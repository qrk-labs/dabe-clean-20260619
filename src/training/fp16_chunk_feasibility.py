from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from lightning import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer

from ..backbone.transformer import TransformerBlock

log = logging.getLogger(__name__)


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(0)


def _extract_text_field(example: dict[str, Any], text_field: str) -> str:
    value = example.get(text_field)
    if isinstance(value, str):
        return value.strip()

    for candidate in ("text", "content", "document", "body"):
        candidate_value = example.get(candidate)
        if isinstance(candidate_value, str):
            return candidate_value.strip()
    return ""


def _load_text_dataset_samples(
    dataset_name: str,
    split: str,
    max_samples: int,
    seed: int,
    text_field: str,
    allow_synthetic_fallback: bool,
    streaming: bool,
) -> list[str]:
    texts: list[str] = []
    try:
        dataset = load_dataset(dataset_name, split=split, streaming=streaming)
        if streaming:
            for idx, example in enumerate(dataset):
                if idx >= max_samples:
                    break
                text = _extract_text_field(example, text_field=text_field)
                if text:
                    texts.append(text)
        else:
            shuffled = dataset.shuffle(seed=seed) if hasattr(dataset, "shuffle") else dataset
            sliced = shuffled.select(range(min(max_samples, len(shuffled))))
            for example in sliced:
                text = _extract_text_field(example, text_field=text_field)
                if text:
                    texts.append(text)
    except Exception as exc:
        if not allow_synthetic_fallback:
            raise RuntimeError(
                (
                    f"Failed to load dataset={dataset_name} split={split} "
                    f"text_field={text_field}: {exc}"
                )
            ) from exc
        log.warning(
            (
                "Failed to load dataset=%s split=%s text_field=%s (%s). "
                "Using synthetic fallback."
            ),
            dataset_name,
            split,
            text_field,
            exc,
        )

    if texts:
        return texts

    if not allow_synthetic_fallback:
        raise RuntimeError(
            (
                f"Dataset yielded no usable text rows for dataset={dataset_name} "
                f"split={split} text_field={text_field}"
            )
        )

    fallback = [
        "A tiny fox found a glowing acorn and shared it with friends.",
        "Mira built a paper boat and the rain carried it to a clock garden.",
        "Tim found a map under his pillow and followed it through moonlight.",
        "Nia asked the wind for help and it returned with a song.",
    ]
    return [fallback[idx % len(fallback)] for idx in range(max_samples)]


def _cpu_util_percent() -> float | None:
    try:
        load, _, _ = os.getloadavg()  # type: ignore[name-defined]
    except Exception:
        return None
    cpu_count = os.cpu_count() or 1
    return max(0.0, min(100.0, (float(load) / float(cpu_count)) * 100.0))


def _resolve_precision(training_cfg: dict[str, Any], accelerator: str) -> str:
    configured = training_cfg.get("precision")
    if configured:
        return str(configured)
    if accelerator == "gpu":
        return "bf16-mixed"
    return "32-true"


@dataclass
class RuntimeMetrics:
    chunk_size_tokens: int
    first_batch_seconds: float | None = None
    steps_per_sec: float = 0.0
    compressed_tokens_per_sec: float = 0.0
    raw_tokens_per_sec: float = 0.0
    gpu_mem_peak_mb: float | None = None
    cpu_util_avg: float | None = None
    gpu_util_avg: float | None = None


class RuntimeMetricsCallback(Callback):
    def __init__(self, runtime_metrics: RuntimeMetrics):
        self.runtime_metrics = runtime_metrics
        self._fit_start: float | None = None
        self._batch_start: float | None = None
        self._batch_durations: list[float] = []
        self._compressed_token_counts: list[int] = []
        self._cpu_utils: list[float] = []

    def _compressed_tokens_in_batch(self, batch: Any) -> int:
        if isinstance(batch, dict):
            values = batch.get("input_scalars")
            if isinstance(values, torch.Tensor):
                if values.dim() == 3:
                    return int(values.shape[0] * values.shape[1])
                return int(values.numel())
        return 0

    def on_train_start(self, trainer: Trainer, pl_module: L.LightningModule) -> None:
        del trainer, pl_module
        self._fit_start = time.perf_counter()

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: L.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del trainer, pl_module, batch_idx
        self._batch_start = time.perf_counter()
        self._compressed_token_counts.append(self._compressed_tokens_in_batch(batch))
        cpu = _cpu_util_percent()
        if cpu is not None:
            self._cpu_utils.append(cpu)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del trainer, pl_module, outputs, batch, batch_idx
        if self._batch_start is None:
            return
        duration = max(1e-6, time.perf_counter() - self._batch_start)
        if self.runtime_metrics.first_batch_seconds is None:
            self.runtime_metrics.first_batch_seconds = duration
        self._batch_durations.append(duration)

    def on_fit_end(self, trainer: Trainer, pl_module: L.LightningModule) -> None:
        del trainer, pl_module
        if self._fit_start is None:
            return
        elapsed = max(1e-6, time.perf_counter() - self._fit_start)
        total_steps = float(len(self._batch_durations))
        total_compressed_tokens = float(sum(self._compressed_token_counts))
        self.runtime_metrics.steps_per_sec = total_steps / elapsed
        self.runtime_metrics.compressed_tokens_per_sec = total_compressed_tokens / elapsed
        self.runtime_metrics.raw_tokens_per_sec = (
            self.runtime_metrics.compressed_tokens_per_sec * self.runtime_metrics.chunk_size_tokens
        )
        if self._cpu_utils:
            self.runtime_metrics.cpu_util_avg = sum(self._cpu_utils) / len(self._cpu_utils)

        if torch.cuda.is_available():
            self.runtime_metrics.gpu_mem_peak_mb = float(torch.cuda.max_memory_allocated()) / (
                1024.0 * 1024.0
            )
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            try:
                self.runtime_metrics.gpu_mem_peak_mb = float(torch.mps.current_allocated_memory()) / (
                    1024.0 * 1024.0
                )
            except Exception:
                self.runtime_metrics.gpu_mem_peak_mb = None


class ValidationTrendCallback(Callback):
    def __init__(self):
        self.val_losses: list[float] = []

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: L.LightningModule) -> None:
        del pl_module
        metric = trainer.callback_metrics.get("val/loss")
        if metric is None:
            return
        self.val_losses.append(float(metric.item()))


class EtaMetricsCallback(Callback):
    """Logs runtime ETA/throughput metrics for fast progress visibility."""

    def __init__(self, log_interval_steps: int, emit_stdout: bool = False):
        self.log_interval_steps = max(1, int(log_interval_steps))
        self.emit_stdout = bool(emit_stdout)
        self._fit_start: float | None = None

    def on_train_start(self, trainer: Trainer, pl_module: L.LightningModule) -> None:
        del trainer, pl_module
        self._fit_start = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, outputs, batch, batch_idx
        if self._fit_start is None:
            return
        global_step = int(trainer.global_step)
        if global_step < 1 or (global_step % self.log_interval_steps) != 0:
            return
        elapsed = max(1e-6, time.perf_counter() - self._fit_start)
        steps_per_sec = float(global_step) / elapsed
        remaining_steps = max(0, int(trainer.max_steps) - global_step)
        eta_seconds = float(remaining_steps) / max(1e-6, steps_per_sec)
        if trainer.logger is not None:
            trainer.logger.log_metrics(
                {
                    "sys/steps_per_sec_live": steps_per_sec,
                    "sys/eta_seconds": eta_seconds,
                },
                step=global_step,
            )
        if self.emit_stdout:
            eta_minutes = eta_seconds / 60.0
            print(
                (
                    f"[eta] step={global_step}/{int(trainer.max_steps)} "
                    f"sps={steps_per_sec:.2f} eta_min={eta_minutes:.1f}"
                ),
                flush=True,
            )


class FP16ChunkCompressor:
    """Deterministic non-learned scalar compressor for EXP-015."""

    def __init__(
        self,
        chunk_size_tokens: int,
        window_overlap_tokens: int,
        tokenizer_name: str = "gpt2",
        force_fast_tokenizer: bool = True,
        dtype: str = "float16",
        tokenizer: Any | None = None,
        scalars_per_chunk: int = 1,
    ):
        self.chunk_size_tokens = int(chunk_size_tokens)
        self.window_overlap_tokens = int(window_overlap_tokens)
        self.scalars_per_chunk = max(1, int(scalars_per_chunk))
        self.dtype = np.float16 if dtype.lower() == "float16" else np.float32
        self.step_tokens = self.chunk_size_tokens - self.window_overlap_tokens
        if self.step_tokens <= 0:
            raise ValueError("window_overlap_tokens must be smaller than chunk_size_tokens.")
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=force_fast_tokenizer,
        )
        if force_fast_tokenizer and not bool(getattr(self.tokenizer, "is_fast", False)):
            raise RuntimeError(
                f"Expected fast tokenizer for {tokenizer_name}, but slow tokenizer was loaded."
            )
        self.vocab_size = max(1, int(getattr(self.tokenizer, "vocab_size", 50257)))

    def _scalar_from_sub_chunk(self, sub_chunk_ids: np.ndarray) -> np.floating[Any]:
        if sub_chunk_ids.size == 0:
            return self.dtype(0.0)
        ids = sub_chunk_ids.astype(np.float32)
        weights = np.arange(1, ids.shape[0] + 1, dtype=np.float32)
        normalized = (ids + 1.0) / float(self.vocab_size)
        weighted_mean = float(np.dot(normalized, weights) / np.sum(weights))
        scalar = np.tanh((weighted_mean * 2.0) - 1.0)
        if not math.isfinite(scalar):
            scalar = 0.0
        return self.dtype(scalar)

    def _chunk_to_scalars(self, chunk_ids: np.ndarray) -> np.ndarray:
        if chunk_ids.size == 0:
            return np.zeros((self.scalars_per_chunk,), dtype=self.dtype)
        num_scalars = self.scalars_per_chunk
        chunk_size = len(chunk_ids)
        sub_window_size = chunk_size // num_scalars
        remainder = chunk_size % num_scalars
        scalars = np.empty((num_scalars,), dtype=self.dtype)
        start_idx = 0
        for i in range(num_scalars):
            end_idx = start_idx + sub_window_size + (1 if i < remainder else 0)
            sub_chunk = chunk_ids[start_idx:end_idx]
            scalars[i] = self._scalar_from_sub_chunk(sub_chunk)
            start_idx = end_idx
        return scalars

    def compress_token_ids(self, token_ids: Sequence[int]) -> np.ndarray:
        scalars, _ = self.compress_token_ids_with_chunks(token_ids)
        return scalars

    def compress_token_ids_with_chunks(self, token_ids: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.shape[0] < self.chunk_size_tokens:
            return (
                np.empty((0, self.scalars_per_chunk), dtype=self.dtype),
                np.empty((0, self.chunk_size_tokens), dtype=np.int32),
            )

        starts = np.arange(
            0,
            ids.shape[0] - self.chunk_size_tokens + 1,
            self.step_tokens,
            dtype=np.int64,
        )
        scalars = np.empty((starts.shape[0], self.scalars_per_chunk), dtype=self.dtype)
        chunk_ids = np.empty((starts.shape[0], self.chunk_size_tokens), dtype=np.int32)
        for i, start in enumerate(starts):
            chunk = ids[start : start + self.chunk_size_tokens]
            scalars[i] = self._chunk_to_scalars(chunk)
            chunk_ids[i] = chunk.astype(np.int32, copy=False)
        return scalars, chunk_ids

    def compress_text(self, text: str) -> np.ndarray:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return self.compress_token_ids(token_ids)

    def compress_texts(
        self,
        texts: Sequence[str],
        tokenizer_batch_size: int,
    ) -> list[np.ndarray]:
        compressed: list[np.ndarray] = []
        batch_size = max(1, int(tokenizer_batch_size))
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = self.tokenizer(batch, add_special_tokens=False)
            ids_per_text = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
            for token_ids in ids_per_text:
                compressed.append(self.compress_token_ids(token_ids))
        return compressed

    def compress_texts_with_chunks(
        self,
        texts: Sequence[str],
        tokenizer_batch_size: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        compressed: list[tuple[np.ndarray, np.ndarray]] = []
        batch_size = max(1, int(tokenizer_batch_size))
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = self.tokenizer(batch, add_special_tokens=False)
            ids_per_text = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
            for token_ids in ids_per_text:
                compressed.append(self.compress_token_ids_with_chunks(token_ids))
        return compressed

    def flatten_stream(self, texts: Sequence[str], tokenizer_batch_size: int) -> np.ndarray:
        scalar_parts: list[np.ndarray] = []
        batch_size = max(1, int(tokenizer_batch_size))
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = self.tokenizer(batch, add_special_tokens=False)
            ids_per_text = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
            batch_scalars: list[np.ndarray] = []
            for token_ids in ids_per_text:
                scalars = self.compress_token_ids(token_ids)
                if scalars.size > 0:
                    batch_scalars.append(scalars)
            if batch_scalars:
                scalar_parts.append(np.concatenate(batch_scalars, axis=0))
        if not scalar_parts:
            return np.empty((0, self.scalars_per_chunk), dtype=self.dtype)
        return np.concatenate(scalar_parts, axis=0).astype(self.dtype, copy=False)

    def flatten_stream_with_chunks(
        self,
        texts: Sequence[str],
        tokenizer_batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        scalar_parts: list[np.ndarray] = []
        chunk_parts: list[np.ndarray] = []
        batch_size = max(1, int(tokenizer_batch_size))
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = self.tokenizer(batch, add_special_tokens=False)
            ids_per_text = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
            batch_scalars: list[np.ndarray] = []
            batch_chunks: list[np.ndarray] = []
            for token_ids in ids_per_text:
                scalars, chunks = self.compress_token_ids_with_chunks(token_ids)
                if scalars.size > 0:
                    batch_scalars.append(scalars)
                    batch_chunks.append(chunks)
            if batch_scalars:
                scalar_parts.append(np.concatenate(batch_scalars, axis=0))
                chunk_parts.append(np.concatenate(batch_chunks, axis=0))
        if not scalar_parts:
            return (
                np.empty((0, self.scalars_per_chunk), dtype=self.dtype),
                np.empty((0, self.chunk_size_tokens), dtype=np.int32),
            )
        scalar_stream = np.concatenate(scalar_parts, axis=0).astype(
            self.dtype,
            copy=False,
        )
        chunk_stream = np.concatenate(chunk_parts, axis=0).astype(
            np.int32,
            copy=False,
        )
        return scalar_stream, chunk_stream


class FP16ChunkSequenceDataset(TorchDataset):
    def __init__(
        self,
        scalar_stream: np.ndarray,
        compressed_seq_len: int,
        window_stride: int = 1,
        chunk_tokens_stream: np.ndarray | None = None,
    ):
        if compressed_seq_len < 1:
            raise ValueError("compressed_seq_len must be >= 1")
        stride = max(1, int(window_stride))
        seq_len = int(compressed_seq_len)
        self._seq_len = seq_len
        stream = np.asarray(scalar_stream, dtype=np.float32)
        tokens = (
            np.asarray(chunk_tokens_stream, dtype=np.int32)
            if chunk_tokens_stream is not None
            else None
        )
        self._chunk_tokens_stream = tokens

        if stream.ndim == 1:
            self._num_scalars = 1
            stream = stream.reshape(-1)
            if tokens is not None and tokens.shape[0] != stream.shape[0]:
                raise ValueError("chunk_tokens_stream must align with scalar_stream length.")
            if stream.shape[0] <= seq_len:
                self._inputs = np.empty((0, seq_len), dtype=np.float32)
                self._targets = np.empty((0, seq_len), dtype=np.float32)
                self._starts = np.empty((0,), dtype=np.int64)
                return
            starts = np.arange(0, stream.shape[0] - seq_len, stride, dtype=np.int64)
            if starts.size == 0:
                self._inputs = np.empty((0, seq_len), dtype=np.float32)
                self._targets = np.empty((0, seq_len), dtype=np.float32)
                self._starts = np.empty((0,), dtype=np.int64)
                return
            indices = starts[:, None] + np.arange(seq_len + 1, dtype=np.int64)[None, :]
            windows = stream[indices]
            self._inputs = windows[:, :-1].astype(np.float32, copy=False)
            self._targets = windows[:, 1:].astype(np.float32, copy=False)
            self._starts = starts
        elif stream.ndim == 2:
            self._num_scalars = stream.shape[1]
            if tokens is not None and tokens.shape[0] != stream.shape[0]:
                raise ValueError("chunk_tokens_stream must align with scalar_stream length.")
            if stream.shape[0] <= seq_len:
                self._inputs = np.empty((0, seq_len, self._num_scalars), dtype=np.float32)
                self._targets = np.empty((0, seq_len, self._num_scalars), dtype=np.float32)
                self._starts = np.empty((0,), dtype=np.int64)
                return
            starts = np.arange(0, stream.shape[0] - seq_len, stride, dtype=np.int64)
            if starts.size == 0:
                self._inputs = np.empty((0, seq_len, self._num_scalars), dtype=np.float32)
                self._targets = np.empty((0, seq_len, self._num_scalars), dtype=np.float32)
                self._starts = np.empty((0,), dtype=np.int64)
                return
            indices = starts[:, None] + np.arange(seq_len + 1, dtype=np.int64)[None, :]
            windows = stream[indices]
            self._inputs = windows[:, :-1, :].astype(np.float32, copy=False)
            self._targets = windows[:, 1:, :].astype(np.float32, copy=False)
            self._starts = starts
        else:
            raise ValueError(f"scalar_stream must be 1D or 2D, got {stream.ndim}D")

    def __len__(self) -> int:
        return int(self._inputs.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = {
            "input_scalars": torch.from_numpy(np.asarray(self._inputs[idx], dtype=np.float32)),
            "target_scalars": torch.from_numpy(np.asarray(self._targets[idx], dtype=np.float32)),
        }
        if self._chunk_tokens_stream is not None:
            start = int(self._starts[idx])
            target_chunk_ids = self._chunk_tokens_stream[start + 1 : start + 1 + self._seq_len]
            sample["target_chunk_ids"] = torch.from_numpy(
                np.asarray(target_chunk_ids, dtype=np.int64),
            )
        return sample


class FP16ChunkDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_texts: Sequence[str],
        val_texts: Sequence[str],
        compressor: FP16ChunkCompressor,
        compressed_seq_len: int,
        window_stride: int,
        tokenizer_batch_size: int,
        batch_size: int,
        num_workers: int,
        seed: int,
        accelerator: str,
        include_chunk_tokens: bool = False,
    ):
        super().__init__()
        self.train_texts = list(train_texts)
        self.val_texts = list(val_texts)
        self.compressor = compressor
        self.compressed_seq_len = int(compressed_seq_len)
        self.window_stride = int(window_stride)
        self.tokenizer_batch_size = int(tokenizer_batch_size)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.seed = int(seed)
        self.accelerator = str(accelerator)
        self.include_chunk_tokens = bool(include_chunk_tokens)
        self._is_setup = False
        self.train_chunk_count = 0
        self.val_chunk_count = 0

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self._is_setup:
            return
        if self.include_chunk_tokens:
            train_stream, train_chunks = self.compressor.flatten_stream_with_chunks(
                texts=self.train_texts,
                tokenizer_batch_size=self.tokenizer_batch_size,
            )
            val_stream, val_chunks = self.compressor.flatten_stream_with_chunks(
                texts=self.val_texts,
                tokenizer_batch_size=self.tokenizer_batch_size,
            )
        else:
            train_stream = self.compressor.flatten_stream(
                texts=self.train_texts,
                tokenizer_batch_size=self.tokenizer_batch_size,
            )
            val_stream = self.compressor.flatten_stream(
                texts=self.val_texts,
                tokenizer_batch_size=self.tokenizer_batch_size,
            )
            train_chunks = None
            val_chunks = None
        self.train_chunk_count = int(train_stream.shape[0])
        self.val_chunk_count = int(val_stream.shape[0])

        self.train_dataset = FP16ChunkSequenceDataset(
            scalar_stream=train_stream,
            compressed_seq_len=self.compressed_seq_len,
            window_stride=self.window_stride,
            chunk_tokens_stream=train_chunks,
        )
        self.val_dataset = FP16ChunkSequenceDataset(
            scalar_stream=val_stream,
            compressed_seq_len=self.compressed_seq_len,
            window_stride=self.window_stride,
            chunk_tokens_stream=val_chunks,
        )

        if len(self.train_dataset) == 0:
            raise RuntimeError(
                "FP16ChunkDataModule produced zero train samples. "
                "Increase train_samples or reduce chunk_size_tokens/compressed_seq_len."
            )
        if len(self.val_dataset) == 0:
            raise RuntimeError(
                "FP16ChunkDataModule produced zero val samples. "
                "Increase val_samples or reduce chunk_size_tokens/compressed_seq_len."
            )
        self._is_setup = True

    def _dataloader_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"num_workers": self.num_workers}
        if self.accelerator == "gpu" and self.num_workers > 0:
            kwargs.update({
                "pin_memory": True,
                "persistent_workers": True,
                "prefetch_factor": 2,
            })
        return kwargs

    def train_dataloader(self) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            **self._dataloader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            **self._dataloader_kwargs(),
        )


class ScalarTransformerLM(nn.Module):
    """Tiny scalar-stream causal LM for EXP-015 throughput feasibility."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.hidden_dim = int(config.get("hidden_dim", 256))
        self.num_layers = int(config.get("num_layers", 4))
        self.num_heads = int(config.get("num_heads", 4))
        self.ff_dim = int(config.get("ff_dim", 1024))
        self.dropout = float(config.get("dropout", 0.1))
        self.attention_cfg = dict(config.get("attention", {}))
        self.num_scalars = int(config.get("num_scalars", 1))

        self.input_proj = nn.Linear(self.num_scalars, self.hidden_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout=self.dropout,
                attention_cfg=self.attention_cfg,
            )
            for _ in range(self.num_layers)
        ])
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.num_scalars)

    def forward(
        self,
        input_scalars: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_scalars.dim() == 2:
            x = input_scalars.unsqueeze(-1)
        else:
            x = input_scalars
        hidden = self.input_proj(x)
        for block in self.blocks:
            hidden = block(hidden, mask=mask)
        hidden = self.norm(hidden)
        output = self.output_proj(hidden)
        if self.num_scalars == 1:
            output = output.squeeze(-1)
        return output, hidden


class ContextConditionedDiffusionDenoiser(nn.Module):
    """Simple context-conditioned denoiser used for latent gist reconstruction."""

    def __init__(self, latent_dim: int, context_dim: int, max_timesteps: int):
        super().__init__()
        self.time_embed = nn.Embedding(max_timesteps + 1, latent_dim)
        hidden_dim = max(latent_dim, context_dim) * 2
        self.net = nn.Sequential(
            nn.Linear(latent_dim + context_dim + latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t: torch.Tensor, context: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_embed = self.time_embed(t.long())
        features = torch.cat([z_t, context, t_embed], dim=-1)
        return self.net(features)


class FP16ChunkLMModule(L.LightningModule):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.model_cfg = dict(config.get("model", {}))
        self.training_cfg = dict(config.get("training", {}))
        self.compression_cfg = dict(config.get("compression", {}))
        self.diffusion_cfg = dict(config.get("diffusion", {}))
        self.model_cfg.setdefault("num_scalars", int(self.compression_cfg.get("scalars_per_chunk", 1)))
        self.model = ScalarTransformerLM(self.model_cfg)
        self.diffusion_enabled = bool(self.diffusion_cfg.get("enabled", False))
        self.diffusion_loss_weight = float(self.diffusion_cfg.get("loss_weight", 0.25))
        self.diffusion_recon_weight = float(self.diffusion_cfg.get("reconstruction_weight", 0.2))
        if self.diffusion_enabled:
            vocab_size = int(self.diffusion_cfg.get("vocab_size", 50257))
            self.diffusion_latent_dim = int(
                self.diffusion_cfg.get("latent_dim", self.model.hidden_dim),
            )
            self.diffusion_timesteps = int(self.diffusion_cfg.get("timesteps", 256))
            self.chunk_token_embed = nn.Embedding(vocab_size, self.diffusion_latent_dim)
            self.chunk_latent_norm = nn.LayerNorm(self.diffusion_latent_dim)
            self.context_to_latent = nn.Linear(self.model.hidden_dim, self.diffusion_latent_dim)
            self.diffusion_denoiser = ContextConditionedDiffusionDenoiser(
                latent_dim=self.diffusion_latent_dim,
                context_dim=self.diffusion_latent_dim,
                max_timesteps=self.diffusion_timesteps,
            )
            self.latent_decoder = nn.Sequential(
                nn.Linear(self.diffusion_latent_dim, self.diffusion_latent_dim),
                nn.SiLU(),
                nn.Linear(self.diffusion_latent_dim, self.diffusion_latent_dim),
            )
        self.nan_batches = 0

    def _diffusion_losses(
        self,
        hidden: torch.Tensor,
        target_chunk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.diffusion_enabled:
            zero = torch.tensor(0.0, device=hidden.device)
            return zero, zero, zero, zero, zero

        z0 = self.chunk_token_embed(target_chunk_ids).mean(dim=-2)
        z0 = self.chunk_latent_norm(z0)
        context = self.context_to_latent(hidden)

        t = torch.randint(
            low=1,
            high=self.diffusion_timesteps + 1,
            size=(z0.shape[0], z0.shape[1]),
            device=hidden.device,
            dtype=torch.long,
        )
        alpha = 1.0 - (t.float() / float(self.diffusion_timesteps + 1))
        alpha = alpha.clamp(min=1e-4, max=0.9999).unsqueeze(-1)
        noise = torch.randn_like(z0)
        z_t = torch.sqrt(alpha) * z0 + torch.sqrt(1.0 - alpha) * noise
        noise_hat = self.diffusion_denoiser(z_t=z_t, context=context, t=t)
        diffusion_noise_loss = F.mse_loss(noise_hat, noise)

        z0_hat = (z_t - torch.sqrt(1.0 - alpha) * noise_hat) / torch.sqrt(alpha)
        decoded = self.latent_decoder(z0_hat)
        diffusion_recon_loss = F.mse_loss(decoded, z0)
        diffusion_loss = diffusion_noise_loss + self.diffusion_recon_weight * diffusion_recon_loss
        gist_cosine = F.cosine_similarity(decoded, z0, dim=-1).mean()
        decoded_flat = F.normalize(decoded.reshape(-1, decoded.shape[-1]), dim=-1)
        z0_flat = F.normalize(z0.reshape(-1, z0.shape[-1]), dim=-1)
        similarity = torch.matmul(decoded_flat, z0_flat.transpose(0, 1))
        top_idx = similarity.argmax(dim=-1)
        expected_idx = torch.arange(similarity.shape[0], device=similarity.device)
        gist_retrieval_top1 = (top_idx == expected_idx).float().mean()
        return diffusion_loss, diffusion_noise_loss, diffusion_recon_loss, gist_cosine, gist_retrieval_top1

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        stage: str,
        log_metrics: bool = True,
    ) -> torch.Tensor:
        input_scalars = batch["input_scalars"].to(self.device)
        target_scalars = batch["target_scalars"].to(self.device)
        mask = build_causal_mask(input_scalars.shape[1], self.device)
        preds, hidden = self.model(input_scalars, mask=mask)
        scalar_loss = F.smooth_l1_loss(preds, target_scalars)
        loss = scalar_loss
        diffusion_loss = torch.tensor(0.0, device=self.device)
        diffusion_noise_loss = torch.tensor(0.0, device=self.device)
        diffusion_recon_loss = torch.tensor(0.0, device=self.device)
        gist_cosine = torch.tensor(0.0, device=self.device)
        gist_retrieval_top1 = torch.tensor(0.0, device=self.device)
        if self.diffusion_enabled:
            if "target_chunk_ids" not in batch:
                raise RuntimeError(
                    "Diffusion-enabled run requires target_chunk_ids in dataloader batch.",
                )
            target_chunk_ids = batch["target_chunk_ids"].to(self.device)
            (
                diffusion_loss,
                diffusion_noise_loss,
                diffusion_recon_loss,
                gist_cosine,
                gist_retrieval_top1,
            ) = self._diffusion_losses(
                hidden=hidden,
                target_chunk_ids=target_chunk_ids,
            )
            loss = loss + self.diffusion_loss_weight * diffusion_loss
        mae = F.l1_loss(preds, target_scalars)

        if not bool(torch.isfinite(loss)):
            self.nan_batches += 1
            raise RuntimeError("Non-finite loss in FP16 chunk feasibility training.")

        if log_metrics:
            self.log(f"{stage}/loss", loss, prog_bar=True)
            self.log(f"{stage}/scalar_loss", scalar_loss)
            self.log(f"{stage}/mae", mae, prog_bar=(stage == "val"))
            self.log(f"{stage}/stable", torch.tensor(1.0, device=self.device))
            if self.diffusion_enabled:
                self.log(f"{stage}/diffusion_loss", diffusion_loss)
                self.log(f"{stage}/diffusion_noise_loss", diffusion_noise_loss)
                self.log(f"{stage}/diffusion_recon_loss", diffusion_recon_loss)
                self.log(f"{stage}/gist_cosine", gist_cosine)
                self.log(f"{stage}/gist_retrieval_top1", gist_retrieval_top1)
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._step(batch, stage="train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._step(batch, stage="val")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=float(self.training_cfg.get("learning_rate", 3e-4)),
            weight_decay=float(self.training_cfg.get("weight_decay", 0.01)),
        )


def run_fp16_chunk_feasibility(
    config: dict[str, Any],
    output_dir: Path,
    accelerator: str,
    devices: Any,
) -> dict[str, Any]:
    fp16_cfg = dict(config.get("fp16_chunk", {}))
    dataset_cfg = dict(fp16_cfg.get("dataset", {}))
    compression_cfg = dict(fp16_cfg.get("compression", {}))
    training_cfg = dict(fp16_cfg.get("training", {}))
    gate_cfg = dict(fp16_cfg.get("gate", {}))
    diffusion_cfg = dict(fp16_cfg.get("diffusion", {}))
    diffusion_enabled = bool(diffusion_cfg.get("enabled", False))
    seed = int(config.get("experiment", {}).get("seed", 42))

    train_texts = _load_text_dataset_samples(
        dataset_name=str(dataset_cfg.get("dataset_name", "roneneldan/TinyStories")),
        split=str(dataset_cfg.get("train_split", "train")),
        max_samples=int(dataset_cfg.get("train_samples", 4096)),
        seed=seed,
        text_field=str(dataset_cfg.get("text_field", "text")),
        allow_synthetic_fallback=bool(dataset_cfg.get("allow_synthetic_fallback", False)),
        streaming=bool(dataset_cfg.get("streaming", True)),
    )
    val_texts = _load_text_dataset_samples(
        dataset_name=str(dataset_cfg.get("dataset_name", "roneneldan/TinyStories")),
        split=str(dataset_cfg.get("val_split", "validation")),
        max_samples=int(dataset_cfg.get("val_samples", 512)),
        seed=seed + 1,
        text_field=str(dataset_cfg.get("text_field", "text")),
        allow_synthetic_fallback=bool(dataset_cfg.get("allow_synthetic_fallback", False)),
        streaming=bool(dataset_cfg.get("streaming", True)),
    )

    compressor = FP16ChunkCompressor(
        chunk_size_tokens=int(compression_cfg.get("chunk_size_tokens", 128)),
        window_overlap_tokens=int(compression_cfg.get("window_overlap", 0)),
        tokenizer_name=str(compression_cfg.get("tokenizer_name", "gpt2")),
        force_fast_tokenizer=bool(compression_cfg.get("force_fast_tokenizer", True)),
        dtype=str(compression_cfg.get("dtype", "float16")),
        scalars_per_chunk=int(compression_cfg.get("scalars_per_chunk", 1)),
    )
    data_module = FP16ChunkDataModule(
        train_texts=train_texts,
        val_texts=val_texts,
        compressor=compressor,
        compressed_seq_len=int(training_cfg.get("compressed_seq_len", 8)),
        window_stride=int(training_cfg.get("window_stride", 1)),
        tokenizer_batch_size=int(compression_cfg.get("tokenizer_batch_size", 64)),
        batch_size=int(training_cfg.get("batch_size", 64)),
        num_workers=int(training_cfg.get("num_workers", 0)),
        seed=seed,
        accelerator=accelerator,
        include_chunk_tokens=diffusion_enabled,
    )
    model_config = {
        **fp16_cfg,
        "diffusion": {
            **diffusion_cfg,
            "vocab_size": int(diffusion_cfg.get("vocab_size", compressor.vocab_size)),
            "chunk_size_tokens": int(
                diffusion_cfg.get("chunk_size_tokens", compressor.chunk_size_tokens),
            ),
        },
    }
    model = FP16ChunkLMModule(config=model_config)

    runtime_metrics = RuntimeMetrics(chunk_size_tokens=compressor.chunk_size_tokens)
    runtime_callback = RuntimeMetricsCallback(runtime_metrics=runtime_metrics)
    trend_callback = ValidationTrendCallback()
    eta_callback = EtaMetricsCallback(
        log_interval_steps=int(training_cfg.get("eta_log_interval_steps", 20)),
        emit_stdout=bool(training_cfg.get("eta_stdout", False)),
    )
    callbacks: list[Callback] = [runtime_callback, trend_callback, eta_callback]

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_every_n_train_steps = int(training_cfg.get("checkpoint_every_n_train_steps", 0))
    checkpoint_save_top_k = int(training_cfg.get("checkpoint_save_top_k", 1))
    checkpoint_save_last = bool(training_cfg.get("checkpoint_save_last", True))
    checkpoint_best_callback: ModelCheckpoint | None = None
    checkpoint_periodic_callback: ModelCheckpoint | None = None
    if checkpoint_every_n_train_steps > 0:
        checkpoint_periodic_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="step-{step:07d}",
            save_top_k=-1,
            save_last=False,
            every_n_train_steps=checkpoint_every_n_train_steps,
            save_on_train_epoch_end=False,
        )
        callbacks.append(checkpoint_periodic_callback)
    if checkpoint_save_top_k != 0 or checkpoint_save_last:
        checkpoint_best_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best-step-{step:07d}",
            monitor="val/loss",
            mode="min",
            save_top_k=checkpoint_save_top_k,
            save_last=checkpoint_save_last,
            auto_insert_metric_name=False,
        )
        callbacks.append(checkpoint_best_callback)

    enable_csv_logger = bool(training_cfg.get("enable_csv_logger", True))
    csv_logger: CSVLogger | bool
    if enable_csv_logger:
        csv_logger = CSVLogger(save_dir=str(output_dir), name="logs")
    else:
        csv_logger = False

    trainer = Trainer(
        max_steps=int(training_cfg.get("max_steps", 1500)),
        accelerator=accelerator,
        devices=devices,
        logger=csv_logger,
        callbacks=callbacks,
        default_root_dir=str(output_dir),
        gradient_clip_val=float(training_cfg.get("grad_clip", 1.0)),
        precision=_resolve_precision(training_cfg, accelerator),
        log_every_n_steps=int(training_cfg.get("log_every_n_steps", 20)),
        val_check_interval=int(training_cfg.get("val_check_interval", 100)),
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=data_module)

    val_loss_first = trend_callback.val_losses[0] if trend_callback.val_losses else None
    val_loss_last = trend_callback.val_losses[-1] if trend_callback.val_losses else None
    min_relative_drop = float(gate_cfg.get("min_relative_drop", 0.0))
    val_trend_pass = False
    if val_loss_first is not None and val_loss_last is not None:
        threshold = val_loss_first * (1.0 - min_relative_drop)
        val_trend_pass = bool(val_loss_last <= threshold)
    stable = model.nan_batches == 0
    require_stable = bool(gate_cfg.get("require_stable", True))
    require_downward_trend = bool(gate_cfg.get("require_downward_trend", True))
    gate_pass = (stable or not require_stable) and (val_trend_pass or not require_downward_trend)
    metrics_csv_path = (
        str(Path(trainer.log_dir) / "metrics.csv")
        if trainer.log_dir is not None
        else None
    )

    return {
        "stable": bool(stable),
        "nan_batches": int(model.nan_batches),
        "val_loss_first": val_loss_first,
        "val_loss_last": val_loss_last,
        "val_trend_pass": bool(val_trend_pass),
        "diffusion_enabled": diffusion_enabled,
        "val_diffusion_loss_last": (
            float(trainer.callback_metrics["val/diffusion_loss"].item())
            if diffusion_enabled and "val/diffusion_loss" in trainer.callback_metrics
            else None
        ),
        "val_gist_cosine_last": (
            float(trainer.callback_metrics["val/gist_cosine"].item())
            if diffusion_enabled and "val/gist_cosine" in trainer.callback_metrics
            else None
        ),
        "val_gist_retrieval_top1_last": (
            float(trainer.callback_metrics["val/gist_retrieval_top1"].item())
            if diffusion_enabled and "val/gist_retrieval_top1" in trainer.callback_metrics
            else None
        ),
        "first_batch_seconds": runtime_metrics.first_batch_seconds,
        "steps_per_sec": runtime_metrics.steps_per_sec,
        "compressed_tokens_per_sec": runtime_metrics.compressed_tokens_per_sec,
        "raw_tokens_per_sec": runtime_metrics.raw_tokens_per_sec,
        "gpu_mem_peak_mb": runtime_metrics.gpu_mem_peak_mb,
        "cpu_util_avg": runtime_metrics.cpu_util_avg,
        "gpu_util_avg": runtime_metrics.gpu_util_avg,
        "device": str(trainer.strategy.root_device),
        "train_samples": len(data_module.train_dataset),
        "val_samples": len(data_module.val_dataset),
        "train_chunks": data_module.train_chunk_count,
        "val_chunks": data_module.val_chunk_count,
        "gate_pass": bool(gate_pass),
        "metrics_csv_path": metrics_csv_path,
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir.exists() else None,
        "best_checkpoint_path": (
            checkpoint_best_callback.best_model_path
            if checkpoint_best_callback is not None and checkpoint_best_callback.best_model_path
            else None
        ),
        "last_checkpoint_path": (
            checkpoint_best_callback.last_model_path
            if checkpoint_best_callback is not None and checkpoint_best_callback.last_model_path
            else None
        ),
        "last_periodic_checkpoint_path": (
            checkpoint_periodic_callback.last_model_path
            if checkpoint_periodic_callback is not None and checkpoint_periodic_callback.last_model_path
            else None
        ),
    }
