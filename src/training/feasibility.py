from __future__ import annotations

import logging
import math
import os
import random
import resource
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from lightning import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer

from ..backbone.adapter_transformer import AdapterTokenTransformerLM
from ..backbone.token_transformer import TokenTransformerLM
from ..backbone.transformer import DABETransformer
from ..bitmask_encoder.lfq_encoder import LFQEncoder
from ..density_router.base import Span
from ..density_router.entropy_based import EntropyRouter
from ..evaluation.compression import CompressionMetrics

log = logging.getLogger(__name__)


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(0)


def _sequence_windows(ids: Sequence[int], seq_len: int) -> list[tuple[list[int], list[int]]]:
    windows: list[tuple[list[int], list[int]]] = []
    if len(ids) <= seq_len:
        return windows
    step = max(1, seq_len // 2)
    for start in range(0, len(ids) - seq_len, step):
        chunk = ids[start : start + seq_len + 1]
        if len(chunk) != seq_len + 1:
            continue
        windows.append((chunk[:-1], chunk[1:]))
    return windows


def _sequence_windows_np(ids: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    if ids.shape[0] <= seq_len:
        return (
            np.empty((0, seq_len), dtype=np.int32),
            np.empty((0, seq_len), dtype=np.int32),
        )
    step = max(1, seq_len // 2)
    starts = np.arange(0, ids.shape[0] - seq_len, step, dtype=np.int64)
    if starts.size == 0:
        return (
            np.empty((0, seq_len), dtype=np.int32),
            np.empty((0, seq_len), dtype=np.int32),
        )
    idx = starts[:, None] + np.arange(seq_len + 1, dtype=np.int64)[None, :]
    windows = ids[idx]
    return windows[:, :-1].astype(np.int32), windows[:, 1:].astype(np.int32)


def _extract_text_field(example: dict[str, Any], text_field: str) -> str:
    value = example.get(text_field)
    if isinstance(value, str):
        return value.strip()

    # Best-effort fallback for common text dataset schemas.
    for candidate in ("text", "content", "document", "body"):
        candidate_value = example.get(candidate)
        if isinstance(candidate_value, str):
            return candidate_value.strip()
    return ""


def _python_code_texts(max_samples: int) -> list[str]:
    """Deterministic Python-code corpus for paper-defense domain probes."""
    snippets = [
        "def add_user(users, name):\n    users.append({'name': name, 'active': True})\n    return users\n",
        "class Cache:\n    def __init__(self):\n        self.items = {}\n\n    def get(self, key, default=None):\n        return self.items.get(key, default)\n",
        "for idx, value in enumerate(values):\n    if value % 2 == 0:\n        total += value\n    else:\n        skipped.append(idx)\n",
        "with open(path, 'r', encoding='utf-8') as handle:\n    lines = [line.strip() for line in handle if line.strip()]\n",
        "try:\n    result = client.fetch(user_id=user_id, timeout=3.0)\nexcept TimeoutError:\n    result = {'error': 'timeout'}\n",
        "def normalize_batch(batch):\n    mean = sum(batch) / max(len(batch), 1)\n    return [(item - mean) for item in batch]\n",
        "async def fetch_json(session, url):\n    async with session.get(url, timeout=10) as response:\n        response.raise_for_status()\n        return await response.json()\n",
    ]
    samples: list[str] = []
    for idx in range(max(1, int(max_samples))):
        snippet = snippets[idx % len(snippets)]
        # Repeat enough times that both BPE tokens and DABE word-span tokens
        # produce non-empty 64-step language-model windows.
        samples.append((snippet + "\n") * 8)
    return samples


def _load_text_dataset_samples(
    dataset_name: str,
    split: str,
    max_samples: int,
    seed: int = 42,
    text_field: str = "text",
    allow_synthetic_fallback: bool = True,
    streaming: bool = True,
    cache_mode: str = "none",
    cache_root: Path | None = None,
) -> list[str]:
    if str(dataset_name) in {"python_code", "__python_code__", "code", "__code__"}:
        return _python_code_texts(max_samples)

    cache_file: Path | None = None
    if cache_root is not None and cache_mode in {"build", "reuse"}:
        key = _hash_key([
            dataset_name,
            split,
            str(max_samples),
            str(seed),
            text_field,
            str(streaming),
        ])
        cache_file = cache_root / "texts" / f"{key}.txt"
        if cache_mode == "reuse":
            cached = _load_cached_texts(cache_file)
            if len(cached) >= max_samples:
                return cached[:max_samples]

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
                "Failed to load text dataset=%s split=%s text_field=%s (%s). "
                "Using fallback synthetic stories."
            ),
            dataset_name,
            split,
            text_field,
            exc,
        )
    if texts:
        if cache_file is not None and cache_mode in {"build", "reuse"}:
            _save_cached_texts(cache_file, texts)
        return texts
    if not allow_synthetic_fallback:
        raise RuntimeError(
            (
                f"Dataset yielded no usable text rows for dataset={dataset_name} "
                f"split={split} text_field={text_field}"
            )
        )

    random.seed(seed)
    fallback = [
        "Tim found a tiny map under his pillow and followed it to a glowing acorn.",
        "Mira built a paper boat and the rain carried it to a garden of clocks.",
        "A fox borrowed the moonlight and painted silver paths for the village.",
        "Nia whispered to the wind, and it returned with a pocket full of stars.",
    ]
    return [random.choice(fallback) for _ in range(max_samples)]


def _compute_width_distribution(widths: Sequence[int]) -> dict[str, float]:
    total = max(1, len(widths))
    counts: dict[int, int] = {}
    for width in widths:
        counts[int(width)] = counts.get(int(width), 0) + 1
    return {f"width_{width}": count / total for width, count in sorted(counts.items())}


def _hamming_distance(a: torch.Tensor, b: torch.Tensor) -> int:
    max_len = max(a.numel(), b.numel())
    padded_a = F.pad(a.long(), (0, max_len - a.numel()))
    padded_b = F.pad(b.long(), (0, max_len - b.numel()))
    return int((padded_a != padded_b).sum().item())


def _hamming_summary(values: Sequence[int]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    float_values = [float(v) for v in values]
    return {
        "mean": sum(float_values) / len(float_values),
        "min": min(float_values),
        "max": max(float_values),
    }


def _collect_spans(router: EntropyRouter, texts: Sequence[str]) -> tuple[list[Span], list[int]]:
    spans: list[Span] = []
    widths: list[int] = []
    for text in texts:
        scores = router.score(text)
        segmented = router.segment(text, scores)
        spans.extend(segmented)
        widths.extend([span.bit_width for span in segmented])
    return spans, widths


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = max(0.0, float(seconds))
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s"


def _normalize_batch_count(value: Any) -> int | None:
    if isinstance(value, (list, tuple)):
        flattened = [_normalize_batch_count(item) for item in value]
        if any(item is None for item in flattened):
            return None
        return int(sum(item for item in flattened if item is not None))
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return int(value)
    if isinstance(value, int):
        return value
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    return coerced


def _hash_key(parts: Sequence[str]) -> str:
    payload = "||".join(parts).encode("utf-8")
    return str(abs(hash(payload)))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _cpu_util_percent() -> float | None:
    try:
        load, _, _ = os.getloadavg()  # type: ignore[name-defined]
    except Exception:
        return None
    cpu_count = os.cpu_count() or 1
    return max(0.0, min(100.0, (float(load) / float(cpu_count)) * 100.0))


def _gpu_util_percent() -> float | None:
    if not torch.cuda.is_available():
        return None
    try:
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(command, text=True, timeout=1.0).strip()
        first = out.splitlines()[0].strip()
        return float(first)
    except Exception:
        return None


def _resolve_precision(config: dict, accelerator: str) -> str:
    opt_cfg = config.get("optimization", {})
    profile = str(opt_cfg.get("device_profile", "mps_safe")).lower()
    if profile == "cuda_fast" and accelerator == "gpu":
        return "bf16-mixed"
    return "32-true"


def _resolve_data_cache_root(config: dict, output_dir: Path) -> Path:
    feasibility_cfg = config.get("feasibility", {})
    opt_cfg = config.get("optimization", {})
    data_cache_cfg = opt_cfg.get("data_cache", {})
    configured = data_cache_cfg.get("root")
    if configured:
        root = Path(str(configured))
        root.mkdir(parents=True, exist_ok=True)
        return root

    artifact_path = feasibility_cfg.get("tokenizer_artifact_path")
    if artifact_path:
        artifact = Path(str(artifact_path))
        root = artifact.parent.parent / "cache"
        root.mkdir(parents=True, exist_ok=True)
        return root
    root = output_dir / "cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_cache_mode(config: dict) -> str:
    mode = (
        config.get("optimization", {})
        .get("data_cache", {})
        .get("mode", "none")
    )
    mode_str = str(mode).lower()
    if mode_str not in {"none", "build", "reuse"}:
        return "none"
    return mode_str


def _load_cached_texts(cache_file: Path) -> list[str]:
    if not cache_file.exists():
        return []
    lines = cache_file.read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line.strip()]


def _save_cached_texts(cache_file: Path, texts: Sequence[str]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text("\n".join(texts), encoding="utf-8")


class ETAProgressCallback(Callback):
    """Logs rolling ETA for long feasibility runs in non-interactive Modal logs."""

    def __init__(
        self,
        stage_name: str,
        log_every_n_steps: int = 100,
        rolling_window: int = 50,
    ):
        self.stage_name = stage_name
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.rolling_window = max(5, int(rolling_window))
        self._fit_start_time: float | None = None
        self._batch_start_time: float | None = None
        self._batch_durations: deque[float] = deque(maxlen=self.rolling_window)
        self._batches_per_epoch: int | None = None
        self._total_batches: int | None = None

    def _is_log_step(self, global_step: int) -> bool:
        if global_step <= 0:
            return False
        if global_step <= 10 and global_step in {1, 2, 5, 10}:
            return True
        return global_step % self.log_every_n_steps == 0

    def _avg_batch_seconds(self) -> float | None:
        if not self._batch_durations:
            return None
        return sum(self._batch_durations) / len(self._batch_durations)

    def on_train_start(self, trainer: Trainer, pl_module: L.LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        self._fit_start_time = time.perf_counter()
        self._batches_per_epoch = _normalize_batch_count(trainer.num_training_batches)
        if self._batches_per_epoch is not None and self._batches_per_epoch > 0:
            self._total_batches = self._batches_per_epoch * max(1, int(trainer.max_epochs))
        else:
            self._total_batches = None
        log.info(
            "[ETA][%s] train start: epochs=%s batches_per_epoch=%s total_batches=%s",
            self.stage_name,
            trainer.max_epochs,
            self._batches_per_epoch,
            self._total_batches,
        )

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: L.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, batch, batch_idx
        if not trainer.is_global_zero:
            return
        self._batch_start_time = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, outputs, batch
        if not trainer.is_global_zero:
            return

        now = time.perf_counter()
        if self._batch_start_time is not None:
            self._batch_durations.append(max(1e-6, now - self._batch_start_time))
        avg_batch_sec = self._avg_batch_seconds()
        global_step = int(trainer.global_step)

        if avg_batch_sec is None or not self._is_log_step(global_step):
            return

        elapsed = (now - self._fit_start_time) if self._fit_start_time is not None else None
        epoch_progress = batch_idx + 1
        epoch_eta = None
        total_eta = None
        if self._batches_per_epoch is not None:
            epoch_remaining = max(self._batches_per_epoch - epoch_progress, 0)
            epoch_eta = epoch_remaining * avg_batch_sec
        if self._total_batches is not None:
            total_remaining = max(self._total_batches - global_step, 0)
            total_eta = total_remaining * avg_batch_sec

        log.info(
            (
                "[ETA][%s] step=%s epoch=%s batch=%s/%s avg_step=%.2fs "
                "elapsed=%s eta_epoch=%s eta_total=%s"
            ),
            self.stage_name,
            global_step,
            int(trainer.current_epoch) + 1,
            epoch_progress,
            self._batches_per_epoch if self._batches_per_epoch is not None else "?",
            avg_batch_sec,
            _format_duration(elapsed),
            _format_duration(epoch_eta),
            _format_duration(total_eta),
        )

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: L.LightningModule) -> None:
        del pl_module
        if not trainer.is_global_zero:
            return
        val_batches = _normalize_batch_count(trainer.num_val_batches)
        log.info(
            "[ETA][%s] validation start: epoch=%s val_batches=%s",
            self.stage_name,
            int(trainer.current_epoch) + 1,
            val_batches,
        )

    def on_fit_end(self, trainer: Trainer, pl_module: L.LightningModule) -> None:
        del pl_module
        if not trainer.is_global_zero:
            return
        elapsed = None
        if self._fit_start_time is not None:
            elapsed = time.perf_counter() - self._fit_start_time
        log.info("[ETA][%s] fit complete: elapsed=%s", self.stage_name, _format_duration(elapsed))


@dataclass
class StageRuntimeMetrics:
    prep_seconds: float = 0.0
    first_batch_seconds: float | None = None
    steps_per_sec: float = 0.0
    tokens_per_sec: float = 0.0
    dataloader_wait_seconds: float = 0.0
    gpu_mem_peak_mb: float | None = None
    gpu_util_avg: float | None = None
    cpu_util_avg: float | None = None
    compile_graph_breaks: int | None = None
    compile_warmup_seconds: float | None = None


class RuntimeMetricsCallback(Callback):
    """Collects runtime health/perf metrics for time-to-signal optimization."""

    def __init__(
        self,
        stage_name: str,
        runtime_metrics: StageRuntimeMetrics,
        profile_enabled: bool,
        profile_steps: int,
        profile_dir: Path,
    ):
        self.stage_name = stage_name
        self.runtime_metrics = runtime_metrics
        self.profile_enabled = profile_enabled
        self.profile_steps = max(1, int(profile_steps))
        self.profile_dir = profile_dir
        self._fit_start: float | None = None
        self._last_batch_end: float | None = None
        self._batch_start: float | None = None
        self._batch_durations: list[float] = []
        self._batch_waits: list[float] = []
        self._token_counts: list[int] = []
        self._cpu_utils: list[float] = []
        self._gpu_utils: list[float] = []
        self._profiler: Any | None = None
        self._profile_step = 0

    def _tokens_in_batch(self, batch: Any) -> int:
        if isinstance(batch, dict):
            input_ids = batch.get("input_ids")
            if isinstance(input_ids, torch.Tensor):
                return int(input_ids.numel())
        return 0

    def on_train_start(self, trainer: Trainer, pl_module: L.LightningModule) -> None:
        del pl_module
        self._fit_start = time.perf_counter()
        if self.profile_enabled:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.profile_dir.mkdir(parents=True, exist_ok=True)
            trace_path = self.profile_dir / f"{self.stage_name}_trace.json"
            self._profiler = torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=self.profile_steps),
                on_trace_ready=lambda prof: prof.export_chrome_trace(str(trace_path)),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )
            self._profiler.start()

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: L.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del trainer, pl_module, batch_idx
        now = time.perf_counter()
        if self._last_batch_end is not None:
            self._batch_waits.append(max(0.0, now - self._last_batch_end))
        self._batch_start = now
        self._token_counts.append(self._tokens_in_batch(batch))
        cpu = _cpu_util_percent()
        gpu = _gpu_util_percent()
        if cpu is not None:
            self._cpu_utils.append(cpu)
        if gpu is not None:
            self._gpu_utils.append(gpu)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del trainer, pl_module, outputs, batch, batch_idx
        now = time.perf_counter()
        if self._batch_start is not None:
            duration = max(1e-6, now - self._batch_start)
            self._batch_durations.append(duration)
            if self.runtime_metrics.first_batch_seconds is None:
                self.runtime_metrics.first_batch_seconds = duration
                if self.runtime_metrics.compile_warmup_seconds is None:
                    self.runtime_metrics.compile_warmup_seconds = duration
        self._last_batch_end = now
        if self._profiler is not None and self._profile_step < self.profile_steps + 2:
            self._profiler.step()
            self._profile_step += 1

    def on_fit_end(self, trainer: Trainer, pl_module: L.LightningModule) -> None:
        del trainer, pl_module
        if self._profiler is not None:
            self._profiler.stop()

        elapsed = 0.0
        if self._fit_start is not None:
            elapsed = max(1e-6, time.perf_counter() - self._fit_start)

        total_tokens = float(sum(self._token_counts))
        total_steps = float(len(self._batch_durations))
        avg_step = sum(self._batch_durations) / max(len(self._batch_durations), 1)
        self.runtime_metrics.steps_per_sec = total_steps / elapsed
        self.runtime_metrics.tokens_per_sec = total_tokens / elapsed
        self.runtime_metrics.dataloader_wait_seconds = (
            sum(self._batch_waits) / max(len(self._batch_waits), 1)
        )
        self.runtime_metrics.cpu_util_avg = (
            sum(self._cpu_utils) / len(self._cpu_utils) if self._cpu_utils else None
        )
        self.runtime_metrics.gpu_util_avg = (
            sum(self._gpu_utils) / len(self._gpu_utils) if self._gpu_utils else None
        )
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
        log.info(
            (
                "[RUNTIME][%s] first_batch=%s avg_step=%.4fs steps/s=%.3f tokens/s=%.2f "
                "wait=%.4fs gpu_mem_peak_mb=%s cpu_util=%s gpu_util=%s"
            ),
            self.stage_name,
            _format_duration(self.runtime_metrics.first_batch_seconds),
            avg_step,
            self.runtime_metrics.steps_per_sec,
            self.runtime_metrics.tokens_per_sec,
            self.runtime_metrics.dataloader_wait_seconds,
            self.runtime_metrics.gpu_mem_peak_mb,
            self.runtime_metrics.cpu_util_avg,
            self.runtime_metrics.gpu_util_avg,
        )


def _batch_to_model_inputs(
    stage_name: str,
    batch: dict[str, torch.Tensor],
    model_device: torch.device,
    encoder: LFQEncoder | None = None,
    max_total_bits: int | None = None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    input_ids = batch["input_ids"].to(model_device)
    mask = build_causal_mask(input_ids.shape[1], model_device)
    if stage_name == "dabe_lm":
        if encoder is None or max_total_bits is None:
            raise ValueError("encoder and max_total_bits are required for DABE diagnostics")
        bit_widths = batch["bit_widths"].to(model_device)
        flat_ids = input_ids.reshape(-1)
        flat_widths = bit_widths.reshape(-1).clamp(min=1, max=encoder.max_bit_width)
        with torch.no_grad():
            hidden = encoder.token_embed(flat_ids)
            logits = encoder.proj(hidden)[:, : encoder.max_bit_width]
            code = (torch.tanh(logits) > 0).long()
        positions = torch.arange(encoder.max_bit_width, device=model_device)
        code = code * (positions.unsqueeze(0) < flat_widths.unsqueeze(1)).long()
        shifts = torch.arange(encoder.num_header_bits, device=model_device, dtype=torch.long)
        header = ((flat_widths.unsqueeze(1) >> shifts) & 1).long()
        encoded = torch.cat([header, code], dim=1)
        if encoded.shape[1] < max_total_bits:
            encoded = F.pad(encoded, (0, max_total_bits - encoded.shape[1]))
        elif encoded.shape[1] > max_total_bits:
            encoded = encoded[:, :max_total_bits]
        bits = encoded.view(input_ids.shape[0], input_ids.shape[1], max_total_bits)
        return (bits,), {"bit_widths": bit_widths, "mask": mask}
    return (input_ids,), {"mask": mask}


def _compile_graph_break_count(
    model: torch.nn.Module,
    stage_name: str,
    batch: dict[str, torch.Tensor],
    encoder: LFQEncoder | None = None,
    max_total_bits: int | None = None,
) -> int | None:
    try:
        import torch._dynamo as dynamo
    except Exception:
        return None
    try:
        args, kwargs = _batch_to_model_inputs(
            stage_name=stage_name,
            batch=batch,
            model_device=next(model.parameters()).device,
            encoder=encoder,
            max_total_bits=max_total_bits,
        )
        explain_output = dynamo.explain(model)(*args, **kwargs)
        return int(getattr(explain_output, "graph_break_count", 0))
    except Exception:
        return None


def _maybe_compile_model(
    model: torch.nn.Module,
    enabled: bool,
    mode: str = "reduce-overhead",
) -> tuple[torch.nn.Module, bool]:
    if not enabled or not hasattr(torch, "compile"):
        return model, False
    try:
        compiled = torch.compile(model, mode=mode)
    except Exception as exc:
        log.warning("torch.compile disabled due to compile error: %s", exc)
        return model, False
    return compiled, True


@dataclass
class TokenizerTrainingResult:
    artifact_path: Path
    vocab_size: int
    avg_train_loss: float
    compression_ratio_vs_bpe: float
    density_distribution: dict[str, float]
    same_hamming: dict[str, float]
    diff_hamming: dict[str, float]
    prep_seconds: float = 0.0


def train_lfq_tokenizer_artifact(
    config: dict,
    output_dir: Path,
    wandb_logger: Any | None = None,
) -> TokenizerTrainingResult:
    feasibility_cfg = config.get("feasibility", {})
    tokenizer_cfg = feasibility_cfg.get("tokenizer_training", {})
    optimization_cfg = config.get("optimization", {})
    cache_mode = _resolve_cache_mode(config)
    cache_root = _resolve_data_cache_root(config, output_dir)
    dataset_name = tokenizer_cfg.get("dataset_name", "roneneldan/TinyStories")
    train_split = tokenizer_cfg.get("train_split", "train")
    eval_split = tokenizer_cfg.get("eval_split", "validation")
    text_field = str(tokenizer_cfg.get("text_field", "text"))
    allow_synthetic_fallback = bool(tokenizer_cfg.get("allow_synthetic_fallback", True))
    streaming = bool(tokenizer_cfg.get("streaming", True))
    prep_start = time.perf_counter()

    train_texts = _load_text_dataset_samples(
        dataset_name=dataset_name,
        split=train_split,
        max_samples=int(tokenizer_cfg.get("train_samples", 2000)),
        seed=int(config.get("experiment", {}).get("seed", 42)),
        text_field=text_field,
        allow_synthetic_fallback=allow_synthetic_fallback,
        streaming=streaming,
        cache_mode=cache_mode,
        cache_root=cache_root,
    )
    eval_texts = _load_text_dataset_samples(
        dataset_name=dataset_name,
        split=eval_split,
        max_samples=int(tokenizer_cfg.get("eval_samples", 256)),
        seed=int(config.get("experiment", {}).get("seed", 42)) + 1,
        text_field=text_field,
        allow_synthetic_fallback=allow_synthetic_fallback,
        streaming=streaming,
        cache_mode=cache_mode,
        cache_root=cache_root,
    )

    router = EntropyRouter(config.get("router", {}))
    spans, all_widths = _collect_spans(router, train_texts)
    span_texts = [span.text for span in spans]

    encoder_cfg = dict(config.get("encoder", {}))
    encoder_cfg["vocab_size"] = int(
        tokenizer_cfg.get("vocab_size", encoder_cfg.get("vocab_size", 4096))
    )
    encoder = LFQEncoder(encoder_cfg)
    fit_stats = encoder.train_tokenizer(
        spans=span_texts,
        bit_widths=all_widths,
        epochs=int(tokenizer_cfg.get("epochs", 2)),
        batch_size=int(tokenizer_cfg.get("batch_size", 128)),
        learning_rate=float(tokenizer_cfg.get("learning_rate", 1e-3)),
        device=str(tokenizer_cfg.get("device", "cpu")),
    )

    artifact_path = Path(feasibility_cfg.get(
        "tokenizer_artifact_path",
        output_dir / "artifacts" / "lfq_tokenizer.pt",
    ))
    artifact_path = encoder.export_artifact(artifact_path)

    bpe_model = feasibility_cfg.get("bpe_model_name", "gpt2")
    bpe_tokenizer = AutoTokenizer.from_pretrained(
        bpe_model,
        use_fast=bool(optimization_cfg.get("force_fast_tokenizer", True)),
    )
    if bool(optimization_cfg.get("force_fast_tokenizer", True)) and not bool(
        getattr(bpe_tokenizer, "is_fast", False)
    ):
        raise RuntimeError(f"Expected fast tokenizer for {bpe_model}, but slow tokenizer was loaded.")
    spans_per_text: list[list[Span]] = []
    bit_widths_per_text: list[list[int]] = []
    bpe_counts: list[int] = []
    for text in eval_texts:
        scores = router.score(text)
        segmented = router.segment(text, scores)
        spans_per_text.append(segmented)
        header_bits = int(config.get("encoder", {}).get("num_header_bits", 4))
        bit_widths_per_text.append([span.bit_width + header_bits for span in segmented])
        bpe_counts.append(len(bpe_tokenizer.encode(text, add_special_tokens=False)))

    compression_report = CompressionMetrics().compute(
        texts=eval_texts,
        bpe_token_counts=bpe_counts,
        spans_per_text=spans_per_text,
        bit_widths_per_text=bit_widths_per_text,
    )

    token_occurrences: dict[str, list[int]] = {}
    for span in spans:
        token_occurrences.setdefault(encoder._normalize_span(span.text), []).append(span.bit_width)

    same_distances: list[int] = []
    diff_distances: list[int] = []
    random.seed(int(config.get("experiment", {}).get("seed", 42)))
    frequent_tokens = [
        token for token, widths in token_occurrences.items()
        if token in encoder.token_to_id and len(widths) >= 2
    ]
    for token in frequent_tokens[:200]:
        widths = token_occurrences[token][:3]
        codes = [encoder.encode(token, width) for width in widths]
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                same_distances.append(_hamming_distance(codes[i], codes[j]))

    vocab_tokens = encoder.id_to_token[1:]
    for _ in range(min(300, len(vocab_tokens) * 2)):
        if len(vocab_tokens) < 2:
            break
        first, second = random.sample(vocab_tokens, 2)
        first_width = encoder.token_id_to_width[encoder.lookup_token_id(first)]
        second_width = encoder.token_id_to_width[encoder.lookup_token_id(second)]
        code_a = encoder.encode(first, first_width)
        code_b = encoder.encode(second, second_width)
        diff_distances.append(_hamming_distance(code_a, code_b))

    density_distribution = _compute_width_distribution(all_widths)
    result = TokenizerTrainingResult(
        artifact_path=artifact_path,
        vocab_size=int(fit_stats["vocab_size"]),
        avg_train_loss=float(fit_stats["avg_train_loss"]),
        compression_ratio_vs_bpe=float(compression_report.compression_ratio_vs_bpe),
        density_distribution=density_distribution,
        same_hamming=_hamming_summary(same_distances),
        diff_hamming=_hamming_summary(diff_distances),
        prep_seconds=time.perf_counter() - prep_start,
    )

    if wandb_logger is not None:
        wandb_logger.log_metrics({
            "tokenizer/vocab_size": result.vocab_size,
            "tokenizer/train_loss": result.avg_train_loss,
            "tokenizer/compression_ratio_vs_bpe": result.compression_ratio_vs_bpe,
            "tokenizer/hamming_same_mean": result.same_hamming["mean"],
            "tokenizer/hamming_diff_mean": result.diff_hamming["mean"],
            "lang/en/tokenizer_samples": len(train_texts),
        })
        for key, value in result.density_distribution.items():
            wandb_logger.log_metrics({f"tokenizer/density_{key}": value})

    return result


class DABESequenceDataset(TorchDataset):
    def __init__(
        self,
        texts: Sequence[str],
        router: EntropyRouter,
        encoder: LFQEncoder,
        seq_len: int,
        progress_label: str | None = None,
        progress_every_texts: int = 200,
        cache_dir: Path | None = None,
        cache_mode: str = "none",
        split_name: str = "train",
        preencode_to_disk: bool = False,
    ):
        self._inputs: np.ndarray | None = None
        self._targets: np.ndarray | None = None
        self._bit_widths: np.ndarray | None = None
        self.samples: list[dict[str, torch.Tensor]] = []
        total_texts = len(texts)
        prep_start = time.perf_counter()
        every = max(1, int(progress_every_texts))
        cache_prefix = "dabe"
        cache_key = _hash_key([cache_prefix, split_name, str(seq_len), str(len(texts))])
        if (
            preencode_to_disk
            and cache_dir is not None
            and cache_mode == "reuse"
            and (cache_dir / f"{cache_key}_inputs.npy").exists()
        ):
            self._inputs = np.load(cache_dir / f"{cache_key}_inputs.npy", mmap_mode="r")
            self._targets = np.load(cache_dir / f"{cache_key}_targets.npy", mmap_mode="r")
            self._bit_widths = np.load(cache_dir / f"{cache_key}_widths.npy", mmap_mode="r")
            return

        window_inputs: list[np.ndarray] = []
        window_targets: list[np.ndarray] = []
        window_widths: list[np.ndarray] = []

        for idx, text in enumerate(texts, start=1):
            scores = router.score(text)
            spans = router.segment(text, scores)
            if len(spans) < 2:
                continue
            ids = np.fromiter(
                (encoder.lookup_token_id(span.text) for span in spans),
                dtype=np.int32,
            )
            widths = np.fromiter((int(span.bit_width) for span in spans), dtype=np.int32)
            input_windows, target_windows = _sequence_windows_np(ids, seq_len)
            width_windows, _ = _sequence_windows_np(widths, seq_len)
            if input_windows.size == 0:
                continue
            window_inputs.append(input_windows)
            window_targets.append(target_windows)
            window_widths.append(width_windows)
            if progress_label is not None and (idx % every == 0 or idx == total_texts):
                elapsed = time.perf_counter() - prep_start
                avg = elapsed / max(1, idx)
                sample_count = int(sum(arr.shape[0] for arr in window_inputs))
                eta = avg * max(total_texts - idx, 0)
                log.info(
                    "[PREP][%s] texts=%s/%s samples=%s elapsed=%s eta=%s",
                    progress_label,
                    idx,
                    total_texts,
                    sample_count,
                    _format_duration(elapsed),
                    _format_duration(eta),
                )

        if window_inputs:
            self._inputs = np.concatenate(window_inputs, axis=0)
            self._targets = np.concatenate(window_targets, axis=0)
            self._bit_widths = np.concatenate(window_widths, axis=0)
        else:
            self._inputs = np.empty((0, seq_len), dtype=np.int32)
            self._targets = np.empty((0, seq_len), dtype=np.int32)
            self._bit_widths = np.empty((0, seq_len), dtype=np.int32)

        if preencode_to_disk and cache_dir is not None and cache_mode in {"build", "reuse"}:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_dir / f"{cache_key}_inputs.npy", self._inputs)
            np.save(cache_dir / f"{cache_key}_targets.npy", self._targets)
            np.save(cache_dir / f"{cache_key}_widths.npy", self._bit_widths)

    def __len__(self) -> int:
        if self._inputs is not None:
            return int(self._inputs.shape[0])
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self._inputs is not None and self._targets is not None and self._bit_widths is not None:
            return {
                "input_ids": torch.from_numpy(np.asarray(self._inputs[idx], dtype=np.int64)),
                "targets": torch.from_numpy(np.asarray(self._targets[idx], dtype=np.int64)),
                "bit_widths": torch.from_numpy(np.asarray(self._bit_widths[idx], dtype=np.int64)),
            }
        return self.samples[idx]


class BPESequenceDataset(TorchDataset):
    def __init__(
        self,
        texts: Sequence[str],
        tokenizer: Any,
        seq_len: int,
        progress_label: str | None = None,
        progress_every_texts: int = 200,
        cache_dir: Path | None = None,
        cache_mode: str = "none",
        split_name: str = "train",
        preencode_to_disk: bool = False,
        tokenizer_batch_size: int = 64,
    ):
        self._inputs: np.ndarray | None = None
        self._targets: np.ndarray | None = None
        self.samples: list[dict[str, torch.Tensor]] = []
        total_texts = len(texts)
        prep_start = time.perf_counter()
        every = max(1, int(progress_every_texts))
        cache_prefix = "bpe"
        cache_key = _hash_key([cache_prefix, split_name, str(seq_len), str(len(texts))])
        if (
            preencode_to_disk
            and cache_dir is not None
            and cache_mode == "reuse"
            and (cache_dir / f"{cache_key}_inputs.npy").exists()
        ):
            self._inputs = np.load(cache_dir / f"{cache_key}_inputs.npy", mmap_mode="r")
            self._targets = np.load(cache_dir / f"{cache_key}_targets.npy", mmap_mode="r")
            return

        window_inputs: list[np.ndarray] = []
        window_targets: list[np.ndarray] = []

        for batch_start in range(0, total_texts, max(1, int(tokenizer_batch_size))):
            batch_texts = list(texts[batch_start : batch_start + max(1, int(tokenizer_batch_size))])
            if callable(tokenizer):
                encoded = tokenizer(
                    batch_texts,
                    add_special_tokens=False,
                )
                ids_per_text = (
                    encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
                )
            else:
                ids_per_text = [
                    tokenizer.encode(text, add_special_tokens=False)
                    for text in batch_texts
                ]
            for text_offset, token_ids in enumerate(ids_per_text):
                token_array = np.asarray(token_ids, dtype=np.int32)
                input_windows, target_windows = _sequence_windows_np(token_array, seq_len)
                if input_windows.size > 0:
                    window_inputs.append(input_windows)
                    window_targets.append(target_windows)
                idx = batch_start + text_offset + 1
                if progress_label is None or (idx % every != 0 and idx != total_texts):
                    continue
                elapsed = time.perf_counter() - prep_start
                avg = elapsed / max(1, idx)
                sample_count = int(sum(arr.shape[0] for arr in window_inputs))
                eta = avg * max(total_texts - idx, 0)
                log.info(
                    "[PREP][%s] texts=%s/%s samples=%s elapsed=%s eta=%s",
                    progress_label,
                    idx,
                    total_texts,
                    sample_count,
                    _format_duration(elapsed),
                    _format_duration(eta),
                )

        if window_inputs:
            self._inputs = np.concatenate(window_inputs, axis=0)
            self._targets = np.concatenate(window_targets, axis=0)
        else:
            self._inputs = np.empty((0, seq_len), dtype=np.int32)
            self._targets = np.empty((0, seq_len), dtype=np.int32)

        if preencode_to_disk and cache_dir is not None and cache_mode in {"build", "reuse"}:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_dir / f"{cache_key}_inputs.npy", self._inputs)
            np.save(cache_dir / f"{cache_key}_targets.npy", self._targets)

    def __len__(self) -> int:
        if self._inputs is not None:
            return int(self._inputs.shape[0])
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self._inputs is not None and self._targets is not None:
            return {
                "input_ids": torch.from_numpy(np.asarray(self._inputs[idx], dtype=np.int64)),
                "targets": torch.from_numpy(np.asarray(self._targets[idx], dtype=np.int64)),
            }
        return self.samples[idx]


class DABEFeasibilityDataModule(L.LightningDataModule):
    def __init__(
        self,
        texts_train: Sequence[str],
        texts_val: Sequence[str],
        router: EntropyRouter,
        encoder: LFQEncoder,
        seq_len: int,
        batch_size: int,
        num_workers: int,
        seed: int = 42,
        progress_every_texts: int = 200,
        cache_dir: Path | None = None,
        cache_mode: str = "none",
        preencode_to_disk: bool = False,
        device_profile: str = "mps_safe",
        accelerator: str = "cpu",
    ):
        super().__init__()
        self.texts_train = texts_train
        self.texts_val = texts_val
        self.router = router
        self.encoder = encoder
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = int(seed)
        self.progress_every_texts = int(progress_every_texts)
        self.cache_dir = cache_dir
        self.cache_mode = cache_mode
        self.preencode_to_disk = bool(preencode_to_disk)
        self.device_profile = str(device_profile)
        self.accelerator = str(accelerator)
        self._is_setup = False
        self.prep_seconds = 0.0

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self._is_setup:
            return
        prep_start = time.perf_counter()
        log.info(
            "[PREP][dabe_datamodule] building datasets: train_texts=%s val_texts=%s seq_len=%s",
            len(self.texts_train),
            len(self.texts_val),
            self.seq_len,
        )
        self.train_dataset = DABESequenceDataset(
            texts=self.texts_train,
            router=self.router,
            encoder=self.encoder,
            seq_len=self.seq_len,
            progress_label="dabe_train",
            progress_every_texts=self.progress_every_texts,
            cache_dir=self.cache_dir,
            cache_mode=self.cache_mode,
            split_name="train",
            preencode_to_disk=self.preencode_to_disk,
        )
        self.val_dataset = DABESequenceDataset(
            texts=self.texts_val,
            router=self.router,
            encoder=self.encoder,
            seq_len=self.seq_len,
            progress_label="dabe_val",
            progress_every_texts=self.progress_every_texts,
            cache_dir=self.cache_dir,
            cache_mode=self.cache_mode,
            split_name="val",
            preencode_to_disk=self.preencode_to_disk,
        )
        self.prep_seconds = time.perf_counter() - prep_start
        self._is_setup = True
        log.info(
            "[PREP][dabe_datamodule] built datasets: train_samples=%s val_samples=%s prep=%s",
            len(self.train_dataset),
            len(self.val_dataset),
            _format_duration(self.prep_seconds),
        )

    def _dataloader_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"num_workers": self.num_workers}
        cuda_fast = self.device_profile == "cuda_fast" and self.accelerator == "gpu"
        if cuda_fast and self.num_workers > 0:
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


class BPEFeasibilityDataModule(L.LightningDataModule):
    def __init__(
        self,
        texts_train: Sequence[str],
        texts_val: Sequence[str],
        tokenizer: Any,
        seq_len: int,
        batch_size: int,
        num_workers: int,
        seed: int = 42,
        progress_every_texts: int = 200,
        cache_dir: Path | None = None,
        cache_mode: str = "none",
        preencode_to_disk: bool = False,
        tokenizer_batch_size: int = 64,
        device_profile: str = "mps_safe",
        accelerator: str = "cpu",
    ):
        super().__init__()
        self.texts_train = texts_train
        self.texts_val = texts_val
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = int(seed)
        self.progress_every_texts = int(progress_every_texts)
        self.cache_dir = cache_dir
        self.cache_mode = cache_mode
        self.preencode_to_disk = bool(preencode_to_disk)
        self.tokenizer_batch_size = int(tokenizer_batch_size)
        self.device_profile = str(device_profile)
        self.accelerator = str(accelerator)
        self._is_setup = False
        self.prep_seconds = 0.0

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self._is_setup:
            return
        prep_start = time.perf_counter()
        log.info(
            "[PREP][bpe_datamodule] building datasets: train_texts=%s val_texts=%s seq_len=%s",
            len(self.texts_train),
            len(self.texts_val),
            self.seq_len,
        )
        self.train_dataset = BPESequenceDataset(
            self.texts_train,
            self.tokenizer,
            self.seq_len,
            progress_label="bpe_train",
            progress_every_texts=self.progress_every_texts,
            cache_dir=self.cache_dir,
            cache_mode=self.cache_mode,
            split_name="train",
            preencode_to_disk=self.preencode_to_disk,
            tokenizer_batch_size=self.tokenizer_batch_size,
        )
        self.val_dataset = BPESequenceDataset(
            self.texts_val,
            self.tokenizer,
            self.seq_len,
            progress_label="bpe_val",
            progress_every_texts=self.progress_every_texts,
            cache_dir=self.cache_dir,
            cache_mode=self.cache_mode,
            split_name="val",
            preencode_to_disk=self.preencode_to_disk,
            tokenizer_batch_size=self.tokenizer_batch_size,
        )
        self.prep_seconds = time.perf_counter() - prep_start
        self._is_setup = True
        log.info(
            "[PREP][bpe_datamodule] built datasets: train_samples=%s val_samples=%s prep=%s",
            len(self.train_dataset),
            len(self.val_dataset),
            _format_duration(self.prep_seconds),
        )

    def _dataloader_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"num_workers": self.num_workers}
        cuda_fast = self.device_profile == "cuda_fast" and self.accelerator == "gpu"
        if cuda_fast and self.num_workers > 0:
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


class DABECausalLMModule(L.LightningModule):
    def __init__(self, config: dict, encoder: LFQEncoder):
        super().__init__()
        self.config = config
        self.training_cfg = config.get("training", {})
        self.optimization_cfg = config.get("optimization", {})
        self.v4_cfg = self.optimization_cfg.get("v4_distill", {})
        self.encoder = encoder
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        self.model = DABETransformer(config.get("backbone", {}))
        self.mtp_heads = int(self.v4_cfg.get("mtp_heads", 0))
        self.mtp_weight = float(self.v4_cfg.get("mtp_loss_weight", 0.1))
        self.nan_batches = 0

    def _ids_to_bits_loop(self, input_ids: torch.Tensor, bit_widths: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        max_total_bits = int(self.model.max_bit_width)
        flat_ids = input_ids.reshape(-1)
        flat_widths = bit_widths.reshape(-1)
        flat_bits = torch.zeros(
            (flat_ids.shape[0], max_total_bits),
            dtype=torch.long,
            device=self.device,
        )

        for width in torch.unique(flat_widths):
            indices = torch.where(flat_widths == width)[0]
            if indices.numel() == 0:
                continue
            token_ids = flat_ids[indices]
            encoded = self.encoder.encode_token_ids(token_ids, int(width.item())).to(self.device)
            usable = min(max_total_bits, encoded.shape[-1])
            flat_bits[indices, :usable] = encoded[:, :usable]

        return flat_bits.view(batch_size, seq_len, max_total_bits)

    def _ids_to_bits(self, input_ids: torch.Tensor, bit_widths: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        max_total_bits = int(self.model.max_bit_width)
        flat_ids = input_ids.reshape(-1)
        flat_widths = bit_widths.reshape(-1).clamp(min=1, max=self.encoder.max_bit_width)

        hidden = self.encoder.token_embed(flat_ids)
        logits = self.encoder.proj(hidden)[:, : self.encoder.max_bit_width]
        code_bits = (torch.tanh(logits) > 0).long()
        positions = torch.arange(self.encoder.max_bit_width, device=flat_ids.device).unsqueeze(0)
        code_mask = positions < flat_widths.unsqueeze(1)
        code_bits = code_bits * code_mask.long()

        shifts = torch.arange(
            self.encoder.num_header_bits, device=flat_ids.device, dtype=torch.long
        ).unsqueeze(0)
        header_bits = ((flat_widths.unsqueeze(1) >> shifts) & 1).long()
        encoded = torch.cat([header_bits, code_bits], dim=1)

        if encoded.shape[1] < max_total_bits:
            encoded = F.pad(encoded, (0, max_total_bits - encoded.shape[1]))
        elif encoded.shape[1] > max_total_bits:
            encoded = encoded[:, :max_total_bits]

        return encoded.view(batch_size, seq_len, max_total_bits)

    def _mtp_loss(self, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.mtp_heads <= 0:
            return torch.tensor(0.0, device=hidden.device)
        mtp_logits = self.model.compute_mtp_logits(hidden)
        if not mtp_logits:
            return torch.tensor(0.0, device=hidden.device)
        losses: list[torch.Tensor] = []
        for head_idx, logits in enumerate(mtp_logits):
            shift = head_idx + 1
            if targets.shape[1] <= shift:
                continue
            shifted_targets = targets[:, shift:]
            shifted_logits = logits[:, :-shift, :]
            losses.append(
                F.cross_entropy(
                    shifted_logits.reshape(-1, shifted_logits.shape[-1]),
                    shifted_targets.reshape(-1),
                )
            )
        if not losses:
            return torch.tensor(0.0, device=hidden.device)
        return torch.stack(losses).mean()

    def _loss_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        input_ids = batch["input_ids"].to(self.device)
        targets = batch["targets"].to(self.device)
        bit_widths = batch["bit_widths"].to(self.device)
        bits = self._ids_to_bits(input_ids, bit_widths)
        mask = build_causal_mask(input_ids.shape[1], self.device)
        logits, hidden = self.model(bits, bit_widths=bit_widths, mask=mask)
        base_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        mtp_loss = self._mtp_loss(hidden, targets)
        loss = base_loss + self.mtp_weight * mtp_loss

        is_finite = torch.isfinite(loss)
        if not bool(is_finite):
            self.nan_batches += 1
            raise RuntimeError("Non-finite loss in DABE feasibility training.")

        self.log(f"dabe/{stage}_loss", loss, prog_bar=True)
        self.log(f"dabe/{stage}_base_loss", base_loss)
        self.log(f"dabe/{stage}_mtp_loss", mtp_loss)
        self.log(f"dabe/{stage}_stable", torch.tensor(1.0, device=self.device))
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._loss_step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._loss_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.training_cfg.get("learning_rate", 3e-4)),
            weight_decay=float(self.training_cfg.get("weight_decay", 0.01)),
        )
        return optimizer


class BPECausalLMModule(L.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.training_cfg = config.get("training", {})
        self.optimization_cfg = config.get("optimization", {})
        self.v4_cfg = self.optimization_cfg.get("v4_distill", {})
        self.model = TokenTransformerLM(config.get("backbone", {}))
        self.mtp_heads = int(self.v4_cfg.get("mtp_heads", 0))
        self.mtp_weight = float(self.v4_cfg.get("mtp_loss_weight", 0.1))
        self.nan_batches = 0

    def _mtp_loss(self, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.mtp_heads <= 0:
            return torch.tensor(0.0, device=hidden.device)
        mtp_logits = self.model.compute_mtp_logits(hidden)
        if not mtp_logits:
            return torch.tensor(0.0, device=hidden.device)
        losses: list[torch.Tensor] = []
        for head_idx, logits in enumerate(mtp_logits):
            shift = head_idx + 1
            if targets.shape[1] <= shift:
                continue
            shifted_targets = targets[:, shift:]
            shifted_logits = logits[:, :-shift, :]
            losses.append(
                F.cross_entropy(
                    shifted_logits.reshape(-1, shifted_logits.shape[-1]),
                    shifted_targets.reshape(-1),
                )
            )
        if not losses:
            return torch.tensor(0.0, device=hidden.device)
        return torch.stack(losses).mean()

    def _loss_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        input_ids = batch["input_ids"].to(self.device)
        targets = batch["targets"].to(self.device)
        mask = build_causal_mask(input_ids.shape[1], self.device)
        logits, hidden = self.model(input_ids, mask=mask)
        base_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        mtp_loss = self._mtp_loss(hidden, targets)
        loss = base_loss + self.mtp_weight * mtp_loss

        is_finite = torch.isfinite(loss)
        if not bool(is_finite):
            self.nan_batches += 1
            raise RuntimeError("Non-finite loss in BPE feasibility training.")

        self.log(f"bpe/{stage}_loss", loss, prog_bar=True)
        self.log(f"bpe/{stage}_base_loss", base_loss)
        self.log(f"bpe/{stage}_mtp_loss", mtp_loss)
        self.log(f"bpe/{stage}_stable", torch.tensor(1.0, device=self.device))
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._loss_step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._loss_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.training_cfg.get("learning_rate", 3e-4)),
            weight_decay=float(self.training_cfg.get("weight_decay", 0.01)),
        )
        return optimizer


class AdapterCausalLMModule(L.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.training_cfg = config.get("training", {})
        self.model = AdapterTokenTransformerLM(config.get("backbone", {}))
        self.nan_batches = 0

    def _loss_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        input_ids = batch["input_ids"].to(self.device)
        targets = batch["targets"].to(self.device)
        mask = build_causal_mask(input_ids.shape[1], self.device)
        logits, _, stats = self.model(input_ids, mask=mask)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

        if not bool(torch.isfinite(loss)):
            self.nan_batches += 1
            raise RuntimeError("Non-finite loss in adapter feasibility training.")

        mean_width = stats["bit_widths"].float().mean()
        width_std = stats["bit_widths"].float().std(unbiased=False)
        gate_mean = stats["gate_mean"]
        residual_scale = stats["residual_scale"].float()
        self.log(f"adapter/{stage}_loss", loss, prog_bar=True)
        self.log(f"adapter/{stage}_mean_width", mean_width)
        self.log(f"adapter/{stage}_width_std", width_std)
        self.log(f"adapter/{stage}_gate_mean", gate_mean)
        self.log(f"adapter/{stage}_residual_scale", residual_scale)
        self.log(f"adapter/{stage}_stable", torch.tensor(1.0, device=self.device))
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._loss_step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._loss_step(batch, "val")

    def configure_optimizers(self):
        base_lr = float(self.training_cfg.get("learning_rate", 3e-4))
        base_wd = float(self.training_cfg.get("weight_decay", 0.01))
        adapter_lr = float(self.training_cfg.get("adapter_learning_rate", base_lr))
        adapter_wd = float(self.training_cfg.get("adapter_weight_decay", base_wd))

        adapter_params: list[torch.nn.Parameter] = []
        base_params: list[torch.nn.Parameter] = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("adapter."):
                adapter_params.append(param)
            else:
                base_params.append(param)

        if adapter_params and (adapter_lr != base_lr or adapter_wd != base_wd):
            optimizer = torch.optim.AdamW(
                [
                    {"params": base_params, "lr": base_lr, "weight_decay": base_wd},
                    {"params": adapter_params, "lr": adapter_lr, "weight_decay": adapter_wd},
                ]
            )
            return optimizer

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=base_lr,
            weight_decay=base_wd,
        )
        return optimizer


def _build_trainer(
    config: dict,
    output_dir: Path,
    wandb_logger: Any | None,
    accelerator: str,
    devices: Any,
    stage_name: str,
    monitor: str,
    runtime_metrics: StageRuntimeMetrics | None = None,
) -> Trainer:
    training_cfg = config.get("training", {})
    logging_cfg = config.get("logging", {})
    evaluation_cfg = config.get("evaluation", {})
    optimization_cfg = config.get("optimization", {})
    eta_log_every_n_steps = int(logging_cfg.get("eta_log_every_n_steps", 100))
    save_top_k = int(evaluation_cfg.get("save_top_k", 1))
    callbacks: list[Callback] = [ETAProgressCallback(stage_name, eta_log_every_n_steps)]
    if runtime_metrics is not None:
        profile_cfg = optimization_cfg.get("profile", {})
        callbacks.append(
            RuntimeMetricsCallback(
                stage_name=stage_name,
                runtime_metrics=runtime_metrics,
                profile_enabled=bool(profile_cfg.get("enabled", False)),
                profile_steps=int(profile_cfg.get("steps", 200)),
                profile_dir=output_dir / "profiler",
            )
        )
    if save_top_k != 0:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(output_dir / f"{stage_name}_checkpoints"),
                filename="{epoch}-{step}",
                monitor=monitor,
                save_top_k=save_top_k,
                auto_insert_metric_name=False,
            )
        )
    grad_preset = str(optimization_cfg.get("grad_accumulation_preset", "none")).lower()
    base_batch_size = int(training_cfg.get("batch_size", 4))
    accumulate_grad_batches = 1
    if grad_preset in {"auto_small", "auto_medium"}:
        target_batch = 32 if grad_preset == "auto_small" else 64
        accumulate_grad_batches = max(1, int(math.ceil(target_batch / max(base_batch_size, 1))))

    return Trainer(
        max_epochs=int(training_cfg.get("max_epochs", 1)),
        accelerator=accelerator,
        devices=devices,
        logger=wandb_logger,
        callbacks=callbacks,
        gradient_clip_val=training_cfg.get("grad_clip"),
        accumulate_grad_batches=accumulate_grad_batches,
        precision=_resolve_precision(config, accelerator),
        log_every_n_steps=int(logging_cfg.get("log_every_n_steps", 10)),
        default_root_dir=str(output_dir),
        enable_progress_bar=bool(logging_cfg.get("enable_progress_bar", True)),
    )


def run_feasibility_experiment(
    config: dict,
    output_dir: Path,
    wandb_logger: Any | None,
    accelerator: str,
    devices: Any,
) -> dict:
    feasibility_cfg = config.get("feasibility", {})
    lm_data_cfg = feasibility_cfg.get("lm_data", {})
    optimization_cfg = config.get("optimization", {})
    data_cache_cfg = optimization_cfg.get("data_cache", {})
    compile_cfg = optimization_cfg.get("compile", {})
    stage = config.get("experiment", {}).get("feasibility_stage", "all")
    cache_root = _resolve_data_cache_root(config, output_dir)
    cache_mode = _resolve_cache_mode(config)
    device_profile = str(optimization_cfg.get("device_profile", "mps_safe"))
    preencode_to_disk = bool(data_cache_cfg.get("preencode_to_disk", False))

    result: dict[str, Any] = {"track": "feasibility", "stage": stage}
    tokenizer_result: TokenizerTrainingResult | None = None
    artifact_path = Path(feasibility_cfg.get(
        "tokenizer_artifact_path",
        output_dir / "artifacts" / "lfq_tokenizer.pt",
    ))

    # Stage-local execution should only retrain tokenizer when explicitly requested
    # (tokenizer/all) or when the shared artifact is missing.
    requires_tokenizer = stage in {"tokenizer", "all"} or not artifact_path.exists()
    if requires_tokenizer:
        tokenizer_result = train_lfq_tokenizer_artifact(config, output_dir, wandb_logger)
        artifact_path = tokenizer_result.artifact_path
        result["tokenizer"] = {
            "artifact_path": str(tokenizer_result.artifact_path),
            "vocab_size": tokenizer_result.vocab_size,
            "avg_train_loss": tokenizer_result.avg_train_loss,
            "compression_ratio_vs_bpe": tokenizer_result.compression_ratio_vs_bpe,
            "density_distribution": tokenizer_result.density_distribution,
            "hamming_same": tokenizer_result.same_hamming,
            "hamming_diff": tokenizer_result.diff_hamming,
            "runtime": {
                "prep_seconds": tokenizer_result.prep_seconds,
                "first_batch_seconds": None,
                "steps_per_sec": None,
                "tokens_per_sec": None,
                "gpu_mem_peak_mb": None,
                "cpu_util_avg": _cpu_util_percent(),
                "compile_graph_breaks": None,
            },
        }
    if stage == "tokenizer":
        return result

    dataset_name = lm_data_cfg.get("dataset_name", "roneneldan/TinyStories")
    text_field = str(lm_data_cfg.get("text_field", "text"))
    allow_synthetic_fallback = bool(lm_data_cfg.get("allow_synthetic_fallback", True))
    streaming = bool(lm_data_cfg.get("streaming", True))
    train_texts = _load_text_dataset_samples(
        dataset_name=dataset_name,
        split=lm_data_cfg.get("train_split", "train"),
        max_samples=int(lm_data_cfg.get("train_samples", 2000)),
        seed=int(config.get("experiment", {}).get("seed", 42)),
        text_field=text_field,
        allow_synthetic_fallback=allow_synthetic_fallback,
        streaming=streaming,
        cache_mode=cache_mode,
        cache_root=cache_root,
    )
    val_texts = _load_text_dataset_samples(
        dataset_name=dataset_name,
        split=lm_data_cfg.get("val_split", "validation"),
        max_samples=int(lm_data_cfg.get("val_samples", 256)),
        seed=int(config.get("experiment", {}).get("seed", 42)) + 7,
        text_field=text_field,
        allow_synthetic_fallback=allow_synthetic_fallback,
        streaming=streaming,
        cache_mode=cache_mode,
        cache_root=cache_root,
    )
    seq_len = int(lm_data_cfg.get("seq_len", 64))
    batch_size = int(config.get("training", {}).get("batch_size", 4))
    num_workers = int(config.get("data", {}).get("num_workers", 0))
    tokenizer_batch_size = int(data_cache_cfg.get("tokenizer_batch_size", 64))

    dabe_val_loss: float | None = None
    bpe_val_loss: float | None = None
    dabe_stable = True
    bpe_stable = True

    if stage in {"all", "dabe_lm"}:
        v4_cfg = optimization_cfg.get("v4_distill", {})
        router_cfg = dict(config.get("router", {}))
        if "balance_bias_update" in v4_cfg and "balance_bias_update" not in router_cfg:
            router_cfg["balance_bias_update"] = bool(v4_cfg.get("balance_bias_update", False))
        router = EntropyRouter(router_cfg)
        encoder = LFQEncoder(config.get("encoder", {}))
        encoder.load_artifact(artifact_path)

        dabe_data = DABEFeasibilityDataModule(
            texts_train=train_texts,
            texts_val=val_texts,
            router=router,
            encoder=encoder,
            seq_len=seq_len,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=int(config.get("experiment", {}).get("seed", 42)),
            cache_dir=cache_root / "windows",
            cache_mode=cache_mode,
            preencode_to_disk=preencode_to_disk,
            device_profile=device_profile,
            accelerator=accelerator,
        )
        dabe_backbone = dict(config.get("backbone", {}))
        dabe_attention_cfg = dict(dabe_backbone.get("attention", {}))
        dabe_attention_cfg["backend"] = str(
            optimization_cfg.get("attention_backend", dabe_attention_cfg.get("backend", "auto"))
        )
        if "compressed_kv_proxy" in v4_cfg:
            dabe_attention_cfg["compressed_kv_proxy"] = bool(v4_cfg.get("compressed_kv_proxy", False))
        if dabe_attention_cfg:
            dabe_backbone["attention"] = dabe_attention_cfg
        if "mtp_heads" in v4_cfg:
            dabe_backbone["mtp_heads"] = int(v4_cfg.get("mtp_heads", 0))
        dabe_cfg = {**config, "backbone": dabe_backbone}
        dabe_model = DABECausalLMModule(config=dabe_cfg, encoder=encoder)
        dabe_metrics = StageRuntimeMetrics()
        dabe_data.setup("fit")
        dabe_metrics.prep_seconds = dabe_data.prep_seconds
        compile_enabled = bool(compile_cfg.get("enabled", False))
        if len(dabe_data.train_dataset) > 0:
            first_batch = next(iter(dabe_data.train_dataloader()))
            dabe_metrics.compile_graph_breaks = _compile_graph_break_count(
                dabe_model.model,
                stage_name="dabe_lm",
                batch=first_batch,
                encoder=encoder,
                max_total_bits=int(dabe_model.model.max_bit_width),
            )
        dabe_model.model, dabe_model_was_compiled = _maybe_compile_model(
            dabe_model.model,
            enabled=compile_enabled,
            mode=str(compile_cfg.get("mode", "reduce-overhead")),
        )
        dabe_trainer = _build_trainer(
            config=config,
            output_dir=output_dir,
            wandb_logger=wandb_logger,
            accelerator=accelerator,
            devices=devices,
            stage_name="dabe_lm",
            monitor="dabe/val_loss",
            runtime_metrics=dabe_metrics,
        )
        try:
            dabe_trainer.fit(dabe_model, datamodule=dabe_data)
        except RuntimeError as exc:
            if dabe_model_was_compiled:
                log.warning("Retrying dabe_lm in eager mode after failure with compile: %s", exc)
                dabe_model.model = dabe_model.model._orig_mod if hasattr(dabe_model.model, "_orig_mod") else dabe_model.model  # type: ignore[attr-defined]
                dabe_trainer = _build_trainer(
                    config=config,
                    output_dir=output_dir,
                    wandb_logger=wandb_logger,
                    accelerator=accelerator,
                    devices=devices,
                    stage_name="dabe_lm",
                    monitor="dabe/val_loss",
                    runtime_metrics=dabe_metrics,
                )
                dabe_trainer.fit(dabe_model, datamodule=dabe_data)
            else:
                raise
        metric = dabe_trainer.callback_metrics.get("dabe/val_loss")
        dabe_val_loss = float(metric.item()) if metric is not None else None
        dabe_stable = dabe_model.nan_batches == 0
        result["dabe_lm"] = {
            "val_loss": dabe_val_loss,
            "val_perplexity": math.exp(dabe_val_loss) if dabe_val_loss is not None else None,
            "stable": dabe_stable,
            "train_samples": len(dabe_data.train_dataset),
            "val_samples": len(dabe_data.val_dataset),
            "runtime": {
                "prep_seconds": dabe_metrics.prep_seconds,
                "first_batch_seconds": dabe_metrics.first_batch_seconds,
                "steps_per_sec": dabe_metrics.steps_per_sec,
                "tokens_per_sec": dabe_metrics.tokens_per_sec,
                "dataloader_wait_seconds": dabe_metrics.dataloader_wait_seconds,
                "gpu_mem_peak_mb": dabe_metrics.gpu_mem_peak_mb,
                "gpu_util_avg": dabe_metrics.gpu_util_avg,
                "cpu_util_avg": dabe_metrics.cpu_util_avg,
                "compile_graph_breaks": dabe_metrics.compile_graph_breaks,
                "compile_warmup_seconds": dabe_metrics.compile_warmup_seconds,
            },
        }

    if stage in {"all", "bpe_baseline"}:
        bpe_model_name = feasibility_cfg.get("bpe_model_name", "gpt2")
        bpe_tokenizer = AutoTokenizer.from_pretrained(
            bpe_model_name,
            use_fast=bool(optimization_cfg.get("force_fast_tokenizer", True)),
        )
        if bool(optimization_cfg.get("force_fast_tokenizer", True)) and not bool(
            getattr(bpe_tokenizer, "is_fast", False)
        ):
            raise RuntimeError(
                f"Expected fast tokenizer for {bpe_model_name}, but slow tokenizer was loaded."
            )
        if bpe_tokenizer.pad_token is None and bpe_tokenizer.eos_token is not None:
            bpe_tokenizer.pad_token = bpe_tokenizer.eos_token

        baseline_backbone = {
            **config.get("backbone", {}),
            **feasibility_cfg.get("baseline_backbone", {}),
        }
        v4_cfg = optimization_cfg.get("v4_distill", {})
        baseline_attention_cfg = dict(baseline_backbone.get("attention", {}))
        baseline_attention_cfg["backend"] = str(
            optimization_cfg.get("attention_backend", baseline_attention_cfg.get("backend", "auto"))
        )
        if "compressed_kv_proxy" in v4_cfg:
            baseline_attention_cfg["compressed_kv_proxy"] = bool(v4_cfg.get("compressed_kv_proxy", False))
        if baseline_attention_cfg:
            baseline_backbone["attention"] = baseline_attention_cfg
        if "mtp_heads" in v4_cfg:
            baseline_backbone["mtp_heads"] = int(v4_cfg.get("mtp_heads", 0))
        baseline_cfg = {
            **config,
            "backbone": {
                **baseline_backbone,
                "vocab_size": int(bpe_tokenizer.vocab_size),
            },
        }
        bpe_data = BPEFeasibilityDataModule(
            texts_train=train_texts,
            texts_val=val_texts,
            tokenizer=bpe_tokenizer,
            seq_len=seq_len,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=int(config.get("experiment", {}).get("seed", 42)),
            cache_dir=cache_root / "windows",
            cache_mode=cache_mode,
            preencode_to_disk=preencode_to_disk,
            tokenizer_batch_size=tokenizer_batch_size,
            device_profile=device_profile,
            accelerator=accelerator,
        )
        bpe_model = BPECausalLMModule(config=baseline_cfg)
        bpe_metrics = StageRuntimeMetrics()
        bpe_data.setup("fit")
        bpe_metrics.prep_seconds = bpe_data.prep_seconds
        compile_enabled = bool(compile_cfg.get("enabled", False))
        if len(bpe_data.train_dataset) > 0:
            first_batch = next(iter(bpe_data.train_dataloader()))
            bpe_metrics.compile_graph_breaks = _compile_graph_break_count(
                bpe_model.model,
                stage_name="bpe_baseline",
                batch=first_batch,
            )
        bpe_model.model, bpe_model_was_compiled = _maybe_compile_model(
            bpe_model.model,
            enabled=compile_enabled,
            mode=str(compile_cfg.get("mode", "reduce-overhead")),
        )
        bpe_trainer = _build_trainer(
            config=config,
            output_dir=output_dir,
            wandb_logger=wandb_logger,
            accelerator=accelerator,
            devices=devices,
            stage_name="bpe_baseline",
            monitor="bpe/val_loss",
            runtime_metrics=bpe_metrics,
        )
        try:
            bpe_trainer.fit(bpe_model, datamodule=bpe_data)
        except RuntimeError as exc:
            if bpe_model_was_compiled:
                log.warning(
                    "Retrying bpe_baseline in eager mode after failure with compile: %s",
                    exc,
                )
                bpe_model.model = bpe_model.model._orig_mod if hasattr(bpe_model.model, "_orig_mod") else bpe_model.model  # type: ignore[attr-defined]
                bpe_trainer = _build_trainer(
                    config=config,
                    output_dir=output_dir,
                    wandb_logger=wandb_logger,
                    accelerator=accelerator,
                    devices=devices,
                    stage_name="bpe_baseline",
                    monitor="bpe/val_loss",
                    runtime_metrics=bpe_metrics,
                )
                bpe_trainer.fit(bpe_model, datamodule=bpe_data)
            else:
                raise
        metric = bpe_trainer.callback_metrics.get("bpe/val_loss")
        bpe_val_loss = float(metric.item()) if metric is not None else None
        bpe_stable = bpe_model.nan_batches == 0
        result["bpe_baseline"] = {
            "val_loss": bpe_val_loss,
            "val_perplexity": math.exp(bpe_val_loss) if bpe_val_loss is not None else None,
            "stable": bpe_stable,
            "train_samples": len(bpe_data.train_dataset),
            "val_samples": len(bpe_data.val_dataset),
            "runtime": {
                "prep_seconds": bpe_metrics.prep_seconds,
                "first_batch_seconds": bpe_metrics.first_batch_seconds,
                "steps_per_sec": bpe_metrics.steps_per_sec,
                "tokens_per_sec": bpe_metrics.tokens_per_sec,
                "dataloader_wait_seconds": bpe_metrics.dataloader_wait_seconds,
                "gpu_mem_peak_mb": bpe_metrics.gpu_mem_peak_mb,
                "gpu_util_avg": bpe_metrics.gpu_util_avg,
                "cpu_util_avg": bpe_metrics.cpu_util_avg,
                "compile_graph_breaks": bpe_metrics.compile_graph_breaks,
                "compile_warmup_seconds": bpe_metrics.compile_warmup_seconds,
            },
        }

    gate_cfg = feasibility_cfg.get("decision_gate", {})
    loss_tolerance = float(gate_cfg.get("loss_tolerance_ratio", 0.15))
    compression_min = float(gate_cfg.get("min_compression_ratio", 1.0))
    compression_value = (
        tokenizer_result.compression_ratio_vs_bpe
        if tokenizer_result is not None
        else result.get("tokenizer", {}).get("compression_ratio_vs_bpe", 0.0)
    )
    competitive = (
        dabe_val_loss is not None
        and bpe_val_loss is not None
        and dabe_stable
        and bpe_stable
        and compression_value >= compression_min
        and dabe_val_loss <= bpe_val_loss * (1.0 + loss_tolerance)
    )

    result["decision_gate"] = {
        "dabe_val_loss": dabe_val_loss,
        "bpe_val_loss": bpe_val_loss,
        "compression_ratio_vs_bpe": compression_value,
        "dabe_stable": dabe_stable,
        "bpe_stable": bpe_stable,
        "competitive": competitive,
    }
    return result


def _validate_adapter_shapes(backbone_cfg: dict) -> dict[str, tuple[int, ...]]:
    model = AdapterTokenTransformerLM(backbone_cfg)
    seq_len = 16
    batch_size = 2
    input_ids = torch.randint(0, int(backbone_cfg["vocab_size"]), (batch_size, seq_len))
    mask = build_causal_mask(seq_len, input_ids.device)
    logits, hidden, stats = model(input_ids, mask=mask)
    if logits.shape != (batch_size, seq_len, int(backbone_cfg["vocab_size"])):
        raise AssertionError(f"Unexpected adapter logits shape: {tuple(logits.shape)}")
    if hidden.shape != (batch_size, seq_len, int(backbone_cfg["hidden_dim"])):
        raise AssertionError(f"Unexpected adapter hidden shape: {tuple(hidden.shape)}")
    if stats["bit_widths"].shape != (batch_size, seq_len):
        raise AssertionError(f"Unexpected bit-width shape: {tuple(stats['bit_widths'].shape)}")
    return {
        "logits": tuple(logits.shape),
        "hidden": tuple(hidden.shape),
        "bit_widths": tuple(stats["bit_widths"].shape),
    }


def run_adapter_feasibility_experiment(
    config: dict,
    output_dir: Path,
    wandb_logger: Any | None,
    accelerator: str,
    devices: Any,
) -> dict:
    feasibility_cfg = config.get("feasibility", {})
    data_cfg = feasibility_cfg.get("adapter_data", feasibility_cfg.get("lm_data", {}))
    gate_cfg = feasibility_cfg.get("decision_gate", {})
    bpe_model_name = feasibility_cfg.get("bpe_model_name", "gpt2")
    bpe_tokenizer = AutoTokenizer.from_pretrained(bpe_model_name)
    if bpe_tokenizer.pad_token is None and bpe_tokenizer.eos_token is not None:
        bpe_tokenizer.pad_token = bpe_tokenizer.eos_token

    text_field = str(data_cfg.get("text_field", "text"))
    allow_synthetic_fallback = bool(data_cfg.get("allow_synthetic_fallback", True))
    train_texts = _load_text_dataset_samples(
        dataset_name=data_cfg.get("dataset_name", "roneneldan/TinyStories"),
        split=data_cfg.get("train_split", "train"),
        max_samples=int(data_cfg.get("train_samples", 2000)),
        seed=int(config.get("experiment", {}).get("seed", 42)),
        text_field=text_field,
        allow_synthetic_fallback=allow_synthetic_fallback,
    )
    val_texts = _load_text_dataset_samples(
        dataset_name=data_cfg.get("dataset_name", "roneneldan/TinyStories"),
        split=data_cfg.get("val_split", "validation"),
        max_samples=int(data_cfg.get("val_samples", 256)),
        seed=int(config.get("experiment", {}).get("seed", 42)) + 11,
        text_field=text_field,
        allow_synthetic_fallback=allow_synthetic_fallback,
    )
    seq_len = int(data_cfg.get("seq_len", 64))
    batch_size = int(config.get("training", {}).get("batch_size", 4))
    num_workers = int(config.get("data", {}).get("num_workers", 0))

    adapter_backbone = {
        **config.get("backbone", {}),
        **feasibility_cfg.get("adapter_backbone", {}),
        "vocab_size": int(bpe_tokenizer.vocab_size),
        "adapter": feasibility_cfg.get("adapter", {}),
    }
    baseline_backbone = {
        **config.get("backbone", {}),
        **feasibility_cfg.get("baseline_backbone", {}),
        "vocab_size": int(bpe_tokenizer.vocab_size),
    }

    shape_report = _validate_adapter_shapes(adapter_backbone)
    if wandb_logger is not None:
        wandb_logger.log_metrics({
            "adapter/shapes_logits_batch": shape_report["logits"][0],
            "adapter/shapes_hidden_dim": shape_report["hidden"][-1],
        })

    data_module = BPEFeasibilityDataModule(
        texts_train=train_texts,
        texts_val=val_texts,
        tokenizer=bpe_tokenizer,
        seq_len=seq_len,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=int(config.get("experiment", {}).get("seed", 42)),
    )

    adapter_cfg = {**config, "backbone": adapter_backbone}
    adapter_model = AdapterCausalLMModule(config=adapter_cfg)
    adapter_trainer = _build_trainer(
        config=config,
        output_dir=output_dir,
        wandb_logger=wandb_logger,
        accelerator=accelerator,
        devices=devices,
        stage_name="adapter_lm",
        monitor="adapter/val_loss",
    )
    adapter_trainer.fit(adapter_model, datamodule=data_module)
    adapter_loss_metric = adapter_trainer.callback_metrics.get("adapter/val_loss")
    adapter_width_metric = adapter_trainer.callback_metrics.get("adapter/val_width_std")
    if adapter_loss_metric is not None:
        adapter_val_loss = float(adapter_loss_metric.item())
    else:
        adapter_val_loss = None
    if adapter_width_metric is not None:
        adapter_width_std = float(adapter_width_metric.item())
    else:
        adapter_width_std = 0.0
    adapter_stable = adapter_model.nan_batches == 0

    baseline_cfg = {**config, "backbone": baseline_backbone}
    baseline_model = BPECausalLMModule(config=baseline_cfg)
    baseline_trainer = _build_trainer(
        config=config,
        output_dir=output_dir,
        wandb_logger=wandb_logger,
        accelerator=accelerator,
        devices=devices,
        stage_name="adapter_baseline",
        monitor="bpe/val_loss",
    )
    baseline_trainer.fit(baseline_model, datamodule=data_module)
    baseline_metric = baseline_trainer.callback_metrics.get("bpe/val_loss")
    baseline_val_loss = float(baseline_metric.item()) if baseline_metric is not None else None
    baseline_stable = baseline_model.nan_batches == 0

    loss_tolerance = float(gate_cfg.get("loss_tolerance_ratio", 0.1))
    min_width_std = float(gate_cfg.get("min_width_std", 0.2))
    competitive = (
        adapter_val_loss is not None
        and baseline_val_loss is not None
        and adapter_stable
        and baseline_stable
        and adapter_width_std >= min_width_std
        and adapter_val_loss <= baseline_val_loss * (1.0 + loss_tolerance)
    )

    return {
        "track": "adapter_feasibility",
        "shape_validation": shape_report,
        "adapter_lm": {
            "val_loss": adapter_val_loss,
            "val_perplexity": math.exp(adapter_val_loss) if adapter_val_loss is not None else None,
            "stable": adapter_stable,
            "val_width_std": adapter_width_std,
            "train_samples": len(data_module.train_dataset),
            "val_samples": len(data_module.val_dataset),
        },
        "baseline_lm": {
            "val_loss": baseline_val_loss,
            "val_perplexity": (
                math.exp(baseline_val_loss)
                if baseline_val_loss is not None
                else None
            ),
            "stable": baseline_stable,
            "train_samples": len(data_module.train_dataset),
            "val_samples": len(data_module.val_dataset),
        },
        "decision_gate": {
            "adapter_val_loss": adapter_val_loss,
            "baseline_val_loss": baseline_val_loss,
            "adapter_width_std": adapter_width_std,
            "min_width_std": min_width_std,
            "adapter_stable": adapter_stable,
            "baseline_stable": baseline_stable,
            "competitive": competitive,
        },
    }
