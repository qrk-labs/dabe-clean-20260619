from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer

from ..backbone.transformer import TransformerBlock
from .fp16_chunk_feasibility import (
    EtaMetricsCallback,
    FP16ChunkCompressor,
    RuntimeMetrics,
    RuntimeMetricsCallback,
    ValidationTrendCallback,
    _load_text_dataset_samples,
    _resolve_precision,
    build_causal_mask,
)


def _validate_hierarchy_sizes(coarse: int, mid: int, fine: int) -> None:
    if coarse <= 0 or mid <= 0 or fine <= 0:
        raise ValueError("Hierarchy chunk sizes must be positive.")
    if not (coarse > mid > fine):
        raise ValueError("Expected hierarchy sizes coarse > mid > fine.")
    if coarse % mid != 0 or coarse % fine != 0:
        raise ValueError("coarse chunk size must be divisible by mid and fine sizes.")


def _positions_from_stride(chunk_size_tokens: int, stride: int) -> list[int]:
    if stride <= 0:
        raise ValueError("stride must be positive.")
    return list(range(0, int(chunk_size_tokens), int(stride)))


def _aggregate_scale_to_coarse(
    scale_stream: np.ndarray,
    coarse_count: int,
    factor: int,
) -> np.ndarray:
    if scale_stream.shape[0] < coarse_count * factor:
        return np.empty((0,), dtype=np.float32)
    trimmed = scale_stream[: coarse_count * factor]
    reshaped = trimmed.reshape(coarse_count, factor)
    return reshaped.mean(axis=1).astype(np.float32, copy=False)


def _build_hierarchical_streams(
    *,
    texts: Sequence[str],
    tokenizer: Any,
    coarse_chunk_size: int,
    mid_chunk_size: int,
    fine_chunk_size: int,
    tokenizer_batch_size: int,
    dtype: str,
    include_chunk_tokens: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    _validate_hierarchy_sizes(coarse_chunk_size, mid_chunk_size, fine_chunk_size)
    coarse_factor_mid = coarse_chunk_size // mid_chunk_size
    coarse_factor_fine = coarse_chunk_size // fine_chunk_size

    coarse_comp = FP16ChunkCompressor(
        chunk_size_tokens=coarse_chunk_size,
        window_overlap_tokens=0,
        tokenizer=tokenizer,
        force_fast_tokenizer=False,
        dtype=dtype,
    )
    mid_comp = FP16ChunkCompressor(
        chunk_size_tokens=mid_chunk_size,
        window_overlap_tokens=0,
        tokenizer=tokenizer,
        force_fast_tokenizer=False,
        dtype=dtype,
    )
    fine_comp = FP16ChunkCompressor(
        chunk_size_tokens=fine_chunk_size,
        window_overlap_tokens=0,
        tokenizer=tokenizer,
        force_fast_tokenizer=False,
        dtype=dtype,
    )

    coarse_parts: list[np.ndarray] = []
    mid_parts: list[np.ndarray] = []
    fine_parts: list[np.ndarray] = []
    chunk_parts: list[np.ndarray] = []

    batch_size = max(1, int(tokenizer_batch_size))
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        encoded = tokenizer(batch, add_special_tokens=False)
        ids_per_text = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        for token_ids in ids_per_text:
            ids = np.asarray(token_ids, dtype=np.int64)
            coarse_count = ids.shape[0] // coarse_chunk_size
            if coarse_count <= 1:
                continue
            usable_ids = ids[: coarse_count * coarse_chunk_size]
            coarse_scalars, coarse_chunks = coarse_comp.compress_token_ids_with_chunks(usable_ids)
            if coarse_scalars.shape[0] != coarse_count:
                continue
            mid_scalars = mid_comp.compress_token_ids(usable_ids).astype(np.float32, copy=False)
            fine_scalars = fine_comp.compress_token_ids(usable_ids).astype(np.float32, copy=False)
            mid_aligned = _aggregate_scale_to_coarse(
                mid_scalars,
                coarse_count=coarse_count,
                factor=coarse_factor_mid,
            )
            fine_aligned = _aggregate_scale_to_coarse(
                fine_scalars,
                coarse_count=coarse_count,
                factor=coarse_factor_fine,
            )
            if mid_aligned.shape[0] != coarse_count or fine_aligned.shape[0] != coarse_count:
                continue

            coarse_parts.append(coarse_scalars.astype(np.float32, copy=False))
            mid_parts.append(mid_aligned)
            fine_parts.append(fine_aligned)
            if include_chunk_tokens:
                chunk_parts.append(coarse_chunks.astype(np.int32, copy=False))

    if not coarse_parts:
        empty_float = np.empty((0,), dtype=np.float32)
        empty_tokens = np.empty((0, coarse_chunk_size), dtype=np.int32) if include_chunk_tokens else None
        return empty_float, empty_float, empty_float, empty_tokens

    coarse_stream = np.concatenate(coarse_parts, axis=0).astype(np.float32, copy=False)
    mid_stream = np.concatenate(mid_parts, axis=0).astype(np.float32, copy=False)
    fine_stream = np.concatenate(fine_parts, axis=0).astype(np.float32, copy=False)
    chunk_stream = (
        np.concatenate(chunk_parts, axis=0).astype(np.int32, copy=False)
        if include_chunk_tokens
        else None
    )
    return coarse_stream, mid_stream, fine_stream, chunk_stream


class HierarchicalFP16SequenceDataset(TorchDataset):
    def __init__(
        self,
        *,
        coarse_stream: np.ndarray,
        mid_stream: np.ndarray,
        fine_stream: np.ndarray,
        compressed_seq_len: int,
        window_stride: int = 1,
        chunk_tokens_stream: np.ndarray | None = None,
    ):
        if compressed_seq_len < 1:
            raise ValueError("compressed_seq_len must be >= 1")
        if not (coarse_stream.shape[0] == mid_stream.shape[0] == fine_stream.shape[0]):
            raise ValueError("Hierarchical streams must be aligned to equal lengths.")

        seq_len = int(compressed_seq_len)
        stride = max(1, int(window_stride))

        coarse = np.asarray(coarse_stream, dtype=np.float32).reshape(-1)
        mid = np.asarray(mid_stream, dtype=np.float32).reshape(-1)
        fine = np.asarray(fine_stream, dtype=np.float32).reshape(-1)
        tokens = (
            np.asarray(chunk_tokens_stream, dtype=np.int32)
            if chunk_tokens_stream is not None
            else None
        )
        if tokens is not None and tokens.shape[0] != coarse.shape[0]:
            raise ValueError("chunk_tokens_stream must align with stream length.")
        self._chunk_tokens_stream = tokens
        self._seq_len = seq_len

        if coarse.shape[0] <= seq_len:
            self._inputs_coarse = np.empty((0, seq_len), dtype=np.float32)
            self._inputs_mid = np.empty((0, seq_len), dtype=np.float32)
            self._inputs_fine = np.empty((0, seq_len), dtype=np.float32)
            self._targets_coarse = np.empty((0, seq_len), dtype=np.float32)
            self._targets_mid = np.empty((0, seq_len), dtype=np.float32)
            self._targets_fine = np.empty((0, seq_len), dtype=np.float32)
            self._starts = np.empty((0,), dtype=np.int64)
            return

        starts = np.arange(0, coarse.shape[0] - seq_len, stride, dtype=np.int64)
        if starts.size == 0:
            self._inputs_coarse = np.empty((0, seq_len), dtype=np.float32)
            self._inputs_mid = np.empty((0, seq_len), dtype=np.float32)
            self._inputs_fine = np.empty((0, seq_len), dtype=np.float32)
            self._targets_coarse = np.empty((0, seq_len), dtype=np.float32)
            self._targets_mid = np.empty((0, seq_len), dtype=np.float32)
            self._targets_fine = np.empty((0, seq_len), dtype=np.float32)
            self._starts = np.empty((0,), dtype=np.int64)
            return

        indices = starts[:, None] + np.arange(seq_len + 1, dtype=np.int64)[None, :]
        coarse_windows = coarse[indices]
        mid_windows = mid[indices]
        fine_windows = fine[indices]

        self._inputs_coarse = coarse_windows[:, :-1].astype(np.float32, copy=False)
        self._inputs_mid = mid_windows[:, :-1].astype(np.float32, copy=False)
        self._inputs_fine = fine_windows[:, :-1].astype(np.float32, copy=False)
        self._targets_coarse = coarse_windows[:, 1:].astype(np.float32, copy=False)
        self._targets_mid = mid_windows[:, 1:].astype(np.float32, copy=False)
        self._targets_fine = fine_windows[:, 1:].astype(np.float32, copy=False)
        self._starts = starts

    def __len__(self) -> int:
        return int(self._inputs_coarse.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        input_coarse = torch.from_numpy(np.asarray(self._inputs_coarse[idx], dtype=np.float32))
        sample = {
            "input_scalars": input_coarse,
            "input_scalars_coarse": input_coarse,
            "input_scalars_mid": torch.from_numpy(np.asarray(self._inputs_mid[idx], dtype=np.float32)),
            "input_scalars_fine": torch.from_numpy(np.asarray(self._inputs_fine[idx], dtype=np.float32)),
            "target_scalars": torch.from_numpy(np.asarray(self._targets_coarse[idx], dtype=np.float32)),
            "target_scalars_coarse": torch.from_numpy(
                np.asarray(self._targets_coarse[idx], dtype=np.float32),
            ),
            "target_scalars_mid": torch.from_numpy(np.asarray(self._targets_mid[idx], dtype=np.float32)),
            "target_scalars_fine": torch.from_numpy(np.asarray(self._targets_fine[idx], dtype=np.float32)),
        }
        if self._chunk_tokens_stream is not None:
            start = int(self._starts[idx])
            target_chunk_ids = self._chunk_tokens_stream[start + 1 : start + 1 + self._seq_len]
            sample["target_chunk_ids"] = torch.from_numpy(
                np.asarray(target_chunk_ids, dtype=np.int64),
            )
        return sample


class HierarchicalFP16DataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        train_texts: Sequence[str],
        val_texts: Sequence[str],
        tokenizer_name: str,
        force_fast_tokenizer: bool,
        dtype: str,
        coarse_chunk_size: int,
        mid_chunk_size: int,
        fine_chunk_size: int,
        compressed_seq_len: int,
        window_stride: int,
        tokenizer_batch_size: int,
        batch_size: int,
        num_workers: int,
        seed: int,
        accelerator: str,
        include_chunk_tokens: bool,
    ):
        super().__init__()
        _validate_hierarchy_sizes(coarse_chunk_size, mid_chunk_size, fine_chunk_size)
        self.train_texts = list(train_texts)
        self.val_texts = list(val_texts)
        self.tokenizer_name = str(tokenizer_name)
        self.force_fast_tokenizer = bool(force_fast_tokenizer)
        self.dtype = str(dtype)
        self.coarse_chunk_size = int(coarse_chunk_size)
        self.mid_chunk_size = int(mid_chunk_size)
        self.fine_chunk_size = int(fine_chunk_size)
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
        self.vocab_size = 50257
        self.train_coarse_stream: np.ndarray | None = None
        self.train_mid_stream: np.ndarray | None = None
        self.train_fine_stream: np.ndarray | None = None
        self.train_chunk_tokens: np.ndarray | None = None
        self.val_coarse_stream: np.ndarray | None = None
        self.val_mid_stream: np.ndarray | None = None
        self.val_fine_stream: np.ndarray | None = None
        self.val_chunk_tokens: np.ndarray | None = None

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self._is_setup:
            return

        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            use_fast=self.force_fast_tokenizer,
        )
        if self.force_fast_tokenizer and not bool(getattr(tokenizer, "is_fast", False)):
            raise RuntimeError(
                f"Expected fast tokenizer for {self.tokenizer_name}, but slow tokenizer was loaded.",
            )
        self.vocab_size = max(1, int(getattr(tokenizer, "vocab_size", 50257)))

        train_coarse, train_mid, train_fine, train_chunks = _build_hierarchical_streams(
            texts=self.train_texts,
            tokenizer=tokenizer,
            coarse_chunk_size=self.coarse_chunk_size,
            mid_chunk_size=self.mid_chunk_size,
            fine_chunk_size=self.fine_chunk_size,
            tokenizer_batch_size=self.tokenizer_batch_size,
            dtype=self.dtype,
            include_chunk_tokens=self.include_chunk_tokens,
        )
        val_coarse, val_mid, val_fine, val_chunks = _build_hierarchical_streams(
            texts=self.val_texts,
            tokenizer=tokenizer,
            coarse_chunk_size=self.coarse_chunk_size,
            mid_chunk_size=self.mid_chunk_size,
            fine_chunk_size=self.fine_chunk_size,
            tokenizer_batch_size=self.tokenizer_batch_size,
            dtype=self.dtype,
            include_chunk_tokens=self.include_chunk_tokens,
        )
        self.train_chunk_count = int(train_coarse.shape[0])
        self.val_chunk_count = int(val_coarse.shape[0])
        self.train_coarse_stream = train_coarse
        self.train_mid_stream = train_mid
        self.train_fine_stream = train_fine
        self.train_chunk_tokens = train_chunks
        self.val_coarse_stream = val_coarse
        self.val_mid_stream = val_mid
        self.val_fine_stream = val_fine
        self.val_chunk_tokens = val_chunks

        self.train_dataset = HierarchicalFP16SequenceDataset(
            coarse_stream=train_coarse,
            mid_stream=train_mid,
            fine_stream=train_fine,
            compressed_seq_len=self.compressed_seq_len,
            window_stride=self.window_stride,
            chunk_tokens_stream=train_chunks,
        )
        self.val_dataset = HierarchicalFP16SequenceDataset(
            coarse_stream=val_coarse,
            mid_stream=val_mid,
            fine_stream=val_fine,
            compressed_seq_len=self.compressed_seq_len,
            window_stride=self.window_stride,
            chunk_tokens_stream=val_chunks,
        )
        if len(self.train_dataset) == 0:
            raise RuntimeError(
                "HierarchicalFP16DataModule produced zero train samples. "
                "Increase train_samples or reduce chunk sizes/compressed_seq_len.",
            )
        if len(self.val_dataset) == 0:
            raise RuntimeError(
                "HierarchicalFP16DataModule produced zero val samples. "
                "Increase val_samples or reduce chunk sizes/compressed_seq_len.",
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


class HierarchicalScalarTransformerLM(nn.Module):
    def __init__(self, model_cfg: dict[str, Any], hierarchy_cfg: dict[str, Any]):
        super().__init__()
        self.hidden_dim = int(model_cfg.get("hidden_dim", 256))
        self.num_layers = int(model_cfg.get("num_layers", 4))
        self.num_heads = int(model_cfg.get("num_heads", 4))
        self.ff_dim = int(model_cfg.get("ff_dim", 1024))
        self.dropout = float(model_cfg.get("dropout", 0.1))
        self.attention_cfg = dict(model_cfg.get("attention", {}))
        self.router_temperature = float(hierarchy_cfg.get("router_temperature", 1.0))
        self.hard_routing = bool(hierarchy_cfg.get("hard_routing", False))
        self.base_residual_weight = float(hierarchy_cfg.get("base_residual_weight", 0.15))

        self.coarse_proj = nn.Linear(1, self.hidden_dim)
        self.mid_proj = nn.Linear(1, self.hidden_dim)
        self.fine_proj = nn.Linear(1, self.hidden_dim)
        self.router = nn.Linear(self.hidden_dim * 3, 3)

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
        self.output_proj = nn.Linear(self.hidden_dim, 1)

    def forward(
        self,
        input_scalars_coarse: torch.Tensor,
        input_scalars_mid: torch.Tensor,
        input_scalars_fine: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_coarse = self.coarse_proj(input_scalars_coarse.unsqueeze(-1))
        h_mid = self.mid_proj(input_scalars_mid.unsqueeze(-1))
        h_fine = self.fine_proj(input_scalars_fine.unsqueeze(-1))

        router_input = torch.cat([h_coarse, h_mid, h_fine], dim=-1)
        logits = self.router(router_input) / max(1e-5, self.router_temperature)
        route_probs = F.softmax(logits, dim=-1)
        if self.hard_routing:
            hard_idx = route_probs.argmax(dim=-1)
            hard_probs = F.one_hot(hard_idx, num_classes=3).float()
            route_probs = hard_probs + (route_probs - route_probs.detach())

        fused = (
            route_probs[..., 0:1] * h_coarse
            + route_probs[..., 1:2] * h_mid
            + route_probs[..., 2:3] * h_fine
        )
        if self.base_residual_weight > 0.0:
            fused = (
                (1.0 - self.base_residual_weight) * fused
                + self.base_residual_weight * h_coarse
            )
        hidden = fused
        for block in self.blocks:
            hidden = block(hidden, mask=mask)
        hidden = self.norm(hidden)
        preds = self.output_proj(hidden).squeeze(-1)
        return preds, hidden, route_probs


class ContextConditionedDiffusionDenoiser(nn.Module):
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
        return self.net(torch.cat([z_t, context, t_embed], dim=-1))


class HierarchicalFP16ChunkLMModule(L.LightningModule):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.model_cfg = dict(config.get("model", {}))
        self.training_cfg = dict(config.get("training", {}))
        self.diffusion_cfg = dict(config.get("diffusion", {}))
        self.hierarchy_cfg = dict(config.get("hierarchical", {}))
        self.decode_mirror_cfg = dict(config.get("decode_mirror", {}))
        self.output_head_cfg = dict(config.get("output_head", {}))
        self.model = HierarchicalScalarTransformerLM(self.model_cfg, self.hierarchy_cfg)

        self.route_balance_weight = float(self.hierarchy_cfg.get("route_balance_weight", 0.02))
        self.route_entropy_weight = float(self.hierarchy_cfg.get("route_entropy_weight", 0.0))
        route_target = self.hierarchy_cfg.get("route_target", [0.55, 0.30, 0.15])
        if not isinstance(route_target, Sequence) or len(route_target) != 3:
            route_target = [0.55, 0.30, 0.15]
        target = np.asarray(route_target, dtype=np.float32)
        target = target / max(1e-8, float(target.sum()))
        self.register_buffer("route_target", torch.from_numpy(target))

        self.diffusion_enabled = bool(self.diffusion_cfg.get("enabled", False))
        self.diffusion_loss_weight = float(self.diffusion_cfg.get("loss_weight", 0.25))
        self.diffusion_recon_weight = float(self.diffusion_cfg.get("reconstruction_weight", 0.2))
        self.compute_train_gist_metrics = bool(
            self.diffusion_cfg.get("compute_train_gist_metrics", False),
        )
        allowed_output_families = {"anchor_sparse", "anchor_refine", "dual_task"}
        self.output_head_explicit = bool(self.output_head_cfg)
        configured_family = str(self.output_head_cfg.get("family", "")).strip().lower()
        if self.output_head_explicit and configured_family and configured_family not in allowed_output_families:
            raise ValueError(
                "fp16_chunk.output_head.family must be one of "
                f"{sorted(allowed_output_families)}; got '{configured_family}'.",
            )
        if self.output_head_explicit and not configured_family:
            configured_family = "anchor_sparse"
        legacy_decode_enabled = bool(self.decode_mirror_cfg.get("enabled", False))
        self.decode_mirror_enabled = bool(
            self.output_head_cfg.get("enabled", legacy_decode_enabled),
        ) if self.output_head_explicit else legacy_decode_enabled
        self.output_head_family = (
            configured_family
            if configured_family
            else ("legacy_decode_mirror" if self.decode_mirror_enabled else "disabled")
        )
        self.output_head_anchor_stride = max(
            1,
            int(self.output_head_cfg.get("anchor_stride", self.decode_mirror_cfg.get("fine_stride", 1))),
        )
        loss_weights_cfg = dict(self.output_head_cfg.get("loss_weights", {}))
        if "diffusion" in loss_weights_cfg:
            self.diffusion_loss_weight = float(loss_weights_cfg["diffusion"])

        self.decode_mirror_loss_weight = float(self.decode_mirror_cfg.get("loss_weight", 0.1))
        default_coarse_weight = 1.0 if self.output_head_family in {"anchor_sparse", "anchor_refine", "dual_task"} else 0.2
        default_mid_weight = 0.0 if self.output_head_family in {"anchor_sparse", "anchor_refine"} else 0.3
        default_fine_weight = 0.0 if self.output_head_family in {"anchor_sparse", "anchor_refine"} else 0.5
        self.decode_mirror_coarse_weight = float(self.decode_mirror_cfg.get("coarse_weight", default_coarse_weight))
        self.decode_mirror_mid_weight = float(self.decode_mirror_cfg.get("mid_weight", default_mid_weight))
        self.decode_mirror_fine_weight = float(self.decode_mirror_cfg.get("fine_weight", default_fine_weight))
        self.decode_mirror_scalar_mid_weight = float(
            self.decode_mirror_cfg.get("scalar_mid_weight", 0.25),
        )
        self.decode_mirror_scalar_fine_weight = float(
            self.decode_mirror_cfg.get("scalar_fine_weight", 0.25),
        )
        self.output_refiner_enabled = False
        self.chunk_supervision_enabled = False
        self.output_chunk_ce_weight = float(loss_weights_cfg.get("chunk_ce", 0.0))
        self.chunk_size_tokens = int(
            self.diffusion_cfg.get(
                "chunk_size_tokens",
                self.hierarchy_cfg.get("coarse_chunk_size", 64),
            ),
        )
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
            if self.decode_mirror_enabled:
                if self.output_head_family in {"anchor_sparse", "anchor_refine", "dual_task"}:
                    coarse_stride = self.output_head_anchor_stride
                    mid_stride = self.output_head_anchor_stride
                    fine_stride = self.output_head_anchor_stride
                else:
                    coarse_stride = int(self.decode_mirror_cfg.get("coarse_stride", 8))
                    mid_stride = int(self.decode_mirror_cfg.get("mid_stride", 4))
                    fine_stride = int(self.decode_mirror_cfg.get("fine_stride", 1))
                self.decode_mirror_coarse_positions = _positions_from_stride(
                    self.chunk_size_tokens,
                    coarse_stride,
                )
                self.decode_mirror_mid_positions = _positions_from_stride(
                    self.chunk_size_tokens,
                    mid_stride,
                )
                self.decode_mirror_fine_positions = _positions_from_stride(
                    self.chunk_size_tokens,
                    fine_stride,
                )
                self.decode_mirror_coarse_head = nn.Linear(
                    self.model.hidden_dim,
                    len(self.decode_mirror_coarse_positions) * self.diffusion_latent_dim,
                )
                self.decode_mirror_mid_head = nn.Linear(
                    self.model.hidden_dim,
                    len(self.decode_mirror_mid_positions) * self.diffusion_latent_dim,
                )
                self.decode_mirror_fine_head = nn.Linear(
                    self.model.hidden_dim,
                    len(self.decode_mirror_fine_positions) * self.diffusion_latent_dim,
                )
                self.mid_scalar_head = nn.Linear(self.model.hidden_dim, 1)
                self.fine_scalar_head = nn.Linear(self.model.hidden_dim, 1)
                self.decode_mirror_infer_coarse_weight = float(
                    self.decode_mirror_cfg.get("infer_coarse_weight", 0.2),
                )
                self.decode_mirror_infer_mid_weight = float(
                    self.decode_mirror_cfg.get("infer_mid_weight", 0.3),
                )
                self.decode_mirror_infer_fine_weight = float(
                    self.decode_mirror_cfg.get("infer_fine_weight", 0.5),
                )
                next_token_cfg = dict(self.decode_mirror_cfg.get("next_token_supervision", {}))
                token_ce_cfg = dict(self.output_head_cfg.get("token_ce", {}))
                next_token_cfg = {**next_token_cfg, **token_ce_cfg}
                self.next_token_supervision_enabled = bool(
                    next_token_cfg.get("enabled", self.output_head_explicit),
                )
                self.next_token_supervision_loss_weight = float(
                    loss_weights_cfg.get("token_ce", next_token_cfg.get("loss_weight", 0.1)),
                )
                self.next_token_supervision_batch_size = max(
                    1,
                    int(next_token_cfg.get("sample_batch_size", 4)),
                )
                self.next_token_supervision_stride = max(
                    1,
                    int(next_token_cfg.get("position_stride", 8)),
                )
                self.next_token_supervision_max_positions = max(
                    1,
                    int(next_token_cfg.get("max_positions", 8)),
                )
                chunk_ce_cfg = dict(self.output_head_cfg.get("chunk_ce", {}))
                self.chunk_supervision_enabled = bool(
                    self.output_head_family == "dual_task"
                    and self.output_chunk_ce_weight > 0.0,
                )
                self.chunk_supervision_batch_size = max(
                    1,
                    int(chunk_ce_cfg.get("sample_batch_size", 4)),
                )
                self.chunk_supervision_sequence_stride = max(
                    1,
                    int(chunk_ce_cfg.get("sequence_stride", 2)),
                )
                self.chunk_supervision_position_stride = max(
                    1,
                    int(chunk_ce_cfg.get("position_stride", self.output_head_anchor_stride)),
                )
                self.chunk_supervision_max_positions = max(
                    1,
                    int(chunk_ce_cfg.get("max_positions", 16)),
                )
                self.chunk_supervision_max_sequences = max(
                    1,
                    int(chunk_ce_cfg.get("max_sequences", 4)),
                )
                refiner_cfg = dict(self.output_head_cfg.get("refiner", {}))
                self.output_refiner_enabled = bool(
                    refiner_cfg.get("enabled", self.output_head_family == "anchor_refine"),
                )
                if self.output_refiner_enabled:
                    self.output_refiner = nn.Sequential(
                        nn.Linear(self.diffusion_latent_dim, self.diffusion_latent_dim),
                        nn.SiLU(),
                        nn.Linear(self.diffusion_latent_dim, self.diffusion_latent_dim),
                    )
                diversity_cfg = dict(self.decode_mirror_cfg.get("diversity_regularization", {}))
                self.diversity_regularization_enabled = bool(diversity_cfg.get("enabled", False))
                self.diversity_regularization_loss_weight = float(
                    diversity_cfg.get("loss_weight", 0.05),
                )
                self.diversity_regularization_batch_size = max(
                    1,
                    int(diversity_cfg.get("sample_batch_size", 4)),
                )
                self.diversity_regularization_temperature = max(
                    1e-3,
                    float(diversity_cfg.get("temperature", 1.0)),
                )
                self.diversity_entropy_target = float(diversity_cfg.get("entropy_target", 6.0))
                self.diversity_entropy_weight = float(diversity_cfg.get("entropy_weight", 1.0))
                self.diversity_top1_max = float(diversity_cfg.get("top1_max", 0.35))
                self.diversity_top1_weight = float(diversity_cfg.get("top1_weight", 1.0))
                self.diversity_adjacent_similarity_weight = float(
                    diversity_cfg.get("adjacent_similarity_weight", 0.1),
                )
        elif self.decode_mirror_enabled:
            raise RuntimeError("decode_mirror requires diffusion.enabled=true.")
        self.nan_batches = 0

    def _project_tier_embeddings(
        self,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.decode_mirror_enabled:
            raise RuntimeError("decode_mirror projections requested while decode_mirror is disabled.")
        coarse = self.decode_mirror_coarse_head(hidden).reshape(
            hidden.shape[0],
            hidden.shape[1],
            len(self.decode_mirror_coarse_positions),
            self.diffusion_latent_dim,
        )
        mid = self.decode_mirror_mid_head(hidden).reshape(
            hidden.shape[0],
            hidden.shape[1],
            len(self.decode_mirror_mid_positions),
            self.diffusion_latent_dim,
        )
        fine = self.decode_mirror_fine_head(hidden).reshape(
            hidden.shape[0],
            hidden.shape[1],
            len(self.decode_mirror_fine_positions),
            self.diffusion_latent_dim,
        )
        return coarse, mid, fine

    def _decode_mirror_losses(
        self,
        hidden: torch.Tensor,
        target_chunk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.decode_mirror_enabled:
            zero = torch.tensor(0.0, device=hidden.device)
            return zero, zero, zero, zero
        pred_coarse, pred_mid, pred_fine = self._project_tier_embeddings(hidden)
        target_emb_full = self.chunk_token_embed(target_chunk_ids)
        target_coarse = target_emb_full[:, :, self.decode_mirror_coarse_positions, :]
        target_mid = target_emb_full[:, :, self.decode_mirror_mid_positions, :]
        target_fine = target_emb_full[:, :, self.decode_mirror_fine_positions, :]

        coarse_loss = F.smooth_l1_loss(pred_coarse, target_coarse)
        mid_loss = F.smooth_l1_loss(pred_mid, target_mid)
        fine_loss = F.smooth_l1_loss(pred_fine, target_fine)
        total = (
            self.decode_mirror_coarse_weight * coarse_loss
            + self.decode_mirror_mid_weight * mid_loss
            + self.decode_mirror_fine_weight * fine_loss
        )
        return total, coarse_loss, mid_loss, fine_loss

    def _fuse_tier_embeddings(
        self,
        pred_coarse: torch.Tensor,
        pred_mid: torch.Tensor,
        pred_fine: torch.Tensor,
        *,
        fill_missing_positions: bool = False,
    ) -> torch.Tensor:
        batch_size = int(pred_fine.shape[0])
        fused = torch.zeros(
            (batch_size, self.chunk_size_tokens, self.diffusion_latent_dim),
            device=pred_fine.device,
            dtype=pred_fine.dtype,
        )
        weights = torch.zeros(
            (batch_size, self.chunk_size_tokens, 1),
            device=pred_fine.device,
            dtype=pred_fine.dtype,
        )
        coarse_w = self.decode_mirror_infer_coarse_weight
        mid_w = self.decode_mirror_infer_mid_weight
        fine_w = self.decode_mirror_infer_fine_weight

        fused[:, self.decode_mirror_coarse_positions, :] += coarse_w * pred_coarse
        weights[:, self.decode_mirror_coarse_positions, :] += coarse_w
        fused[:, self.decode_mirror_mid_positions, :] += mid_w * pred_mid
        weights[:, self.decode_mirror_mid_positions, :] += mid_w
        fused[:, self.decode_mirror_fine_positions, :] += fine_w * pred_fine
        weights[:, self.decode_mirror_fine_positions, :] += fine_w
        if fill_missing_positions:
            present_mask = weights[0, :, 0] > 0
            if not bool(torch.all(present_mask)):
                present_positions = torch.nonzero(present_mask, as_tuple=False).squeeze(-1).tolist()
                if present_positions:
                    nearest_positions: list[int] = []
                    for pos in range(self.chunk_size_tokens):
                        if pos in present_positions:
                            nearest_positions.append(pos)
                            continue
                        nearest_positions.append(
                            min(present_positions, key=lambda src: abs(src - pos)),
                        )
                    nearest_idx = torch.tensor(
                        nearest_positions,
                        device=fused.device,
                        dtype=torch.long,
                    )
                    filled_fused = fused.index_select(dim=1, index=nearest_idx)
                    filled_weights = weights.index_select(dim=1, index=nearest_idx)
                    missing_mask = (~present_mask).to(dtype=fused.dtype).view(1, -1, 1)
                    fused = fused * (1.0 - missing_mask) + filled_fused * missing_mask
                    weights = weights * (1.0 - missing_mask) + filled_weights * missing_mask
        fused_out = fused / weights.clamp(min=1e-6)
        if self.output_refiner_enabled:
            fused_out = fused_out + self.output_refiner(fused_out)
        return fused_out

    def _next_token_supervision_metrics(
        self,
        *,
        hidden: torch.Tensor,
        target_chunk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.decode_mirror_enabled or not self.next_token_supervision_enabled:
            zero = torch.tensor(0.0, device=hidden.device)
            return zero, zero
        sample_batch = min(int(hidden.shape[0]), self.next_token_supervision_batch_size)
        if sample_batch <= 0:
            zero = torch.tensor(0.0, device=hidden.device)
            return zero, zero

        hidden_last = hidden[:sample_batch, -1, :].unsqueeze(1)
        pred_coarse, pred_mid, pred_fine = self._project_tier_embeddings(hidden_last)
        fused = self._fuse_tier_embeddings(
            pred_coarse=pred_coarse[:, 0, :, :],
            pred_mid=pred_mid[:, 0, :, :],
            pred_fine=pred_fine[:, 0, :, :],
        )
        target_last = target_chunk_ids[:sample_batch, -1, :].to(dtype=torch.long)

        positions = list(range(0, self.chunk_size_tokens, self.next_token_supervision_stride))
        if not positions:
            positions = [0]
        positions = positions[: self.next_token_supervision_max_positions]
        pos_idx = torch.tensor(positions, device=hidden.device, dtype=torch.long)

        pred_sel = fused.index_select(dim=1, index=pos_idx)
        target_sel = target_last.index_select(dim=1, index=pos_idx)

        pred_flat = F.normalize(pred_sel.reshape(-1, pred_sel.shape[-1]), dim=-1)
        target_flat = target_sel.reshape(-1)
        vocab_embed = F.normalize(self.chunk_token_embed.weight, dim=-1)
        logits = torch.matmul(pred_flat, vocab_embed.transpose(0, 1))
        ce_loss = F.cross_entropy(logits, target_flat)
        token_acc = (logits.argmax(dim=-1) == target_flat).float().mean()
        return ce_loss, token_acc

    def _chunk_supervision_metrics(
        self,
        *,
        hidden: torch.Tensor,
        target_chunk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.decode_mirror_enabled or not self.chunk_supervision_enabled:
            zero = torch.tensor(0.0, device=hidden.device)
            return zero, zero
        sample_batch = min(int(hidden.shape[0]), self.chunk_supervision_batch_size)
        if sample_batch <= 0:
            zero = torch.tensor(0.0, device=hidden.device)
            return zero, zero

        hidden_batch = hidden[:sample_batch, :, :]
        target_batch = target_chunk_ids[:sample_batch, :, :].to(dtype=torch.long)
        seq_positions = list(range(0, hidden_batch.shape[1], self.chunk_supervision_sequence_stride))
        if not seq_positions:
            seq_positions = [max(0, hidden_batch.shape[1] - 1)]
        seq_positions = seq_positions[: self.chunk_supervision_max_sequences]
        seq_idx = torch.tensor(seq_positions, device=hidden.device, dtype=torch.long)

        hidden_sel = hidden_batch.index_select(dim=1, index=seq_idx)
        target_sel = target_batch.index_select(dim=1, index=seq_idx)
        pred_coarse, pred_mid, pred_fine = self._project_tier_embeddings(hidden_sel)
        bsz, steps = int(hidden_sel.shape[0]), int(hidden_sel.shape[1])
        fused = self._fuse_tier_embeddings(
            pred_coarse=pred_coarse.reshape(
                bsz * steps,
                pred_coarse.shape[2],
                pred_coarse.shape[3],
            ),
            pred_mid=pred_mid.reshape(
                bsz * steps,
                pred_mid.shape[2],
                pred_mid.shape[3],
            ),
            pred_fine=pred_fine.reshape(
                bsz * steps,
                pred_fine.shape[2],
                pred_fine.shape[3],
            ),
        )
        target_flat = target_sel.reshape(bsz * steps, target_sel.shape[-1])
        token_positions = list(range(0, self.chunk_size_tokens, self.chunk_supervision_position_stride))
        if not token_positions:
            token_positions = [0]
        token_positions = token_positions[: self.chunk_supervision_max_positions]
        tok_idx = torch.tensor(token_positions, device=hidden.device, dtype=torch.long)

        pred_tokens = fused.index_select(dim=1, index=tok_idx)
        tgt_tokens = target_flat.index_select(dim=1, index=tok_idx)
        pred_flat = F.normalize(pred_tokens.reshape(-1, pred_tokens.shape[-1]), dim=-1)
        tgt_flat = tgt_tokens.reshape(-1)
        vocab_embed = F.normalize(self.chunk_token_embed.weight, dim=-1)
        logits = torch.matmul(pred_flat, vocab_embed.transpose(0, 1))
        ce_loss = F.cross_entropy(logits, tgt_flat)
        chunk_acc = (logits.argmax(dim=-1) == tgt_flat).float().mean()
        return ce_loss, chunk_acc

    def _diversity_regularization_metrics(
        self,
        *,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.decode_mirror_enabled or not self.diversity_regularization_enabled:
            zero = torch.tensor(0.0, device=hidden.device)
            return zero, zero, zero, zero
        sample_batch = min(int(hidden.shape[0]), self.diversity_regularization_batch_size)
        if sample_batch <= 0:
            zero = torch.tensor(0.0, device=hidden.device)
            return zero, zero, zero, zero

        hidden_last = hidden[:sample_batch, -1, :].unsqueeze(1)
        pred_coarse, pred_mid, pred_fine = self._project_tier_embeddings(hidden_last)
        fused = self._fuse_tier_embeddings(
            pred_coarse=pred_coarse[:, 0, :, :],
            pred_mid=pred_mid[:, 0, :, :],
            pred_fine=pred_fine[:, 0, :, :],
        )

        vocab_embed = F.normalize(self.chunk_token_embed.weight, dim=-1)
        fused_norm = F.normalize(fused.reshape(-1, fused.shape[-1]), dim=-1)
        logits = torch.matmul(fused_norm, vocab_embed.transpose(0, 1))
        logits = logits / self.diversity_regularization_temperature
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs.clamp(min=1e-8))

        entropy = -(probs * log_probs).sum(dim=-1)
        entropy_penalty = torch.relu(self.diversity_entropy_target - entropy).mean()
        top1_prob = probs.max(dim=-1).values
        top1_penalty = torch.relu(top1_prob - self.diversity_top1_max).mean()

        fused_seq = fused
        if fused_seq.shape[1] > 1:
            adj_cos = F.cosine_similarity(fused_seq[:, 1:, :], fused_seq[:, :-1, :], dim=-1).mean()
        else:
            adj_cos = torch.tensor(0.0, device=hidden.device)

        diversity_loss = (
            self.diversity_entropy_weight * entropy_penalty
            + self.diversity_top1_weight * top1_penalty
            + self.diversity_adjacent_similarity_weight * adj_cos
        )
        return diversity_loss, entropy.mean(), top1_prob.mean(), adj_cos

    def decode_chunk_ids_from_hidden(self, hidden_last: torch.Tensor) -> torch.Tensor:
        """Decode one coarse chunk of token ids from the final hidden state without train-memory lookup."""
        if not self.decode_mirror_enabled:
            raise RuntimeError("decode_mirror is disabled for this checkpoint/config.")
        if hidden_last.dim() == 3:
            hidden_last = hidden_last[:, -1, :]
        if hidden_last.dim() != 2:
            raise ValueError("hidden_last must have shape [batch, hidden_dim] or [batch, seq, hidden_dim].")

        context = hidden_last.unsqueeze(1)  # [B,1,H]
        pred_coarse, pred_mid, pred_fine = self._project_tier_embeddings(context)
        pred_coarse = pred_coarse[:, 0, :, :]
        pred_mid = pred_mid[:, 0, :, :]
        pred_fine = pred_fine[:, 0, :, :]
        fused = self._fuse_tier_embeddings(
            pred_coarse=pred_coarse,
            pred_mid=pred_mid,
            pred_fine=pred_fine,
            fill_missing_positions=True,
        )
        vocab_embed = F.normalize(self.chunk_token_embed.weight, dim=-1)
        fused_norm = F.normalize(fused.reshape(-1, fused.shape[-1]), dim=-1)
        similarity = torch.matmul(fused_norm, vocab_embed.transpose(0, 1))
        token_ids = similarity.argmax(dim=-1).reshape(hidden_last.shape[0], self.chunk_size_tokens)
        return token_ids

    def _diffusion_losses(
        self,
        hidden: torch.Tensor,
        target_chunk_ids: torch.Tensor,
        compute_gist_metrics: bool,
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
        if compute_gist_metrics:
            gist_cosine = F.cosine_similarity(decoded, z0, dim=-1).mean()
            decoded_flat = F.normalize(decoded.reshape(-1, decoded.shape[-1]), dim=-1)
            z0_flat = F.normalize(z0.reshape(-1, z0.shape[-1]), dim=-1)
            similarity = torch.matmul(decoded_flat, z0_flat.transpose(0, 1))
            top_idx = similarity.argmax(dim=-1)
            expected_idx = torch.arange(similarity.shape[0], device=similarity.device)
            gist_retrieval_top1 = (top_idx == expected_idx).float().mean()
        else:
            gist_cosine = torch.tensor(0.0, device=hidden.device)
            gist_retrieval_top1 = torch.tensor(0.0, device=hidden.device)
        return diffusion_loss, diffusion_noise_loss, diffusion_recon_loss, gist_cosine, gist_retrieval_top1

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        stage: str,
        log_metrics: bool = True,
    ) -> torch.Tensor:
        input_coarse = batch["input_scalars_coarse"].to(self.device)
        input_mid = batch["input_scalars_mid"].to(self.device)
        input_fine = batch["input_scalars_fine"].to(self.device)
        target_scalars = batch["target_scalars"].to(self.device)
        target_scalars_mid = batch.get("target_scalars_mid")
        target_scalars_fine = batch.get("target_scalars_fine")
        if isinstance(target_scalars_mid, torch.Tensor):
            target_scalars_mid = target_scalars_mid.to(self.device)
        if isinstance(target_scalars_fine, torch.Tensor):
            target_scalars_fine = target_scalars_fine.to(self.device)

        mask = build_causal_mask(input_coarse.shape[1], self.device)
        preds, hidden, route_probs = self.model(
            input_scalars_coarse=input_coarse,
            input_scalars_mid=input_mid,
            input_scalars_fine=input_fine,
            mask=mask,
        )
        scalar_loss = F.smooth_l1_loss(preds, target_scalars)
        mae = F.l1_loss(preds, target_scalars)
        mid_scalar_loss = torch.tensor(0.0, device=self.device)
        fine_scalar_loss = torch.tensor(0.0, device=self.device)
        if self.decode_mirror_enabled and target_scalars_mid is not None and target_scalars_fine is not None:
            mid_preds = self.mid_scalar_head(hidden).squeeze(-1)
            fine_preds = self.fine_scalar_head(hidden).squeeze(-1)
            mid_scalar_loss = F.smooth_l1_loss(mid_preds, target_scalars_mid)
            fine_scalar_loss = F.smooth_l1_loss(fine_preds, target_scalars_fine)

        route_usage = route_probs.mean(dim=(0, 1))
        route_balance_loss = ((route_usage - self.route_target) ** 2).mean()
        route_entropy = -(
            route_probs * torch.log(route_probs.clamp(min=1e-8))
        ).sum(dim=-1).mean()
        loss = scalar_loss + self.route_balance_weight * route_balance_loss
        if self.decode_mirror_enabled:
            loss = (
                loss
                + self.decode_mirror_scalar_mid_weight * mid_scalar_loss
                + self.decode_mirror_scalar_fine_weight * fine_scalar_loss
            )
        if self.route_entropy_weight > 0.0:
            loss = loss - self.route_entropy_weight * route_entropy

        diffusion_loss = torch.tensor(0.0, device=self.device)
        diffusion_noise_loss = torch.tensor(0.0, device=self.device)
        diffusion_recon_loss = torch.tensor(0.0, device=self.device)
        gist_cosine = torch.tensor(0.0, device=self.device)
        gist_retrieval_top1 = torch.tensor(0.0, device=self.device)
        decode_mirror_loss = torch.tensor(0.0, device=self.device)
        decode_mirror_coarse_loss = torch.tensor(0.0, device=self.device)
        decode_mirror_mid_loss = torch.tensor(0.0, device=self.device)
        decode_mirror_fine_loss = torch.tensor(0.0, device=self.device)
        next_token_ce_loss = torch.tensor(0.0, device=self.device)
        next_token_acc = torch.tensor(0.0, device=self.device)
        next_chunk_ce_loss = torch.tensor(0.0, device=self.device)
        next_chunk_acc = torch.tensor(0.0, device=self.device)
        diversity_loss = torch.tensor(0.0, device=self.device)
        diversity_entropy = torch.tensor(0.0, device=self.device)
        diversity_top1_prob = torch.tensor(0.0, device=self.device)
        diversity_adjacent_cos = torch.tensor(0.0, device=self.device)
        if self.diffusion_enabled:
            if "target_chunk_ids" not in batch:
                raise RuntimeError("Diffusion-enabled hierarchical run requires target_chunk_ids.")
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
                compute_gist_metrics=(stage != "train" or self.compute_train_gist_metrics),
            )
            loss = loss + self.diffusion_loss_weight * diffusion_loss
            if self.decode_mirror_enabled:
                (
                    decode_mirror_loss,
                    decode_mirror_coarse_loss,
                    decode_mirror_mid_loss,
                    decode_mirror_fine_loss,
                ) = self._decode_mirror_losses(
                    hidden=hidden,
                    target_chunk_ids=target_chunk_ids,
                )
                loss = loss + self.decode_mirror_loss_weight * decode_mirror_loss
                (
                    next_token_ce_loss,
                    next_token_acc,
                ) = self._next_token_supervision_metrics(
                    hidden=hidden,
                    target_chunk_ids=target_chunk_ids,
                )
                if self.next_token_supervision_enabled:
                    loss = loss + self.next_token_supervision_loss_weight * next_token_ce_loss
                (
                    next_chunk_ce_loss,
                    next_chunk_acc,
                ) = self._chunk_supervision_metrics(
                    hidden=hidden,
                    target_chunk_ids=target_chunk_ids,
                )
                if self.chunk_supervision_enabled:
                    loss = loss + self.output_chunk_ce_weight * next_chunk_ce_loss
                (
                    diversity_loss,
                    diversity_entropy,
                    diversity_top1_prob,
                    diversity_adjacent_cos,
                ) = self._diversity_regularization_metrics(hidden=hidden)
                if self.diversity_regularization_enabled:
                    loss = loss + self.diversity_regularization_loss_weight * diversity_loss

        if not bool(torch.isfinite(loss)):
            self.nan_batches += 1
            raise RuntimeError("Non-finite loss in hierarchical FP16 training.")

        if log_metrics:
            self.log(f"{stage}/loss", loss, prog_bar=True)
            self.log(f"{stage}/scalar_loss", scalar_loss)
            if self.decode_mirror_enabled:
                self.log(f"{stage}/scalar_mid_loss", mid_scalar_loss)
                self.log(f"{stage}/scalar_fine_loss", fine_scalar_loss)
            self.log(f"{stage}/mae", mae, prog_bar=(stage == "val"))
            self.log(f"{stage}/route_balance_loss", route_balance_loss)
            self.log(f"{stage}/route_entropy", route_entropy)
            self.log(f"{stage}/route_usage_coarse", route_usage[0])
            self.log(f"{stage}/route_usage_mid", route_usage[1])
            self.log(f"{stage}/route_usage_fine", route_usage[2])
            self.log(f"{stage}/stable", torch.tensor(1.0, device=self.device))
            if self.diffusion_enabled:
                self.log(f"{stage}/diffusion_loss", diffusion_loss)
                self.log(f"{stage}/diffusion_noise_loss", diffusion_noise_loss)
                self.log(f"{stage}/diffusion_recon_loss", diffusion_recon_loss)
                if self.decode_mirror_enabled:
                    self.log(f"{stage}/decode_mirror_loss", decode_mirror_loss)
                    self.log(f"{stage}/decode_mirror_coarse_loss", decode_mirror_coarse_loss)
                    self.log(f"{stage}/decode_mirror_mid_loss", decode_mirror_mid_loss)
                    self.log(f"{stage}/decode_mirror_fine_loss", decode_mirror_fine_loss)
                    if self.next_token_supervision_enabled:
                        self.log(f"{stage}/next_token_ce", next_token_ce_loss)
                        self.log(f"{stage}/next_token_acc", next_token_acc, prog_bar=(stage == "val"))
                    if self.chunk_supervision_enabled:
                        self.log(f"{stage}/next_chunk_ce", next_chunk_ce_loss)
                        self.log(f"{stage}/next_chunk_acc", next_chunk_acc, prog_bar=(stage == "val"))
                    if self.diversity_regularization_enabled:
                        self.log(f"{stage}/diversity_loss", diversity_loss)
                        self.log(f"{stage}/diversity_entropy", diversity_entropy)
                        self.log(f"{stage}/diversity_top1_prob", diversity_top1_prob)
                        self.log(f"{stage}/diversity_adjacent_cos", diversity_adjacent_cos)
                if stage != "train" or self.compute_train_gist_metrics:
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


def run_fp16_hierarchical_feasibility(
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
    hierarchy_cfg = dict(fp16_cfg.get("hierarchical", {}))
    if not bool(hierarchy_cfg.get("enabled", False)):
        raise RuntimeError("Hierarchical run requested but fp16_chunk.hierarchical.enabled is false.")

    coarse_chunk_size = int(hierarchy_cfg.get("coarse_chunk_size", compression_cfg.get("chunk_size_tokens", 64)))
    mid_chunk_size = int(hierarchy_cfg.get("mid_chunk_size", 8))
    fine_chunk_size = int(hierarchy_cfg.get("fine_chunk_size", 1))
    _validate_hierarchy_sizes(coarse_chunk_size, mid_chunk_size, fine_chunk_size)
    if coarse_chunk_size != int(compression_cfg.get("chunk_size_tokens", coarse_chunk_size)):
        raise RuntimeError("compression.chunk_size_tokens must match hierarchical.coarse_chunk_size.")

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

    data_module = HierarchicalFP16DataModule(
        train_texts=train_texts,
        val_texts=val_texts,
        tokenizer_name=str(compression_cfg.get("tokenizer_name", "gpt2")),
        force_fast_tokenizer=bool(compression_cfg.get("force_fast_tokenizer", True)),
        dtype=str(compression_cfg.get("dtype", "float16")),
        coarse_chunk_size=coarse_chunk_size,
        mid_chunk_size=mid_chunk_size,
        fine_chunk_size=fine_chunk_size,
        compressed_seq_len=int(training_cfg.get("compressed_seq_len", 8)),
        window_stride=int(training_cfg.get("window_stride", 1)),
        tokenizer_batch_size=int(compression_cfg.get("tokenizer_batch_size", 64)),
        batch_size=int(training_cfg.get("batch_size", 64)),
        num_workers=int(training_cfg.get("num_workers", 0)),
        seed=seed,
        accelerator=accelerator,
        include_chunk_tokens=diffusion_enabled,
    )
    # Build vocab_size in setup before model init.
    data_module.setup()
    train_coarse_path = output_dir / "train_scalars_coarse.npy"
    train_mid_path = output_dir / "train_scalars_mid.npy"
    train_fine_path = output_dir / "train_scalars_fine.npy"
    train_chunk_tokens_path: Path | None = None
    if data_module.train_coarse_stream is None or data_module.train_mid_stream is None or data_module.train_fine_stream is None:
        raise RuntimeError("Hierarchical train streams were not built during setup.")
    np.save(train_coarse_path, data_module.train_coarse_stream)
    np.save(train_mid_path, data_module.train_mid_stream)
    np.save(train_fine_path, data_module.train_fine_stream)
    if data_module.train_chunk_tokens is not None:
        train_chunk_tokens_path = output_dir / "train_chunk_tokens.npy"
        np.save(train_chunk_tokens_path, data_module.train_chunk_tokens)

    model_config = {
        **fp16_cfg,
        "hierarchical": hierarchy_cfg,
        "diffusion": {
            **diffusion_cfg,
            "vocab_size": int(diffusion_cfg.get("vocab_size", data_module.vocab_size)),
            "chunk_size_tokens": coarse_chunk_size,
        },
    }
    model = HierarchicalFP16ChunkLMModule(config=model_config)

    runtime_metrics = RuntimeMetrics(chunk_size_tokens=coarse_chunk_size)
    runtime_callback = RuntimeMetricsCallback(runtime_metrics=runtime_metrics)
    trend_callback = ValidationTrendCallback()
    eta_callback = EtaMetricsCallback(
        log_interval_steps=int(training_cfg.get("eta_log_interval_steps", 20)),
        emit_stdout=bool(training_cfg.get("eta_stdout", True)),
    )
    callbacks: list[Callback] = [runtime_callback, trend_callback, eta_callback]

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_every_n_train_steps = int(training_cfg.get("checkpoint_every_n_train_steps", 0))
    checkpoint_save_top_k = int(training_cfg.get("checkpoint_save_top_k", 1))
    checkpoint_save_last = bool(training_cfg.get("checkpoint_save_last", True))
    checkpoint_monitor = str(training_cfg.get("checkpoint_monitor", "val/loss"))
    checkpoint_mode = str(training_cfg.get("checkpoint_mode", "")).strip().lower()
    if checkpoint_mode not in {"min", "max"}:
        checkpoint_mode = "max" if checkpoint_monitor.endswith("_acc") else "min"
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
            monitor=checkpoint_monitor,
            mode=checkpoint_mode,
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
        "val_next_token_ce_last": (
            float(trainer.callback_metrics["val/next_token_ce"].item())
            if "val/next_token_ce" in trainer.callback_metrics
            else None
        ),
        "val_next_token_acc_last": (
            float(trainer.callback_metrics["val/next_token_acc"].item())
            if "val/next_token_acc" in trainer.callback_metrics
            else None
        ),
        "val_next_chunk_ce_last": (
            float(trainer.callback_metrics["val/next_chunk_ce"].item())
            if "val/next_chunk_ce" in trainer.callback_metrics
            else None
        ),
        "val_next_chunk_acc_last": (
            float(trainer.callback_metrics["val/next_chunk_acc"].item())
            if "val/next_chunk_acc" in trainer.callback_metrics
            else None
        ),
        "val_route_usage_coarse_last": (
            float(trainer.callback_metrics["val/route_usage_coarse"].item())
            if "val/route_usage_coarse" in trainer.callback_metrics
            else None
        ),
        "val_route_usage_mid_last": (
            float(trainer.callback_metrics["val/route_usage_mid"].item())
            if "val/route_usage_mid" in trainer.callback_metrics
            else None
        ),
        "val_route_usage_fine_last": (
            float(trainer.callback_metrics["val/route_usage_fine"].item())
            if "val/route_usage_fine" in trainer.callback_metrics
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
        "coarse_chunk_size": coarse_chunk_size,
        "mid_chunk_size": mid_chunk_size,
        "fine_chunk_size": fine_chunk_size,
        "output_head_family": getattr(model, "output_head_family", "disabled"),
        "output_head_anchor_stride": int(getattr(model, "output_head_anchor_stride", 0)),
        "output_head_refiner_enabled": bool(getattr(model, "output_refiner_enabled", False)),
        "gate_pass": bool(gate_pass),
        "metrics_csv_path": metrics_csv_path,
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir.exists() else None,
        "train_scalars_coarse_path": str(train_coarse_path),
        "train_scalars_mid_path": str(train_mid_path),
        "train_scalars_fine_path": str(train_fine_path),
        "train_chunk_tokens_path": str(train_chunk_tokens_path) if train_chunk_tokens_path is not None else None,
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
