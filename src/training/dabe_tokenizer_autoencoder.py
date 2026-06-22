from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
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

from .fp16_chunk_feasibility import EtaMetricsCallback, _load_text_dataset_samples, _resolve_precision


def _synthetic_codec_texts(max_samples: int) -> list[str]:
    base = [
        "Lina packed a red kite, a brass key, and a tiny notebook before sunrise.",
        "The robot gardener counted blue seeds while rain tapped the greenhouse roof.",
        "Milo found three maps inside the clock and followed the one with silver ink.",
        "A patient dog carried warm bread across the bridge for the sleepy baker.",
    ]
    return [" ".join([base[idx % len(base)]] * 8) for idx in range(max(1, int(max_samples)))]


@dataclass(frozen=True)
class DABETokenizerAutoencoderOutput:
    """Forward output for DABE tokenizer objective, paper section 3."""

    logits: torch.Tensor
    bits: torch.Tensor
    bit_logits: torch.Tensor
    gist_logits: torch.Tensor | None = None
    diffusion_embed_mse: torch.Tensor | None = None
    lookup_positions: torch.Tensor | None = None
    lookup_token_ids: torch.Tensor | None = None
    lookup_gate: torch.Tensor | None = None
    lookup_active_mask: torch.Tensor | None = None
    lookup_budget_k: torch.Tensor | None = None
    lookup_keep_logits: torch.Tensor | None = None
    lookup_keep_probs: torch.Tensor | None = None
    selector_logits: torch.Tensor | None = None
    selector_target_mask: torch.Tensor | None = None
    budget_logits: torch.Tensor | None = None
    budget_target: torch.Tensor | None = None
    residual_router_logits: torch.Tensor | None = None
    residual_router_target: torch.Tensor | None = None
    residual_info_scores: torch.Tensor | None = None
    variable_window_logits: torch.Tensor | None = None
    variable_window_target: torch.Tensor | None = None
    variable_window_probs: torch.Tensor | None = None
    variable_window_soft_probs: torch.Tensor | None = None
    variable_window_bits_per_chunk: torch.Tensor | None = None
    variable_window_action_deviation: torch.Tensor | None = None
    variable_window_base_deviation: torch.Tensor | None = None
    variable_window_oracle_deviation: torch.Tensor | None = None


def chunk_deviation_stats(matches: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return fixed-length chunk Hamming deviation stats for section-6 analysis."""
    if matches.ndim != 2:
        raise ValueError("matches must have shape (batch, chunk_size_tokens).")
    deviation = (~matches.bool()).float().sum(dim=1)
    chunk_size = max(1, int(matches.shape[1]))
    return {
        "mean": deviation.mean(),
        "rate_mean": deviation.mean() / float(chunk_size),
        "p50": torch.quantile(deviation, 0.50),
        "p90": torch.quantile(deviation, 0.90),
        "p95": torch.quantile(deviation, 0.95),
        "max": deviation.max(),
    }


def _python_code_codec_texts(max_samples: int) -> list[str]:
    """Deterministic Python-like snippets for repair-trace domain diagnostics."""
    snippets = [
        'def normalize_counts(items):\n    totals = {}\n    for name, value in items:\n        key = name.strip().lower()\n        totals[key] = totals.get(key, 0) + int(value)\n    return {k: v / max(1, sum(totals.values())) for k, v in totals.items()}\n',
        'class RollingMean:\n    def __init__(self, window=8):\n        self.window = window\n        self.values = []\n\n    def update(self, x):\n        self.values.append(float(x))\n        self.values = self.values[-self.window:]\n        return sum(self.values) / len(self.values)\n',
        "def parse_record(line):\n    user_id, score, flag = line.rstrip().split(',')\n    if flag == 'skip':\n        return None\n    return {'user_id': user_id, 'score': float(score), 'active': flag == 'active'}\n",
        "async def fetch_json(session, url):\n    async with session.get(url, timeout=10) as response:\n        response.raise_for_status()\n        payload = await response.json()\n    return payload.get('items', [])\n",
    ]
    repeated = []
    for idx in range(max(1, int(max_samples))):
        snippet = snippets[idx % len(snippets)]
        repeated.append((snippet + "\n") * 4)
    return repeated


def _resolve_tokenizer_texts(dataset_cfg: dict[str, Any], experiment_cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    dataset_name = str(dataset_cfg.get("dataset_name", "roneneldan/TinyStories"))
    if dataset_name in {"synthetic", "__synthetic__"}:
        return (
            _synthetic_codec_texts(int(dataset_cfg.get("train_samples", 256))),
            _synthetic_codec_texts(int(dataset_cfg.get("val_samples", 64))),
        )
    if dataset_name in {"python_code", "__python_code__", "code", "__code__"}:
        return (
            _python_code_codec_texts(int(dataset_cfg.get("train_samples", 256))),
            _python_code_codec_texts(int(dataset_cfg.get("val_samples", 64))),
        )
    train_texts = _load_text_dataset_samples(
        dataset_name=dataset_name,
        split=str(dataset_cfg.get("train_split", "train")),
        text_field=str(dataset_cfg.get("text_field", "text")),
        max_samples=int(dataset_cfg.get("train_samples", 256)),
        seed=int(experiment_cfg.get("seed", 42)),
        allow_synthetic_fallback=bool(dataset_cfg.get("allow_synthetic_fallback", True)),
        streaming=bool(dataset_cfg.get("streaming", True)),
    )
    val_texts = _load_text_dataset_samples(
        dataset_name=dataset_name,
        split=str(dataset_cfg.get("val_split", "validation")),
        text_field=str(dataset_cfg.get("text_field", "text")),
        max_samples=int(dataset_cfg.get("val_samples", 64)),
        seed=int(experiment_cfg.get("seed", 42)) + 1,
        allow_synthetic_fallback=bool(dataset_cfg.get("allow_synthetic_fallback", True)),
        streaming=bool(dataset_cfg.get("streaming", True)),
    )
    return train_texts, val_texts


def build_token_chunk_array(
    *,
    texts: Sequence[str],
    tokenizer: Any,
    chunk_size_tokens: int,
    chunk_stride_tokens: int,
    tokenizer_batch_size: int,
) -> np.ndarray:
    """Convert text samples to full BPE-token chunks for section 3 codec training."""
    chunk_size = int(chunk_size_tokens)
    stride = max(1, int(chunk_stride_tokens))
    if chunk_size <= 0:
        raise ValueError("chunk_size_tokens must be positive.")

    chunks: list[np.ndarray] = []
    batch_size = max(1, int(tokenizer_batch_size))
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        encoded = tokenizer(batch, add_special_tokens=False)
        ids_per_text = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        for token_ids in ids_per_text:
            ids = np.asarray(token_ids, dtype=np.int64)
            if ids.shape[0] < chunk_size:
                continue
            starts = np.arange(0, ids.shape[0] - chunk_size + 1, stride, dtype=np.int64)
            for chunk_start in starts:
                chunks.append(ids[chunk_start : chunk_start + chunk_size])

    if not chunks:
        return np.empty((0, chunk_size), dtype=np.int64)
    return np.stack(chunks, axis=0).astype(np.int64, copy=False)


class TokenChunkDataset(TorchDataset):
    def __init__(self, chunks: np.ndarray):
        if chunks.ndim != 2:
            raise ValueError("chunks must have shape (num_chunks, chunk_size_tokens).")
        self.chunks = np.asarray(chunks, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.chunks.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.from_numpy(np.asarray(self.chunks[idx], dtype=np.int64))}


class DABETokenizerDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        train_texts: Sequence[str],
        val_texts: Sequence[str],
        tokenizer_name: str,
        force_fast_tokenizer: bool,
        chunk_size_tokens: int,
        chunk_stride_tokens: int,
        tokenizer_batch_size: int,
        batch_size: int,
        num_workers: int,
        seed: int,
        accelerator: str,
    ):
        super().__init__()
        self.train_texts = list(train_texts)
        self.val_texts = list(val_texts)
        self.tokenizer_name = str(tokenizer_name)
        self.force_fast_tokenizer = bool(force_fast_tokenizer)
        self.chunk_size_tokens = int(chunk_size_tokens)
        self.chunk_stride_tokens = int(chunk_stride_tokens)
        self.tokenizer_batch_size = int(tokenizer_batch_size)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.seed = int(seed)
        self.accelerator = str(accelerator)
        self.vocab_size = 50257
        self.train_chunks: np.ndarray | None = None
        self.val_chunks: np.ndarray | None = None
        self._is_setup = False

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self._is_setup:
            return

        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            use_fast=self.force_fast_tokenizer,
        )
        if self.force_fast_tokenizer and not bool(getattr(tokenizer, "is_fast", False)):
            raise RuntimeError(f"Expected fast tokenizer for {self.tokenizer_name}.")
        self.vocab_size = max(1, int(getattr(tokenizer, "vocab_size", 50257)))

        self.train_chunks = build_token_chunk_array(
            texts=self.train_texts,
            tokenizer=tokenizer,
            chunk_size_tokens=self.chunk_size_tokens,
            chunk_stride_tokens=self.chunk_stride_tokens,
            tokenizer_batch_size=self.tokenizer_batch_size,
        )
        self.val_chunks = build_token_chunk_array(
            texts=self.val_texts,
            tokenizer=tokenizer,
            chunk_size_tokens=self.chunk_size_tokens,
            chunk_stride_tokens=self.chunk_stride_tokens,
            tokenizer_batch_size=self.tokenizer_batch_size,
        )
        if self.train_chunks.shape[0] == 0:
            raise RuntimeError("DABE tokenizer autoencoder produced zero train chunks.")
        if self.val_chunks.shape[0] == 0:
            raise RuntimeError("DABE tokenizer autoencoder produced zero val chunks.")

        self.train_dataset = TokenChunkDataset(self.train_chunks)
        self.val_dataset = TokenChunkDataset(self.val_chunks)
        self._is_setup = True

    def _dataloader_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"num_workers": self.num_workers}
        if self.accelerator == "gpu" and self.num_workers > 0:
            kwargs.update({"pin_memory": True, "persistent_workers": True, "prefetch_factor": 2})
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
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, **self._dataloader_kwargs())


class DABEChunkTokenizerAutoencoder(nn.Module):
    """Fixed-width chunk codec for DABE tokenizer training, paper section 3."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.vocab_size = int(config.get("vocab_size", 50257))
        self.chunk_size_tokens = int(config.get("chunk_size_tokens", 64))
        self.code_bits = int(config.get("code_bits", 256))
        self.embed_dim = int(config.get("embed_dim", 128))
        self.hidden_dim = int(config.get("hidden_dim", 512))
        self.dropout = float(config.get("dropout", 0.1))
        self.decoder_mode = str(config.get("decoder_mode", "mirror"))
        self.diffusion_steps = max(1, int(config.get("diffusion_steps", 64)))
        self.diffusion_layers = max(1, int(config.get("diffusion_layers", 2)))
        self.diffusion_heads = max(1, int(config.get("diffusion_heads", 4)))
        self.code_decoder_layers = max(1, int(config.get("code_decoder_layers", 2)))
        self.code_decoder_heads = max(1, int(config.get("code_decoder_heads", 4)))
        self.hierarchical_block_tokens = max(1, int(config.get("hierarchical_block_tokens", 16)))
        self.hierarchical_refine_layers = max(0, int(config.get("hierarchical_refine_layers", 1)))
        self.hierarchical_decoder_layers = max(1, int(config.get("hierarchical_decoder_layers", 2)))
        self.hierarchical_decoder_heads = max(1, int(config.get("hierarchical_decoder_heads", 4)))
        self.hierarchical_local_refine_layers = max(0, int(config.get("hierarchical_local_refine_layers", 1)))
        self.hierarchical_local_refine_causal = bool(config.get("hierarchical_local_refine_causal", True))
        self.lexical_lookup_k = max(0, int(config.get("lexical_lookup_k", 8)))
        self.lexical_lookup_heads = max(1, int(config.get("lexical_lookup_heads", self.hierarchical_decoder_heads)))
        self.lexical_lookup_copy_scale = float(config.get("lexical_lookup_copy_scale", 4.0))
        self.lexical_lookup_selector = str(config.get("lexical_lookup_selector", "oracle_loss"))
        self.lexical_lookup_k_schedule = self._parse_lookup_k_schedule(config.get("lexical_lookup_k_schedule", ""))
        self.lexical_lookup_slot_policy = str(config.get("lexical_lookup_slot_policy", "fixed"))
        self.lexical_lookup_keep_threshold = float(config.get("lexical_lookup_keep_threshold", 0.5))
        self.residual_router_levels = 3
        self.residual_router_high_quantile = float(config.get("residual_router_high_quantile", 0.75))
        self.residual_router_medium_quantile = float(config.get("residual_router_medium_quantile", 0.50))
        self.residual_router_med_score = float(config.get("residual_router_med_score", 0.5))
        self.variable_window_levels = 3
        self.variable_window_fine_quantile = float(config.get("variable_window_fine_quantile", 0.75))
        self.variable_window_medium_quantile = float(config.get("variable_window_medium_quantile", 0.50))
        self.variable_window_refine_layers = max(1, int(config.get("variable_window_refine_layers", 1)))
        self.variable_window_target_mode = str(config.get("variable_window_target_mode", "quantile"))
        self.variable_window_oracle_cost_lambda = float(config.get("variable_window_oracle_cost_lambda", 0.0))
        self.variable_window_mixing_mode = str(config.get("variable_window_mixing_mode", "soft"))
        self.variable_window_temperature = max(1e-3, float(config.get("variable_window_temperature", 1.0)))
        self.sliding_global_bits = max(1, int(config.get("sliding_global_bits", 96)))
        self.sliding_medium_window = max(1, int(config.get("sliding_medium_window", 32)))
        self.sliding_medium_stride = max(1, int(config.get("sliding_medium_stride", 16)))
        self.sliding_medium_bits = max(1, int(config.get("sliding_medium_bits", 64)))
        self.sliding_fine_window = max(1, int(config.get("sliding_fine_window", 16)))
        self.sliding_fine_stride = max(1, int(config.get("sliding_fine_stride", 8)))
        self.sliding_fine_bits = max(1, int(config.get("sliding_fine_bits", 32)))
        self.sliding_refine_layers = max(0, int(config.get("sliding_refine_layers", 1)))
        self.sliding_decoder_layers = max(1, int(config.get("sliding_decoder_layers", 1)))
        self.sliding_decoder_heads = max(1, int(config.get("sliding_decoder_heads", 4)))

        if self.chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be positive.")
        if self.code_bits <= 0:
            raise ValueError("code_bits must be positive.")
        if self.decoder_mode not in {
            "mirror",
            "diffusion",
            "code_transformer",
            "hierarchical_local",
            "hierarchical_local_refine",
            "hierarchical_lookup",
            "gist_residual_lookup",
            "gist_residual_variable_windows",
            "sliding_progressive",
        }:
            raise ValueError(
                "decoder_mode must be 'mirror', 'diffusion', 'code_transformer', "
                "'hierarchical_local', 'hierarchical_local_refine', 'hierarchical_lookup', "
                "'gist_residual_lookup', 'gist_residual_variable_windows', or 'sliding_progressive'."
            )
        if self.lexical_lookup_selector not in {"oracle_loss", "learned"}:
            raise ValueError("lexical_lookup_selector must be 'oracle_loss' or 'learned'.")
        if self.lexical_lookup_slot_policy not in {"fixed", "halting"}:
            raise ValueError("lexical_lookup_slot_policy must be 'fixed' or 'halting'.")
        if self.variable_window_target_mode not in {"quantile", "action_value"}:
            raise ValueError("variable_window_target_mode must be 'quantile' or 'action_value'.")
        if self.variable_window_mixing_mode not in {"soft", "straight_through", "hard"}:
            raise ValueError("variable_window_mixing_mode must be 'soft', 'straight_through', or 'hard'.")
        if self.embed_dim % self.diffusion_heads != 0:
            raise ValueError("embed_dim must be divisible by diffusion_heads.")
        if self.embed_dim % self.code_decoder_heads != 0:
            raise ValueError("embed_dim must be divisible by code_decoder_heads.")
        if self.embed_dim % self.hierarchical_decoder_heads != 0:
            raise ValueError("embed_dim must be divisible by hierarchical_decoder_heads.")
        if self.embed_dim % self.lexical_lookup_heads != 0:
            raise ValueError("embed_dim must be divisible by lexical_lookup_heads.")
        if self.embed_dim % self.sliding_decoder_heads != 0:
            raise ValueError("embed_dim must be divisible by sliding_decoder_heads.")
        if self.chunk_size_tokens % self.hierarchical_block_tokens != 0:
            raise ValueError("chunk_size_tokens must be divisible by hierarchical_block_tokens.")
        self.hierarchical_num_blocks = self.chunk_size_tokens // self.hierarchical_block_tokens
        if self.decoder_mode == "gist_residual_variable_windows" and self.hierarchical_num_blocks % 2 != 0:
            raise ValueError("gist_residual_variable_windows requires an even number of hierarchical blocks.")
        if self.code_bits % self.hierarchical_num_blocks != 0:
            raise ValueError("code_bits must be divisible by the number of hierarchical blocks.")
        self.hierarchical_code_bits_per_block = self.code_bits // self.hierarchical_num_blocks
        self.sliding_specs = self._build_sliding_specs()
        self.sliding_total_bits = sum(len(starts) * bits for _, _window, bits, starts in self.sliding_specs)
        if self.decoder_mode == "sliding_progressive" and self.sliding_total_bits != self.code_bits:
            raise ValueError(
                "sliding_progressive bit allocation must equal code_bits; "
                f"got {self.sliding_total_bits} vs {self.code_bits}."
            )

        self.token_embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.encoder_pos = nn.Parameter(torch.randn(self.chunk_size_tokens, self.embed_dim) * 0.02)
        self.encoder = nn.Sequential(
            nn.LayerNorm(self.chunk_size_tokens * self.embed_dim),
            nn.Linear(self.chunk_size_tokens * self.embed_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.bit_proj = nn.Linear(self.hidden_dim, self.code_bits)
        self.hierarchical_block_encoder = nn.Sequential(
            nn.LayerNorm(self.hierarchical_block_tokens * self.embed_dim),
            nn.Linear(self.hierarchical_block_tokens * self.embed_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.hierarchical_bit_proj = nn.Linear(self.hidden_dim, self.hierarchical_code_bits_per_block)
        self.sliding_window_encoders = nn.ModuleDict()
        self.sliding_bit_projs = nn.ModuleDict()
        self.sliding_code_to_window = nn.ModuleDict()
        for name, window_tokens, bits_per_window, _starts in self.sliding_specs:
            self.sliding_window_encoders[name] = nn.Sequential(
                nn.LayerNorm(window_tokens * self.embed_dim),
                nn.Linear(window_tokens * self.embed_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
            )
            self.sliding_bit_projs[name] = nn.Linear(self.hidden_dim, bits_per_window)
            self.sliding_code_to_window[name] = nn.Sequential(
                nn.Linear(bits_per_window, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, window_tokens * self.embed_dim),
            )
        self.code_to_chunk = nn.Sequential(
            nn.Linear(self.code_bits, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.chunk_size_tokens * self.embed_dim),
        )
        self.hierarchical_code_to_block = nn.Sequential(
            nn.Linear(self.hierarchical_code_bits_per_block, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hierarchical_block_tokens * self.embed_dim),
        )
        self.decoder_pos = nn.Parameter(torch.randn(self.chunk_size_tokens, self.embed_dim) * 0.02)
        self.decoder = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.embed_dim),
            nn.GELU(),
        )
        self.diffusion_time_embed = nn.Embedding(self.diffusion_steps, self.embed_dim)
        diffusion_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.diffusion_heads,
            dim_feedforward=self.hidden_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.diffusion_denoiser = nn.TransformerEncoder(diffusion_layer, num_layers=self.diffusion_layers)
        self.diffusion_norm = nn.LayerNorm(self.embed_dim)
        code_decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.code_decoder_heads,
            dim_feedforward=self.hidden_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.code_transformer_decoder = nn.TransformerEncoder(code_decoder_layer, num_layers=self.code_decoder_layers)
        self.code_transformer_norm = nn.LayerNorm(self.embed_dim)
        hierarchical_decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.hierarchical_decoder_heads,
            dim_feedforward=self.hidden_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.hierarchical_block_decoder = nn.TransformerEncoder(
            hierarchical_decoder_layer,
            num_layers=self.hierarchical_decoder_layers,
        )
        if self.hierarchical_local_refine_layers > 0:
            hierarchical_local_refine_layer = nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=self.hierarchical_decoder_heads,
                dim_feedforward=self.hidden_dim,
                dropout=self.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.hierarchical_local_refiner = nn.TransformerEncoder(
                hierarchical_local_refine_layer,
                num_layers=self.hierarchical_local_refine_layers,
            )
        else:
            self.hierarchical_local_refiner = nn.Identity()
        self.hierarchical_local_refine_norm = nn.LayerNorm(self.embed_dim)
        self.register_buffer(
            "hierarchical_local_causal_mask",
            torch.triu(
                torch.ones(self.hierarchical_block_tokens, self.hierarchical_block_tokens, dtype=torch.bool),
                diagonal=1,
            ),
            persistent=False,
        )
        lookup_slots = max(1, self.max_lexical_lookup_k)
        self.lexical_lookup_slot_embed = nn.Parameter(torch.randn(lookup_slots, self.embed_dim) * 0.02)
        self.lexical_lookup_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.lexical_lookup_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.lexical_lookup_norm = nn.LayerNorm(self.embed_dim)
        self.lexical_lookup_gate = nn.Linear(self.embed_dim, 1)
        self.lexical_lookup_query = nn.Linear(self.embed_dim, self.embed_dim)
        self.lexical_lookup_key = nn.Linear(self.embed_dim, self.embed_dim)
        self.lexical_selector = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        self.lexical_budget_selector = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, len(self.lexical_lookup_k_choices)),
        )
        self.lexical_slot_halting = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        self.residual_router = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.residual_router_levels),
        )
        self.residual_level_embed = nn.Parameter(torch.randn(self.residual_router_levels, self.embed_dim) * 0.02)
        self.variable_window_router = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.variable_window_levels),
        )
        variable_window_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.hierarchical_decoder_heads,
            dim_feedforward=self.hidden_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.variable_window_refiner = nn.TransformerEncoder(
            variable_window_layer,
            num_layers=self.variable_window_refine_layers,
        )
        self.variable_window_norm = nn.LayerNorm(self.embed_dim)
        self.variable_window_level_embed = nn.Parameter(
            torch.randn(self.variable_window_levels, self.embed_dim) * 0.02
        )
        if self.hierarchical_refine_layers > 0:
            hierarchical_refine_layer = nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=self.hierarchical_decoder_heads,
                dim_feedforward=self.hidden_dim,
                dropout=self.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.hierarchical_refiner = nn.TransformerEncoder(
                hierarchical_refine_layer,
                num_layers=self.hierarchical_refine_layers,
            )
        else:
            self.hierarchical_refiner = nn.Identity()
        self.hierarchical_norm = nn.LayerNorm(self.embed_dim)
        sliding_window_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.sliding_decoder_heads,
            dim_feedforward=self.hidden_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sliding_window_decoder = nn.TransformerEncoder(
            sliding_window_layer,
            num_layers=self.sliding_decoder_layers,
        )
        if self.sliding_refine_layers > 0:
            sliding_refine_layer = nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=self.sliding_decoder_heads,
                dim_feedforward=self.hidden_dim,
                dropout=self.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.sliding_refiner = nn.TransformerEncoder(sliding_refine_layer, num_layers=self.sliding_refine_layers)
        else:
            self.sliding_refiner = nn.Identity()
        self.sliding_level_embed = nn.Parameter(torch.randn(len(self.sliding_specs), self.embed_dim) * 0.02)
        self.sliding_norm = nn.LayerNorm(self.embed_dim)
        self.output_proj = nn.Linear(self.embed_dim, self.vocab_size)

    def _parse_lookup_k_schedule(self, raw: Any) -> list[int]:
        if raw is None or raw == "":
            return []
        if isinstance(raw, str):
            normalized = raw.replace("|", ",").replace(";", ",")
            parts = [part.strip() for part in normalized.split(",") if part.strip()]
        elif isinstance(raw, Sequence):
            parts = list(raw)
        else:
            parts = [raw]
        values = sorted({max(1, int(part)) for part in parts})
        return values

    @property
    def lexical_lookup_k_choices(self) -> list[int]:
        choices = self.lexical_lookup_k_schedule or [self.lexical_lookup_k]
        return sorted({min(max(0, int(choice)), self.chunk_size_tokens) for choice in choices})

    @property
    def max_lexical_lookup_k(self) -> int:
        return max(self.lexical_lookup_k_choices)

    def _build_sliding_specs(self) -> list[tuple[str, int, int, list[int]]]:
        def _starts(window: int, stride: int) -> list[int]:
            if window > self.chunk_size_tokens:
                raise ValueError("sliding window cannot exceed chunk_size_tokens.")
            starts = list(range(0, self.chunk_size_tokens - window + 1, stride))
            final_start = self.chunk_size_tokens - window
            if starts[-1] != final_start:
                starts.append(final_start)
            return starts

        return [
            ("global", self.chunk_size_tokens, self.sliding_global_bits, [0]),
            (
                "medium",
                self.sliding_medium_window,
                self.sliding_medium_bits,
                _starts(self.sliding_medium_window, self.sliding_medium_stride),
            ),
            (
                "fine",
                self.sliding_fine_window,
                self.sliding_fine_bits,
                _starts(self.sliding_fine_window, self.sliding_fine_stride),
            ),
        ]

    @property
    def bits_per_token(self) -> float:
        return float(self.code_bits) / float(self.chunk_size_tokens)

    @property
    def lookup_bits_per_chunk(self) -> int:
        if (
            self.decoder_mode
            not in {"hierarchical_lookup", "gist_residual_lookup", "gist_residual_variable_windows"}
            or self.max_lexical_lookup_k <= 0
        ):
            return 0
        return int(self.max_lexical_lookup_k * self.lookup_entry_bits)

    @property
    def lookup_entry_bits(self) -> int:
        token_bits = max(1, int(math.ceil(math.log2(max(2, self.vocab_size)))))
        position_bits = max(1, int(math.ceil(math.log2(max(2, self.chunk_size_tokens)))))
        return int(token_bits + position_bits)

    @property
    def effective_bits_per_token(self) -> float:
        return float(self.code_bits + self.lookup_bits_per_chunk) / float(self.chunk_size_tokens)

    def encode_bits(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.decoder_mode in {
            "hierarchical_local",
            "hierarchical_local_refine",
            "hierarchical_lookup",
            "gist_residual_lookup",
            "gist_residual_variable_windows",
        }:
            return self._encode_hierarchical_bits(input_ids)
        if self.decoder_mode == "sliding_progressive":
            return self._encode_sliding_bits(input_ids)
        embedded = self.token_embed(input_ids) + self.encoder_pos.unsqueeze(0)
        hidden = self.encoder(embedded.reshape(embedded.shape[0], -1))
        bit_logits = self.bit_proj(hidden)
        soft_bits = torch.tanh(bit_logits)
        hard_bits = (soft_bits > 0).float()
        bits = hard_bits + soft_bits - soft_bits.detach()
        return bits, bit_logits

    def _encode_hierarchical_bits(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(input_ids.shape[0])
        embedded = self.token_embed(input_ids) + self.encoder_pos.unsqueeze(0)
        blocks = embedded.reshape(
            batch_size,
            self.hierarchical_num_blocks,
            self.hierarchical_block_tokens,
            self.embed_dim,
        )
        hidden = self.hierarchical_block_encoder(
            blocks.reshape(batch_size * self.hierarchical_num_blocks, -1)
        )
        bit_logits = self.hierarchical_bit_proj(hidden).reshape(
            batch_size,
            self.hierarchical_num_blocks,
            self.hierarchical_code_bits_per_block,
        )
        soft_bits = torch.tanh(bit_logits)
        hard_bits = (soft_bits > 0).float()
        bits = hard_bits + soft_bits - soft_bits.detach()
        return bits.reshape(batch_size, self.code_bits), bit_logits.reshape(batch_size, self.code_bits)

    def _encode_sliding_bits(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(input_ids.shape[0])
        embedded = self.token_embed(input_ids) + self.encoder_pos.unsqueeze(0)
        bits_parts: list[torch.Tensor] = []
        logits_parts: list[torch.Tensor] = []
        for name, window_tokens, _bits_per_window, starts in self.sliding_specs:
            windows = torch.stack([embedded[:, start : start + window_tokens, :] for start in starts], dim=1)
            hidden = self.sliding_window_encoders[name](
                windows.reshape(batch_size * len(starts), window_tokens * self.embed_dim)
            )
            bit_logits = self.sliding_bit_projs[name](hidden).reshape(batch_size, len(starts), -1)
            soft_bits = torch.tanh(bit_logits)
            hard_bits = (soft_bits > 0).float()
            bits = hard_bits + soft_bits - soft_bits.detach()
            bits_parts.append(bits.reshape(batch_size, -1))
            logits_parts.append(bit_logits.reshape(batch_size, -1))
        return torch.cat(bits_parts, dim=1), torch.cat(logits_parts, dim=1)

    def _diffusion_alpha_bar(self, timesteps: torch.Tensor) -> torch.Tensor:
        position = (timesteps.float() + 1.0) / float(self.diffusion_steps)
        angle = (position + 0.008) / 1.008 * (torch.pi / 2.0)
        return torch.cos(angle).square().clamp(1e-4, 0.999)

    def _forward_mirror(
        self,
        *,
        bits: torch.Tensor,
        bit_logits: torch.Tensor,
        batch_size: int,
    ) -> DABETokenizerAutoencoderOutput:
        decoded = self.code_to_chunk(bits).reshape(batch_size, self.chunk_size_tokens, self.embed_dim)
        decoded = decoded + self.decoder_pos.unsqueeze(0)
        decoded = self.decoder(decoded)
        logits = self.output_proj(decoded)
        return DABETokenizerAutoencoderOutput(logits=logits, bits=bits, bit_logits=bit_logits)

    def _forward_hierarchical_lookup(
        self,
        *,
        input_ids: torch.Tensor,
        bits: torch.Tensor,
        bit_logits: torch.Tensor,
        batch_size: int,
    ) -> DABETokenizerAutoencoderOutput:
        """Decode section-3 DABE codes with a sparse exact-token side memory."""
        base_hidden = self._decode_hierarchical_hidden(
            bits=bits,
            batch_size=batch_size,
        )
        base_logits = self.output_proj(base_hidden)
        max_lookup_k = self.max_lexical_lookup_k
        if max_lookup_k <= 0:
            return DABETokenizerAutoencoderOutput(logits=base_logits, bits=bits, bit_logits=bit_logits)

        selector_logits = None
        budget_logits = None
        budget_target = None
        choices = torch.tensor(self.lexical_lookup_k_choices, device=input_ids.device, dtype=torch.long)
        adaptive_budget = len(self.lexical_lookup_k_choices) > 1
        with torch.no_grad():
            draft_losses = F.cross_entropy(
                base_logits.reshape(-1, base_logits.shape[-1]),
                input_ids.reshape(-1),
                reduction="none",
            ).reshape_as(input_ids)
            ranked_losses, oracle_positions = torch.topk(draft_losses, k=max_lookup_k, dim=1)
            if adaptive_budget:
                hardness = ranked_losses.mean(dim=1)
                order = torch.argsort(hardness, descending=False)
                budget_target = torch.empty(batch_size, device=input_ids.device, dtype=torch.long)
                buckets = torch.arange(batch_size, device=input_ids.device) * len(self.lexical_lookup_k_choices)
                buckets = torch.div(buckets, max(1, batch_size), rounding_mode="floor")
                budget_target[order] = buckets.clamp(max=len(self.lexical_lookup_k_choices) - 1)
                target_budget_k = choices[budget_target]
            else:
                target_budget_k = torch.full((batch_size,), max_lookup_k, device=input_ids.device, dtype=torch.long)
            target_active_mask = torch.arange(max_lookup_k, device=input_ids.device).unsqueeze(0) < target_budget_k.unsqueeze(1)
            selector_target_mask = torch.zeros_like(input_ids, dtype=base_logits.dtype)
            selector_target_mask.scatter_(
                dim=1,
                index=oracle_positions,
                src=target_active_mask.to(dtype=base_logits.dtype),
            )

        selector_features = self.token_embed(input_ids) + self.encoder_pos.unsqueeze(0)
        if adaptive_budget:
            budget_logits = self.lexical_budget_selector(selector_features.mean(dim=1))
            budget_class = budget_logits.argmax(dim=1)
            lookup_budget_k = choices[budget_class]
        else:
            lookup_budget_k = torch.full((batch_size,), max_lookup_k, device=input_ids.device, dtype=torch.long)
        if self.lexical_lookup_selector == "learned":
            selector_logits = self.lexical_selector(selector_features).squeeze(-1)
            lookup_positions = torch.topk(selector_logits, k=max_lookup_k, dim=1).indices
        else:
            lookup_positions = oracle_positions

        lookup_token_ids = input_ids.gather(dim=1, index=lookup_positions)
        slot_embed = self.lexical_lookup_slot_embed[: lookup_positions.shape[1]].unsqueeze(0)
        lookup_memory = (
            self.token_embed(lookup_token_ids)
            + self.decoder_pos[lookup_positions]
            + slot_embed
        )
        lookup_keep_logits = None
        lookup_keep_probs = None
        if self.lexical_lookup_slot_policy == "halting":
            lookup_keep_logits = self.lexical_slot_halting(lookup_memory).squeeze(-1)
            lookup_keep_probs = torch.sigmoid(lookup_keep_logits)
            lookup_budget_k = lookup_keep_probs.sum(dim=1)
            lookup_active_mask = lookup_keep_probs >= self.lexical_lookup_keep_threshold
            empty_rows = ~lookup_active_mask.any(dim=1)
            if bool(empty_rows.any()):
                fallback = lookup_keep_probs.argmax(dim=1, keepdim=True)
                lookup_active_mask = lookup_active_mask.scatter(1, fallback, True)
            attention_memory = lookup_memory * lookup_keep_probs.unsqueeze(-1)
            key_padding_mask = None
        else:
            lookup_active_mask = torch.arange(max_lookup_k, device=input_ids.device).unsqueeze(0) < lookup_budget_k.unsqueeze(1)
            attention_memory = lookup_memory
            key_padding_mask = ~lookup_active_mask
        context, _ = self.lexical_lookup_attention(
            query=self.lexical_lookup_norm(base_hidden),
            key=attention_memory,
            value=attention_memory,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        repaired = self.lexical_lookup_norm(base_hidden + context)
        normal_logits = self.output_proj(repaired)

        gate = torch.sigmoid(self.lexical_lookup_gate(repaired))
        query = self.lexical_lookup_query(repaired)
        key = self.lexical_lookup_key(lookup_memory)
        copy_scores = torch.einsum("btd,bkd->btk", query, key) / math.sqrt(float(self.embed_dim))
        copy_scores = copy_scores * gate * self.lexical_lookup_copy_scale
        if lookup_keep_probs is not None:
            copy_scores = copy_scores * lookup_keep_probs.unsqueeze(1)
        else:
            copy_scores = copy_scores.masked_fill(~lookup_active_mask.unsqueeze(1), 0.0)
        copy_indices = lookup_token_ids.unsqueeze(1).expand(-1, self.chunk_size_tokens, -1)
        logits = normal_logits.scatter_add(dim=2, index=copy_indices, src=copy_scores)
        return DABETokenizerAutoencoderOutput(
            logits=logits,
            bits=bits,
            bit_logits=bit_logits,
            lookup_positions=lookup_positions,
            lookup_token_ids=lookup_token_ids,
            lookup_gate=gate.squeeze(-1),
            lookup_active_mask=lookup_active_mask,
            lookup_budget_k=lookup_budget_k,
            lookup_keep_logits=lookup_keep_logits,
            lookup_keep_probs=lookup_keep_probs,
            selector_logits=selector_logits,
            selector_target_mask=selector_target_mask,
            budget_logits=budget_logits,
            budget_target=budget_target,
        )

    def _residual_router_target_from_losses(self, losses: torch.Tensor) -> torch.Tensor:
        high_q = min(0.99, max(0.01, self.residual_router_high_quantile))
        med_q = min(high_q, max(0.01, self.residual_router_medium_quantile))
        high_threshold = torch.quantile(losses.float(), high_q, dim=1, keepdim=True)
        med_threshold = torch.quantile(losses.float(), med_q, dim=1, keepdim=True)
        target = torch.zeros_like(losses, dtype=torch.long)
        target = torch.where(losses >= med_threshold, torch.ones_like(target), target)
        target = torch.where(losses >= high_threshold, torch.full_like(target, 2), target)
        return target

    def _variable_window_target_from_block_losses(self, block_losses: torch.Tensor) -> torch.Tensor:
        """Map block residual losses to fine/medium/full window targets for section 4."""
        fine_q = min(0.99, max(0.01, self.variable_window_fine_quantile))
        med_q = min(fine_q, max(0.01, self.variable_window_medium_quantile))
        fine_threshold = torch.quantile(block_losses.float(), fine_q, dim=1, keepdim=True)
        med_threshold = torch.quantile(block_losses.float(), med_q, dim=1, keepdim=True)
        target = torch.full_like(block_losses, 2, dtype=torch.long)
        target = torch.where(block_losses >= med_threshold, torch.ones_like(target), target)
        target = torch.where(block_losses >= fine_threshold, torch.zeros_like(target), target)
        return target

    def _variable_window_bits_per_chunk(self, window_probs: torch.Tensor) -> torch.Tensor:
        """Expected code bits when fine=block, medium=two-block, full=whole-chunk windows."""
        bits_per_block = float(self.hierarchical_code_bits_per_block)
        bits_by_level = torch.tensor(
            [
                bits_per_block,
                bits_per_block / 2.0,
                bits_per_block / float(self.hierarchical_num_blocks),
            ],
            device=window_probs.device,
            dtype=window_probs.dtype,
        )
        return (window_probs * bits_by_level.view(1, 1, -1)).sum(dim=-1).sum(dim=1)

    def _variable_window_bits_by_level(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        bits_per_block = float(self.hierarchical_code_bits_per_block)
        return torch.tensor(
            [
                bits_per_block,
                bits_per_block / 2.0,
                bits_per_block / float(self.hierarchical_num_blocks),
            ],
            device=device,
            dtype=dtype,
        )

    def _variable_window_selection_probs(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return decode-routing probabilities and soft probabilities for section-4 router analysis."""
        soft_probs = torch.softmax(logits / float(self.variable_window_temperature), dim=-1)
        if self.variable_window_mixing_mode == "soft":
            return soft_probs, soft_probs

        hard_indices = soft_probs.argmax(dim=-1)
        hard_probs = F.one_hot(hard_indices, num_classes=self.variable_window_levels).to(dtype=soft_probs.dtype)
        if self.variable_window_mixing_mode == "hard":
            return hard_probs, soft_probs

        straight_through_probs = hard_probs + soft_probs - soft_probs.detach()
        return straight_through_probs, soft_probs

    def _refine_nonoverlap_windows(
        self,
        hidden: torch.Tensor,
        *,
        window_tokens: int,
        level_idx: int,
    ) -> torch.Tensor:
        batch_size = int(hidden.shape[0])
        num_windows = self.chunk_size_tokens // int(window_tokens)
        windows = hidden.reshape(batch_size, num_windows, int(window_tokens), self.embed_dim)
        windows = windows + self.variable_window_level_embed[level_idx].view(1, 1, 1, self.embed_dim)
        refined = self.variable_window_refiner(
            windows.reshape(batch_size * num_windows, int(window_tokens), self.embed_dim)
        )
        return refined.reshape(batch_size, self.chunk_size_tokens, self.embed_dim)

    def _variable_window_candidate_hiddens(
        self,
        *,
        bits: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Return fine/medium/full candidate hidden states without averaging local code bits."""
        block_bits = bits.reshape(
            batch_size,
            self.hierarchical_num_blocks,
            self.hierarchical_code_bits_per_block,
        )

        block_states = self.hierarchical_code_to_block(
            block_bits.reshape(batch_size * self.hierarchical_num_blocks, -1)
        ).reshape(
            batch_size,
            self.hierarchical_num_blocks,
            self.hierarchical_block_tokens,
            self.embed_dim,
        )
        block_pos = self.decoder_pos.reshape(
            self.hierarchical_num_blocks,
            self.hierarchical_block_tokens,
            self.embed_dim,
        )
        block_states = block_states + block_pos.unsqueeze(0)
        decoded_blocks = self.hierarchical_block_decoder(
            block_states.reshape(
                batch_size * self.hierarchical_num_blocks,
                self.hierarchical_block_tokens,
                self.embed_dim,
            )
        )
        fine_hidden = decoded_blocks.reshape(batch_size, self.chunk_size_tokens, self.embed_dim)
        fine_hidden = self._refine_nonoverlap_windows(
            fine_hidden,
            window_tokens=self.hierarchical_block_tokens,
            level_idx=0,
        )
        medium_hidden = self._refine_nonoverlap_windows(
            fine_hidden,
            window_tokens=self.hierarchical_block_tokens * 2,
            level_idx=1,
        )
        full_hidden = self._refine_nonoverlap_windows(
            fine_hidden,
            window_tokens=self.chunk_size_tokens,
            level_idx=2,
        )
        return torch.stack([fine_hidden, medium_hidden, full_hidden], dim=2)

    def _decode_variable_window_hidden(
        self,
        *,
        bits: torch.Tensor,
        batch_size: int,
        window_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode hierarchical bits with differentiable fine/medium/full window mixing."""
        candidate_hiddens = self._variable_window_candidate_hiddens(bits=bits, batch_size=batch_size)
        token_window_probs = window_probs.repeat_interleave(self.hierarchical_block_tokens, dim=1)
        mixed_hidden = (candidate_hiddens * token_window_probs.unsqueeze(-1)).sum(dim=2)
        return self.variable_window_norm(mixed_hidden), candidate_hiddens

    @torch.no_grad()
    def _variable_window_action_targets(
        self,
        *,
        candidate_hiddens: torch.Tensor,
        provisional_logits: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Choose fine/medium/full actions by measured block Hamming deviation."""
        batch_size = int(input_ids.shape[0])
        block_targets = input_ids.reshape(
            batch_size,
            self.hierarchical_num_blocks,
            self.hierarchical_block_tokens,
        )
        base_pred = provisional_logits.argmax(dim=-1).reshape_as(block_targets)
        base_deviation = (base_pred != block_targets).float().sum(dim=2)
        action_deviations: list[torch.Tensor] = []
        for action_idx in range(self.variable_window_levels):
            logits = self.output_proj(candidate_hiddens[:, :, action_idx, :])
            pred = logits.argmax(dim=-1).reshape_as(block_targets)
            action_deviations.append((pred != block_targets).float().sum(dim=2))
        action_deviation = torch.stack(action_deviations, dim=-1)
        if self.variable_window_oracle_cost_lambda > 0.0:
            bits_by_level = self._variable_window_bits_by_level(
                device=action_deviation.device,
                dtype=action_deviation.dtype,
            )
            normalized_cost = bits_by_level / bits_by_level.max().clamp_min(1.0)
            action_score = action_deviation + float(self.variable_window_oracle_cost_lambda) * normalized_cost.view(1, 1, -1)
        else:
            action_score = action_deviation
        target = action_score.argmin(dim=-1)
        oracle_deviation = action_deviation.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        return target, action_deviation, base_deviation, oracle_deviation

    def _forward_gist_residual_lookup(
        self,
        *,
        input_ids: torch.Tensor,
        bits: torch.Tensor,
        bit_logits: torch.Tensor,
        batch_size: int,
    ) -> DABETokenizerAutoencoderOutput:
        """Decode a coarse gist, then spend fine attention on residual high-information zones."""
        gist_hidden = self._decode_hierarchical_hidden(bits=bits, batch_size=batch_size)
        gist_logits = self.output_proj(gist_hidden)
        max_lookup_k = self.max_lexical_lookup_k
        if max_lookup_k <= 0:
            return DABETokenizerAutoencoderOutput(
                logits=gist_logits,
                bits=bits,
                bit_logits=bit_logits,
                gist_logits=gist_logits,
            )

        with torch.no_grad():
            residual_losses = F.cross_entropy(
                gist_logits.reshape(-1, gist_logits.shape[-1]),
                input_ids.reshape(-1),
                reduction="none",
            ).reshape_as(input_ids)
            residual_router_target = self._residual_router_target_from_losses(residual_losses)

        residual_router_logits = self.residual_router(gist_hidden)
        residual_router_probs = torch.softmax(residual_router_logits, dim=-1)
        residual_info_scores = (
            residual_router_probs[..., 2]
            + self.residual_router_med_score * residual_router_probs[..., 1]
        )
        lookup_positions = torch.topk(residual_info_scores, k=max_lookup_k, dim=1).indices
        lookup_token_ids = input_ids.gather(dim=1, index=lookup_positions)
        slot_embed = self.lexical_lookup_slot_embed[:max_lookup_k].unsqueeze(0)
        gathered_router_probs = residual_router_probs.gather(
            dim=1,
            index=lookup_positions.unsqueeze(-1).expand(-1, -1, self.residual_router_levels),
        )
        lookup_level_embed = torch.matmul(gathered_router_probs, self.residual_level_embed)
        lookup_memory = (
            self.token_embed(lookup_token_ids)
            + self.decoder_pos[lookup_positions]
            + slot_embed
            + lookup_level_embed
        )

        lookup_keep_logits = self.lexical_slot_halting(lookup_memory).squeeze(-1)
        lookup_keep_probs = torch.sigmoid(lookup_keep_logits)
        lookup_budget_k = lookup_keep_probs.sum(dim=1)
        lookup_active_mask = lookup_keep_probs >= self.lexical_lookup_keep_threshold
        empty_rows = ~lookup_active_mask.any(dim=1)
        if bool(empty_rows.any()):
            fallback = lookup_keep_probs.argmax(dim=1, keepdim=True)
            lookup_active_mask = lookup_active_mask.scatter(1, fallback, True)

        dense_level_context = torch.matmul(residual_router_probs, self.residual_level_embed)
        attention_memory = lookup_memory * lookup_keep_probs.unsqueeze(-1)
        context, _ = self.lexical_lookup_attention(
            query=self.lexical_lookup_norm(gist_hidden + dense_level_context),
            key=attention_memory,
            value=attention_memory,
            need_weights=False,
        )
        repaired = self.lexical_lookup_norm(gist_hidden + dense_level_context + context)
        normal_logits = self.output_proj(repaired)

        gate = torch.sigmoid(self.lexical_lookup_gate(repaired))
        query = self.lexical_lookup_query(repaired)
        key = self.lexical_lookup_key(lookup_memory)
        copy_scores = torch.einsum("btd,bkd->btk", query, key) / math.sqrt(float(self.embed_dim))
        copy_scores = copy_scores * gate * self.lexical_lookup_copy_scale * lookup_keep_probs.unsqueeze(1)
        copy_indices = lookup_token_ids.unsqueeze(1).expand(-1, self.chunk_size_tokens, -1)
        logits = normal_logits.scatter_add(dim=2, index=copy_indices, src=copy_scores)
        return DABETokenizerAutoencoderOutput(
            logits=logits,
            bits=bits,
            bit_logits=bit_logits,
            gist_logits=gist_logits,
            lookup_positions=lookup_positions,
            lookup_token_ids=lookup_token_ids,
            lookup_gate=gate.squeeze(-1),
            lookup_active_mask=lookup_active_mask,
            lookup_budget_k=lookup_budget_k,
            lookup_keep_logits=lookup_keep_logits,
            lookup_keep_probs=lookup_keep_probs,
            selector_logits=None,
            selector_target_mask=(residual_router_target == 2).to(dtype=gist_logits.dtype),
            residual_router_logits=residual_router_logits,
            residual_router_target=residual_router_target,
            residual_info_scores=residual_info_scores,
        )

    def _forward_gist_residual_variable_windows(
        self,
        *,
        input_ids: torch.Tensor,
        bits: torch.Tensor,
        bit_logits: torch.Tensor,
        batch_size: int,
    ) -> DABETokenizerAutoencoderOutput:
        """Decode with learned 16/32/64-token windows before residual fine lookup."""
        provisional_hidden = self._decode_hierarchical_hidden(bits=bits, batch_size=batch_size)
        provisional_logits = self.output_proj(provisional_hidden)
        block_features = provisional_hidden.reshape(
            batch_size,
            self.hierarchical_num_blocks,
            self.hierarchical_block_tokens,
            self.embed_dim,
        ).mean(dim=2)
        variable_window_logits = self.variable_window_router(block_features)
        variable_window_probs, variable_window_soft_probs = self._variable_window_selection_probs(variable_window_logits)
        variable_window_bits_per_chunk = self._variable_window_bits_per_chunk(variable_window_probs)

        gist_hidden, candidate_hiddens = self._decode_variable_window_hidden(
            bits=bits,
            batch_size=batch_size,
            window_probs=variable_window_probs,
        )
        variable_window_action_deviation = None
        variable_window_base_deviation = None
        variable_window_oracle_deviation = None
        with torch.no_grad():
            if self.variable_window_target_mode == "action_value":
                (
                    variable_window_target,
                    variable_window_action_deviation,
                    variable_window_base_deviation,
                    variable_window_oracle_deviation,
                ) = self._variable_window_action_targets(
                    candidate_hiddens=candidate_hiddens,
                    provisional_logits=provisional_logits,
                    input_ids=input_ids,
                )
            else:
                provisional_losses = F.cross_entropy(
                    provisional_logits.reshape(-1, provisional_logits.shape[-1]),
                    input_ids.reshape(-1),
                    reduction="none",
                ).reshape_as(input_ids)
                block_losses = provisional_losses.reshape(
                    batch_size,
                    self.hierarchical_num_blocks,
                    self.hierarchical_block_tokens,
                ).mean(dim=2)
                variable_window_target = self._variable_window_target_from_block_losses(block_losses)
        gist_logits = self.output_proj(gist_hidden)
        max_lookup_k = self.max_lexical_lookup_k
        if max_lookup_k <= 0:
            return DABETokenizerAutoencoderOutput(
                logits=gist_logits,
                bits=bits,
                bit_logits=bit_logits,
                gist_logits=gist_logits,
                variable_window_logits=variable_window_logits,
                variable_window_target=variable_window_target,
                variable_window_probs=variable_window_probs,
                variable_window_soft_probs=variable_window_soft_probs,
                variable_window_bits_per_chunk=variable_window_bits_per_chunk,
                variable_window_action_deviation=variable_window_action_deviation,
                variable_window_base_deviation=variable_window_base_deviation,
                variable_window_oracle_deviation=variable_window_oracle_deviation,
            )

        with torch.no_grad():
            residual_losses = F.cross_entropy(
                gist_logits.reshape(-1, gist_logits.shape[-1]),
                input_ids.reshape(-1),
                reduction="none",
            ).reshape_as(input_ids)
            residual_router_target = self._residual_router_target_from_losses(residual_losses)

        residual_router_logits = self.residual_router(gist_hidden)
        residual_router_probs = torch.softmax(residual_router_logits, dim=-1)
        residual_info_scores = (
            residual_router_probs[..., 2]
            + self.residual_router_med_score * residual_router_probs[..., 1]
        )
        lookup_positions = torch.topk(residual_info_scores, k=max_lookup_k, dim=1).indices
        lookup_token_ids = input_ids.gather(dim=1, index=lookup_positions)
        slot_embed = self.lexical_lookup_slot_embed[:max_lookup_k].unsqueeze(0)
        gathered_router_probs = residual_router_probs.gather(
            dim=1,
            index=lookup_positions.unsqueeze(-1).expand(-1, -1, self.residual_router_levels),
        )
        lookup_level_embed = torch.matmul(gathered_router_probs, self.residual_level_embed)
        lookup_memory = (
            self.token_embed(lookup_token_ids)
            + self.decoder_pos[lookup_positions]
            + slot_embed
            + lookup_level_embed
        )

        lookup_keep_logits = self.lexical_slot_halting(lookup_memory).squeeze(-1)
        lookup_keep_probs = torch.sigmoid(lookup_keep_logits)
        lookup_budget_k = lookup_keep_probs.sum(dim=1)
        lookup_active_mask = lookup_keep_probs >= self.lexical_lookup_keep_threshold
        empty_rows = ~lookup_active_mask.any(dim=1)
        if bool(empty_rows.any()):
            fallback = lookup_keep_probs.argmax(dim=1, keepdim=True)
            lookup_active_mask = lookup_active_mask.scatter(1, fallback, True)

        dense_level_context = torch.matmul(residual_router_probs, self.residual_level_embed)
        attention_memory = lookup_memory * lookup_keep_probs.unsqueeze(-1)
        context, _ = self.lexical_lookup_attention(
            query=self.lexical_lookup_norm(gist_hidden + dense_level_context),
            key=attention_memory,
            value=attention_memory,
            need_weights=False,
        )
        repaired = self.lexical_lookup_norm(gist_hidden + dense_level_context + context)
        normal_logits = self.output_proj(repaired)

        gate = torch.sigmoid(self.lexical_lookup_gate(repaired))
        query = self.lexical_lookup_query(repaired)
        key = self.lexical_lookup_key(lookup_memory)
        copy_scores = torch.einsum("btd,bkd->btk", query, key) / math.sqrt(float(self.embed_dim))
        copy_scores = copy_scores * gate * self.lexical_lookup_copy_scale * lookup_keep_probs.unsqueeze(1)
        copy_indices = lookup_token_ids.unsqueeze(1).expand(-1, self.chunk_size_tokens, -1)
        logits = normal_logits.scatter_add(dim=2, index=copy_indices, src=copy_scores)
        return DABETokenizerAutoencoderOutput(
            logits=logits,
            bits=bits,
            bit_logits=bit_logits,
            gist_logits=gist_logits,
            lookup_positions=lookup_positions,
            lookup_token_ids=lookup_token_ids,
            lookup_gate=gate.squeeze(-1),
            lookup_active_mask=lookup_active_mask,
            lookup_budget_k=lookup_budget_k,
            lookup_keep_logits=lookup_keep_logits,
            lookup_keep_probs=lookup_keep_probs,
            selector_logits=None,
            selector_target_mask=(residual_router_target == 2).to(dtype=gist_logits.dtype),
            residual_router_logits=residual_router_logits,
            residual_router_target=residual_router_target,
            residual_info_scores=residual_info_scores,
            variable_window_logits=variable_window_logits,
            variable_window_target=variable_window_target,
            variable_window_probs=variable_window_probs,
            variable_window_soft_probs=variable_window_soft_probs,
            variable_window_bits_per_chunk=variable_window_bits_per_chunk,
            variable_window_action_deviation=variable_window_action_deviation,
            variable_window_base_deviation=variable_window_base_deviation,
            variable_window_oracle_deviation=variable_window_oracle_deviation,
        )

    def _forward_diffusion(
        self,
        *,
        input_ids: torch.Tensor,
        bits: torch.Tensor,
        bit_logits: torch.Tensor,
    ) -> DABETokenizerAutoencoderOutput:
        batch_size = int(input_ids.shape[0])
        target_embeddings = self.token_embed(input_ids)
        timesteps = torch.randint(0, self.diffusion_steps, (batch_size,), device=input_ids.device)
        noise = torch.randn_like(target_embeddings)
        alpha_bar = self._diffusion_alpha_bar(timesteps).view(batch_size, 1, 1).to(target_embeddings.dtype)
        noisy_embeddings = alpha_bar.sqrt() * target_embeddings + (1.0 - alpha_bar).sqrt() * noise

        denoised = self.diffusion_denoise_embeddings(noisy_embeddings=noisy_embeddings, bits=bits, timesteps=timesteps)
        logits = self.output_proj(denoised)
        diffusion_embed_mse = F.mse_loss(denoised, target_embeddings.detach())
        return DABETokenizerAutoencoderOutput(
            logits=logits,
            bits=bits,
            bit_logits=bit_logits,
            diffusion_embed_mse=diffusion_embed_mse,
        )

    def _forward_code_transformer(
        self,
        *,
        bits: torch.Tensor,
        bit_logits: torch.Tensor,
        batch_size: int,
    ) -> DABETokenizerAutoencoderOutput:
        code_states = self.code_to_chunk(bits).reshape(batch_size, self.chunk_size_tokens, self.embed_dim)
        code_states = code_states + self.decoder_pos.unsqueeze(0)
        decoded = self.code_transformer_norm(self.code_transformer_decoder(code_states))
        logits = self.output_proj(decoded)
        return DABETokenizerAutoencoderOutput(logits=logits, bits=bits, bit_logits=bit_logits)

    def _forward_hierarchical_local(
        self,
        *,
        bits: torch.Tensor,
        bit_logits: torch.Tensor,
        batch_size: int,
        local_refine: bool = False,
    ) -> DABETokenizerAutoencoderOutput:
        """Decode local section-3 DABE codes, optionally with code-only causal block refinement."""
        decoded = self._decode_hierarchical_hidden(
            bits=bits,
            batch_size=batch_size,
            local_refine=local_refine,
        )
        logits = self.output_proj(decoded)
        return DABETokenizerAutoencoderOutput(logits=logits, bits=bits, bit_logits=bit_logits)

    def _decode_hierarchical_hidden(
        self,
        *,
        bits: torch.Tensor,
        batch_size: int,
        local_refine: bool = False,
    ) -> torch.Tensor:
        block_bits = bits.reshape(
            batch_size,
            self.hierarchical_num_blocks,
            self.hierarchical_code_bits_per_block,
        )
        block_states = self.hierarchical_code_to_block(block_bits).reshape(
            batch_size,
            self.hierarchical_num_blocks,
            self.hierarchical_block_tokens,
            self.embed_dim,
        )
        block_pos = self.decoder_pos.reshape(
            self.hierarchical_num_blocks,
            self.hierarchical_block_tokens,
            self.embed_dim,
        )
        block_states = block_states + block_pos.unsqueeze(0)
        decoded_blocks = self.hierarchical_block_decoder(
            block_states.reshape(batch_size * self.hierarchical_num_blocks, self.hierarchical_block_tokens, self.embed_dim)
        )
        if local_refine and self.hierarchical_local_refine_layers > 0:
            local_mask = self.hierarchical_local_causal_mask if self.hierarchical_local_refine_causal else None
            refined_blocks = self.hierarchical_local_refiner(decoded_blocks, mask=local_mask)
            decoded_blocks = self.hierarchical_local_refine_norm(decoded_blocks + refined_blocks)
        decoded = decoded_blocks.reshape(batch_size, self.chunk_size_tokens, self.embed_dim)
        return self.hierarchical_norm(self.hierarchical_refiner(decoded))

    def _forward_sliding_progressive(
        self,
        *,
        bits: torch.Tensor,
        bit_logits: torch.Tensor,
        batch_size: int,
    ) -> DABETokenizerAutoencoderOutput:
        accum = torch.zeros(
            batch_size,
            self.chunk_size_tokens,
            self.embed_dim,
            device=bits.device,
            dtype=bits.dtype,
        )
        counts = torch.zeros(self.chunk_size_tokens, device=bits.device, dtype=bits.dtype)
        cursor = 0
        for level_idx, (name, window_tokens, bits_per_window, starts) in enumerate(self.sliding_specs):
            width = len(starts) * bits_per_window
            level_bits = bits[:, cursor : cursor + width].reshape(batch_size, len(starts), bits_per_window)
            cursor += width
            window_states = self.sliding_code_to_window[name](
                level_bits.reshape(batch_size * len(starts), bits_per_window)
            ).reshape(batch_size * len(starts), window_tokens, self.embed_dim)
            window_pos = self.decoder_pos[:window_tokens].unsqueeze(0)
            window_states = window_states + window_pos + self.sliding_level_embed[level_idx].view(1, 1, self.embed_dim)
            window_states = self.sliding_window_decoder(window_states).reshape(
                batch_size,
                len(starts),
                window_tokens,
                self.embed_dim,
            )
            for window_idx, start in enumerate(starts):
                accum[:, start : start + window_tokens, :] += window_states[:, window_idx, :, :]
                counts[start : start + window_tokens] += 1.0

        decoded = accum / counts.clamp_min(1.0).view(1, self.chunk_size_tokens, 1)
        decoded = self.sliding_norm(self.sliding_refiner(decoded))
        logits = self.output_proj(decoded)
        return DABETokenizerAutoencoderOutput(logits=logits, bits=bits, bit_logits=bit_logits)

    def diffusion_denoise_embeddings(
        self,
        *,
        noisy_embeddings: torch.Tensor,
        bits: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(noisy_embeddings.shape[0])
        code_context = self.code_to_chunk(bits).reshape(batch_size, self.chunk_size_tokens, self.embed_dim)
        time_context = self.diffusion_time_embed(timesteps).unsqueeze(1)
        denoiser_input = noisy_embeddings + code_context + self.decoder_pos.unsqueeze(0) + time_context
        return self.diffusion_norm(self.diffusion_denoiser(denoiser_input))

    @torch.no_grad()
    def reverse_diffusion_decode(
        self,
        *,
        bits: torch.Tensor,
        sample_steps: int | None = None,
        stochastic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode from Gaussian noise using DDIM-style x0 prediction conditioned only on code bits."""
        steps = max(1, int(sample_steps or self.diffusion_steps))
        device = bits.device
        batch_size = int(bits.shape[0])
        noisy = torch.randn(batch_size, self.chunk_size_tokens, self.embed_dim, device=device, dtype=bits.dtype)
        timesteps = torch.linspace(
            self.diffusion_steps - 1,
            0,
            steps=min(steps, self.diffusion_steps),
            device=device,
        ).round().long()
        timesteps = torch.unique_consecutive(timesteps)
        if int(timesteps[-1].item()) != 0:
            timesteps = torch.cat([timesteps, torch.zeros(1, device=device, dtype=torch.long)])

        pred_clean = noisy
        for idx, timestep in enumerate(timesteps):
            t = timestep.expand(batch_size)
            pred_clean = self.diffusion_denoise_embeddings(noisy_embeddings=noisy, bits=bits, timesteps=t)
            if idx == len(timesteps) - 1:
                break

            next_t = timesteps[idx + 1].expand(batch_size)
            alpha_t = self._diffusion_alpha_bar(t).view(batch_size, 1, 1).to(noisy.dtype)
            alpha_next = self._diffusion_alpha_bar(next_t).view(batch_size, 1, 1).to(noisy.dtype)
            eps = (noisy - alpha_t.sqrt() * pred_clean) / (1.0 - alpha_t).sqrt().clamp_min(1e-6)
            if stochastic:
                eps = torch.randn_like(eps)
            noisy = alpha_next.sqrt() * pred_clean + (1.0 - alpha_next).sqrt() * eps

        logits = self.output_proj(pred_clean)
        return logits, pred_clean

    def forward(self, input_ids: torch.Tensor) -> DABETokenizerAutoencoderOutput:
        if input_ids.ndim != 2 or input_ids.shape[1] != self.chunk_size_tokens:
            raise ValueError(
                "input_ids must have shape (batch, chunk_size_tokens); "
                f"got {tuple(input_ids.shape)}"
            )
        bits, bit_logits = self.encode_bits(input_ids)
        if self.decoder_mode == "diffusion":
            return self._forward_diffusion(input_ids=input_ids, bits=bits, bit_logits=bit_logits)
        if self.decoder_mode == "code_transformer":
            return self._forward_code_transformer(
                bits=bits,
                bit_logits=bit_logits,
                batch_size=int(input_ids.shape[0]),
            )
        if self.decoder_mode == "hierarchical_local":
            return self._forward_hierarchical_local(
                bits=bits,
                bit_logits=bit_logits,
                batch_size=int(input_ids.shape[0]),
            )
        if self.decoder_mode == "hierarchical_local_refine":
            return self._forward_hierarchical_local(
                bits=bits,
                bit_logits=bit_logits,
                batch_size=int(input_ids.shape[0]),
                local_refine=True,
            )
        if self.decoder_mode == "hierarchical_lookup":
            return self._forward_hierarchical_lookup(
                input_ids=input_ids,
                bits=bits,
                bit_logits=bit_logits,
                batch_size=int(input_ids.shape[0]),
            )
        if self.decoder_mode == "gist_residual_lookup":
            return self._forward_gist_residual_lookup(
                input_ids=input_ids,
                bits=bits,
                bit_logits=bit_logits,
                batch_size=int(input_ids.shape[0]),
            )
        if self.decoder_mode == "gist_residual_variable_windows":
            return self._forward_gist_residual_variable_windows(
                input_ids=input_ids,
                bits=bits,
                bit_logits=bit_logits,
                batch_size=int(input_ids.shape[0]),
            )
        if self.decoder_mode == "sliding_progressive":
            return self._forward_sliding_progressive(
                bits=bits,
                bit_logits=bit_logits,
                batch_size=int(input_ids.shape[0]),
            )
        return self._forward_mirror(bits=bits, bit_logits=bit_logits, batch_size=int(input_ids.shape[0]))


class DABETokenizerAutoencoderModule(L.LightningModule):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.model_cfg = dict(config.get("model", {}))
        self.training_cfg = dict(config.get("training", {}))
        self.model = DABEChunkTokenizerAutoencoder(self.model_cfg)
        self.bit_balance_weight = float(self.training_cfg.get("bit_balance_weight", 0.01))
        self.diffusion_loss_weight = float(self.training_cfg.get("diffusion_loss_weight", 0.1))
        self.selector_loss_weight = float(self.training_cfg.get("selector_loss_weight", 0.1))
        self.budget_loss_weight = float(self.training_cfg.get("budget_loss_weight", 0.1))
        self.gist_loss_weight = float(self.training_cfg.get("gist_loss_weight", 0.0))
        self.residual_router_loss_weight = float(self.training_cfg.get("residual_router_loss_weight", 0.0))
        self.variable_window_loss_weight = float(self.training_cfg.get("variable_window_loss_weight", 0.0))
        self.variable_window_cost_weight = float(self.training_cfg.get("variable_window_cost_weight", 0.0))
        self.variable_window_nonimprove_weight = float(self.training_cfg.get("variable_window_nonimprove_weight", 0.0))
        self.variable_window_nonimprove_margin = float(self.training_cfg.get("variable_window_nonimprove_margin", 0.5))
        self.lookup_slot_cost_weight = float(self.training_cfg.get("lookup_slot_cost_weight", 0.0))
        self.lookup_slot_target_weight = float(self.training_cfg.get("lookup_slot_target_weight", 0.0))
        self.lookup_slot_target_k = self.training_cfg.get("lookup_slot_target_k")
        self.block_loss_weights = [float(weight) for weight in self.training_cfg.get("block_loss_weights", [])]
        self.nan_batches = 0

    def _step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        input_ids = batch["input_ids"].to(self.device)
        output = self.model(input_ids)
        token_losses = F.cross_entropy(
            output.logits.reshape(-1, output.logits.shape[-1]),
            input_ids.reshape(-1),
            reduction="none",
        ).reshape_as(input_ids)
        token_loss = token_losses.mean()
        weighted_token_loss = token_loss
        block_metrics: list[dict[str, torch.Tensor]] = []
        block_size = int(getattr(self.model, "hierarchical_block_tokens", input_ids.shape[1]))
        if self.block_loss_weights and input_ids.shape[1] % block_size == 0:
            num_blocks = input_ids.shape[1] // block_size
            if len(self.block_loss_weights) != num_blocks:
                raise ValueError(
                    "block_loss_weights length must match the number of blocks; "
                    f"got {len(self.block_loss_weights)} vs {num_blocks}."
                )
            block_weights = torch.tensor(self.block_loss_weights, device=self.device, dtype=token_loss.dtype)
            block_losses = token_losses.reshape(input_ids.shape[0], num_blocks, block_size).mean(dim=(0, 2))
            weighted_token_loss = torch.sum(block_losses * block_weights) / block_weights.sum().clamp_min(1e-6)
        bit_mean = output.bits.mean(dim=0)
        bit_balance_loss = torch.mean((bit_mean - 0.5) ** 2)
        diffusion_embed_mse = output.diffusion_embed_mse
        diffusion_loss = (
            diffusion_embed_mse
            if diffusion_embed_mse is not None
            else torch.zeros((), device=self.device, dtype=token_loss.dtype)
        )
        selector_loss = torch.zeros((), device=self.device, dtype=token_loss.dtype)
        if output.selector_logits is not None and output.selector_target_mask is not None:
            positives = output.selector_target_mask.sum().clamp_min(1.0)
            negatives = (output.selector_target_mask.numel() - output.selector_target_mask.sum()).clamp_min(1.0)
            selector_loss = F.binary_cross_entropy_with_logits(
                output.selector_logits,
                output.selector_target_mask.to(output.selector_logits.dtype),
                pos_weight=(negatives / positives).to(output.selector_logits.dtype),
            )
        budget_loss = torch.zeros((), device=self.device, dtype=token_loss.dtype)
        if output.budget_logits is not None and output.budget_target is not None:
            budget_loss = F.cross_entropy(output.budget_logits, output.budget_target)
        gist_loss = torch.zeros((), device=self.device, dtype=token_loss.dtype)
        if output.gist_logits is not None:
            gist_loss = F.cross_entropy(
                output.gist_logits.reshape(-1, output.gist_logits.shape[-1]),
                input_ids.reshape(-1),
            )
        residual_router_loss = torch.zeros((), device=self.device, dtype=token_loss.dtype)
        if output.residual_router_logits is not None and output.residual_router_target is not None:
            residual_router_loss = F.cross_entropy(
                output.residual_router_logits.reshape(-1, output.residual_router_logits.shape[-1]),
                output.residual_router_target.reshape(-1),
            )
        variable_window_loss = torch.zeros((), device=self.device, dtype=token_loss.dtype)
        variable_window_cost_loss = torch.zeros((), device=self.device, dtype=token_loss.dtype)
        variable_window_nonimprove_loss = torch.zeros((), device=self.device, dtype=token_loss.dtype)
        if output.variable_window_logits is not None and output.variable_window_target is not None:
            variable_window_loss = F.cross_entropy(
                output.variable_window_logits.reshape(-1, output.variable_window_logits.shape[-1]),
                output.variable_window_target.reshape(-1),
            )
        if output.variable_window_bits_per_chunk is not None:
            variable_window_cost_loss = output.variable_window_bits_per_chunk.float().mean() / float(self.model.code_bits)
        if (
            output.variable_window_probs is not None
            and output.variable_window_action_deviation is not None
            and output.variable_window_base_deviation is not None
        ):
            action_penalty = torch.relu(
                output.variable_window_action_deviation.to(output.variable_window_probs.dtype)
                - output.variable_window_base_deviation.unsqueeze(-1).to(output.variable_window_probs.dtype)
                + float(self.variable_window_nonimprove_margin)
            )
            variable_window_nonimprove_loss = (
                output.variable_window_probs * action_penalty.detach()
            ).sum(dim=-1).mean() / float(self.model.hierarchical_block_tokens)
        lookup_slot_cost_loss = torch.zeros((), device=self.device, dtype=token_loss.dtype)
        lookup_slot_target_loss = torch.zeros((), device=self.device, dtype=token_loss.dtype)
        if output.lookup_keep_probs is not None:
            lookup_slot_cost_loss = output.lookup_keep_probs.mean()
            if self.lookup_slot_target_k is not None:
                target_k = torch.tensor(float(self.lookup_slot_target_k), device=self.device, dtype=token_loss.dtype)
                keep_k = output.lookup_keep_probs.sum(dim=1).mean()
                max_k = torch.tensor(float(output.lookup_keep_probs.shape[1]), device=self.device, dtype=token_loss.dtype)
                lookup_slot_target_loss = ((keep_k - target_k) / max_k.clamp_min(1.0)).square()
        loss = (
            weighted_token_loss
            + self.bit_balance_weight * bit_balance_loss
            + self.diffusion_loss_weight * diffusion_loss
            + self.selector_loss_weight * selector_loss
            + self.budget_loss_weight * budget_loss
            + self.gist_loss_weight * gist_loss
            + self.residual_router_loss_weight * residual_router_loss
            + self.variable_window_loss_weight * variable_window_loss
            + self.variable_window_cost_weight * variable_window_cost_loss
            + self.variable_window_nonimprove_weight * variable_window_nonimprove_loss
            + self.lookup_slot_cost_weight * lookup_slot_cost_loss
            + self.lookup_slot_target_weight * lookup_slot_target_loss
        )
        if not bool(torch.isfinite(loss)):
            self.nan_batches += 1
            raise RuntimeError("Non-finite loss in DABE tokenizer autoencoder training.")

        with torch.no_grad():
            pred_ids = output.logits.argmax(dim=-1)
            matches = pred_ids == input_ids
            token_acc = matches.float().mean()
            topk_limit = min(10, int(output.logits.shape[-1]))
            topk_ids = torch.topk(output.logits, k=topk_limit, dim=-1).indices
            top5_acc = (topk_ids[..., : min(5, topk_limit)] == input_ids.unsqueeze(-1)).any(dim=-1).float().mean()
            top10_acc = (topk_ids == input_ids.unsqueeze(-1)).any(dim=-1).float().mean()
            exact_chunk_acc = matches.all(dim=-1).float().mean()
            chunk_deviation = chunk_deviation_stats(matches)
            bit_density = output.bits.mean()
            lookup_selected_fraction = torch.zeros((), device=self.device, dtype=token_acc.dtype)
            lookup_token_acc = torch.zeros((), device=self.device, dtype=token_acc.dtype)
            non_lookup_token_acc = token_acc
            lookup_gate_mean = torch.zeros((), device=self.device, dtype=token_acc.dtype)
            selector_recall_at_k = torch.zeros((), device=self.device, dtype=token_acc.dtype)
            selector_precision_at_k = torch.zeros((), device=self.device, dtype=token_acc.dtype)
            lookup_budget_k_mean = torch.zeros((), device=self.device, dtype=token_acc.dtype)
            lookup_active_k_mean = torch.zeros((), device=self.device, dtype=token_acc.dtype)
            lookup_keep_prob_mean = torch.zeros((), device=self.device, dtype=token_acc.dtype)
            observed_effective_bits_per_token = torch.tensor(
                self.model.effective_bits_per_token,
                device=self.device,
                dtype=token_acc.dtype,
            )
            budget_acc = torch.zeros((), device=self.device, dtype=token_acc.dtype)
            residual_router_acc = torch.zeros((), device=self.device, dtype=token_acc.dtype)
            residual_high_recall = torch.zeros((), device=self.device, dtype=token_acc.dtype)
            residual_high_precision = torch.zeros((), device=self.device, dtype=token_acc.dtype)
            residual_high_fraction = torch.zeros((), device=self.device, dtype=token_acc.dtype)
            residual_medium_fraction = torch.zeros((), device=self.device, dtype=token_acc.dtype)
            residual_fine_region_fraction = torch.zeros((), device=self.device, dtype=token_acc.dtype)
            variable_window_acc = torch.zeros((), device=self.device, dtype=token_acc.dtype)
            variable_window_fine_fraction = torch.zeros((), device=self.device, dtype=token_acc.dtype)
            variable_window_medium_fraction = torch.zeros((), device=self.device, dtype=token_acc.dtype)
            variable_window_full_fraction = torch.zeros((), device=self.device, dtype=token_acc.dtype)
            variable_window_target_fine_fraction = torch.zeros((), device=self.device, dtype=token_acc.dtype)
            variable_window_target_medium_fraction = torch.zeros((), device=self.device, dtype=token_acc.dtype)
            variable_window_target_full_fraction = torch.zeros((), device=self.device, dtype=token_acc.dtype)
            variable_window_expected_tokens = torch.zeros((), device=self.device, dtype=token_acc.dtype)
            variable_window_soft_expected_tokens = torch.zeros((), device=self.device, dtype=token_acc.dtype)
            variable_window_soft_entropy = torch.zeros((), device=self.device, dtype=token_acc.dtype)
            variable_window_code_bits = torch.tensor(
                float(self.model.code_bits),
                device=self.device,
                dtype=token_acc.dtype,
            )
            variable_window_base_deviation_mean = torch.zeros((), device=self.device, dtype=token_acc.dtype)
            variable_window_oracle_deviation_mean = torch.zeros((), device=self.device, dtype=token_acc.dtype)
            variable_window_chosen_deviation_mean = torch.zeros((), device=self.device, dtype=token_acc.dtype)
            variable_window_expected_deviation_mean = torch.zeros((), device=self.device, dtype=token_acc.dtype)
            variable_window_regret_mean = torch.zeros((), device=self.device, dtype=token_acc.dtype)
            variable_window_nonimprove_rate = torch.zeros((), device=self.device, dtype=token_acc.dtype)
            variable_window_fine_deviation_mean = torch.zeros((), device=self.device, dtype=token_acc.dtype)
            variable_window_medium_deviation_mean = torch.zeros((), device=self.device, dtype=token_acc.dtype)
            variable_window_full_deviation_mean = torch.zeros((), device=self.device, dtype=token_acc.dtype)
            if output.residual_router_logits is not None and output.residual_router_target is not None:
                router_pred = output.residual_router_logits.argmax(dim=-1)
                residual_router_acc = (router_pred == output.residual_router_target).float().mean()
                target_high = output.residual_router_target == 2
                pred_high = router_pred == 2
                residual_high_recall = (
                    (pred_high & target_high).float().sum()
                    / target_high.float().sum().clamp_min(1.0)
                )
                residual_high_precision = (
                    (pred_high & target_high).float().sum()
                    / pred_high.float().sum().clamp_min(1.0)
                )
                residual_high_fraction = pred_high.float().mean()
                residual_medium_fraction = (router_pred == 1).float().mean()
                residual_fine_region_fraction = (router_pred > 0).float().mean()
            if output.variable_window_probs is not None:
                window_pred = output.variable_window_probs.argmax(dim=-1)
                variable_window_fine_fraction = (window_pred == 0).float().mean()
                variable_window_medium_fraction = (window_pred == 1).float().mean()
                variable_window_full_fraction = (window_pred == 2).float().mean()
                window_tokens = torch.tensor(
                    [
                        float(self.model.hierarchical_block_tokens),
                        float(self.model.hierarchical_block_tokens * 2),
                        float(self.model.chunk_size_tokens),
                    ],
                    device=self.device,
                    dtype=output.variable_window_probs.dtype,
                )
                variable_window_expected_tokens = (
                    output.variable_window_probs * window_tokens.view(1, 1, -1)
                ).sum(dim=-1).mean()
                if output.variable_window_soft_probs is not None:
                    soft_probs = output.variable_window_soft_probs.float()
                    variable_window_soft_expected_tokens = (
                        soft_probs * window_tokens.float().view(1, 1, -1)
                    ).sum(dim=-1).mean()
                    variable_window_soft_entropy = (
                        -(soft_probs * soft_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
                    )
                if output.variable_window_bits_per_chunk is not None:
                    variable_window_code_bits = output.variable_window_bits_per_chunk.float().mean()
                if output.variable_window_target is not None:
                    variable_window_acc = (window_pred == output.variable_window_target).float().mean()
                    variable_window_target_fine_fraction = (output.variable_window_target == 0).float().mean()
                    variable_window_target_medium_fraction = (output.variable_window_target == 1).float().mean()
                    variable_window_target_full_fraction = (output.variable_window_target == 2).float().mean()
                if output.variable_window_action_deviation is not None:
                    action_deviation = output.variable_window_action_deviation.float()
                    variable_window_fine_deviation_mean = action_deviation[..., 0].mean()
                    variable_window_medium_deviation_mean = action_deviation[..., 1].mean()
                    variable_window_full_deviation_mean = action_deviation[..., 2].mean()
                    chosen_deviation = action_deviation.gather(
                        dim=-1,
                        index=window_pred.unsqueeze(-1),
                    ).squeeze(-1)
                    expected_deviation = (
                        output.variable_window_probs.float() * action_deviation
                    ).sum(dim=-1)
                    variable_window_chosen_deviation_mean = chosen_deviation.mean()
                    variable_window_expected_deviation_mean = expected_deviation.mean()
                    if output.variable_window_oracle_deviation is not None:
                        oracle_deviation = output.variable_window_oracle_deviation.float()
                        variable_window_oracle_deviation_mean = oracle_deviation.mean()
                        variable_window_regret_mean = (chosen_deviation - oracle_deviation).mean()
                    if output.variable_window_base_deviation is not None:
                        base_deviation = output.variable_window_base_deviation.float()
                        variable_window_base_deviation_mean = base_deviation.mean()
                        variable_window_nonimprove_rate = (chosen_deviation >= base_deviation).float().mean()
            lookup_metrics_enabled = output.lookup_positions is not None
            if output.lookup_positions is not None:
                active_mask = (
                    output.lookup_active_mask
                    if output.lookup_active_mask is not None
                    else torch.ones_like(output.lookup_positions, dtype=torch.bool)
                )
                lookup_mask = torch.zeros_like(input_ids, dtype=torch.bool)
                lookup_mask.scatter_(dim=1, index=output.lookup_positions, src=active_mask)
                lookup_selected_fraction = lookup_mask.float().mean()
                lookup_active_k_mean = active_mask.float().sum(dim=1).mean()
                lookup_budget_k_mean = (
                    output.lookup_budget_k.float().mean()
                    if output.lookup_budget_k is not None
                    else lookup_active_k_mean
                )
                if output.lookup_keep_probs is not None:
                    lookup_keep_prob_mean = output.lookup_keep_probs.float().mean()
                observed_effective_bits_per_token = (
                    variable_window_code_bits + lookup_budget_k_mean * float(self.model.lookup_entry_bits)
                ) / float(self.model.chunk_size_tokens)
                if bool(lookup_mask.any()):
                    lookup_token_acc = matches[lookup_mask].float().mean()
                    lookup_gate_mean = output.lookup_gate[lookup_mask].float().mean() if output.lookup_gate is not None else lookup_gate_mean
                non_lookup_mask = ~lookup_mask
                if bool(non_lookup_mask.any()):
                    non_lookup_token_acc = matches[non_lookup_mask].float().mean()
                if output.selector_target_mask is not None:
                    target_mask = output.selector_target_mask.bool()
                    overlap = (lookup_mask & target_mask).float().sum(dim=1)
                    selector_recall_at_k = (overlap / target_mask.float().sum(dim=1).clamp_min(1.0)).mean()
                    selector_precision_at_k = (overlap / lookup_mask.float().sum(dim=1).clamp_min(1.0)).mean()
                if output.budget_logits is not None and output.budget_target is not None:
                    budget_acc = (output.budget_logits.argmax(dim=1) == output.budget_target).float().mean()
            if input_ids.shape[1] % block_size == 0:
                num_blocks = input_ids.shape[1] // block_size
                block_matches = matches.reshape(input_ids.shape[0], num_blocks, block_size)
                for block_idx in range(num_blocks):
                    block_metrics.append(
                        {
                            "token_acc": block_matches[:, block_idx, :].float().mean(),
                            "exact_acc": block_matches[:, block_idx, :].all(dim=-1).float().mean(),
                        }
                    )

        self.log(f"{stage}/loss", loss, prog_bar=stage == "val")
        self.log(f"{stage}/token_ce", token_loss)
        self.log(f"{stage}/weighted_token_ce", weighted_token_loss)
        self.log(f"{stage}/selector_loss", selector_loss)
        self.log(f"{stage}/budget_loss", budget_loss)
        self.log(f"{stage}/gist_loss", gist_loss)
        self.log(f"{stage}/residual_router_loss", residual_router_loss)
        self.log(f"{stage}/variable_window_loss", variable_window_loss)
        self.log(f"{stage}/variable_window_cost_loss", variable_window_cost_loss)
        self.log(f"{stage}/variable_window_nonimprove_loss", variable_window_nonimprove_loss)
        self.log(f"{stage}/lookup_slot_cost_loss", lookup_slot_cost_loss)
        self.log(f"{stage}/lookup_slot_target_loss", lookup_slot_target_loss)
        self.log(f"{stage}/token_acc", token_acc, prog_bar=stage == "val")
        self.log(f"{stage}/token_top5_acc", top5_acc)
        self.log(f"{stage}/token_top10_acc", top10_acc)
        self.log(f"{stage}/exact_chunk_acc", exact_chunk_acc)
        self.log(f"{stage}/chunk_deviation_mean", chunk_deviation["mean"], prog_bar=stage == "val")
        self.log(f"{stage}/chunk_deviation_rate_mean", chunk_deviation["rate_mean"])
        self.log(f"{stage}/chunk_deviation_p50", chunk_deviation["p50"])
        self.log(f"{stage}/chunk_deviation_p90", chunk_deviation["p90"])
        self.log(f"{stage}/chunk_deviation_p95", chunk_deviation["p95"])
        self.log(f"{stage}/chunk_deviation_max", chunk_deviation["max"])
        for block_idx, metrics in enumerate(block_metrics):
            self.log(f"{stage}/block{block_idx}_token_acc", metrics["token_acc"])
            self.log(f"{stage}/block{block_idx}_exact_acc", metrics["exact_acc"])
        self.log(f"{stage}/bit_balance_loss", bit_balance_loss)
        self.log(f"{stage}/bit_density", bit_density)
        self.log(f"{stage}/diffusion_embed_mse", diffusion_loss)
        self.log(f"{stage}/bits_per_token", torch.tensor(self.model.bits_per_token, device=self.device))
        self.log(f"{stage}/effective_bits_per_token", torch.tensor(self.model.effective_bits_per_token, device=self.device))
        if output.residual_router_logits is not None and output.residual_router_target is not None:
            self.log(f"{stage}/residual_router_acc", residual_router_acc)
            self.log(f"{stage}/residual_high_recall", residual_high_recall)
            self.log(f"{stage}/residual_high_precision", residual_high_precision)
            self.log(f"{stage}/residual_high_fraction", residual_high_fraction)
            self.log(f"{stage}/residual_medium_fraction", residual_medium_fraction)
            self.log(f"{stage}/residual_fine_region_fraction", residual_fine_region_fraction)
        if output.variable_window_probs is not None:
            self.log(f"{stage}/variable_window_acc", variable_window_acc)
            self.log(f"{stage}/variable_window_fine_fraction", variable_window_fine_fraction)
            self.log(f"{stage}/variable_window_medium_fraction", variable_window_medium_fraction)
            self.log(f"{stage}/variable_window_full_fraction", variable_window_full_fraction)
            self.log(f"{stage}/variable_window_target_fine_fraction", variable_window_target_fine_fraction)
            self.log(f"{stage}/variable_window_target_medium_fraction", variable_window_target_medium_fraction)
            self.log(f"{stage}/variable_window_target_full_fraction", variable_window_target_full_fraction)
            self.log(f"{stage}/variable_window_expected_tokens", variable_window_expected_tokens)
            self.log(f"{stage}/variable_window_soft_expected_tokens", variable_window_soft_expected_tokens)
            self.log(f"{stage}/variable_window_soft_entropy", variable_window_soft_entropy)
            self.log(f"{stage}/variable_window_code_bits_per_chunk", variable_window_code_bits)
            self.log(f"{stage}/variable_window_base_deviation_mean", variable_window_base_deviation_mean)
            self.log(f"{stage}/variable_window_oracle_deviation_mean", variable_window_oracle_deviation_mean)
            self.log(f"{stage}/variable_window_chosen_deviation_mean", variable_window_chosen_deviation_mean)
            self.log(f"{stage}/variable_window_expected_deviation_mean", variable_window_expected_deviation_mean)
            self.log(f"{stage}/variable_window_regret_mean", variable_window_regret_mean)
            self.log(f"{stage}/variable_window_nonimprove_rate", variable_window_nonimprove_rate)
            self.log(f"{stage}/variable_window_fine_deviation_mean", variable_window_fine_deviation_mean)
            self.log(f"{stage}/variable_window_medium_deviation_mean", variable_window_medium_deviation_mean)
            self.log(f"{stage}/variable_window_full_deviation_mean", variable_window_full_deviation_mean)
        if lookup_metrics_enabled:
            self.log(f"{stage}/lookup_selected_fraction", lookup_selected_fraction)
            self.log(f"{stage}/lookup_token_acc", lookup_token_acc)
            self.log(f"{stage}/non_lookup_token_acc", non_lookup_token_acc)
            self.log(f"{stage}/lookup_gate_mean", lookup_gate_mean)
            self.log(f"{stage}/selector_recall_at_k", selector_recall_at_k)
            self.log(f"{stage}/selector_precision_at_k", selector_precision_at_k)
            self.log(f"{stage}/lookup_budget_k_mean", lookup_budget_k_mean)
            self.log(f"{stage}/lookup_active_k_mean", lookup_active_k_mean)
            self.log(f"{stage}/lookup_keep_prob_mean", lookup_keep_prob_mean)
            self.log(f"{stage}/observed_effective_bits_per_token", observed_effective_bits_per_token)
            self.log(f"{stage}/budget_acc", budget_acc)
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=float(self.training_cfg.get("learning_rate", 3e-4)),
            weight_decay=float(self.training_cfg.get("weight_decay", 0.01)),
        )


class _ValidationLossTrend(Callback):
    def __init__(self) -> None:
        self.val_losses: list[float] = []

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: L.LightningModule) -> None:
        del pl_module
        metric = trainer.callback_metrics.get("val/loss")
        if metric is not None:
            self.val_losses.append(float(metric.detach().cpu().item()))


def _build_tokenizer_data_module(
    *,
    config: dict[str, Any],
    accelerator: str,
) -> tuple[DABETokenizerDataModule, dict[str, Any], dict[str, Any], dict[str, Any]]:
    experiment_cfg = dict(config.get("experiment", {}))
    tokenizer_cfg = dict(config.get("dabe_tokenizer", {}))
    dataset_cfg = dict(tokenizer_cfg.get("dataset", {}))
    codec_cfg = dict(tokenizer_cfg.get("codec", {}))
    model_cfg = dict(tokenizer_cfg.get("model", {}))
    training_cfg = dict(tokenizer_cfg.get("training", {}))

    train_texts, val_texts = _resolve_tokenizer_texts(dataset_cfg, experiment_cfg)

    data_module = DABETokenizerDataModule(
        train_texts=train_texts,
        val_texts=val_texts,
        tokenizer_name=str(codec_cfg.get("tokenizer_name", "gpt2")),
        force_fast_tokenizer=bool(codec_cfg.get("force_fast_tokenizer", True)),
        chunk_size_tokens=int(codec_cfg.get("chunk_size_tokens", 64)),
        chunk_stride_tokens=int(codec_cfg.get("chunk_stride_tokens", codec_cfg.get("chunk_size_tokens", 64))),
        tokenizer_batch_size=int(codec_cfg.get("tokenizer_batch_size", 64)),
        batch_size=int(training_cfg.get("batch_size", 32)),
        num_workers=int(training_cfg.get("num_workers", 0)),
        seed=int(experiment_cfg.get("seed", 42)),
        accelerator=accelerator,
    )
    data_module.setup()
    return data_module, codec_cfg, model_cfg, training_cfg


def _reverse_probe_metrics(
    *,
    model: DABEChunkTokenizerAutoencoder,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int,
    sample_steps: int,
) -> dict[str, Any]:
    modes = {
        "one_step_tmax": 1,
        "reverse_ddim": sample_steps,
    }
    totals: dict[str, dict[str, Any]] = {
        name: {
            "tokens_correct": 0.0,
            "tokens_total": 0.0,
            "top5_correct": 0.0,
            "top10_correct": 0.0,
            "chunks_correct": 0.0,
            "chunks_total": 0.0,
            "chunk_deviation_sum": 0.0,
            "chunk_deviations": [],
            "embed_mse_sum": 0.0,
        }
        for name in modes
    }
    bit_density_sum = 0.0
    batches_seen = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            target_embeddings = model.token_embed(input_ids)
            bits, _ = model.encode_bits(input_ids)
            bit_density_sum += float(bits.mean().detach().cpu().item())
            batches_seen += 1

            for name, steps in modes.items():
                logits, decoded_embeddings = model.reverse_diffusion_decode(bits=bits, sample_steps=steps, stochastic=False)
                pred_ids = logits.argmax(dim=-1)
                topk_limit = min(10, int(logits.shape[-1]))
                topk_ids = torch.topk(logits, k=topk_limit, dim=-1).indices
                top5 = (topk_ids[..., : min(5, topk_limit)] == input_ids.unsqueeze(-1)).any(dim=-1)
                top10 = (topk_ids == input_ids.unsqueeze(-1)).any(dim=-1)
                matches = pred_ids == input_ids
                exact = matches.all(dim=-1)
                chunk_deviation = (~matches).float().sum(dim=1)
                metrics = totals[name]
                metrics["tokens_correct"] += float(matches.sum().detach().cpu().item())
                metrics["tokens_total"] += float(input_ids.numel())
                metrics["top5_correct"] += float(top5.sum().detach().cpu().item())
                metrics["top10_correct"] += float(top10.sum().detach().cpu().item())
                metrics["chunks_correct"] += float(exact.sum().detach().cpu().item())
                metrics["chunks_total"] += float(input_ids.shape[0])
                metrics["chunk_deviation_sum"] += float(chunk_deviation.sum().detach().cpu().item())
                metrics["chunk_deviations"].extend(float(x) for x in chunk_deviation.detach().cpu().tolist())
                metrics["embed_mse_sum"] += float(
                    F.mse_loss(decoded_embeddings, target_embeddings, reduction="sum").detach().cpu().item()
                )

    result: dict[str, Any] = {
        "batches": int(batches_seen),
        "bit_density": bit_density_sum / max(1, batches_seen),
    }
    for name, metrics in totals.items():
        token_total = max(1.0, metrics["tokens_total"])
        chunk_total = max(1.0, metrics["chunks_total"])
        deviations = torch.tensor(metrics["chunk_deviations"], dtype=torch.float64)
        if deviations.numel() == 0:
            deviations = torch.zeros(1, dtype=torch.float64)
        result[name] = {
            "token_acc": metrics["tokens_correct"] / token_total,
            "token_top5_acc": metrics["top5_correct"] / token_total,
            "token_top10_acc": metrics["top10_correct"] / token_total,
            "exact_chunk_acc": metrics["chunks_correct"] / chunk_total,
            "chunk_deviation_mean": metrics["chunk_deviation_sum"] / chunk_total,
            "chunk_deviation_rate_mean": metrics["chunk_deviation_sum"] / token_total,
            "chunk_deviation_p50": float(torch.quantile(deviations, 0.50).item()),
            "chunk_deviation_p90": float(torch.quantile(deviations, 0.90).item()),
            "chunk_deviation_p95": float(torch.quantile(deviations, 0.95).item()),
            "chunk_deviation_max": float(deviations.max().item()),
            "embed_mse": metrics["embed_mse_sum"] / token_total,
        }
    return result


def run_dabe_tokenizer_reverse_diffusion_probe(
    *,
    config: dict[str, Any],
    checkpoint_path: Path,
    output_dir: Path,
    accelerator: str,
    devices: int | str,
    max_batches: int = 16,
    sample_steps: int = 64,
) -> dict[str, Any]:
    del devices
    data_module, codec_cfg, _model_cfg, _training_cfg = _build_tokenizer_data_module(
        config=config,
        accelerator=accelerator,
    )
    device = torch.device("cuda" if accelerator == "gpu" and torch.cuda.is_available() else "cpu")
    module = DABETokenizerAutoencoderModule.load_from_checkpoint(str(checkpoint_path), map_location=device, strict=False)
    module.to(device)
    model = module.model
    if model.decoder_mode != "diffusion":
        raise ValueError("Reverse diffusion probe requires a checkpoint trained with decoder_mode=diffusion.")

    metrics = _reverse_probe_metrics(
        model=model,
        dataloader=data_module.val_dataloader(),
        device=device,
        max_batches=max(1, int(max_batches)),
        sample_steps=max(1, int(sample_steps)),
    )
    result = {
        "checkpoint_path": str(checkpoint_path),
        "max_batches": int(max_batches),
        "sample_steps": int(sample_steps),
        "train_chunks": int(data_module.train_chunks.shape[0] if data_module.train_chunks is not None else 0),
        "val_chunks": int(data_module.val_chunks.shape[0] if data_module.val_chunks is not None else 0),
        "chunk_size_tokens": int(codec_cfg.get("chunk_size_tokens", 64)),
        "code_bits": int(model.code_bits),
        "bits_per_token": float(model.bits_per_token),
        "lookup_bits_per_chunk": int(model.lookup_bits_per_chunk),
        "effective_bits_per_token": float(model.effective_bits_per_token),
        "decoder_mode": str(model.decoder_mode),
        **metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reverse_diffusion_probe.json").write_text(json.dumps(result, indent=2))
    return result


def _build_repair_sample(
    *,
    tokenizer: Any,
    batch_idx: int,
    row_idx: int,
    sample_index: int,
    input_cpu: torch.Tensor,
    pred_cpu: torch.Tensor,
    matches_cpu: torch.Tensor,
    chunk_deviation_cpu: torch.Tensor,
    output: DABETokenizerAutoencoderOutput,
    gist_pred_ids: torch.Tensor | None,
    gist_matches: torch.Tensor | None,
    decode_token: Any,
    raw_token: Any,
) -> dict[str, Any]:
    """Build a per-position repair trace for Section 5 interpretability diagnostics."""
    gist_cpu = gist_pred_ids.detach().cpu() if gist_pred_ids is not None else None
    gist_matches_cpu = gist_matches.detach().cpu() if gist_matches is not None else None
    lookup_positions_cpu = output.lookup_positions.detach().cpu() if output.lookup_positions is not None else None
    lookup_active_cpu = output.lookup_active_mask.detach().cpu() if output.lookup_active_mask is not None else None
    lookup_keep_cpu = output.lookup_keep_probs.detach().cpu() if output.lookup_keep_probs is not None else None

    target_tokens = [int(x) for x in input_cpu[row_idx].tolist()]
    pred_tokens = [int(x) for x in pred_cpu[row_idx].tolist()]
    gist_tokens = [int(x) for x in gist_cpu[row_idx].tolist()] if gist_cpu is not None else []
    lookup_positions = (
        [int(x) for x in lookup_positions_cpu[row_idx].tolist()] if lookup_positions_cpu is not None else []
    )
    lookup_active = (
        [bool(x) for x in lookup_active_cpu[row_idx].tolist()]
        if lookup_active_cpu is not None
        else ([True] * len(lookup_positions))
    )
    lookup_keep_probs = (
        [float(x) for x in lookup_keep_cpu[row_idx].tolist()] if lookup_keep_cpu is not None else []
    )
    lookup_slot_by_position = {pos: slot_idx for slot_idx, pos in enumerate(lookup_positions)}
    repair_trace = []
    outcome_counts: dict[str, int] = {}
    for pos, target_id in enumerate(target_tokens):
        pred_id = pred_tokens[pos]
        gist_id = gist_tokens[pos] if gist_tokens else None
        match = bool(matches_cpu[row_idx, pos].item())
        gist_match = bool(gist_matches_cpu[row_idx, pos].item()) if gist_matches_cpu is not None else None
        lookup_slot = lookup_slot_by_position.get(pos)
        lookup_is_active = bool(lookup_active[lookup_slot]) if lookup_slot is not None and lookup_slot < len(lookup_active) else False
        keep_prob = (
            float(lookup_keep_probs[lookup_slot])
            if lookup_slot is not None and lookup_slot < len(lookup_keep_probs)
            else None
        )
        if gist_id is None or gist_match is None:
            outcome = "no_gist_trace"
        elif (not gist_match) and match:
            outcome = "corrected"
        elif gist_match and (not match):
            outcome = "damaged"
        elif gist_id != pred_id and not match:
            outcome = "changed_wrong_to_wrong"
        elif match:
            outcome = "preserved_correct"
        else:
            outcome = "missed"
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        repair_trace.append(
            {
                "position": int(pos),
                "target_id": int(target_id),
                "target_token": raw_token(target_id),
                "target_text": decode_token(target_id),
                "gist_pred_id": int(gist_id) if gist_id is not None else None,
                "gist_pred_token": raw_token(gist_id) if gist_id is not None else None,
                "gist_pred_text": decode_token(gist_id) if gist_id is not None else None,
                "pred_id": int(pred_id),
                "pred_token": raw_token(pred_id),
                "pred_text": decode_token(pred_id),
                "match": match,
                "gist_match": gist_match,
                "lookup_candidate": lookup_slot is not None,
                "lookup_active": lookup_is_active,
                "lookup_slot_index": int(lookup_slot) if lookup_slot is not None else None,
                "lookup_keep_prob": keep_prob,
                "repair_outcome": outcome,
            }
        )
    return {
        "sample_index": int(sample_index),
        "batch_index": int(batch_idx),
        "row_index": int(row_idx),
        "target_text": tokenizer.decode(target_tokens),
        "pred_text": tokenizer.decode(pred_tokens),
        "gist_pred_text": tokenizer.decode(gist_tokens) if gist_tokens else None,
        "target_token_ids": target_tokens,
        "pred_token_ids": pred_tokens,
        "gist_pred_token_ids": gist_tokens,
        "match_mask": [bool(x) for x in matches_cpu[row_idx].tolist()],
        "gist_match_mask": [bool(x) for x in gist_matches_cpu[row_idx].tolist()] if gist_matches_cpu is not None else [],
        "lookup_positions": lookup_positions,
        "lookup_active_mask": lookup_active,
        "lookup_budget_k": (
            float(output.lookup_budget_k.detach().cpu()[row_idx].item()) if output.lookup_budget_k is not None else 0.0
        ),
        "lookup_keep_probs": lookup_keep_probs,
        "repair_trace": repair_trace,
        "repair_outcome_counts": outcome_counts,
        "repair_corrected_count": int(outcome_counts.get("corrected", 0)),
        "repair_changed_count": int(
            outcome_counts.get("corrected", 0)
            + outcome_counts.get("damaged", 0)
            + outcome_counts.get("changed_wrong_to_wrong", 0)
        ),
        "token_acc": float(matches_cpu[row_idx].float().mean().item()),
        "gist_token_acc": float(gist_matches_cpu[row_idx].float().mean().item()) if gist_matches_cpu is not None else None,
        "chunk_deviation": float(chunk_deviation_cpu[row_idx].item()),
    }


def run_dabe_tokenizer_decode_diagnostics(
    *,
    config: dict[str, Any],
    checkpoint_path: Path,
    output_dir: Path,
    accelerator: str,
    devices: int | str,
    max_batches: int = 128,
    num_samples: int = 12,
) -> dict[str, Any]:
    del devices
    data_module, codec_cfg, _model_cfg, _training_cfg = _build_tokenizer_data_module(
        config=config,
        accelerator=accelerator,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(codec_cfg.get("tokenizer_name", "gpt2")),
        use_fast=bool(codec_cfg.get("force_fast_tokenizer", True)),
    )
    device = torch.device("cuda" if accelerator == "gpu" and torch.cuda.is_available() else "cpu")
    module = DABETokenizerAutoencoderModule.load_from_checkpoint(str(checkpoint_path), map_location=device, strict=False)
    module.to(device)
    model = module.model
    model.eval()

    chunk_size = int(model.chunk_size_tokens)
    position_correct = torch.zeros(chunk_size, dtype=torch.float64)
    position_top5 = torch.zeros(chunk_size, dtype=torch.float64)
    position_top10 = torch.zeros(chunk_size, dtype=torch.float64)
    position_total = torch.zeros(chunk_size, dtype=torch.float64)
    token_correct = 0.0
    top5_correct = 0.0
    top10_correct = 0.0
    token_total = 0.0
    gist_token_correct = 0.0
    gist_token_total = 0.0
    repair_changed_total = 0.0
    repair_corrected_total = 0.0
    repair_damaged_total = 0.0
    repair_target_total = 0.0
    lookup_repair_corrected_total = 0.0
    lookup_repair_changed_total = 0.0
    lookup_repair_total = 0.0
    chunk_correct = 0.0
    chunk_total = 0.0
    chunk_deviation_sum = 0.0
    chunk_deviations: list[float] = []
    block_size = int(getattr(model, "hierarchical_block_tokens", 16))
    if chunk_size % block_size != 0:
        block_size = chunk_size
    num_blocks = chunk_size // block_size
    half_size = chunk_size // 2 if chunk_size % 2 == 0 else chunk_size
    num_halves = chunk_size // half_size
    block_correct = torch.zeros(num_blocks, dtype=torch.float64)
    block_total = torch.zeros(num_blocks, dtype=torch.float64)
    half_correct = torch.zeros(num_halves, dtype=torch.float64)
    half_total = torch.zeros(num_halves, dtype=torch.float64)
    loss_sum = 0.0
    bit_density_sum = 0.0
    lookup_token_correct = 0.0
    lookup_token_total = 0.0
    non_lookup_token_correct = 0.0
    non_lookup_token_total = 0.0
    lookup_gate_sum = 0.0
    lookup_gate_total = 0.0
    selector_overlap_total = 0.0
    selector_target_total = 0.0
    selector_selected_total = 0.0
    lookup_budget_k_sum = 0.0
    lookup_budget_k_total = 0.0
    lookup_active_k_sum = 0.0
    lookup_keep_prob_sum = 0.0
    lookup_keep_prob_total = 0.0
    budget_correct = 0.0
    budget_total = 0.0
    batches_seen = 0
    samples: list[dict[str, Any]] = []
    corrected_samples: list[dict[str, Any]] = []

    def _decode_token(token_id: int) -> str:
        return tokenizer.decode([int(token_id)])

    def _raw_token(token_id: int) -> str:
        convert = getattr(tokenizer, "convert_ids_to_tokens", None)
        if convert is None:
            return str(int(token_id))
        return str(convert(int(token_id)))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    diagnostic_start = time.perf_counter()

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_module.val_dataloader()):
            if batch_idx >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            output = model(input_ids)
            logits = output.logits
            pred_ids = logits.argmax(dim=-1)
            topk_limit = min(10, int(logits.shape[-1]))
            topk_ids = torch.topk(logits, k=topk_limit, dim=-1).indices
            matches = pred_ids == input_ids
            top5_matches = (topk_ids[..., : min(5, topk_limit)] == input_ids.unsqueeze(-1)).any(dim=-1)
            top10_matches = (topk_ids == input_ids.unsqueeze(-1)).any(dim=-1)
            exact = matches.all(dim=-1)
            chunk_deviation = (~matches).float().sum(dim=1)
            gist_pred_ids = None
            gist_matches = None
            repair_changed = None
            repair_corrected = None
            repair_damaged = None
            if output.gist_logits is not None:
                gist_pred_ids = output.gist_logits.argmax(dim=-1)
                gist_matches = gist_pred_ids == input_ids
                repair_changed = gist_pred_ids != pred_ids
                repair_corrected = (~gist_matches) & matches
                repair_damaged = gist_matches & (~matches)
                gist_token_correct += float(gist_matches.sum().detach().cpu().item())
                gist_token_total += float(input_ids.numel())
                repair_changed_total += float(repair_changed.sum().detach().cpu().item())
                repair_corrected_total += float(repair_corrected.sum().detach().cpu().item())
                repair_damaged_total += float(repair_damaged.sum().detach().cpu().item())
                repair_target_total += float((~gist_matches).sum().detach().cpu().item())
            block_exact = matches.reshape(input_ids.shape[0], num_blocks, block_size).all(dim=-1)
            half_exact = matches.reshape(input_ids.shape[0], num_halves, half_size).all(dim=-1)
            batch_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), input_ids.reshape(-1), reduction="sum")
            lookup_mask = None
            if output.lookup_positions is not None:
                active_mask = (
                    output.lookup_active_mask
                    if output.lookup_active_mask is not None
                    else torch.ones_like(output.lookup_positions, dtype=torch.bool)
                )
                lookup_mask = torch.zeros_like(input_ids, dtype=torch.bool)
                lookup_mask.scatter_(dim=1, index=output.lookup_positions, src=active_mask)
                non_lookup_mask = ~lookup_mask
                if repair_corrected is not None and repair_changed is not None:
                    lookup_repair_corrected_total += float((repair_corrected & lookup_mask).sum().detach().cpu().item())
                    lookup_repair_changed_total += float((repair_changed & lookup_mask).sum().detach().cpu().item())
                    lookup_repair_total += float(lookup_mask.sum().detach().cpu().item())
                lookup_token_correct += float(matches[lookup_mask].sum().detach().cpu().item())
                lookup_token_total += float(lookup_mask.sum().detach().cpu().item())
                non_lookup_token_correct += float(matches[non_lookup_mask].sum().detach().cpu().item())
                non_lookup_token_total += float(non_lookup_mask.sum().detach().cpu().item())
                if output.lookup_gate is not None:
                    lookup_gate_sum += float(output.lookup_gate[lookup_mask].sum().detach().cpu().item())
                    lookup_gate_total += float(lookup_mask.sum().detach().cpu().item())
                if output.selector_target_mask is not None:
                    target_mask = output.selector_target_mask.bool()
                    selector_overlap_total += float((lookup_mask & target_mask).sum().detach().cpu().item())
                    selector_target_total += float(target_mask.sum().detach().cpu().item())
                    selector_selected_total += float(lookup_mask.sum().detach().cpu().item())
                if output.lookup_budget_k is not None:
                    lookup_budget_k_sum += float(output.lookup_budget_k.float().sum().detach().cpu().item())
                    lookup_budget_k_total += float(output.lookup_budget_k.numel())
                lookup_active_k_sum += float(active_mask.float().sum(dim=1).sum().detach().cpu().item())
                if output.lookup_keep_probs is not None:
                    lookup_keep_prob_sum += float(output.lookup_keep_probs.float().sum().detach().cpu().item())
                    lookup_keep_prob_total += float(output.lookup_keep_probs.numel())
                if output.budget_logits is not None and output.budget_target is not None:
                    budget_correct += float(
                        (output.budget_logits.argmax(dim=1) == output.budget_target).sum().detach().cpu().item()
                    )
                    budget_total += float(output.budget_target.numel())

            token_correct += float(matches.sum().detach().cpu().item())
            top5_correct += float(top5_matches.sum().detach().cpu().item())
            top10_correct += float(top10_matches.sum().detach().cpu().item())
            token_total += float(input_ids.numel())
            chunk_correct += float(exact.sum().detach().cpu().item())
            chunk_total += float(input_ids.shape[0])
            chunk_deviation_sum += float(chunk_deviation.sum().detach().cpu().item())
            chunk_deviations.extend(float(x) for x in chunk_deviation.detach().cpu().tolist())
            block_correct += block_exact.detach().cpu().double().sum(dim=0)
            block_total += torch.full((num_blocks,), float(input_ids.shape[0]), dtype=torch.float64)
            half_correct += half_exact.detach().cpu().double().sum(dim=0)
            half_total += torch.full((num_halves,), float(input_ids.shape[0]), dtype=torch.float64)
            loss_sum += float(batch_loss.detach().cpu().item())
            bit_density_sum += float(output.bits.mean().detach().cpu().item())
            batches_seen += 1

            position_correct += matches.detach().cpu().double().sum(dim=0)
            position_top5 += top5_matches.detach().cpu().double().sum(dim=0)
            position_top10 += top10_matches.detach().cpu().double().sum(dim=0)
            position_total += torch.full((chunk_size,), float(input_ids.shape[0]), dtype=torch.float64)

            if len(samples) < num_samples:
                input_cpu = input_ids.detach().cpu()
                pred_cpu = pred_ids.detach().cpu()
                matches_cpu = matches.detach().cpu()
                chunk_deviation_cpu = chunk_deviation.detach().cpu()
                for row_idx in range(input_cpu.shape[0]):
                    if len(samples) >= num_samples:
                        break
                    samples.append(
                        _build_repair_sample(
                            tokenizer=tokenizer,
                            batch_idx=batch_idx,
                            row_idx=row_idx,
                            sample_index=len(samples),
                            input_cpu=input_cpu,
                            pred_cpu=pred_cpu,
                            matches_cpu=matches_cpu,
                            chunk_deviation_cpu=chunk_deviation_cpu,
                            output=output,
                            gist_pred_ids=gist_pred_ids,
                            gist_matches=gist_matches,
                            decode_token=_decode_token,
                            raw_token=_raw_token,
                        )
                    )

            if gist_pred_ids is not None and gist_matches is not None:
                input_cpu = input_ids.detach().cpu()
                pred_cpu = pred_ids.detach().cpu()
                matches_cpu = matches.detach().cpu()
                chunk_deviation_cpu = chunk_deviation.detach().cpu()
                corrected_cpu = repair_corrected.detach().cpu() if repair_corrected is not None else None
                changed_cpu = repair_changed.detach().cpu() if repair_changed is not None else None
                for row_idx in range(input_cpu.shape[0]):
                    corrected_count = int(corrected_cpu[row_idx].sum().item()) if corrected_cpu is not None else 0
                    changed_count = int(changed_cpu[row_idx].sum().item()) if changed_cpu is not None else 0
                    if corrected_count <= 0:
                        continue
                    sample_payload = _build_repair_sample(
                        tokenizer=tokenizer,
                        batch_idx=batch_idx,
                        row_idx=row_idx,
                        sample_index=len(corrected_samples),
                        input_cpu=input_cpu,
                        pred_cpu=pred_cpu,
                        matches_cpu=matches_cpu,
                        chunk_deviation_cpu=chunk_deviation_cpu,
                        output=output,
                        gist_pred_ids=gist_pred_ids,
                        gist_matches=gist_matches,
                        decode_token=_decode_token,
                        raw_token=_raw_token,
                    )
                    sample_payload["repair_corrected_count"] = corrected_count
                    sample_payload["repair_changed_count"] = changed_count
                    corrected_samples.append(sample_payload)
                    corrected_samples.sort(
                        key=lambda item: (
                            int(item.get("repair_corrected_count", 0)),
                            float(item.get("token_acc", 0.0)),
                        ),
                        reverse=True,
                    )
                    del corrected_samples[max(1, int(num_samples)) :]

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    diagnostic_wall_seconds = time.perf_counter() - diagnostic_start
    gpu_peak_memory_allocated_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)) if device.type == "cuda" else None
    )
    gpu_peak_memory_reserved_mb = (
        float(torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)) if device.type == "cuda" else None
    )

    per_pos_acc = (position_correct / position_total.clamp_min(1.0)).tolist()
    per_pos_top5 = (position_top5 / position_total.clamp_min(1.0)).tolist()
    per_pos_top10 = (position_top10 / position_total.clamp_min(1.0)).tolist()
    ranked_positions = sorted(
        [{"position": idx, "token_acc": float(acc)} for idx, acc in enumerate(per_pos_acc)],
        key=lambda item: item["token_acc"],
    )
    per_block_exact = (block_correct / block_total.clamp_min(1.0)).tolist()
    per_half_exact = (half_correct / half_total.clamp_min(1.0)).tolist()
    chunk_deviation_values = torch.tensor(chunk_deviations, dtype=torch.float64)
    if chunk_deviation_values.numel() == 0:
        chunk_deviation_values = torch.zeros(1, dtype=torch.float64)
    lookup_budget_k_mean = lookup_budget_k_sum / max(1.0, lookup_budget_k_total)
    lookup_active_k_mean = lookup_active_k_sum / max(1.0, lookup_budget_k_total)
    lookup_keep_prob_mean = lookup_keep_prob_sum / max(1.0, lookup_keep_prob_total)
    observed_effective_bits_per_token = (
        float(model.code_bits) + lookup_budget_k_mean * float(model.lookup_entry_bits)
    ) / float(chunk_size)
    result = {
        "checkpoint_path": str(checkpoint_path),
        "max_batches": int(max_batches),
        "batches": int(batches_seen),
        "num_samples": int(len(samples)),
        "train_chunks": int(data_module.train_chunks.shape[0] if data_module.train_chunks is not None else 0),
        "val_chunks": int(data_module.val_chunks.shape[0] if data_module.val_chunks is not None else 0),
        "chunk_size_tokens": chunk_size,
        "diagnostic_block_size": int(block_size),
        "diagnostic_num_blocks": int(num_blocks),
        "diagnostic_half_size": int(half_size),
        "code_bits": int(model.code_bits),
        "bits_per_token": float(model.bits_per_token),
        "lookup_bits_per_chunk": int(model.lookup_bits_per_chunk),
        "effective_bits_per_token": float(model.effective_bits_per_token),
        "lookup_k_choices": [int(choice) for choice in model.lexical_lookup_k_choices],
        "lookup_budget_k_mean": lookup_budget_k_mean,
        "lookup_active_k_mean": lookup_active_k_mean,
        "lookup_keep_prob_mean": lookup_keep_prob_mean,
        "observed_effective_bits_per_token": observed_effective_bits_per_token,
        "decoder_mode": str(model.decoder_mode),
        "token_ce": loss_sum / max(1.0, token_total),
        "token_acc": token_correct / max(1.0, token_total),
        "token_top5_acc": top5_correct / max(1.0, token_total),
        "token_top10_acc": top10_correct / max(1.0, token_total),
        "gist_token_acc": gist_token_correct / max(1.0, gist_token_total) if gist_token_total > 0.0 else None,
        "repair_changed_fraction": repair_changed_total / max(1.0, token_total),
        "repair_correction_fraction": repair_corrected_total / max(1.0, repair_target_total),
        "repair_damage_fraction": repair_damaged_total / max(1.0, gist_token_total),
        "lookup_repair_correction_fraction": lookup_repair_corrected_total / max(1.0, lookup_repair_total),
        "lookup_repair_changed_fraction": lookup_repair_changed_total / max(1.0, lookup_repair_total),
        "exact_chunk_acc": chunk_correct / max(1.0, chunk_total),
        "chunk_deviation_mean": chunk_deviation_sum / max(1.0, chunk_total),
        "chunk_deviation_rate_mean": chunk_deviation_sum / max(1.0, token_total),
        "chunk_deviation_p50": float(torch.quantile(chunk_deviation_values, 0.50).item()),
        "chunk_deviation_p90": float(torch.quantile(chunk_deviation_values, 0.90).item()),
        "chunk_deviation_p95": float(torch.quantile(chunk_deviation_values, 0.95).item()),
        "chunk_deviation_max": float(chunk_deviation_values.max().item()),
        "exact_block_acc": float(block_correct.sum().item() / max(1.0, block_total.sum().item())),
        "exact_half_acc": float(half_correct.sum().item() / max(1.0, half_total.sum().item())),
        "per_block_exact_acc": per_block_exact,
        "per_half_exact_acc": per_half_exact,
        "bit_density": bit_density_sum / max(1, batches_seen),
        "lookup_token_acc": lookup_token_correct / max(1.0, lookup_token_total),
        "non_lookup_token_acc": non_lookup_token_correct / max(1.0, non_lookup_token_total),
        "lookup_gate_mean": lookup_gate_sum / max(1.0, lookup_gate_total),
        "selector_recall_at_k": selector_overlap_total / max(1.0, selector_target_total),
        "selector_precision_at_k": selector_overlap_total / max(1.0, selector_selected_total),
        "budget_acc": budget_correct / max(1.0, budget_total),
        "per_position_token_acc": per_pos_acc,
        "per_position_top5_acc": per_pos_top5,
        "per_position_top10_acc": per_pos_top10,
        "worst_positions": ranked_positions[:10],
        "best_positions": list(reversed(ranked_positions[-10:])),
        "diagnostic_wall_seconds": diagnostic_wall_seconds,
        "diagnostic_chunks_per_second": chunk_total / max(1.0e-9, diagnostic_wall_seconds),
        "diagnostic_tokens_per_second": token_total / max(1.0e-9, diagnostic_wall_seconds),
        "gpu_peak_memory_allocated_mb": gpu_peak_memory_allocated_mb,
        "gpu_peak_memory_reserved_mb": gpu_peak_memory_reserved_mb,
        "samples": samples,
        "corrected_samples": corrected_samples,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "decode_diagnostics.json").write_text(json.dumps(result, indent=2))
    return result


def run_dabe_tokenizer_autoencoder(
    *,
    config: dict[str, Any],
    output_dir: Path,
    accelerator: str,
    devices: int | str,
) -> dict[str, Any]:
    experiment_cfg = dict(config.get("experiment", {}))
    tokenizer_cfg = dict(config.get("dabe_tokenizer", {}))
    dataset_cfg = dict(tokenizer_cfg.get("dataset", {}))
    codec_cfg = dict(tokenizer_cfg.get("codec", {}))
    model_cfg = dict(tokenizer_cfg.get("model", {}))
    training_cfg = dict(tokenizer_cfg.get("training", {}))

    train_texts, val_texts = _resolve_tokenizer_texts(dataset_cfg, experiment_cfg)

    data_module = DABETokenizerDataModule(
        train_texts=train_texts,
        val_texts=val_texts,
        tokenizer_name=str(codec_cfg.get("tokenizer_name", "gpt2")),
        force_fast_tokenizer=bool(codec_cfg.get("force_fast_tokenizer", True)),
        chunk_size_tokens=int(codec_cfg.get("chunk_size_tokens", 64)),
        chunk_stride_tokens=int(codec_cfg.get("chunk_stride_tokens", codec_cfg.get("chunk_size_tokens", 64))),
        tokenizer_batch_size=int(codec_cfg.get("tokenizer_batch_size", 64)),
        batch_size=int(training_cfg.get("batch_size", 32)),
        num_workers=int(training_cfg.get("num_workers", 0)),
        seed=int(experiment_cfg.get("seed", 42)),
        accelerator=accelerator,
    )
    data_module.setup()

    model_config = {
        "model": {
            **model_cfg,
            "vocab_size": data_module.vocab_size,
            "chunk_size_tokens": int(codec_cfg.get("chunk_size_tokens", 64)),
            "code_bits": int(codec_cfg.get("code_bits", 256)),
        },
        "training": training_cfg,
    }
    module = DABETokenizerAutoencoderModule(model_config)

    trend = _ValidationLossTrend()
    eta_callback = EtaMetricsCallback(
        log_interval_steps=int(training_cfg.get("eta_log_interval_steps", 20)),
        emit_stdout=bool(training_cfg.get("eta_stdout", False)),
    )
    callbacks: list[Callback] = [trend, eta_callback]
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-step-{step:07d}",
        monitor="val/loss",
        mode="min",
        save_top_k=int(training_cfg.get("checkpoint_save_top_k", 1)),
        save_last=bool(training_cfg.get("checkpoint_save_last", True)),
        every_n_train_steps=int(training_cfg.get("checkpoint_every_n_train_steps", 100)),
        auto_insert_metric_name=False,
    )
    callbacks.append(checkpoint_callback)
    logger = (
        CSVLogger(save_dir=str(output_dir / "logs"), name="")
        if bool(training_cfg.get("enable_csv_logger", True))
        else False
    )
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        max_steps=int(training_cfg.get("max_steps", 100)),
        val_check_interval=int(training_cfg.get("val_check_interval", 25)),
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=float(training_cfg.get("grad_clip", 1.0)),
        log_every_n_steps=int(training_cfg.get("log_every_n_steps", 10)),
        precision=_resolve_precision(training_cfg, accelerator),
        enable_checkpointing=True,
        enable_progress_bar=bool(training_cfg.get("enable_progress_bar", False)),
    )
    trainer.fit(module, datamodule=data_module)

    val_loss_first = trend.val_losses[0] if trend.val_losses else None
    val_loss_last = trend.val_losses[-1] if trend.val_losses else None
    result = {
        "stable": module.nan_batches == 0,
        "nan_batches": int(module.nan_batches),
        "train_chunks": int(data_module.train_chunks.shape[0] if data_module.train_chunks is not None else 0),
        "val_chunks": int(data_module.val_chunks.shape[0] if data_module.val_chunks is not None else 0),
        "chunk_size_tokens": int(codec_cfg.get("chunk_size_tokens", 64)),
        "code_bits": int(codec_cfg.get("code_bits", 256)),
        "bits_per_token": float(module.model.bits_per_token),
        "lookup_bits_per_chunk": int(module.model.lookup_bits_per_chunk),
        "effective_bits_per_token": float(module.model.effective_bits_per_token),
        "lookup_k_choices": [int(choice) for choice in module.model.lexical_lookup_k_choices],
        "decoder_mode": str(module.model.decoder_mode),
        "val_loss_first": val_loss_first,
        "val_loss_last": val_loss_last,
        "val_token_acc_last": (
            float(trainer.callback_metrics["val/token_acc"].item())
            if "val/token_acc" in trainer.callback_metrics
            else None
        ),
        "val_exact_chunk_acc_last": (
            float(trainer.callback_metrics["val/exact_chunk_acc"].item())
            if "val/exact_chunk_acc" in trainer.callback_metrics
            else None
        ),
        "val_chunk_deviation_mean_last": (
            float(trainer.callback_metrics["val/chunk_deviation_mean"].item())
            if "val/chunk_deviation_mean" in trainer.callback_metrics
            else None
        ),
        "val_chunk_deviation_rate_mean_last": (
            float(trainer.callback_metrics["val/chunk_deviation_rate_mean"].item())
            if "val/chunk_deviation_rate_mean" in trainer.callback_metrics
            else None
        ),
        "val_chunk_deviation_p50_last": (
            float(trainer.callback_metrics["val/chunk_deviation_p50"].item())
            if "val/chunk_deviation_p50" in trainer.callback_metrics
            else None
        ),
        "val_chunk_deviation_p90_last": (
            float(trainer.callback_metrics["val/chunk_deviation_p90"].item())
            if "val/chunk_deviation_p90" in trainer.callback_metrics
            else None
        ),
        "val_chunk_deviation_p95_last": (
            float(trainer.callback_metrics["val/chunk_deviation_p95"].item())
            if "val/chunk_deviation_p95" in trainer.callback_metrics
            else None
        ),
        "val_chunk_deviation_max_last": (
            float(trainer.callback_metrics["val/chunk_deviation_max"].item())
            if "val/chunk_deviation_max" in trainer.callback_metrics
            else None
        ),
        "val_token_top5_acc_last": (
            float(trainer.callback_metrics["val/token_top5_acc"].item())
            if "val/token_top5_acc" in trainer.callback_metrics
            else None
        ),
        "val_token_top10_acc_last": (
            float(trainer.callback_metrics["val/token_top10_acc"].item())
            if "val/token_top10_acc" in trainer.callback_metrics
            else None
        ),
        "val_bit_density_last": (
            float(trainer.callback_metrics["val/bit_density"].item())
            if "val/bit_density" in trainer.callback_metrics
            else None
        ),
        "val_diffusion_embed_mse_last": (
            float(trainer.callback_metrics["val/diffusion_embed_mse"].item())
            if "val/diffusion_embed_mse" in trainer.callback_metrics
            else None
        ),
        "val_weighted_token_ce_last": (
            float(trainer.callback_metrics["val/weighted_token_ce"].item())
            if "val/weighted_token_ce" in trainer.callback_metrics
            else None
        ),
        "val_gist_loss_last": (
            float(trainer.callback_metrics["val/gist_loss"].item())
            if "val/gist_loss" in trainer.callback_metrics
            else None
        ),
        "val_residual_router_loss_last": (
            float(trainer.callback_metrics["val/residual_router_loss"].item())
            if "val/residual_router_loss" in trainer.callback_metrics
            else None
        ),
        "val_residual_router_acc_last": (
            float(trainer.callback_metrics["val/residual_router_acc"].item())
            if "val/residual_router_acc" in trainer.callback_metrics
            else None
        ),
        "val_residual_high_recall_last": (
            float(trainer.callback_metrics["val/residual_high_recall"].item())
            if "val/residual_high_recall" in trainer.callback_metrics
            else None
        ),
        "val_residual_high_precision_last": (
            float(trainer.callback_metrics["val/residual_high_precision"].item())
            if "val/residual_high_precision" in trainer.callback_metrics
            else None
        ),
        "val_residual_high_fraction_last": (
            float(trainer.callback_metrics["val/residual_high_fraction"].item())
            if "val/residual_high_fraction" in trainer.callback_metrics
            else None
        ),
        "val_residual_medium_fraction_last": (
            float(trainer.callback_metrics["val/residual_medium_fraction"].item())
            if "val/residual_medium_fraction" in trainer.callback_metrics
            else None
        ),
        "val_residual_fine_region_fraction_last": (
            float(trainer.callback_metrics["val/residual_fine_region_fraction"].item())
            if "val/residual_fine_region_fraction" in trainer.callback_metrics
            else None
        ),
        "val_variable_window_loss_last": (
            float(trainer.callback_metrics["val/variable_window_loss"].item())
            if "val/variable_window_loss" in trainer.callback_metrics
            else None
        ),
        "val_variable_window_cost_loss_last": (
            float(trainer.callback_metrics["val/variable_window_cost_loss"].item())
            if "val/variable_window_cost_loss" in trainer.callback_metrics
            else None
        ),
        "val_variable_window_nonimprove_loss_last": (
            float(trainer.callback_metrics["val/variable_window_nonimprove_loss"].item())
            if "val/variable_window_nonimprove_loss" in trainer.callback_metrics
            else None
        ),
        "val_variable_window_acc_last": (
            float(trainer.callback_metrics["val/variable_window_acc"].item())
            if "val/variable_window_acc" in trainer.callback_metrics
            else None
        ),
        "val_variable_window_fine_fraction_last": (
            float(trainer.callback_metrics["val/variable_window_fine_fraction"].item())
            if "val/variable_window_fine_fraction" in trainer.callback_metrics
            else None
        ),
        "val_variable_window_medium_fraction_last": (
            float(trainer.callback_metrics["val/variable_window_medium_fraction"].item())
            if "val/variable_window_medium_fraction" in trainer.callback_metrics
            else None
        ),
        "val_variable_window_full_fraction_last": (
            float(trainer.callback_metrics["val/variable_window_full_fraction"].item())
            if "val/variable_window_full_fraction" in trainer.callback_metrics
            else None
        ),
        "val_variable_window_target_fine_fraction_last": (
            float(trainer.callback_metrics["val/variable_window_target_fine_fraction"].item())
            if "val/variable_window_target_fine_fraction" in trainer.callback_metrics
            else None
        ),
        "val_variable_window_target_medium_fraction_last": (
            float(trainer.callback_metrics["val/variable_window_target_medium_fraction"].item())
            if "val/variable_window_target_medium_fraction" in trainer.callback_metrics
            else None
        ),
        "val_variable_window_target_full_fraction_last": (
            float(trainer.callback_metrics["val/variable_window_target_full_fraction"].item())
            if "val/variable_window_target_full_fraction" in trainer.callback_metrics
            else None
        ),
        "val_variable_window_expected_tokens_last": (
            float(trainer.callback_metrics["val/variable_window_expected_tokens"].item())
            if "val/variable_window_expected_tokens" in trainer.callback_metrics
            else None
        ),
        "val_variable_window_soft_expected_tokens_last": (
            float(trainer.callback_metrics["val/variable_window_soft_expected_tokens"].item())
            if "val/variable_window_soft_expected_tokens" in trainer.callback_metrics
            else None
        ),
        "val_variable_window_soft_entropy_last": (
            float(trainer.callback_metrics["val/variable_window_soft_entropy"].item())
            if "val/variable_window_soft_entropy" in trainer.callback_metrics
            else None
        ),
        "val_variable_window_code_bits_per_chunk_last": (
            float(trainer.callback_metrics["val/variable_window_code_bits_per_chunk"].item())
            if "val/variable_window_code_bits_per_chunk" in trainer.callback_metrics
            else None
        ),
        "val_variable_window_base_deviation_mean_last": (
            float(trainer.callback_metrics["val/variable_window_base_deviation_mean"].item())
            if "val/variable_window_base_deviation_mean" in trainer.callback_metrics
            else None
        ),
        "val_variable_window_oracle_deviation_mean_last": (
            float(trainer.callback_metrics["val/variable_window_oracle_deviation_mean"].item())
            if "val/variable_window_oracle_deviation_mean" in trainer.callback_metrics
            else None
        ),
        "val_variable_window_chosen_deviation_mean_last": (
            float(trainer.callback_metrics["val/variable_window_chosen_deviation_mean"].item())
            if "val/variable_window_chosen_deviation_mean" in trainer.callback_metrics
            else None
        ),
        "val_variable_window_expected_deviation_mean_last": (
            float(trainer.callback_metrics["val/variable_window_expected_deviation_mean"].item())
            if "val/variable_window_expected_deviation_mean" in trainer.callback_metrics
            else None
        ),
        "val_variable_window_regret_mean_last": (
            float(trainer.callback_metrics["val/variable_window_regret_mean"].item())
            if "val/variable_window_regret_mean" in trainer.callback_metrics
            else None
        ),
        "val_variable_window_nonimprove_rate_last": (
            float(trainer.callback_metrics["val/variable_window_nonimprove_rate"].item())
            if "val/variable_window_nonimprove_rate" in trainer.callback_metrics
            else None
        ),
        "val_variable_window_fine_deviation_mean_last": (
            float(trainer.callback_metrics["val/variable_window_fine_deviation_mean"].item())
            if "val/variable_window_fine_deviation_mean" in trainer.callback_metrics
            else None
        ),
        "val_variable_window_medium_deviation_mean_last": (
            float(trainer.callback_metrics["val/variable_window_medium_deviation_mean"].item())
            if "val/variable_window_medium_deviation_mean" in trainer.callback_metrics
            else None
        ),
        "val_variable_window_full_deviation_mean_last": (
            float(trainer.callback_metrics["val/variable_window_full_deviation_mean"].item())
            if "val/variable_window_full_deviation_mean" in trainer.callback_metrics
            else None
        ),
        "val_lookup_token_acc_last": (
            float(trainer.callback_metrics["val/lookup_token_acc"].item())
            if "val/lookup_token_acc" in trainer.callback_metrics
            else None
        ),
        "val_non_lookup_token_acc_last": (
            float(trainer.callback_metrics["val/non_lookup_token_acc"].item())
            if "val/non_lookup_token_acc" in trainer.callback_metrics
            else None
        ),
        "val_lookup_gate_mean_last": (
            float(trainer.callback_metrics["val/lookup_gate_mean"].item())
            if "val/lookup_gate_mean" in trainer.callback_metrics
            else None
        ),
        "val_selector_loss_last": (
            float(trainer.callback_metrics["val/selector_loss"].item())
            if "val/selector_loss" in trainer.callback_metrics
            else None
        ),
        "val_selector_recall_at_k_last": (
            float(trainer.callback_metrics["val/selector_recall_at_k"].item())
            if "val/selector_recall_at_k" in trainer.callback_metrics
            else None
        ),
        "val_selector_precision_at_k_last": (
            float(trainer.callback_metrics["val/selector_precision_at_k"].item())
            if "val/selector_precision_at_k" in trainer.callback_metrics
            else None
        ),
        "val_budget_loss_last": (
            float(trainer.callback_metrics["val/budget_loss"].item())
            if "val/budget_loss" in trainer.callback_metrics
            else None
        ),
        "val_lookup_slot_cost_loss_last": (
            float(trainer.callback_metrics["val/lookup_slot_cost_loss"].item())
            if "val/lookup_slot_cost_loss" in trainer.callback_metrics
            else None
        ),
        "val_lookup_slot_target_loss_last": (
            float(trainer.callback_metrics["val/lookup_slot_target_loss"].item())
            if "val/lookup_slot_target_loss" in trainer.callback_metrics
            else None
        ),
        "val_budget_acc_last": (
            float(trainer.callback_metrics["val/budget_acc"].item())
            if "val/budget_acc" in trainer.callback_metrics
            else None
        ),
        "val_lookup_budget_k_mean_last": (
            float(trainer.callback_metrics["val/lookup_budget_k_mean"].item())
            if "val/lookup_budget_k_mean" in trainer.callback_metrics
            else None
        ),
        "val_lookup_active_k_mean_last": (
            float(trainer.callback_metrics["val/lookup_active_k_mean"].item())
            if "val/lookup_active_k_mean" in trainer.callback_metrics
            else None
        ),
        "val_lookup_keep_prob_mean_last": (
            float(trainer.callback_metrics["val/lookup_keep_prob_mean"].item())
            if "val/lookup_keep_prob_mean" in trainer.callback_metrics
            else None
        ),
        "val_observed_effective_bits_per_token_last": (
            float(trainer.callback_metrics["val/observed_effective_bits_per_token"].item())
            if "val/observed_effective_bits_per_token" in trainer.callback_metrics
            else None
        ),
        "best_checkpoint_path": checkpoint_callback.best_model_path,
    }
    for idx in range(int(getattr(module.model, "hierarchical_num_blocks", 0))):
        token_key = f"val/block{idx}_token_acc"
        exact_key = f"val/block{idx}_exact_acc"
        if token_key in trainer.callback_metrics:
            result[f"val_block{idx}_token_acc_last"] = float(trainer.callback_metrics[token_key].item())
        if exact_key in trainer.callback_metrics:
            result[f"val_block{idx}_exact_acc_last"] = float(trainer.callback_metrics[exact_key].item())
    return result
