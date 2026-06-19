from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.training.fp16_chunk_feasibility import FP16ChunkCompressor, build_causal_mask
from src.training.fp16_hierarchical_feasibility import HierarchicalFP16ChunkLMModule


def resolve_device(preferred: str = "auto") -> torch.device:
    value = str(preferred).strip().lower()
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if value == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if value == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _prepare_context(stream: np.ndarray, seq_len: int) -> np.ndarray:
    if stream.shape[0] == 0:
        return np.zeros((seq_len,), dtype=np.float32)
    if stream.shape[0] >= seq_len:
        return stream[-seq_len:].astype(np.float32, copy=True)
    pad = np.zeros((seq_len - stream.shape[0],), dtype=np.float32)
    return np.concatenate([pad, stream.astype(np.float32, copy=False)], axis=0)


@dataclass
class HierarchicalGenerationConfig:
    compressed_seq_len: int
    continuation_chunks: int
    decode_mode: str = "mirror_decoder"  # mirror_decoder|memory_lookup
    runtime_mode: str = "quality"  # quality|fast
    fast_decode_interval: int = 2
    fast_refine_tokens: int = 16
    repetition_ngram: int = 4
    repetition_fraction_threshold: float = 0.35
    ttt_enabled: bool = False
    ttt_steps: int = 8
    ttt_lr: float = 1e-3
    ttt_batch_size: int = 8
    ttt_max_windows: int = 32
    ttt_position_stride: int = 8
    ttt_max_positions: int = 8
    ttt_scalar_loss_weight: float = 0.05
    ttt_next_step_targets: bool = True
    ttt_grad_clip: float = 1.0


@dataclass
class GenerationOutput:
    continuation: str
    latency_ms: float
    generated_token_ids: list[int]
    chunk_repeat_fraction: float
    collapse_detected: bool
    ttt_applied: bool = False
    ttt_steps_run: int = 0
    ttt_last_loss: float = 0.0


class HierarchicalGenerator:
    def __init__(
        self,
        *,
        model_module: HierarchicalFP16ChunkLMModule,
        coarse_compressor: FP16ChunkCompressor,
        mid_compressor: FP16ChunkCompressor,
        fine_compressor: FP16ChunkCompressor,
        config: HierarchicalGenerationConfig,
        device: torch.device,
        train_coarse: np.ndarray | None = None,
        train_mid: np.ndarray | None = None,
        train_fine: np.ndarray | None = None,
        train_chunk_tokens: np.ndarray | None = None,
    ):
        self.model_module = model_module
        self.coarse_compressor = coarse_compressor
        self.mid_compressor = mid_compressor
        self.fine_compressor = fine_compressor
        self.config = config
        self.device = device
        self.train_coarse = train_coarse
        self.train_mid = train_mid
        self.train_fine = train_fine
        self.train_chunk_tokens = train_chunk_tokens

        self.chunk_size_tokens = int(getattr(self.model_module, "chunk_size_tokens", 64))
        self.fast_decode_interval = max(1, int(self.config.fast_decode_interval))
        self.fast_refine_tokens = max(0, int(self.config.fast_refine_tokens))
        self.ttt_steps = max(1, int(self.config.ttt_steps))
        self.ttt_batch_size = max(1, int(self.config.ttt_batch_size))
        self.ttt_max_windows = max(1, int(self.config.ttt_max_windows))
        self.ttt_position_stride = max(1, int(self.config.ttt_position_stride))
        self.ttt_max_positions = max(1, int(self.config.ttt_max_positions))
        self.ttt_scalar_loss_weight = max(0.0, float(self.config.ttt_scalar_loss_weight))
        self.ttt_grad_clip = max(0.0, float(self.config.ttt_grad_clip))

        if str(self.config.decode_mode) == "memory_lookup":
            if (
                self.train_coarse is None
                or self.train_mid is None
                or self.train_fine is None
                or self.train_chunk_tokens is None
            ):
                raise RuntimeError("memory_lookup mode requires preloaded train streams and chunk tokens.")
            if self.train_coarse.shape[0] == 0:
                raise RuntimeError("memory_lookup mode requires non-empty train streams.")
        elif str(self.config.decode_mode) != "mirror_decoder":
            raise ValueError(f"Unknown decode_mode: {self.config.decode_mode}")

        if str(self.config.runtime_mode) not in {"quality", "fast"}:
            raise ValueError(f"Unknown runtime_mode: {self.config.runtime_mode}")

    def _align_scale_stream(
        self,
        *,
        scale_stream: np.ndarray,
        coarse_count: int,
        coarse_fallback: np.ndarray,
        factor: int,
    ) -> np.ndarray:
        usable = int(coarse_count) * int(factor)
        if usable <= 0 or scale_stream.shape[0] < usable:
            return coarse_fallback.astype(np.float32, copy=True)
        trimmed = scale_stream[:usable].reshape(coarse_count, factor)
        return trimmed.mean(axis=1).astype(np.float32, copy=False)

    def _build_prompt_hierarchical_streams(
        self,
        prompt_ids: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        coarse_scalars, coarse_chunks = self.coarse_compressor.compress_token_ids_with_chunks(prompt_ids)
        coarse_scalars = coarse_scalars.astype(np.float32, copy=False)
        coarse_chunks = coarse_chunks.astype(np.int64, copy=False)
        if coarse_scalars.shape[0] == 0:
            empty_f = np.empty((0,), dtype=np.float32)
            empty_i = np.empty((0, self.chunk_size_tokens), dtype=np.int64)
            return empty_f, empty_f, empty_f, empty_i

        mid_scalars = self.mid_compressor.compress_token_ids(prompt_ids).astype(np.float32, copy=False)
        fine_scalars = self.fine_compressor.compress_token_ids(prompt_ids).astype(np.float32, copy=False)
        coarse_count = int(coarse_scalars.shape[0])
        mid_factor = max(1, self.chunk_size_tokens // max(1, int(self.mid_compressor.chunk_size_tokens)))
        fine_factor = max(1, self.chunk_size_tokens // max(1, int(self.fine_compressor.chunk_size_tokens)))
        mid_aligned = self._align_scale_stream(
            scale_stream=mid_scalars,
            coarse_count=coarse_count,
            coarse_fallback=coarse_scalars,
            factor=mid_factor,
        )
        fine_aligned = self._align_scale_stream(
            scale_stream=fine_scalars,
            coarse_count=coarse_count,
            coarse_fallback=coarse_scalars,
            factor=fine_factor,
        )
        return coarse_scalars, mid_aligned, fine_aligned, coarse_chunks

    def _build_ttt_windows(
        self,
        *,
        coarse_stream: np.ndarray,
        mid_stream: np.ndarray,
        fine_stream: np.ndarray,
        chunk_tokens: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        seq_len = int(self.config.compressed_seq_len)
        coarse_count = int(coarse_stream.shape[0])
        if coarse_count == 0:
            return (
                np.empty((0, seq_len), dtype=np.float32),
                np.empty((0, seq_len), dtype=np.float32),
                np.empty((0, seq_len), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0, self.chunk_size_tokens), dtype=np.int64),
            )

        rows_coarse: list[np.ndarray] = []
        rows_mid: list[np.ndarray] = []
        rows_fine: list[np.ndarray] = []
        tgt_coarse: list[float] = []
        tgt_mid: list[float] = []
        tgt_fine: list[float] = []
        tgt_chunks: list[np.ndarray] = []

        use_next = bool(self.config.ttt_next_step_targets) and coarse_count >= 2
        start_index = 1 if use_next else 0
        end_index = coarse_count
        for target_idx in range(start_index, end_index):
            if use_next:
                source_end = target_idx
            else:
                source_end = target_idx + 1
            ctx_coarse = _prepare_context(coarse_stream[:source_end], seq_len)
            ctx_mid = _prepare_context(mid_stream[:source_end], seq_len)
            ctx_fine = _prepare_context(fine_stream[:source_end], seq_len)
            rows_coarse.append(ctx_coarse)
            rows_mid.append(ctx_mid)
            rows_fine.append(ctx_fine)
            tgt_coarse.append(float(coarse_stream[target_idx]))
            tgt_mid.append(float(mid_stream[target_idx]))
            tgt_fine.append(float(fine_stream[target_idx]))
            tgt_chunks.append(chunk_tokens[target_idx].astype(np.int64, copy=False))

        if not rows_coarse:
            return (
                np.empty((0, seq_len), dtype=np.float32),
                np.empty((0, seq_len), dtype=np.float32),
                np.empty((0, seq_len), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0, self.chunk_size_tokens), dtype=np.int64),
            )

        keep = min(len(rows_coarse), self.ttt_max_windows)
        rows_coarse = rows_coarse[-keep:]
        rows_mid = rows_mid[-keep:]
        rows_fine = rows_fine[-keep:]
        tgt_coarse = tgt_coarse[-keep:]
        tgt_mid = tgt_mid[-keep:]
        tgt_fine = tgt_fine[-keep:]
        tgt_chunks = tgt_chunks[-keep:]
        return (
            np.asarray(rows_coarse, dtype=np.float32),
            np.asarray(rows_mid, dtype=np.float32),
            np.asarray(rows_fine, dtype=np.float32),
            np.asarray(tgt_coarse, dtype=np.float32),
            np.asarray(tgt_mid, dtype=np.float32),
            np.asarray(tgt_fine, dtype=np.float32),
            np.asarray(tgt_chunks, dtype=np.int64),
        )

    def _ttt_trainable_parameters(self) -> list[torch.nn.Parameter]:
        if not getattr(self.model_module, "decode_mirror_enabled", False):
            return []
        names = (
            "decode_mirror_coarse_head",
            "decode_mirror_mid_head",
            "decode_mirror_fine_head",
            "mid_scalar_head",
            "fine_scalar_head",
        )
        params: list[torch.nn.Parameter] = []
        for name in names:
            module = getattr(self.model_module, name, None)
            if module is None:
                continue
            params.extend(list(module.parameters()))
        return [param for param in params if param.requires_grad]

    def _snapshot_parameters(self, parameters: list[torch.nn.Parameter]) -> list[torch.Tensor]:
        return [param.detach().clone() for param in parameters]

    def _restore_parameters(
        self,
        *,
        parameters: list[torch.nn.Parameter],
        snapshot: list[torch.Tensor],
    ) -> None:
        for param, tensor in zip(parameters, snapshot):
            param.data.copy_(tensor)

    def _run_ttt(
        self,
        *,
        prompt_ids: list[int],
        parameters: list[torch.nn.Parameter],
    ) -> tuple[bool, int, float]:
        if not bool(self.config.ttt_enabled):
            return False, 0, 0.0
        if str(self.config.decode_mode) != "mirror_decoder":
            return False, 0, 0.0
        if not parameters:
            return False, 0, 0.0
        if not hasattr(self.model_module, "_project_tier_embeddings") or not hasattr(self.model_module, "_fuse_tier_embeddings"):
            return False, 0, 0.0

        coarse_stream, mid_stream, fine_stream, chunk_tokens = self._build_prompt_hierarchical_streams(prompt_ids)
        (
            windows_coarse,
            windows_mid,
            windows_fine,
            target_coarse,
            target_mid,
            target_fine,
            target_chunks,
        ) = self._build_ttt_windows(
            coarse_stream=coarse_stream,
            mid_stream=mid_stream,
            fine_stream=fine_stream,
            chunk_tokens=chunk_tokens,
        )
        if windows_coarse.shape[0] == 0:
            return False, 0, 0.0

        optimizer = torch.optim.AdamW(parameters, lr=float(self.config.ttt_lr), weight_decay=0.0)
        seq_len = int(self.config.compressed_seq_len)
        position_idx = list(range(0, self.chunk_size_tokens, self.ttt_position_stride))
        if not position_idx:
            position_idx = [0]
        position_idx = position_idx[: self.ttt_max_positions]
        pos_idx = torch.tensor(position_idx, device=self.device, dtype=torch.long)

        last_loss = 0.0
        steps_run = 0
        was_training = self.model_module.training
        self.model_module.eval()
        with torch.enable_grad():
            for _ in range(self.ttt_steps):
                sample_size = min(self.ttt_batch_size, windows_coarse.shape[0])
                sample_idx = np.random.choice(windows_coarse.shape[0], size=sample_size, replace=False)
                in_coarse = torch.from_numpy(windows_coarse[sample_idx]).to(device=self.device, dtype=torch.float32)
                in_mid = torch.from_numpy(windows_mid[sample_idx]).to(device=self.device, dtype=torch.float32)
                in_fine = torch.from_numpy(windows_fine[sample_idx]).to(device=self.device, dtype=torch.float32)
                tgt_coarse = torch.from_numpy(target_coarse[sample_idx]).to(device=self.device, dtype=torch.float32)
                tgt_mid = torch.from_numpy(target_mid[sample_idx]).to(device=self.device, dtype=torch.float32)
                tgt_fine = torch.from_numpy(target_fine[sample_idx]).to(device=self.device, dtype=torch.float32)
                tgt_chunks = torch.from_numpy(target_chunks[sample_idx]).to(device=self.device, dtype=torch.long)

                mask = build_causal_mask(seq_len, self.device)
                preds, hidden, _ = self.model_module.model(
                    input_scalars_coarse=in_coarse,
                    input_scalars_mid=in_mid,
                    input_scalars_fine=in_fine,
                    mask=mask,
                )
                hidden_last = hidden[:, -1, :]
                context = hidden_last.unsqueeze(1)
                pred_coarse, pred_mid, pred_fine = self.model_module._project_tier_embeddings(context)
                fused = self.model_module._fuse_tier_embeddings(
                    pred_coarse=pred_coarse[:, 0, :, :],
                    pred_mid=pred_mid[:, 0, :, :],
                    pred_fine=pred_fine[:, 0, :, :],
                )

                pred_sel = fused.index_select(dim=1, index=pos_idx)
                tgt_sel = tgt_chunks.index_select(dim=1, index=pos_idx)
                pred_flat = torch.nn.functional.normalize(pred_sel.reshape(-1, pred_sel.shape[-1]), dim=-1)
                tgt_flat = tgt_sel.reshape(-1)
                vocab_embed = torch.nn.functional.normalize(self.model_module.chunk_token_embed.weight, dim=-1)
                logits = torch.matmul(pred_flat, vocab_embed.transpose(0, 1))
                ce_loss = torch.nn.functional.cross_entropy(logits, tgt_flat)

                scalar_loss = torch.nn.functional.smooth_l1_loss(preds[:, -1], tgt_coarse)
                if hasattr(self.model_module, "mid_scalar_head") and hasattr(self.model_module, "fine_scalar_head"):
                    pred_mid_scalar = self.model_module.mid_scalar_head(hidden_last).squeeze(-1)
                    pred_fine_scalar = self.model_module.fine_scalar_head(hidden_last).squeeze(-1)
                    scalar_loss = scalar_loss + torch.nn.functional.smooth_l1_loss(pred_mid_scalar, tgt_mid)
                    scalar_loss = scalar_loss + torch.nn.functional.smooth_l1_loss(pred_fine_scalar, tgt_fine)

                loss = ce_loss + self.ttt_scalar_loss_weight * scalar_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.ttt_grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(parameters, self.ttt_grad_clip)
                optimizer.step()
                last_loss = float(loss.item())
                steps_run += 1

        if was_training:
            self.model_module.train()
        else:
            self.model_module.eval()
        return True, steps_run, last_loss

    def _decode_memory_lookup(self, next_coarse: float) -> tuple[list[int], float, float]:
        nearest_idx = int(np.argmin(np.abs(self.train_coarse - next_coarse)))
        next_mid = float(self.train_mid[nearest_idx])
        next_fine = float(self.train_fine[nearest_idx])
        chunk_ids = self.train_chunk_tokens[nearest_idx].astype(np.int64, copy=False).tolist()
        return chunk_ids, next_mid, next_fine

    def _decode_mirror_quality(self, hidden_last: torch.Tensor) -> list[int]:
        chunk_ids = self.model_module.decode_chunk_ids_from_hidden(hidden_last)[0]
        return chunk_ids.to(dtype=torch.int64).tolist()

    def _decode_mirror_fast(
        self,
        *,
        hidden_last: torch.Tensor,
        step_idx: int,
        cached_chunk: torch.Tensor | None,
    ) -> tuple[list[int], torch.Tensor]:
        should_refresh = cached_chunk is None or (step_idx % self.fast_decode_interval == 0)
        if not should_refresh:
            return cached_chunk.to(dtype=torch.int64).tolist(), cached_chunk

        refreshed = self.model_module.decode_chunk_ids_from_hidden(hidden_last)[0].to(dtype=torch.int64)
        if cached_chunk is None:
            merged = refreshed
        else:
            merged = cached_chunk.clone()
            refine = min(self.chunk_size_tokens, self.fast_refine_tokens)
            if refine > 0:
                merged[:refine] = refreshed[:refine]
        return merged.tolist(), merged

    def _chunk_repeat_fraction(self, chunk_history: list[tuple[int, ...]]) -> float:
        if not chunk_history:
            return 0.0
        counts: dict[tuple[int, ...], int] = {}
        max_count = 0
        for key in chunk_history:
            value = counts.get(key, 0) + 1
            counts[key] = value
            if value > max_count:
                max_count = value
        return float(max_count) / float(len(chunk_history))

    def generate(self, prompt: str) -> GenerationOutput:
        prompt_ids = self.coarse_compressor.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_coarse = self.coarse_compressor.compress_token_ids(prompt_ids).astype(np.float32, copy=False)
        prompt_mid = self.mid_compressor.compress_token_ids(prompt_ids).astype(np.float32, copy=False)
        prompt_fine = self.fine_compressor.compress_token_ids(prompt_ids).astype(np.float32, copy=False)

        seq_len = int(self.config.compressed_seq_len)
        ctx_coarse = _prepare_context(prompt_coarse, seq_len)
        ctx_mid = _prepare_context(prompt_mid, seq_len)
        ctx_fine = _prepare_context(prompt_fine, seq_len)

        generated_token_ids: list[int] = []
        chunk_history: list[tuple[int, ...]] = []
        cached_chunk: torch.Tensor | None = None
        start = time.perf_counter()
        ttt_applied = False
        ttt_steps_run = 0
        ttt_last_loss = 0.0
        ttt_params = self._ttt_trainable_parameters()
        ttt_snapshot: list[torch.Tensor] = []
        if bool(self.config.ttt_enabled) and ttt_params:
            ttt_snapshot = self._snapshot_parameters(ttt_params)
            try:
                ttt_applied, ttt_steps_run, ttt_last_loss = self._run_ttt(
                    prompt_ids=prompt_ids,
                    parameters=ttt_params,
                )
            except Exception:
                self._restore_parameters(parameters=ttt_params, snapshot=ttt_snapshot)
                raise

        try:
            for step_idx in range(int(self.config.continuation_chunks)):
                input_coarse = torch.from_numpy(ctx_coarse[None, :]).to(device=self.device, dtype=torch.float32)
                input_mid = torch.from_numpy(ctx_mid[None, :]).to(device=self.device, dtype=torch.float32)
                input_fine = torch.from_numpy(ctx_fine[None, :]).to(device=self.device, dtype=torch.float32)
                mask = build_causal_mask(seq_len, self.device)

                with torch.no_grad():
                    preds, hidden, _ = self.model_module.model(
                        input_scalars_coarse=input_coarse,
                        input_scalars_mid=input_mid,
                        input_scalars_fine=input_fine,
                        mask=mask,
                    )
                    hidden_last = hidden[:, -1, :]
                    next_coarse = float(preds[0, -1].item())

                    if str(self.config.decode_mode) == "memory_lookup":
                        chunk_ids, next_mid, next_fine = self._decode_memory_lookup(next_coarse)
                    else:
                        if str(self.config.runtime_mode) == "fast":
                            chunk_ids, cached_chunk = self._decode_mirror_fast(
                                hidden_last=hidden_last,
                                step_idx=step_idx,
                                cached_chunk=cached_chunk,
                            )
                        else:
                            chunk_ids = self._decode_mirror_quality(hidden_last)

                        if getattr(self.model_module, "decode_mirror_enabled", False):
                            next_mid = float(self.model_module.mid_scalar_head(hidden_last).squeeze(-1).item())
                            next_fine = float(self.model_module.fine_scalar_head(hidden_last).squeeze(-1).item())
                        else:
                            next_mid = next_coarse
                            next_fine = next_coarse

                generated_token_ids.extend(chunk_ids)
                chunk_history.append(tuple(int(token_id) for token_id in chunk_ids))

                ctx_coarse = np.concatenate([ctx_coarse[1:], np.asarray([next_coarse], dtype=np.float32)], axis=0)
                ctx_mid = np.concatenate([ctx_mid[1:], np.asarray([next_mid], dtype=np.float32)], axis=0)
                ctx_fine = np.concatenate([ctx_fine[1:], np.asarray([next_fine], dtype=np.float32)], axis=0)
        finally:
            if ttt_snapshot and ttt_params:
                self._restore_parameters(parameters=ttt_params, snapshot=ttt_snapshot)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        continuation = self.coarse_compressor.tokenizer.decode(
            generated_token_ids,
            clean_up_tokenization_spaces=False,
        ).strip()

        repeat_fraction = self._chunk_repeat_fraction(chunk_history)
        collapse_detected = bool(repeat_fraction >= float(self.config.repetition_fraction_threshold))
        return GenerationOutput(
            continuation=continuation,
            latency_ms=elapsed_ms,
            generated_token_ids=generated_token_ids,
            chunk_repeat_fraction=repeat_fraction,
            collapse_detected=collapse_detected,
            ttt_applied=ttt_applied,
            ttt_steps_run=ttt_steps_run,
            ttt_last_loss=ttt_last_loss,
        )
