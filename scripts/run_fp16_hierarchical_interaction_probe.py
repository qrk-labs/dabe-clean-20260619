#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.run_fp16_interaction_probe import (
    _aggregate_llm_judge,
    _aggregate_probe_quality,
    _coherence_score,
    _judge_rows_with_openrouter,
    _load_fixed_prompt_set,
    _load_probe_prompts,
    _resolve_openrouter_api_key,
    _rubric_score,
    _strict_quality_score,
)
from src.training.fp16_chunk_feasibility import FP16ChunkCompressor, build_causal_mask
from src.training.fp16_hierarchical_feasibility import HierarchicalFP16ChunkLMModule


def _choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_train_config(run_root: Path) -> dict[str, Any]:
    candidates = [
        run_root / "results.json",
        run_root / "stage_result.json",
        run_root / "pipeline_summary.json",
    ]
    for config_path in candidates:
        if not config_path.exists():
            continue
        payload = json.loads(config_path.read_text())
        config = payload.get("config")
        if isinstance(config, dict):
            return config
    raise RuntimeError(
        (
            "Missing config in run root. Expected one of "
            "results.json, stage_result.json, or pipeline_summary.json."
        ),
    )


def _resolve_checkpoint_path(run_root: Path, checkpoint_name: str) -> Path:
    candidate = Path(checkpoint_name)
    if candidate.is_absolute():
        if candidate.exists():
            return candidate
        raise RuntimeError(f"Absolute checkpoint path does not exist: {candidate}")

    roots = [run_root, run_root / "checkpoints"]
    for root in roots:
        path = root / checkpoint_name
        if path.exists():
            return path
    raise RuntimeError(
        (
            f"Checkpoint not found for name={checkpoint_name}. "
            f"Tried {run_root / checkpoint_name} and {run_root / 'checkpoints' / checkpoint_name}."
        ),
    )


def _prepare_context(stream: np.ndarray, seq_len: int) -> np.ndarray:
    if stream.shape[0] == 0:
        return np.zeros((seq_len,), dtype=np.float32)
    if stream.shape[0] >= seq_len:
        return stream[-seq_len:].astype(np.float32, copy=True)
    pad = np.zeros((seq_len - stream.shape[0],), dtype=np.float32)
    return np.concatenate([pad, stream.astype(np.float32, copy=False)], axis=0)


def _generate_continuation_memory_lookup(
    *,
    prompt: str,
    model_module: HierarchicalFP16ChunkLMModule,
    coarse_compressor: FP16ChunkCompressor,
    mid_compressor: FP16ChunkCompressor,
    fine_compressor: FP16ChunkCompressor,
    train_coarse: np.ndarray,
    train_mid: np.ndarray,
    train_fine: np.ndarray,
    train_chunk_tokens: np.ndarray,
    compressed_seq_len: int,
    continuation_chunks: int,
    device: torch.device,
) -> tuple[str, float]:
    prompt_ids = coarse_compressor.tokenizer.encode(prompt, add_special_tokens=False)
    prompt_coarse = coarse_compressor.compress_token_ids(prompt_ids).astype(np.float32, copy=False)
    prompt_mid = mid_compressor.compress_token_ids(prompt_ids).astype(np.float32, copy=False)
    prompt_fine = fine_compressor.compress_token_ids(prompt_ids).astype(np.float32, copy=False)

    ctx_coarse = _prepare_context(prompt_coarse, compressed_seq_len)
    ctx_mid = _prepare_context(prompt_mid, compressed_seq_len)
    ctx_fine = _prepare_context(prompt_fine, compressed_seq_len)

    generated_token_ids: list[int] = []
    start = time.perf_counter()
    for _ in range(continuation_chunks):
        input_coarse = torch.from_numpy(ctx_coarse[None, :]).to(device=device, dtype=torch.float32)
        input_mid = torch.from_numpy(ctx_mid[None, :]).to(device=device, dtype=torch.float32)
        input_fine = torch.from_numpy(ctx_fine[None, :]).to(device=device, dtype=torch.float32)
        mask = build_causal_mask(compressed_seq_len, device)
        with torch.no_grad():
            preds, _, _ = model_module.model(
                input_scalars_coarse=input_coarse,
                input_scalars_mid=input_mid,
                input_scalars_fine=input_fine,
                mask=mask,
            )
        next_coarse = float(preds[0, -1].item())
        nearest_idx = int(np.argmin(np.abs(train_coarse - next_coarse)))
        next_mid = float(train_mid[nearest_idx])
        next_fine = float(train_fine[nearest_idx])
        chunk_ids = train_chunk_tokens[nearest_idx].astype(np.int64, copy=False).tolist()
        generated_token_ids.extend(chunk_ids)

        ctx_coarse = np.concatenate([ctx_coarse[1:], np.asarray([next_coarse], dtype=np.float32)], axis=0)
        ctx_mid = np.concatenate([ctx_mid[1:], np.asarray([next_mid], dtype=np.float32)], axis=0)
        ctx_fine = np.concatenate([ctx_fine[1:], np.asarray([next_fine], dtype=np.float32)], axis=0)

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    continuation = coarse_compressor.tokenizer.decode(
        generated_token_ids,
        clean_up_tokenization_spaces=False,
    ).strip()
    return continuation, elapsed_ms


def _generate_continuation_mirror_decoder(
    *,
    prompt: str,
    model_module: HierarchicalFP16ChunkLMModule,
    coarse_compressor: FP16ChunkCompressor,
    mid_compressor: FP16ChunkCompressor,
    fine_compressor: FP16ChunkCompressor,
    compressed_seq_len: int,
    continuation_chunks: int,
    device: torch.device,
) -> tuple[str, float]:
    prompt_ids = coarse_compressor.tokenizer.encode(prompt, add_special_tokens=False)
    prompt_coarse = coarse_compressor.compress_token_ids(prompt_ids).astype(np.float32, copy=False)
    prompt_mid = mid_compressor.compress_token_ids(prompt_ids).astype(np.float32, copy=False)
    prompt_fine = fine_compressor.compress_token_ids(prompt_ids).astype(np.float32, copy=False)

    ctx_coarse = _prepare_context(prompt_coarse, compressed_seq_len)
    ctx_mid = _prepare_context(prompt_mid, compressed_seq_len)
    ctx_fine = _prepare_context(prompt_fine, compressed_seq_len)

    generated_token_ids: list[int] = []
    start = time.perf_counter()
    for _ in range(continuation_chunks):
        input_coarse = torch.from_numpy(ctx_coarse[None, :]).to(device=device, dtype=torch.float32)
        input_mid = torch.from_numpy(ctx_mid[None, :]).to(device=device, dtype=torch.float32)
        input_fine = torch.from_numpy(ctx_fine[None, :]).to(device=device, dtype=torch.float32)
        mask = build_causal_mask(compressed_seq_len, device)
        with torch.no_grad():
            preds, hidden, _ = model_module.model(
                input_scalars_coarse=input_coarse,
                input_scalars_mid=input_mid,
                input_scalars_fine=input_fine,
                mask=mask,
            )
            hidden_last = hidden[:, -1, :]
            chunk_ids = model_module.decode_chunk_ids_from_hidden(hidden_last)[0]
            next_coarse = float(preds[0, -1].item())
            if getattr(model_module, "decode_mirror_enabled", False):
                next_mid = float(model_module.mid_scalar_head(hidden_last).squeeze(-1).item())
                next_fine = float(model_module.fine_scalar_head(hidden_last).squeeze(-1).item())
            else:
                next_mid = next_coarse
                next_fine = next_coarse
        generated_token_ids.extend(chunk_ids.to(dtype=torch.int64).tolist())
        ctx_coarse = np.concatenate([ctx_coarse[1:], np.asarray([next_coarse], dtype=np.float32)], axis=0)
        ctx_mid = np.concatenate([ctx_mid[1:], np.asarray([next_mid], dtype=np.float32)], axis=0)
        ctx_fine = np.concatenate([ctx_fine[1:], np.asarray([next_fine], dtype=np.float32)], axis=0)

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    continuation = coarse_compressor.tokenizer.decode(
        generated_token_ids,
        clean_up_tokenization_spaces=False,
    ).strip()
    return continuation, elapsed_ms


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--continuation-chunks", type=int, default=2)
    parser.add_argument("--skip-rows", type=int, default=-1)
    parser.add_argument("--prompt-set-file", default="")
    parser.add_argument("--output-name", default="interaction_probe_hierarchical.json")
    parser.add_argument(
        "--decode-mode",
        choices=("mirror_decoder", "memory_lookup"),
        default="mirror_decoder",
    )
    parser.add_argument("--allow-memory-lookup", action="store_true")
    parser.add_argument("--memory-lookup-diagnostic", action="store_true")
    parser.add_argument("--openrouter-judge", action="store_true")
    parser.add_argument("--openrouter-model", default="deepseek/deepseek-v4-flash")
    parser.add_argument("--openrouter-api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--opencode-config-path", default="~/.config/opencode/opencode.json")
    parser.add_argument("--judge-chunk-size", type=int, default=8)
    parser.add_argument("--judge-timeout-seconds", type=int, default=60)
    parser.add_argument("--judge-max-retries", type=int, default=5)
    parser.add_argument("--judge-retry-backoff-seconds", type=float, default=1.5)
    parser.add_argument("--judge-inter-chunk-sleep-seconds", type=float, default=1.0)
    return parser.parse_args()


def _validate_scored_decode_contract(
    *,
    decode_mode: str,
    allow_memory_lookup: bool,
    memory_lookup_diagnostic: bool,
) -> None:
    if decode_mode != "memory_lookup":
        return
    if not bool(allow_memory_lookup):
        raise RuntimeError(
            "memory_lookup decode mode is disabled by default due train/eval leakage risk. "
            "Re-run with --allow-memory-lookup to use it explicitly.",
        )
    if not bool(memory_lookup_diagnostic):
        raise RuntimeError(
            "Scored evaluation with memory_lookup is disallowed. "
            "Use mirror_decoder for scored comparisons, or pass --memory-lookup-diagnostic "
            "for diagnostic-only runs.",
        )


def main() -> None:
    args = _parse_args()
    run_root = Path(args.run_root)
    config = _load_train_config(run_root)
    checkpoint_path = _resolve_checkpoint_path(run_root, args.checkpoint_name)
    output_path = run_root / args.output_name

    fp16_cfg = dict(config.get("fp16_chunk", {}))
    dataset_cfg = dict(fp16_cfg.get("dataset", {}))
    compression_cfg = dict(fp16_cfg.get("compression", {}))
    training_cfg = dict(fp16_cfg.get("training", {}))
    hierarchy_cfg = dict(fp16_cfg.get("hierarchical", {}))
    if not bool(hierarchy_cfg.get("enabled", False)):
        raise RuntimeError("Config does not enable hierarchical mode.")

    coarse_chunk_size = int(hierarchy_cfg.get("coarse_chunk_size", compression_cfg.get("chunk_size_tokens", 64)))
    mid_chunk_size = int(hierarchy_cfg.get("mid_chunk_size", 8))
    fine_chunk_size = int(hierarchy_cfg.get("fine_chunk_size", 1))

    tokenizer_name = str(compression_cfg.get("tokenizer_name", "gpt2"))
    force_fast = bool(compression_cfg.get("force_fast_tokenizer", True))
    dtype = str(compression_cfg.get("dtype", "float16"))
    coarse_compressor = FP16ChunkCompressor(
        chunk_size_tokens=coarse_chunk_size,
        window_overlap_tokens=0,
        tokenizer_name=tokenizer_name,
        force_fast_tokenizer=force_fast,
        dtype=dtype,
    )
    shared_tokenizer = coarse_compressor.tokenizer
    mid_compressor = FP16ChunkCompressor(
        chunk_size_tokens=mid_chunk_size,
        window_overlap_tokens=0,
        tokenizer=shared_tokenizer,
        force_fast_tokenizer=False,
        dtype=dtype,
    )
    fine_compressor = FP16ChunkCompressor(
        chunk_size_tokens=fine_chunk_size,
        window_overlap_tokens=0,
        tokenizer=shared_tokenizer,
        force_fast_tokenizer=False,
        dtype=dtype,
    )

    if str(args.prompt_set_file).strip():
        prompts = _load_fixed_prompt_set(
            prompt_set_file=Path(args.prompt_set_file),
            num_prompts=int(args.num_prompts),
        )
    else:
        prompts = _load_probe_prompts(
            dataset_name=str(dataset_cfg.get("dataset_name", "roneneldan/TinyStories")),
            split=str(dataset_cfg.get("val_split", "validation")),
            text_field=str(dataset_cfg.get("text_field", "text")),
            tokenizer=shared_tokenizer,
            num_prompts=int(args.num_prompts),
            skip_rows=(
                int(args.skip_rows)
                if int(args.skip_rows) >= 0
                else int(dataset_cfg.get("val_samples", 2048))
            ),
            prompt_tokens=int(args.prompt_tokens),
        )

    train_coarse = np.empty((0,), dtype=np.float32)
    train_mid = np.empty((0,), dtype=np.float32)
    train_fine = np.empty((0,), dtype=np.float32)
    train_chunk_tokens = np.empty((0, coarse_chunk_size), dtype=np.int32)
    _validate_scored_decode_contract(
        decode_mode=str(args.decode_mode),
        allow_memory_lookup=bool(args.allow_memory_lookup),
        memory_lookup_diagnostic=bool(args.memory_lookup_diagnostic),
    )
    if args.decode_mode == "memory_lookup":
        train_coarse_path = run_root / "train_scalars_coarse.npy"
        train_mid_path = run_root / "train_scalars_mid.npy"
        train_fine_path = run_root / "train_scalars_fine.npy"
        train_chunks_path = run_root / "train_chunk_tokens.npy"
        if not (
            train_coarse_path.exists()
            and train_mid_path.exists()
            and train_fine_path.exists()
            and train_chunks_path.exists()
        ):
            raise RuntimeError(
                "memory_lookup mode requires saved train streams/chunks. "
                "Expected train_scalars_{coarse,mid,fine}.npy and train_chunk_tokens.npy in run root.",
            )
        train_coarse = np.load(train_coarse_path, mmap_mode="r").astype(np.float32, copy=False)
        train_mid = np.load(train_mid_path, mmap_mode="r").astype(np.float32, copy=False)
        train_fine = np.load(train_fine_path, mmap_mode="r").astype(np.float32, copy=False)
        train_chunk_tokens = np.load(train_chunks_path, mmap_mode="r")

    device = _choose_device()
    model_config = {
        **fp16_cfg,
        "diffusion": {
            **dict(fp16_cfg.get("diffusion", {})),
            "vocab_size": int(fp16_cfg.get("diffusion", {}).get("vocab_size", coarse_compressor.vocab_size)),
        },
    }
    model_module = HierarchicalFP16ChunkLMModule(config=model_config).to(device)
    model_module.eval()
    state = torch.load(checkpoint_path, map_location=device)
    model_module.load_state_dict(state["state_dict"], strict=True)
    if args.decode_mode == "mirror_decoder" and not bool(getattr(model_module, "decode_mirror_enabled", False)):
        raise RuntimeError(
            "Checkpoint/config does not enable decode_mirror. "
            "Use a mirror-decoder checkpoint or run with --decode-mode memory_lookup --allow-memory-lookup.",
        )

    compressed_seq_len = int(training_cfg.get("compressed_seq_len", 8))
    continuation_chunks = int(args.continuation_chunks)

    rows: list[dict[str, Any]] = []
    coherent_count = 0
    strict_coherent_count = 0
    latencies: list[float] = []
    for idx, prompt in enumerate(prompts):
        if args.decode_mode == "memory_lookup":
            continuation, latency_ms = _generate_continuation_memory_lookup(
                prompt=prompt,
                model_module=model_module,
                coarse_compressor=coarse_compressor,
                mid_compressor=mid_compressor,
                fine_compressor=fine_compressor,
                train_coarse=train_coarse,
                train_mid=train_mid,
                train_fine=train_fine,
                train_chunk_tokens=train_chunk_tokens,
                compressed_seq_len=compressed_seq_len,
                continuation_chunks=continuation_chunks,
                device=device,
            )
        else:
            continuation, latency_ms = _generate_continuation_mirror_decoder(
                prompt=prompt,
                model_module=model_module,
                coarse_compressor=coarse_compressor,
                mid_compressor=mid_compressor,
                fine_compressor=fine_compressor,
                compressed_seq_len=compressed_seq_len,
                continuation_chunks=continuation_chunks,
                device=device,
            )
        coherent, stats = _coherence_score(prompt, continuation)
        strict_coherent, strict_stats = _strict_quality_score(
            prompt=prompt,
            continuation=continuation,
            base_coherent=coherent,
            base_stats=stats,
        )
        rubric_pass, rubric = _rubric_score(
            prompt=prompt,
            continuation=continuation,
            base_stats=stats,
            strict_stats=strict_stats,
        )
        coherent_count += int(coherent)
        strict_coherent_count += int(strict_coherent)
        latencies.append(latency_ms)
        rows.append(
            {
                "idx": idx,
                "prompt": prompt,
                "continuation": continuation,
                "coherent": coherent,
                "strict_coherent": strict_coherent,
                "rubric_pass": rubric_pass,
                "latency_ms": latency_ms,
                "stats": stats,
                "strict_stats": strict_stats,
                "rubric": rubric,
            }
        )

    aggregate = _aggregate_probe_quality(rows)
    llm_aggregate = {
        "llm_rubric_avg_score": 0.0,
        "llm_rubric_pass_rate": 0.0,
        "llm_rubric_format_avg": 0.0,
        "llm_rubric_relevance_avg": 0.0,
        "llm_rubric_fluency_avg": 0.0,
        "llm_rubric_diversity_avg": 0.0,
    }
    llm_meta = {"llm_judge_requests": 0.0, "llm_judge_fallback_rows": 0.0}
    llm_judge_error = ""
    if bool(args.openrouter_judge):
        try:
            openrouter_api_key = _resolve_openrouter_api_key(
                opencode_config_path=Path(str(args.opencode_config_path)).expanduser(),
                primary_env_name=str(args.openrouter_api_key_env),
            )
            llm_meta = _judge_rows_with_openrouter(
                rows=rows,
                api_key=openrouter_api_key,
                model=str(args.openrouter_model),
                chunk_size=int(args.judge_chunk_size),
                timeout_seconds=int(args.judge_timeout_seconds),
                max_retries=int(args.judge_max_retries),
                retry_backoff_seconds=float(args.judge_retry_backoff_seconds),
                inter_chunk_sleep_seconds=float(args.judge_inter_chunk_sleep_seconds),
            )
            llm_aggregate = _aggregate_llm_judge(rows)
        except Exception as err:
            llm_judge_error = str(err)

    coherent_rate = float(coherent_count) / float(len(rows) or 1)
    strict_coherent_rate = float(strict_coherent_count) / float(len(rows) or 1)
    avg_latency_ms = float(np.mean(latencies)) if latencies else 0.0
    payload = {
        "run_id": run_root.name,
        "num_prompts": len(rows),
        "coherent_count": coherent_count,
        "coherent_rate": coherent_rate,
        "strict_coherent_count": strict_coherent_count,
        "strict_coherent_rate": strict_coherent_rate,
        "avg_decode_latency_ms": avg_latency_ms,
        "device": str(device),
        "continuation_chunks": continuation_chunks,
        "decode_mode": str(args.decode_mode),
        "uses_train_memory_lookup": bool(args.decode_mode == "memory_lookup"),
        "memory_lookup_diagnostic": bool(args.memory_lookup_diagnostic),
        "unique_continuations": int(aggregate["unique_continuations"]),
        "dominant_continuation_share": aggregate["dominant_continuation_share"],
        "mojibake_rows": int(aggregate["mojibake_rows"]),
        "strict_gate_pass": bool(int(aggregate["strict_gate_pass"])),
        "rubric_avg_score": aggregate["rubric_avg_score"],
        "rubric_pass_rate": aggregate["rubric_pass_rate"],
        "rubric_format_avg": aggregate["rubric_format_avg"],
        "rubric_relevance_avg": aggregate["rubric_relevance_avg"],
        "rubric_fluency_avg": aggregate["rubric_fluency_avg"],
        "rubric_diversity_avg": aggregate["rubric_diversity_avg"],
        "llm_judge_enabled": bool(args.openrouter_judge),
        "llm_judge_model": str(args.openrouter_model) if bool(args.openrouter_judge) else "",
        "llm_judge_chunk_size": int(args.judge_chunk_size),
        "llm_judge_requests": int(llm_meta["llm_judge_requests"]),
        "llm_judge_fallback_rows": int(llm_meta["llm_judge_fallback_rows"]),
        "llm_judge_error": llm_judge_error,
        "llm_rubric_avg_score": llm_aggregate["llm_rubric_avg_score"],
        "llm_rubric_pass_rate": llm_aggregate["llm_rubric_pass_rate"],
        "llm_rubric_format_avg": llm_aggregate["llm_rubric_format_avg"],
        "llm_rubric_relevance_avg": llm_aggregate["llm_rubric_relevance_avg"],
        "llm_rubric_fluency_avg": llm_aggregate["llm_rubric_fluency_avg"],
        "llm_rubric_diversity_avg": llm_aggregate["llm_rubric_diversity_avg"],
        "rows": rows,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in payload if k != "rows"}, indent=2))
    print(f"Saved probe: {output_path}")


if __name__ == "__main__":
    main()
