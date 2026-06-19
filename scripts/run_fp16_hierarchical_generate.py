#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from omegaconf import OmegaConf

from src.inference.hierarchical_generation import (
    HierarchicalGenerationConfig,
    HierarchicalGenerator,
    resolve_device,
)
from src.training.fp16_chunk_feasibility import FP16ChunkCompressor, _extract_text_field
from src.training.fp16_hierarchical_feasibility import HierarchicalFP16ChunkLMModule


def _load_train_config(run_root: Path) -> dict[str, Any]:
    candidates = [
        run_root / "results.json",
        run_root / "stage_result.json",
        run_root / "pipeline_summary.json",
        run_root / "config.yaml",
    ]
    for config_path in candidates:
        if not config_path.exists():
            continue
        if config_path.suffix == ".yaml":
            cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
            if isinstance(cfg, dict):
                return cfg
            continue
        payload = json.loads(config_path.read_text())
        config = payload.get("config")
        if isinstance(config, dict):
            return config
    raise RuntimeError(
        "Missing config in run root. Expected results/stage/pipeline JSON or config.yaml.",
    )


def _resolve_checkpoint_path(run_root: Path, checkpoint_name: str) -> Path:
    if checkpoint_name.strip():
        candidate = Path(checkpoint_name)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        roots = [run_root, run_root / "checkpoints"]
        for root in roots:
            path = root / checkpoint_name
            if path.exists():
                return path
        raise RuntimeError(f"Checkpoint not found for checkpoint_name={checkpoint_name}")

    candidates: list[Path] = []
    candidates.extend(sorted((run_root / "checkpoints").glob("best-step-*.ckpt")))
    candidates.extend(sorted(run_root.glob("best-step-*.ckpt")))
    candidates.extend(sorted((run_root / "checkpoints").glob("*.ckpt")))
    candidates.extend(sorted(run_root.glob("*.ckpt")))
    if not candidates:
        raise RuntimeError("No checkpoint files found in run root.")
    return candidates[-1]


def _load_inference_defaults(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(cfg, dict):
        return {}
    section = cfg.get("fp16_hierarchical_inference")
    if isinstance(section, dict):
        return section
    section = cfg.get("inference")
    if isinstance(section, dict):
        return section
    return {}


def _load_prompts(prompt_file: Path) -> list[str]:
    rows: list[str] = []
    for raw_line in prompt_file.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                rows.append(line)
                continue
            prompt = payload.get("prompt", payload.get("text", ""))
            if isinstance(prompt, str) and prompt.strip():
                rows.append(prompt.strip())
            continue
        rows.append(line)
    return rows


def _load_probe_prompts(
    *,
    dataset_name: str,
    split: str,
    text_field: str,
    tokenizer: Any,
    num_prompts: int,
    skip_rows: int,
    prompt_tokens: int,
) -> list[str]:
    prompts: list[str] = []
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    for idx, row in enumerate(dataset):
        if idx < skip_rows:
            continue
        text = _extract_text_field(row, text_field=text_field)
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < prompt_tokens:
            continue
        prompt_ids = ids[:prompt_tokens]
        prompt = tokenizer.decode(prompt_ids, clean_up_tokenization_spaces=False).strip()
        if prompt:
            prompts.append(prompt)
        if len(prompts) >= num_prompts:
            break
    if len(prompts) < num_prompts:
        raise RuntimeError(f"Only collected {len(prompts)} prompts (requested {num_prompts}).")
    return prompts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--checkpoint-name", default="")
    parser.add_argument("--output-name", default="hierarchical_generation.json")
    parser.add_argument("--inference-config", default="configs/fp16_hierarchical_inference.yaml")
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--skip-rows", type=int, default=-1)
    parser.add_argument("--decode-mode", choices=("mirror_decoder", "memory_lookup"), default="")
    parser.add_argument("--allow-memory-lookup", action="store_true")
    parser.add_argument("--runtime-mode", choices=("quality", "fast"), default="")
    parser.add_argument("--continuation-chunks", type=int, default=0)
    parser.add_argument("--compressed-seq-len", type=int, default=0)
    parser.add_argument("--fast-decode-interval", type=int, default=0)
    parser.add_argument("--fast-refine-tokens", type=int, default=-1)
    parser.add_argument("--ttt-enabled", action="store_true")
    parser.add_argument("--ttt-steps", type=int, default=0)
    parser.add_argument("--ttt-lr", type=float, default=0.0)
    parser.add_argument("--ttt-batch-size", type=int, default=0)
    parser.add_argument("--ttt-max-windows", type=int, default=0)
    parser.add_argument("--ttt-position-stride", type=int, default=0)
    parser.add_argument("--ttt-max-positions", type=int, default=0)
    parser.add_argument("--ttt-scalar-loss-weight", type=float, default=-1.0)
    parser.add_argument("--ttt-grad-clip", type=float, default=-1.0)
    parser.add_argument("--ttt-current-step-targets", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_root = Path(args.run_root)
    output_path = run_root / args.output_name

    config = _load_train_config(run_root)
    checkpoint_path = _resolve_checkpoint_path(run_root, args.checkpoint_name)
    inference_defaults = _load_inference_defaults(Path(args.inference_config))

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

    prompts: list[str] = [prompt.strip() for prompt in args.prompt if str(prompt).strip()]
    if not prompts and str(args.prompt_file).strip():
        prompt_file = Path(args.prompt_file)
        if not prompt_file.exists():
            raise RuntimeError(f"Prompt file does not exist: {prompt_file}")
        prompts = _load_prompts(prompt_file)
    if not prompts:
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

    decode_mode = str(args.decode_mode or inference_defaults.get("decode_mode", "mirror_decoder"))
    runtime_mode = str(args.runtime_mode or inference_defaults.get("runtime_mode", "quality"))
    continuation_chunks = int(
        args.continuation_chunks
        if int(args.continuation_chunks) > 0
        else inference_defaults.get("continuation_chunks", 4),
    )
    compressed_seq_len = int(
        args.compressed_seq_len
        if int(args.compressed_seq_len) > 0
        else inference_defaults.get("compressed_seq_len", training_cfg.get("compressed_seq_len", 8)),
    )
    fast_decode_interval = int(
        args.fast_decode_interval
        if int(args.fast_decode_interval) > 0
        else inference_defaults.get("fast_decode_interval", 2),
    )
    fast_refine_tokens_default = int(inference_defaults.get("fast_refine_tokens", 16))
    fast_refine_tokens = int(args.fast_refine_tokens if int(args.fast_refine_tokens) >= 0 else fast_refine_tokens_default)
    repetition_ngram = int(inference_defaults.get("repetition_ngram", 4))
    repetition_fraction_threshold = float(inference_defaults.get("repetition_fraction_threshold", 0.35))
    ttt_enabled = bool(args.ttt_enabled) or bool(inference_defaults.get("ttt_enabled", False))
    ttt_steps = int(args.ttt_steps if int(args.ttt_steps) > 0 else inference_defaults.get("ttt_steps", 8))
    ttt_lr = float(args.ttt_lr if float(args.ttt_lr) > 0.0 else inference_defaults.get("ttt_lr", 1e-3))
    ttt_batch_size = int(
        args.ttt_batch_size if int(args.ttt_batch_size) > 0 else inference_defaults.get("ttt_batch_size", 8),
    )
    ttt_max_windows = int(
        args.ttt_max_windows if int(args.ttt_max_windows) > 0 else inference_defaults.get("ttt_max_windows", 32),
    )
    ttt_position_stride = int(
        args.ttt_position_stride
        if int(args.ttt_position_stride) > 0
        else inference_defaults.get("ttt_position_stride", 8),
    )
    ttt_max_positions = int(
        args.ttt_max_positions
        if int(args.ttt_max_positions) > 0
        else inference_defaults.get("ttt_max_positions", 8),
    )
    ttt_scalar_loss_weight = float(
        args.ttt_scalar_loss_weight
        if float(args.ttt_scalar_loss_weight) >= 0.0
        else inference_defaults.get("ttt_scalar_loss_weight", 0.05),
    )
    ttt_grad_clip = float(
        args.ttt_grad_clip if float(args.ttt_grad_clip) >= 0.0 else inference_defaults.get("ttt_grad_clip", 1.0),
    )
    ttt_next_step_targets = (
        False
        if bool(args.ttt_current_step_targets)
        else bool(inference_defaults.get("ttt_next_step_targets", True))
    )

    if decode_mode == "memory_lookup" and not bool(args.allow_memory_lookup):
        raise RuntimeError(
            "memory_lookup decode mode is disabled by default. Re-run with --allow-memory-lookup.",
        )

    train_coarse = None
    train_mid = None
    train_fine = None
    train_chunk_tokens = None
    if decode_mode == "memory_lookup":
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
                "memory_lookup mode requires train_scalars_{coarse,mid,fine}.npy and train_chunk_tokens.npy",
            )
        train_coarse = np.load(train_coarse_path, mmap_mode="r").astype(np.float32, copy=False)
        train_mid = np.load(train_mid_path, mmap_mode="r").astype(np.float32, copy=False)
        train_fine = np.load(train_fine_path, mmap_mode="r").astype(np.float32, copy=False)
        train_chunk_tokens = np.load(train_chunks_path, mmap_mode="r")

    device = resolve_device(args.device)
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
    if decode_mode == "mirror_decoder" and not bool(getattr(model_module, "decode_mirror_enabled", False)):
        raise RuntimeError("Checkpoint/config does not enable decode_mirror.")

    generator = HierarchicalGenerator(
        model_module=model_module,
        coarse_compressor=coarse_compressor,
        mid_compressor=mid_compressor,
        fine_compressor=fine_compressor,
        config=HierarchicalGenerationConfig(
            compressed_seq_len=compressed_seq_len,
            continuation_chunks=continuation_chunks,
            decode_mode=decode_mode,
            runtime_mode=runtime_mode,
            fast_decode_interval=fast_decode_interval,
            fast_refine_tokens=fast_refine_tokens,
            repetition_ngram=repetition_ngram,
            repetition_fraction_threshold=repetition_fraction_threshold,
            ttt_enabled=ttt_enabled,
            ttt_steps=ttt_steps,
            ttt_lr=ttt_lr,
            ttt_batch_size=ttt_batch_size,
            ttt_max_windows=ttt_max_windows,
            ttt_position_stride=ttt_position_stride,
            ttt_max_positions=ttt_max_positions,
            ttt_scalar_loss_weight=ttt_scalar_loss_weight,
            ttt_next_step_targets=ttt_next_step_targets,
            ttt_grad_clip=ttt_grad_clip,
        ),
        device=device,
        train_coarse=train_coarse,
        train_mid=train_mid,
        train_fine=train_fine,
        train_chunk_tokens=train_chunk_tokens,
    )

    rows: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    total_generated_tokens = 0
    collapse_count = 0
    ttt_applied_count = 0
    ttt_steps_total = 0
    ttt_loss_values: list[float] = []
    for idx, prompt in enumerate(prompts):
        result = generator.generate(prompt)
        latencies_ms.append(float(result.latency_ms))
        total_generated_tokens += int(len(result.generated_token_ids))
        collapse_count += int(result.collapse_detected)
        ttt_applied_count += int(result.ttt_applied)
        ttt_steps_total += int(result.ttt_steps_run)
        if result.ttt_applied:
            ttt_loss_values.append(float(result.ttt_last_loss))
        rows.append(
            {
                "idx": idx,
                "prompt": prompt,
                "continuation": result.continuation,
                "latency_ms": float(result.latency_ms),
                "generated_tokens": int(len(result.generated_token_ids)),
                "chunk_repeat_fraction": float(result.chunk_repeat_fraction),
                "collapse_detected": bool(result.collapse_detected),
                "ttt_applied": bool(result.ttt_applied),
                "ttt_steps_run": int(result.ttt_steps_run),
                "ttt_last_loss": float(result.ttt_last_loss),
            },
        )

    total_seconds = float(sum(latencies_ms) / 1000.0)
    payload = {
        "run_id": run_root.name,
        "checkpoint_path": str(checkpoint_path),
        "num_prompts": len(rows),
        "decode_mode": decode_mode,
        "runtime_mode": runtime_mode,
        "device": str(device),
        "continuation_chunks": continuation_chunks,
        "compressed_seq_len": compressed_seq_len,
        "fast_decode_interval": fast_decode_interval,
        "fast_refine_tokens": fast_refine_tokens,
        "ttt_enabled": bool(ttt_enabled),
        "ttt_steps": int(ttt_steps),
        "ttt_lr": float(ttt_lr),
        "ttt_batch_size": int(ttt_batch_size),
        "ttt_max_windows": int(ttt_max_windows),
        "ttt_position_stride": int(ttt_position_stride),
        "ttt_max_positions": int(ttt_max_positions),
        "ttt_scalar_loss_weight": float(ttt_scalar_loss_weight),
        "ttt_next_step_targets": bool(ttt_next_step_targets),
        "ttt_grad_clip": float(ttt_grad_clip),
        "ttt_applied_count": int(ttt_applied_count),
        "ttt_steps_total": int(ttt_steps_total),
        "ttt_avg_last_loss": float(np.mean(ttt_loss_values)) if ttt_loss_values else 0.0,
        "avg_latency_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        "total_generated_tokens": int(total_generated_tokens),
        "tokens_per_sec": float(total_generated_tokens / total_seconds) if total_seconds > 0.0 else 0.0,
        "collapse_count": int(collapse_count),
        "collapse_rate": float(collapse_count / float(len(rows) or 1)),
        "rows": rows,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in payload if k != "rows"}, indent=2))
    print(f"Saved generation output: {output_path}")


if __name__ == "__main__":
    main()
