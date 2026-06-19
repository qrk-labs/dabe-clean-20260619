#!/usr/bin/env python3
"""Modal launcher for hierarchical 64/8/1 FP16 feasibility runs."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO = Path("/workspace/dabe")
VOLUME_ROOT = Path("/vol")
VOLUME_EXPERIMENTS = VOLUME_ROOT / "experiments"

DEFAULT_CONFIG_NAME = "fp16_hierarchical_64_8_1_modal_t4"

app = modal.App("dabe-fp16-hierarchical")
runs_volume = modal.Volume.from_name("dabe-experiments", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0",
        "lightning>=2.0",
        "hydra-core>=1.3",
        "omegaconf>=2.3",
        "wandb>=0.15",
        "datasets>=2.14",
        "transformers>=4.35",
        "sentencepiece>=0.1",
        "numpy>=1.24",
        "tqdm>=4.65",
        "certifi>=2024.2.2",
    )
    .add_local_dir(str(REPO_ROOT / "src"), remote_path=str(REMOTE_REPO / "src"))
    .add_local_dir(str(REPO_ROOT / "configs"), remote_path=str(REMOTE_REPO / "configs"))
    .add_local_dir(str(REPO_ROOT / "scripts"), remote_path=str(REMOTE_REPO / "scripts"))
)


def _bootstrap_runtime() -> None:
    os.chdir(str(REMOTE_REPO))
    if str(REMOTE_REPO) not in sys.path:
        sys.path.insert(0, str(REMOTE_REPO))

    VOLUME_EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("HF_HOME", "/vol/hf-cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/vol/hf-cache/transformers")
    os.environ.setdefault("HF_DATASETS_CACHE", "/vol/hf-cache/datasets")
    os.environ.setdefault("WANDB_DIR", "/vol/wandb")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _run_with_periodic_commits(fn, interval_seconds: int) -> dict[str, Any]:
    interval = max(0, int(interval_seconds))
    if interval <= 0:
        return fn()

    stop_event = threading.Event()

    def _periodic_commit_loop() -> None:
        while not stop_event.wait(interval):
            try:
                runs_volume.commit()
            except Exception:
                pass

    thread = threading.Thread(target=_periodic_commit_loop, daemon=True)
    thread.start()
    try:
        return fn()
    finally:
        stop_event.set()
        thread.join(timeout=2.0)
        runs_volume.commit()


def _compose_config(config_name: str, overrides: str) -> dict[str, Any]:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    config_dir = REMOTE_REPO / "configs"
    override_list = shlex.split(overrides) if overrides else []
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=override_list)
    return OmegaConf.to_container(cfg, resolve=True)


def _merge_override_string(base: str, extra_tokens: list[str]) -> str:
    base_tokens = shlex.split(base) if base else []
    merged = [*base_tokens, *extra_tokens]
    return " ".join(shlex.quote(token) for token in merged)


@app.function(
    image=image,
    gpu="T4",
    cpu=8,
    memory=32768,
    timeout=12 * 60 * 60,
    volumes={"/vol": runs_volume},
)
def run_hierarchical_train(
    config_name: str = DEFAULT_CONFIG_NAME,
    overrides: str = "",
    run_id: str = "",
    periodic_commit_seconds: int = 300,
) -> dict[str, Any]:
    from lightning import seed_everything
    from omegaconf import OmegaConf

    _bootstrap_runtime()
    from src.runtime.device import resolve_trainer_accelerator
    from src.training.fp16_hierarchical_feasibility import run_fp16_hierarchical_feasibility

    runtime_overrides = _merge_override_string(
        overrides,
        [
            "experiment.device=cuda",
        ],
    )
    config = _compose_config(config_name=config_name, overrides=runtime_overrides)
    experiment_cfg = config.get("experiment", {})
    experiment_name = str(experiment_cfg.get("name", "fp16_hierarchical_modal"))
    resolved_run_id = run_id or f"{experiment_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    output_dir = VOLUME_EXPERIMENTS / resolved_run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(OmegaConf.create(config), output_dir / "config.yaml")
    seed = experiment_cfg.get("seed")
    if seed is not None:
        seed_everything(int(seed), workers=True)

    accelerator, devices = resolve_trainer_accelerator(str(experiment_cfg.get("device", "cuda")))

    def _run() -> dict[str, Any]:
        result = run_fp16_hierarchical_feasibility(
            config=config,
            output_dir=output_dir,
            accelerator=accelerator,
            devices=devices,
        )
        payload = {
            "status": "ok",
            "run_id": resolved_run_id,
            "output_dir": str(output_dir),
            "config_name": config_name,
            "overrides": runtime_overrides,
            "config": config,
            "result": {"train": result},
        }
        (output_dir / "pipeline_summary.json").write_text(json.dumps(payload, indent=2))
        (output_dir / "stage_result.json").write_text(json.dumps(payload, indent=2))
        return payload

    payload = _run_with_periodic_commits(_run, interval_seconds=periodic_commit_seconds)
    runs_volume.commit()
    return payload


@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    memory=16384,
    timeout=2 * 60 * 60,
    volumes={"/vol": runs_volume},
)
def run_hierarchical_probe(
    run_id: str,
    checkpoint_name: str = "",
    num_prompts: int = 100,
    prompt_set_file: str = "research/eval/fixed_prompt_set_v1.jsonl",
    output_name: str = "interaction_probe_fixedset100_openrouter_modal.json",
    openrouter_model: str = "deepseek/deepseek-v4-flash",
    judge_chunk_size: int = 8,
    periodic_commit_seconds: int = 120,
) -> dict[str, Any]:
    _bootstrap_runtime()
    run_root = VOLUME_EXPERIMENTS / run_id
    if not run_root.exists():
        raise RuntimeError(f"Run root not found in volume: {run_root}")

    checkpoint = checkpoint_name
    if not checkpoint:
        ckpt_dir = run_root / "checkpoints"
        ckpts = sorted(ckpt_dir.glob("best-step-*.ckpt"))
        if not ckpts:
            raise RuntimeError(f"No best checkpoint found in run root: {run_root}")
        checkpoint = ckpts[-1].name

    command = [
        "python",
        "scripts/run_fp16_hierarchical_interaction_probe.py",
        "--run-root",
        str(run_root),
        "--checkpoint-name",
        checkpoint,
        "--prompt-set-file",
        prompt_set_file,
        "--num-prompts",
        str(int(num_prompts)),
        "--output-name",
        output_name,
        "--openrouter-judge",
        "--openrouter-model",
        openrouter_model,
        "--judge-chunk-size",
        str(int(judge_chunk_size)),
    ]

    def _run() -> dict[str, Any]:
        proc = subprocess.run(
            command,
            cwd=str(REMOTE_REPO),
            capture_output=True,
            text=True,
            check=False,
        )
        payload = {
            "status": "ok" if proc.returncode == 0 else "failed",
            "run_id": run_id,
            "checkpoint_name": checkpoint,
            "command": command,
            "returncode": int(proc.returncode),
            "stdout": proc.stdout[-8000:],
            "stderr": proc.stderr[-8000:],
            "output_path": str(run_root / output_name),
        }
        return payload

    payload = _run_with_periodic_commits(_run, interval_seconds=periodic_commit_seconds)
    runs_volume.commit()
    return payload


@app.local_entrypoint()
def main(
    config_name: str = DEFAULT_CONFIG_NAME,
    overrides: str = "",
    run_id: str = "",
    periodic_commit_seconds: int = 300,
):
    summary = run_hierarchical_train.remote(
        config_name=config_name,
        overrides=overrides,
        run_id=run_id,
        periodic_commit_seconds=periodic_commit_seconds,
    )
    print(json.dumps(summary, indent=2))
