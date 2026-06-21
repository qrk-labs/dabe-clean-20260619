#!/usr/bin/env python3
"""Modular Modal runner for DABE feasibility experiments."""

from __future__ import annotations

import json
import os
import shlex
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO = Path("/workspace/dabe")
VOLUME_ROOT = Path("/vol")
VOLUME_EXPERIMENTS = VOLUME_ROOT / "experiments"

DEFAULT_CONFIG_NAME = "feasibility_modal_olmo_smoke"
DEFAULT_STAGES = "tokenizer,dabe_lm,bpe_baseline"

app = modal.App("dabe-feasibility-smoke")
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
    )
    .add_local_dir(str(REPO_ROOT / "src"), remote_path=str(REMOTE_REPO / "src"))
    .add_local_dir(str(REPO_ROOT / "configs"), remote_path=str(REMOTE_REPO / "configs"))
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


def _stage_runner(config_name: str, overrides: str, run_id: str | None):
    _bootstrap_runtime()
    from src.training.modal_feasibility_pipeline import (
        FeasibilityRunContext,
        FeasibilityStageRunner,
    )

    context = FeasibilityRunContext.create(
        repo_root=REMOTE_REPO,
        run_id=run_id,
        base_output_root=VOLUME_EXPERIMENTS,
        config_name=config_name,
        overrides=overrides,
    )
    return FeasibilityStageRunner(context=context)


def _commit_with_result(payload: dict[str, Any]) -> dict[str, Any]:
    runs_volume.commit()
    return payload


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
                # Non-fatal best-effort checkpointing during long stages.
                pass

    thread = threading.Thread(target=_periodic_commit_loop, daemon=True)
    thread.start()
    try:
        return fn()
    finally:
        stop_event.set()
        thread.join(timeout=2.0)
        runs_volume.commit()


def _merge_override_string(base: str, extra_tokens: list[str]) -> str:
    base_tokens = shlex.split(base) if base else []
    merged = [*base_tokens, *extra_tokens]
    return " ".join(shlex.quote(token) for token in merged)


@app.function(
    image=image,
    gpu="T4",
    cpu=8,
    memory=32768,
    timeout=75 * 60,
    volumes={"/vol": runs_volume},
)
def run_tokenizer_stage(
    config_name: str = DEFAULT_CONFIG_NAME,
    overrides: str = "",
    run_id: str = "",
    periodic_commit_seconds: int = 300,
) -> dict[str, Any]:
    tokenizer_overrides = _merge_override_string(
        overrides,
        [
            "+router.device=cuda",
            "feasibility.tokenizer_training.device=cuda",
        ],
    )
    runner = _stage_runner(
        config_name=config_name,
        overrides=tokenizer_overrides,
        run_id=run_id or None,
    )
    payload = _run_with_periodic_commits(
        lambda: runner.run_stage("tokenizer"),
        interval_seconds=periodic_commit_seconds,
    )
    return _commit_with_result(payload)


@app.function(
    image=image,
    gpu="T4",
    cpu=8,
    memory=32768,
    timeout=75 * 60,
    volumes={"/vol": runs_volume},
)
def run_dabe_stage(
    config_name: str = DEFAULT_CONFIG_NAME,
    overrides: str = "",
    run_id: str = "",
    periodic_commit_seconds: int = 300,
) -> dict[str, Any]:
    dabe_overrides = _merge_override_string(overrides, ["+router.device=cuda"])
    runner = _stage_runner(
        config_name=config_name,
        overrides=dabe_overrides,
        run_id=run_id or None,
    )
    payload = _run_with_periodic_commits(
        lambda: runner.run_stage("dabe_lm"),
        interval_seconds=periodic_commit_seconds,
    )
    return _commit_with_result(payload)


@app.function(
    image=image,
    gpu="T4",
    cpu=8,
    memory=32768,
    timeout=75 * 60,
    volumes={"/vol": runs_volume},
)
def run_bpe_stage(
    config_name: str = DEFAULT_CONFIG_NAME,
    overrides: str = "",
    run_id: str = "",
    periodic_commit_seconds: int = 300,
) -> dict[str, Any]:
    runner = _stage_runner(config_name=config_name, overrides=overrides, run_id=run_id or None)
    payload = _run_with_periodic_commits(
        lambda: runner.run_stage("bpe_baseline"),
        interval_seconds=periodic_commit_seconds,
    )
    return _commit_with_result(payload)


@app.function(
    image=image,
    gpu="T4",
    cpu=8,
    memory=32768,
    timeout=3 * 60 * 60,
    volumes={"/vol": runs_volume},
)
def run_pipeline(
    config_name: str = DEFAULT_CONFIG_NAME,
    overrides: str = "",
    run_id: str = "",
    stages: str = DEFAULT_STAGES,
    periodic_commit_seconds: int = 300,
) -> dict[str, Any]:
    runner = _stage_runner(config_name=config_name, overrides=overrides, run_id=run_id or None)
    from src.training.modal_feasibility_pipeline import parse_stage_list

    payload = _run_with_periodic_commits(
        lambda: runner.run_stages(parse_stage_list(stages)),
        interval_seconds=periodic_commit_seconds,
    )
    return _commit_with_result(payload)


@app.function(
    image=image,
    gpu="T4",
    cpu=8,
    memory=32768,
    timeout=40 * 60,
    volumes={"/vol": runs_volume},
)
def run_python_code_dabe_bpe_concurrent(
    config_name: str = "feasibility_python_code_smoke",
    overrides: str = "",
    run_id: str = "",
    periodic_commit_seconds: int = 120,
) -> dict[str, Any]:
    """Train DABE and BPE LM baselines concurrently after shared tokenizer prep."""
    runtime_overrides = _merge_override_string(
        overrides,
        [
            "experiment.device=cuda",
            "optimization.device_profile=cuda_fast",
            "feasibility.tokenizer_training.device=cuda",
            "logging.eta_log_every_n_steps=25",
        ],
    )
    runner = _stage_runner(
        config_name=config_name,
        overrides=runtime_overrides,
        run_id=run_id or None,
    )

    def _run() -> dict[str, Any]:
        tokenizer_payload = runner.run_stage("tokenizer")
        stage_payloads: dict[str, Any] = {"tokenizer": tokenizer_payload}
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        def _run_stage(stage: str) -> dict[str, Any]:
            stage_runner = _stage_runner(
                config_name=config_name,
                overrides=runtime_overrides,
                run_id=runner.context.run_id,
            )
            return stage_runner.run_stage(stage)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(_run_stage, stage): stage
                for stage in ("dabe_lm", "bpe_baseline")
            }
            for future in as_completed(futures):
                stage = futures[future]
                stage_payloads[stage] = future.result()
                runs_volume.commit()

        summary = runner._load_summary()
        summary["status"] = (
            "ok"
            if all(payload.get("status") == "ok" for payload in stage_payloads.values())
            else "partial_or_failed"
        )
        summary["execution_mode"] = "tokenizer_then_concurrent_dabe_lm_and_bpe_baseline"
        summary["concurrent_stages"] = ["dabe_lm", "bpe_baseline"]
        summary["stage_statuses"] = {
            stage: payload.get("status") for stage, payload in stage_payloads.items()
        }
        runner._write_json(runner.context.summary_path, summary)
        return summary

    payload = _run_with_periodic_commits(_run, interval_seconds=periodic_commit_seconds)
    return _commit_with_result(payload)


@app.function(
    image=image,
    cpu=1,
    memory=2048,
    timeout=5 * 60,
    volumes={"/vol": runs_volume},
)
def estimate_stage_timeout_seconds(
    run_id: str,
    stage: str,
    safety_factor: float = 1.6,
) -> dict[str, Any]:
    _bootstrap_runtime()
    summary_path = VOLUME_EXPERIMENTS / run_id / "pipeline_summary.json"
    if not summary_path.exists():
        return {"run_id": run_id, "stage": stage, "estimated_timeout_seconds": None}
    summary = json.loads(summary_path.read_text())
    stage_payload = ((summary.get("stages", {}) or {}).get(stage, {}))
    config = stage_payload.get("config", {})
    runtime = ((stage_payload.get("result", {}) or {}).get(stage, {}) or {}).get("runtime", {})
    steps_per_sec = runtime.get("steps_per_sec")
    prep_seconds = runtime.get("prep_seconds")
    if not steps_per_sec:
        return {"run_id": run_id, "stage": stage, "estimated_timeout_seconds": None}
    training_cfg = config.get("training", {})
    epochs = int(training_cfg.get("max_epochs", 1))
    batch_size = int(training_cfg.get("batch_size", 1))
    train_samples = ((stage_payload.get("result", {}) or {}).get(stage, {}) or {}).get("train_samples")
    if not train_samples:
        return {"run_id": run_id, "stage": stage, "estimated_timeout_seconds": None}
    estimated_steps = max(1, int((int(train_samples) + max(batch_size - 1, 0)) / max(batch_size, 1))) * epochs
    train_seconds = float(estimated_steps) / max(float(steps_per_sec), 1e-6)
    total_seconds = (float(prep_seconds or 0.0) + train_seconds) * max(float(safety_factor), 1.0)
    return {
        "run_id": run_id,
        "stage": stage,
        "estimated_timeout_seconds": int(total_seconds),
        "prep_seconds": prep_seconds,
        "steps_per_sec": steps_per_sec,
        "estimated_steps": estimated_steps,
    }


@app.local_entrypoint()
def main(
    config_name: str = DEFAULT_CONFIG_NAME,
    overrides: str = "",
    run_id: str = "",
    stages: str = DEFAULT_STAGES,
    periodic_commit_seconds: int = 300,
):
    summary = run_pipeline.remote(
        config_name=config_name,
        overrides=overrides,
        run_id=run_id,
        stages=stages,
        periodic_commit_seconds=periodic_commit_seconds,
    )
    print(json.dumps(summary, indent=2))
