#!/usr/bin/env python3
"""Modular Modal runner for FP16 chunk feasibility experiments."""

from __future__ import annotations

import json
import os
import shlex
import sys
import threading
from pathlib import Path
from typing import Any

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO = Path("/workspace/dabe")
VOLUME_ROOT = Path("/vol")
VOLUME_EXPERIMENTS = VOLUME_ROOT / "experiments"

DEFAULT_CONFIG_NAME = "fp16_chunk_modal_2m_unique_1b_total"
DEFAULT_STAGES = "preprocess,train"

app = modal.App("dabe-fp16-chunk")
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
    from src.training.modal_fp16_chunk_pipeline import FP16ChunkModalStageRunner, FP16ModalRunContext

    context = FP16ModalRunContext.create(
        repo_root=REMOTE_REPO,
        run_id=run_id,
        base_output_root=VOLUME_EXPERIMENTS,
        config_name=config_name,
        overrides=overrides,
    )
    return FP16ChunkModalStageRunner(context=context)


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
    cpu=16,
    memory=65536,
    timeout=2 * 60 * 60,
    volumes={"/vol": runs_volume},
)
def run_preprocess_stage(
    config_name: str = DEFAULT_CONFIG_NAME,
    overrides: str = "",
    run_id: str = "",
    periodic_commit_seconds: int = 300,
) -> dict[str, Any]:
    preprocess_overrides = _merge_override_string(
        overrides,
        [
            "experiment.device=cpu",
            "fp16_chunk.training.num_workers=16",
        ],
    )
    runner = _stage_runner(
        config_name=config_name,
        overrides=preprocess_overrides,
        run_id=run_id or None,
    )
    payload = _run_with_periodic_commits(
        lambda: runner.run_stage("preprocess"),
        interval_seconds=periodic_commit_seconds,
    )
    return _commit_with_result(payload)


@app.function(
    image=image,
    gpu="T4",
    cpu=16,
    memory=65536,
    timeout=8 * 60 * 60,
    volumes={"/vol": runs_volume},
)
def run_train_stage(
    config_name: str = DEFAULT_CONFIG_NAME,
    overrides: str = "",
    run_id: str = "",
    periodic_commit_seconds: int = 300,
) -> dict[str, Any]:
    train_overrides = _merge_override_string(
        overrides,
        [
            "experiment.device=cuda",
            "fp16_chunk.training.precision=bf16-mixed",
            "fp16_chunk.training.num_workers=8",
        ],
    )
    runner = _stage_runner(
        config_name=config_name,
        overrides=train_overrides,
        run_id=run_id or None,
    )
    payload = _run_with_periodic_commits(
        lambda: runner.run_stage("train"),
        interval_seconds=periodic_commit_seconds,
    )
    return _commit_with_result(payload)


@app.function(
    image=image,
    gpu="T4",
    cpu=16,
    memory=65536,
    timeout=1 * 60 * 60,
    volumes={"/vol": runs_volume},
)
def run_pipeline(
    config_name: str = DEFAULT_CONFIG_NAME,
    overrides: str = "",
    run_id: str = "",
    stages: str = DEFAULT_STAGES,
    periodic_commit_seconds: int = 300,
) -> dict[str, Any]:
    pipeline_overrides = _merge_override_string(
        overrides,
        [
            "experiment.device=cuda",
            "fp16_chunk.training.precision=bf16-mixed",
            "fp16_chunk.training.num_workers=8",
            "fp16_chunk.training.eta_stdout=true",
        ],
    )
    runner = _stage_runner(
        config_name=config_name,
        overrides=pipeline_overrides,
        run_id=run_id or None,
    )
    from src.training.modal_fp16_chunk_pipeline import parse_stage_list

    payload = _run_with_periodic_commits(
        lambda: runner.run_stages(parse_stage_list(stages)),
        interval_seconds=periodic_commit_seconds,
    )
    return _commit_with_result(payload)


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

