#!/usr/bin/env python3
"""Modal runner for DABE tokenizer autoencoder experiments."""

from __future__ import annotations

import json
import os
import shlex
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

DEFAULT_CONFIG_NAME = "dabe_tokenizer_autoencoder_smoke"

app = modal.App("dabe-tokenizer-autoencoder")
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
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/vol/hf-cache/transformers")
    os.environ.setdefault("HF_DATASETS_CACHE", "/vol/hf-cache/datasets")
    os.environ.setdefault("WANDB_DIR", "/vol/wandb")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _compose_config(config_name: str, overrides: str) -> dict[str, Any]:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    override_list = shlex.split(overrides) if overrides else []
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(REMOTE_REPO / "configs")):
        cfg = compose(config_name=config_name, overrides=override_list)
    return OmegaConf.to_container(cfg, resolve=True)


def _merge_override_string(base: str, extra_tokens: list[str]) -> str:
    base_tokens = shlex.split(base) if base else []
    merged = [*base_tokens, *extra_tokens]
    return " ".join(shlex.quote(token) for token in merged)


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


@app.function(
    image=image,
    gpu="T4",
    cpu=8,
    memory=32768,
    timeout=30 * 60,
    volumes={"/vol": runs_volume},
)
def run_tokenizer_autoencoder(
    config_name: str = DEFAULT_CONFIG_NAME,
    overrides: str = "",
    run_id: str = "",
    stages: str = "train",
    periodic_commit_seconds: int = 120,
) -> dict[str, Any]:
    del stages
    from lightning import seed_everything
    from omegaconf import OmegaConf

    _bootstrap_runtime()
    from src.runtime.device import resolve_trainer_accelerator
    from src.training.dabe_tokenizer_autoencoder import run_dabe_tokenizer_autoencoder

    runtime_overrides = _merge_override_string(
        overrides,
        [
            "experiment.device=cuda",
            "dabe_tokenizer.training.precision=bf16-mixed",
            "dabe_tokenizer.training.num_workers=4",
            "dabe_tokenizer.training.eta_stdout=true",
            "dabe_tokenizer.training.eta_log_interval_steps=100",
        ],
    )
    config = _compose_config(config_name=config_name, overrides=runtime_overrides)
    experiment_cfg = dict(config.get("experiment", {}))
    experiment_name = str(experiment_cfg.get("name", "dabe_tokenizer_autoencoder_modal"))
    resolved_run_id = run_id or f"{experiment_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    output_dir = VOLUME_EXPERIMENTS / resolved_run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(OmegaConf.create(config), output_dir / "config.yaml")
    seed = experiment_cfg.get("seed")
    if seed is not None:
        seed_everything(int(seed), workers=True)

    accelerator, devices = resolve_trainer_accelerator(str(experiment_cfg.get("device", "cuda")))

    def _run() -> dict[str, Any]:
        result = run_dabe_tokenizer_autoencoder(
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
    cpu=8,
    memory=32768,
    timeout=30 * 60,
    volumes={"/vol": runs_volume},
)
def run_tokenizer_autoencoder_sweep(
    config_name: str = DEFAULT_CONFIG_NAME,
    overrides: str = "",
    run_id: str = "",
    stages: str = "train",
    periodic_commit_seconds: int = 120,
    code_bits_csv: str = "512,1024",
) -> dict[str, Any]:
    del stages
    from lightning import seed_everything
    from omegaconf import OmegaConf

    _bootstrap_runtime()
    from src.runtime.device import resolve_trainer_accelerator
    from src.training.dabe_tokenizer_autoencoder import run_dabe_tokenizer_autoencoder

    base_runtime_overrides = _merge_override_string(
        overrides,
        [
            "experiment.device=cuda",
            "dabe_tokenizer.training.precision=bf16-mixed",
            "dabe_tokenizer.training.num_workers=4",
            "dabe_tokenizer.training.eta_stdout=true",
            "dabe_tokenizer.training.eta_log_interval_steps=100",
            "dabe_tokenizer.model.decoder_mode=diffusion",
        ],
    )
    base_config = _compose_config(config_name=config_name, overrides=base_runtime_overrides)
    experiment_cfg = dict(base_config.get("experiment", {}))
    experiment_name = str(experiment_cfg.get("name", "dabe_tokenizer_autoencoder_sweep"))
    resolved_run_id = run_id or f"{experiment_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    output_dir = VOLUME_EXPERIMENTS / resolved_run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    code_bits_values = [int(item.strip()) for item in code_bits_csv.split(",") if item.strip()]
    if not code_bits_values:
        raise ValueError("code_bits_csv must contain at least one integer bit width.")

    accelerator, devices = resolve_trainer_accelerator(str(experiment_cfg.get("device", "cuda")))
    sweep_results: dict[str, Any] = {}

    def _run_sweep() -> dict[str, Any]:
        for code_bits in code_bits_values:
            child_overrides = _merge_override_string(
                base_runtime_overrides,
                [f"dabe_tokenizer.codec.code_bits={code_bits}"],
            )
            child_config = _compose_config(config_name=config_name, overrides=child_overrides)
            child_experiment_cfg = dict(child_config.get("experiment", {}))
            seed = child_experiment_cfg.get("seed")
            if seed is not None:
                seed_everything(int(seed), workers=True)

            child_dir = output_dir / f"bits_{code_bits}"
            child_dir.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(OmegaConf.create(child_config), child_dir / "config.yaml")
            result = run_dabe_tokenizer_autoencoder(
                config=child_config,
                output_dir=child_dir,
                accelerator=accelerator,
                devices=devices,
            )
            sweep_results[str(code_bits)] = {
                "run_id": f"{resolved_run_id}_bits_{code_bits}",
                "output_dir": str(child_dir),
                "overrides": child_overrides,
                "result": result,
            }
            (child_dir / "stage_result.json").write_text(json.dumps(sweep_results[str(code_bits)], indent=2))
            runs_volume.commit()

        payload = {
            "status": "ok",
            "run_id": resolved_run_id,
            "output_dir": str(output_dir),
            "config_name": config_name,
            "base_overrides": base_runtime_overrides,
            "code_bits_values": code_bits_values,
            "result": {"sweep": sweep_results},
        }
        (output_dir / "pipeline_summary.json").write_text(json.dumps(payload, indent=2))
        (output_dir / "stage_result.json").write_text(json.dumps(payload, indent=2))
        return payload

    payload = _run_with_periodic_commits(_run_sweep, interval_seconds=periodic_commit_seconds)
    runs_volume.commit()
    return payload


@app.function(
    image=image,
    gpu="T4",
    cpu=8,
    memory=32768,
    timeout=60 * 60,
    volumes={"/vol": runs_volume},
)
def run_tokenizer_lookup_pair(
    config_name: str = DEFAULT_CONFIG_NAME,
    overrides: str = "",
    run_id: str = "",
    stages: str = "train",
    periodic_commit_seconds: int = 120,
) -> dict[str, Any]:
    del stages
    from lightning import seed_everything
    from omegaconf import OmegaConf

    _bootstrap_runtime()
    from src.runtime.device import resolve_trainer_accelerator
    from src.training.dabe_tokenizer_autoencoder import run_dabe_tokenizer_autoencoder

    base_runtime_overrides = _merge_override_string(
        overrides,
        [
            "experiment.device=cuda",
            "dabe_tokenizer.training.precision=bf16-mixed",
            "dabe_tokenizer.training.num_workers=4",
            "dabe_tokenizer.training.eta_stdout=true",
            "dabe_tokenizer.training.eta_log_interval_steps=100",
        ],
    )
    base_config = _compose_config(config_name=config_name, overrides=base_runtime_overrides)
    experiment_cfg = dict(base_config.get("experiment", {}))
    experiment_name = str(experiment_cfg.get("name", "dabe_tokenizer_lookup_pair"))
    resolved_run_id = run_id or f"{experiment_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    output_dir = VOLUME_EXPERIMENTS / resolved_run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator, devices = resolve_trainer_accelerator(str(experiment_cfg.get("device", "cuda")))
    variants = [
        (
            "exp083_fixed_k32",
            [
                "dabe_tokenizer.model.lexical_lookup_k=32",
                "dabe_tokenizer.model.lexical_lookup_slot_policy=fixed",
                "dabe_tokenizer.training.lookup_slot_cost_weight=0.0",
                "dabe_tokenizer.training.lookup_slot_target_weight=0.0",
                "dabe_tokenizer.training.lookup_slot_target_k=null",
            ],
        ),
        (
            "exp084_ranked_halting_k32",
            [
                "dabe_tokenizer.model.lexical_lookup_k=32",
                "dabe_tokenizer.model.lexical_lookup_slot_policy=halting",
                "dabe_tokenizer.model.lexical_lookup_keep_threshold=0.5",
                "dabe_tokenizer.training.lookup_slot_cost_weight=0.02",
                "dabe_tokenizer.training.lookup_slot_target_weight=0.02",
                "dabe_tokenizer.training.lookup_slot_target_k=16",
            ],
        ),
    ]
    pair_results: dict[str, Any] = {}

    def _run_pair() -> dict[str, Any]:
        for variant_name, variant_tokens in variants:
            child_overrides = _merge_override_string(base_runtime_overrides, variant_tokens)
            child_config = _compose_config(config_name=config_name, overrides=child_overrides)
            child_experiment_cfg = dict(child_config.get("experiment", {}))
            seed = child_experiment_cfg.get("seed")
            if seed is not None:
                seed_everything(int(seed), workers=True)

            child_dir = output_dir / variant_name
            child_dir.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(OmegaConf.create(child_config), child_dir / "config.yaml")
            result = run_dabe_tokenizer_autoencoder(
                config=child_config,
                output_dir=child_dir,
                accelerator=accelerator,
                devices=devices,
            )
            pair_results[variant_name] = {
                "run_id": f"{resolved_run_id}_{variant_name}",
                "output_dir": str(child_dir),
                "overrides": child_overrides,
                "result": result,
            }
            (child_dir / "stage_result.json").write_text(json.dumps(pair_results[variant_name], indent=2))
            runs_volume.commit()

        payload = {
            "status": "ok",
            "run_id": resolved_run_id,
            "output_dir": str(output_dir),
            "config_name": config_name,
            "base_overrides": base_runtime_overrides,
            "variants": [name for name, _tokens in variants],
            "result": {"pair": pair_results},
        }
        (output_dir / "pipeline_summary.json").write_text(json.dumps(payload, indent=2))
        (output_dir / "stage_result.json").write_text(json.dumps(payload, indent=2))
        return payload

    payload = _run_with_periodic_commits(_run_pair, interval_seconds=periodic_commit_seconds)
    runs_volume.commit()
    return payload


@app.function(
    image=image,
    gpu="T4",
    cpu=8,
    memory=32768,
    timeout=90 * 60,
    volumes={"/vol": runs_volume},
)
def run_tokenizer_halting_rate_sweep(
    config_name: str = DEFAULT_CONFIG_NAME,
    overrides: str = "",
    run_id: str = "",
    stages: str = "train",
    periodic_commit_seconds: int = 120,
    target_k_csv: str = "12,16,20",
) -> dict[str, Any]:
    del stages
    from lightning import seed_everything
    from omegaconf import OmegaConf

    _bootstrap_runtime()
    from src.runtime.device import resolve_trainer_accelerator
    from src.training.dabe_tokenizer_autoencoder import run_dabe_tokenizer_autoencoder

    base_runtime_overrides = _merge_override_string(
        overrides,
        [
            "experiment.device=cuda",
            "dabe_tokenizer.training.precision=bf16-mixed",
            "dabe_tokenizer.training.num_workers=4",
            "dabe_tokenizer.training.eta_stdout=true",
            "dabe_tokenizer.training.eta_log_interval_steps=100",
            "dabe_tokenizer.model.lexical_lookup_k=32",
            "dabe_tokenizer.model.lexical_lookup_slot_policy=halting",
            "dabe_tokenizer.model.lexical_lookup_keep_threshold=0.5",
            "dabe_tokenizer.training.lookup_slot_cost_weight=0.02",
            "dabe_tokenizer.training.lookup_slot_target_weight=0.02",
        ],
    )
    base_config = _compose_config(config_name=config_name, overrides=base_runtime_overrides)
    experiment_cfg = dict(base_config.get("experiment", {}))
    experiment_name = str(experiment_cfg.get("name", "dabe_tokenizer_halting_rate_sweep"))
    resolved_run_id = run_id or f"{experiment_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    output_dir = VOLUME_EXPERIMENTS / resolved_run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    target_k_values = [float(item.strip()) for item in target_k_csv.split(",") if item.strip()]
    if not target_k_values:
        raise ValueError("target_k_csv must contain at least one target K value.")

    accelerator, devices = resolve_trainer_accelerator(str(experiment_cfg.get("device", "cuda")))
    sweep_results: dict[str, Any] = {}

    def _target_label(target_k: float) -> str:
        return f"k{target_k:g}".replace(".", "p")

    def _run_sweep() -> dict[str, Any]:
        for target_k in target_k_values:
            target_label = _target_label(target_k)
            variant_name = f"exp085_halting_target_{target_label}"
            child_overrides = _merge_override_string(
                base_runtime_overrides,
                [f"dabe_tokenizer.training.lookup_slot_target_k={target_k:g}"],
            )
            child_config = _compose_config(config_name=config_name, overrides=child_overrides)
            child_experiment_cfg = dict(child_config.get("experiment", {}))
            seed = child_experiment_cfg.get("seed")
            if seed is not None:
                seed_everything(int(seed), workers=True)

            child_dir = output_dir / variant_name
            child_dir.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(OmegaConf.create(child_config), child_dir / "config.yaml")
            result = run_dabe_tokenizer_autoencoder(
                config=child_config,
                output_dir=child_dir,
                accelerator=accelerator,
                devices=devices,
            )
            sweep_results[target_label] = {
                "run_id": f"{resolved_run_id}_{variant_name}",
                "variant_name": variant_name,
                "target_k": target_k,
                "output_dir": str(child_dir),
                "overrides": child_overrides,
                "result": result,
            }
            (child_dir / "stage_result.json").write_text(json.dumps(sweep_results[target_label], indent=2))
            runs_volume.commit()

        payload = {
            "status": "ok",
            "run_id": resolved_run_id,
            "output_dir": str(output_dir),
            "config_name": config_name,
            "base_overrides": base_runtime_overrides,
            "target_k_values": target_k_values,
            "result": {"halting_rate_sweep": sweep_results},
        }
        (output_dir / "pipeline_summary.json").write_text(json.dumps(payload, indent=2))
        (output_dir / "stage_result.json").write_text(json.dumps(payload, indent=2))
        return payload

    payload = _run_with_periodic_commits(_run_sweep, interval_seconds=periodic_commit_seconds)
    runs_volume.commit()
    return payload


@app.function(
    image=image,
    gpu="T4",
    cpu=8,
    memory=32768,
    timeout=150 * 60,
    volumes={"/vol": runs_volume},
)
def run_tokenizer_gist_residual_sweep(
    config_name: str = DEFAULT_CONFIG_NAME,
    overrides: str = "",
    run_id: str = "",
    stages: str = "train",
    periodic_commit_seconds: int = 120,
    router_loss_csv: str = "0.1,0.2",
    slot_cost_csv: str = "0.02,0.04",
) -> dict[str, Any]:
    del stages
    from lightning import seed_everything
    from omegaconf import OmegaConf

    _bootstrap_runtime()
    from src.runtime.device import resolve_trainer_accelerator
    from src.training.dabe_tokenizer_autoencoder import run_dabe_tokenizer_autoencoder

    base_runtime_overrides = _merge_override_string(
        overrides,
        [
            "experiment.device=cuda",
            "dabe_tokenizer.training.precision=bf16-mixed",
            "dabe_tokenizer.training.num_workers=4",
            "dabe_tokenizer.training.eta_stdout=true",
            "dabe_tokenizer.training.eta_log_interval_steps=100",
            "dabe_tokenizer.model.decoder_mode=gist_residual_lookup",
            "dabe_tokenizer.model.lexical_lookup_k=32",
            "dabe_tokenizer.model.lexical_lookup_slot_policy=halting",
            "dabe_tokenizer.model.lexical_lookup_keep_threshold=0.5",
            "dabe_tokenizer.training.selector_loss_weight=0.0",
            "dabe_tokenizer.training.budget_loss_weight=0.0",
            "dabe_tokenizer.training.lookup_slot_target_weight=0.0",
            "dabe_tokenizer.training.lookup_slot_target_k=null",
        ],
    )
    base_config = _compose_config(config_name=config_name, overrides=base_runtime_overrides)
    experiment_cfg = dict(base_config.get("experiment", {}))
    experiment_name = str(experiment_cfg.get("name", "dabe_tokenizer_gist_residual_sweep"))
    resolved_run_id = run_id or f"{experiment_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    output_dir = VOLUME_EXPERIMENTS / resolved_run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    router_loss_values = [float(item.strip()) for item in router_loss_csv.split(",") if item.strip()]
    slot_cost_values = [float(item.strip()) for item in slot_cost_csv.split(",") if item.strip()]
    if not router_loss_values:
        raise ValueError("router_loss_csv must contain at least one residual router loss weight.")
    if not slot_cost_values:
        raise ValueError("slot_cost_csv must contain at least one lookup slot cost weight.")

    accelerator, devices = resolve_trainer_accelerator(str(experiment_cfg.get("device", "cuda")))
    sweep_results: dict[str, Any] = {}

    def _float_label(value: float) -> str:
        return f"{value:g}".replace(".", "p")

    def _run_sweep() -> dict[str, Any]:
        for router_loss in router_loss_values:
            for slot_cost in slot_cost_values:
                router_label = _float_label(router_loss)
                cost_label = _float_label(slot_cost)
                variant_name = f"exp087_router_w{router_label}_cost{cost_label}"
                child_overrides = _merge_override_string(
                    base_runtime_overrides,
                    [
                        f"dabe_tokenizer.training.residual_router_loss_weight={router_loss:g}",
                        f"dabe_tokenizer.training.lookup_slot_cost_weight={slot_cost:g}",
                    ],
                )
                child_config = _compose_config(config_name=config_name, overrides=child_overrides)
                child_experiment_cfg = dict(child_config.get("experiment", {}))
                seed = child_experiment_cfg.get("seed")
                if seed is not None:
                    seed_everything(int(seed), workers=True)

                child_dir = output_dir / variant_name
                child_dir.mkdir(parents=True, exist_ok=True)
                OmegaConf.save(OmegaConf.create(child_config), child_dir / "config.yaml")
                result = run_dabe_tokenizer_autoencoder(
                    config=child_config,
                    output_dir=child_dir,
                    accelerator=accelerator,
                    devices=devices,
                )
                sweep_key = f"router_w{router_label}_cost{cost_label}"
                sweep_results[sweep_key] = {
                    "run_id": f"{resolved_run_id}_{variant_name}",
                    "variant_name": variant_name,
                    "residual_router_loss_weight": router_loss,
                    "lookup_slot_cost_weight": slot_cost,
                    "output_dir": str(child_dir),
                    "overrides": child_overrides,
                    "result": result,
                }
                (child_dir / "stage_result.json").write_text(json.dumps(sweep_results[sweep_key], indent=2))
                runs_volume.commit()

        payload = {
            "status": "ok",
            "run_id": resolved_run_id,
            "output_dir": str(output_dir),
            "config_name": config_name,
            "base_overrides": base_runtime_overrides,
            "router_loss_values": router_loss_values,
            "slot_cost_values": slot_cost_values,
            "result": {"gist_residual_sweep": sweep_results},
        }
        (output_dir / "pipeline_summary.json").write_text(json.dumps(payload, indent=2))
        (output_dir / "stage_result.json").write_text(json.dumps(payload, indent=2))
        return payload

    payload = _run_with_periodic_commits(_run_sweep, interval_seconds=periodic_commit_seconds)
    runs_volume.commit()
    return payload


@app.function(
    image=image,
    gpu="T4",
    cpu=8,
    memory=32768,
    timeout=30 * 60,
    volumes={"/vol": runs_volume},
)
def run_tokenizer_reverse_diffusion_probe(
    config_name: str = DEFAULT_CONFIG_NAME,
    overrides: str = "",
    run_id: str = "",
    stages: str = "probe",
    periodic_commit_seconds: int = 120,
    source_run_id: str = "exp065_modal_dabe_tokae_diffusion_bits_sweep_001",
    code_bits_csv: str = "512,1024",
    max_batches: int = 64,
    sample_steps: int = 64,
) -> dict[str, Any]:
    del stages
    from lightning import seed_everything
    from omegaconf import OmegaConf

    _bootstrap_runtime()
    from src.runtime.device import resolve_trainer_accelerator
    from src.training.dabe_tokenizer_autoencoder import run_dabe_tokenizer_reverse_diffusion_probe

    base_runtime_overrides = _merge_override_string(
        overrides,
        [
            "experiment.device=cuda",
            "dabe_tokenizer.training.precision=bf16-mixed",
            "dabe_tokenizer.training.num_workers=4",
            "dabe_tokenizer.model.decoder_mode=diffusion",
        ],
    )
    base_config = _compose_config(config_name=config_name, overrides=base_runtime_overrides)
    experiment_cfg = dict(base_config.get("experiment", {}))
    experiment_name = str(experiment_cfg.get("name", "dabe_tokenizer_reverse_diffusion_probe"))
    resolved_run_id = run_id or f"{experiment_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    output_dir = VOLUME_EXPERIMENTS / resolved_run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    code_bits_values = [int(item.strip()) for item in code_bits_csv.split(",") if item.strip()]
    if not code_bits_values:
        raise ValueError("code_bits_csv must contain at least one integer bit width.")

    seed = experiment_cfg.get("seed")
    if seed is not None:
        seed_everything(int(seed), workers=True)
    accelerator, devices = resolve_trainer_accelerator(str(experiment_cfg.get("device", "cuda")))
    probe_results: dict[str, Any] = {}

    def _run_probe() -> dict[str, Any]:
        for code_bits in code_bits_values:
            child_overrides = _merge_override_string(
                base_runtime_overrides,
                [f"dabe_tokenizer.codec.code_bits={code_bits}"],
            )
            child_config = _compose_config(config_name=config_name, overrides=child_overrides)
            checkpoint_path = (
                VOLUME_EXPERIMENTS
                / source_run_id
                / f"bits_{code_bits}"
                / "checkpoints"
                / "best-step-0006000.ckpt"
            )
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Missing checkpoint for {code_bits} bits: {checkpoint_path}")

            child_dir = output_dir / f"bits_{code_bits}"
            child_dir.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(OmegaConf.create(child_config), child_dir / "config.yaml")
            result = run_dabe_tokenizer_reverse_diffusion_probe(
                config=child_config,
                checkpoint_path=checkpoint_path,
                output_dir=child_dir,
                accelerator=accelerator,
                devices=devices,
                max_batches=int(max_batches),
                sample_steps=int(sample_steps),
            )
            probe_results[str(code_bits)] = {
                "run_id": f"{resolved_run_id}_bits_{code_bits}",
                "source_checkpoint": str(checkpoint_path),
                "output_dir": str(child_dir),
                "overrides": child_overrides,
                "result": result,
            }
            (child_dir / "stage_result.json").write_text(json.dumps(probe_results[str(code_bits)], indent=2))
            runs_volume.commit()

        payload = {
            "status": "ok",
            "run_id": resolved_run_id,
            "source_run_id": source_run_id,
            "output_dir": str(output_dir),
            "config_name": config_name,
            "base_overrides": base_runtime_overrides,
            "code_bits_values": code_bits_values,
            "max_batches": int(max_batches),
            "sample_steps": int(sample_steps),
            "result": {"probe": probe_results},
        }
        (output_dir / "pipeline_summary.json").write_text(json.dumps(payload, indent=2))
        (output_dir / "stage_result.json").write_text(json.dumps(payload, indent=2))
        return payload

    payload = _run_with_periodic_commits(_run_probe, interval_seconds=periodic_commit_seconds)
    runs_volume.commit()
    return payload


@app.function(
    image=image,
    gpu="T4",
    cpu=8,
    memory=32768,
    timeout=30 * 60,
    volumes={"/vol": runs_volume},
)
def run_tokenizer_decode_diagnostics(
    config_name: str = DEFAULT_CONFIG_NAME,
    overrides: str = "",
    run_id: str = "",
    stages: str = "probe",
    periodic_commit_seconds: int = 120,
    source_run_id: str = "exp067_modal_dabe_code_transformer_512_001",
    checkpoint_name: str = "best-step-0008000.ckpt",
    max_batches: int = 128,
    num_samples: int = 12,
    runtime_extra_overrides: str = "",
) -> dict[str, Any]:
    del stages
    from lightning import seed_everything
    from omegaconf import OmegaConf

    _bootstrap_runtime()
    from src.runtime.device import resolve_trainer_accelerator
    from src.training.dabe_tokenizer_autoencoder import run_dabe_tokenizer_decode_diagnostics

    extra_override_tokens = shlex.split(runtime_extra_overrides) if runtime_extra_overrides else []
    runtime_overrides = _merge_override_string(
        overrides,
        [
            "experiment.device=cuda",
            "dabe_tokenizer.training.precision=bf16-mixed",
            "dabe_tokenizer.training.num_workers=4",
            *extra_override_tokens,
        ],
    )
    config = _compose_config(config_name=config_name, overrides=runtime_overrides)
    experiment_cfg = dict(config.get("experiment", {}))
    experiment_name = str(experiment_cfg.get("name", "dabe_tokenizer_decode_diagnostics"))
    resolved_run_id = run_id or f"{experiment_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    output_dir = VOLUME_EXPERIMENTS / resolved_run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = experiment_cfg.get("seed")
    if seed is not None:
        seed_everything(int(seed), workers=True)
    accelerator, devices = resolve_trainer_accelerator(str(experiment_cfg.get("device", "cuda")))
    checkpoint_path = VOLUME_EXPERIMENTS / source_run_id / "checkpoints" / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    OmegaConf.save(OmegaConf.create(config), output_dir / "config.yaml")

    def _run_probe() -> dict[str, Any]:
        result = run_dabe_tokenizer_decode_diagnostics(
            config=config,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            accelerator=accelerator,
            devices=devices,
            max_batches=int(max_batches),
            num_samples=int(num_samples),
        )
        payload = {
            "status": "ok",
            "run_id": resolved_run_id,
            "source_run_id": source_run_id,
            "checkpoint_name": checkpoint_name,
            "output_dir": str(output_dir),
            "config_name": config_name,
            "overrides": runtime_overrides,
            "max_batches": int(max_batches),
            "num_samples": int(num_samples),
            "result": {"decode_diagnostics": result},
        }
        (output_dir / "pipeline_summary.json").write_text(json.dumps(payload, indent=2))
        (output_dir / "stage_result.json").write_text(json.dumps(payload, indent=2))
        return payload

    payload = _run_with_periodic_commits(_run_probe, interval_seconds=periodic_commit_seconds)
    runs_volume.commit()
    return payload


@app.local_entrypoint()
def main(
    config_name: str = DEFAULT_CONFIG_NAME,
    overrides: str = "",
    run_id: str = "",
    stages: str = "train",
    periodic_commit_seconds: int = 120,
):
    summary = run_tokenizer_autoencoder.remote(
        config_name=config_name,
        overrides=overrides,
        run_id=run_id,
        stages=stages,
        periodic_commit_seconds=periodic_commit_seconds,
    )
    print(json.dumps(summary, indent=2))
