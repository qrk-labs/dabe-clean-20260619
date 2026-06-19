from __future__ import annotations

import json
import logging
import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from lightning import seed_everything
from omegaconf import OmegaConf

from ..runtime.device import resolve_trainer_accelerator
from .feasibility import run_feasibility_experiment

log = logging.getLogger(__name__)

VALID_FEASIBILITY_STAGES = ("tokenizer", "dabe_lm", "bpe_baseline")


def parse_stage_list(stages: str | Sequence[str]) -> list[str]:
    if isinstance(stages, str):
        stage_list = [part.strip() for part in stages.split(",") if part.strip()]
    else:
        stage_list = [str(part).strip() for part in stages if str(part).strip()]

    unknown = [stage for stage in stage_list if stage not in VALID_FEASIBILITY_STAGES]
    if unknown:
        raise ValueError(
            f"Unknown feasibility stages: {unknown}. "
            f"Valid stages: {list(VALID_FEASIBILITY_STAGES)}"
        )
    if not stage_list:
        raise ValueError("No feasibility stages requested.")
    return stage_list


def compute_competitive_gate(
    dabe_val_loss: float | None,
    bpe_val_loss: float | None,
    compression_ratio_vs_bpe: float | None,
    dabe_stable: bool,
    bpe_stable: bool,
    loss_tolerance_ratio: float,
    min_compression_ratio: float,
) -> bool:
    if dabe_val_loss is None or bpe_val_loss is None:
        return False
    if compression_ratio_vs_bpe is None:
        return False
    if not dabe_stable or not bpe_stable:
        return False
    if compression_ratio_vs_bpe < min_compression_ratio:
        return False
    return dabe_val_loss <= bpe_val_loss * (1.0 + loss_tolerance_ratio)


@dataclass
class FeasibilityRunContext:
    repo_root: Path
    run_id: str
    run_root: Path
    config_name: str
    overrides: list[str]

    @classmethod
    def create(
        cls,
        *,
        repo_root: Path,
        run_id: str | None,
        base_output_root: Path,
        config_name: str,
        overrides: str | Sequence[str] | None,
    ) -> "FeasibilityRunContext":
        resolved_run_id = run_id or datetime.now(timezone.utc).strftime(
            "modal_feasibility_%Y%m%d_%H%M%S"
        )
        if isinstance(overrides, str):
            parsed_overrides = shlex.split(overrides) if overrides else []
        else:
            parsed_overrides = list(overrides or [])
        run_root = base_output_root / resolved_run_id
        return cls(
            repo_root=repo_root,
            run_id=resolved_run_id,
            run_root=run_root,
            config_name=config_name,
            overrides=parsed_overrides,
        )

    @property
    def tokenizer_artifact_path(self) -> Path:
        return self.run_root / "artifacts" / "lfq_tokenizer.pt"

    @property
    def summary_path(self) -> Path:
        return self.run_root / "pipeline_summary.json"

    def stage_output_dir(self, stage: str) -> Path:
        return self.run_root / stage


class FeasibilityStageRunner:
    """Programmatic stage runner for Modal feasibility experiments.

    Keeps each stage reproducible and independently restartable while sharing
    run-level artifacts (tokenizer + summaries) through a common run root.
    """

    def __init__(self, context: FeasibilityRunContext):
        self.context = context
        self.config_dir = self.context.repo_root / "configs"
        self.context.run_root.mkdir(parents=True, exist_ok=True)
        self.context.tokenizer_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        self._maybe_adopt_existing_overrides()

    def _maybe_adopt_existing_overrides(self) -> None:
        if self.context.overrides:
            return
        if not self.context.summary_path.exists():
            return
        try:
            existing = json.loads(self.context.summary_path.read_text())
        except Exception:
            return
        prior_overrides = existing.get("overrides")
        if isinstance(prior_overrides, list) and prior_overrides:
            self.context.overrides = [str(item) for item in prior_overrides]

    def _compose_stage_config(self, stage: str) -> dict[str, Any]:
        GlobalHydra.instance().clear()
        stage_overrides = [
            *self.context.overrides,
            "experiment.track=feasibility",
            f"experiment.feasibility_stage={stage}",
            f"+feasibility.tokenizer_artifact_path={self.context.tokenizer_artifact_path}",
            "logging.wandb_mode=offline",
        ]

        with initialize_config_dir(version_base=None, config_dir=str(self.config_dir)):
            cfg = compose(config_name=self.context.config_name, overrides=stage_overrides)
        return OmegaConf.to_container(cfg, resolve=True)

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str))

    def _load_summary(self) -> dict[str, Any]:
        if not self.context.summary_path.exists():
            return {
                "run_id": self.context.run_id,
                "config_name": self.context.config_name,
                "overrides": self.context.overrides,
                "run_root": str(self.context.run_root),
                "tokenizer_artifact_path": str(self.context.tokenizer_artifact_path),
                "stages": {},
            }
        return json.loads(self.context.summary_path.read_text())

    def _update_summary(self, stage_payload: dict[str, Any]) -> dict[str, Any]:
        summary = self._load_summary()
        stage_name = stage_payload["stage"]
        summary["stages"][stage_name] = stage_payload
        summary["updated_at"] = datetime.now(timezone.utc).isoformat()

        gate = self._build_aggregate_gate(summary)
        if gate is not None:
            summary["decision_gate"] = gate

        self._write_json(self.context.summary_path, summary)
        return summary

    def _build_aggregate_gate(self, summary: dict[str, Any]) -> dict[str, Any] | None:
        stages = summary.get("stages", {})
        tokenizer_result = stages.get("tokenizer", {}).get("result", {})
        dabe_result = stages.get("dabe_lm", {}).get("result", {})
        bpe_result = stages.get("bpe_baseline", {}).get("result", {})

        compression_value = (
            tokenizer_result.get("tokenizer", {}) or {}
        ).get("compression_ratio_vs_bpe")
        dabe_val_loss = ((dabe_result.get("dabe_lm", {}) or {}).get("val_loss"))
        bpe_val_loss = ((bpe_result.get("bpe_baseline", {}) or {}).get("val_loss"))
        dabe_stable = bool((dabe_result.get("dabe_lm", {}) or {}).get("stable", False))
        bpe_stable = bool((bpe_result.get("bpe_baseline", {}) or {}).get("stable", False))

        if not stages:
            return None

        # Gate thresholds are shared across stages; take from any present stage config.
        any_stage = next(iter(stages.values()))
        stage_config = any_stage.get("config", {})
        gate_cfg = stage_config.get("feasibility", {}).get("decision_gate", {})
        loss_tolerance = float(gate_cfg.get("loss_tolerance_ratio", 0.15))
        min_compression = float(gate_cfg.get("min_compression_ratio", 1.0))

        return {
            "dabe_val_loss": dabe_val_loss,
            "bpe_val_loss": bpe_val_loss,
            "compression_ratio_vs_bpe": compression_value,
            "dabe_stable": dabe_stable,
            "bpe_stable": bpe_stable,
            "loss_tolerance_ratio": loss_tolerance,
            "min_compression_ratio": min_compression,
            "competitive": compute_competitive_gate(
                dabe_val_loss=dabe_val_loss,
                bpe_val_loss=bpe_val_loss,
                compression_ratio_vs_bpe=compression_value,
                dabe_stable=dabe_stable,
                bpe_stable=bpe_stable,
                loss_tolerance_ratio=loss_tolerance,
                min_compression_ratio=min_compression,
            ),
        }

    def run_stage(self, stage: str) -> dict[str, Any]:
        parse_stage_list([stage])  # validates stage value
        stage_output_dir = self.context.stage_output_dir(stage)
        stage_output_dir.mkdir(parents=True, exist_ok=True)

        config = self._compose_stage_config(stage)
        OmegaConf.save(OmegaConf.create(config), stage_output_dir / "config.yaml")

        seed = config.get("experiment", {}).get("seed")
        if seed is not None:
            seed_everything(int(seed), workers=True)
            torch.manual_seed(int(seed))

        accelerator, devices = resolve_trainer_accelerator(
            config.get("experiment", {}).get("device", "auto")
        )

        try:
            stage_result = run_feasibility_experiment(
                config=config,
                output_dir=stage_output_dir,
                wandb_logger=None,
                accelerator=accelerator,
                devices=devices,
            )
            payload = {
                "status": "ok",
                "run_id": self.context.run_id,
                "stage": stage,
                "output_dir": str(stage_output_dir),
                "config_name": self.context.config_name,
                "overrides": self.context.overrides,
                "config": config,
                "result": stage_result,
            }
        except Exception as exc:
            payload = {
                "status": "failed",
                "run_id": self.context.run_id,
                "stage": stage,
                "output_dir": str(stage_output_dir),
                "config_name": self.context.config_name,
                "overrides": self.context.overrides,
                "config": config,
                "error": str(exc),
            }
            self._write_json(stage_output_dir / "stage_result.json", payload)
            self._update_summary(payload)
            raise

        self._write_json(stage_output_dir / "stage_result.json", payload)
        self._update_summary(payload)
        return payload

    def run_stages(self, stages: Iterable[str]) -> dict[str, Any]:
        stage_list = parse_stage_list(list(stages))
        results: dict[str, Any] = {
            "status": "ok",
            "run_id": self.context.run_id,
            "stages": {},
        }
        for stage in stage_list:
            log.info("Starting feasibility stage: %s (run_id=%s)", stage, self.context.run_id)
            results["stages"][stage] = self.run_stage(stage)
        results["summary_path"] = str(self.context.summary_path)
        return results
