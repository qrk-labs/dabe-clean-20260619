#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import hydra
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf

from src.runtime.device import resolve_trainer_accelerator
from src.training.fp16_hierarchical_feasibility import run_fp16_hierarchical_feasibility

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="fp16_hierarchical_64_8_1_exp031_seed")
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    experiment_cfg = config.get("experiment", {})
    experiment_name = str(experiment_cfg.get("name", "fp16_hierarchical_64_8_1"))
    run_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path("experiments") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(cfg, output_dir / "config.yaml")
    log.info("Config saved to %s", output_dir / "config.yaml")

    seed = experiment_cfg.get("seed")
    if seed is not None:
        seed_everything(int(seed), workers=True)

    accelerator, devices = resolve_trainer_accelerator(str(experiment_cfg.get("device", "auto")))
    log.info("Trainer resolved to accelerator=%s devices=%s", accelerator, devices)

    result = run_fp16_hierarchical_feasibility(
        config=config,
        output_dir=output_dir,
        accelerator=accelerator,
        devices=devices,
    )
    payload = {
        "run_id": run_id,
        "experiment": experiment_name,
        "track": "fp16_hierarchical_64_8_1",
        "config": config,
        "fp16_hierarchical": result,
    }
    with open(output_dir / "results.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    log.info("Results saved to %s", output_dir / "results.json")
    log.info("Run complete: %s", run_id)


if __name__ == "__main__":
    main()
