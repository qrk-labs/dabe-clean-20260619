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
from src.training.dabe_tokenizer_autoencoder import run_dabe_tokenizer_autoencoder

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="dabe_tokenizer_autoencoder_smoke")
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    experiment_cfg = config.get("experiment", {})
    experiment_name = str(experiment_cfg.get("name", "dabe_tokenizer_autoencoder_smoke"))
    run_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path("experiments") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(cfg, output_dir / "config.yaml")
    seed = experiment_cfg.get("seed")
    if seed is not None:
        seed_everything(int(seed), workers=True)

    accelerator, devices = resolve_trainer_accelerator(str(experiment_cfg.get("device", "auto")))
    result = run_dabe_tokenizer_autoencoder(
        config=config,
        output_dir=output_dir,
        accelerator=accelerator,
        devices=devices,
    )
    payload = {
        "run_id": run_id,
        "experiment": experiment_name,
        "track": "dabe_tokenizer_autoencoder",
        "config": config,
        "result": result,
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2))

    log.info("Results saved to %s", output_dir / "results.json")
    log.info("Run complete: %s", run_id)


if __name__ == "__main__":
    main()
