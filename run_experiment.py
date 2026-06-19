#!/usr/bin/env python3
import json
import logging
from datetime import datetime
from pathlib import Path

import hydra
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

import wandb
from src.data.multilingual import MultilingualDataModule
from src.pipeline import DABEPipeline
from src.runtime.device import resolve_trainer_accelerator
from src.training.feasibility import (
    run_adapter_feasibility_experiment,
    run_feasibility_experiment,
)
from src.training.pretrain import PretrainLightningModule

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="density_adaptive")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    experiment_name = config["experiment"]["name"]
    run_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path("experiments") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(cfg, output_dir / "config.yaml")
    log.info(f"Experiment config saved to {output_dir / 'config.yaml'}")

    seed = config["experiment"].get("seed")
    if seed is not None:
        seed_everything(int(seed), workers=True)
        torch.manual_seed(int(seed))

    logging_cfg = config.get("logging", {})
    offline_wandb = logging_cfg.get("wandb_mode", "online") == "offline"
    wandb_logger = WandbLogger(
        project=logging_cfg.get("wandb_project", "dabe"),
        entity=logging_cfg.get("wandb_entity"),
        name=run_id,
        config=config,
        offline=offline_wandb,
    )

    experiment_track = config["experiment"].get("track", "legacy")
    accelerator, devices = resolve_trainer_accelerator(config["experiment"].get("device", "auto"))
    log.info(f"Trainer device resolved to accelerator={accelerator}, devices={devices}")

    if experiment_track == "feasibility":
        log.info("Running single-track feasibility pipeline...")
        feasibility_results = run_feasibility_experiment(
            config=config,
            output_dir=output_dir,
            wandb_logger=wandb_logger,
            accelerator=accelerator,
            devices=devices,
        )
        results = {
            "run_id": run_id,
            "experiment": experiment_name,
            "track": experiment_track,
            "config": config,
            "feasibility": feasibility_results,
        }
    elif experiment_track == "adapter_feasibility":
        log.info("Running adapter-route feasibility pipeline...")
        adapter_results = run_adapter_feasibility_experiment(
            config=config,
            output_dir=output_dir,
            wandb_logger=wandb_logger,
            accelerator=accelerator,
            devices=devices,
        )
        results = {
            "run_id": run_id,
            "experiment": experiment_name,
            "track": experiment_track,
            "config": config,
            "adapter_feasibility": adapter_results,
        }
    else:
        log.info("Running shape validation via DABEPipeline...")
        pipeline = DABEPipeline(config)
        shape_report = pipeline.run_dummy_forward()
        log.info(f"Shape validation passed: {json.dumps(shape_report, default=str)}")

        wandb_logger.log_metrics({"shapes/logits_0": shape_report["shapes"]["logits_shape"][0]})
        wandb_logger.log_metrics({"shapes/hidden_0": shape_report["shapes"]["hidden_shape"][0]})

        if config["experiment"].get("train", True):
            log.info("Initializing data module...")
            data_cfg = config.get("data", {})
            training_cfg = config.get("training", {})
            datamodule = MultilingualDataModule(
                source_langs=data_cfg["source_langs"],
                target_langs=data_cfg.get("target_langs"),
                dataset_name=data_cfg.get("dataset_name", "wikiann"),
                synthetic_only=data_cfg.get("synthetic_only", False),
                batch_size=training_cfg["batch_size"],
                max_samples=data_cfg.get("max_samples", 10000),
                num_workers=data_cfg.get("num_workers", 0),
            )

            log.info("Initializing Lightning module...")
            model = PretrainLightningModule(config)

            callbacks: list[Callback] = [
                ModelCheckpoint(
                    dirpath=str(output_dir / "checkpoints"),
                    filename="{epoch}-{step}-{val_loss:.4f}",
                    monitor="val_loss",
                    save_top_k=3,
                    auto_insert_metric_name=False,
                ),
            ]

            if training_cfg.get("early_stop_patience"):
                callbacks.append(
                    EarlyStopping(monitor="val_loss", patience=training_cfg["early_stop_patience"])
                )

            trainer = Trainer(
                max_epochs=training_cfg.get("max_epochs", 10),
                accelerator=accelerator,
                devices=devices,
                logger=wandb_logger,
                callbacks=callbacks,
                gradient_clip_val=training_cfg.get("grad_clip"),
                log_every_n_steps=logging_cfg.get("log_every_n_steps", 10),
                default_root_dir=str(output_dir),
            )

            log.info("Starting training...")
            trainer.fit(model, datamodule=datamodule)

            final_loss = (
                trainer.callback_metrics.get("train/loss_epoch")
                or trainer.callback_metrics.get("train_loss")
                or trainer.callback_metrics.get("train/loss")
            )

            results = {
                "run_id": run_id,
                "experiment": experiment_name,
                "config": config,
                "final_loss": final_loss,
                "checkpoints": str(output_dir / "checkpoints"),
            }
        else:
            log.info("Shape validation only; no training run.")
            results = {
                "run_id": run_id,
                "experiment": experiment_name,
                "config": config,
                "shape_validation": shape_report,
                "training": "skipped",
            }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Results saved to {output_dir / 'results.json'}")

    wandb.finish()
    log.info(f"Experiment {run_id} complete. Output in {output_dir}")


if __name__ == "__main__":
    main()
