from __future__ import annotations

import json
import logging
import shlex
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import lightning as L
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from ..runtime.device import resolve_trainer_accelerator
from .fp16_chunk_feasibility import (
    EtaMetricsCallback,
    FP16ChunkCompressor,
    FP16ChunkLMModule,
    FP16ChunkSequenceDataset,
    RuntimeMetrics,
    RuntimeMetricsCallback,
    ValidationTrendCallback,
    _load_text_dataset_samples,
    _resolve_precision,
)

log = logging.getLogger(__name__)

VALID_FP16_MODAL_STAGES = ("preprocess", "train")


def parse_stage_list(stages: str | Sequence[str]) -> list[str]:
    if isinstance(stages, str):
        stage_list = [part.strip() for part in stages.split(",") if part.strip()]
    else:
        stage_list = [str(part).strip() for part in stages if str(part).strip()]

    unknown = [stage for stage in stage_list if stage not in VALID_FP16_MODAL_STAGES]
    if unknown:
        raise ValueError(
            f"Unknown fp16 modal stages: {unknown}. "
            f"Valid stages: {list(VALID_FP16_MODAL_STAGES)}"
        )
    if not stage_list:
        raise ValueError("No fp16 modal stages requested.")
    return stage_list


@dataclass
class FP16ModalRunContext:
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
    ) -> "FP16ModalRunContext":
        resolved_run_id = run_id or datetime.now(timezone.utc).strftime(
            "modal_fp16_chunk_%Y%m%d_%H%M%S",
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
    def preprocessed_dir(self) -> Path:
        return self.run_root / "artifacts" / "preprocessed"

    @property
    def preprocessed_metadata_path(self) -> Path:
        return self.preprocessed_dir / "metadata.json"

    @property
    def summary_path(self) -> Path:
        return self.run_root / "pipeline_summary.json"

    def stage_output_dir(self, stage: str) -> Path:
        return self.run_root / stage


class _CachedFP16ChunkDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        train_stream: np.ndarray,
        val_stream: np.ndarray,
        compressed_seq_len: int,
        window_stride: int,
        batch_size: int,
        num_workers: int,
        seed: int,
        accelerator: str,
        train_chunk_tokens: np.ndarray | None = None,
        val_chunk_tokens: np.ndarray | None = None,
    ):
        super().__init__()
        self.train_stream = train_stream
        self.val_stream = val_stream
        self.train_chunk_tokens = train_chunk_tokens
        self.val_chunk_tokens = val_chunk_tokens
        self.compressed_seq_len = int(compressed_seq_len)
        self.window_stride = int(window_stride)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.seed = int(seed)
        self.accelerator = str(accelerator)
        self.train_chunk_count = int(train_stream.shape[0])
        self.val_chunk_count = int(val_stream.shape[0])
        self._is_setup = False

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self._is_setup:
            return
        self.train_dataset = FP16ChunkSequenceDataset(
            scalar_stream=self.train_stream,
            compressed_seq_len=self.compressed_seq_len,
            window_stride=self.window_stride,
            chunk_tokens_stream=self.train_chunk_tokens,
        )
        self.val_dataset = FP16ChunkSequenceDataset(
            scalar_stream=self.val_stream,
            compressed_seq_len=self.compressed_seq_len,
            window_stride=self.window_stride,
            chunk_tokens_stream=self.val_chunk_tokens,
        )
        if len(self.train_dataset) == 0:
            raise RuntimeError(
                "Cached preprocess produced zero train windows. "
                "Increase train_samples or reduce chunk_size_tokens/compressed_seq_len.",
            )
        if len(self.val_dataset) == 0:
            raise RuntimeError(
                "Cached preprocess produced zero val windows. "
                "Increase val_samples or reduce chunk_size_tokens/compressed_seq_len.",
            )
        self._is_setup = True

    def _dataloader_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"num_workers": self.num_workers}
        if self.accelerator == "gpu" and self.num_workers > 0:
            kwargs.update({
                "pin_memory": True,
                "persistent_workers": True,
                "prefetch_factor": 2,
            })
        return kwargs

    def train_dataloader(self) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            **self._dataloader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            **self._dataloader_kwargs(),
        )


class FP16ChunkModalStageRunner:
    def __init__(self, context: FP16ModalRunContext):
        self.context = context
        self.config_dir = self.context.repo_root / "configs"
        self.context.run_root.mkdir(parents=True, exist_ok=True)
        self.context.preprocessed_dir.mkdir(parents=True, exist_ok=True)
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

    def _compose_config(self) -> dict[str, Any]:
        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base=None, config_dir=str(self.config_dir)):
            cfg = compose(config_name=self.context.config_name, overrides=self.context.overrides)
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
                "preprocessed_dir": str(self.context.preprocessed_dir),
                "stages": {},
            }
        return json.loads(self.context.summary_path.read_text())

    def _update_summary(self, stage_payload: dict[str, Any]) -> dict[str, Any]:
        summary = self._load_summary()
        summary["stages"][stage_payload["stage"]] = stage_payload
        summary["updated_at"] = datetime.now(timezone.utc).isoformat()

        preprocess_result = (summary["stages"].get("preprocess", {}).get("result") or {}).get("preprocess", {})
        train_result = (summary["stages"].get("train", {}).get("result") or {}).get("train", {})
        if preprocess_result or train_result:
            summary["aggregate"] = {
                "stable": bool(train_result.get("stable", False)),
                "nan_batches": int(train_result.get("nan_batches", 0)),
                "unique_raw_tokens_equivalent": int(preprocess_result.get("train_chunks", 0))
                * int(preprocess_result.get("chunk_size_tokens", 0)),
                "processed_raw_tokens_target": int(train_result.get("processed_raw_tokens_target", 0)),
                "val_loss_last": train_result.get("val_loss_last"),
                "raw_tokens_per_sec": train_result.get("raw_tokens_per_sec"),
            }

        self._write_json(self.context.summary_path, summary)
        return summary

    def _preprocess_arrays(self, config: dict[str, Any], stage_output_dir: Path) -> dict[str, Any]:
        started = time.perf_counter()
        fp16_cfg = dict(config.get("fp16_chunk", {}))
        dataset_cfg = dict(fp16_cfg.get("dataset", {}))
        compression_cfg = dict(fp16_cfg.get("compression", {}))
        diffusion_cfg = dict(fp16_cfg.get("diffusion", {}))
        diffusion_enabled = bool(diffusion_cfg.get("enabled", False))
        seed = int(config.get("experiment", {}).get("seed", 42))

        train_texts = _load_text_dataset_samples(
            dataset_name=str(dataset_cfg.get("dataset_name", "roneneldan/TinyStories")),
            split=str(dataset_cfg.get("train_split", "train")),
            max_samples=int(dataset_cfg.get("train_samples", 4096)),
            seed=seed,
            text_field=str(dataset_cfg.get("text_field", "text")),
            allow_synthetic_fallback=bool(dataset_cfg.get("allow_synthetic_fallback", False)),
            streaming=bool(dataset_cfg.get("streaming", True)),
        )
        val_texts = _load_text_dataset_samples(
            dataset_name=str(dataset_cfg.get("dataset_name", "roneneldan/TinyStories")),
            split=str(dataset_cfg.get("val_split", "validation")),
            max_samples=int(dataset_cfg.get("val_samples", 512)),
            seed=seed + 1,
            text_field=str(dataset_cfg.get("text_field", "text")),
            allow_synthetic_fallback=bool(dataset_cfg.get("allow_synthetic_fallback", False)),
            streaming=bool(dataset_cfg.get("streaming", True)),
        )

        compressor = FP16ChunkCompressor(
            chunk_size_tokens=int(compression_cfg.get("chunk_size_tokens", 128)),
            window_overlap_tokens=int(compression_cfg.get("window_overlap", 0)),
            tokenizer_name=str(compression_cfg.get("tokenizer_name", "gpt2")),
            force_fast_tokenizer=bool(compression_cfg.get("force_fast_tokenizer", True)),
            dtype=str(compression_cfg.get("dtype", "float16")),
            scalars_per_chunk=int(compression_cfg.get("scalars_per_chunk", 1)),
        )
        tokenizer_batch_size = int(compression_cfg.get("tokenizer_batch_size", 64))
        if diffusion_enabled:
            train_stream, train_chunks = compressor.flatten_stream_with_chunks(
                texts=train_texts,
                tokenizer_batch_size=tokenizer_batch_size,
            )
            val_stream, val_chunks = compressor.flatten_stream_with_chunks(
                texts=val_texts,
                tokenizer_batch_size=tokenizer_batch_size,
            )
        else:
            train_stream = compressor.flatten_stream(
                texts=train_texts,
                tokenizer_batch_size=tokenizer_batch_size,
            )
            val_stream = compressor.flatten_stream(
                texts=val_texts,
                tokenizer_batch_size=tokenizer_batch_size,
            )
            train_chunks = None
            val_chunks = None

        self.context.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        train_scalars_path = self.context.preprocessed_dir / "train_scalars.npy"
        val_scalars_path = self.context.preprocessed_dir / "val_scalars.npy"
        np.save(train_scalars_path, train_stream)
        np.save(val_scalars_path, val_stream)

        train_chunks_path: Path | None = None
        val_chunks_path: Path | None = None
        if train_chunks is not None and val_chunks is not None:
            train_chunks_path = self.context.preprocessed_dir / "train_chunk_tokens.npy"
            val_chunks_path = self.context.preprocessed_dir / "val_chunk_tokens.npy"
            np.save(train_chunks_path, train_chunks)
            np.save(val_chunks_path, val_chunks)

        preprocess_seconds = max(1e-6, time.perf_counter() - started)
        metadata = {
            "run_id": self.context.run_id,
            "diffusion_enabled": diffusion_enabled,
            "chunk_size_tokens": compressor.chunk_size_tokens,
            "tokenizer_batch_size": tokenizer_batch_size,
            "train_text_samples": len(train_texts),
            "val_text_samples": len(val_texts),
            "train_chunks": int(train_stream.shape[0]),
            "val_chunks": int(val_stream.shape[0]),
            "train_scalars_path": str(train_scalars_path),
            "val_scalars_path": str(val_scalars_path),
            "train_chunk_tokens_path": str(train_chunks_path) if train_chunks_path is not None else None,
            "val_chunk_tokens_path": str(val_chunks_path) if val_chunks_path is not None else None,
            "preprocess_seconds": preprocess_seconds,
        }
        self._write_json(self.context.preprocessed_metadata_path, metadata)
        self._write_json(stage_output_dir / "preprocess_metadata.json", metadata)
        return metadata

    def _load_cached_arrays(self, use_memmap: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
        if not self.context.preprocessed_metadata_path.exists():
            raise RuntimeError(
                "Missing cached preprocess metadata. Run 'preprocess' stage before 'train'.",
            )
        metadata = json.loads(self.context.preprocessed_metadata_path.read_text())
        mmap_mode = "r" if use_memmap else None
        train_stream = np.load(metadata["train_scalars_path"], mmap_mode=mmap_mode)
        val_stream = np.load(metadata["val_scalars_path"], mmap_mode=mmap_mode)
        train_chunk_tokens = None
        val_chunk_tokens = None
        train_chunks_path = metadata.get("train_chunk_tokens_path")
        val_chunks_path = metadata.get("val_chunk_tokens_path")
        if train_chunks_path and val_chunks_path:
            train_chunk_tokens = np.load(train_chunks_path, mmap_mode=mmap_mode)
            val_chunk_tokens = np.load(val_chunks_path, mmap_mode=mmap_mode)
        return train_stream, val_stream, train_chunk_tokens, val_chunk_tokens, metadata

    def _train_from_cache(self, config: dict[str, Any], stage_output_dir: Path) -> dict[str, Any]:
        prep_started = time.perf_counter()
        fp16_cfg = dict(config.get("fp16_chunk", {}))
        training_cfg = dict(fp16_cfg.get("training", {}))
        gate_cfg = dict(fp16_cfg.get("gate", {}))
        diffusion_cfg = dict(fp16_cfg.get("diffusion", {}))
        diffusion_enabled = bool(diffusion_cfg.get("enabled", False))
        seed = int(config.get("experiment", {}).get("seed", 42))
        accelerator, devices = resolve_trainer_accelerator(str(config.get("experiment", {}).get("device", "auto")))

        modal_cfg = dict(config.get("fp16_chunk_modal", {}))
        use_memmap = bool(modal_cfg.get("use_memmap", True))
        (
            train_stream,
            val_stream,
            train_chunk_tokens,
            val_chunk_tokens,
            cache_metadata,
        ) = self._load_cached_arrays(use_memmap=use_memmap)

        data_module = _CachedFP16ChunkDataModule(
            train_stream=train_stream,
            val_stream=val_stream,
            train_chunk_tokens=train_chunk_tokens,
            val_chunk_tokens=val_chunk_tokens,
            compressed_seq_len=int(training_cfg.get("compressed_seq_len", 8)),
            window_stride=int(training_cfg.get("window_stride", 1)),
            batch_size=int(training_cfg.get("batch_size", 64)),
            num_workers=int(training_cfg.get("num_workers", 0)),
            seed=seed,
            accelerator=accelerator,
        )
        model_config = {
            **fp16_cfg,
            "diffusion": {
                **diffusion_cfg,
                "vocab_size": int(diffusion_cfg.get("vocab_size", 50257)),
            },
        }
        model = FP16ChunkLMModule(config=model_config)
        prep_seconds = max(1e-6, time.perf_counter() - prep_started)

        runtime_metrics = RuntimeMetrics(
            chunk_size_tokens=int(fp16_cfg.get("compression", {}).get("chunk_size_tokens", 64)),
        )
        runtime_callback = RuntimeMetricsCallback(runtime_metrics=runtime_metrics)
        trend_callback = ValidationTrendCallback()
        eta_callback = EtaMetricsCallback(
            log_interval_steps=int(training_cfg.get("eta_log_interval_steps", 20)),
            emit_stdout=bool(training_cfg.get("eta_stdout", False)),
        )
        callbacks: list[Callback] = [runtime_callback, trend_callback, eta_callback]

        checkpoint_dir = stage_output_dir / "checkpoints"
        checkpoint_every_n_train_steps = int(training_cfg.get("checkpoint_every_n_train_steps", 0))
        checkpoint_save_top_k = int(training_cfg.get("checkpoint_save_top_k", 1))
        checkpoint_save_last = bool(training_cfg.get("checkpoint_save_last", True))
        checkpoint_best_callback: ModelCheckpoint | None = None
        checkpoint_periodic_callback: ModelCheckpoint | None = None
        if checkpoint_every_n_train_steps > 0:
            checkpoint_periodic_callback = ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="step-{step:07d}",
                save_top_k=-1,
                save_last=False,
                every_n_train_steps=checkpoint_every_n_train_steps,
                save_on_train_epoch_end=False,
            )
            callbacks.append(checkpoint_periodic_callback)
        if checkpoint_save_top_k != 0 or checkpoint_save_last:
            checkpoint_best_callback = ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="best-step-{step:07d}",
                monitor="val/loss",
                mode="min",
                save_top_k=checkpoint_save_top_k,
                save_last=checkpoint_save_last,
                auto_insert_metric_name=False,
            )
            callbacks.append(checkpoint_best_callback)

        enable_csv_logger = bool(training_cfg.get("enable_csv_logger", True))
        csv_logger: CSVLogger | bool
        if enable_csv_logger:
            csv_logger = CSVLogger(save_dir=str(stage_output_dir), name="logs")
        else:
            csv_logger = False

        trainer = Trainer(
            max_steps=int(training_cfg.get("max_steps", 1500)),
            accelerator=accelerator,
            devices=devices,
            logger=csv_logger,
            callbacks=callbacks,
            default_root_dir=str(stage_output_dir),
            gradient_clip_val=float(training_cfg.get("grad_clip", 1.0)),
            precision=_resolve_precision(training_cfg, accelerator),
            log_every_n_steps=int(training_cfg.get("log_every_n_steps", 20)),
            val_check_interval=int(training_cfg.get("val_check_interval", 100)),
            check_val_every_n_epoch=None,
            num_sanity_val_steps=0,
        )
        trainer.fit(model, datamodule=data_module)

        val_loss_first = trend_callback.val_losses[0] if trend_callback.val_losses else None
        val_loss_last = trend_callback.val_losses[-1] if trend_callback.val_losses else None
        min_relative_drop = float(gate_cfg.get("min_relative_drop", 0.0))
        val_trend_pass = False
        if val_loss_first is not None and val_loss_last is not None:
            threshold = val_loss_first * (1.0 - min_relative_drop)
            val_trend_pass = bool(val_loss_last <= threshold)
        stable = model.nan_batches == 0
        require_stable = bool(gate_cfg.get("require_stable", True))
        require_downward_trend = bool(gate_cfg.get("require_downward_trend", True))
        gate_pass = (stable or not require_stable) and (val_trend_pass or not require_downward_trend)
        metrics_csv_path = str(Path(trainer.log_dir) / "metrics.csv") if trainer.log_dir else None
        processed_raw_tokens_target = (
            int(training_cfg.get("max_steps", 1500))
            * int(training_cfg.get("batch_size", 64))
            * int(training_cfg.get("compressed_seq_len", 8))
            * int(fp16_cfg.get("compression", {}).get("chunk_size_tokens", 64))
        )

        return {
            "stable": bool(stable),
            "nan_batches": int(model.nan_batches),
            "val_loss_first": val_loss_first,
            "val_loss_last": val_loss_last,
            "val_trend_pass": bool(val_trend_pass),
            "diffusion_enabled": diffusion_enabled,
            "val_diffusion_loss_last": (
                float(trainer.callback_metrics["val/diffusion_loss"].item())
                if diffusion_enabled and "val/diffusion_loss" in trainer.callback_metrics
                else None
            ),
            "val_gist_cosine_last": (
                float(trainer.callback_metrics["val/gist_cosine"].item())
                if diffusion_enabled and "val/gist_cosine" in trainer.callback_metrics
                else None
            ),
            "val_gist_retrieval_top1_last": (
                float(trainer.callback_metrics["val/gist_retrieval_top1"].item())
                if diffusion_enabled and "val/gist_retrieval_top1" in trainer.callback_metrics
                else None
            ),
            "first_batch_seconds": runtime_metrics.first_batch_seconds,
            "prep_seconds": prep_seconds,
            "steps_per_sec": runtime_metrics.steps_per_sec,
            "compressed_tokens_per_sec": runtime_metrics.compressed_tokens_per_sec,
            "raw_tokens_per_sec": runtime_metrics.raw_tokens_per_sec,
            "gpu_mem_peak_mb": runtime_metrics.gpu_mem_peak_mb,
            "cpu_util_avg": runtime_metrics.cpu_util_avg,
            "gpu_util_avg": runtime_metrics.gpu_util_avg,
            "device": str(trainer.strategy.root_device),
            "train_samples": len(data_module.train_dataset),
            "val_samples": len(data_module.val_dataset),
            "train_chunks": data_module.train_chunk_count,
            "val_chunks": data_module.val_chunk_count,
            "processed_raw_tokens_target": int(processed_raw_tokens_target),
            "cached_unique_raw_tokens_equivalent": int(cache_metadata.get("train_chunks", 0))
            * int(cache_metadata.get("chunk_size_tokens", 0)),
            "gate_pass": bool(gate_pass),
            "metrics_csv_path": metrics_csv_path,
            "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir.exists() else None,
            "best_checkpoint_path": (
                checkpoint_best_callback.best_model_path
                if checkpoint_best_callback is not None and checkpoint_best_callback.best_model_path
                else None
            ),
            "last_checkpoint_path": (
                checkpoint_best_callback.last_model_path
                if checkpoint_best_callback is not None and checkpoint_best_callback.last_model_path
                else None
            ),
            "last_periodic_checkpoint_path": (
                checkpoint_periodic_callback.last_model_path
                if checkpoint_periodic_callback is not None and checkpoint_periodic_callback.last_model_path
                else None
            ),
        }

    def run_stage(self, stage: str) -> dict[str, Any]:
        parse_stage_list([stage])
        stage_output_dir = self.context.stage_output_dir(stage)
        stage_output_dir.mkdir(parents=True, exist_ok=True)

        config = self._compose_config()
        OmegaConf.save(OmegaConf.create(config), stage_output_dir / "config.yaml")

        seed = config.get("experiment", {}).get("seed")
        if seed is not None:
            seed_everything(int(seed), workers=True)
            torch.manual_seed(int(seed))

        try:
            if stage == "preprocess":
                result = {"preprocess": self._preprocess_arrays(config=config, stage_output_dir=stage_output_dir)}
            elif stage == "train":
                result = {"train": self._train_from_cache(config=config, stage_output_dir=stage_output_dir)}
            else:
                raise ValueError(f"Unsupported stage: {stage}")

            payload = {
                "status": "ok",
                "run_id": self.context.run_id,
                "stage": stage,
                "output_dir": str(stage_output_dir),
                "config_name": self.context.config_name,
                "overrides": self.context.overrides,
                "config": config,
                "result": result,
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
            log.info("Starting fp16 modal stage: %s (run_id=%s)", stage, self.context.run_id)
            results["stages"][stage] = self.run_stage(stage)
        results["summary_path"] = str(self.context.summary_path)
        return results

