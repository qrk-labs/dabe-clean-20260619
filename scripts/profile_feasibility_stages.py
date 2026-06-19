#!/usr/bin/env python3
"""Run feasibility stages independently and emit optimization runtime metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.modal_feasibility_pipeline import FeasibilityRunContext, FeasibilityStageRunner, parse_stage_list


def _stage_metrics(payload: dict) -> dict:
    stage = payload.get("stage")
    result = (payload.get("result") or {}).get(stage, {})
    runtime = result.get("runtime", {})
    return {
        "stage": stage,
        "prep_seconds": runtime.get("prep_seconds"),
        "first_batch_seconds": runtime.get("first_batch_seconds"),
        "steps_per_sec": runtime.get("steps_per_sec"),
        "tokens_per_sec": runtime.get("tokens_per_sec"),
        "gpu_mem_peak_mb": runtime.get("gpu_mem_peak_mb"),
        "gpu_util_avg": runtime.get("gpu_util_avg"),
        "cpu_util_avg": runtime.get("cpu_util_avg"),
        "dataloader_wait_seconds": runtime.get("dataloader_wait_seconds"),
        "compile_graph_breaks": runtime.get("compile_graph_breaks"),
        "compile_warmup_seconds": runtime.get("compile_warmup_seconds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="feasibility_opt_v3")
    parser.add_argument("--run-id", default="opt_v3_profile_local")
    parser.add_argument("--overrides", default="")
    parser.add_argument("--stages", default="tokenizer,dabe_lm,bpe_baseline")
    parser.add_argument("--output-root", default="experiments")
    args = parser.parse_args()

    context = FeasibilityRunContext.create(
        repo_root=Path.cwd(),
        run_id=args.run_id,
        base_output_root=Path(args.output_root),
        config_name=args.config_name,
        overrides=args.overrides,
    )
    runner = FeasibilityStageRunner(context=context)

    metrics = []
    for stage in parse_stage_list(args.stages):
        payload = runner.run_stage(stage)
        metrics.append(_stage_metrics(payload))

    summary = {
        "run_id": context.run_id,
        "config_name": args.config_name,
        "metrics": metrics,
    }
    out_path = context.run_root / "optimization_profile_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
