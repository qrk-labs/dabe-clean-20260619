#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_stage_train(run_root: Path) -> dict[str, Any]:
    candidates = [run_root / "stage_result.json", run_root / "pipeline_summary.json"]
    for candidate in candidates:
        if candidate.exists():
            payload = _load_json(candidate)
            result = payload.get("result", {})
            train = result.get("train", {})
            if isinstance(train, dict):
                return train
    raise RuntimeError(f"Missing stage_result.json/pipeline_summary.json under: {run_root}")


def _load_best_probe(run_root: Path) -> tuple[dict[str, Any], Path]:
    probe_files = sorted(run_root.glob("interaction_probe*.json"))
    best: tuple[dict[str, Any], Path] | None = None
    for probe_path in probe_files:
        payload = _load_json(probe_path)
        if str(payload.get("decode_mode", "")) != "mirror_decoder":
            continue
        if bool(payload.get("uses_train_memory_lookup", False)):
            continue
        score = float(payload.get("strict_coherent_rate", 0.0))
        if best is None or score > float(best[0].get("strict_coherent_rate", 0.0)):
            best = (payload, probe_path)
    if best is None:
        raise RuntimeError(f"No mirror_decoder no-memory probe found under: {run_root}")
    return best


def _rank_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row.get("strict_coherent_rate", 0.0)),
        float(row.get("coherent_rate", 0.0)),
        float(row.get("val_next_token_acc_last", 0.0)),
        float(row.get("raw_tokens_per_sec", 0.0)),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", action="append", required=True)
    parser.add_argument(
        "--baseline-exp056-stage",
        default="experiments/modal_downloads/exp056_modal_hier_2m1b_001/stage_result.json",
    )
    parser.add_argument(
        "--baseline-exp057-stage",
        default="experiments/modal_downloads/exp057_modal_hier_2m1b_singledecode_001/stage_result.json",
    )
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows: list[dict[str, Any]] = []
    for run_root_raw in args.run_root:
        run_root = Path(run_root_raw)
        train = _load_stage_train(run_root)
        probe, probe_path = _load_best_probe(run_root)
        rows.append(
            {
                "run_root": str(run_root),
                "probe_path": str(probe_path),
                "strict_coherent_rate": float(probe.get("strict_coherent_rate", 0.0)),
                "coherent_rate": float(probe.get("coherent_rate", 0.0)),
                "val_next_token_acc_last": float(train.get("val_next_token_acc_last", 0.0) or 0.0),
                "raw_tokens_per_sec": float(train.get("raw_tokens_per_sec", 0.0) or 0.0),
                "val_next_chunk_acc_last": float(train.get("val_next_chunk_acc_last", 0.0) or 0.0),
                "output_head_family": str(train.get("output_head_family", "")),
            }
        )

    ranked = sorted(rows, key=_rank_key, reverse=True)
    top_k = max(1, int(args.top_k))
    selected = ranked[:top_k]

    exp056_payload = _load_json(Path(args.baseline_exp056_stage))
    exp057_payload = _load_json(Path(args.baseline_exp057_stage))
    exp056_next_token_acc = float(
        exp056_payload.get("result", {}).get("train", {}).get("val_next_token_acc_last", 0.0) or 0.0,
    )
    exp057_tps = float(
        exp057_payload.get("result", {}).get("train", {}).get("raw_tokens_per_sec", 0.0) or 0.0,
    )

    finalists: list[dict[str, Any]] = []
    for row in selected:
        relative_token_acc_gain = (
            ((row["val_next_token_acc_last"] - exp056_next_token_acc) / exp056_next_token_acc)
            if exp056_next_token_acc > 0
            else 0.0
        )
        throughput_ratio = (row["raw_tokens_per_sec"] / exp057_tps) if exp057_tps > 0 else 0.0
        passes = (
            row["strict_coherent_rate"] >= 0.25
            and row["coherent_rate"] >= 0.50
            and relative_token_acc_gain >= 0.25
            and throughput_ratio >= 0.75
        )
        finalists.append(
            {
                **row,
                "relative_token_acc_gain_vs_exp056": relative_token_acc_gain,
                "throughput_ratio_vs_exp057": throughput_ratio,
                "passes_final_gate": bool(passes),
            },
        )

    payload = {
        "rank_order": ranked,
        "selected_top_k": finalists,
        "baselines": {
            "exp056_val_next_token_acc_last": exp056_next_token_acc,
            "exp057_raw_tokens_per_sec": exp057_tps,
        },
    }
    if str(args.output).strip():
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"Saved ranking: {output_path}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
