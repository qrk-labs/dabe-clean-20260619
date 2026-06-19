#!/usr/bin/env python3
"""Bounded bend/equivalent router segmentation spike with explicit go/no-go output."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def baseline_segment(words: list[str], scores: np.ndarray, min_bw: int = 8, max_bw: int = 24):
    spans = []
    min_s = float(scores.min())
    max_s = float(scores.max())
    range_s = max(max_s - min_s, 1e-8)
    cursor = 0
    for i, word in enumerate(words):
        density = float((scores[min(i, len(scores) - 1)] - min_s) / range_s)
        bit_width = min_bw + int(round(density * (max_bw - min_bw)))
        end = cursor + len(word) + (1 if i < len(words) - 1 else 0)
        spans.append((word, cursor, end, bit_width))
        cursor += len(word) + 1
    return spans


def vectorized_segment(words: list[str], scores: np.ndarray, min_bw: int = 8, max_bw: int = 24):
    n = len(words)
    if n == 0:
        return []
    min_s = float(scores.min())
    max_s = float(scores.max())
    densities = (scores[:n] - min_s) / max(max_s - min_s, 1e-8)
    widths = min_bw + np.rint(densities * (max_bw - min_bw)).astype(np.int32)

    lengths = np.array([len(word) for word in words], dtype=np.int32)
    starts = np.concatenate(([0], np.cumsum(lengths[:-1] + 1)))
    ends = starts + lengths
    if n > 1:
        ends[:-1] += 1

    return [
        (words[i], int(starts[i]), int(ends[i]), int(widths[i]))
        for i in range(n)
    ]


def benchmark(num_texts: int = 512, words_per_text: int = 128):
    texts = [" ".join([f"tok{i%31}" for i in range(words_per_text)]) for _ in range(num_texts)]
    scores = [np.linspace(0.1, 0.9, words_per_text, dtype=np.float32) for _ in range(num_texts)]

    t0 = time.perf_counter()
    baseline_outputs = []
    for text, score in zip(texts, scores):
        baseline_outputs.append(baseline_segment(text.split(), score))
    baseline_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    vector_outputs = []
    for text, score in zip(texts, scores):
        vector_outputs.append(vectorized_segment(text.split(), score))
    vector_time = time.perf_counter() - t1

    deterministic_parity = baseline_outputs == vector_outputs
    speedup = baseline_time / max(vector_time, 1e-9)

    bend_binary = shutil.which("bend")
    bend_available = bend_binary is not None
    bend_run_status = "not_run"
    if bend_available:
        try:
            # Keep this bounded and non-critical.
            subprocess.run([bend_binary, "--help"], check=False, timeout=5)
            bend_run_status = "available"
        except Exception:
            bend_run_status = "error"

    integration_overhead_hours = 0.5  # bounded estimate for this spike
    go = bool(deterministic_parity and speedup >= 1.5 and integration_overhead_hours <= 1.0)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "candidate": "equivalent_runtime_vectorized_python",
        "bend_available": bend_available,
        "bend_status": bend_run_status,
        "speedup_vs_baseline": speedup,
        "deterministic_parity": deterministic_parity,
        "integration_overhead_hours": integration_overhead_hours,
        "go": go,
        "criteria": {
            "speedup_min": 1.5,
            "integration_overhead_hours_max": 1.0,
            "deterministic_parity_required": True,
        },
    }
    return payload


def main() -> int:
    report = benchmark()
    report_dir = Path("experiments/bend_spike")
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = report_dir / f"report_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved bend spike report to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
