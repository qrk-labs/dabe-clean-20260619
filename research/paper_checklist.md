> **Development freeze:** DABE is frozen as the publication artifact for the preprint at https://doi.org/10.5281/zenodo.20797432. Do not run new experiments or alter scientific claims unless the user explicitly unfreezes development. See [`FREEZE_NOTICE.md`](../FREEZE_NOTICE.md).

# DABE Paper Checklist

## Goal

Build a publication-ready evidence pack for the FP16 chunk-scalar technique with clear quality/throughput tradeoffs, strong reproducibility, and matched baselines.

## Execution Order

- [ ] **1) Evaluation Hardening (must-do first)**
  - [x] Replace heuristic-only coherence with stricter quality checks.
  - [x] Track mojibake rate, diversity, repetition, and prompt relevance.
  - [x] Add a fixed judge set (100 prompts) and rubric-based judge runner.
  - [ ] Freeze this evaluation protocol and reuse it across all subsequent experiments.

- [ ] **2) Pareto Frontier Sweep (core systems claim)**
  - [ ] Keep model/codec fixed.
  - [ ] Sweep 4-6 saturation profiles from quality-first to throughput-first.
  - [ ] Report frontier: quality vs throughput vs memory.

- [ ] **3) Update-Count vs Batch-Size Disentanglement**
  - [ ] Hold total processed tokens fixed.
  - [ ] Vary update count independently from per-step workload.
  - [ ] Show whether quality shifts come from fewer updates or batch regime itself.

- [ ] **4) Unique-vs-Repeated Exposure Grid (post-fix only)**
  - [ ] Run corrected grid: `2M/3B`, `2M/10B`, `10M/10B`, `20M/10B`.
  - [ ] Use best-checkpoint selection, not only last checkpoint.
  - [ ] Compare interaction and gist metrics across the grid.

- [ ] **5) Decode-Path Ablation (core quality bottleneck)**
  - [ ] Compare current nearest-neighbor decode to improved decode alternatives.
  - [ ] Quantify collapse reduction and semantic consistency improvements.

- [ ] **6) Matched Baselines**
  - [ ] Add equal-budget BPE transformer baseline.
  - [ ] Add compressed-scalar without diffusion baseline.
  - [ ] Ensure same data budget and reporting protocol.

- [ ] **7) Reproducibility / Variance**
  - [ ] Run 3 seeds on top quality profile and top saturation profile.
  - [ ] Report mean/std for primary metrics.

- [ ] **8) Cost-Efficiency Analysis**
  - [ ] Report wall-clock, tokens/sec, memory footprint, and cost per run.
  - [ ] Plot quality-per-dollar and throughput-per-dollar.

## Reporting Rules

- [ ] Every checklist item must map to `EXP-XXX` entries in `research/experiment_log.md`.
- [ ] Every completed item must include pass/fail rationale and next action.
- [ ] No paper claims without at least one matched baseline and one variance check.
