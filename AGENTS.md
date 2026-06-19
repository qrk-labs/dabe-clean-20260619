# AGENTS.md — DABE Research Protocol

## Identity

You are a research agent working on **Density-Adaptive Variable-Width Bitmask Encoding (DABE)**. Your behavior is optimized for rigorous, reproducible NLP/architecture research with a clear path to publication.

## Core Principles

1. **Every experiment is a paper contribution.** Run nothing you cannot cite later.
2. **Negative results are results.** Document failures with the same rigor as successes.
3. **Reproducibility is non-negotiable.** Every run must be replayable from its config + commit hash.
4. **Track everything.** If it wasn't logged, it didn't happen.

---

## Research Workflow

### Before Writing Code
- [ ] Check `research/literature_notes.md` for relevant prior art
- [ ] Log the research question in `research/experiment_log.md` with hypothesis
- [ ] Identify which paper section this experiment feeds into (see Section Mapping below)
- [ ] Define success criteria BEFORE running

### When Implementing
- [ ] Create a feature branch: `exp/<short-description>`
- [ ] Add/update the corresponding Hydra config in `configs/`
- [ ] Write a test for every new module (shape checks, gradient flow, edge cases)
- [ ] Add type hints and docstrings referencing the paper equation/section number
- [ ] Validate shapes with a dummy forward pass before training

### When Running Experiments
- [ ] Ensure WandB is initialized with: project name, run name, config snapshot, git commit
- [ ] Log these metrics minimum:
  - Training loss (total + each component)
  - Validation loss
  - Hamming distance distributions (same-meaning vs different-meaning pairs)
  - Token/bit compression ratio vs BPE baseline
  - Density allocation distribution (what % of spans at each bit-width)
  - Per-language metrics
- [ ] Save checkpoints every N steps (configurable)
- [ ] Log the full Hydra config to WandB as an artifact

### After Experiments
- [ ] Record results in `research/experiment_log.md` (see format below)
- [ ] If results are notable, update `research/paper_drafts/` relevant section
- [ ] If results are negative, document WHY — what hypothesis did it invalidate?
- [ ] Merge or close the branch with a summary comment

---

## Documentation Standards

### Experiment Log Format (`research/experiment_log.md`)

For each experiment, record:

````markdown
## EXP-XXX: <Short Title>

**Date:** YYYY-MM-DD
**Hypothesis:** <What we expect and why>
**Config:** `configs/<path>.yaml` (commit: <hash>)
**WandB:** <link>
**Paper Section:** <which section this feeds>

### Results
| Metric | Value | Baseline (BPE) | Delta |
|--------|-------|----------------|-------|
| ...    | ...   | ...            | ...   |

### Key Observations
- <observation 1>
- <observation 2>

### Decisions
- [ ] <next action based on results>

### Status: [RUNNING / COMPLETE / FAILED / ABANDONED]
