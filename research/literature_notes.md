> **Development freeze:** DABE is frozen as the publication artifact for the preprint at https://doi.org/10.5281/zenodo.20797432. Do not run new experiments or alter scientific claims unless the user explicitly unfreezes development. See [`FREEZE_NOTICE.md`](../FREEZE_NOTICE.md).

## <Author, Year> — <Short Title>
- **Venue:** <conference/journal>
- **Key Idea:** <1-2 sentences>
- **Relevance to DABE:** <why we care>
- **Borrow from:** <specific technique/equation we want to adapt>
- **Cite as:** <full citation string>
```

### Paper Draft Organization (`research/paper_drafts/`)

```
paper_drafts/
├── 00_abstract.md
├── 01_introduction.md
├── 02_related_work.md
├── 03_methodology.md
├── 04_experiments.md
├── 05_results.md
├── 06_analysis.md
├── 07_conclusion.md
└── figures/
    └── (plot scripts and saved figures)
```

Each section file should be continuously updated as experiments complete. Do NOT wait until the end to write — draft as you go.

---

## Paper Section Mapping

| Section | What Feeds It | Current Status |
|---------|---------------|----------------|
| 1. Introduction | Motivation, problem statement | Not started |
| 2. Related Work | BitLM, MaskBit, SemToken, H-Net, NASH, SONAR, LaBSE, AdaTok, UTF8Tokenizer | Literature notes partial |
| 3. Methodology | Density router design, LFQ/semantic hash encoder, variable-width projection, training objectives | Not started |
| 4. Experimental Setup | Datasets, baselines, hyperparameters, compute | Not started |
| 5. Results | Main benchmarks, compression ratios, multilingual alignment | Not started |
| 6. Analysis | Hamming distance structure, density allocation visualizations, ablations | Not started |
| 7. Conclusion | Summary, limitations, future work | Not started |

---

## Ablation Protocol

Ablations are mandatory before claiming any result. Minimum ablation set:

1. **Fixed-width vs variable-width:** Does density adaptation actually help?
2. **Bit-width sweep:** 8, 16, 24, 32, 48 bits — where is the knee?
3. **Density signal source:** Entropy-based vs learned router vs uniform baseline
4. **Number of density levels:** 2 (fine/coarse) vs 4 (fine/medium/coarse/filler)
5. **Multilingual objective:** With vs without contrastive alignment loss
6. **Binary code learning:** LFQ vs Gumbel-Softmax vs VAE (semantic hash)

Each ablation gets its own experiment log entry and its own Hydra config.

---

## Milestones

- [ ] **M1:** Skeleton project with shape-validated dummy forward pass
- [ ] **M2:** Fixed-width 32-bit LFQ encoder training on single-language data
- [ ] **M3:** Add contrastive multilingual alignment loss
- [ ] **M4:** Add density router (start with entropy-based)
- [ ] **M5:** Variable-width encoding with density signal
- [ ] **M6:** Full ablation suite
- [ ] **M7:** Baseline comparisons (BPE, BitLM, SemToken)
- [ ] **M8:** Paper draft complete

---

## Git Conventions

- `main` — stable, reproducible experiments only
- `exp/<desc>` — experiment branches
- Commit messages: `[EXP-XXX] <description>`
- Never force-push `main`
- Tag stable checkpoints: `v0.1-m1`, `v0.2-m2`, etc.

---

## Reminders

- If uncertain about a design choice, run a small-scale pilot FIRST (1M tokens, 1 layer, 1 language). Log it. Decide from data.
- When in doubt, measure. When not in doubt, measure anyway.
- A 2-page writeup of a negative result is better than a 10-page writeup of an unvalidated claim.