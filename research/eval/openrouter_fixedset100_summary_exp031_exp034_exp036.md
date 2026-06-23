> **Development freeze:** DABE is frozen as the publication artifact for the preprint at https://doi.org/10.5281/zenodo.20797432. Do not run new experiments or alter scientific claims unless the user explicitly unfreezes development. See [`FREEZE_NOTICE.md`](../../FREEZE_NOTICE.md).

# OpenRouter Fixed-Set(100) Summary: EXP-031 vs EXP-034 vs EXP-036

| Run | Coherent | Strict | Unique | Dominant | Mojibake | Heuristic Rubric | LLM Rubric | LLM Pass | Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exp031_modal_fp16_2m3b_optfix_001 | 0.90 | 0.44 | 63 | 0.07 | 49 | 6.69 (0.50) | 2.41 | 0.03 | 14.79 |
| exp034_modal_fp16_2m3b_sat_v2_001 | 0.93 | 0.07 | 33 | 0.07 | 75 | 6.17 (0.25) | 2.07 | 0.00 | 14.64 |
| exp036_modal_fp16_2m32b_8gb_v2_001 | 0.96 | 0.00 | 22 | 0.14 | 100 | 5.86 (0.00) | 1.23 | 0.00 | 11.20 |

Sorted by `llm_rubric_avg_score` (descending).
