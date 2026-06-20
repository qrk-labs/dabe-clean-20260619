# Analysis Notes

## Fixed Chunks As The Carrier

The architectural interpretation belongs in the method: DABE keeps the 64-token slab fixed and adapts the sparse repair mask over that slab. The variable-window experiments remain important evidence, and they are now presented as direct ablations of the carrier geometry. EXP-088, EXP-089, and EXP-090 test quantile-supervised routing, action-value routing, and hard straight-through routing respectively; none beats the fixed-chunk gist-residual lookup anchor.

Reviewer-facing conclusion: adaptive granularity is useful, but in this evidence stack it belongs in repair allocation, not in the primary token-window geometry.

## Cost-Aware Lookup Knee

EXP-091 and EXP-092 tested whether bluntly increasing sparse lookup cost can reduce bitrate without destroying reconstruction. The result is a clear local knee rather than a monotonic improvement.

![Fig. 3. Cost-knee curve](figures/fig03_cost_knee_curve.png)

### Replicated Knee

| Slot cost | observed bits/token | active K | token_acc | chunk deviation mean | chunk deviation p90 |
|-----------|--------------------:|---------:|----------:|---------------------:|--------------------:|
| `0.0225` | `20.26545` | `8.10659` | `0.89255` | `6.87713` | `11.14981` |
| `0.025` | `20.06207` | `6.63953` | `0.89274` | `6.86454` | `10.96876` |
| `0.0275` | `19.84857` | `4.54922` | `0.89002` | `7.03849` | `11.12805` |

`0.025` is the best local tradeoff: it has the best token accuracy, mean chunk deviation, and p90 deviation among the adjacent settings. Lower cost spends more lookup budget without improving quality; higher cost saves bits but begins to over-prune lexical repair.

### Interpretation

The cost term is doing useful work, but it is not the main source of quality. EXP-087's quality anchor remains substantially stronger:

| Setting | observed bits/token | token_acc | chunk deviation mean |
|---------|--------------------:|----------:|---------------------:|
| EXP-087 quality anchor, cost `0.02` | `20.37236` | `0.91547` | `5.41007` |
| EXP-092 cost-aware knee, cost `0.025` | `20.06207` | `0.89274` | `6.86454` |

This suggests that the next architectural improvement should not simply increase slot-cost pressure. It should improve router and lookup accuracy at a fixed or lower active K.

### Paper Placement

- Section 5 Results: include EXP-092 as a small replicated rate-distortion curve.
- Section 6 Analysis: describe `0.025` as the cost-aware knee and EXP-087 as the quality anchor.
- Limitations: cost-aware compression currently trades away too much quality relative to EXP-087.

## Fixed-Rate Comparator

EXP-093 provides the matched fixed-rate learned tokenizer baseline that the results table needed. It uses the same TinyStories setup and a no-lookup `hierarchical_local` decoder with `1280` code bits per 64-token chunk (`20.0` bits/token).

![Fig. 1. Matched 20bpt comparator](figures/fig01_matched_20bpt_comparator.png)

| Setting | bits/token | token_acc | chunk deviation mean | chunk deviation p90 |
|---------|-----------:|----------:|---------------------:|--------------------:|
| EXP-093 fixed-rate no-lookup | `20.0` | `0.72045` | `17.89119` | `24.14597` |
| EXP-092 cost-aware sparse repair | `20.06207` observed | `0.89274` | `6.86454` | `10.96876` |

This is the cleanest support for the central mechanism claim. The fixed-rate code is healthy rather than collapsed (`bit_density=0.48766`), but increasing no-lookup capacity to the same bitrate scale does not approach the adaptive repair model. The gain therefore comes from *where* the model spends lexical precision, not only from *how many* bits it spends.

## Active Repair Budget

![Fig. 6. Active repair budget vs distortion](figures/fig06_active_k_vs_quality.png)

The active-K plot shows why budget alone is not enough. More active slots often help, but the best points are not simply the largest-K points. The gist-residual models reduce distortion by spending fewer slots more selectively, which is the core mechanism behind the paper's "adaptive repair budget" framing.

## Runtime Cost

The sparse-repair path has a measurable training cost. EXP-093's fixed-rate no-lookup run logged `14.34` steps/s on T4, while EXP-092's cost-aware sparse-repair sweep logged `7.80` steps/s on the first child. With batch size `32` and chunk size `64`, that corresponds to about `29.4k` tokens/s versus `16.0k` tokens/s. EXP-094's decode diagnostic processed `13.46k` tokens/s with CUDA peak allocated/reserved memory of `2031.74`/`2152.00` MiB.

## Code-Like Text Diagnostics

Python code is a useful stress test because its hard tokens differ from TinyStories: identifiers, indentation, delimiters, operators, and literals carry exact meaning. The diagnostic data path now supports a deterministic Python-code corpus, so a future checkpoint probe can test whether repair slots move from narrative names/content words toward syntax-critical tokens. This should be treated as a diagnostic extension until we run it on the final checkpoint.
