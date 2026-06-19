# Analysis Notes

## Anticipated Question: Would Sliding Or Variable Token Windows Have Worked?

Short answer: we tested this directly in EXP-088, EXP-089, and EXP-090. The result is a useful negative finding. Variable token-window geometry improved some local routing diagnostics, but did not beat the simpler EXP-087 gist-residual lookup architecture on the global rate-distortion objective.

### Baseline To Beat

EXP-087 used a fixed 64-token chunk with a coarse gist decoder and a residual-routed sparse lexical lookup. The winning setting was:

- `decoder_mode=gist_residual_lookup`
- `code_bits=1024`
- `hierarchical_block_tokens=16`
- `lexical_lookup_k=32`
- `lexical_lookup_slot_policy=halting`
- `residual_router_loss_weight=0.2`
- `lookup_slot_cost_weight=0.02`

This produced the current anchor result:

| Metric | EXP-087 Best |
|--------|--------------|
| token accuracy | `0.91547` |
| mean chunk deviation | `5.41007` |
| p90 chunk deviation | `9.04678` |
| observed effective bits/token | `20.37236` |
| soft lookup K | `12.71961` |
| threshold-active K | `9.03257` |

The key idea was not to change token geometry. Instead, the model learned where to spend a sparse repair budget after producing a coarse whole-chunk gist.

### Variable-Window Detour

The variable-window line tested whether we could do better by letting the tokenizer choose fine/medium/full windows inside a fixed 64-token slab.

| Experiment | Core Change | token_acc | chunk deviation mean | observed bits/token | Outcome |
|------------|-------------|-----------|----------------------|---------------------|---------|
| EXP-088 | quantile-supervised variable windows | `0.85820` | `9.07550` | `19.46092` | lower bitrate, large quality drop |
| EXP-089 | action-value teacher from measured candidate deviation | `0.87508` | `7.99482` | `18.29175` | better than EXP-088, still far behind EXP-087 |
| EXP-090 | straight-through hard window routing | `0.87209` | `8.18653` | `21.70567` | completed cleanly but collapsed to all-fine routing |
| EXP-087 | fixed chunk + residual lookup | `0.91547` | `5.41007` | `20.37236` | best rate-distortion anchor |

### What We Learned

- EXP-088 answered the naive version of the question. Quantile-based window targets were not enough: the router collapsed to fine-window argmax while soft probabilities still claimed a lower expected bitrate.
- EXP-089 answered the fairer version. We replaced quantile targets with an action-value teacher that measured actual block Hamming deviation for fine/medium/full candidate decodes. This improved EXP-088 substantially, but reconstruction remained far behind EXP-087.
- EXP-090 answered the discrete-routing objection. We replaced soft window mixing with straight-through hard routing. This reduced local action regret, but the router collapsed to all-fine windows, erased the bitrate advantage, and still did not recover EXP-087 quality.
- The evidence points away from sliding token-window size as the primary mechanism. The stronger formulation is a fixed stable chunk representation plus a learned, density-adaptive attention/repair budget.

### Reviewer-Facing Answer

If asked "Would a sliding-window tokenizer have worked?", the answer should be:

> We tested adaptive token-window granularity in three variants: quantile-supervised routing, action-value routing based on measured candidate distortion, and hard straight-through routing. These variants either reduced bitrate at substantial reconstruction cost or collapsed to all-fine routing. The best variable-window result reached `0.87508` token accuracy and `7.99482` mean chunk deviation, still well behind the fixed-window gist-residual lookup baseline at `0.91547` token accuracy and `5.41007` mean chunk deviation. This motivated the paper's focus on sliding attention/repair budget rather than sliding token window size.

### Paper Placement

- Section 5 Results: include the table above as an ablation/negative-result summary.
- Section 6 Analysis: use this as evidence that variable granularity should be applied to repair budget, not to the primary token-window geometry.
- Limitations/Future Work: mention cost-aware hard routing as a possible future variant, but do not present it as the main path.

## Cost-Aware Lookup Knee

EXP-091 and EXP-092 tested whether bluntly increasing sparse lookup cost can reduce bitrate without destroying reconstruction. The result is a clear local knee rather than a monotonic improvement.

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

| Setting | bits/token | token_acc | chunk deviation mean | chunk deviation p90 |
|---------|-----------:|----------:|---------------------:|--------------------:|
| EXP-093 fixed-rate no-lookup | `20.0` | `0.72045` | `17.89119` | `24.14597` |
| EXP-092 cost-aware sparse repair | `20.06207` observed | `0.89274` | `6.86454` | `10.96876` |

This is the cleanest support for the central mechanism claim. The fixed-rate code is healthy rather than collapsed (`bit_density=0.48766`), but increasing no-lookup capacity to the same bitrate scale does not approach the adaptive repair model. The gain therefore comes from *where* the model spends lexical precision, not only from *how many* bits it spends.
