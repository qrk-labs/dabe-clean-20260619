# Limitations

## Tokenizer-Autoencoder Scope

This study is intentionally framed as tokenizer-autoencoder rate-distortion rather than full downstream language-model pretraining. That choice isolates the tokenization mechanism: the experiments measure how chunk codes, sparse repair, and routing policies affect reconstruction before adding the confounds of language-model scale, optimizer schedules, and downstream task selection. The matched 20 bits/token comparison is therefore the central evidence unit for the paper.

## Controlled Dataset Scope

The main experiments use TinyStories as a controlled text domain with exact GPT-2-token reconstruction metrics. This setting makes it possible to compare architectural variants under the same token basis, chunk length, training budget, and validation protocol. Domain generalization remains a separate evaluation target; this setting establishes a clean mechanism result that should next be tested on broader corpora, code, multilingual data, and full LM training. EXP-097 makes this boundary explicit: the EXP-087 TinyStories-trained checkpoint reaches only `0.41761` token accuracy on a deterministic Python-code corpus, so the current evidence should not be read as zero-shot code-tokenizer performance.

## Focused Run Design

The run plan prioritizes focused, objective comparisons over broad exploratory sweeps. The experiment set includes a matched fixed-rate comparator, a replicated cost-knee neighborhood around EXP-092, a quality anchor in EXP-087, and direct variable-window ablations. Additional seeds and larger-scale runs would support confidence intervals and scaling laws, but they are not required to interpret the main matched-rate separation.

## Rate-Distortion Rather Than Losslessness

The current models are rate-distortion tokenizers, not lossless compressors. Exact full-chunk reconstruction remains low, while token accuracy, exact local blocks, and mean chunk deviation improve substantially with sparse repair. This is the correct operating point for the paper's claim: DABE demonstrates that adaptive repair allocation can sharply reduce reconstruction distortion at a matched observed bitrate.
