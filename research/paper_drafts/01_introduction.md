> **Development freeze:** DABE is frozen as the publication artifact for the preprint at https://doi.org/10.5281/zenodo.20797432. Do not run new experiments or alter scientific claims unless the user explicitly unfreezes development. See [`FREEZE_NOTICE.md`](../../FREEZE_NOTICE.md).

# Introduction

Subword tokenizers define the discrete interface between text and modern language models. Before a model sees language, text is committed to a sequence of symbols chosen by a tokenizer, usually a subword segmentation scheme such as byte-pair encoding or a related unigram model. These methods are efficient, robust, and easy to deploy, but their rate allocation is primarily corpus-level: frequent fragments receive compact symbols, rare lexical details are decomposed into longer sequences, and local span difficulty is not represented as an explicit allocation decision.

Natural text has nonuniform information density. A span may contain predictable function words, repeated narrative scaffolding, rare names, punctuation-sensitive dialogue, or local details that are easy to paraphrase but hard to reconstruct exactly. This creates a tokenizer-level rate-distortion question: which parts of a span require lexical precision, and which parts can be represented by a coarse semantic gist?

We study this question through Density-Adaptive Bitmask Encoding (DABE), a learned tokenizer-autoencoder that compresses fixed 64-token chunks into binary codes and reconstructs the original token sequence. The goal is not merely to replace one vocabulary with another. Instead, DABE treats tokenization as a rate-distortion problem: given a finite bit budget, how should a learned tokenizer allocate capacity across a span so that reconstruction error falls where it matters most?

The central design choice in this work is to decouple coarse meaning from lexical repair. A DABE chunk first receives a compact gist representation intended to capture broad local structure. A second, sparse repair pathway then allocates additional lexical lookup capacity to positions predicted to be difficult. This produces a "sliding attention budget" over a stable chunk, rather than a sliding token window. The distinction matters: changing the primary token-window geometry can reduce nominal bitrate, but it can also disrupt local code specificity and make reconstruction brittle. In contrast, keeping the chunk stable while adapting the repair budget lets the model preserve a consistent coarse representation and spend precision only where residual error suggests it is useful.

Our experiments on TinyStories establish this framing empirically. The strongest setting, a fixed 64-token chunk with gist encoding and residual-routed sparse lexical lookup, reaches `0.91547` token accuracy and `5.41007` mean chunk deviation at `20.37236` observed bits/token. A cost-aware operating point replicated across adjacent cost settings reaches `0.89274` token accuracy and `6.86454` mean chunk deviation at `20.06207` observed bits/token. A matched fixed-rate no-lookup baseline at `20.0` bits/token is stable and non-collapsed, but reaches only `0.72045` token accuracy and `17.89119` mean chunk deviation. The result is a matched-rate separation: the gain is not explained by total bit budget alone, but by the model's ability to place lexical precision where reconstruction requires it.

This paper makes four contributions:

- We formulate DABE tokenization as a learned rate-distortion problem over fixed token chunks, with binary codes and explicit reconstruction metrics.
- We introduce a gist-plus-residual-repair architecture that separates coarse span representation from sparse lexical precision.
- We show that adaptive sparse repair substantially outperforms a matched fixed-rate no-lookup learned tokenizer at approximately `20` bits/token.
- We provide controlled ablations showing that adaptive token-window geometry is not, by itself, the right mechanism for this setting.

The paper therefore makes a specific rate-distortion claim: within a controlled tokenizer-autoencoder setting, adaptive sparse lexical repair over stable chunks gives a stronger reconstruction frontier than either fixed-rate learned compression or adaptive token-window geometry. This is the mechanism result that future full language-model integration should build on.
