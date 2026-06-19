# Working Abstract

Modern subword tokenizers are efficient and robust, but they allocate lexical capacity through a fixed segmentation policy rather than through the local information demand of a span. We study Density-Adaptive Bitmask Encoding (DABE), a learned tokenizer-autoencoder that compresses fixed 64-token chunks into binary codes and spends additional sparse lexical repair capacity only where reconstruction is difficult.

Across TinyStories tokenizer-autoencoder experiments, we find that the most effective design is not adaptive token-window geometry. Instead, a stable fixed chunk paired with a coarse gist stream and a residual-routed sparse lookup stream gives the best rate-distortion tradeoff. The best quality setting reaches `0.91547` token accuracy and `5.41007` mean chunk deviation at `20.37236` observed bits/token. A replicated cost-aware setting identifies `lookup_slot_cost_weight=0.025` as a local compression knee, reaching `0.89274` token accuracy and `6.86454` mean chunk deviation at `20.06207` observed bits/token.

Negative ablations show that sliding or variable token windows either reduce bitrate at substantial reconstruction cost or collapse to all-fine routing. Matched fixed-rate baselines show that sparse lexical repair improves reconstruction beyond simply adding more no-lookup code bits. These results suggest that learned tokenizers should adapt the repair/attention budget over stable chunks, rather than primarily varying token-window size.

The remaining publication-critical baseline is a fixed-rate no-lookup learned chunk tokenizer near `20` bits/token, which would directly compare against the cost-aware DABE operating point.
