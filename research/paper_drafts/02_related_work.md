# Related Work

## Subword Tokenization

Most contemporary language models use fixed tokenization policies such as byte-pair encoding, WordPiece, unigram language-model tokenization, or byte-level variants. These methods are robust and efficient, but they allocate sequence length through corpus-level frequency statistics rather than through the local information density of a span. DABE keeps the deployment motivation of compact discrete representations, but studies whether a learned tokenizer can allocate reconstruction capacity adaptively inside a fixed chunk.

## Learned Discrete Tokenizers

Discrete autoencoders, vector-quantized models, residual quantizers, and semantic hashing methods learn compact latent codes that can reconstruct or represent text. These approaches motivate treating tokenization as a learned compression problem rather than a fixed segmentation problem. The fixed-rate no-lookup DABE baselines in EXP-071, EXP-079, and EXP-093 play this role in our experiments: they ask how much reconstruction quality a learned chunk code obtains when it spends all capacity uniformly.

## Adaptive Computation And Routing

Adaptive computation mechanisms spend more computation on difficult examples, tokens, or positions. DABE applies the same broad principle to tokenizer reconstruction capacity: the model predicts where sparse lexical repair is worth spending after producing a coarse gist of the chunk. This differs from adaptive windowing because the primary chunk geometry remains stable while the repair budget changes.

## Retrieval, Copy, And Lexical Memory

Copy and retrieval mechanisms are often used to recover rare or exact lexical details that parametric sequence models blur. DABE's sparse lexical lookup stream plays a similar role inside the tokenizer itself: it gives the decoder a small budget for high-precision lexical holes while leaving easier positions to the gist representation.

## Positioning

The closest comparator for the present paper is a fixed-rate learned chunk tokenizer under the same bit budget. EXP-093 provides that comparator at `20.0` bits/token. BPE and SentencePiece remain important deployment references, while this paper isolates tokenizer-autoencoder rate-distortion before full language-model pretraining.
