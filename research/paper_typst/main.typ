// DEVELOPMENT FREEZE: DABE is frozen as the publication artifact for
// https://doi.org/10.5281/zenodo.20797432. Do not alter scientific claims,
// experiments, or paper content unless the user explicitly unfreezes development.

#set page(margin: 1in)
#set text(size: 10.5pt)
#set heading(numbering: "1.")
#set math.equation(numbering: "(1)")
#show heading: it => block(
  above: if it.level == 1 { 1.35em } else { 1.0em },
  below: if it.level == 1 { 0.8em } else { 0.55em },
)[#it]

#let dabe = smallcaps[DABE]
#let bpt = [bits/token]
#let exp(body) = smallcaps[EXP-#body]

#align(center)[
  #text(size: 18pt, weight: "bold")[Density-Adaptive Bitmask Encoding: Fixed Chunks with Sparse Lexical Repair]

  #v(0.35em)
  #text(size: 9.4pt)[Mainasara Al-amin Tsowa ]#text(size: 9.4pt, "<mainasara@qrk.ng>")
  #v(0.08em)
  #text(size: 9.4pt)[Babangida Usman Tsowa ]#text(size: 9.4pt, "<babangida@qrk.ng>")
  #v(0.08em)
  #text(size: 9.4pt)[Abdul-malik Abdullahi Mustapha ]#text(size: 9.4pt, "<maleek@qrk.ng>")
]

#v(1em)

#block(inset: 10pt, fill: luma(96%), stroke: luma(82%), radius: 3pt)[
  #strong[Abstract.] Modern subword tokenizers are efficient and robust, but they allocate lexical capacity through a fixed segmentation policy rather than through the local information demand of a span. We study Density-Adaptive Bitmask Encoding (#dabe), a learned tokenizer-autoencoder that compresses fixed 64-token chunks into binary codes and spends additional sparse lexical repair capacity only where reconstruction is difficult.

  Across TinyStories tokenizer-autoencoder experiments, we find that the most effective design is not adaptive token-window geometry. Instead, a stable fixed chunk paired with a coarse gist stream and a residual-routed sparse lookup stream gives the best rate-distortion tradeoff. The best quality setting reaches 0.91547 token accuracy and 5.41007 mean chunk deviation at 20.37236 observed #bpt. A replicated cost-aware setting identifies `lookup_slot_cost_weight=0.025` as a local compression knee, reaching 0.89274 token accuracy and 6.86454 mean chunk deviation at 20.06207 observed #bpt.

  Controlled ablations show that sliding or variable token windows either reduce bitrate at substantial reconstruction cost or collapse to all-fine routing. A matched fixed-rate no-lookup baseline at 20.0 #bpt reaches 0.72045 token accuracy and 17.89119 mean chunk deviation; a longer convergence-defense run improves this to 0.77324 and 14.51295, but remains far behind the cost-aware #dabe point's 0.89274 token accuracy and 6.86454 mean chunk deviation at 20.06207 observed #bpt. These results identify stable chunks plus adaptive repair as the stronger learned-tokenizer mechanism at matched bitrate.
]

= Introduction

Subword tokenizers define the discrete interface between text and modern language models. Before a model sees language, text is committed to a sequence of symbols chosen by a tokenizer, usually a subword segmentation scheme such as byte-pair encoding or a related unigram model @sennrich2016bpe @kudo2018sentencepiece. These methods are efficient, robust, and easy to deploy, but their rate allocation is primarily corpus-level: frequent fragments receive compact symbols, rare lexical details are decomposed into longer sequences, and local span difficulty is not represented as an explicit allocation decision.

Natural text has nonuniform information density. A span may contain predictable function words, repeated narrative scaffolding, rare names, punctuation-sensitive dialogue, or local details that are easy to paraphrase but hard to reconstruct exactly. This creates a tokenizer-level rate-distortion question: which parts of a span require lexical precision, and which parts can be represented by a coarse semantic gist?

We study this question through #dabe, a learned tokenizer-autoencoder that compresses fixed 64-token chunks into binary codes and reconstructs the original token sequence. The goal is not merely to replace one vocabulary with another. Instead, #dabe treats tokenization as a rate-distortion problem: given a finite bit budget, how should a learned tokenizer allocate capacity across a span so that reconstruction error falls where it matters most?

The central design choice in this work is to decouple coarse meaning from lexical repair. A #dabe chunk first receives a compact gist representation intended to capture broad local structure. A second, sparse repair pathway then allocates additional lexical lookup capacity to positions predicted to be difficult. This produces a sliding repair budget over a stable chunk, rather than a sliding token window. The distinction matters: changing the primary token-window geometry can reduce nominal bitrate, but it can also disrupt local code specificity and make reconstruction brittle. In contrast, keeping the chunk stable while adapting the repair budget lets the model preserve a consistent coarse representation and spend precision only where residual error suggests it is useful.

Our experiments on TinyStories establish this framing empirically. The strongest setting, a fixed 64-token chunk with gist encoding and residual-routed sparse lexical lookup, reaches 0.91547 token accuracy and 5.41007 mean chunk deviation at 20.37236 observed #bpt. A cost-aware operating point replicated across adjacent cost settings reaches 0.89274 token accuracy and 6.86454 mean chunk deviation at 20.06207 observed #bpt. A matched fixed-rate no-lookup baseline at 20.0 #bpt is stable and non-collapsed, reaching 0.72045 token accuracy and 17.89119 mean chunk deviation at the original horizon; a longer convergence-defense run improves to 0.77324 and 14.51295 but still remains far behind sparse repair. The result is a matched-rate separation: the gain is not explained by total bit budget alone, but by the model's ability to place lexical precision where reconstruction requires it.

This paper makes four contributions:

- We formulate #dabe tokenization as a learned rate-distortion problem over fixed token chunks, with binary codes and explicit reconstruction metrics.
- We introduce a gist-plus-residual-repair architecture that separates coarse span representation from sparse lexical precision.
- We show that adaptive sparse repair substantially outperforms a matched fixed-rate no-lookup learned tokenizer at approximately 20 #bpt.
- We provide controlled ablations showing that adaptive token-window geometry is not, by itself, the right mechanism for this setting.

The paper therefore makes a specific rate-distortion claim: within a controlled tokenizer-autoencoder setting, adaptive sparse lexical repair over stable chunks gives a stronger reconstruction frontier than either fixed-rate learned compression or adaptive token-window geometry. This is the mechanism result that future full language-model integration should build on.

= Related Work

#strong[Subword tokenization.] Most contemporary language models use fixed tokenization policies such as byte-pair encoding, WordPiece, unigram language-model tokenization, or byte-level variants @sennrich2016bpe @kudo2018sentencepiece @schuster2012japanese. These methods are robust and efficient, but they allocate sequence length through corpus-level frequency statistics rather than through the local information density of a span. #dabe keeps the deployment motivation of compact discrete representations, but studies whether a learned tokenizer can allocate reconstruction capacity adaptively inside a fixed chunk.

#strong[Learned discrete tokenizers.] Discrete autoencoders, vector-quantized models, residual quantizers, and semantic hashing methods learn compact latent codes that can reconstruct or represent text @vandenoord2017vqvae @kaiser2018discrete. These approaches motivate treating tokenization as a learned compression problem rather than a fixed segmentation problem. The fixed-rate no-lookup #dabe baselines in #exp[071], #exp[079], and #exp[093] play this role in our experiments: they ask how much reconstruction quality a learned chunk code obtains when it spends all capacity uniformly.

#strong[Adaptive computation and routing.] Adaptive computation mechanisms spend more computation on difficult examples, tokens, or positions @graves2016act @bengio2013estimating. #dabe applies the same broad principle to tokenizer reconstruction capacity: the model predicts where sparse lexical repair is worth spending after producing a coarse gist of the chunk. This differs from adaptive windowing because the primary chunk geometry remains stable while the repair budget changes.

#strong[Retrieval, copy, and lexical memory.] Copy and retrieval mechanisms are often used to recover rare or exact lexical details that parametric sequence models blur @gu2016copy @guu2020realm. #dabe's sparse lexical lookup stream plays a similar role inside the tokenizer itself: it gives the decoder a small budget for high-precision lexical holes while leaving easier positions to the gist representation.

= Method

== Problem Setup

Let a tokenized text chunk be

$ x = (x_1, ..., x_T), quad x_i in cal(V), quad T = 64. $

The tokenizer-autoencoder maps this chunk into a binary code and reconstructs a distribution over the original GPT-2 token IDs. We treat the system as a rate-distortion model. A method is useful only when it reduces reconstruction distortion at a comparable observed bitrate.

== Fixed-Rate Chunk Code

The learned chunk encoder produces a binary representation

$ b = q_phi(x), quad b in {0, 1}^B, $

where $B$ is the base code size. The fixed-rate no-lookup comparator reconstructs directly from this code:

$ h_i = F_theta(b, i), quad ell_i^"base" = W_o h_i, quad p_i^"base" = op("softmax")(ell_i^"base"). $

The fixed-rate bitrate is therefore

$ R_"fixed" = B / T. $

For #exp[093], $B = 1280$ and $T = 64$, giving exactly 20.0 #bpt. This is the matched learned-compression baseline for the cost-aware #dabe point.

== Gist Stream

The #dabe decoder separates coarse reconstruction from sparse repair. First, a gist decoder maps the binary code into position-wise hidden states and logits:

$ g_i &= G_theta(b, i) \
  ell_i^"gist" &= W_g g_i \
  p_i^"gist" &= op("softmax")(ell_i^"gist"). $

This stream is responsible for predictable local structure. It should recover easy tokens without spending lexical lookup slots everywhere.

== Residual Difficulty Signal

The residual router estimates where the gist stream is likely to be wrong. In supervised analysis this difficulty can be written as the per-position gist loss

$ d_i = op("CE")(x_i, p_i^"gist") = -log p_i^"gist"(x_i). $

During decoding, the router uses available hidden features rather than ground-truth labels:

$ r_i = f_psi(g_i), quad r_i in RR^(K_"max"), $

where $K_"max"$ is the maximum number of lexical repair slots that could be exposed for a position or local region. The experiments with residual-router supervision train these scores to correlate with reconstruction difficulty, so high-information holes receive more repair capacity.

== Sparse Lexical Reconstruction

Sparse lexical lookup augments the gist logits with a small set of candidate repair vectors. Let

$ m_(i,k) in RR^(|cal(V)|), quad k = 1, ..., K_"max" $

be the logit-space contribution of lookup slot $k$ at position $i$. The router converts scores into keep probabilities

$ a_(i,k) = sigma(r_(i,k)), quad a_(i,k) in [0, 1], $

and the discrete active-slot mask is

$ z_(i,k) = bb(1)[a_(i,k) > tau]. $

The repaired reconstruction logits are

$ ell_i^"repair" = ell_i^"gist" + lambda_"lookup" sum_(k=1)^(K_"max") z_(i,k) m_(i,k), $

and the final token distribution is

$ p_i^"repair" = op("softmax")(ell_i^"repair"). $

This equation is the core sparse-reconstruction claim: the binary chunk code supplies a smooth whole-span gist, while the lookup path fills a small number of lexical holes selected by the residual router.

== Cost-Aware Objective

The reconstruction objective is token cross-entropy over the repaired distribution:

$ cal(L)_"rec" = 1 / T sum_(i=1)^T op("CE")(x_i, p_i^"repair"). $

Cost-aware runs add a penalty for using lookup capacity:

$ cal(L) = cal(L)_"rec" + alpha_"slot" 1 / (T K_"max") sum_(i=1)^T sum_(k=1)^(K_"max") a_(i,k) + beta_"router" cal(L)_"router". $

Here $alpha_"slot"$ is the lookup-slot cost weight and $cal(L)_"router"$ denotes optional residual-router supervision. The observed effective bitrate charges the base code plus active lexical slots:

$ R_"obs" = (B + c_"slot" sum_(i=1)^T sum_(k=1)^(K_"max") z_(i,k)) / T. $ <eq-observed-rate>

where $c_"slot"$ is the accounting cost per active lookup slot. In the experiments, reported #bpt values use this observed active-slot budget rather than the maximum possible lookup budget. This is why cost-aware points can be compared fairly against fixed-rate no-lookup baselines.

== Distortion Metrics

Token accuracy reports the fraction of positions reconstructed exactly:

$ A_"tok" = 1 / T sum_(i=1)^T bb(1)[hat(x)_i = x_i], quad hat(x)_i = arg max_(v in cal(V)) p_i(v). $

Chunk deviation is the Hamming distortion over a 64-token slab:

$ D(x, hat(x)) = sum_(i=1)^T bb(1)[hat(x)_i != x_i]. $ <eq-chunk-deviation>

We report the mean and p90 of $D$ across validation chunks. Chunk deviation is more informative than exact full-chunk reconstruction in the current regime because even strong models rarely reconstruct all 64 tokens exactly.

== Why Fixed Chunks

The method keeps $T = 64$ fixed and adapts repair allocation through $z_(i,k)$. This is the architectural distinction tested by the paper. A sliding-window tokenizer would make chunk geometry itself adaptive: easy regions could be represented with wider windows, while hard regions could be represented with finer windows. That design is attractive because it appears to align token length with information density. In this setting, however, it also changes the carrier on which the binary code is trained. The decoder must reconstruct not only lexical content, but also a moving segmentation geometry; errors in routing can therefore alter both what is represented and where reconstruction evidence is available.

#dabe separates these two problems. The 64-token slab supplies a stable reconstruction coordinate system, while sparse repair changes only the local precision budget. The gist stream can learn a consistent whole-chunk prior, and the residual path can then allocate lexical detail to the holes left by that prior. This is why the model adapts $z_(i,k)$ rather than $T$: information density changes the repair mask, not the coordinate frame.

We evaluate the sliding-window alternative directly with quantile-supervised variable windows, action-value-supervised window routing, and hard straight-through routing. These variants are reported as controlled ablations: they distinguish whether adaptive granularity should live in token geometry or in repair allocation. The results in Section 5 show that variable windows either reduce bitrate at substantial reconstruction cost or collapse toward all-fine routing, while fixed chunks plus sparse repair maintain the stronger rate-distortion frontier.

= Experimental Setup

== Dataset and Token Basis

All main tokenizer-autoencoder experiments use TinyStories with 4096 training samples and 512 validation samples in streaming mode. Text is converted to GPT-2 token IDs, and each training example is chunked into fixed 64-token slabs. This keeps the evaluation focused on learned chunk reconstruction rather than on changing the underlying text preprocessing pipeline.

== Training Protocol

The main runs use a Modal T4 GPU, bf16 mixed precision, batch size 32, 12000 maximum training steps, validation every 300 steps, and checkpointing every 2000 train steps. #exp[095] extends the fixed-rate comparator toward 24000 steps as a convergence-defense run under the same architecture and data basis. Each experiment logs its configuration, run ID, launch contract, and final summary into `research/experiment_log.md` and `experiments/modal_downloads/`.

== Metrics

We report token accuracy, top-5 and top-10 token accuracy, exact 16-token block accuracy, exact 64-token chunk accuracy, mean chunk deviation, p90 chunk deviation, observed effective #bpt, active lookup $K$, and bit density. Chunk deviation is the primary distortion metric for comparing reconstructions that are close but not exactly identical.

== Runtime and Compute Accounting

The headline runs are small enough for single-T4 execution but not free: sparse repair adds selector, halting, and lookup-attention work on top of the fixed chunk decoder. Each optimizer step processes $32 times 64 = 2048$ GPT-2 tokens. The fixed-rate #exp[093] run logged an ETA sample of 14.34 steps/s, or approximately 29368 tokens/s. The cost-aware #exp[092] sweep logged 7.80 steps/s on its first child, or approximately 15974 tokens/s. This indicates roughly a 1.8x training-throughput cost for the sparse repair path relative to the no-lookup matched baseline under the logged T4 runs.

#figure(
  table(
    columns: (1.9fr, 0.75fr, 0.85fr, 0.8fr, 0.8fr, 0.85fr),
    inset: 5pt,
    table.header([Setting], [Base bits], [Slot bits], [Budget $K$], [Active $K$], [#bpt]),
    [#exp[093] fixed-rate no lookup], [1280], [0], [0.00], [0.00], [20.000],
    [#exp[092] cost-aware sparse repair], [1024], [22], [11.82], [6.64], [20.062],
    [#exp[087] quality anchor], [1024], [22], [12.72], [9.03], [20.372],
  ),
  caption: [Bitrate and repair-budget accounting for the main settings. Budget $K$ determines the charged side budget; active $K$ is the threshold-active lookup usage.],
) <tab-compute>

== Bitrate Calibration

#figure(
  table(
    columns: (2.4fr, 1.2fr, 3.1fr),
    inset: 5pt,
    table.header([Quantity], [Value], [Interpretation]),
    [GPT-2 vocabulary index information cost], [$log_2(50257) approx 15.62$ #bpt], [raw token-ID index reference],
    [#dabe fixed-rate baseline], [20.00 #bpt], [learned reconstruction from a fixed chunk code],
    [#dabe sparse repair range], [20.06--20.37 observed #bpt], [base code plus active lookup-slot accounting],
  ),
  caption: [Bitrate calibration. These are learned GPT-2-token-ID reconstruction rates, not claims of raw-text compression superiority over BPE.],
) <tab-bitrate-calibration>

EXP-094 adds a measured decode-profile pass on the EXP-087 checkpoint: 1351 validation chunks were processed in 6.42 seconds, or 210.33 chunks/s and 13461 tokens/s, with CUDA peak allocated/reserved memory of 2031.74/2152.00 MiB. EXP-096 runs the same diagnostic path for the fixed-rate EXP-093 checkpoint: 1351 validation chunks were processed in 2.63 seconds, or 514.54 chunks/s and 32930 tokens/s, with CUDA peak allocated/reserved memory of 852.80/974.00 MiB. The measured inference headline is therefore about 2.45x slower throughput and 2.38x higher peak allocated memory for sparse repair, in exchange for much lower reconstruction distortion. Computationally, the lookup path adds memory and attention activations proportional to $O(B_"batch" K_"max" d + B_"batch" T K_"max")$ for hidden width $d$, chunk length $T = 64$, and maximum lookup list $K_"max" = 32$.

== Baselines and Ablations

The main fixed-rate comparator is #exp[093], a no-lookup hierarchical-local decoder at 20.0 #bpt. The main #dabe cost-aware point is #exp[092] at 20.06207 observed #bpt. #exp[087] is the quality anchor. #exp[088], #exp[089], and #exp[090] test variable token-window alternatives. #exp[077] through #exp[087] provide the adaptive-budget development path.

= Results

== Matched 20 #bpt Comparator

@fig-matched20 gives the cleanest test of the mechanism claim. At the same bitrate scale, adaptive sparse repair is far more accurate than a fixed-rate no-lookup learned chunk tokenizer.

#figure(
  image("../paper_drafts/figures/fig01_matched_20bpt_comparator.svg", width: 95%),
  caption: [Matched 20 #bpt comparator. Cost-aware sparse repair improves token accuracy and chunk deviation relative to a fixed-rate no-lookup learned compressor at nearly the same bitrate.],
) <fig-matched20>

#figure(
  table(
    columns: (2.5fr, 0.8fr, 0.9fr, 0.9fr, 0.8fr),
    inset: 5pt,
    table.header([Setting], [#bpt], [Token acc.], [Mean dev.], [p90 dev.]),
    [#exp[093] fixed-rate no lookup], [20.00000], [0.72045], [17.89119], [24.14597],
    [#exp[095] longer fixed-rate], [20.00000], [0.77324], [14.51295], [20.75514],
    [#exp[092] cost-aware sparse repair], [20.06207], [0.89274], [6.86454], [10.96876],
  ),
  caption: [Matched 20 #bpt result.],
) <tab-matched20>

The original matched-horizon token-accuracy gain is 0.17229, and mean chunk deviation falls by 11.02665 tokens per 64-token chunk. #exp[095] shows that longer fixed-rate training improves the comparator to 0.77324 token accuracy and 14.51295 mean deviation, but the sparse-repair point remains 0.11950 absolute token accuracy higher and 7.64841 tokens/chunk lower in mean deviation. The fixed-rate baseline is healthy rather than collapsed: its bit density remains near 0.49. The gap therefore supports the claim that the gain comes from where lexical precision is spent, not only from how many bits are spent or whether the fixed-rate baseline had converged.

== Rate-Distortion Frontier

@fig-frontier places the main experiments on a shared rate-distortion plot. Lower chunk deviation at comparable #bpt is better.

#figure(
  image("../paper_drafts/figures/fig02_rate_distortion_frontier.svg", width: 95%),
  caption: [Rate-distortion frontier across fixed-rate baselines, sparse lookup models, cost-aware points, and variable-window ablations. Fixed chunks plus sparse repair form the strongest current frontier.],
) <fig-frontier>

The best quality anchor is #exp[087], with 0.91547 token accuracy and 5.41007 mean chunk deviation at 20.37236 observed #bpt. The best lower-cost knee is #exp[092] with slot cost 0.025, which reaches 0.89274 token accuracy and 6.86454 mean chunk deviation at 20.06207 observed #bpt.

== Fixed-Rate Convergence Check

#exp[095] directly tests whether #exp[093] underperformed because the fixed-rate no-lookup baseline was undertrained. The longer run timed out operationally near 23500/24000 steps, but it passed the predeclared 12000-step minimum, wrote metrics through step 23244, and saved a best checkpoint at step 22000. Its best validation point is 0.77324 token accuracy and 14.51295 mean chunk deviation. This improves the fixed-rate baseline materially, but does not approach #exp[092]'s 0.89274 token accuracy and 6.86454 mean deviation.

== Cost-Knee Replication

#figure(
  image("../paper_drafts/figures/fig03_cost_knee_curve.svg", width: 95%),
  caption: [Cost-knee curve for lookup-slot cost. The 0.025 setting is the replicated local knee: lower cost spends more without improving quality, while higher cost over-prunes repair.],
) <fig-costknee>

#figure(
  table(
    columns: (0.9fr, 0.9fr, 0.9fr, 0.9fr, 0.9fr),
    inset: 5pt,
    table.header([Slot cost], [#bpt], [Active $K$], [Token acc.], [Mean dev.]),
    [0.0225], [20.26545], [8.10659], [0.89255], [6.87713],
    [0.0250], [20.06207], [6.63953], [0.89274], [6.86454],
    [0.0275], [19.84857], [4.54922], [0.89002], [7.03849],
  ),
  caption: [Lookup-slot cost sweep around the local knee.],
) <tab-costknee>

The cost term is useful but not sufficient by itself. It controls active slot usage, but the quality anchor remains #exp[087]. This suggests that future gains should improve the router and lookup content at fixed active $K$, rather than simply increasing cost pressure.

== Adaptive-Budget Ablation

#figure(
  image("../paper_drafts/figures/fig04_adaptive_budget_ablation.svg", width: 95%),
  caption: [Adaptive-budget ablation. Quality improves as the architecture learns to allocate repair slots more selectively, not merely as raw maximum $K$ increases.],
) <fig-budgetablation>

The ablation path starts with fixed sparse lookup, moves to ranked halting, then adds gist and residual-router supervision. Fixed $K = 32$ obtains strong quality at high cost. Ranked halting and gist-residual routing recover much of that quality near the 20--22 #bpt range, motivating the final fixed-chunk plus adaptive sparse repair architecture.

== Variable-Window Ablation

#figure(
  image("../paper_drafts/figures/fig05_variable_window_negative_result.svg", width: 95%),
  caption: [Variable-window ablation. Sliding token-window geometry underperforms the fixed-chunk sparse-repair anchor.],
) <fig-variablewindow>

#figure(
  table(
    columns: (2.4fr, 0.9fr, 0.9fr, 0.9fr),
    inset: 5pt,
    table.header([Setting], [#bpt], [Token acc.], [Mean dev.]),
    [#exp[088] quantile variable windows], [19.46092], [0.85820], [9.07550],
    [#exp[089] action-value variable windows], [18.29175], [0.87508], [7.99482],
    [#exp[090] hard straight-through windows], [21.70567], [0.87209], [8.18653],
    [#exp[087] fixed chunk + residual lookup], [20.37236], [0.91547], [5.41007],
  ),
  caption: [Variable-window ablations compared with the fixed-chunk quality anchor.],
) <tab-variablewindow>

This ablation narrows the paper's mechanism claim. Adaptive granularity is useful, but in our experiments it belongs in the repair budget rather than in the primary token-window geometry.

== Active $K$ and Quality

#figure(
  image("../paper_drafts/figures/fig06_active_k_vs_quality.svg", width: 95%),
  caption: [Active repair budget versus distortion. The best points are not simply the largest active-$K$ points; selective allocation matters.],
) <fig-activek>

@fig-activek supports the mechanism interpretation. More active slots often help, but the strongest points are produced by architectures that spend slots selectively. This is the empirical counterpart of @eq-observed-rate: the side budget must buy useful lexical repairs rather than simply activate more capacity.

== Sparse Repair Trace

#figure(
  image("../paper_drafts/figures/fig07_sparse_repair_trace.svg", width: 95%),
  caption: [Sparse repair trace from EXP-094. The gist stream misses scene and dialogue details; the sparse repair path restores selected tokens in the final decode.],
) <fig-repairtrace>

#block(inset: 7pt, fill: luma(97%), stroke: luma(84%), radius: 3pt)[
  #text(size: 8.6pt)[#strong[Uncorrected gist decode.] really toys. It was never a big, Sue,'s one. It are the hole and saw the new up. "Thank is little one, pictures?" Ben asked, feeling scared. "I are't know, , it is a new car. They it like a bad dragon!" And]

  #v(0.35em)
  #text(size: 8.6pt)[#strong[Corrected sparse-repair decode.] like nothing. It was just a big, white, cloud cloud. It covered the sun and made the sky darker. "What is that cloud, Anna?" Ben asked, feeling scared. "I don't know, Ben, it is a strange cloud. Maybe it is a monster cloud!" Anna]

  #v(0.35em)
  #text(size: 8.6pt)[#strong[Target reference.] like nothing. It was just a big, white, fluffy cloud. It covered the sun and made the sky darker. "What is that cloud, Anna?" Ben asked, feeling scared. "I don't know, Ben, it is a strange cloud. Maybe it is a monster cloud!" Anna]
]

@fig-repairtrace makes the repair mechanism observable. In this diagnostic chunk, the gist stream reconstructs only 62.5% of tokens, confusing scene words such as "sun", "sky", and "cloud" with plausible but wrong alternatives. The repaired decode reaches 98.4% token accuracy on the same chunk, with 23 positions corrected. The trace is the intended interpretation of sparse repair as an inspectable budget, not an opaque black box.

== Code-Domain Completion Control

#exp[099] and #exp[100] test whether the code failure in #exp[097] is a domain-exposure issue rather than a hard representational limit. These runs use the deterministic Python-code generator as a controlled probe, not as a broad Python benchmark. On next-token completion, the #dabe LM reaches 0.98964 accuracy versus 0.98012 for the GPT-2 BPE baseline. On reconstruction, the code-trained sparse-repair tokenizer reaches 1.0 token accuracy, 1.0 exact chunk accuracy, and 0.0 mean chunk deviation at 16.20189 observed #bpt.

#figure(
  table(
    columns: (1.4fr, 2.1fr, 2.1fr),
    inset: 5pt,
    table.header([Context], [#dabe completion], [GPT-2 BPE completion]),
    [#raw("def")],
    [#raw("add_user(users,") => #raw("name):")],
    [#raw(" add") => #raw("_") => #raw("user")],
    [#raw("def add_user(users,")],
    [#raw("name):")],
    [#raw("_")],
    [#raw("... name):")],
    [#raw("users.append({'name':")],
    [#raw("user")],
    [#raw("... 'active':")],
    [#raw("true})")],
    [#raw("(")],
  ),
  caption: [Completion examples from #exp[099]. #dabe predicts learned code-span tokens, while the BPE baseline predicts subword pieces.],
) <tab-code-completions>

@tab-code-completions makes the tokenization difference visible. #dabe's successful predictions are larger code-local units such as `add_user(users,`, `name):`, and `users.append({'name':`. Its observed mistakes are also chunkier: repeated loop contexts can confuse `value` with `skipped.append(idx)`, and repeated `result =` contexts can prefer `{'error':` over `client.fetch(user_id=user_id,`. By contrast, BPE's observed mistakes are often whitespace or subword-local, such as predicting a space where the target is ` self`. This supports the mechanism interpretation without overstating the dataset: code-domain training can make the learned span tokenizer behave sensibly on code-like text, but the current corpus is still small and repeated.

= Analysis

#strong[Why sparse repair works.] The gist path turns the chunk code into a smooth whole-span prior. This is useful for predictable text, but exact token reconstruction is brittle around names, punctuation, dialogue markers, and other locally high-information regions. Sparse repair gives the model an explicit way to allocate lexical precision to these holes. @eq-chunk-deviation makes the consequence measurable: the target is not vague semantic similarity, but fewer wrong tokens per chunk.

#strong[Why the fixed-rate comparator matters.] #exp[093] is the rate-matched comparator that isolates the mechanism. It spends the same budget scale inside a learned chunk autoencoder, but without the sparse lexical repair path. Its much higher chunk deviation shows that uniform capacity is a weak substitute for adaptive repair. #exp[095] addresses convergence directly: longer training improves fixed-rate reconstruction, but the remaining gap to #exp[092] is still large.

#strong[Inference overhead.] #exp[096] provides the matched no-lookup decode profile for #exp[094]. Fixed-rate decoding reaches 32930 tokens/s and 852.80 MiB peak allocated memory, while sparse repair reaches 13461 tokens/s and 2031.74 MiB. Sparse repair is therefore about 2.45x slower and 2.38x larger in allocated memory on this T4 diagnostic path, traded for a token-accuracy increase from 0.72025 to 0.91549.

#strong[Code-like text.] #exp[097] evaluates the #exp[087] quality-anchor checkpoint zero-shot on a deterministic Python-code corpus. Code stresses different reconstruction behavior than TinyStories: identifiers, indentation, brackets, operators, string delimiters, and numeric literals carry exact meaning. The result is a deliberate scope check rather than a success claim: token accuracy drops to 0.41761 with 37.27273 mean chunk deviation, while active $K$ rises to 16.0. The model recognizes code as difficult and spends repair budget, but the TinyStories-trained lexical repair content is not code-calibrated. #exp[099] and #exp[100] provide the complementary in-domain control: after code-domain training on the deterministic generator, the LM completion metric and tokenizer reconstruction metric both improve sharply.

#strong[What remains open.] The cost-aware knee shows a path toward better compression, but it also shows that cost pressure alone diminishes on either side of 0.025. The next architectural question is how to improve lookup correctness and router calibration at a fixed active $K$.

= Limitations

This study is intentionally framed as tokenizer-autoencoder rate-distortion rather than full downstream language-model pretraining. That choice isolates the tokenization mechanism: the experiments measure how chunk codes, sparse repair, and routing policies affect reconstruction before adding the confounds of language-model scale, optimizer schedules, and downstream task selection. The matched 20 #bpt comparison is therefore the central evidence unit for the paper.

The main experiments use TinyStories as a controlled text domain with exact GPT-2-token reconstruction metrics. This setting makes it possible to compare architectural variants under the same token basis, chunk length, training budget, and validation protocol. Domain generalization remains a separate evaluation target; #exp[097] makes that boundary explicit by reaching only 0.41761 token accuracy on deterministic Python code. #exp[099] and #exp[100] show that code-domain training can produce favorable completion and reconstruction results on the same deterministic generator, but that generator is intentionally small and repetitive. The current paper establishes a clean mechanism result that should next be tested on broader corpora, real code repositories, multilingual data, and full LM training.

The run plan prioritizes focused, objective comparisons over broad exploratory sweeps. The experiment set includes a matched fixed-rate comparator, a replicated cost-knee neighborhood around #exp[092], a quality anchor in #exp[087], and direct variable-window ablations. Additional seeds and larger-scale runs would support confidence intervals and scaling laws, but they are not required to interpret the main matched-rate separation.

The current models are rate-distortion tokenizers, not lossless compressors. Exact full-chunk reconstruction remains low, while token accuracy, exact local blocks, and mean chunk deviation improve substantially with sparse repair. This is the correct operating point for the paper's claim: #dabe demonstrates that adaptive repair allocation can sharply reduce reconstruction distortion at a matched observed bitrate.

= Conclusion

#dabe studies tokenization as a learned rate-distortion problem over fixed text chunks. The strongest result is not obtained by changing token-window geometry, but by keeping a stable 64-token chunk and adapting a sparse lexical repair budget over it.

The matched comparison between #exp[093]/#exp[095] and #exp[092] is the central result: a healthy fixed-rate no-lookup learned tokenizer at 20.0 #bpt reaches 0.72045 token accuracy and 17.89119 mean chunk deviation at the original horizon, and improves to 0.77324 and 14.51295 under longer training. Cost-aware #dabe still reaches 0.89274 token accuracy and 6.86454 mean chunk deviation at 20.06207 observed #bpt. This shows that total bit budget and fixed-baseline convergence alone do not explain the gain.

The broader ablation story is consistent. Fixed sparse lookup improves over no-lookup codes; ranked halting recovers much of fixed $K = 32$ quality at lower observed bitrate; the gist-residual architecture improves repair allocation further; and variable token-window geometry fails to beat fixed chunks plus repair. The architectural implication is clear: for learned tokenizers, adapt the repair budget before adapting the token window.

#bibliography("references.bib")
