# Methodology

## Problem Setup

DABE compresses a fixed 64-token input chunk into a binary representation and reconstructs the original GPT-2 token sequence. We evaluate the tokenizer as a rate-distortion system: lower bits/token is better only when reconstruction quality is preserved. The main metrics are token accuracy, top-k token accuracy, exact chunk or block reconstruction, observed effective bits/token, active lookup budget, and chunk deviation statistics.

## Fixed-Rate Chunk Code

The baseline learned tokenizer encodes each 64-token chunk into a fixed number of code bits and decodes with a hierarchical local decoder. This no-lookup path is the standard learned-compression comparator. EXP-093 uses `1280` code bits, or exactly `20.0` bits/token, making it directly comparable to the cost-aware DABE operating point at `20.06207` observed bits/token.

## Gist Stream

The gist stream produces a coarse reconstruction from the chunk-level binary code. Its role is to capture predictable local structure and broad semantic shape without spending lexical precision everywhere. This stream gives the model a stable base representation over the full 64-token slab.

## Residual Router

After the gist decode, the residual router predicts high- and medium-information regions from reconstruction difficulty. In the lead architecture, router supervision improves the model's ability to identify where lexical detail is likely to matter. EXP-087 shows that increasing residual-router supervision improves token accuracy and chunk deviation at modest bitrate cost.

## Sparse Lexical Lookup

The sparse lookup stream provides a candidate list of lexical repair slots. A halting or cost-aware policy controls how many slots remain active. The observed effective bitrate includes the base code plus the active lookup side budget, so models are compared by actual repair usage rather than by maximum possible K.

The decode diagnostic path records token-level traces of the sparse repair process: target token, gist prediction, final repaired prediction, lookup slot index, keep probability, and repair outcome. This makes the mechanism inspectable rather than treating the lookup path as a black box.

## Why Fixed Chunks

DABE keeps the 64-token chunk fixed and adapts repair allocation over that stable coordinate system. A sliding-window tokenizer would make token geometry adaptive, using wider windows for easy regions and finer windows for hard regions. That design is attractive because it appears to align token length with information density, but it also changes the carrier on which the binary code is trained. Routing errors can alter both what is represented and where reconstruction evidence is available.

DABE separates those problems: the slab remains fixed, while information density changes the sparse repair mask. We evaluate quantile-supervised variable windows, action-value-supervised routing, and hard straight-through routing as direct ablations. These controlled ablations distinguish whether adaptive granularity should live in token geometry or repair allocation; the results support fixed chunks plus adaptive repair.
