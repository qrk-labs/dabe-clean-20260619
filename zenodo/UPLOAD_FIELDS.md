> **Development freeze:** DABE is frozen as the publication artifact for the preprint at https://doi.org/10.5281/zenodo.20797432. Do not run new experiments or alter scientific claims unless the user explicitly unfreezes development. See [`FREEZE_NOTICE.md`](../FREEZE_NOTICE.md).

# Zenodo Upload Fields

Use this as the copy-paste version of the metadata JSON files.

## Paper / Preprint Record

Resource type:

```text
Publication / Preprint
```

Title:

```text
Density-Adaptive Bitmask Encoding: Fixed Chunks with Sparse Lexical Repair
```

Creators:

```text
Tsowa, Mainasara Al-amin <mainasara@qrk.ng>
Tsowa, Babangida Usman <babangida@qrk.ng>
Mustapha, Abdul-malik Abdullahi <maleek@qrk.ng>
```

Description / abstract:

```text
Modern subword tokenizers allocate lexical capacity through a fixed segmentation policy rather than through local span difficulty. This paper studies Density-Adaptive Bitmask Encoding (DABE), a learned tokenizer-autoencoder that compresses fixed 64-token GPT-2-token chunks into binary codes and spends sparse lexical repair capacity only where reconstruction is difficult. Across controlled TinyStories experiments, fixed chunks plus adaptive sparse lexical repair outperform matched fixed-rate learned compression and variable-window alternatives at approximately 20 bits/token.
```

Keywords:

```text
learned tokenization; rate-distortion; sparse lexical repair; tokenizer autoencoder; DABE; BPE; TinyStories; adaptive computation
```

License:

```text
Creative Commons Attribution 4.0 International
```

Version:

```text
paper-v1
```

Language:

```text
English
```

Related identifiers:

```text
https://github.com/qrk-labs/dabe-clean-20260619
Relation: is supplemented by
Resource type: software
```

Files to upload:

```text
zenodo/deposit/paper/research/paper_typst/main.pdf
zenodo/deposit/paper/research/paper_typst/main.typ
zenodo/deposit/paper/research/paper_typst/references.bib
zenodo/deposit/paper/research/paper_drafts/
zenodo/deposit/paper/research/paper_drafts/figures/
```

## Software / Artifact Record

Resource type:

```text
Software
```

Title:

```text
DABE Paper Artifact: Fixed Chunks with Sparse Lexical Repair
```

Description:

```text
Software and lightweight reproducibility artifacts for the DABE paper, including source code, selected configs, figure generation data, experiment logs, launch contracts, and Modal JSON/CSV summaries. The artifact supports the paper's central claim that fixed 64-token chunks plus adaptive sparse lexical repair outperform fixed-rate learned chunk compression and adaptive token-window geometry in the tested tokenizer-autoencoder setting.
```

License:

```text
MIT License
```

Version:

```text
paper-v1
```

Files to upload:

```text
zenodo/deposit/artifact/
```

or the convenience archive:

```text
zenodo/dabe_zenodo_deposit_2026-06-22.zip
```

## Reserved DOI

Current reserved DOI:

```text
10.5281/zenodo.20797432
```

After reserving a DOI, run:

```bash
python3 zenodo/scripts/replace_doi.py 10.5281/zenodo.YOUR_RESERVED_ID
python3 zenodo/scripts/build_deposit_bundle.py
```
