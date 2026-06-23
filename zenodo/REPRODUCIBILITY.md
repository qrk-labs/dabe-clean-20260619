> **Development freeze:** DABE is frozen as the publication artifact for the preprint at https://doi.org/10.5281/zenodo.20797432. Do not run new experiments or alter scientific claims unless the user explicitly unfreezes development. See [`FREEZE_NOTICE.md`](../FREEZE_NOTICE.md).

# Reproducibility Notes

## Core Claim

DABE works best in the tested regime as fixed 64-token chunks plus adaptive
sparse lexical repair, not as adaptive token-window geometry.

The paper's primary evidence is:

- EXP-092 cost-aware sparse repair at approximately 20 observed bits/token.
- EXP-093 fixed-rate 20 bits/token no-lookup comparator.
- EXP-095 longer fixed-rate convergence-defense run.
- EXP-087 quality anchor.
- EXP-088, EXP-089, and EXP-090 variable-window negative ablations.

## Figure Pipeline

The paper figure pipeline is:

```bash
python3 scripts/build_paper_figures.py
```

Expected generated outputs live in:

```text
research/paper_drafts/figures/
```

The Zenodo bundle copies the generated SVG, PDF, PNG, and JSON figure sources.

## Paper Build

The Typst paper source lives in:

```text
research/paper_typst/main.typ
```

The current build output is:

```text
research/paper_typst/main.pdf
```

The repository-local Typst binary is expected at:

```text
research/.tools/bin/typst
```

Build command:

```bash
PATH="research/.tools/bin:$PATH" typst compile --root research research/paper_typst/main.typ research/paper_typst/main.pdf
```

## Included Experiment Artifacts

The bundle includes lightweight JSON/CSV artifacts from EXP-071, EXP-077,
EXP-079, EXP-080, EXP-083 through EXP-100, plus launch contracts where present.
It excludes checkpoints and cache files.

## Key Metrics To Verify

- EXP-092 cost-aware knee:
  - token accuracy: `0.89274`
  - mean chunk deviation: `6.86454`
  - observed bits/token: `20.06207`
- EXP-093 fixed-rate baseline:
  - token accuracy: `0.72045`
  - mean chunk deviation: `17.89119`
  - bits/token: `20.0`
- EXP-095 longer fixed-rate baseline:
  - token accuracy: `0.77324`
  - mean chunk deviation: `14.51295`
  - bits/token: `20.0`
- EXP-087 quality anchor:
  - token accuracy: `0.91547`
  - mean chunk deviation: `5.41007`
  - observed bits/token: `20.37236`
