> **Development freeze:** DABE is frozen as the publication artifact for the preprint at https://doi.org/10.5281/zenodo.20797432. Do not run new experiments or alter scientific claims unless the user explicitly unfreezes development. See [`FREEZE_NOTICE.md`](../../FREEZE_NOTICE.md).

# DABE Paper Typst Draft

This directory contains a native Typst manuscript for the current DABE paper draft. It mirrors the TeX draft while using Typst syntax for headings, figures, tables, equations, citations, and bibliography.

## Workspace-Local Typst Install

Install Typst into the repository-local tool directory:

```sh
make install-typst
```

This installs to:

```text
../.tools/bin/typst
```

The current Codex environment could not complete network installs because outbound DNS/network access is blocked (`Could not resolve host: index.crates.io`). Once network access is available, rerun `make install-typst` from this directory.

## Build

After installation, build the paper PDF from this directory with:

```sh
make pdf
```

The Makefile passes `--root ..` so Typst can read both `paper_typst/` and the sibling generated figures in `paper_drafts/figures/`.

The Typst source uses the generated SVG figures from `../paper_drafts/figures/`. Regenerate those figures from the repository root with:

```sh
python3 scripts/build_paper_figures.py
```
