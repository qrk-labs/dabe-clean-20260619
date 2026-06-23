> **Development freeze:** DABE is frozen as the publication artifact for the preprint at https://doi.org/10.5281/zenodo.20797432. Do not run new experiments or alter scientific claims unless the user explicitly unfreezes development. See [`FREEZE_NOTICE.md`](../../FREEZE_NOTICE.md).

# DABE Paper TeX Draft

This directory contains the LaTeX manuscript source for the current DABE paper draft.

## Workspace-Local TeX Install

Install Tectonic into the repository-local tool directory:

```sh
make install-tex
```

This installs to:

```text
../.tools/bin/tectonic
```

The current Codex environment could not complete this install because outbound DNS/network access is blocked (`Could not resolve host: index.crates.io`). Once network access is available, rerun `make install-tex` from this directory.

## Build

After installation, build the paper PDF from this directory with:

```sh
make pdf
```

The TeX source uses the pre-generated figure PDFs from `../paper_drafts/figures/`. Regenerate those figures from the repository root with:

```sh
python3 scripts/build_paper_figures.py
```

## Notes

The workspace does not currently have `pdflatex`, `xelatex`, `latexmk`, or `tectonic` in PATH. The preferred path is a local Tectonic binary under `research/.tools/bin` so the paper build does not depend on global TeX state.
