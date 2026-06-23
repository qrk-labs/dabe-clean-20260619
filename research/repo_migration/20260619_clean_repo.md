> **Development freeze:** DABE is frozen as the publication artifact for the preprint at https://doi.org/10.5281/zenodo.20797432. Do not run new experiments or alter scientific claims unless the user explicitly unfreezes development. See [`FREEZE_NOTICE.md`](../../FREEZE_NOTICE.md).

# Clean Repository Migration - 2026-06-19

This workspace was converted from the historical DABE Git repository into a fresh Git repository to remove large accumulated Git/LFS object storage.

## Previous Git State

- Previous branch: `exp/dabe-tokenizer-autoencoder`
- Previous HEAD: `bee2a0f0fdcaf114919c5b3ddb68c36b5e2528d4`
- Previous remotes:
  - `origin`: `git@github.com:qrk-labs/dabe.git`
  - `export-dabe-export-20260615`: `git@github.com:qrk-labs/dabe-export-20260615.git`

## Reason

The old `.git` directory had grown to roughly `45G` after initial garbage cleanup, including large Git object and LFS storage. Earlier in the cleanup we removed about `20G` of orphaned temporary Git object garbage without changing history or the working tree.

## New Repository Policy

The fresh repo is intended to track source, configs, tests, research notes, and lightweight experiment summaries. Heavy artifacts such as checkpoints, tensors, arrays, local logs, WandB runs, and virtual environments are ignored and should live in Modal volumes or artifact storage.
