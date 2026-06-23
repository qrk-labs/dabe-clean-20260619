> **Development freeze:** DABE is frozen as the publication artifact for the preprint at https://doi.org/10.5281/zenodo.20797432. Do not run new experiments or alter scientific claims unless the user explicitly unfreezes development. See [`FREEZE_NOTICE.md`](../FREEZE_NOTICE.md).

# Zenodo Prepublish Checklist

## Paper Record

- [ ] Create a new Zenodo upload.
- [ ] Set resource type to `Publication / Preprint`.
- [ ] Set title to `Density-Adaptive Bitmask Encoding: Fixed Chunks with Sparse Lexical Repair`.
- [ ] Add all creators exactly:
  - Mainasara Al-amin Tsowa `<mainasara@qrk.ng>`
  - Babangida Usman Tsowa `<babangida@qrk.ng>`
  - Abdul-malik Abdullahi Mustapha `<maleek@qrk.ng>`
- [ ] Add ORCID IDs if available.
- [ ] Paste the abstract from the paper.
- [ ] Set access to open.
- [ ] Set license to `Creative Commons Attribution 4.0 International`.
- [ ] Reserve DOI before final publication if the DOI should appear in the PDF.
- [ ] Verify the reserved DOI `10.5281/zenodo.20797432` appears consistently in the metadata.
- [ ] Upload paper PDF, Typst source, bibliography, figures, and markdown drafts.
- [ ] Add related identifiers:
  - GitHub repository URL.
  - Software/artifact Zenodo DOI, if using a separate artifact record.
  - arXiv DOI/URL later, if applicable.
- [ ] Preview the Zenodo record and verify the rendered metadata.

## Software / Artifact Record

- [ ] Choose either GitHub release integration or manual software upload.
- [ ] If using GitHub release integration, copy `zenodo/metadata/.zenodo.json`
      and `zenodo/metadata/CITATION.cff` to the repository root before tagging.
- [ ] Set resource type to `Software`.
- [ ] Use MIT for repository code.
- [ ] Include `zenodo/LICENSES.md` so paper text/figures remain CC BY 4.0.
- [ ] Upload or archive only lightweight artifacts:
  - `research/experiment_log.md`
  - `research/paper_drafts/figures/figure_data.json`
  - selected Modal JSON/CSV summaries
  - launch contracts
  - figure builder script
  - relevant configs, source, and tests
- [ ] Do not upload checkpoints unless a future reviewer explicitly needs them.
- [ ] Verify `zenodo/deposit/file_manifest.json` checksums after upload.

## Final Sanity Checks

- [ ] Title matches across PDF, Zenodo metadata, `CITATION.cff`, and `.zenodo.json`.
- [ ] Authors match across PDF, Zenodo metadata, `CITATION.cff`, and `.zenodo.json`.
- [ ] DOI placeholder has been replaced or intentionally left out of the PDF.
- [ ] Main claim is backed by EXP-092 vs EXP-093/095 and figures/tables.
- [ ] Limitations stay objective: controlled tokenizer-autoencoder study,
      focused runs, single main text domain, deterministic code probe.
- [ ] No cache folders, pyc files, checkpoints, or local-only temp paths are in
      the upload bundle.
