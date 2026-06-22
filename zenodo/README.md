# DABE Zenodo Deposit Kit

This folder contains the upload-ready metadata and packaging plan for the
DABE paper and reproducibility artifact.

The recommended release has two Zenodo records:

1. Paper/preprint record
   - Resource type: Publication / Preprint
   - License: CC BY 4.0
   - Files: paper PDF, Typst source, bibliography, figures, and paper draft
     markdown.
2. Software/artifact record
   - Resource type: Software
   - License: MIT for code, with paper/text files separately marked CC BY 4.0.
   - Files: repository snapshot or GitHub release archive, selected configs,
     scripts, tests, experiment logs, figure pipeline, and lightweight Modal
     artifacts.

## Fast Path

From the repository root:

```bash
python3 zenodo/scripts/build_deposit_bundle.py
```

This creates:

- `zenodo/deposit/` with paper and artifact subfolders.
- `zenodo/deposit/file_manifest.json` with SHA-256 checksums.
- `zenodo/dabe_zenodo_deposit_2026-06-22.zip` as a convenience upload bundle.

The script intentionally excludes checkpoints, caches, pyc files, large model
artifacts, and unrelated temporary files.

## DOI Reservation Flow

1. Create a Zenodo draft for the paper record.
2. Use Zenodo's reserve DOI option.
3. Verify the reserved DOI `10.5281/zenodo.20796190` in the files under
   `zenodo/metadata/` and, if desired, in the manuscript title page or
   code/data availability section.
4. Rebuild the PDF and rerun the bundle script.
5. Upload the final bundle contents and publish.

Do not publish until the PDF, metadata, and citation files agree on title,
authors, version, DOI, and license.

## Current Paper Identity

- Title: Density-Adaptive Bitmask Encoding: Fixed Chunks with Sparse Lexical Repair
- Authors:
  - Mainasara Al-amin Tsowa `<mainasara@qrk.ng>`
  - Babangida Usman Tsowa `<babangida@qrk.ng>`
  - Abdul-malik Abdullahi Mustapha `<maleek@qrk.ng>`
- Repository: `git@github.com:qrk-labs/dabe-clean-20260619.git`
- Current local branch at packaging time: `exp/093-fixed-rate-20bpt-baseline`
- Source commit: recorded dynamically in `zenodo/deposit/bundle_summary.json`
  and `zenodo/deposit/file_manifest.json` when the deposit bundle is built.

## Notes For Upload

- Upload the paper as a preprint/publication record.
- Upload the artifact bundle separately as software, or create a GitHub release
  and let Zenodo archive that release.
- Keep the paper claim narrow: learned GPT-2-token-ID reconstruction, not
  raw-text compression superiority over BPE.
- Keep deterministic Python-code experiments framed as controlled
  domain-adaptation probes, not broad Python-code benchmarks.
