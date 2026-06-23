> **Development freeze:** DABE is frozen as the publication artifact for the preprint at https://doi.org/10.5281/zenodo.20797432. Do not run new experiments or alter scientific claims unless the user explicitly unfreezes development. See [`FREEZE_NOTICE.md`](../FREEZE_NOTICE.md).

# Code And Data Availability

The source code, paper source, figure scripts, selected experiment summaries,
and lightweight diagnostic artifacts are prepared for archival in the DABE
Zenodo artifact record. The corresponding repository is:

`git@github.com:qrk-labs/dabe-clean-20260619.git`

The paper-facing bundle records the exact source branch and commit in
`zenodo/deposit/bundle_summary.json` and `zenodo/deposit/file_manifest.json`
when `zenodo/scripts/build_deposit_bundle.py` is run.

The main text-domain experiments use TinyStories text tokenized into GPT-2 token
IDs and evaluated as fixed 64-token reconstruction chunks. The reported bitrate
values are learned GPT-2-token-ID reconstruction rates. They should not be read
as claims that DABE is a raw-text compressor superior to BPE.

The Python-code experiments are deterministic controlled probes. EXP-097 tests
zero-shot transfer of the TinyStories-trained sparse-repair checkpoint to
generated Python-like code. EXP-098 through EXP-100 test in-domain adaptation on
the same deterministic generator. These experiments support mechanism and scope
claims, not broad Python-code benchmark claims.

Large checkpoints are intentionally excluded from the default Zenodo package.
The included artifacts are sufficient to inspect the paper's reported metrics,
figure data, launch contracts, diagnostic traces, and source/configuration
state. Checkpoints can be archived separately if required for review.
