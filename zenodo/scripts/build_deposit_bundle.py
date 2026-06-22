#!/usr/bin/env python3
"""Build a lightweight Zenodo deposit bundle for the DABE paper.

The script copies only paper-facing sources, figure outputs, selected configs,
source/test files, experiment logs, launch contracts, and lightweight JSON/CSV
artifacts. It deliberately excludes checkpoints, caches, pyc files, and local
temporary outputs.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import shutil
import subprocess
import zipfile
from datetime import date
from pathlib import Path


TODAY = date(2026, 6, 22).isoformat()
EXPERIMENT_IDS = {
    "071",
    "077",
    "079",
    "080",
    "083",
    "084",
    "085",
    "086",
    "087",
    "088",
    "089",
    "090",
    "091",
    "092",
    "093",
    "094",
    "095",
    "096",
    "097",
    "098",
    "099",
    "100",
}
LIGHTWEIGHT_PATTERNS = [
    "*.json",
    "*.csv",
    "*.md",
    "*.txt",
    "*.yaml",
    "*.yml",
    "*.typ",
    "*.bib",
    "*.pdf",
    "*.svg",
    "*.png",
    "*.py",
    "*.cff",
    "LICENSE",
    "README*",
    "pyproject.toml",
]
EXCLUDE_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".git",
    "checkpoints",
    "lightning_logs",
    "wandb",
}
EXCLUDE_SUFFIXES = {
    ".ckpt",
    ".pt",
    ".pth",
    ".safetensors",
    ".pyc",
    ".DS_Store",
}


def run_git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "UNKNOWN"


def is_lightweight(path: Path) -> bool:
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return False
    if path.suffix in EXCLUDE_SUFFIXES:
        return False
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in LIGHTWEIGHT_PATTERNS)


def experiment_selected(path: Path) -> bool:
    text = "/".join(path.parts).lower()
    return any(f"exp{exp_id}" in text for exp_id in EXPERIMENT_IDS)


def copy_file(repo: Path, dest_root: Path, rel_path: str | Path) -> Path:
    rel = Path(rel_path)
    src = repo / rel
    if not src.exists():
        raise FileNotFoundError(f"Required source file is missing: {rel}")
    dest = dest_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def copy_tree_filtered(repo: Path, dest_root: Path, rel_root: str | Path) -> list[Path]:
    rel_base = Path(rel_root)
    src_base = repo / rel_base
    copied: list[Path] = []
    if not src_base.exists():
        return copied
    for src in sorted(src_base.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(repo)
        if is_lightweight(rel):
            copied.append(copy_file(repo, dest_root, rel))
    return copied


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(deposit: Path, repo: Path) -> None:
    files = []
    for path in sorted(p for p in deposit.rglob("*") if p.is_file()):
        rel = path.relative_to(deposit)
        files.append(
            {
                "path": str(rel),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    manifest = {
        "created_date": TODAY,
        "source_commit": run_git(repo, "rev-parse", "HEAD"),
        "source_branch": run_git(repo, "branch", "--show-current"),
        "file_count": len(files),
        "files": files,
    }
    (deposit / "file_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def write_bundle_summary(deposit: Path, repo: Path) -> None:
    summary = {
        "title": "DABE Zenodo deposit bundle",
        "created_date": TODAY,
        "source_commit": run_git(repo, "rev-parse", "HEAD"),
        "source_branch": run_git(repo, "branch", "--show-current"),
        "paper_record": {
            "resource_type": "Publication / Preprint",
            "license": "CC-BY-4.0",
            "doi_placeholder": "10.5281/zenodo.TODO",
        },
        "software_artifact_record": {
            "resource_type": "Software",
            "license": "MIT",
            "doi_placeholder": "10.5281/zenodo.TODO",
        },
        "excluded_by_design": [
            "model checkpoints",
            "large tensor/model binaries",
            "Python bytecode caches",
            "local temp folders",
            "WandB caches",
        ],
    }
    (deposit / "bundle_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def make_zip(deposit: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(p for p in deposit.rglob("*") if p.is_file()):
            archive.write(path, path.relative_to(deposit.parent))


def main() -> None:
    script_path = Path(__file__).resolve()
    repo = script_path.parents[2]
    zenodo_root = repo / "zenodo"
    deposit = zenodo_root / "deposit"

    if deposit.exists():
        shutil.rmtree(deposit)
    deposit.mkdir(parents=True)

    # Paper record payload.
    paper_files = [
        "research/paper_typst/main.pdf",
        "research/paper_typst/main.typ",
        "research/paper_typst/references.bib",
        "research/paper_typst/README.md",
        "research/paper_typst/Makefile",
        "research/paper_typst/install_workspace_typst.sh",
    ]
    for rel in paper_files:
        copy_file(repo, deposit / "paper", rel)
    copy_tree_filtered(repo, deposit / "paper", "research/paper_drafts")

    # Software/artifact record payload.
    core_files = [
        "LICENSE",
        "README.md",
        "pyproject.toml",
        "research/experiment_log.md",
        "research/literature_notes.md",
        "research/paper_checklist.md",
        "configs/dabe_tokenizer_autoencoder_smoke.yaml",
        "configs/feasibility_python_code_smoke.yaml",
        "scripts/build_paper_figures.py",
        "scripts/modal_dabe_tokenizer_autoencoder.py",
        "scripts/run_dabe_tokenizer_autoencoder.py",
        "src/training/dabe_tokenizer_autoencoder.py",
        "tests/test_dabe_tokenizer_autoencoder.py",
    ]
    for rel in core_files:
        copy_file(repo, deposit / "artifact", rel)

    # Zenodo-specific docs and metadata.
    for rel in [
        "zenodo/README.md",
        "zenodo/CHECKLIST.md",
        "zenodo/UPLOAD_FIELDS.md",
        "zenodo/CODE_AND_DATA_AVAILABILITY.md",
        "zenodo/LICENSES.md",
        "zenodo/REPRODUCIBILITY.md",
        "zenodo/artifact_manifest.json",
    ]:
        copy_file(repo, deposit / "artifact", rel)
    copy_tree_filtered(repo, deposit / "artifact", "zenodo/metadata")
    copy_tree_filtered(repo, deposit / "artifact", "zenodo/scripts")

    # Lightweight Modal summaries and launch contracts for cited experiments.
    for root in ["experiments/modal_downloads", "experiments/modal_launches"]:
        source_root = repo / root
        if not source_root.exists():
            continue
        for src in sorted(source_root.rglob("*")):
            if not src.is_file():
                continue
            rel = src.relative_to(repo)
            if experiment_selected(rel) and is_lightweight(rel):
                copy_file(repo, deposit / "artifact", rel)

    write_bundle_summary(deposit, repo)
    write_manifest(deposit, repo)
    make_zip(deposit, zenodo_root / f"dabe_zenodo_deposit_{TODAY}.zip")

    print(f"Built {deposit}")
    print(f"Wrote {deposit / 'file_manifest.json'}")
    print(f"Wrote {zenodo_root / f'dabe_zenodo_deposit_{TODAY}.zip'}")


if __name__ == "__main__":
    main()
