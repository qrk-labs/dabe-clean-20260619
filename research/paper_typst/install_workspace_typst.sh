#!/usr/bin/env bash
set -euo pipefail

# Installs Typst into the repository-local tools directory.
# This avoids global package-manager writes and keeps the paper build reproducible.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TOOLS_DIR="${REPO_ROOT}/research/.tools"
BIN_DIR="${TOOLS_DIR}/bin"

mkdir -p "${BIN_DIR}"

if command -v "${BIN_DIR}/typst" >/dev/null 2>&1; then
  "${BIN_DIR}/typst" --version
  exit 0
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo is required for the workspace-local Typst install." >&2
  exit 1
fi

cargo install typst-cli --root "${TOOLS_DIR}" --locked
"${BIN_DIR}/typst" --version
