#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEARCH_ROOT="${1:-${ROOT_DIR}/exps}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "[ERROR] Export WANDB_API_KEY before syncing offline runs." >&2
  exit 2
fi

found=0
while IFS= read -r run; do
  found=1
  wandb sync "${run}"
done < <(find "${SEARCH_ROOT}" -type d -name 'offline-run-*' -print)

if (( found == 0 )); then
  echo "[INFO] No offline W&B runs found under ${SEARCH_ROOT}."
fi
