#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${VAGEN_CPU_ENV:-vagen}"
CONDA_EXE_PATH="${CONDA_EXE:-$(command -v conda || true)}"

if [[ -z "${CONDA_EXE_PATH}" ]]; then
  echo "[ERROR] conda is required for the pinned Python 3.12 CPU environment." >&2
  exit 1
fi

if ! "${CONDA_EXE_PATH}" env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  "${CONDA_EXE_PATH}" create \
    --name "${ENV_NAME}" \
    --override-channels \
    --channel conda-forge \
    python=3.12 \
    pip \
    --yes
fi

"${CONDA_EXE_PATH}" run --name "${ENV_NAME}" \
  python -m pip install --retries 20 --resume-retries 20 --timeout 120 \
  --requirement "${ROOT_DIR}/requirements/cpu-test.txt"
"${CONDA_EXE_PATH}" run --name "${ENV_NAME}" \
  python -m pip install --no-deps --editable "${ROOT_DIR}/verl" --editable "${ROOT_DIR}"
"${CONDA_EXE_PATH}" run --name "${ENV_NAME}" python -m pip check

echo "[OK] CPU environment '${ENV_NAME}' is ready."
echo "     conda run -n ${ENV_NAME} bash scripts/run_smoke.sh"
