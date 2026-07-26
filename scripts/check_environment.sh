#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
REQUIRE_GPU="${REQUIRE_GPU:-0}"

cd "${ROOT_DIR}"

if [[ ! -f "verl/verl/version/version" ]]; then
  echo "[ERROR] verl submodule is not initialized." >&2
  exit 1
fi
if git submodule status | grep -Eq '^[+-]'; then
  echo "[ERROR] verl submodule does not match the gitlink recorded by the parent repository." >&2
  git submodule status >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import json
import platform

import fastapi
import fire
import httpx
import numpy
import openai
import ray
import tensordict
import torch
import transformers

print(json.dumps({
    "platform": platform.platform(),
    "python": platform.python_version(),
    "numpy": numpy.__version__,
    "fastapi": fastapi.__version__,
    "fire": fire.__version__,
    "httpx": httpx.__version__,
    "openai": openai.__version__,
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "ray": ray.__version__,
    "tensordict": tensordict.__version__,
    "transformers": transformers.__version__,
}, indent=2, sort_keys=True))
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
  echo "[INFO] No NVIDIA GPU detected; CPU tests remain available."
  if [[ "${REQUIRE_GPU}" == "1" ]]; then
    echo "[ERROR] This command requires a CUDA-capable NVIDIA GPU." >&2
    exit 2
  fi
fi

df -h "${ROOT_DIR}" | tail -n 1
echo "[OK] Environment checks passed."
