#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${VAGEN_GPU_ENV:-vagen}"
MIN_FREE_GB="${MIN_FREE_GB:-600}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] No NVIDIA GPU is visible. Start the AutoDL instance with a GPU first." >&2
  exit 2
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda is required. Use an AutoDL PyTorch/Conda image with Python 3.12." >&2
  exit 1
fi

if command -v apt-get >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install --yes libvulkan1 vulkan-tools
fi

FREE_GB="$(df -Pk "${ROOT_DIR}" | awk 'NR==2 {printf "%d", $4/1024/1024}')"
if (( FREE_GB < MIN_FREE_GB )); then
  echo "[ERROR] Only ${FREE_GB} GiB is free; at least ${MIN_FREE_GB} GiB is required." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  conda create \
    --name "${ENV_NAME}" \
    --override-channels \
    --channel conda-forge \
    python=3.12 \
    pip \
    --yes
fi
conda activate "${ENV_NAME}"

export MAX_JOBS="${MAX_JOBS:-16}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"
export WANDB_MODE="${WANDB_MODE:-offline}"
mkdir -p "${HF_HOME}" "${ROOT_DIR}/artifacts/environment"

pushd "${ROOT_DIR}/verl" >/dev/null
# Install verl and dependencies
USE_MEGATRON=0 USE_SGLANG=1 bash scripts/install_vllm_sglang_mcore.sh
python -m pip install --no-deps --editable .
# Clean up any downloaded wheel files from install script to keep submodule clean
find . -maxdepth 1 -name "*.whl" -type f -delete
popd >/dev/null

python -m pip install --requirement "${ROOT_DIR}/requirements/cpu-test.txt"
python -m pip install \
  "opencv-python==4.11.0.86" \
  "transformers==4.57.1" \
  "trl==0.26.2" \
  "ai2thor==5.0.0" \
  "fire==0.7.1" \
  "uvicorn<0.41" \
  "pytest==8.4.1" \
  "pytest-asyncio==1.1.0" \
  "ruff==0.12.7"
python -m pip install --editable "${ROOT_DIR}"
python -m pip check

python -m pip freeze > "${ROOT_DIR}/artifacts/environment/gpu-pip-freeze.txt"
nvidia-smi -q > "${ROOT_DIR}/artifacts/environment/nvidia-smi.txt"

PYTHON_BIN=python REQUIRE_GPU=1 bash "${ROOT_DIR}/scripts/check_environment.sh"

PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/verl" python -m pytest -q \
  "${ROOT_DIR}/vagen/tests/test_value_mask_regression.py" \
  "${ROOT_DIR}/vagen/tests/test_no_concat_episode_grpo.py" \
  "${ROOT_DIR}/vagen/tests/test_logprob_parity.py"

if [[ "${DOWNLOAD_MODEL:-0}" == "1" ]]; then
  BOOTSTRAP_MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}" \
    python - <<'PY'
import os

from huggingface_hub import snapshot_download

snapshot_download(repo_id=os.environ["BOOTSTRAP_MODEL_PATH"])
PY
fi
if [[ "${PRELOAD_NAVIGATION:-0}" == "1" ]]; then
  python -m vagen.envs.navigation.pre_download_scenes
fi

echo "[OK] AutoDL bootstrap completed in conda env '${ENV_NAME}'."
echo "     Next: bash scripts/run_smoke.sh"
