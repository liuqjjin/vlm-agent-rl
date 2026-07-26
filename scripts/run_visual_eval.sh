#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVIRONMENT="${ENVIRONMENT:-sokoban}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
OBSERVATION_ABLATION="${OBSERVATION_ABLATION:-none}"
N_ENVS="${N_ENVS:-60}"
SEED_START="${SEED_START:-}"
MAX_CONCURRENT_JOBS="${MAX_CONCURRENT_JOBS:-4}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "${OBSERVATION_ABLATION}" in
  none|remove|shuffle_tiles) ;;
  *)
    echo "[ERROR] OBSERVATION_ABLATION must be none, remove, or shuffle_tiles." >&2
    exit 1
    ;;
esac

case "${ENVIRONMENT}" in
  sokoban)
    EVAL_SCRIPT="${ROOT_DIR}/examples/evaluate/sokoban/sglang/eval_qwen25_vl_3b.sh"
    DEFAULT_SEED_START=10000
    ;;
  navigation)
    EVAL_SCRIPT="${ROOT_DIR}/examples/evaluate/navigation/sglang/eval_qwen25_vl_3b.sh"
    DEFAULT_SEED_START=0
    ;;
  frozenlake)
    EVAL_SCRIPT="${ROOT_DIR}/examples/evaluate/frozenlake/sglang/eval_qwen25_vl_3b.sh"
    DEFAULT_SEED_START=10000
    ;;
  *)
    echo "[ERROR] ENVIRONMENT must be frozenlake, sokoban, or navigation." >&2
    exit 1
    ;;
esac

if [[ -z "${SEED_START}" ]]; then
  SEED_START="${DEFAULT_SEED_START}"
fi
SEED_END=$((SEED_START + N_ENVS))
TAG="${TAG:-${ENVIRONMENT}_${OBSERVATION_ABLATION}}"
DUMP_DIR="${DUMP_DIR:-${ROOT_DIR}/exps/eval/${MODEL_PATH//\//_}/${TAG}}"
METRICS_DIR="${DUMP_DIR}/gpu_metrics"

COMMAND=(
  bash "${EVAL_SCRIPT}"
  "envs.0.n_envs=${N_ENVS}"
  "envs.0.seed=[${SEED_START},${SEED_END},1]"
  "envs.0.tag_id=${TAG}"
  "envs.0.observation_ablation=${OBSERVATION_ABLATION}"
  "run.max_concurrent_jobs=${MAX_CONCURRENT_JOBS}"
  "run.resume=force_rerun"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[DRY RUN] MODEL_PATH=%q DUMP_DIR=%q ' "${MODEL_PATH}" "${DUMP_DIR}"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] visual Qwen2.5-VL evaluation requires an NVIDIA GPU." >&2
  exit 2
fi

mkdir -p "${DUMP_DIR}"
EVAL_METHOD="${EVAL_METHOD:-base}" \
EVAL_ENVIRONMENT="${ENVIRONMENT}" \
EVAL_MODEL_PATH="${MODEL_PATH}" \
EVAL_ABLATION="${OBSERVATION_ABLATION}" \
EVAL_SEED_START="${SEED_START}" \
EVAL_SEED_END="${SEED_END}" \
EVAL_DUMP_DIR="${DUMP_DIR}" \
EVAL_COMMIT="$(git -C "${ROOT_DIR}" rev-parse HEAD)" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

manifest = {
    "commit": os.environ["EVAL_COMMIT"],
    "method": os.environ["EVAL_METHOD"],
    "environment": os.environ["EVAL_ENVIRONMENT"],
    "model": os.environ["EVAL_MODEL_PATH"],
    "observation_ablation": os.environ["EVAL_ABLATION"],
    "seed_start": int(os.environ["EVAL_SEED_START"]),
    "seed_end_exclusive": int(os.environ["EVAL_SEED_END"]),
}
Path(os.environ["EVAL_DUMP_DIR"], "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
PY

MODEL_PATH="${MODEL_PATH}" \
DUMP_DIR="${DUMP_DIR}" \
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/run_with_gpu_metrics.py" \
  --output-dir "${METRICS_DIR}" \
  -- "${COMMAND[@]}"
