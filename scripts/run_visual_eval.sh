#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVIRONMENT="${ENVIRONMENT:-sokoban}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
OBSERVATION_ABLATION="${OBSERVATION_ABLATION:-none}"
N_ENVS="${N_ENVS:-}"
SEED_START="${SEED_START:-}"
MAX_CONCURRENT_JOBS="${MAX_CONCURRENT_JOBS:-4}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"
PORT_VALUE="${PORT:-30000}"
DP_SIZE_VALUE="${DP_SIZE:-1}"
TP_SIZE_VALUE="${TP_SIZE:-1}"
MEM_FRACTION_VALUE="${MEM_FRACTION:-0.80}"

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
    DEFAULT_SEED_START=10001
    DEFAULT_N_ENVS=128
    ;;
  navigation)
    EVAL_SCRIPT="${ROOT_DIR}/examples/evaluate/navigation/sglang/eval_qwen25_vl_3b.sh"
    DEFAULT_SEED_START=30
    DEFAULT_N_ENVS=30
    ;;
  frozenlake)
    EVAL_SCRIPT="${ROOT_DIR}/examples/evaluate/frozenlake/sglang/eval_qwen25_vl_3b.sh"
    DEFAULT_SEED_START=10001
    DEFAULT_N_ENVS=128
    ;;
  *)
    echo "[ERROR] ENVIRONMENT must be frozenlake, sokoban, or navigation." >&2
    exit 1
    ;;
esac

if [[ -z "${N_ENVS}" ]]; then
  N_ENVS="${DEFAULT_N_ENVS}"
fi
if ! [[ "${N_ENVS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] N_ENVS must be a positive integer." >&2
  exit 1
fi
if [[ -z "${SEED_START}" ]]; then
  SEED_START="${DEFAULT_SEED_START}"
fi
if ! [[ "${SEED_START}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] SEED_START must be a non-negative integer." >&2
  exit 1
fi
SEED_END_EXCLUSIVE=$((SEED_START + N_ENVS))
SEED_MAX=$((SEED_END_EXCLUSIVE - 1))
TAG="${TAG:-${ENVIRONMENT}_${OBSERVATION_ABLATION}}"
DUMP_DIR="${DUMP_DIR:-${ROOT_DIR}/exps/eval/${MODEL_PATH//\//_}/${TAG}}"
METRICS_DIR="${DUMP_DIR}/gpu_metrics"
EVAL_BACKEND=sglang
LOG_DIR_VALUE="${LOG_DIR:-${DUMP_DIR}/logs}"
CONFIG_CHECK_OUTPUT="${DUMP_DIR}/resolved_config.txt"

COMMAND=(
  bash "${EVAL_SCRIPT}"
  "envs.0.n_envs=${N_ENVS}"
  "envs.0.seed=[${SEED_START},${SEED_MAX},1]"
  "envs.0.tag_id=${TAG}"
  "envs.0.observation_ablation=${OBSERVATION_ABLATION}"
  "run.max_concurrent_jobs=${MAX_CONCURRENT_JOBS}"
  "run.resume=force_rerun"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[DRY RUN] '
  printf 'MODEL_PATH=%q DUMP_DIR=%q PYTHON_BIN=%q ' \
    "${MODEL_PATH}" "${DUMP_DIR}" "${PYTHON_BIN}"
  printf 'PORT=%q DP_SIZE=%q TP_SIZE=%q MEM_FRACTION=%q ' \
    "${PORT_VALUE}" "${DP_SIZE_VALUE}" "${TP_SIZE_VALUE}" "${MEM_FRACTION_VALUE}"
  printf 'LOG_DIR=%q CONFIG_CHECK_OUTPUT=%q ' \
    "${LOG_DIR_VALUE}" "${CONFIG_CHECK_OUTPUT}"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] visual Qwen2.5-VL evaluation requires an NVIDIA GPU." >&2
  exit 2
fi

GIT_DIRTY=False
if [[ -n "$(git -C "${ROOT_DIR}" status --porcelain)" ]]; then
  GIT_DIRTY=True
fi
if [[ "${GIT_DIRTY}" == "True" && "${ALLOW_DIRTY}" != "1" ]]; then
  echo "[ERROR] Refusing a formal evaluation from a dirty worktree; commit changes or set ALLOW_DIRTY=1." >&2
  exit 2
fi

mkdir -p "${DUMP_DIR}"
EVAL_METHOD="${EVAL_METHOD:-base}" \
EVAL_ENVIRONMENT="${ENVIRONMENT}" \
EVAL_MODEL_PATH="${MODEL_PATH}" \
EVAL_ABLATION="${OBSERVATION_ABLATION}" \
EVAL_SEED_START="${SEED_START}" \
EVAL_SEED_END_EXCLUSIVE="${SEED_END_EXCLUSIVE}" \
EVAL_DUMP_DIR="${DUMP_DIR}" \
EVAL_COMMIT="$(git -C "${ROOT_DIR}" rev-parse HEAD)" \
EVAL_VERL_COMMIT="$(git -C "${ROOT_DIR}/verl" rev-parse HEAD)" \
EVAL_GIT_DIRTY="${GIT_DIRTY}" \
EVAL_N_ENVS="${N_ENVS}" \
EVAL_BACKEND="${EVAL_BACKEND}" \
EVAL_PORT="${PORT_VALUE}" \
EVAL_DP_SIZE="${DP_SIZE_VALUE}" \
EVAL_TP_SIZE="${TP_SIZE_VALUE}" \
EVAL_MEM_FRACTION="${MEM_FRACTION_VALUE}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

manifest = {
    "commit": os.environ["EVAL_COMMIT"],
    "verl_commit": os.environ["EVAL_VERL_COMMIT"],
    "git_dirty": os.environ["EVAL_GIT_DIRTY"].lower() == "true",
    "method": os.environ["EVAL_METHOD"],
    "environment": os.environ["EVAL_ENVIRONMENT"],
    "model": os.environ["EVAL_MODEL_PATH"],
    "backend": os.environ["EVAL_BACKEND"],
    "port": int(os.environ["EVAL_PORT"]),
    "data_parallel_size": int(os.environ["EVAL_DP_SIZE"]),
    "tensor_parallel_size": int(os.environ["EVAL_TP_SIZE"]),
    "memory_fraction": float(os.environ["EVAL_MEM_FRACTION"]),
    "n_envs": int(os.environ["EVAL_N_ENVS"]),
    "observation_ablation": os.environ["EVAL_ABLATION"],
    "seed_start": int(os.environ["EVAL_SEED_START"]),
    "seed_end_exclusive": int(os.environ["EVAL_SEED_END_EXCLUSIVE"]),
}
Path(os.environ["EVAL_DUMP_DIR"], "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
PY

{
  printf '#!/usr/bin/env bash\n'
  printf 'set -euo pipefail\n'
  printf 'MODEL_PATH=%q DUMP_DIR=%q PYTHON_BIN=%q ' \
    "${MODEL_PATH}" "${DUMP_DIR}" "${PYTHON_BIN}"
  printf 'PORT=%q DP_SIZE=%q TP_SIZE=%q MEM_FRACTION=%q ' \
    "${PORT_VALUE}" "${DP_SIZE_VALUE}" "${TP_SIZE_VALUE}" "${MEM_FRACTION_VALUE}"
  printf 'LOG_DIR=%q CONFIG_CHECK_OUTPUT=%q ' \
    "${LOG_DIR_VALUE}" "${CONFIG_CHECK_OUTPUT}"
  printf '%q ' "${COMMAND[@]}"
  printf '"$@"\n'
} > "${DUMP_DIR}/eval_command.sh"
chmod +x "${DUMP_DIR}/eval_command.sh"

MODEL_PATH="${MODEL_PATH}" \
DUMP_DIR="${DUMP_DIR}" \
PYTHON_BIN="${PYTHON_BIN}" \
PORT="${PORT_VALUE}" \
DP_SIZE="${DP_SIZE_VALUE}" \
TP_SIZE="${TP_SIZE_VALUE}" \
MEM_FRACTION="${MEM_FRACTION_VALUE}" \
LOG_DIR="${LOG_DIR_VALUE}" \
CONFIG_CHECK_OUTPUT="${CONFIG_CHECK_OUTPUT}" \
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/run_with_gpu_metrics.py" \
  --output-dir "${METRICS_DIR}" \
  -- "${COMMAND[@]}"
