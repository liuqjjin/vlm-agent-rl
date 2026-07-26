#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${SCRIPT_DIR}/../config_base.yaml}"
PORT="${PORT:-30000}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
DP_SIZE="${DP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-1}"
MEM_FRACTION="${MEM_FRACTION:-0.80}"
DUMP_DIR="${DUMP_DIR:-${ROOT_DIR}/rollouts/qwen25_vl_3b_navigation}"
mkdir -p "${LOG_DIR}" "${DUMP_DIR}"

if ! curl --silent --show-error --fail \
  "${NAVIGATION_SERVER_URL:-http://127.0.0.1:8000}/health" >/dev/null; then
  echo "[ERROR] Navigation environment server is not ready." >&2
  exit 2
fi

SERVER_LOG="${LOG_DIR}/qwen25_vl_3b_server.log"
EVAL_LOG="${LOG_DIR}/qwen25_vl_3b_eval.log"

python -m sglang.launch_server \
  --host 0.0.0.0 \
  --log-level warning \
  --port "${PORT}" \
  --model-path "${MODEL_PATH}" \
  --dp-size "${DP_SIZE}" \
  --tp "${TP_SIZE}" \
  --trust-remote-code \
  --mem-fraction-static "${MEM_FRACTION}" \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "${SERVER_PID}" >/dev/null 2>&1 || true
  wait "${SERVER_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

source "${SCRIPT_DIR}/../../frozenlake/sglang/wait_for_server.sh"
wait_for_server

PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/verl" python -m vagen.evaluate.run_eval \
  --config "${CONFIG}" \
  run.backend=sglang \
  "backends.sglang.base_url=http://127.0.0.1:${PORT}/v1" \
  "backends.sglang.model=${MODEL_PATH}" \
  "experiment.dump_dir=${DUMP_DIR}" \
  "fileroot=${ROOT_DIR}" \
  "$@" \
  2>&1 | tee "${EVAL_LOG}"
