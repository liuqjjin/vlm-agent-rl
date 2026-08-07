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
EVALUATION_ROLE="${EVALUATION_ROLE:-diagnostic}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-}"
SOURCE_SELECTION_MANIFEST="${SOURCE_SELECTION_MANIFEST:-}"
SOURCE_EXPORT_MANIFEST="${SOURCE_EXPORT_MANIFEST:-}"
SOURCE_METHOD="${SOURCE_METHOD:-}"
SOURCE_ENVIRONMENT="${SOURCE_ENVIRONMENT:-}"
SOURCE_TRAIN_SEED="${SOURCE_TRAIN_SEED:-}"
SOURCE_CHECKPOINT_STEP="${SOURCE_CHECKPOINT_STEP:-}"
WRITE_MANIFEST_ON_DRY_RUN="${WRITE_MANIFEST_ON_DRY_RUN:-0}"

case "${OBSERVATION_ABLATION}" in
  none|remove|shuffle_tiles) ;;
  *)
    echo "[ERROR] OBSERVATION_ABLATION must be none, remove, or shuffle_tiles." >&2
    exit 1
    ;;
esac

case "${EVALUATION_ROLE}" in
  diagnostic|base_eval|anti_cheat|final_test) ;;
  *)
    echo "[ERROR] EVALUATION_ROLE must be diagnostic, base_eval, anti_cheat, or final_test." >&2
    exit 1
    ;;
esac

case "${ENVIRONMENT}" in
  sokoban)
    EVAL_SCRIPT="${ROOT_DIR}/examples/evaluate/sokoban/sglang/eval_qwen25_vl_3b.sh"
    DEFAULT_SEED_START=10129  # Sokoban test: [10129, 10256], disjoint from train+val
    DEFAULT_N_ENVS=128
    ;;
  navigation)
    EVAL_SCRIPT="${ROOT_DIR}/examples/evaluate/navigation/sglang/eval_qwen25_vl_3b.sh"
    DEFAULT_SEED_START=30  # Navigation test: base 30-59
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
if ! [[ "${DP_SIZE_VALUE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] DP_SIZE must be a positive integer." >&2
  exit 1
fi
if ! [[ "${TP_SIZE_VALUE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] TP_SIZE must be a positive integer." >&2
  exit 1
fi
EXPECTED_GPU_COUNT=$((DP_SIZE_VALUE * TP_SIZE_VALUE))
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
  "run.resume=skip_completed"
)

GIT_DIRTY=False
if [[ -n "$(git -C "${ROOT_DIR}" status --porcelain)" ]]; then
  GIT_DIRTY=True
fi
if [[ "${GIT_DIRTY}" == "True" && "${ALLOW_DIRTY}" != "1" ]]; then
  if [[ "${DRY_RUN}" != "1" ]]; then
    echo "[ERROR] Refusing a formal evaluation from a dirty worktree; commit changes or set ALLOW_DIRTY=1." >&2
    exit 2
  fi
fi

if [[ "${EVALUATION_ROLE}" == "final_test" ]]; then
  if [[ "${OBSERVATION_ABLATION}" != "none" ]]; then
    echo "[ERROR] final_test requires OBSERVATION_ABLATION=none." >&2
    exit 2
  fi
  for required_value in \
    SOURCE_RUN_DIR SOURCE_SELECTION_MANIFEST SOURCE_EXPORT_MANIFEST \
    SOURCE_METHOD SOURCE_ENVIRONMENT SOURCE_TRAIN_SEED SOURCE_CHECKPOINT_STEP; do
    if [[ -z "${!required_value}" ]]; then
      echo "[ERROR] final_test requires ${required_value}." >&2
      exit 2
    fi
  done
  if [[ "${SOURCE_ENVIRONMENT}" != "${ENVIRONMENT}" ]]; then
    echo "[ERROR] final_test cannot evaluate a checkpoint on a different environment." >&2
    exit 2
  fi
fi

write_manifest() {
  mkdir -p "${DUMP_DIR}"
  EVAL_METHOD="${EVAL_METHOD:-base}" \
EVAL_ENVIRONMENT="${ENVIRONMENT}" \
EVAL_MODEL_PATH="${MODEL_PATH}" \
EVAL_ABLATION="${OBSERVATION_ABLATION}" \
EVAL_ROLE="${EVALUATION_ROLE}" \
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
EVAL_SOURCE_RUN_DIR="${SOURCE_RUN_DIR}" \
EVAL_SOURCE_SELECTION_MANIFEST="${SOURCE_SELECTION_MANIFEST}" \
EVAL_SOURCE_EXPORT_MANIFEST="${SOURCE_EXPORT_MANIFEST}" \
EVAL_SOURCE_METHOD="${SOURCE_METHOD}" \
EVAL_SOURCE_ENVIRONMENT="${SOURCE_ENVIRONMENT}" \
EVAL_SOURCE_TRAIN_SEED="${SOURCE_TRAIN_SEED}" \
EVAL_SOURCE_CHECKPOINT_STEP="${SOURCE_CHECKPOINT_STEP}" \
PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/verl${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" - "${COMMAND[@]}" <<'PY'
import os
import sys

from vagen.utils.run_manifest import write_compatible_manifest

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
    "evaluation_role": os.environ["EVAL_ROLE"],
    "seed_start": int(os.environ["EVAL_SEED_START"]),
    "seed_end_exclusive": int(os.environ["EVAL_SEED_END_EXCLUSIVE"]),
    "resume_mode": "skip_completed",
    "command": sys.argv[1:],
}
optional_strings = {
    "source_run_dir": "EVAL_SOURCE_RUN_DIR",
    "source_selection_manifest": "EVAL_SOURCE_SELECTION_MANIFEST",
    "source_export_manifest": "EVAL_SOURCE_EXPORT_MANIFEST",
    "source_method": "EVAL_SOURCE_METHOD",
    "source_environment": "EVAL_SOURCE_ENVIRONMENT",
}
for key, environment_name in optional_strings.items():
    if os.environ[environment_name]:
        manifest[key] = os.environ[environment_name]
for key, environment_name in {
    "source_train_seed": "EVAL_SOURCE_TRAIN_SEED",
    "source_checkpoint_step": "EVAL_SOURCE_CHECKPOINT_STEP",
}.items():
    if os.environ[environment_name]:
        manifest[key] = int(os.environ[environment_name])
write_compatible_manifest(
    os.path.join(os.environ["EVAL_DUMP_DIR"], "manifest.json"),
    manifest,
    require_existing_match=True,
)
PY
}

if [[ "${DRY_RUN}" == "1" ]]; then
  if [[ "${WRITE_MANIFEST_ON_DRY_RUN}" == "1" ]]; then
    write_manifest
  fi
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

if [[ "${EVALUATION_ROLE}" == "final_test" ]]; then
  EVAL_EXPORT_PATH="${SOURCE_EXPORT_MANIFEST}" EVAL_EXPECTED_MODEL="${MODEL_PATH}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["EVAL_EXPORT_PATH"])
payload = json.loads(path.read_text())
if payload.get("artifact_type") != "fsdp_lora_checkpoint_export":
    raise SystemExit(f"[ERROR] invalid export manifest: {path}")
if payload.get("status") != "complete":
    raise SystemExit(f"[ERROR] checkpoint export is not complete: {path}")
if Path(payload["model_path"]).resolve() != Path(os.environ["EVAL_EXPECTED_MODEL"]).resolve():
    raise SystemExit("[ERROR] MODEL_PATH does not match source export manifest")
PY
fi

write_manifest

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

RUN_RESUME_STATE="$(
  RUN_STATE_DIR="${DUMP_DIR}" \
  PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/verl${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" - <<'PY'
import os

from vagen.utils.run_manifest import classify_run_for_resume

print(classify_run_for_resume(os.environ["RUN_STATE_DIR"]))
PY
)"
case "${RUN_RESUME_STATE}" in
  complete)
    echo "[SKIP] Evaluation run is already complete: ${DUMP_DIR}"
    exit 0
    ;;
  tainted-gpu-metrics)
    echo "[ERROR] This run has incomplete GPU sampling evidence; use a new dump directory." >&2
    exit 2
    ;;
  failed-parity)
    echo "[ERROR] Unexpected parity evidence exists in this evaluation directory; use a new dump directory." >&2
    exit 2
    ;;
  resumable) ;;
  *)
    echo "[ERROR] Unknown run resume state: ${RUN_RESUME_STATE}" >&2
    exit 2
    ;;
esac

env \
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
  --expected-device-count "${EXPECTED_GPU_COUNT}" \
  -- "${COMMAND[@]}"
