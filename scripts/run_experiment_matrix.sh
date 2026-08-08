#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE="${1:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NAVIGATION_SERVER_PID=""

usage() {
  echo "Usage: $0 describe|validate-matrix|dry-run|smoke|base-eval|core-screening|episode-screening|confirmatory|select-checkpoints|export-checkpoints|final-test|final-results|publish-results|anti-cheat|state-preflight|analyze [args]" >&2
}

cleanup() {
  if [[ -n "${NAVIGATION_SERVER_PID}" ]]; then
    kill "${NAVIGATION_SERVER_PID}" >/dev/null 2>&1 || true
    wait "${NAVIGATION_SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

ensure_navigation_server() {
  local url="${NAVIGATION_SERVER_URL:-http://127.0.0.1:8000}"
  if curl --silent --fail "${url}/health" >/dev/null 2>&1; then
    return
  fi

  local log_dir="${ROOT_DIR}/exps/system"
  mkdir -p "${log_dir}"
  PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/verl" \
    "${PYTHON_BIN}" -m vagen.envs.navigation.serve \
    --host=127.0.0.1 \
    --port=8000 \
    "--devices=${NAVIGATION_DEVICES:-[0]}" \
    "--max_envs=${NAVIGATION_MAX_ENVS:-8}" \
    "--max_inflight=${NAVIGATION_MAX_ENVS:-8}" \
    "--thread_pool_size=${NAVIGATION_MAX_ENVS:-8}" \
    >"${log_dir}/navigation_server.log" 2>&1 &
  NAVIGATION_SERVER_PID=$!

  local attempts=0
  until curl --silent --fail "${url}/health" >/dev/null 2>&1; do
    if ! kill -0 "${NAVIGATION_SERVER_PID}" >/dev/null 2>&1; then
      echo "[ERROR] Navigation server exited during startup." >&2
      tail -n 100 "${log_dir}/navigation_server.log" >&2 || true
      exit 2
    fi
    attempts=$((attempts + 1))
    if (( attempts >= 60 )); then
      echo "[ERROR] Navigation server did not become ready in 120 seconds." >&2
      exit 2
    fi
    sleep 2
  done
}

run_training() {
  local method="$1"
  local environment="$2"
  local seed="$3"
  shift 3
  if [[ "${environment}" == "navigation" && "${DRY_RUN:-0}" != "1" ]]; then
    ensure_navigation_server
  fi
  METHOD="${method}" \
  ENVIRONMENT="${environment}" \
  SEED="${seed}" \
    bash "${ROOT_DIR}/scripts/run_training_method.sh" "$@"
}

run_eval() {
  local environment="$1"
  local ablation="$2"
  if [[ "${environment}" == "navigation" && "${DRY_RUN:-0}" != "1" ]]; then
    ensure_navigation_server
  fi
  ENVIRONMENT="${environment}" \
  OBSERVATION_ABLATION="${ablation}" \
  EVAL_METHOD="${EVAL_METHOD:-base}" \
  MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}" \
    bash "${ROOT_DIR}/scripts/run_visual_eval.sh"
}

if [[ -z "${PHASE}" ]]; then
  usage
  exit 1
fi

case "${PHASE}" in
  describe)
    cat "${ROOT_DIR}/experiments/matrix.yaml"
    ;;
  validate-matrix)
    PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/verl" \
      "${PYTHON_BIN}" -m vagen.analysis.experiment_contract \
      --matrix "${ROOT_DIR}/experiments/matrix.yaml" \
      --repo-root "${ROOT_DIR}" \
      --output "${MATRIX_VALIDATION_OUTPUT:-${ROOT_DIR}/results/gpu/matrix_contract.json}"
    ;;
  dry-run)
    export DRY_RUN=1
    export PYTHON_BIN
    export TOTAL_STEPS=5
    export TRAIN_BATCH_SIZE=2
    if [[ -z "${DRY_RUN_ROOT:-}" ]]; then
      DRY_RUN_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/vlm-agent-rl-matrix-dry-run.XXXXXX")"
    fi
    echo "[INFO] matrix dry-run artifacts: ${DRY_RUN_ROOT}"
    for environment in sokoban navigation; do
      for method in concat_grpo no_concat_gae no_concat_episode_grpo; do
        EXPERIMENT_DIR="${DRY_RUN_ROOT}/${environment}_${method}_seed0" \
          run_training "${method}" "${environment}" 0
      done
      DUMP_DIR="${DRY_RUN_ROOT}/eval/${environment}_none" \
      N_ENVS=2 \
        run_eval "${environment}" none
    done
    ;;
  smoke)
    REQUIRE_GPU=1 bash "${ROOT_DIR}/scripts/run_smoke.sh"
    ;;
  base-eval)
    IFS=',' read -r -a environments <<< "${ENVIRONMENTS:-sokoban,navigation}"
    for environment in "${environments[@]}"; do
      EVALUATION_ROLE=base_eval run_eval "${environment}" none
    done
    ;;
  core-screening)
    IFS=',' read -r -a environments <<< "${ENVIRONMENTS:-sokoban,navigation}"
    IFS=',' read -r -a methods <<< "${METHODS:-concat_grpo,no_concat_gae,no_concat_episode_grpo}"
    for environment in "${environments[@]}"; do
      for method in "${methods[@]}"; do
        experiment_name="${environment}_core_screening_${method}_seed0"
        if [[ "${method}" == "no_concat_episode_grpo" ]]; then
          experiment_name="${environment}_core_screening_${method}_${REWARD_MODE:-outcome}_${LOSS_WEIGHTING:-trajectory}_seed0"
        fi
        TOTAL_STEPS="${SCREENING_STEPS:-50}" \
        TRAIN_BATCH_SIZE="${SCREENING_BATCH_SIZE:-4}" \
        TEST_FREQ="${SCREENING_TEST_FREQ:-25}" \
        SAVE_FREQ=-1 \
        EXPERIMENT_NAME="${experiment_name}" \
          run_training "${method}" "${environment}" 0
      done
    done
    ;;
  episode-screening)
    for reward_mode in outcome bounded_process format_gate; do
      for loss_weighting in token turn trajectory; do
        REWARD_MODE="${reward_mode}" \
        LOSS_WEIGHTING="${loss_weighting}" \
        TOTAL_STEPS="${SCREENING_STEPS:-50}" \
        TRAIN_BATCH_SIZE="${SCREENING_BATCH_SIZE:-4}" \
        ROLLOUT_N="${SCREENING_ROLLOUT_N:-4}" \
        TEST_FREQ="${SCREENING_TEST_FREQ:-25}" \
        SAVE_FREQ=-1 \
        EXPERIMENT_NAME="sokoban_episode_screening_no_concat_episode_grpo_${reward_mode}_${loss_weighting}_seed0" \
          run_training no_concat_episode_grpo sokoban 0
      done
    done
    ;;
  confirmatory)
    IFS=',' read -r -a environments <<< "${ENVIRONMENTS:-sokoban,navigation}"
    IFS=',' read -r -a methods <<< "${SELECTED_METHODS:-no_concat_episode_grpo}"
    IFS=',' read -r -a seeds <<< "${CONFIRMATORY_SEEDS:-0,1,2}"
    for environment in "${environments[@]}"; do
      for method in "${methods[@]}"; do
        for seed in "${seeds[@]}"; do
          rollout_n="${CONFIRMATORY_ROLLOUT_N:-4}"
          if [[ "${method}" == "no_concat_gae" ]]; then
            rollout_n=1
          fi
          experiment_name="${environment}_confirmatory_${method}_seed${seed}"
          if [[ "${method}" == "no_concat_episode_grpo" ]]; then
            experiment_name="${environment}_confirmatory_${method}_${REWARD_MODE:-outcome}_${LOSS_WEIGHTING:-trajectory}_seed${seed}"
          fi
          TOTAL_STEPS="${CONFIRMATORY_STEPS:-401}" \
          TRAIN_BATCH_SIZE="${CONFIRMATORY_BATCH_SIZE:-8}" \
          ROLLOUT_N="${rollout_n}" \
          EXPERIMENT_NAME="${experiment_name}" \
            run_training "${method}" "${environment}" "${seed}"
        done
      done
    done
    ;;
  select-checkpoints)
    shift
    run_dirs=("$@")
    if (( ${#run_dirs[@]} == 0 )); then
      while IFS= read -r -d '' run_dir; do
        run_dirs+=("${run_dir}")
      done < <(
        find "${EXPERIMENT_ROOT:-${ROOT_DIR}/exps/vlm_agent_rl}" \
          -mindepth 1 -maxdepth 1 -type d -name '*_confirmatory_*' -print0
      )
    fi
    if (( ${#run_dirs[@]} == 0 )); then
      echo "[ERROR] no confirmatory training runs found." >&2
      exit 2
    fi
    for run_dir in "${run_dirs[@]}"; do
      PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/verl" \
        "${PYTHON_BIN}" -m vagen.analysis.final_evaluation select \
        --run "${run_dir}" \
        --output "${run_dir}/selection/checkpoint_selection.json"
    done
    ;;
  export-checkpoints)
    shift
    selections=("$@")
    if (( ${#selections[@]} == 0 )); then
      while IFS= read -r -d '' selection; do
        selections+=("${selection}")
      done < <(
        find "${EXPERIMENT_ROOT:-${ROOT_DIR}/exps/vlm_agent_rl}" \
          -type f -path '*/selection/checkpoint_selection.json' -print0
      )
    fi
    if (( ${#selections[@]} == 0 )); then
      echo "[ERROR] no checkpoint-selection manifests found." >&2
      exit 2
    fi
    export_root="${EXPORT_ROOT:-${ROOT_DIR}/exps/vlm_agent_rl_exports}"
    for selection in "${selections[@]}"; do
      run_dir="$(cd "$(dirname "${selection}")/.." && pwd)"
      output_dir="${export_root}/$(basename "${run_dir}")"
      export_args=()
      if [[ "${DRY_RUN:-0}" == "1" ]]; then
        export_args+=(--dry-run)
      fi
      PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/verl" \
        "${PYTHON_BIN}" -m vagen.analysis.final_evaluation export \
        --selection "${selection}" \
        --output-dir "${output_dir}" \
        "${export_args[@]}"
    done
    ;;
  final-test)
    shift
    export_manifests=("$@")
    if (( ${#export_manifests[@]} == 0 )); then
      while IFS= read -r -d '' export_manifest; do
        export_manifests+=("${export_manifest}")
      done < <(
        find "${EXPORT_ROOT:-${ROOT_DIR}/exps/vlm_agent_rl_exports}" \
          -mindepth 2 -maxdepth 2 -type f -name export_manifest.json -print0
      )
    fi
    if (( ${#export_manifests[@]} == 0 )); then
      echo "[ERROR] no checkpoint-export manifests found." >&2
      exit 2
    fi
    for export_manifest in "${export_manifests[@]}"; do
      export_fields=()
      while IFS= read -r -d '' field; do
        export_fields+=("${field}")
      done < <(
        "${PYTHON_BIN}" - "${export_manifest}" <<'PY'
import json
import os
import sys

payload = json.load(open(sys.argv[1]))
fields = (
    payload["environment"],
    payload["method"],
    payload["train_seed"],
    payload["checkpoint_step"],
    payload["model_path"],
    payload["source_run_dir"],
    payload["selection_manifest"],
)
for value in fields:
    os.write(sys.stdout.fileno(), str(value).encode() + b"\0")
PY
      )
      if (( ${#export_fields[@]} != 7 )); then
        echo "[ERROR] invalid export manifest fields: ${export_manifest}" >&2
        exit 2
      fi
      environment="${export_fields[0]}"
      method="${export_fields[1]}"
      train_seed="${export_fields[2]}"
      checkpoint_step="${export_fields[3]}"
      model_path="${export_fields[4]}"
      source_run_dir="${export_fields[5]}"
      selection_manifest="${export_fields[6]}"
      dump_dir="${FINAL_TEST_ROOT:-${ROOT_DIR}/exps/eval/final_test}/${environment}/${method}/train_seed_${train_seed}/checkpoint_${checkpoint_step}"
      EVALUATION_ROLE=final_test \
      EVAL_METHOD="${method}" \
      MODEL_PATH="${model_path}" \
      DUMP_DIR="${dump_dir}" \
      TAG="final_test_${environment}_${method}_train_seed_${train_seed}" \
      SOURCE_RUN_DIR="${source_run_dir}" \
      SOURCE_SELECTION_MANIFEST="${selection_manifest}" \
      SOURCE_EXPORT_MANIFEST="${export_manifest}" \
      SOURCE_METHOD="${method}" \
      SOURCE_ENVIRONMENT="${environment}" \
      SOURCE_TRAIN_SEED="${train_seed}" \
      SOURCE_CHECKPOINT_STEP="${checkpoint_step}" \
      WRITE_MANIFEST_ON_DRY_RUN="${DRY_RUN:-0}" \
        run_eval "${environment}" none
    done
    ;;
  final-results)
    shift
    final_runs=("$@")
    if (( ${#final_runs[@]} == 0 )); then
      while IFS= read -r -d '' manifest; do
        final_runs+=("$(dirname "${manifest}")")
      done < <(
        find "${FINAL_TEST_ROOT:-${ROOT_DIR}/exps/eval/final_test}" \
          -type f -name manifest.json -print0
      )
    fi
    if (( ${#final_runs[@]} == 0 )); then
      echo "[ERROR] no final-test runs found." >&2
      exit 2
    fi
    final_args=()
    for final_run in "${final_runs[@]}"; do
      final_args+=(--run "${final_run}")
    done
    base_args=()
    while IFS= read -r -d '' base_manifest; do
      manifest_role="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("evaluation_role", ""))' "${base_manifest}")"
      if [[ "${manifest_role}" == "base_eval" ]]; then
        base_args+=(--base-run "$(dirname "${base_manifest}")")
      fi
    done < <(find "${ROOT_DIR}/exps/eval" -type f -name manifest.json -print0)
    PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/verl" \
      "${PYTHON_BIN}" -m vagen.analysis.final_evaluation aggregate \
      "${final_args[@]}" \
      "${base_args[@]}" \
      --expected-train-seeds "${CONFIRMATORY_SEEDS:-0,1,2}" \
      --expected-methods "${FINAL_EXPECTED_METHODS:-${SELECTED_METHODS:-no_concat_episode_grpo}}" \
      --expected-environments "${ENVIRONMENTS:-sokoban,navigation}" \
      --output-dir "${FINAL_RESULTS_DIR:-${ROOT_DIR}/results/gpu/final}"
    ;;
  publish-results)
    PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/verl" \
      "${PYTHON_BIN}" -m vagen.analysis.final_evaluation publish \
      --results-dir "${FINAL_RESULTS_DIR:-${ROOT_DIR}/results/gpu/final}" \
      --output "${PUBLISHED_RESULTS_PATH:-${ROOT_DIR}/results/main_results.csv}"
    ;;
  anti-cheat)
    if [[ -z "${EVAL_MODEL_PATH:-}" ]]; then
      echo "[ERROR] Set EVAL_MODEL_PATH to an exported model directory." >&2
      exit 2
    fi
    if [[ -z "${EVAL_ENVIRONMENT:-}" ]]; then
      echo "[ERROR] Set EVAL_ENVIRONMENT to the checkpoint's training environment." >&2
      exit 2
    fi
    if [[ -z "${EVAL_METHOD:-}" ]]; then
      echo "[ERROR] Set EVAL_METHOD to the checkpoint's training method; it decides the evaluation context protocol." >&2
      exit 2
    fi
    model_run_name="$(basename "$(dirname "${EVAL_MODEL_PATH}")")"
    anti_cheat_root="${ANTI_CHEAT_ROOT:-${ROOT_DIR}/exps/eval/anti_cheat/${model_run_name}}"
    for ablation in none remove shuffle_tiles; do
      EVALUATION_ROLE=anti_cheat \
      EVAL_METHOD="${EVAL_METHOD}" \
      MODEL_PATH="${EVAL_MODEL_PATH}" \
      DUMP_DIR="${anti_cheat_root}/${EVAL_ENVIRONMENT}_${ablation}" \
        run_eval "${EVAL_ENVIRONMENT}" "${ablation}"
    done
    ;;
  state-preflight)
    shift
    if (( $# == 0 )); then
      echo "[ERROR] provide one or more no-concat rollout JSONL files." >&2
      exit 1
    fi
    PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/verl" \
      "${PYTHON_BIN}" -m vagen.analysis.state_relative_preflight \
      "$@" \
      --output "${STATE_PREFLIGHT_OUTPUT:-${ROOT_DIR}/results/state-relative-preflight.json}" \
      --fail-on-stop
    ;;
  analyze)
    shift
    if (( $# == 0 )); then
      echo "[ERROR] pass analyzer inputs such as --run exps/... or --eval-dump exps/eval/... ." >&2
      exit 1
    fi
    PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/verl" \
      "${PYTHON_BIN}" -m vagen.analysis.analyze_rollouts \
      --output-dir "${ANALYSIS_OUTPUT_DIR:-${ROOT_DIR}/results/gpu-analysis}" \
      "$@"
    ;;
  *)
    usage
    exit 1
    ;;
esac
