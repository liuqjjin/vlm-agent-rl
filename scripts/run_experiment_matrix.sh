#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE="${1:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NAVIGATION_SERVER_PID=""

usage() {
  echo "Usage: $0 describe|dry-run|smoke|base-eval|core-screening|episode-screening|confirmatory|anti-cheat|state-preflight|analyze [args]" >&2
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
  dry-run)
    export DRY_RUN=1
    export PYTHON_BIN
    export TOTAL_STEPS=5
    export TRAIN_BATCH_SIZE=2
    DRY_RUN_ROOT="${DRY_RUN_ROOT:-/private/tmp/vlm-agent-rl-matrix-dry-run}"
    for environment in sokoban navigation; do
      for method in concat_grpo no_concat_gae no_concat_episode_grpo; do
        EXPERIMENT_DIR="${DRY_RUN_ROOT}/${environment}_${method}_seed0" \
          run_training "${method}" "${environment}" 0
      done
      N_ENVS=2 run_eval "${environment}" none
    done
    ;;
  smoke)
    REQUIRE_GPU=1 bash "${ROOT_DIR}/scripts/run_smoke.sh"
    ;;
  base-eval)
    IFS=',' read -r -a environments <<< "${ENVIRONMENTS:-sokoban,navigation}"
    for environment in "${environments[@]}"; do
      run_eval "${environment}" none
    done
    ;;
  core-screening)
    IFS=',' read -r -a environments <<< "${ENVIRONMENTS:-sokoban}"
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
  anti-cheat)
    if [[ -z "${EVAL_MODEL_PATH:-}" ]]; then
      echo "[ERROR] Set EVAL_MODEL_PATH to the selected checkpoint or base model." >&2
      exit 2
    fi
    IFS=',' read -r -a environments <<< "${ENVIRONMENTS:-sokoban,navigation}"
    for environment in "${environments[@]}"; do
      for ablation in none remove shuffle_tiles; do
        EVAL_METHOD="${EVAL_METHOD:-selected_checkpoint}" \
        MODEL_PATH="${EVAL_MODEL_PATH}" \
          run_eval "${environment}" "${ablation}"
      done
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
