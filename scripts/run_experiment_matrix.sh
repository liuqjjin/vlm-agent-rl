#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE="${1:-}"

if [[ -z "${PHASE}" ]]; then
  echo "Usage: $0 smoke|screening|confirmatory|state-preflight [rollout.jsonl ...]" >&2
  exit 1
fi

case "${PHASE}" in
  smoke)
    REQUIRE_GPU=1 bash "${ROOT_DIR}/scripts/run_smoke.sh"
    ;;
  screening)
    for reward_mode in outcome bounded_process format_gate; do
      for loss_weighting in token turn trajectory; do
        ENVIRONMENT=sokoban \
        REWARD_MODE="${reward_mode}" \
        LOSS_WEIGHTING="${loss_weighting}" \
        SEED=0 \
        TOTAL_STEPS="${SCREENING_STEPS:-50}" \
        TRAIN_BATCH_SIZE="${SCREENING_BATCH_SIZE:-4}" \
        ROLLOUT_N="${SCREENING_ROLLOUT_N:-4}" \
        TEST_FREQ="${SCREENING_TEST_FREQ:-25}" \
        SAVE_FREQ=-1 \
        bash "${ROOT_DIR}/scripts/run_no_concat_episode_grpo.sh"
      done
    done
    ;;
  confirmatory)
    IFS=',' read -r -a selected <<< "${SELECTED_CONFIGS:-outcome:trajectory}"
    IFS=',' read -r -a seeds <<< "${CONFIRMATORY_SEEDS:-0,1,2}"
    for specification in "${selected[@]}"; do
      reward_mode="${specification%%:*}"
      loss_weighting="${specification##*:}"
      for seed in "${seeds[@]}"; do
        ENVIRONMENT="${CONFIRMATORY_ENVIRONMENT:-sokoban}" \
        REWARD_MODE="${reward_mode}" \
        LOSS_WEIGHTING="${loss_weighting}" \
        SEED="${seed}" \
        TOTAL_STEPS="${CONFIRMATORY_STEPS:-401}" \
        TRAIN_BATCH_SIZE="${CONFIRMATORY_BATCH_SIZE:-8}" \
        ROLLOUT_N="${CONFIRMATORY_ROLLOUT_N:-4}" \
        bash "${ROOT_DIR}/scripts/run_no_concat_episode_grpo.sh"
      done
    done
    ;;
  state-preflight)
    shift
    if (( $# == 0 )); then
      echo "[ERROR] provide one or more no-concat rollout JSONL files." >&2
      exit 1
    fi
    PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/verl" python -m vagen.analysis.state_relative_preflight \
      "$@" \
      --output "${STATE_PREFLIGHT_OUTPUT:-${ROOT_DIR}/results/state-relative-preflight.json}" \
      --fail-on-stop
    ;;
  *)
    echo "[ERROR] unknown phase '${PHASE}'." >&2
    exit 1
    ;;
esac
