#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
REQUIRE_GPU="${REQUIRE_GPU:-0}"

cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN}" REQUIRE_GPU=0 bash scripts/check_environment.sh
PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/verl" "${PYTHON_BIN}" -m pytest -q \
  vagen/tests/test_concat_val_multi_turn.py \
  vagen/tests/test_value_mask_regression.py \
  vagen/tests/test_no_concat_episode_grpo.py \
  vagen/tests/test_no_concat_agent_loop.py \
  vagen/tests/test_logprob_parity.py \
  vagen/tests/test_multimodal_support.py \
  vagen/tests/test_state_relative_preflight.py \
  vagen/tests/test_sokoban_reward_bias.py \
  vagen/tests/test_sokoban_seeding.py \
  vagen/tests/test_seed_partitions.py \
  vagen/tests/test_navigation_state_anchor.py \
  vagen/tests/test_observation_ablation.py \
  vagen/tests/test_gpu_metrics.py \
  vagen/tests/test_rollout_analysis.py \
  vagen/tests/test_experiment_entrypoints.py \
  verl/tests/trainer/ppo/test_sparse_value_supervision_on_cpu.py \
  verl/tests/trainer/ppo/test_policy_weights_on_cpu.py

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[SKIP] GPU smoke was not run because no NVIDIA GPU is available."
  if [[ "${REQUIRE_GPU}" == "1" ]]; then
    exit 2
  fi
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN}" REQUIRE_GPU=1 bash scripts/check_environment.sh

EVAL_ENVS="${EVAL_ENVS:-4}"
if [[ "${RUN_LOCAL_EVAL:-1}" == "1" ]]; then
  ENVIRONMENT=frozenlake \
  N_ENVS="${EVAL_ENVS}" \
  TAG=frozenlake_smoke \
  DUMP_DIR="${ROOT_DIR}/exps/smoke/frozenlake_eval" \
    bash scripts/run_visual_eval.sh

  ENVIRONMENT=sokoban \
  N_ENVS="${EVAL_ENVS}" \
  TAG=sokoban_smoke \
  DUMP_DIR="${ROOT_DIR}/exps/smoke/sokoban_eval" \
    bash scripts/run_visual_eval.sh
fi

SMOKE_ENVIRONMENT_VALUE="${SMOKE_ENVIRONMENT:-sokoban}"
SMOKE_BATCH_SIZE_VALUE="${SMOKE_BATCH_SIZE:-2}"

if [[ "${RUN_CORE_METHOD_SMOKES:-1}" == "1" ]]; then
  METHOD=concat_grpo \
  ENVIRONMENT="${SMOKE_ENVIRONMENT_VALUE}" \
  TOTAL_STEPS="${SMOKE_CORE_STEPS:-1}" \
  TRAIN_BATCH_SIZE="${SMOKE_BATCH_SIZE_VALUE}" \
  ROLLOUT_N="${SMOKE_ROLLOUT_N:-2}" \
  VAL_BEFORE_TRAIN=False \
  TEST_FREQ=-1 \
  SAVE_FREQ=-1 \
  EXPERIMENT_NAME="${SMOKE_ENVIRONMENT_VALUE}_concat_grpo_smoke" \
    bash scripts/run_training_method.sh

  METHOD=no_concat_gae \
  ENVIRONMENT="${SMOKE_ENVIRONMENT_VALUE}" \
  TOTAL_STEPS="${SMOKE_CORE_STEPS:-1}" \
  TRAIN_BATCH_SIZE="${SMOKE_BATCH_SIZE_VALUE}" \
  ROLLOUT_N=1 \
  VAL_BEFORE_TRAIN=False \
  TEST_FREQ=-1 \
  SAVE_FREQ=-1 \
  EXPERIMENT_NAME="${SMOKE_ENVIRONMENT_VALUE}_no_concat_gae_smoke" \
    bash scripts/run_training_method.sh
fi

ENVIRONMENT="${SMOKE_ENVIRONMENT_VALUE}" \
TOTAL_STEPS="${SMOKE_STEPS:-5}" \
TRAIN_BATCH_SIZE="${SMOKE_BATCH_SIZE_VALUE}" \
ROLLOUT_N="${SMOKE_ROLLOUT_N:-2}" \
VAL_BEFORE_TRAIN=False \
TEST_FREQ=-1 \
SAVE_FREQ=-1 \
EXPERIMENT_NAME="${SMOKE_ENVIRONMENT_VALUE}_episode_grpo_smoke" \
bash scripts/run_no_concat_episode_grpo.sh
