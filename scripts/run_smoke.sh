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
  vagen/tests/test_logprob_parity.py \
  vagen/tests/test_state_relative_preflight.py \
  vagen/tests/test_sokoban_reward_bias.py \
  verl/tests/trainer/ppo/test_sparse_value_supervision_on_cpu.py \
  verl/tests/trainer/ppo/test_policy_weights_on_cpu.py

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[SKIP] GPU smoke was not run because no NVIDIA GPU is available."
  if [[ "${REQUIRE_GPU}" == "1" ]]; then
    exit 2
  fi
  exit 0
fi

EVAL_ENVS="${EVAL_ENVS:-4}"
if [[ "${RUN_LOCAL_EVAL:-1}" == "1" ]]; then
  DUMP_DIR="${ROOT_DIR}/exps/smoke/frozenlake_eval" \
    bash examples/evaluate/frozenlake/sglang/eval_qwen25_vl_3b.sh \
    "envs.0.n_envs=${EVAL_ENVS}" \
    "envs.0.seed=[10000,$((10000 + EVAL_ENVS)),1]" \
    run.resume=force_rerun

  DUMP_DIR="${ROOT_DIR}/exps/smoke/sokoban_eval" \
    bash examples/evaluate/sokoban/sglang/eval_qwen25_vl_3b.sh \
    "envs.0.n_envs=${EVAL_ENVS}" \
    "envs.0.seed=[10000,$((10000 + EVAL_ENVS)),1]" \
    run.resume=force_rerun
fi

ENVIRONMENT="${SMOKE_ENVIRONMENT:-sokoban}" \
TOTAL_STEPS="${SMOKE_STEPS:-5}" \
TRAIN_BATCH_SIZE="${SMOKE_BATCH_SIZE:-2}" \
ROLLOUT_N="${SMOKE_ROLLOUT_N:-2}" \
VAL_BEFORE_TRAIN=False \
TEST_FREQ=-1 \
SAVE_FREQ=-1 \
EXPERIMENT_NAME="${SMOKE_ENVIRONMENT:-sokoban}_episode_grpo_smoke" \
bash scripts/run_no_concat_episode_grpo.sh
