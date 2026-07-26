#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METHOD="${METHOD:-no_concat_episode_grpo}"
ENVIRONMENT="${ENVIRONMENT:-sokoban}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
PROJECT_NAME="${PROJECT_NAME:-vlm_agent_rl}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
TOTAL_STEPS="${TOTAL_STEPS:-401}"
SEED="${SEED:-0}"
REWARD_MODE="${REWARD_MODE:-outcome}"
LOSS_WEIGHTING="${LOSS_WEIGHTING:-trajectory}"
N_GPUS="${N_GPUS:-1}"
LORA_RANK="${LORA_RANK:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.50}"
TRAINER_LOGGER="${TRAINER_LOGGER:-console}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
TEST_FREQ="${TEST_FREQ:-20}"
SAVE_FREQ="${SAVE_FREQ:-100}"
DRY_RUN="${DRY_RUN:-0}"

case "${METHOD}" in
  concat_grpo)
    ADV_ESTIMATOR=grpo
    CONCAT_MULTI_TURN=True
    CRITIC_ENABLED=False
    AGENT_CONFIG=agent.yaml
    ROLLOUT_N="${ROLLOUT_N:-4}"
    ;;
  no_concat_gae)
    ADV_ESTIMATOR=no_concat_gae
    CONCAT_MULTI_TURN=False
    CRITIC_ENABLED=True
    AGENT_CONFIG=agent_no_concat.yaml
    ROLLOUT_N="${ROLLOUT_N:-1}"
    ;;
  no_concat_episode_grpo)
    ADV_ESTIMATOR=no_concat_episode_grpo
    CONCAT_MULTI_TURN=False
    CRITIC_ENABLED=False
    AGENT_CONFIG=agent_no_concat.yaml
    ROLLOUT_N="${ROLLOUT_N:-4}"
    ;;
  *)
    echo "[ERROR] METHOD must be concat_grpo, no_concat_gae, or no_concat_episode_grpo." >&2
    exit 1
    ;;
esac

case "${ENVIRONMENT}" in
  frozenlake)
    TRAIN_FILE="${ROOT_DIR}/examples/train/frozenlake/train_frozenlake_vision.yaml"
    VAL_FILE="${ROOT_DIR}/examples/train/frozenlake/val_frozenlake_vision.yaml"
    MAX_PROMPT_LENGTH=1000
    CONCAT_RESPONSE_LENGTH=4000
    ;;
  sokoban)
    TRAIN_FILE="${ROOT_DIR}/examples/train/sokoban/train_sokoban_vision.yaml"
    VAL_FILE="${ROOT_DIR}/examples/train/sokoban/val_sokoban_vision.yaml"
    MAX_PROMPT_LENGTH=1000
    CONCAT_RESPONSE_LENGTH=4000
    ;;
  navigation)
    TRAIN_FILE="${ROOT_DIR}/examples/train/navigation/train_navigation.yaml"
    VAL_FILE="${ROOT_DIR}/examples/train/navigation/val_navigation.yaml"
    MAX_PROMPT_LENGTH=3000
    CONCAT_RESPONSE_LENGTH=10000
    ;;
  *)
    echo "[ERROR] ENVIRONMENT must be frozenlake, sokoban, or navigation." >&2
    exit 1
    ;;
esac

if [[ "${CONCAT_MULTI_TURN}" == "True" ]]; then
  MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-${CONCAT_RESPONSE_LENGTH}}"
else
  MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
fi

if [[ "${MODEL_PATH}" == *"Qwen3-VL"* || "${MODEL_PATH}" == *"Qwen3VL"* ]]; then
  echo "[ERROR] Qwen3-VL is blocked from formal runs until its no-concat processor/M-RoPE path passes parity." >&2
  exit 2
fi
if [[ "${METHOD}" == "no_concat_gae" && "${ROLLOUT_N}" != "1" ]]; then
  echo "[ERROR] no_concat_gae requires ROLLOUT_N=1 in the core comparison." >&2
  exit 1
fi
if [[ "${METHOD}" != "no_concat_gae" ]] && (( ROLLOUT_N < 2 )); then
  echo "[ERROR] group-relative methods require ROLLOUT_N >= 2." >&2
  exit 1
fi

EXPERIMENT_SUFFIX="${METHOD}_seed${SEED}"
if [[ "${METHOD}" == "no_concat_episode_grpo" ]]; then
  EXPERIMENT_SUFFIX="${METHOD}_${REWARD_MODE}_${LOSS_WEIGHTING}_seed${SEED}"
fi
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${ENVIRONMENT}_${EXPERIMENT_SUFFIX}}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${ROOT_DIR}/exps/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
CHECKPOINT_DIR="${EXPERIMENT_DIR}/checkpoints"
AGENT_LOOP_CONFIG="${ROOT_DIR}/vagen/configs/${AGENT_CONFIG}"
mkdir -p "${EXPERIMENT_DIR}" "${CHECKPOINT_DIR}"

LOGGER_CONFIG="[console]"
if [[ "${TRAINER_LOGGER}" == "wandb" ]]; then
  LOGGER_CONFIG="[console,wandb]"
fi

SGLANG_GPU_ARGS=()
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
  if [[ "${GPU_NAME}" == *"B200"* || "${GPU_NAME}" == *"RTX 6000 Pro"* ]]; then
    SGLANG_GPU_ARGS+=(
      "+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer"
      "+actor_rollout_ref.rollout.engine_kwargs.sglang.mm_attention_backend=triton_attn"
    )
  fi
elif [[ "${DRY_RUN}" != "1" ]]; then
  echo "[ERROR] Qwen2.5-VL training requires an NVIDIA GPU." >&2
  exit 2
fi

if [[ "${ENVIRONMENT}" == "navigation" && "${DRY_RUN}" != "1" ]]; then
  NAVIGATION_SERVER_URL="${NAVIGATION_SERVER_URL:-http://127.0.0.1:8000}"
  if ! curl --silent --show-error --fail "${NAVIGATION_SERVER_URL}/health" >/dev/null; then
    echo "[ERROR] Navigation server is not ready at ${NAVIGATION_SERVER_URL}." >&2
    exit 2
  fi
fi

export PYTHONHASHSEED="${SEED}"
export WANDB_MODE="${WANDB_MODE:-offline}"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

manifest = {
    "commit": "$(git -C "${ROOT_DIR}" rev-parse HEAD)",
    "verl_commit": "$(git -C "${ROOT_DIR}/verl" rev-parse HEAD)",
    "environment": "${ENVIRONMENT}",
    "method": "${METHOD}",
    "advantage_estimator": "${ADV_ESTIMATOR}",
    "model": "${MODEL_PATH}",
    "seed": ${SEED},
    "rollout_n": ${ROLLOUT_N},
    "train_batch_size": ${TRAIN_BATCH_SIZE},
    "total_steps": ${TOTAL_STEPS},
    "reward_mode": "${REWARD_MODE}",
    "loss_weighting": "${LOSS_WEIGHTING}",
    "critic_enabled": "${CRITIC_ENABLED}" == "True",
    "concat_multi_turn": "${CONCAT_MULTI_TURN}" == "True",
    "filter_enabled": False,
    "parity_gate_enabled": True,
    "train_file": "${TRAIN_FILE}",
    "validation_file": "${VAL_FILE}",
}
Path("${EXPERIMENT_DIR}/manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
PY

TRAIN_COMMAND=(
  "${PYTHON_BIN}" -m vagen.main_ppo
  "--config-path=${ROOT_DIR}/vagen/configs"
  "--config-name=vagen_multiturn"
  "data.train_files=${TRAIN_FILE}"
  "data.val_files=${VAL_FILE}"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH}"
  "algorithm.adv_estimator=${ADV_ESTIMATOR}"
  "algorithm.use_kl_in_reward=False"
  "algorithm.no_concat_episode_grpo.reward_mode=${REWARD_MODE}"
  "algorithm.no_concat_episode_grpo.loss_weighting=${LOSS_WEIGHTING}"
  "algorithm.no_concat_episode_grpo.incomplete_group_action=error"
  "algorithm.rollout_train_parity.enabled=True"
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "actor_rollout_ref.model.use_remove_padding=True"
  "actor_rollout_ref.model.use_fused_kernels=True"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  "actor_rollout_ref.model.lora_rank=${LORA_RANK}"
  "actor_rollout_ref.model.lora_alpha=${LORA_RANK}"
  "actor_rollout_ref.model.target_modules=all-linear"
  "actor_rollout_ref.actor.optim.lr=1e-6"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.actor.use_kl_loss=False"
  "actor_rollout_ref.actor.entropy_coeff=0.0"
  "actor_rollout_ref.actor.fsdp_config.param_offload=True"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True"
  "actor_rollout_ref.actor.checkpoint.save_contents=[model,hf_model,optimizer,extra]"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.ref.fsdp_config.param_offload=True"
  "actor_rollout_ref.rollout.name=sglang"
  "actor_rollout_ref.rollout.mode=async"
  "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
  "actor_rollout_ref.rollout.calculate_log_probs=True"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.rollout.max_num_batched_tokens=10000"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
  "actor_rollout_ref.rollout.enforce_eager=True"
  "actor_rollout_ref.rollout.free_cache_engine=True"
  "actor_rollout_ref.rollout.enable_chunked_prefill=True"
  "actor_rollout_ref.rollout.multi_turn.enable=True"
  "actor_rollout_ref.rollout.agent.agent_loop_config_path=${AGENT_LOOP_CONFIG}"
  "actor_rollout_ref.rollout.disable_log_stats=False"
  "critic.enable=${CRITIC_ENABLED}"
  "critic.optim.lr=1e-5"
  "critic.model.path=${MODEL_PATH}"
  "critic.model.use_remove_padding=True"
  "critic.model.enable_gradient_checkpointing=True"
  "critic.ppo_micro_batch_size_per_gpu=1"
  "critic.model.fsdp_config.param_offload=True"
  "critic.model.fsdp_config.optimizer_offload=True"
  "filter.enable=False"
  "trainer.concat_multi_turn=${CONCAT_MULTI_TURN}"
  "trainer.balance_batch=True"
  "trainer.critic_warmup=0"
  "trainer.logger=${LOGGER_CONFIG}"
  "trainer.seed=${SEED}"
  "trainer.val_before_train=${VAL_BEFORE_TRAIN}"
  "trainer.n_gpus_per_node=${N_GPUS}"
  "trainer.nnodes=1"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.test_freq=${TEST_FREQ}"
  "trainer.project_name=${PROJECT_NAME}"
  "trainer.experiment_name=${EXPERIMENT_NAME}"
  "trainer.default_local_dir=${CHECKPOINT_DIR}"
  "trainer.validation_data_dir=${EXPERIMENT_DIR}/validation"
  "trainer.rollout_data_dir=${EXPERIMENT_DIR}/rollouts"
  "trainer.parity_report_path=${EXPERIMENT_DIR}/parity.json"
  "trainer.log_val_generations=16"
  "trainer.total_training_steps=${TOTAL_STEPS}"
)
if (( ${#SGLANG_GPU_ARGS[@]} > 0 )); then
  TRAIN_COMMAND+=("${SGLANG_GPU_ARGS[@]}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[DRY RUN] '
  printf '%q ' "${TRAIN_COMMAND[@]}"
  printf '\n'
  exit 0
fi

cd "${ROOT_DIR}"
set +e
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" \
  "${ROOT_DIR}/scripts/run_with_gpu_metrics.py" \
  --output-dir "${EXPERIMENT_DIR}/gpu_metrics" \
  -- "${TRAIN_COMMAND[@]}" \
  2>&1 | tee "${EXPERIMENT_DIR}/train.log"
status=${PIPESTATUS[0]}
set -e
exit "${status}"
