#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVIRONMENT="${ENVIRONMENT:-sokoban}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
PROJECT_NAME="${PROJECT_NAME:-vlm_agent_rl}"
ROLLOUT_N="${ROLLOUT_N:-4}"
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

case "${ENVIRONMENT}" in
  frozenlake)
    TRAIN_FILE="${ROOT_DIR}/examples/train/frozenlake/train_frozenlake_vision.yaml"
    VAL_FILE="${ROOT_DIR}/examples/train/frozenlake/val_frozenlake_vision.yaml"
    ;;
  sokoban)
    TRAIN_FILE="${ROOT_DIR}/examples/train/sokoban/train_sokoban_vision.yaml"
    VAL_FILE="${ROOT_DIR}/examples/train/sokoban/val_sokoban_vision.yaml"
    ;;
  *)
    echo "[ERROR] ENVIRONMENT must be 'frozenlake' or 'sokoban', got '${ENVIRONMENT}'." >&2
    exit 1
    ;;
esac

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] no-concat Qwen2.5-VL training requires an NVIDIA GPU." >&2
  exit 2
fi
if (( ROLLOUT_N < 2 )); then
  echo "[ERROR] ROLLOUT_N must be at least 2 for group-relative advantages." >&2
  exit 1
fi

EXPERIMENT_NAME="${EXPERIMENT_NAME:-${ENVIRONMENT}_episode_grpo_${REWARD_MODE}_${LOSS_WEIGHTING}_seed${SEED}}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${ROOT_DIR}/exps/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
CHECKPOINT_DIR="${EXPERIMENT_DIR}/checkpoints"
AGENT_LOOP_CONFIG="${ROOT_DIR}/vagen/configs/agent_no_concat.yaml"
mkdir -p "${EXPERIMENT_DIR}" "${CHECKPOINT_DIR}"

LOGGER_CONFIG="[console]"
if [[ "${TRAINER_LOGGER}" == "wandb" ]]; then
  LOGGER_CONFIG="[console,wandb]"
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
SGLANG_GPU_ARGS=()
if [[ "${GPU_NAME}" == *"B200"* || "${GPU_NAME}" == *"RTX 6000 Pro"* ]]; then
  SGLANG_GPU_ARGS+=(
    "+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer"
    "+actor_rollout_ref.rollout.engine_kwargs.sglang.mm_attention_backend=triton_attn"
  )
fi

export PYTHONHASHSEED="${SEED}"
export WANDB_MODE="${WANDB_MODE:-offline}"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
manifest = {
    "commit": "$(git -C "${ROOT_DIR}" rev-parse HEAD)",
    "environment": "${ENVIRONMENT}",
    "model": "${MODEL_PATH}",
    "seed": ${SEED},
    "rollout_n": ${ROLLOUT_N},
    "train_batch_size": ${TRAIN_BATCH_SIZE},
    "total_steps": ${TOTAL_STEPS},
    "reward_mode": "${REWARD_MODE}",
    "loss_weighting": "${LOSS_WEIGHTING}",
    "critic_enabled": False,
    "filter_enabled": False,
    "parity_gate_enabled": True,
}
Path("${EXPERIMENT_DIR}/manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

cd "${ROOT_DIR}"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m vagen.main_ppo \
  --config-path="${ROOT_DIR}/vagen/configs" \
  --config-name=vagen_multiturn \
  "data.train_files=${TRAIN_FILE}" \
  "data.val_files=${VAL_FILE}" \
  "data.train_batch_size=${TRAIN_BATCH_SIZE}" \
  "data.max_prompt_length=1000" \
  "data.max_response_length=512" \
  "algorithm.adv_estimator=no_concat_episode_grpo" \
  "algorithm.use_kl_in_reward=False" \
  "algorithm.no_concat_episode_grpo.reward_mode=${REWARD_MODE}" \
  "algorithm.no_concat_episode_grpo.loss_weighting=${LOSS_WEIGHTING}" \
  "algorithm.no_concat_episode_grpo.incomplete_group_action=error" \
  "algorithm.rollout_train_parity.enabled=True" \
  "actor_rollout_ref.model.path=${MODEL_PATH}" \
  "actor_rollout_ref.model.use_remove_padding=True" \
  "actor_rollout_ref.model.use_fused_kernels=True" \
  "actor_rollout_ref.model.enable_gradient_checkpointing=True" \
  "actor_rollout_ref.model.lora_rank=${LORA_RANK}" \
  "actor_rollout_ref.model.lora_alpha=${LORA_RANK}" \
  "actor_rollout_ref.model.target_modules=all-linear" \
  "actor_rollout_ref.actor.optim.lr=1e-6" \
  "actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE}" \
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1" \
  "actor_rollout_ref.actor.use_kl_loss=False" \
  "actor_rollout_ref.actor.entropy_coeff=0.0" \
  "actor_rollout_ref.actor.fsdp_config.param_offload=True" \
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True" \
  "actor_rollout_ref.actor.checkpoint.save_contents=[model,hf_model,optimizer,extra]" \
  "actor_rollout_ref.rollout.name=sglang" \
  "actor_rollout_ref.rollout.mode=async" \
  "actor_rollout_ref.rollout.n=${ROLLOUT_N}" \
  "actor_rollout_ref.rollout.calculate_log_probs=True" \
  "actor_rollout_ref.rollout.tensor_model_parallel_size=1" \
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1" \
  "actor_rollout_ref.rollout.max_num_batched_tokens=10000" \
  "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}" \
  "actor_rollout_ref.rollout.enforce_eager=True" \
  "actor_rollout_ref.rollout.free_cache_engine=True" \
  "actor_rollout_ref.rollout.enable_chunked_prefill=True" \
  "actor_rollout_ref.rollout.multi_turn.enable=True" \
  "actor_rollout_ref.rollout.agent.agent_loop_config_path=${AGENT_LOOP_CONFIG}" \
  "actor_rollout_ref.rollout.disable_log_stats=False" \
  "critic.enable=False" \
  "filter.enable=False" \
  "trainer.concat_multi_turn=False" \
  "trainer.balance_batch=True" \
  "trainer.logger=${LOGGER_CONFIG}" \
  "trainer.seed=${SEED}" \
  "trainer.val_before_train=${VAL_BEFORE_TRAIN}" \
  "trainer.n_gpus_per_node=${N_GPUS}" \
  "trainer.nnodes=1" \
  "trainer.save_freq=${SAVE_FREQ}" \
  "trainer.test_freq=${TEST_FREQ}" \
  "trainer.project_name=${PROJECT_NAME}" \
  "trainer.experiment_name=${EXPERIMENT_NAME}" \
  "trainer.default_local_dir=${CHECKPOINT_DIR}" \
  "trainer.validation_data_dir=${EXPERIMENT_DIR}/validation" \
  "trainer.rollout_data_dir=${EXPERIMENT_DIR}/rollouts" \
  "trainer.log_val_generations=16" \
  "trainer.total_training_steps=${TOTAL_STEPS}" \
  "${SGLANG_GPU_ARGS[@]}" \
  2>&1 | tee "${EXPERIMENT_DIR}/train.log"
