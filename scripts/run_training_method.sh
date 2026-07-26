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
FORMAT_REWARD_THRESHOLD="${FORMAT_REWARD_THRESHOLD:-}"
SUCCESS_REWARD="${SUCCESS_REWARD:-1.0}"
PROCESS_REWARD_CAP="${PROCESS_REWARD_CAP:-0.2}"
N_GPUS="${N_GPUS:-1}"
LORA_RANK="${LORA_RANK:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.50}"
TRAINER_LOGGER="${TRAINER_LOGGER:-wandb}"
WANDB_MODE_VALUE="${WANDB_MODE:-offline}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
TEST_FREQ="${TEST_FREQ:-20}"
SAVE_FREQ="${SAVE_FREQ:-100}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"

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

case "${TRAINER_LOGGER}" in
  console|wandb) ;;
  *)
    echo "[ERROR] TRAINER_LOGGER must be console or wandb." >&2
    exit 1
    ;;
esac

if ! [[ "${SEED}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] SEED must be a non-negative integer." >&2
  exit 1
fi
if ! [[ "${ROLLOUT_N}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] ROLLOUT_N must be a positive integer." >&2
  exit 1
fi
if ! [[ "${N_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] N_GPUS must be a positive integer." >&2
  exit 1
fi
if ! [[ "${TRAIN_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] TRAIN_BATCH_SIZE must be a positive integer." >&2
  exit 1
fi
if ! [[ "${TOTAL_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] TOTAL_STEPS must be a positive integer." >&2
  exit 1
fi

if [[ "${METHOD}" == "no_concat_episode_grpo" ]]; then
  case "${REWARD_MODE}" in
    outcome|bounded_process|format_gate) ;;
    *)
      echo "[ERROR] REWARD_MODE must be outcome, bounded_process, or format_gate." >&2
      exit 1
      ;;
  esac
  case "${LOSS_WEIGHTING}" in
    token|turn|trajectory) ;;
    *)
      echo "[ERROR] LOSS_WEIGHTING must be token, turn, or trajectory." >&2
      exit 1
      ;;
  esac
fi

case "${ENVIRONMENT}" in
  frozenlake)
    TRAIN_FILE="${ROOT_DIR}/examples/train/frozenlake/train_frozenlake_vision.yaml"
    VAL_FILE="${ROOT_DIR}/examples/train/frozenlake/val_frozenlake_vision.yaml"
    MAX_PROMPT_LENGTH=1000
    CONCAT_RESPONSE_LENGTH=4000
    DEFAULT_FORMAT_REWARD_THRESHOLD=0.02
    ;;
  sokoban)
    TRAIN_FILE="${ROOT_DIR}/examples/train/sokoban/train_sokoban_vision.yaml"
    VAL_FILE="${ROOT_DIR}/examples/train/sokoban/val_sokoban_vision.yaml"
    MAX_PROMPT_LENGTH=1000
    CONCAT_RESPONSE_LENGTH=4000
    DEFAULT_FORMAT_REWARD_THRESHOLD=0.1
    ;;
  navigation)
    TRAIN_FILE="${ROOT_DIR}/examples/train/navigation/train_navigation.yaml"
    VAL_FILE="${ROOT_DIR}/examples/train/navigation/val_navigation.yaml"
    MAX_PROMPT_LENGTH=3000
    CONCAT_RESPONSE_LENGTH=10000
    DEFAULT_FORMAT_REWARD_THRESHOLD=0.01
    ;;
  *)
    echo "[ERROR] ENVIRONMENT must be frozenlake, sokoban, or navigation." >&2
    exit 1
    ;;
esac

if [[ -z "${FORMAT_REWARD_THRESHOLD}" ]]; then
  FORMAT_REWARD_THRESHOLD="${DEFAULT_FORMAT_REWARD_THRESHOLD}"
fi
if [[ "${CONCAT_MULTI_TURN}" == "True" ]]; then
  MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-${CONCAT_RESPONSE_LENGTH}}"
else
  MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
fi

MODEL_PATH_LOWER="$(printf '%s' "${MODEL_PATH}" | tr '[:upper:]' '[:lower:]')"
if [[ "${MODEL_PATH_LOWER}" == *"qwen3-vl"* || "${MODEL_PATH_LOWER}" == *"qwen3vl"* ]]; then
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
if [[ "${METHOD}" == "no_concat_episode_grpo" && "${N_GPUS}" != "1" ]]; then
  echo "[ERROR] no_concat_episode_grpo is currently restricted to N_GPUS=1; distributed policy-weight scaling is not yet validated." >&2
  exit 1
fi

EXPERIMENT_SUFFIX="${METHOD}_seed${SEED}"
if [[ "${METHOD}" == "no_concat_episode_grpo" ]]; then
  EXPERIMENT_SUFFIX="${METHOD}_${REWARD_MODE}_${LOSS_WEIGHTING}_seed${SEED}"
fi
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${ENVIRONMENT}_${EXPERIMENT_SUFFIX}}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${ROOT_DIR}/exps/${PROJECT_NAME}}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${EXPERIMENT_ROOT}/${EXPERIMENT_NAME}}"
CHECKPOINT_DIR="${EXPERIMENT_DIR}/checkpoints"
AGENT_LOOP_CONFIG="${ROOT_DIR}/vagen/configs/${AGENT_CONFIG}"
WANDB_DIR_VALUE="${WANDB_DIR:-${EXPERIMENT_DIR}/wandb}"
mkdir -p "${EXPERIMENT_DIR}" "${CHECKPOINT_DIR}" "${WANDB_DIR_VALUE}"

LOGGER_CONFIG="[console]"
if [[ "${TRAINER_LOGGER}" == "wandb" ]]; then
  LOGGER_CONFIG="[console,wandb]"
fi

SGLANG_GPU_ARGS=()
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_QUERY_ARGS=()
  VISIBLE_GPU_SELECTORS="${CUDA_VISIBLE_DEVICES:-}"
  if [[ -n "${VISIBLE_GPU_SELECTORS}" \
    && "${VISIBLE_GPU_SELECTORS}" != "all" \
    && "${VISIBLE_GPU_SELECTORS}" != "-1" \
    && "${VISIBLE_GPU_SELECTORS}" != "none" \
    && "${VISIBLE_GPU_SELECTORS}" != "void" ]]; then
    FIRST_GPU_SELECTOR="${VISIBLE_GPU_SELECTORS%%,*}"
    FIRST_GPU_SELECTOR="${FIRST_GPU_SELECTOR//[[:space:]]/}"
    GPU_QUERY_ARGS+=("--id=${FIRST_GPU_SELECTOR}")
  fi
  GPU_NAME="$(
    nvidia-smi "${GPU_QUERY_ARGS[@]}" \
      --query-gpu=name --format=csv,noheader \
      | head -n 1
  )"
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
export WANDB_MODE="${WANDB_MODE_VALUE}"
export WANDB_DIR="${WANDB_DIR_VALUE}"

GIT_COMMIT="$(git -C "${ROOT_DIR}" rev-parse HEAD)"
VERL_COMMIT="$(git -C "${ROOT_DIR}/verl" rev-parse HEAD)"
GIT_DIRTY=False
if [[ -n "$(git -C "${ROOT_DIR}" status --porcelain)" ]]; then
  GIT_DIRTY=True
fi
if [[ "${DRY_RUN}" != "1" && "${GIT_DIRTY}" == "True" && "${ALLOW_DIRTY}" != "1" ]]; then
  echo "[ERROR] Refusing a formal run from a dirty worktree; commit changes or set ALLOW_DIRTY=1." >&2
  exit 2
fi

TRAIN_COMMAND=(
  "${PYTHON_BIN}" -m vagen.main_ppo
  "--config-path=${ROOT_DIR}/vagen/configs"
  "--config-name=vagen_multiturn"
  "data.train_files=${TRAIN_FILE}"
  "data.val_files=${VAL_FILE}"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH}"
  "data.seed=${SEED}"
  "algorithm.adv_estimator=${ADV_ESTIMATOR}"
  "algorithm.use_kl_in_reward=False"
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
  "filter.enable=False"
  "trainer.concat_multi_turn=${CONCAT_MULTI_TURN}"
  "trainer.balance_batch=True"
  "trainer.critic_warmup=0"
  "trainer.logger=${LOGGER_CONFIG}"
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
if [[ "${METHOD}" == "no_concat_episode_grpo" ]]; then
  TRAIN_COMMAND+=(
    "algorithm.no_concat_episode_grpo.reward_mode=${REWARD_MODE}"
    "algorithm.no_concat_episode_grpo.loss_weighting=${LOSS_WEIGHTING}"
    "algorithm.no_concat_episode_grpo.incomplete_group_action=error"
    "algorithm.no_concat_episode_grpo.success_reward=${SUCCESS_REWARD}"
    "algorithm.no_concat_episode_grpo.process_reward_cap=${PROCESS_REWARD_CAP}"
    "algorithm.no_concat_episode_grpo.format_reward=${FORMAT_REWARD_THRESHOLD}"
  )
fi
if [[ "${METHOD}" == "no_concat_gae" ]]; then
  TRAIN_COMMAND+=(
    "critic.optim.lr=1e-5"
    "critic.model.path=${MODEL_PATH}"
    "critic.model.use_remove_padding=True"
    "critic.model.enable_gradient_checkpointing=True"
    "critic.ppo_micro_batch_size_per_gpu=1"
    "critic.model.fsdp_config.param_offload=True"
    "critic.model.fsdp_config.optimizer_offload=True"
  )
fi
if (( ${#SGLANG_GPU_ARGS[@]} > 0 )); then
  TRAIN_COMMAND+=("${SGLANG_GPU_ARGS[@]}")
fi

MANIFEST_COMMIT="${GIT_COMMIT}" \
MANIFEST_VERL_COMMIT="${VERL_COMMIT}" \
MANIFEST_GIT_DIRTY="${GIT_DIRTY}" \
MANIFEST_ENVIRONMENT="${ENVIRONMENT}" \
MANIFEST_METHOD="${METHOD}" \
MANIFEST_ADV_ESTIMATOR="${ADV_ESTIMATOR}" \
MANIFEST_MODEL="${MODEL_PATH}" \
MANIFEST_SEED="${SEED}" \
MANIFEST_ROLLOUT_N="${ROLLOUT_N}" \
MANIFEST_TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
MANIFEST_TOTAL_STEPS="${TOTAL_STEPS}" \
MANIFEST_REWARD_MODE="${REWARD_MODE}" \
MANIFEST_LOSS_WEIGHTING="${LOSS_WEIGHTING}" \
MANIFEST_SUCCESS_REWARD="${SUCCESS_REWARD}" \
MANIFEST_PROCESS_REWARD_CAP="${PROCESS_REWARD_CAP}" \
MANIFEST_FORMAT_REWARD_THRESHOLD="${FORMAT_REWARD_THRESHOLD}" \
MANIFEST_CRITIC_ENABLED="${CRITIC_ENABLED}" \
MANIFEST_CONCAT_MULTI_TURN="${CONCAT_MULTI_TURN}" \
MANIFEST_TRAIN_FILE="${TRAIN_FILE}" \
MANIFEST_VAL_FILE="${VAL_FILE}" \
MANIFEST_LOGGER="${TRAINER_LOGGER}" \
MANIFEST_WANDB_MODE="${WANDB_MODE_VALUE}" \
MANIFEST_WANDB_DIR="${WANDB_DIR_VALUE}" \
MANIFEST_MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH}" \
MANIFEST_MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}" \
MANIFEST_LORA_RANK="${LORA_RANK}" \
MANIFEST_N_GPUS="${N_GPUS}" \
MANIFEST_GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
MANIFEST_VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN}" \
MANIFEST_TEST_FREQ="${TEST_FREQ}" \
MANIFEST_SAVE_FREQ="${SAVE_FREQ}" \
MANIFEST_EXPERIMENT_DIR="${EXPERIMENT_DIR}" \
PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/verl${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" - "${TRAIN_COMMAND[@]}" <<'PY'
import os
import sys

from vagen.utils.run_manifest import write_compatible_manifest

def as_bool(name: str) -> bool:
    return os.environ[name].lower() == "true"

manifest = {
    "commit": os.environ["MANIFEST_COMMIT"],
    "verl_commit": os.environ["MANIFEST_VERL_COMMIT"],
    "git_dirty": as_bool("MANIFEST_GIT_DIRTY"),
    "environment": os.environ["MANIFEST_ENVIRONMENT"],
    "method": os.environ["MANIFEST_METHOD"],
    "advantage_estimator": os.environ["MANIFEST_ADV_ESTIMATOR"],
    "model": os.environ["MANIFEST_MODEL"],
    "seed": int(os.environ["MANIFEST_SEED"]),
    "seed_scope": ["python_hash", "training_dataloader_order"],
    "bitwise_cuda_reproducible": False,
    "rollout_n": int(os.environ["MANIFEST_ROLLOUT_N"]),
    "train_batch_size": int(os.environ["MANIFEST_TRAIN_BATCH_SIZE"]),
    "total_steps": int(os.environ["MANIFEST_TOTAL_STEPS"]),
    "reward_mode": (
        os.environ["MANIFEST_REWARD_MODE"]
        if os.environ["MANIFEST_METHOD"] == "no_concat_episode_grpo"
        else None
    ),
    "loss_weighting": (
        os.environ["MANIFEST_LOSS_WEIGHTING"]
        if os.environ["MANIFEST_METHOD"] == "no_concat_episode_grpo"
        else None
    ),
    "success_reward": (
        float(os.environ["MANIFEST_SUCCESS_REWARD"])
        if os.environ["MANIFEST_METHOD"] == "no_concat_episode_grpo"
        else None
    ),
    "process_reward_cap": (
        float(os.environ["MANIFEST_PROCESS_REWARD_CAP"])
        if os.environ["MANIFEST_METHOD"] == "no_concat_episode_grpo"
        else None
    ),
    "format_reward_threshold": (
        float(os.environ["MANIFEST_FORMAT_REWARD_THRESHOLD"])
        if os.environ["MANIFEST_METHOD"] == "no_concat_episode_grpo"
        else None
    ),
    "critic_enabled": as_bool("MANIFEST_CRITIC_ENABLED"),
    "concat_multi_turn": as_bool("MANIFEST_CONCAT_MULTI_TURN"),
    "filter_enabled": False,
    "parity_gate_enabled": True,
    "parity_thresholds": {
        "clip_low": 0.8,
        "clip_high": 1.2,
        "max_p95_ratio_deviation": 0.1,
        "max_p99_ratio_deviation": 0.2,
        "max_mean_abs_logprob_delta": 0.05,
        "max_clip_fraction": 0.01,
    },
    "train_file": os.environ["MANIFEST_TRAIN_FILE"],
    "validation_file": os.environ["MANIFEST_VAL_FILE"],
    "logger": os.environ["MANIFEST_LOGGER"],
    "wandb_mode": os.environ["MANIFEST_WANDB_MODE"],
    "wandb_dir": os.environ["MANIFEST_WANDB_DIR"],
    "max_prompt_length": int(os.environ["MANIFEST_MAX_PROMPT_LENGTH"]),
    "max_response_length": int(os.environ["MANIFEST_MAX_RESPONSE_LENGTH"]),
    "lora_rank": int(os.environ["MANIFEST_LORA_RANK"]),
    "n_gpus": int(os.environ["MANIFEST_N_GPUS"]),
    "gpu_memory_utilization": float(
        os.environ["MANIFEST_GPU_MEMORY_UTILIZATION"]
    ),
    "val_before_train": as_bool("MANIFEST_VAL_BEFORE_TRAIN"),
    "test_freq": int(os.environ["MANIFEST_TEST_FREQ"]),
    "save_freq": int(os.environ["MANIFEST_SAVE_FREQ"]),
    "resume_mode": "auto",
    "command": sys.argv[1:],
}
write_compatible_manifest(
    os.path.join(os.environ["MANIFEST_EXPERIMENT_DIR"], "manifest.json"),
    manifest,
    require_existing_match=True,
)
PY

{
  printf '#!/usr/bin/env bash\n'
  printf 'set -euo pipefail\n'
  printf 'PYTHONHASHSEED=%q WANDB_MODE=%q WANDB_DIR=%q ' \
    "${SEED}" "${WANDB_MODE_VALUE}" "${WANDB_DIR_VALUE}"
  if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${CUDA_VISIBLE_DEVICES}"
  fi
  printf '%q ' "${TRAIN_COMMAND[@]}"
  printf '"$@"\n'
} > "${EXPERIMENT_DIR}/train_command.sh"
chmod +x "${EXPERIMENT_DIR}/train_command.sh"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[DRY RUN] '
  printf '%q ' "${TRAIN_COMMAND[@]}"
  printf '\n'
  exit 0
fi

RUN_RESUME_STATE="$(
  RUN_STATE_DIR="${EXPERIMENT_DIR}" \
  PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/verl${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" - <<'PY'
import os

from vagen.utils.run_manifest import classify_run_for_resume

print(classify_run_for_resume(os.environ["RUN_STATE_DIR"]))
PY
)"
case "${RUN_RESUME_STATE}" in
  complete)
    echo "[SKIP] Training run is already complete: ${EXPERIMENT_DIR}"
    exit 0
    ;;
  failed-parity)
    echo "[ERROR] This run has a failed rollout/train parity attempt; use a new experiment directory." >&2
    exit 2
    ;;
  tainted-gpu-metrics)
    echo "[ERROR] This run has incomplete GPU sampling evidence; use a new experiment directory." >&2
    exit 2
    ;;
  resumable) ;;
  *)
    echo "[ERROR] Unknown run resume state: ${RUN_RESUME_STATE}" >&2
    exit 2
    ;;
esac

cd "${ROOT_DIR}"
"${TRAIN_COMMAND[@]}" --cfg job --resolve > "${EXPERIMENT_DIR}/resolved_config.yaml"
printf '\n===== session %s =====\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  | tee -a "${EXPERIMENT_DIR}/train.log"
set +e
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" \
  "${ROOT_DIR}/scripts/run_with_gpu_metrics.py" \
  --output-dir "${EXPERIMENT_DIR}/gpu_metrics" \
  --expected-device-count "${N_GPUS}" \
  -- "${TRAIN_COMMAND[@]}" \
  2>&1 | tee -a "${EXPERIMENT_DIR}/train.log"
status=${PIPESTATUS[0]}
set -e
exit "${status}"
