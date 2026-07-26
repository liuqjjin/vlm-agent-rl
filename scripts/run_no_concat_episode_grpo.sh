#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export METHOD=no_concat_episode_grpo
exec bash "${ROOT_DIR}/scripts/run_training_method.sh" "$@"
