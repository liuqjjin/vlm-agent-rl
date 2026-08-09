# GPU Execution Minimal Checklist

**Purpose:** Step-by-step commands for external GPU execution. No theory, no background—just what to run and what to expect.

---

## Prerequisites

✅ Linux machine with NVIDIA GPU (48GB+ VRAM recommended, 80GB if smoke OOMs)
✅ SSH access configured
✅ At least 600 GiB free disk for the winner-only route; about 1.5 TiB for the full three-method route
✅ Internet access for model download

**Optional but recommended:**
- Hugging Face token (if public download throttled): `export HF_TOKEN=<your_token>`
- W&B API key (for online logging): `export WANDB_API_KEY=<your_key>`

---

## Phase 0: Bootstrap (First Time Only)

**Time:** ~30 minutes (depends on download speed)

```bash
# SSH into the GPU instance
ssh -p <port> root@<host>

# Clone repository
git clone --recurse-submodules https://github.com/liuqjjin/vlm-agent-rl.git
cd vlm-agent-rl

# Bootstrap environment + download model + preload Navigation assets
# Winner-only route (the bootstrap enforces 600 GiB by default)
DOWNLOAD_MODEL=1 PRELOAD_NAVIGATION=1 bash scripts/autodl_bootstrap.sh

# Full three-method route instead: enforce at least 1.5 TiB before installing
MIN_FREE_GB=1536 DOWNLOAD_MODEL=1 PRELOAD_NAVIGATION=1 \
  bash scripts/autodl_bootstrap.sh
```

**Expected output:**
- Conda environment `vagen` created
- PyTorch 2.8.0 + CUDA 12.8 installed
- SGLang 0.5.2, vLLM 0.11.0, FlashAttention 2.8.1 installed
- Qwen2.5-VL-3B model downloaded to `${ROOT_DIR}/.cache/huggingface/hub/`
- AI2-THOR assets preloaded
- Vulkan renderer tested

**Success check:**
```bash
conda run -n vagen python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
# Should print: CUDA: True, GPUs: 1 (or more)

nvidia-smi
# Should show your GPU(s)
```

**If bootstrap fails:**
- Check `autodl_bootstrap.log` for errors
- Verify disk space: `df -h` (need 600 GiB+ winner-only or 1.5 TiB full comparison)
- Verify CUDA: `nvcc --version`
- Try manual conda environment: `conda create -n vagen python=3.12`

---

## Phase 1: Smoke Test (Required Before Screening)

**Purpose:** Verify GPU inference, training, parity, and memory before committing budget. Use the measured GPU summary below; do not budget from an unverified time estimate.

```bash
cd vlm-agent-rl
conda activate vagen  # or: conda run -n vagen bash

# Validate matrix paths, task identities, and disjoint train/validation/test splits
bash scripts/run_experiment_matrix.sh validate-matrix

# Run 5-step smoke with FrozenLake eval + Sokoban eval + 3 core method 1-step updates
bash scripts/run_experiment_matrix.sh smoke
```

**Expected output:**
```
[✓] FrozenLake Qwen2.5-VL eval: 4 episodes
[✓] Sokoban Qwen2.5-VL eval: 4 episodes
[✓] concat_grpo 1-step update: parity PASSED
[✓] no_concat_gae 1-step update: parity PASSED (includes critic)
[✓] no_concat_episode_grpo 5 updates: parity PASSED

GPU metrics written to exps/vlm_agent_rl/*/gpu_metrics/gpu_summary.json
Peak VRAM and GPU·h are recorded from the rented card; no value is assumed.
```

**Failure modes:**

| Error | Cause | Fix |
|---|---|---|
| `OOM: out of memory` during `no_concat_gae` | Critic + LoRA + offload > 48GB | Need 80GB card, or reduce batch size |
| `Parity gate failed: P95 deviation > 0.1` | Processor mismatch | Should not happen with Qwen2.5-VL; check logs |
| `SGLang server failed to start` | Port conflict or GPU busy | Check `exps/system/sglang_server.log` |
| `Navigation server timeout` | Unity/Vulkan issue | Check `exps/system/navigation_server.log` |

**If smoke OOMs on 48GB:**
- Rent 80GB A100/A800 or 96GB 6000 Pro
- Re-run smoke on larger card before proceeding

**Success check:**
```bash
# Check parity passed for all methods
grep -r "gate_passed.*true" exps/vlm_agent_rl/*/parity.json
# Should show 3 files (concat_grpo, no_concat_gae, no_concat_episode_grpo)

# Check GPU summary exists
find exps/vlm_agent_rl -name gpu_summary.json | wc -l
# Should be >= 3
```

---

## Phase 2: Base Evaluation (Zero-Shot Baseline)

**Purpose:** Measure Qwen2.5-VL zero-shot performance on Sokoban and Navigation

```bash
bash scripts/run_experiment_matrix.sh base-eval
```

**Expected output:**
```
Sokoban: 128 episodes (board-disjoint seeds enumerated in experiments/sokoban_board_split.json, 20003-20645)
Navigation: 30 episodes (seeds 30-59; includes Unity rendering)

Results written to:
 exps/eval/Qwen_Qwen2.5-VL-3B-Instruct/sokoban_none/
 exps/eval/Qwen_Qwen2.5-VL-3B-Instruct/navigation_none/
```

**Success check:**
```bash
python -m vagen.analysis.analyze_rollouts \
  --eval-dump exps/eval/Qwen_Qwen2.5-VL-3B-Instruct/sokoban_none \
  --eval-dump exps/eval/Qwen_Qwen2.5-VL-3B-Instruct/navigation_none \
  --output-dir results/gpu/base_eval

# Should print success rates, mean turns, template concentration
```

## Phase 3: Core Screening (50 Steps, 3 Methods)

**Purpose:** Fast comparison of concat GRPO vs fixed no-concat GAE vs no-concat episode GRPO

```bash
# Sokoban + Navigation, 50 steps each, seed 0
bash scripts/run_experiment_matrix.sh core-screening
```

**Expected output:**
```
6 runs total (3 methods × 2 environments):
- sokoban_core_screening_concat_grpo_seed0
- sokoban_core_screening_no_concat_gae_seed0
- sokoban_core_screening_no_concat_episode_grpo_outcome_trajectory_seed0
- navigation_core_screening_concat_grpo_seed0
- navigation_core_screening_no_concat_gae_seed0
- navigation_core_screening_no_concat_episode_grpo_outcome_trajectory_seed0
```

**Success check:**
```bash
# Check all runs have parity passed and GPU metrics
find exps/vlm_agent_rl/*core_screening* -name parity.json -exec grep -l '"gate_passed": true' {} \;
# Should list 6 files

# Analyze all screening runs
for run in exps/vlm_agent_rl/*core_screening*; do
  python -m vagen.analysis.analyze_rollouts --run "$run" --output-dir "results/gpu/$(basename $run)"
done
```

**Decision point:**
- If any method has non-finite loss or parity failure → debug before confirmatory
- If memory issues persist → adjust `TRAIN_BATCH_SIZE` or use 80GB card
- Select best method for confirmatory based on: (1) success rate, (2) mean turns on success, (3) stability

---

## Phase 4: Episode Screening (3×3 Ablation, Sokoban Only)

**Purpose:** Test all 9 combinations of reward mode × policy objective for episode GRPO

```bash
# All 9 configs on Sokoban, 50 steps, seed 0
bash scripts/run_experiment_matrix.sh episode-screening
```

**Expected output:**
```
9 runs:
- outcome × {token, turn, trajectory}
- bounded_process × {token, turn, trajectory}
- format_gate × {token, turn, trajectory}
```

**Selection criteria (in order):**
1. No parity failures or non-finite loss
2. Highest visual success rate
3. Lowest mean turns on success (efficiency)
4. Stability across validation points
5. Lowest GPU-hours if tied

**Best configuration saved to:**
```bash
# Example: outcome reward + trajectory weighting wins
export SELECTED_REWARD=outcome
export SELECTED_WEIGHTING=trajectory
```

---

## Phase 5: Confirmatory Runs (401 Steps, 3 Seeds)

**Purpose:** Final comparison with statistical power (multiple seeds)

**⚠️ Only run on screening winners! Do not run all 3 core methods unless you have budget.**

```bash
# Option A: Only best episode GRPO config (conservative)
SELECTED_METHODS=no_concat_episode_grpo \
CONFIRMATORY_SEEDS=0,1,2 \
REWARD_MODE=outcome \
LOSS_WEIGHTING=trajectory \
  bash scripts/run_experiment_matrix.sh confirmatory

# Option B: Full comparison (run only after pricing it from measured screening GPU·h)
SELECTED_METHODS=concat_grpo,no_concat_gae,no_concat_episode_grpo \
CONFIRMATORY_SEEDS=0,1,2 \
  bash scripts/run_experiment_matrix.sh confirmatory
```

**Expected output:**
```
Per method per environment: 3 training seeds × 401 steps
Checkpoints saved every 100 steps to exps/vlm_agent_rl/*/checkpoints/
The wrapper retains five actor checkpoints (steps 100, 200, 300, 400, and 401)
and one critic checkpoint. Do not reduce `MAX_ACTOR_CKPTS_TO_KEEP` below 5.
```

**Monitor progress:**
```bash
# Check latest validation JSONL for each run
tail -f exps/vlm_agent_rl/sokoban_confirmatory_*/validation/*.jsonl

# Live GPU usage
watch -n 1 nvidia-smi
```

---

## Phase 6: Select, Export, and Run the Independent Final Test

**Purpose:** Select checkpoints using validation only, export FSDP+LoRA to a loadable model, then evaluate each environment × training seed on the held-out final test.

```bash
# Select among complete saved steps using validation success, successful-turn
# count as a tie-breaker, then later step. This reads validation/*.jsonl only.
bash scripts/run_experiment_matrix.sh select-checkpoints

# Validate every FSDP shard and LoRA adapter without loading model weights.
DRY_RUN=1 bash scripts/run_experiment_matrix.sh export-checkpoints

# Perform the real FSDP conversion and merge LoRA into a standalone HF model.
bash scripts/run_experiment_matrix.sh export-checkpoints

# Inspect all final-test commands and linked manifests without starting SGLang.
DRY_RUN=1 bash scripts/run_experiment_matrix.sh final-test

# Run the held-out final test for every environment × method × training seed.
bash scripts/run_experiment_matrix.sh final-test

# Winner-only route: base + one selected method in both environments (4 rows).
FINAL_EXPECTED_METHODS=no_concat_episode_grpo \
  bash scripts/run_experiment_matrix.sh final-results

# Full-comparison route instead: base + all three methods (8 rows).
# Use this only if all 18 confirmatory runs and final tests were executed.
FINAL_EXPECTED_METHODS=concat_grpo,no_concat_gae,no_concat_episode_grpo \
  bash scripts/run_experiment_matrix.sh final-results
```

**Expected artifacts:**

```text
<training-run>/selection/checkpoint_selection.json
exps/vlm_agent_rl_exports/<training-run>/export_manifest.json
exps/vlm_agent_rl_exports/<training-run>/model/
exps/eval/final_test/<environment>/<method>/train_seed_<seed>/checkpoint_<step>/manifest.json
results/gpu/final/final_test_runs.csv
results/gpu/final/final_test_aggregates.csv
results/gpu/final/base_eval_results.csv
results/gpu/final/main_results.csv
```

`main_results.csv` remains `incomplete-artifacts` with null performance fields if any required seed is missing or any final-test run lacks complete provenance/GPU evidence. Validation metrics are never used as final results.
Its `GPU·h` column is the mean training cost per environment/method/training seed,
matching the resume/registry convention; `final_test_aggregates.csv` additionally
records `GPU·h Total` across all three training seeds.

---

## Phase 7: Visual Ablation (Anti-Cheating)

**Purpose:** Verify model uses visual input, not just memorizing templates

```bash
# Use the standalone model produced by Phase 6, never the raw actor checkpoint.
TRAINING_RUN_NAME=sokoban_confirmatory_no_concat_episode_grpo_outcome_trajectory_seed0
EVAL_MODEL_PATH="/path/to/vlm-agent-rl/exps/vlm_agent_rl_exports/${TRAINING_RUN_NAME}/model" \
EVAL_ENVIRONMENT=sokoban \
EVAL_METHOD=no_concat_episode_grpo \
ANTI_CHEAT_ROOT="exps/eval/anti_cheat/${TRAINING_RUN_NAME}" \
EVALUATION_ROLE=anti_cheat \
  bash scripts/run_experiment_matrix.sh anti-cheat
```

`EVAL_METHOD` is required: it selects the evaluation context protocol for the
checkpoint. Omitting it exits 2 rather than silently falling back to the full
dialogue history.

**Expected output:**
```
3 conditions × 128 episodes:
- none: normal images → baseline success rate
- remove: no image → should drop significantly if visual reasoning matters
- shuffle_tiles: spatial scrambling → drops but less than remove if using coarse features

Results in exps/eval/*/
```

**Analysis:**
```bash
ABLATION_ROOT="exps/eval/anti_cheat/${TRAINING_RUN_NAME}"
python -m vagen.analysis.analyze_rollouts \
  --eval-dump "${ABLATION_ROOT}/sokoban_none" \
  --eval-dump "${ABLATION_ROOT}/sokoban_remove" \
  --eval-dump "${ABLATION_ROOT}/sokoban_shuffle_tiles" \
  --output-dir results/gpu/visual_ablation
```

**Interpretation:**
- `success(none) >> success(remove)` → Model uses visual input ✅
- `success(none) ≈ success(remove)` → Likely template memorization ⚠️
- `success(none) > success(shuffle) > success(remove)` → Uses coarse visual stats

---

## Phase 8: Result Collection

**Purpose:** Extract all metrics into main_results.csv

```bash
# Re-run the leakage-safe aggregator after any resumed final-test job. Choose
# exactly the route that was actually run; winner-only is a 4-row registry.
FINAL_EXPECTED_METHODS=no_concat_episode_grpo \
  bash scripts/run_experiment_matrix.sh final-results

# Refuse publication if any environment/method group is incomplete.
python - <<'PY'
import csv
rows = list(csv.DictReader(open("results/gpu/final/main_results.csv")))
bad = [row for row in rows if row["Status"] != "complete"]
if bad:
    raise SystemExit(f"incomplete final results: {bad}")
print(f"complete final result groups: {len(rows)}")
PY

# Atomically publish base + trained final-test rows to the root registry. This
# refuses incomplete results and preserves the old table under results/gpu/.
bash scripts/run_experiment_matrix.sh publish-results
```

---

## Backup and Shutdown

**Before stopping the instance:**

```bash
# Create a Linux-safe evidence archive without relying on shallow globs.
cd vlm-agent-rl
find exps/vlm_agent_rl exps/vlm_agent_rl_exports exps/eval results/gpu \
  -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.csv' \
  -o -name '*.txt' -o -name '*.log' -o -name '*.sh' \) -print0 \
  > /tmp/vlm_agent_evidence_files
tar --null -czf "results_$(date +%Y%m%d).tar.gz" \
  --files-from=/tmp/vlm_agent_evidence_files

# Download to local machine
scp -P <port> root@<host>:/path/to/results_*.tar.gz ~/Downloads/

# Optional: sync W&B if you want online tracking
WANDB_API_KEY=<your_key> bash scripts/sync_wandb.sh
```

**What to download:**
1. `results_<date>.tar.gz` - All experiment artifacts
2. `exps/vlm_agent_rl_exports/<training-run>/model/` - Selected standalone model (if disk allows)
3. `artifacts/environment/gpu-pip-freeze.txt` - Exact environment
4. `artifacts/environment/nvidia-smi.txt` - GPU info

---

## Cost Accounting (Measured, Single GPU)

Conservative planning envelope: about **134–137 GPU·h** for winner-only
(including six independent final tests), or **344–347 GPU·h** for the full
three-method route (including eighteen independent final tests). Checkpoint
export/LoRA merge is added from its measured time rather than a guessed value.

Do not use a fixed hour or currency estimate across GPU types. After each phase, calculate cost from that instance's measured GPU·h and the price shown by AutoDL:

```bash
export AUTODL_RATE_RMB_PER_HOUR=2.88  # Replace with the actual price of this instance.
python - <<'PY'
import glob
import json
import os

paths = glob.glob("exps/**/gpu_summary.json", recursive=True)
gpu_hours = 0.0
for path in paths:
    payload = json.load(open(path))
    if payload.get("return_code") == 0:
        gpu_hours += float(payload.get("gpu_hours") or 0.0)
rate = float(os.environ["AUTODL_RATE_RMB_PER_HOUR"])
print(f"measured single-GPU hours: {gpu_hours:.3f}")
print(f"measured cost: RMB {gpu_hours * rate:.2f}")
PY
```

Use smoke and screening measurements to decide whether to run the winner-only or full confirmatory matrix. If the smoke OOMs, change the card and measure again; do not extrapolate from the failed run.

---

## Troubleshooting

### OOM (Out of Memory)

```bash
# Reduce batch size
export TRAIN_BATCH_SIZE=2  # default is 8
export ROLLOUT_N=2         # default is 4 for GRPO

# Reduce GPU memory utilization for inference
export GPU_MEMORY_UTILIZATION=0.40  # default 0.50
```

### Parity Gate Failure

```bash
# Check parity report
cat exps/vlm_agent_rl/<run>/parity.json | jq .

# Common causes:
# - Processor mismatch → should not happen with Qwen2.5-VL
# - Chat template issue → check resolved_config.yaml
# - Image token alignment → check train.log for warnings
```

### Navigation Server Hangs

```bash
# Check server log
tail -f exps/system/navigation_server.log

# Manually restart
pkill -f "vagen.envs.navigation.serve"
python -m vagen.envs.navigation.serve --host=127.0.0.1 --port=8000

# Check health
curl http://127.0.0.1:8000/health
```

### Slow Convergence

```bash
# Check validation success rate trends
grep -r "success_rate" exps/vlm_agent_rl/<run>/validation/*.jsonl

# If not improving after 100 steps → may need hyperparameter adjustment
# But proceed to 401 steps before declaring failure
```

---

## Success Criteria Summary

| Phase | Pass Condition | Fail Action |
|---|---|---|
| Smoke | All 3 methods pass parity; peak VRAM < 48GB (or < 80GB on large card) | Investigate logs; check model/processor |
| Base Eval | Completes without errors; success > 0% | Check environment setup |
| Core Screening | Parity passed; no non-finite loss; success rate > baseline | Debug before confirmatory |
| Episode Screening | At least 1 config with parity pass + improvement | Select best config |
| Confirmatory | Completes 401 steps; parity passed all seeds | Select checkpoints on validation only |
| Final test | Export linked; held-out episodes complete for training seeds 0,1,2 | Only this is the final result |
| Visual Ablation | success(none) >> success(remove) | Indicates visual reasoning |

---

## Quick Reference Commands

```bash
# Check all parity gates and print each source path (the JSON has no run_name).
find exps/vlm_agent_rl -name parity.json \
  -exec jq -r 'input_filename + ": " + (.gate_passed|tostring)' {} +

# Check all GPU summaries and print each source path.
find exps/vlm_agent_rl -name gpu_summary.json \
  -exec jq -r 'input_filename + ": " + (.peak_vram_mib|tostring) + " MiB, " + (.gpu_hours|tostring) + " GPU·h"' {} +

# Analyze specific run
python -m vagen.analysis.analyze_rollouts --run exps/vlm_agent_rl/<name> --output-dir results/gpu/<name>

# Live training log
tail -f exps/vlm_agent_rl/<name>/train.log | grep -E "step|success|loss"

# Check conda env
conda run -n vagen python -m pytest vagen/tests/ -q
```

**This checklist should be sufficient for a new user with GPU access to reproduce all experiments without reading the full documentation.**
