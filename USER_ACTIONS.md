# External actions required for CUDA runs

Everything that can be completed on the current Mac has been completed. The remaining blocker is access to a Linux NVIDIA instance.

## Do these four things

1. Sign in to [AutoDL](https://www.autodl.com/), complete any required payment or identity verification, and open the live GPU rental page. Prices and stock change; use the live page rather than a number copied into this repository.
2. Create one instance with:
   - **lowest-cost first choice:** the cheapest available single CUDA GPU with
     at least 48 GiB VRAM; 32 GiB cards are outside the formal configuration;
   - **fallback only if the critic-bearing smoke OOMs:** one A800/A100-class
     80 GiB GPU or a 96 GiB PRO 6000;
   - **image:** a Conda-based Linux image with a CUDA-12-compatible NVIDIA
     driver; the bootstrap creates the pinned Python 3.12/PyTorch environment;
   - **host memory:** at least 96 GiB for the offloaded critic comparison;
   - **disk:** at least 600 GiB available for the winner-only route; about
     1.5 TiB for the full three-method route;
   - public network and SSH enabled.
3. Add your SSH public key in the AutoDL console. Do not send a private key or account password. The official instructions are [AutoDL SSH](https://api.autodl.com/docs/ssh/).
4. Send the instance’s SSH command, for example `ssh -p <port> root@<host>`. If Hugging Face throttles the public Qwen download, provide an HF read token through a secure environment-variable mechanism. A W&B key is optional because runs default to offline logging.

That is all the external interaction required. Login, payment, verification codes, instance creation, and secret provision must remain user actions.

## Why this configuration

- Qwen2.5-VL-3B uses LoRA, 50% inference-engine allocation, and
  parameter/optimizer offload; 48 GiB is the lowest credible first attempt.
  The fixed no-concat GAE comparison also holds a critic and is the
  memory-risk condition, and the smoke now executes that exact path before
  screening, so it decides whether 80 GiB is necessary.
- AI2-THOR Navigation runs a Unity renderer alongside model inference, which is
  why 24 GiB cards are not the formal default even if episode-only inference
  might start on one.
- The bootstrap pins PyTorch 2.8/CUDA-12-compatible SGLang/vLLM dependencies,
  installs Vulkan/AI2-THOR, and requires at least 600 GiB free by default for
  five actor candidates across the winner-only runs and exported models. Set
  `MIN_FREE_GB=1536` before bootstrap for the full three-method route.
- Escalate from 48 GiB only on measured OOM/peak-VRAM evidence; do not pay for
  80–96 GiB in advance from an estimate.

AutoDL’s current supported base-image table is at [Base configurations](https://www.autodl.com/docs/base_config/), and instance creation/storage guidance is at [Quick start](https://www.autodl.com/docs/quick_start/).

## What will run after SSH is available

The repository already contains the commands; no manual dependency work is needed:

```bash
DOWNLOAD_MODEL=1 PRELOAD_NAVIGATION=1 bash scripts/autodl_bootstrap.sh
bash scripts/run_experiment_matrix.sh smoke
bash scripts/run_experiment_matrix.sh base-eval
bash scripts/run_experiment_matrix.sh core-screening
bash scripts/run_experiment_matrix.sh episode-screening
```

Only after reviewing screening evidence will confirmatory seeds be launched. Long-running work will use `tmux`; manifests, raw logs, checkpoints, W&B offline data, parity reports, GPU samples, and result summaries will be copied off the instance before shutdown.

## Cost control

- Start with one GPU and the five-step smoke.
- Stop immediately on environment failure, non-finite loss, or parity-gate failure.
- Do not launch the 3×3 ablation until smoke and base evaluation pass.
- Use the winner-only route to control cost. The completion-state resume's
  three-method comparison requires the separate full confirmatory route; do
  not use those comparison numbers after a winner-only run.
- Plan about 134–137 GPU·h for winner-only (six independent final tests) or
  344–347 GPU·h for all three methods (eighteen final tests). Add checkpoint
  export/LoRA merge from measured time and multiply by AutoDL's live RMB rate.
- Stop the instance after artifacts are backed up. AutoDL documents persistence and image-saving behavior at [Image management](https://www.autodl.com/docs/image/).
