# Upstream provenance and reproducibility

This file records the source state audited on 2026-07-27. Commit identifiers,
not moving branch names, define the implementation reviewed for this project.

## Source ledger

| Component | Role | Repository | Audited branch | Audited commit | License |
|---|---|---|---|---|---|
| VAGEN | Top-level framework and environment loop | [mll-lab-nu/VAGEN](https://github.com/mll-lab-nu/VAGEN) | `main` | [`2936322a6f6c02fbd29ca28e4b6ec37eefefc081`](https://github.com/mll-lab-nu/VAGEN/commit/2936322a6f6c02fbd29ca28e4b6ec37eefefc081) | MIT |
| VAGEN historical no-concat line | Rename/history audit only; not a separate dependency | [mll-lab-nu/VAGEN](https://github.com/mll-lab-nu/VAGEN) | `vagen-lite` | [`527c82de00a44a0b07327676d5c55d9bf77d0f82`](https://github.com/mll-lab-nu/VAGEN/commit/527c82de00a44a0b07327676d5c55d9bf77d0f82) | MIT |
| VAGEN verl fork | Ray/FSDP/SGLang/PPO submodule base | [JamesKrW/verl](https://github.com/JamesKrW/verl) | `vagen-lite` | [`3fe0a29975e1b02ae2bd1dec249f7807dd7966f5`](https://github.com/JamesKrW/verl/commit/3fe0a29975e1b02ae2bd1dec249f7807dd7966f5) | Apache-2.0 |
| Local verl delta | Sparse critic mask and policy weights | [liuqjjin/verl](https://github.com/liuqjjin/verl) | `vagen-value-mask-policy-weights` | [`fecc1520af10cf266ba1947c6b3b9bd5259fe926`](https://github.com/liuqjjin/verl/commit/fecc1520af10cf266ba1947c6b3b9bd5259fe926) | Apache-2.0 |
| verl-agent | Read-only GiGPO/state-grouping design reference | [langfengQ/verl-agent](https://github.com/langfengQ/verl-agent) | `master` | [`20bd331bdbc9026a5668e11362178e10ab7400c8`](https://github.com/langfengQ/verl-agent/commit/20bd331bdbc9026a5668e11362178e10ab7400c8) | Apache-2.0 |

At audit time, `git ls-remote` returned both recorded VAGEN heads, the
JamesKrW/verl head, and the verl-agent head above. The verl-agent checkout was
kept outside this repository, was not installed into the VAGEN environment,
and contributed no copied source.

## Historical `vagen-lite` finding

VAGEN `vagen-lite` is the merge base of the audited `main`, not a fourth
runtime component. It registered the first-token sparse estimator as
`no_concat_gae_first`, and the trainer attached `value_mask` for that legacy
name. Later `main` renamed the registered estimator to `no_concat_gae` while
the trainer condition still listed only the two historical names. Comparing
these exact branch heads identified the first value-mask break; the second was
the pinned verl critic worker dropping the optional field during batch
selection.

## Local changes relative to VAGEN

The top-level fork starts at VAGEN commit `2936322a`. Its changes are limited
to the research question and the evidence needed to evaluate it:

- preserve sparse value supervision for the active `no_concat_gae` path;
- reconstruct complete no-concat trajectories and compute episode GRPO;
- implement normalized token, turn, and trajectory policy objectives;
- fail closed on unverified visual processors and measure rollout/training
  log-probability parity before the first update;
- add canonical state anchors and a state-relative signal preflight;
- make Sokoban retries deterministic across Python hash salts;
- add local visual baselines, observation ablations, resource accounting,
  rollout analysis, experiment orchestration, and CPU evidence.

The complete commit sequence is visible with:

```bash
git log --oneline 2936322a6f6c02fbd29ca28e4b6ec37eefefc081..HEAD
```

## Local changes inside `verl/`

The submodule has exactly two implementation commits on top of `3fe0a299`:

1. `71b5c5c7` — preserve the optional `value_mask` in critic update batches.
2. `fecc1520` — consume normalized `policy_weights` in the actor objective.

The delta touches only:

```text
verl/workers/critic/dp_critic.py
verl/workers/actor/dp_actor.py
verl/trainer/ppo/core_algos.py
tests/trainer/ppo/test_sparse_value_supervision_on_cpu.py
tests/trainer/ppo/test_policy_weights_on_cpu.py
```

It contains 148 added lines and one removed line. Ray orchestration, FSDP,
rollout engines, schedulers, and upstream advantage implementations were not
forked or replaced.

Inspect the exact delta with:

```bash
git -C verl diff --stat \
  3fe0a29975e1b02ae2bd1dec249f7807dd7966f5..fecc1520af10cf266ba1947c6b3b9bd5259fe926
```

## Runtime pins

The reproducible CPU correctness environment is fully enumerated in
[`requirements/cpu-test.txt`](requirements/cpu-test.txt). The run committed
under `results/cpu/20260727-mac-arm64` used:

```text
Python 3.12.13
NumPy 1.26.4
PyTorch 2.8.0
Ray 2.48.0
TensorDict 0.10.0
Transformers 4.57.1
pytest 8.4.1
```

The Linux GPU bootstrap uses Python 3.12 and the pinned VAGEN verl installer:

```text
SGLang 0.5.2
vLLM 0.11.0
FlashAttention 2.8.1 for CUDA 12 / PyTorch 2.8 / Python 3.12
FlashInfer 0.3.1
Transformers 4.57.1
AI2-THOR 5.0.0
Fire 0.7.1
```

Some transitive CUDA packages in the upstream installer are expressed as
compatible ranges. Every actual GPU run therefore writes
`artifacts/environment/gpu-pip-freeze.txt` and `nvidia-smi.txt`; those files,
the run manifest, and the two git commits are the definitive environment
record for reported GPU results.

## Model and external runtime boundary

The scripts download but do not redistribute
[`Qwen/Qwen2.5-VL-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct).
Its weights are governed by the
[Qwen Research License Agreement](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE),
including non-commercial-use and attribution conditions. This is separate
from the repository's MIT code license.

AI2-THOR 5.0.0 and gym-sokoban 0.0.6 are installed dependencies rather than
vendored source. They remain under their own upstream licenses:
[AI2-THOR Apache-2.0](https://github.com/allenai/ai2thor/blob/main/LICENSE)
and
[gym-sokoban MIT](https://github.com/mpSchrader/gym-sokoban/blob/default/LICENSE).

## Research sources

- [VAGEN](https://arxiv.org/abs/2510.16907) — the host framework and visual
  agent training setup.
- [Qwen2-VL](https://arxiv.org/abs/2409.12191) — multimodal processor and
  M-RoPE model family.
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — GRPO.
- [DAPO](https://arxiv.org/abs/2503.14476) — group-relative RL engineering
  and diagnostics.
- [GiGPO](https://arxiv.org/abs/2505.10978) and
  [verl-agent](https://github.com/langfengQ/verl-agent) — state-relative
  grouping as a studied P2 direction.

Design choices and rejected alternatives are recorded in
[`DECISIONS.md`](DECISIONS.md).
