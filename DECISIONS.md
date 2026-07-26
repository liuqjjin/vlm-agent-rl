# Engineering and research decisions

Decisions are dated 2026-07-27 unless noted. “Evidence” means source code, a paper, or an executed artifact; it does not mean an expected GPU outcome.

## D1 — Keep VAGEN as the host system

**Decision:** build on VAGEN main at `2936322a6f6c02fbd29ca28e4b6ec37eefefc081` and its pinned `JamesKrW/verl` branch.

**Evidence:** VAGEN already owns the environment interface, agent loops, Ray trainer, SGLang/vLLM rollout engines, FSDP workers, advantage registry, filters, metrics, and validation dumps. Replacing those layers would create a larger unreviewable delta without advancing the research question.

**Rejected:** merging VAGEN and `langfengQ/verl-agent` into one environment. Both provide a `verl` Python package and have different dependency assumptions. `verl-agent` was cloned at `20bd331bdbc9026a5668e11362178e10ab7400c8` and used read-only.

## D2 — Qwen2.5-VL-3B is the first formal visual model

**Decision:** all formal no-concat entry points default to `Qwen/Qwen2.5-VL-3B-Instruct`; Qwen3-VL is fail-closed.

**Evidence:** the current no-concat path has verified Qwen2/Qwen2.5 processor semantics. Qwen3-VL introduces a different processor and M-RoPE/position-ID path. A model-name change is not a valid adaptation. The local guard and contract tests prevent a silent 1-D fallback.

**Exit condition for Qwen3-VL:** processor-specific position IDs, image token alignment, chat-template equivalence, and a passing first-update parity report.

**References:** [Qwen2-VL](https://arxiv.org/abs/2409.12191), [VAGEN](https://arxiv.org/abs/2510.16907).

## D3 — Repair both sparse-value links

**Decision:** attach `value_mask` for the active estimator name `no_concat_gae` and preserve it in critic batch selection.

**Evidence:** a tensor counterexample showed an ignored position receiving gradient when the worker dropped the mask. In the 20-update controlled run, the fixed ignored prediction remained `0.5`, while the legacy counterfactual moved to `-87.7815` toward the `-100` sentinel. The supervised position reached `1.9654` toward `2.0`.

**Upstream status:** VAGEN PR [#109](https://github.com/mll-lab-nu/VAGEN/pull/109) concerns no-concat validation logging; it does not repair these two sparse-value links at the pinned commit.

## D4 — Episode GRPO must reconstruct trajectories first

**Decision:** no-concat episode GRPO groups rows by `(group_idx, traj_idx)`, validates completeness, then computes one score and advantage per trajectory.

**Evidence:** a no-concat row is one turn. Treating each row as an independent GRPO sample changes the statistical unit, overweights long trajectories, and mixes incomplete distributed padding with real samples.

**Rejected:** direct turn-level GRPO. It cannot represent an episode objective without an explicit trajectory reconstruction step.

**References:** [DeepSeekMath/GRPO](https://arxiv.org/abs/2402.03300), [DAPO](https://arxiv.org/abs/2503.14476).

## D5 — Disable the existing filter for formal no-concat runs

**Decision:** `filter.enable=False` in every formal matrix run.

**Evidence:** the pinned reward-variance filter assumes sample rows are the group-comparison unit. In no-concat mode those rows are turns, so filtering them before trajectory reconstruction has the wrong semantics. Episode GRPO already handles zero-variance groups by returning zero advantages.

**Exit condition:** a future filter must operate on reconstructed trajectories and preserve whole groups.

## D6 — Treat reward and policy aggregation as ablations

**Decision:** expose three reward reductions and three policy objectives; do not name a universal winner in code or documentation.

**Evidence:** the 20-seed shortest-path control held environment state and action path fixed. Splitting actions over turns increased raw reward by `0.20–0.30` on every seed, mean `0.245`. Outcome, bounded-process, and format-gate reductions produced zero same-path delta in this control. This proves the raw bias and validates the reductions on the control; it does not establish which trains best.

Token, turn, and trajectory weighting assign different mass when turns and trajectories have unequal lengths. They are different estimands, not numerical fixes.

## D7 — Use state-relative advantage as the only P2 direction

**Decision:** add canonical pre-action state anchors and a strict preflight, but do not enable a state-relative optimizer before model-rollout evidence.

**Evidence:** visual Sokoban has an exact grid state; Navigation exposes an exact simulator pose. Both can provide a text grouping key while the policy continues to consume pixels. `group_idx` scopes the comparison to the same task. The preflight measures singleton groups, comparable rows, action diversity, and return-to-go variance.

**Stop rule:** if real base-policy rollouts fail any configured signal threshold, record a negative result and do not spend training GPU-hours on this direction.

**Reference implementation studied:** [verl-agent](https://github.com/langfengQ/verl-agent) at the commit recorded in `UPSTREAM.md`. **Research reference:** [GiGPO](https://arxiv.org/abs/2505.10978).

## D8 — Parity is a hard pre-update gate

**Decision:** recompute training-forward log-probabilities and compare them to rollout-engine log-probabilities before the first update.

**Evidence:** PPO-family clipping assumes the logged behavior probability and optimization probability are meaningfully aligned. Processor, image-token, M-RoPE, template, and truncation mismatches can violate that assumption before learning begins.

The report is persisted before raising, including failed-gate evidence. Bypass mode is rejected because it removes the independent training forward needed for the check.

## D9 — Replace Python’s randomized hash in Sokoban retry seeding

**Decision:** use a stable BLAKE2s seed transition.

**Evidence:** two executions with the same public seeds produced different constrained maps because `hash(str(seed))` depends on `PYTHONHASHSEED`. After the change, the four raw CPU files had identical SHA-256 values under two different hash salts.

## D10 — Anti-cheating controls must be deterministic and traceable

**Decision:** evaluate `none`, `remove`, and `shuffle_tiles` as distinct observation conditions; include the condition in metrics and resume identity.

**Evidence:** without the ablation in the resume key, a completed unablated episode could be incorrectly reused for an ablated run. Tile shuffle preserves dimensions and the pixel multiset while destroying spatial layout.

**Limitation:** tile shuffle is not cross-episode image reassignment. It is a spatial visual-dependence control.

## D11 — Report missing GPU results as missing

**Decision:** the current result registry contains null metrics with `pending-external-gpu`, not projected values.

**Evidence:** the workstation reports `torch.cuda.is_available() == False` and has no `nvidia-smi`. CPU-valid code and experiments were completed; SGLang visual inference, AI2-THOR rendering, parity measurement, VRAM, GPU-hours, training, and checkpoint evaluation require external CUDA hardware.

## D12 — Keep the verl fork delta to two changes

**Decision:** the submodule contains only:

1. conditional preservation of `value_mask` in critic batches;
2. optional normalized `policy_weights` in the actor loss.

Both changes have CPU tests in the submodule. No Ray, FSDP, inference engine, PPO scheduler, or GAE infrastructure was rewritten.
