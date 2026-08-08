# Architecture and invariants

## Scope

The project changes the learning semantics around VAGEN’s existing execution stack. Ray orchestration, FSDP, SGLang/vLLM inference, PPO plumbing, and environment interfaces remain upstream infrastructure. The local code owns:

1. how multi-turn samples are represented;
2. how sparse value targets remain masked;
3. how complete trajectories are reconstructed;
4. how trajectory scores become advantages;
5. how per-token policy losses represent token, turn, or trajectory objectives;
6. which checks must pass before a parameter update.

## End-to-end data flow

```mermaid
sequenceDiagram
    participant Env as Visual environment
    participant Loop as Agent loop
    participant Rollout as vLLM async rollout
    participant Trainer as Ray trainer
    participant Algo as Advantage estimator
    participant Worker as Actor / critic workers

    Env->>Loop: pre-action image + text observation
    Loop->>Loop: canonical state_anchor + remaining budget
    Loop->>Rollout: system + current or accumulated context
    Rollout-->>Loop: action tokens + rollout log-probabilities
    Loop->>Env: decoded action
    Env-->>Loop: reward, done, metrics, next observation
    Loop-->>Trainer: turn row(s) with group/traj/turn identity
    Trainer->>Worker: recompute old log-probabilities
    Trainer->>Trainer: first-update rollout/train parity gate
    Trainer->>Algo: rewards, masks, trajectory metadata
    Algo-->>Trainer: advantages + optional policy_weights
    Trainer->>Worker: critic update when enabled
    Trainer->>Worker: weighted actor update
```

## Sample identity and ownership

Every no-concat row has the following identity:

```text
(group_idx, traj_idx, turn_idx)
```

- `group_idx` identifies the original prompt/task group.
- `traj_idx` identifies one of the `rollout.n` sampled trajectories.
- `turn_idx` is contiguous within a trajectory.
- `last_turn` is true exactly once and only on the final turn.
- `traj_success` records whether that environment step reports success; episode
  reduction takes `any(...)` across the reconstructed trajectory.
- `state_anchor` is the canonical pre-action state for that row.

The agent loop creates these fields. `ray_trainer.py` carries them through padding and logging. The advantage estimator consumes them but does not reinterpret their ownership.

Distributed padding may duplicate a row. A duplicate with the same identity and the same reward/mask/outcome/terminal data is ignored. A duplicate with conflicting content is an error.

## Concat and no-concat semantics

### Concat

One optimized sample represents a full trajectory. Intermediate prompt tokens are inserted with response mask zero. `concat_val_multi_turn` strips each turn’s local padding before concatenation, so decoded text, masks, and image count stay aligned.

### No-concat

One optimized sample represents one environment turn:

```text
prompt  = system prompt + current observation
target  = current assistant action
history = absent
```

This is a deliberate short-context POMDP approximation. A turn row must not be treated as an independent trajectory when computing episode-level statistics.

## Sparse critic supervision

No-concat GAE produces one supervised value position per turn. Other return positions carry the sentinel `-100`.

```text
value_mask = (returns != -100)
effective critic mask = response_mask * value_mask
```

Two links are necessary:

1. the active estimator name `no_concat_gae` must trigger `compute_value_mask`;
2. the critic worker’s batch selection must preserve the optional `value_mask`.

The tests check both static dispatch and 20 optimizer steps. With the mask, the ignored prediction has exactly zero gradient; without it, the prediction is trained toward `-100`.

## No-concat episode GRPO

`compute_no_concat_episode_grpo` runs in this order:

1. validate shapes and `rollout.n >= 2`;
2. remove exact padding duplicates;
3. reconstruct rows by `(group_idx, traj_idx)`;
4. require contiguous turns and one final terminal marker;
5. require exactly `rollout.n` trajectories per group, or explicitly drop an incomplete group;
6. reduce turn rewards to one trajectory score;
7. center and optionally standardize trajectory scores within `group_idx`;
8. broadcast one trajectory advantage to valid action tokens in every turn;
9. construct normalized policy weights.

If a group has zero score variance, its standardized advantages are zero. It is never divided by an epsilon to manufacture signal.

## Reward reductions

Let `r_t` be VAGEN’s turn reward, with the success bonus removed from the final turn when constructing process reward.

- `outcome`: `1` for success, else `0`.
- `bounded_process`: outcome plus the clipped sum of process rewards.
- `format_gate`: outcome only if every turn meets its environment's
  per-turn format-reward threshold (`0.10` Sokoban, `0.02` FrozenLake,
  `0.01` Navigation).

These are experiment conditions, not interchangeable implementations. The controlled CPU experiment shows that all three remove the measured same-path turn-splitting delta on its 20 seeded rooms; model behavior still requires evaluation.

## Policy objectives

`policy_weights` always sum to one over active, non-duplicate response tokens.

- `token`: every action token has equal mass.
- `turn`: every turn has equal mass, divided across that turn’s tokens.
- `trajectory`: every trajectory has equal mass, divided across all its action tokens.

The verl actor compensates for its token-mean reduction so the weighted policy-gradient sum is invariant to microbatch partitioning.
The formal episode-GRPO entry point is restricted to one GPU until equivalent
cross-rank scaling has been validated.

## Rollout/training parity

Before the first update, training recomputes `old_log_probs` and compares them with `rollout_log_probs` on valid action tokens:

```text
ratio = exp(log p_train - log p_rollout)
```

The report contains mean, median, P95, P99, mean absolute log-probability delta, clip fraction, and token count. A failed gate is written to `parity.json` before the exception aborts training.

The default hard gates are:

- `abs(P95 - 1) <= 0.10`;
- `abs(P99 - 1) <= 0.20`;
- mean absolute log-probability delta `<= 0.05`;
- fraction outside `[0.8, 1.2] <= 0.01`.

This gate protects the expensive run from processor, chat-template, truncation, image-token, position-ID, or M-RoPE mismatches.

## State-relative preflight

The policy still receives images. A separate text-only anchor exists solely for grouping:

- Sokoban: canonical grid text;
- Navigation: rounded position, rotation, camera horizon, and standing state;
- both: remaining turn budget appended before the action.

The remote Navigation protocol explicitly transports the anchor. Groups remain scoped by `group_idx`, preventing states from unrelated tasks from being compared.

No state-relative advantage is enabled until real base-policy rows pass:

- anchor coverage;
- comparable-row fraction;
- action diversity;
- return-to-go variance;
- minimum sample count.

## Evaluation and artifacts

Every training run writes:

- `manifest.json`;
- `train_command.sh`;
- `resolved_config.yaml`;
- `train.log`;
- `parity.json`;
- raw rollout/validation JSONL;
- `gpu_metrics/gpu_samples.csv`, scoped to `CUDA_VISIBLE_DEVICES`;
- `gpu_metrics/gpu_summary.json`;
- checkpoints according to the phase.

The metrics wrapper verifies that the sampled visible-device count matches the
run's declared training GPU count or evaluation `DP_SIZE × TP_SIZE`; it refuses
to launch when accounting would include an unused visible GPU.
When a manifest-compatible run resumes, raw samples and active session
durations are appended on a continuous runtime axis, so GPU-hours include every
attempt without charging the instance's offline gap. Resume also requires the
same device index, model name, and total memory inventory; any sampling error
keeps the analyzed result incomplete.
An identity-matched complete run is skipped before another process starts.
Parity-failed or sampling-tainted directories are immutable negative evidence;
a replacement attempt must use a new run directory.

The manifest distinguishes the controlled run seed (Python hashing and
dataloader order) from bitwise CUDA determinism, which the asynchronous
rollout stack does not promise.

Every local visual evaluation writes `manifest.json`, `eval_command.sh`,
`resolved_config.txt`, its log, episode metrics/transcripts/images, a tag
summary, and GPU metrics. Image removal and tile shuffle are applied before
adapter formatting and are included in resume identity, so an unablated
episode cannot incorrectly satisfy an ablated resume.

`vagen.analysis.analyze_rollouts` derives success, successful mean turns, invalid-action rate, template concentration, reward/turn correlation, group reward variance, and representative failures from these raw artifacts.
