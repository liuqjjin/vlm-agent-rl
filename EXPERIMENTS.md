# Experiment protocol

## Research questions

1. Does short-context no-concat training remain competitive with full-trajectory concat training on visual multi-turn tasks?
2. Does repaired sparse critic supervision make no-concat GAE stable enough to be a meaningful baseline?
3. Can a critic-free, trajectory-aware group-relative objective improve sample efficiency or stability?
4. How do token, turn, and trajectory policy objectives behave when output and trajectory lengths vary?
5. How much of the measured performance depends on current visual observations rather than prompt templates or reward length?

The declarative source of truth is [experiments/matrix.yaml](experiments/matrix.yaml). Shell entry points read environment variables for resource scaling, but formal invariants are not optional: Qwen2.5-VL-3B, filter off, parity gate on, and explicit concat/critic semantics.

## Core methods

| Method | Context row | Advantage unit | Critic | Default `rollout.n` |
|---|---|---|---:|---:|
| Base Qwen2.5-VL-3B | evaluation only | n/a | no | n/a |
| concat GRPO | complete trajectory | trajectory | no | 4 |
| fixed no-concat GAE | one turn | reconstructed temporal GAE | yes | 1 |
| no-concat episode GRPO | one turn | reconstructed trajectory/group | no | 4 |

The no-concat episode condition is screened across:

```text
reward ∈ {outcome, bounded_process, format_gate}
policy objective ∈ {token, turn, trajectory}
```

No existing reward-variance filter is used because it operates before no-concat trajectory reconstruction.
The format gate uses each environment's actual per-turn format reward, recorded
in the run manifest, rather than a cross-environment constant.
The normalized episode policy-weight path is fail-closed to one GPU until its
cross-rank scaling is validated; the declared funnel is therefore a single-GPU
protocol.

## Environments and held-out seeds

### Sokoban

- visual 6×6 rooms, one box;
- train config: `examples/train/sokoban/train_sokoban_vision.yaml`;
- validation config: `examples/train/sokoban/val_sokoban_vision.yaml`;
- standalone evaluation: 128 held-out seeds `[10001, 10129)`;
- success and successful-trajectory mean turns are primary behavioral metrics.

### Navigation

- egocentric RGB AI2-THOR, partially observable;
- base split only for the first controlled comparison;
- train seeds `[0, 30)`, validation seeds `[30, 60)`;
- standalone evaluation covers the 30 held-out seeds `[30, 60)`;
- the remote protocol transports canonical pose anchors but never exposes them to the model.

The same held-out seed sets are reused for the base model and every trained
method. They are disjoint from each environment's training seed domain.

The confirmatory `SEED` controls Python hashing and training-dataloader order.
It is recorded as such in `manifest.json`; asynchronous inference and CUDA
kernels are not claimed to be bitwise deterministic.

## Funnel

### 1. CPU correctness

Required before CUDA:

```bash
conda run -n vagen bash scripts/run_smoke.sh
```

This covers trajectory reconstruction, incomplete groups, duplicates, zero variance, critic masks, 20-step dynamics, objective weights, microbatch invariance, parity metrics/reporting, processor guard, deterministic environment seeding, state anchors, remote transport, observation ablation, GPU metric parsing, and rollout analysis.

Current result: **104 passed**.

### 2. GPU smoke

```bash
bash scripts/run_experiment_matrix.sh smoke
```

Order:

1. FrozenLake Qwen2.5-VL-3B local SGLang visual evaluation;
2. Sokoban Qwen2.5-VL-3B local SGLang visual evaluation;
3. one concat-GRPO update;
4. one fixed no-concat-GAE update, including the critic;
5. five no-concat episode-GRPO updates.

The default OpenAI Sokoban evaluator is not accepted as the local baseline.
The one-step core method runs are configuration, parity, and peak-memory
checks, not behavioral experiments.

The smoke passes only if the first-update parity gate passes. A failed gate stops before the first actor update.

### 3. Base evaluation

```bash
bash scripts/run_experiment_matrix.sh base-eval
```

This runs the same local Qwen2.5-VL-3B server interface on Sokoban and Navigation. Each run writes episode metrics, transcripts, images, a manifest, and GPU samples.

### 4. Core screening

```bash
bash scripts/run_experiment_matrix.sh core-screening
```

Default: all three trained core methods on both Sokoban and Navigation, seed 0,
50 updates. This phase is for fatal instability, parity, memory, throughput,
reward-variance, and coarse success screening—not final claims.

### 5. Episode-objective screening

```bash
bash scripts/run_experiment_matrix.sh episode-screening
```

Default: all 3×3 reward/objective combinations, Sokoban, seed 0, 50 updates. Rank configurations by:

1. parity pass and absence of non-finite loss;
2. visual success;
3. successful mean turns;
4. stability across validation points;
5. GPU-hours.

Do not select on training reward alone.

### 6. Confirmatory runs

Only screening winners receive 2–3 seeds:

```bash
SELECTED_METHODS=no_concat_episode_grpo \
CONFIRMATORY_SEEDS=0,1,2 \
  bash scripts/run_experiment_matrix.sh confirmatory
```

To confirm all core comparisons after resource review:

```bash
SELECTED_METHODS=concat_grpo,no_concat_gae,no_concat_episode_grpo \
CONFIRMATORY_SEEDS=0,1,2 \
  bash scripts/run_experiment_matrix.sh confirmatory
```

The second command is substantially more expensive, especially the critic-bearing condition. It should not be started before screening.

## State-relative P2 gate

State-relative optimization is not part of the core matrix until real base-policy no-concat rows pass:

```bash
bash scripts/run_experiment_matrix.sh state-preflight \
  exps/<base-rollout-dir>/*.jsonl
```

Default pass criteria:

- at least 64 unique turn rows;
- missing-anchor fraction at most 1%;
- at least 20% of anchored rows in state groups of size two or more;
- diverse actions in at least 10% of comparable groups;
- mean within-state return-to-go variance at least `1e-4`.

A stop decision is a valid negative result. It blocks implementation/training of that method for the environment and checkpoint that produced the rows.

## Anti-cheating and failure analysis

For a selected checkpoint:

```bash
EVAL_MODEL_PATH=/absolute/path/to/hf_model \
  bash scripts/run_experiment_matrix.sh anti-cheat
```

The three visual conditions are:

- `none`: normal images;
- `remove`: no image is sent to the model;
- `shuffle_tiles`: deterministic 5×5 spatial tile permutation.

Interpretation:

- unchanged success under removal is evidence that the benchmark, prompt, or policy may be solvable without current pixels;
- a small tile-shuffle drop but large removal drop suggests coarse visual statistics rather than spatial reasoning;
- both controls must use identical task seeds and deterministic decoding.

After runs:

```bash
bash scripts/run_experiment_matrix.sh analyze \
  --run exps/vlm_agent_rl/<training-run> \
  --eval-dump exps/eval/<checkpoint>/<condition>
```

The analyzer reports:

- answer and full-response template concentration;
- unique-template fraction;
- invalid-action fraction;
- within-group trajectory reward variance and zero-variance fraction;
- reward/turn correlation and reward gained per extra turn;
- success and successful mean turns by visual ablation;
- representative failed episode transcripts.

High reward/turn correlation is not automatically reward hacking, but it triggers manual review of successful long trajectories and the reward-mode ablation.

## Metrics and hard gates

### Primary behavioral metrics

- visual task success;
- mean turns among successful trajectories;
- success by held-out seed.

### Optimization diagnostics

- group reward variance and zero-variance groups;
- actor/critic loss and gradient norms;
- response lengths;
- invalid actions and format correctness;
- template concentration.

### Rollout/training parity

Required report:

- ratio mean;
- ratio median;
- ratio P95;
- ratio P99;
- mean absolute log-probability delta;
- pre-update clip fraction;
- number of valid action tokens.

Default abort thresholds are recorded in the run manifest/config and in `parity.json`.
The report retains an append-only `attempts` history inside the run directory.
If any resumed attempt fails the gate, result analysis keeps the run failed even
when a later retry passes; use a new run directory for a clean replacement run.
The entry point skips runs already classified complete and refuses to reuse a
parity-failed directory.

### Resources

`scripts/run_with_gpu_metrics.py` polls only devices selected by
`CUDA_VISIBLE_DEVICES` (or all devices when it is unset):

- peak VRAM by device and overall;
- wall-clock duration;
- GPU-hours = wall-clock hours × visible GPU count;
- utilization samples;
- trapezoidal power-based energy estimate.

GPU-hours describe occupied devices, not normalized accelerator-equivalent compute.
The wrapper checks the sampled device count against training `N_GPUS` or
evaluation `DP_SIZE × TP_SIZE` before launching.
Resumed accounting requires the same device inventory, and a sampling error
keeps the result incomplete rather than silently understating peak memory.
If GPU sampling never succeeds, GPU-hours and peak VRAM remain null rather than
being reported as zero.

## Result schema

The required first six columns are:

```text
Method | Visual Success | Peak VRAM | GPU·h | Mean Turns | Ratio P95
```

The machine-readable table adds status, environment, seed, commit, and evidence paths. Until CUDA runs exist, [results/main_results.csv](results/main_results.csv) contains null metrics and `pending-external-gpu`.

Every non-null row must be traceable to:

- parent and verl commits;
- `manifest.json`;
- exact `train_command.sh`/`eval_command.sh`, resolved train/eval config, and seed;
- raw rollout or episode files;
- `parity.json` for trained methods;
- `gpu_summary.json`;
- local W&B directory or synced run.

The analyzer marks a row `complete` only when the declared episode count (or
final training step), clean source provenance, both commits, nonempty replay
command and resolved config, valid GPU samples, and—where applicable—a passed
parity gate are all present. Any model/environment error episode marks the
evaluation failed rather than turning infrastructure failure into a behavioral
zero. The evaluator also exits nonzero after persisting such episodes, so the
experiment matrix cannot silently advance. Partial or failed runs remain
explicitly labeled.

## Completed CPU results

Run: `results/cpu/20260727-mac-arm64`, code commit `d2dcd3f3c274254167c6e0177e02b3ad63d8cc8b`, deterministic seed `20260727`.

### Critic mask dynamics

- ignored value: `0.5 → 0.5` with mask;
- legacy counterfactual: `0.5 → -87.7815`;
- supervised value: `-1.0 → 1.9654`, target `2.0`;
- 20 SGD updates.

### Sokoban turn-splitting control

- 20 generated rooms, shortest solution length 3–5;
- both packed and split action sequences solved every room;
- raw split-minus-packed reward: mean `0.245`, range `0.20–0.30`;
- positive delta: 20/20;
- mean delta after outcome, bounded-process, and format-gate trajectory reduction: `0.0`.

### Objective mass control

For two synthetic trajectories with 10 and 4 action tokens:

- token objective mass: `0.714 / 0.286`;
- turn objective mass: `0.667 / 0.333`;
- trajectory objective mass: `0.500 / 0.500`.

These values verify the estimands; they are not training results.

## Results not yet available

The current workstation has no NVIDIA GPU. Therefore the following remain unmeasured:

- local Qwen2.5-VL visual success;
- first-update real-model ratio statistics;
- peak VRAM and GPU-hours;
- trained method comparisons;
- state-relative base-policy preflight;
- image-ablation deltas and model failure cases;
- confirmatory seed variance.

No placeholder value in the repository should be interpreted as zero.
