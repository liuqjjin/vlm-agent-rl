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

## Environments and held-out seeds

### Sokoban

- visual 6×6 rooms, one box;
- train config: `examples/train/sokoban/train_sokoban_vision.yaml`;
- validation config: `examples/train/sokoban/val_sokoban_vision.yaml`;
- standalone base evaluation: seeds `[10000, 10128)`;
- success and successful-trajectory mean turns are primary behavioral metrics.

### Navigation

- egocentric RGB AI2-THOR, partially observable;
- base split only for the first controlled comparison;
- train seeds `[0, 30)`, validation seeds `[30, 60)`;
- standalone base evaluation covers seeds `[0, 60)`;
- the remote protocol transports canonical pose anchors but never exposes them to the model.

The same held-out seed sets are reused across methods.

## Funnel

### 1. CPU correctness

Required before CUDA:

```bash
conda run -n vagen bash scripts/run_smoke.sh
```

This covers trajectory reconstruction, incomplete groups, duplicates, zero variance, critic masks, 20-step dynamics, objective weights, microbatch invariance, parity metrics/reporting, processor guard, deterministic environment seeding, state anchors, remote transport, observation ablation, GPU metric parsing, and rollout analysis.

Current result: **58 passed**.

### 2. GPU smoke

```bash
bash scripts/run_experiment_matrix.sh smoke
```

Order:

1. FrozenLake Qwen2.5-VL-3B local SGLang visual evaluation;
2. Sokoban Qwen2.5-VL-3B local SGLang visual evaluation;
3. five no-concat episode-GRPO updates.

The default OpenAI Sokoban evaluator is not accepted as the local baseline.

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

Default: three trained core methods, Sokoban, seed 0, 50 updates. This phase is for fatal instability, parity, memory, throughput, reward-variance, and coarse success screening—not final claims.

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

### Resources

`scripts/run_with_gpu_metrics.py` polls every visible device:

- peak VRAM by device and overall;
- wall-clock duration;
- GPU-hours = wall-clock hours × visible GPU count;
- utilization samples;
- trapezoidal power-based energy estimate.

GPU-hours describe occupied devices, not normalized accelerator-equivalent compute.

## Result schema

The required first six columns are:

```text
Method | Visual Success | Peak VRAM | GPU·h | Mean Turns | Ratio P95
```

The machine-readable table adds status, environment, seed, commit, and evidence paths. Until CUDA runs exist, [results/main_results.csv](results/main_results.csv) contains null metrics and `pending-external-gpu`.

Every non-null row must be traceable to:

- parent and verl commits;
- `manifest.json`;
- exact train/eval config and seed;
- raw rollout or episode files;
- `parity.json` for trained methods;
- `gpu_summary.json`;
- local W&B directory or synced run.

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
