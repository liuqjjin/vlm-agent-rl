# 实验协议

本文定义正式实验的比较对象、数据划分、执行阶段、统计粒度与证据要求。声明式配置 [experiments/matrix.yaml](experiments/matrix.yaml) 是方法、环境、seed、steps 和硬门槛的机器可读真值源；若本文与矩阵冲突，应先停止运行并同步二者。

当前代码与交付状态见 [PROJECT_STATUS.md](PROJECT_STATUS.md)。GPU 预算和 8 行规划中央值见 [PREDICTED_METRICS.md](PREDICTED_METRICS.md)。

## 1. 研究问题

1. 短上下文 no-concat 训练能否在视觉多轮任务上保留完整轨迹的奖励归因语义？
2. 修复稀疏 critic 监督后，no-concat GAE 是否构成稳定且公平的 baseline？
3. critic-free、trajectory-aware 的 group-relative 目标能否改善成功率、路径效率或资源效率？
4. token、turn、trajectory 三种策略目标在响应长度和轨迹长度变化时有何差异？
5. 成功率多大程度依赖当前视觉观测，而不是提示模板、任务先验或奖励长度偏差？

核心实验不以“训练奖励更高”作为最终结论；held-out 成功率、成功 episode 平均回合、三训练 seed 稳定性和证据完整性共同决定结果是否可报告。

## 2. 比较方法与贡献边界

| 方法 | 上下文行 | 优势统计单元 | Critic | 默认 `rollout.n` | 来源定位 |
|---|---|---|---:|---:|---|
| Base Qwen2.5-VL-3B | evaluation only | 不适用 | 否 | 不适用 | 统一零样本基线 |
| concat GRPO | 完整轨迹 | trajectory/group | 否 | 4 | 上游 baseline |
| fixed no-concat GAE | 单个 turn | 重构后的时序 GAE | 是 | 1 | 既有路径 + 本项目 mask 修复 |
| no-concat episode GRPO | 单个 turn | 重构后的 trajectory/group | 否 | 4 | 本项目核心算法设计 |

本项目比较的是**三条完整 recipe**，不是只隔离一个变量的严格消融。上游提供分布式训练与 rollout 基础设施；本项目负责轨迹重构、稀疏监督修复、episode GRPO、策略权重、一致性门控和实验审计扩展。

### 2.1 已知的共变量

下面几项在三条 recipe 之间同时变化。它们不影响每条 recipe 自身的正确性，但决定了结论的措辞：可以说“这条 recipe 整体更好”，不能说“差异只来自信用分配方式”。

| 共变量 | 差异 | 位置 |
|---|---|---|
| 每个 global step 的 optimizer step 数 | mini-batch 按 `rollout.n` 放大后逐个执行 optimizer step；no-concat 把一条轨迹展开成多行，每步的 Adam 更新次数随平均回合数增加 | `verl/workers/fsdp_workers.py`、`verl/workers/actor/dp_actor.py` |
| 奖励目标 | concat GRPO 用逐 token 奖励求和；no-concat GAE 用逐 turn shaping reward；episode GRPO 默认 `outcome` 只看成败 | `vagen/custom_advantage/` |
| 组内标准差口径 | episode GRPO 用 population std（`unbiased=False`），上游 concat GRPO 用样本 std；`rollout.n=4` 时优势尺度相差约 15.5% | `no_concat_episode_grpo.py`、`verl/trainer/ppo/core_algos.py` |
| 每回合响应预算 | no-concat 固定 512 token；concat 为 4000（Sokoban）/ 10000（Navigation） | `scripts/run_training_method.sh` |

`policy_weights` 保证的是同一个 optimizer step 内、跨 micro-batch 切分的等价性，不消除上表第一行的差异。

报告结果时至少同时给出 global steps、optimizer steps、environment trajectories 和 generated action tokens 四种口径，读者才能判断计算量是否可比。若要把差异归因到单一机制，需要先统一每个 global step 的 optimizer step 数、奖励定义和标准差口径，再单独跑一次因果消融。

episode GRPO screening 覆盖：

```text
reward_mode ∈ {outcome, bounded_process, format_gate}
loss_weighting ∈ {token, turn, trajectory}
```

正式协议限制为单 GPU。跨 rank 的 normalized policy weight 等价性未验证，因此不得将当前 episode GRPO 描述为已验证的多 GPU 算法。

## 3. 数据划分

### 3.1 Sokoban

Sokoban 的 seed **不是**任务身份。`PatchedSokobanEnv.reset` 会在生成的房间不满足 `min_solution_steps` 时继续换 seed 重试，因此不同的 requested seed 可能收敛到同一个棋盘：10,000 个训练 seed 只产生 3,902 个不同棋盘。仅按 seed 区间划分会让大量测试棋盘在训练中出现过。

正式划分因此按**生成后的棋盘**进行。[`vagen/analysis/sokoban_board_split.py`](vagen/analysis/sokoban_board_split.py) 对每个 requested seed 实际生成的 `(room_fixed, room_state)` 取指纹，依次选择与 train 互斥的 validation 棋盘，以及与 train/validation 都互斥的测试棋盘；结果提交在 [`experiments/sokoban_board_split.json`](experiments/sokoban_board_split.json)，运行配置直接引用其中的 `seed_list`。

| 用途 | 配置 | Requested seeds | Episode 数 | 不同棋盘 |
|---|---|---|---:|---:|
| Train | `examples/train/sokoban/train_sokoban_vision.yaml` | `[1, 10000]` | 10,000 | 3,902 |
| Validation | `examples/train/sokoban/val_sokoban_vision.yaml` | 枚举 `seed_list`（`10004`–`10579`） | 128 | 128 |
| Final test | `examples/evaluate/sokoban/config.yaml` | 枚举 `seed_list`（`20003`–`20655`） | 128 | 128 |

Validation 和 final test 各包含 128 个唯一棋盘，train/validation/test 三份集合两两互斥。三份配置使用同一个 `min_solution_steps: [1,5]` 难度窗口，因此 held-out 的差异是棋盘身份，不是题目难度。

重新生成与校验：

```bash
python -m vagen.analysis.sokoban_board_split build --output experiments/sokoban_board_split.json
python -m vagen.analysis.sokoban_board_split verify --split experiments/sokoban_board_split.json --sample 8
```

### 3.2 Navigation

Navigation 的 seed 是**各自数据文件内部的任务索引**，不能只比较数字是否相同而忽略 `eval_set`。

| 用途 | `eval_set` | Tasks | 数量 |
|---|---|---:|---:|
| Train | `base_train` | `[0, 1199]` | 1,200 |
| Validation | `base` | `[0, 29]` | 30 |
| Final test | `base` | `[30, 59]` | 30 |

`base_train.json` 与 `base.json` 使用互斥的 AI2-THOR scene 集；validation 和 final test 又在 `base` 内互斥。因此这是面向未见场景的 held-out 评测。数据划分测试必须同时读取 `eval_set` 与 index range，不能把 `base_train:30` 和 `base:30` 当成同一任务。

### 3.3 训练 seed

Confirmatory training seeds 固定为 `{0,1,2}`。它们控制 Python hash 和训练数据顺序等已声明来源；异步推理和 CUDA kernel 不保证 bitwise deterministic。

每个训练 seed 的 checkpoint 都在完全相同的 held-out 任务集上评测。不能把环境 task seed 与 training seed 混为同一列含义。

## 4. 正式不变量

正式运行必须满足：

- 模型为 `Qwen/Qwen2.5-VL-3B-Instruct`；
- reward-variance filter 关闭；
- rollout/train parity gate 开启；
- concat/no-concat、critic 和 `rollout.n` 与方法定义一致；
- **评测的上下文协议必须与训练一致**：no-concat 训练出的 checkpoint 只能在 no-concat 协议下评测，concat 与 base 用完整历史。该值由 `EVAL_METHOD` 推出、写入 evaluation manifest、参与 resume 身份；聚合器在训练与评测的 `concat_multi_turn` 不一致时拒绝发布该结果；
- Sokoban 的 final test 使用 `experiments/sokoban_board_split.json` 校验过的棋盘级划分；
- episode GRPO 使用单 GPU；
- 工作树干净，顶层与 verl commit 均写入 manifest；
- Qwen3-VL 在 processor、M-RoPE position ID 与 parity 验证前 fail-closed；
- state-relative 方法在真实 base rollout 通过 preflight 前不进入核心矩阵。

## 5. 阶段化实验流程

### Phase 0：CPU 正确性

```bash
conda run -n vagen bash scripts/run_smoke.sh
```

该门槛覆盖轨迹重构、稀疏 mask、奖励归约、策略权重、parity 指标、seed/data split、观测消融、GPU 指标解析和结果分析。通过数量不在协议中写死；只有命令退出码为 0 且 [PROJECT_STATUS.md](PROJECT_STATUS.md) 已记录验收，才可进入 GPU 阶段。

### Phase 1：GPU smoke

```bash
bash scripts/run_experiment_matrix.sh smoke
```

用途：验证本地视觉推理、三个核心方法的真实更新路径、首次 parity、critic-bearing 显存风险与基础环境服务。Smoke 是配置和正确性门槛，不产生行为优劣结论。

### Phase 2：Base evaluation

```bash
bash scripts/run_experiment_matrix.sh base-eval
```

在两个环境各运行一次正式 held-out 零样本评测，共 2 runs。评测应写出 manifest、resolved config、episode metrics/transcript、GPU samples 和环境身份；evaluation run 不要求 training parity 或 W&B。

### Phase 3：Core screening

```bash
bash scripts/run_experiment_matrix.sh core-screening
```

默认规模：3 methods × 2 environments × seed 0 × 50 updates，共 6 runs。用于排除 parity、OOM、非有限 loss、吞吐和明显无学习信号的配置；不得把 seed 0 screening 当作最终性能。

### Phase 4：Episode objective screening

```bash
bash scripts/run_experiment_matrix.sh episode-screening
```

默认规模：3 reward modes × 3 loss weightings × Sokoban × seed 0 × 50 updates，共 9 runs。筛选顺序：

1. parity 通过且无非有限 loss；
2. validation 成功率；
3. 成功 episode 平均回合；
4. validation 点间稳定性；
5. GPU·h；
6. 同等表现时选择更简单的目标。

### Phase 5：Confirmatory

只确认 screening 获胜方法：

```bash
SELECTED_METHODS=no_concat_episode_grpo \
CONFIRMATORY_SEEDS=0,1,2 \
REWARD_MODE=outcome \
LOSS_WEIGHTING=trajectory \
  bash scripts/run_experiment_matrix.sh confirmatory
```

规模：1 method × 2 environments × 3 training seeds × 401 updates，共 6 runs。计入每个冻结 checkpoint 的独立 final test 后，该路线规划计算量约 134–137 GPU·h（export 另按实测追加）。

预算允许时确认全部核心方法：

```bash
SELECTED_METHODS=concat_grpo,no_concat_gae,no_concat_episode_grpo \
CONFIRMATORY_SEEDS=0,1,2 \
  bash scripts/run_experiment_matrix.sh confirmatory
```

规模：3 methods × 2 environments × 3 training seeds × 401 updates，共 18 runs。计入 18 次独立 final test 后，该路线的规划计算量约 344–347 GPU·h（export 另按实测追加），启动前必须单独确认预算。

### Phase 6：Final test

1. 仅使用 validation 指标选择 checkpoint。
2. 冻结方法、checkpoint 和所有评测参数。
3. 每个训练 seed 的 checkpoint 只在正式 final-test split 上执行一次。
4. 任何依据 test 结果重新选择 checkpoint 或超参数的行为都必须披露，并需要新的未触碰 test split。

### Phase 7：视觉依赖评测

每个环境使用该环境自己的 selected checkpoint：

```bash
EVAL_METHOD=no_concat_episode_grpo \
EVAL_ENVIRONMENT=sokoban EVAL_MODEL_PATH=/absolute/path/to/sokoban_hf_checkpoint \
  bash scripts/run_experiment_matrix.sh anti-cheat

EVAL_METHOD=no_concat_episode_grpo \
EVAL_ENVIRONMENT=navigation EVAL_MODEL_PATH=/absolute/path/to/navigation_hf_checkpoint \
  bash scripts/run_experiment_matrix.sh anti-cheat
```

`EVAL_METHOD` 是必填项：它决定该 checkpoint 用哪种上下文协议评测，缺省会直接报错而不是悄悄退回完整历史。

三个条件为：

- `none`：正常视觉输入；
- `remove`：不向模型发送图像；
- `shuffle_tiles`：确定性 5×5 空间块打乱。

解释边界：

- `none` 明显高于 `remove`，支持“当前图像对决策有贡献”；
- `none` 高于 `shuffle_tiles`，支持“模型对空间布局敏感”；
- 这些控制不能单独证明像素级因果推理，也不能排除所有模板或任务先验；
- 三个条件必须使用完全相同的任务、解码设置和 checkpoint。

## 6. 结果粒度与统计报告

### 6.1 Raw per-run 层

每个训练 run 保留：

- `manifest.json`：commit、method、environment、training seed、steps 和关键超参；
- `train_command.sh` 与 resolved config；
- validation / rollout 原始 JSONL；
- `parity.json` 及 append-only attempts；
- `gpu_metrics/gpu_summary.json` 与原始采样；
- 本地日志、checkpoint 和离线 W&B（若训练入口启用）。

每个 evaluation run 保留 manifest、eval command、resolved config、episode metrics/transcript/image 和 GPU samples。不要要求 evaluation 目录包含 training-only 的 parity 或 W&B。

### 6.2 Final aggregate 层

按 `(Method, Environment)` 聚合三个 training seeds。最终主表至少包含：

- 成功率均值、标准差和评测 episode 数；
- 二项成功率区间，并明确是按 seed 分别计算还是合并 episode；
- 成功 episode 平均回合、标准差和成功样本量；
- 各 seed GPU·h、合计 GPU·h、峰值显存最大值；
- Ratio P95/P99、mean absolute log-prob delta 和 gate 状态；
- 与 concat baseline 的绝对差值和效应量；
- 多重比较或样本量不足时的限制。

不满足统计证据时可写“观察到更高均值”，不能写“显著提升”。

## 7. Parity 与失败闭合

首次更新前必须报告：

- ratio mean / median / P95 / P99；
- mean absolute log-probability delta；
- pre-update clip fraction；
- 有效 action token 数。

默认硬门槛：

```text
|ratio_p95 - 1.0| <= 0.10
|ratio_p99 - 1.0| <= 0.20
mean_abs_logprob_delta <= 0.05
pre_update_clip_fraction <= 0.01
```

任何 attempt 失败都使该 run 目录保持失败；不得在同目录重跑后覆盖负面证据。应修复原因并创建新目录。

## 8. GPU 资源口径

- GPU·h = wall-clock hours × 实际占用 GPU 数；
- 该数表示设备占用，不是跨卡型归一化算力；
- peak VRAM 按选定设备取峰值；
- 设备数、设备身份和采样错误必须写入 summary；
- 无有效采样时 GPU·h 与 peak VRAM 保持空值，不能写 0；
- resumed run 只有在设备清单和 manifest 兼容时才能合并证据。

预算公式和中央值见 [PREDICTED_METRICS.md](PREDICTED_METRICS.md)。AutoDL 金额使用租用时的人民币实时单价，文档不固定历史报价。

## 9. 安全汇总多个运行目录

Analyzer 的每个 `--run` 或 `--eval-dump` 只接收一个路径。多个目录必须构造重复参数：

```bash
run_args=()
for run_dir in exps/vlm_agent_rl/*; do
  [[ -d "${run_dir}" ]] && run_args+=(--run "${run_dir}")
done

eval_args=()
for eval_dir in exps/eval/*/*; do
  [[ -d "${eval_dir}" ]] && eval_args+=(--eval-dump "${eval_dir}")
done

python -m vagen.analysis.analyze_rollouts \
  "${run_args[@]}" \
  "${eval_args[@]}" \
  --output-dir results/gpu/raw_runs
```

不要写 `--run exps/.../*` 后假设 parser 会自动接受多个展开路径，也不要用文本替换命令把 incomplete 机械改为 complete。

## 10. State-relative preflight

State-relative optimization 不属于当前核心矩阵。真实 no-concat base-policy 行应先运行：

```bash
bash scripts/run_experiment_matrix.sh state-preflight \
  exps/<base-rollout-dir>/*.jsonl
```

默认要求：

- 至少 64 个唯一 turn rows；
- missing-anchor fraction ≤ 1%；
- 至少 20% anchored rows 位于样本数 ≥ 2 的可比状态组；
- 至少 10% 可比组具有动作多样性；
- mean within-state return-to-go variance ≥ `1e-4`。

preflight 给出 stop 是有效的负面结果；在信号不可识别时不继续实现或训练该方法。

## 11. 最终可报告条件

只有同时满足以下条件，项目状态才可从“GPU 待执行”更新为“完整实测”：

1. CPU smoke、完整 CPU regression 和 CI focused lint 全部通过；
2. fresh clone 能恢复正确的 verl commit 和本项目改动；
3. base、screening、confirmatory、final test 与视觉消融按协议完成；
4. 三训练 seed 聚合与 per-run 原始结果可互相追溯；
5. parity、GPU accounting、manifest 和行为证据完整；
6. README、`results/main_results.csv` 与完成态简历引用同一份聚合实测结果；
7. 保守预测仍保留，用于披露预测误差而非事后删除。
