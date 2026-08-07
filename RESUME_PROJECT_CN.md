# 短上下文多轮 VLM 智能体强化学习后训练

> **使用条件：** 本文是 GPU 阶段达到 [PREDICTED_METRICS.md](PREDICTED_METRICS.md) 中保守预测中央值后的**最终完成态模板**，不是当前项目状态。只有 `results/main_results.csv` 已由三训练 seed 实测聚合结果填充，且 [PROJECT_STATUS.md](PROJECT_STATUS.md) 的全部验收项完成后，才可直接用于投递。若实测不同，必须逐项替换，不能保留更好看的预测值。

**优先岗位：** VLM / 多模态后训练、强化学习算法、AI Agent、训练与推理系统
**可迁移方向：** 图像处理、视觉算法、计算成像与影像算法中的视觉数据流、空间鲁棒性和实验评测

## 一页简历版（4 条）

**短上下文多轮 VLM 智能体强化学习后训练｜个人研究项目**

针对逐轮短上下文训练丢失完整轨迹奖励归因语义的问题，在 VAGEN/verl 既有训练基础设施上构建 concat GRPO、修复后的 no-concat GAE 与 critic-free no-concat episode GRPO 三路受控比较。

- **算法设计：** 设计 no-concat episode GRPO，依据 `(group_idx, traj_idx, turn_idx)` 对逐轮 rollout 去重并重建完整 episode，再执行轨迹奖励归约与组内标准化；实现 token、turn、trajectory 三种策略目标权重，在不拼接完整历史的条件下保持 episode-level 统计单元。
- **正确性修复：** 定位并修复既有 no-concat GAE 中 `value_mask` 激活与 critic worker 传递的两处稀疏监督断链；20 步对照中，被忽略 value 保持 `0.500`，legacy 路径漂移至 `-87.782`。构建首次更新前 rollout/train log-probability 门控，对 P95/P99、绝对 log-prob delta 和 clip fraction 进行 fail-closed 校验。
- **实验结果：** 在视觉 Sokoban 与部分可观测 Navigation 上完成 3 训练 seeds × 401 updates 的 confirmatory 实验；episode GRPO 的 held-out 成功率达到 `48% / 32%`，高于 base `15% / 8%`、concat GRPO `45% / 28%` 和修复后 GAE `42% / 25%`，成功 episode 平均回合为 `3.0 / 6.5`。结果按三 seed 聚合，并保留 per-run manifest、原始 rollout 和统计证据。
- **实验系统：** 组织 3 方法 × 2 环境 core screening、3 奖励模式 × 3 策略目标 screening、两环境三条件视觉依赖评测及每个冻结 checkpoint 的一次性 final test；完整三方法 funnel 占用约 `344–347 GPU·h`，其中 episode GRPO 两环境 confirmatory 为 `96 GPU·h`；其单卡峰值 `45,500 MiB`，低于 critic-bearing GAE 的 `47,500 MiB`。

**核心能力：** 多模态强化学习后训练｜轨迹级信用分配｜参数高效策略优化｜异步采样与训练一致性｜统计评测与实验审计

---

## 面试详版

### 1. 项目定位

多轮视觉语言智能体有两种常见数据组织方式：

- concat：把完整历史拼入当前样本，轨迹语义直接，但视觉与文本上下文随回合累积；
- no-concat：每个 turn 独立训练，上下文短，但一行数据不再等于一条轨迹。

研究核心：当数据拆成逐轮行后，如何重建完整轨迹的统计单元与奖励归因，并保证训练数据链未被 padding 或 processor 差异污染。

### 2. 个人贡献边界

| 范围 | 归属 |
|---|---|
| Ray/FSDP、vLLM 异步 rollout、SGLang 独立评测与环境交互基础设施 | 上游 VAGEN / verl |
| concat GRPO | 上游对照方法 |
| no-concat GAE 初始路径 | 上游既有实现 |
| no-concat GAE 的稀疏 critic mask 修复 | 个人定位并修复 |
| no-concat episode GRPO、轨迹重构、奖励归约、策略权重 | 个人设计与实现 |
| parity gate、实验矩阵、GPU accounting、结果审计和统计分析 | 个人设计与集成 |

个人贡献聚焦于成熟训练栈上的算法、正确性与实验系统扩展。

### 3. 核心算法

#### 3.1 完整轨迹重构

no-concat rollout 中每个 turn 是独立数据行。算法使用 `(group_idx, traj_idx, turn_idx)` 建立身份并执行：

1. 识别数据并行 padding 产生的完整行副本；
2. 检查相同身份的副本内容是否一致，冲突时立即失败；
3. 验证每条轨迹的 turn 从 1 连续增长；
4. 验证恰有一个 terminal marker；
5. 验证每个 group 拥有声明的 `rollout.n` 条完整轨迹；
6. 只有完成上述检查，才计算 episode reward 和 group statistics。

关键思想是把“统计单元”从物理数据行恢复为逻辑轨迹，避免逐轮样本数量、响应长度或 padding 改变方法含义。

#### 3.2 稀疏 Critic 监督修复

既有 no-concat GAE 路径存在两处断链：

- estimator 名称变化后没有同步激活 `value_mask`；
- critic worker 重建 batch 时没有保留可选的 mask 字段。

修复后 critic loss 使用 `response_mask × value_mask`。20 步受控实验中：

- ignored value：修复路径保持 `0.500`，legacy 路径变为 `-87.782`；
- supervised value：从 `-1.0` 收敛至 `1.965`，目标为 `2.0`。

实验同时验证最终状态与优化动态：被忽略位置未收到错误梯度。

#### 3.3 Critic-free episode GRPO

episode GRPO 不训练 value network。对每条重构轨迹先计算单一 episode score，再在同一 group 内标准化，并把轨迹优势广播回该轨迹的 action tokens。

实现包含：

- `outcome`：仅使用最终任务成功；
- `bounded_process`：成功信号叠加有上限的过程奖励；
- `format_gate`：成功前提下约束回合格式质量；
- 零方差 group 保护；
- token / turn / trajectory 三种策略目标归一化。

在保守预测达到后的完成态结果中，episode GRPO 在两环境取得最高观察均值，同时比 critic-bearing GAE 少约 `2,000 MiB` 峰值显存。若最终统计不能支持显著性，简历只写“最高观察均值”，不写“显著提升”。

#### 3.4 奖励长度偏差

Sokoban 原始过程奖励会因同一路径被拆成更多 turn 而增加。20 个确定性 seeds 的对照中，额外奖励均值为 `+0.245`，且 20/20 为正。三种 episode reward 归约在该受控集合中均把同一路径拆分带来的 score 增量压缩至 `0.000`。

这一实验说明为什么不能直接把逐轮奖励求和后用于轨迹组间比较，也为 reward-mode screening 提供了可复现依据。

### 4. 多模态训练正确性

#### 4.1 Rollout/train parity gate

首次 actor update 前比较 rollout engine 与 training forward 对相同 action tokens 的 log probability，并计算：

- ratio mean / median / P95 / P99；
- mean absolute log-probability delta；
- pre-update clip fraction；
- 有效 action token 数。

正式硬门槛为：

```text
|ratio_p95 - 1.0| <= 0.10
|ratio_p99 - 1.0| <= 0.20
mean_abs_logprob_delta <= 0.05
pre_update_clip_fraction <= 0.01
```

失败时先写入 append-only parity report，再终止更新。该门控用于发现 processor、chat template、图像 token、checkpoint 或 position ID 不一致，并提供训练输入链正确性的运行时证据。

#### 4.2 视觉依赖评测

对每个环境使用自己的 selected checkpoint，执行：

- `none`：正常图像；
- `remove`：移除图像；
- `shuffle_tiles`：确定性 5×5 空间块打乱。

`none` 与 `remove` 的差异衡量当前视觉输入贡献；`none` 与 `shuffle_tiles` 的差异衡量空间布局敏感性。

### 5. 实验设计与规模

#### 5.1 数据域

- Sokoban：10,000 train seeds、128 validation seeds、128 final-test seeds，三个集合互斥；
- Navigation：`base_train` 1,200 个训练任务，`base` 中 30 validation + 30 final-test 任务；train 与 held-out 使用互斥 scene 集。

#### 5.2 Funnel

| 阶段 | 规模 | 目的 |
|---|---:|---|
| GPU smoke | 最小推理 + 三方法真实更新 | 配置、parity、OOM gate |
| Base evaluation | 2 environments | 统一零样本基线 |
| Core screening | 3 methods × 2 env × 50 updates | 排除不稳定配置 |
| Episode screening | 3 rewards × 3 objectives × 50 updates | 选择 episode 配置 |
| Confirmatory | 3 methods × 2 env × 3 seeds × 401 updates | 三 seed 同口径比较 |
| Final test | 每 seed checkpoint × 固定 held-out set | 冻结后的最终行为指标 |
| Visual controls | 2 env × 3 conditions | 视觉依赖与空间敏感性 |

获胜方法路线共有 35 个正式训练/评测单元，其中每个冻结 checkpoint 各执行一次 final test；完整确认三种方法为 59 个。Smoke 不进入行为结论，seed 0 screening 不冒充 confirmatory 结果。

### 6. 完成态结果口径

GPU 达到保守预测后的聚合结果：

| 方法 | Sokoban 成功率 / 成功回合 | Navigation 成功率 / 成功回合 | 单 seed GPU·h（Sok / Nav） |
|---|---:|---:|---:|
| Base Qwen2.5-VL-3B | 15% / 3.8 | 8% / 8.2 | 0.8 / 1.2（整次 eval） |
| concat GRPO | 45% / 3.2 | 28% / 6.8 | 12.5 / 18.5 |
| fixed no-concat GAE | 42% / 3.5 | 25% / 7.2 | 14.2 / 20.8 |
| no-concat episode GRPO | **48% / 3.0** | **32% / 6.5** | 12.8 / 19.2 |

成功率与回合数为三训练 seed 聚合；最终投递版还应从真实聚合报告补充标准差或置信区间。GPU·h 是单环境单 seed：episode GRPO confirmatory 合计为 `3 × (12.8 + 19.2) = 96 GPU·h`，三方法完整 funnel 含 18 次独立 final test 后约 `344–347 GPU·h`。

### 7. 工程与复现

每个 training run 保存：

- 顶层仓库与 verl commit、method、environment、training seed 和超参 manifest；
- 实际 command 与 resolved config；
- 原始 validation/rollout JSONL；
- parity attempts；
- GPU 峰值、占卡时长、设备清单、利用率与能耗估计；
- checkpoint、本地日志和离线实验记录。

Analyzer 只有在行为结果、GPU sampling、parity 和 provenance 均完整时才允许 `complete`。负面证据不会被后续重跑覆盖；配置变化使用新 run directory。

### 8. 面试中最值得展开的问题

#### 为什么不能直接按逐轮行计算 GRPO？

因为 group-relative statistics 的样本应是一条完整 rollout，而不是一个 turn。不同轨迹的 turn 数不同，按行计算会让长轨迹在统计中重复出现，并改变均值、方差和策略目标权重。

#### 为什么 no-concat GAE 的 bug 难发现？

训练不会必然 crash；mask 丢失后仍能得到有限 loss，只是 critic 在错误位置被监督。20-step dynamics 比单次 shape/assert 更能揭示这种 silent failure。

#### 为什么需要 parity gate？

rollout engine 与 training forward 可能使用不同 processor、模板或多模态 position IDs。PPO/GRPO 依赖行为策略与训练概率的正确比值，差异会在程序正常运行时破坏优化语义。

#### 为什么只用单 GPU？

episode-normalized policy weights 在跨 rank 聚合下的等价性尚未验证。当前选择 fail-closed 单 GPU，优先保证算法语义；多 GPU 是需要独立证明的后续工作，不作为已完成能力。

#### 为什么结果只写“最高观察均值”？

三训练 seed 能展示稳定性，但小样本下统计功效有限。只有真实区间和检验支持时才写“显著”；否则报告均值、离散程度和效应方向。

### 9. 岗位表达建议

#### VLM / 多模态后训练

重点讲多模态 rollout 与 training forward 的一致性、轨迹级优化、processor/position ID 风险、参数高效训练和视觉依赖评测。

#### 强化学习算法

重点讲统计单元、episode advantage、critic mask、reward reduction、loss weighting、零方差 group 和三 seed 对照。

#### AI Agent

重点讲短上下文多轮交互、环境反馈、轨迹身份、长期奖励归因、invalid action 与代表性失败分析。

#### 训练与推理系统

重点讲异步 rollout、首次更新 gate、失败闭合、manifest、GPU accounting、resume 兼容性和 fresh-clone 复现。

#### 图像处理 / 计算成像 / 影像算法

只强调可迁移能力：视觉输入数据流、空间布局破坏实验、鲁棒性评测、统计设计和工程复现。明确该项目没有实现成像物理、逆问题、重建、分割、配准或医学影像任务；若应聘这些岗位，应搭配一个直接相关项目，而不是把本项目改写成不存在的影像任务。

### 10. 不应写入简历的表述

- “从零搭建 Ray/FSDP/SGLang 分布式训练系统”；
- “独立设计并实现三种强化学习算法”；
- “已验证多 GPU episode GRPO”；
- “视觉消融证明了像素级推理”；
- 没有同硬件、同 checkpoint、同 batch 与 token 口径的固定延迟降幅；
- 当前尚未由真实 GPU artifacts 支持的显存、成功率或 GPU·h；
- 没有最终验收记录支撑的固定测试数量。

---

## 投递前替换清单

1. 确认 [PROJECT_STATUS.md](PROJECT_STATUS.md) 已标记非 GPU 与 GPU gate 全部完成。
2. 确认 `results/main_results.csv` 的 8 行来自真实三训练 seed 聚合，而非预测表复制。
3. 用真实均值、标准差/区间、峰值显存和 GPU·h 替换本文所有完成态中央值。
4. 若 episode GRPO 没有最高均值，按实际排序重写，不保留“最佳”叙述。
5. 若 48 GiB 卡 OOM，删除“可在单张 48 GiB GPU 完成”的表述并记录实际卡型。
6. 若视觉消融没有明显下降，诚实写成 benchmark/policy 视觉依赖不足的负面结果。
7. 保留上游边界，不把基础设施能力写成个人从零实现。
