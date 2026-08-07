# GPU 保守预测：口径、预算与替换协议

> **状态：保守预测，尚未实测。** `results/main_results_predicted.csv` 的全部数值用于租卡、预算、表结构预演和异常检测，不能作为已取得的实验结果。真实 CUDA 运行完成后，保留预测用于复盘，并将实测结果写入 `results/main_results.csv`。

## 1. 预测表的统计粒度

机器可读表固定保留 8 行中央值：4 种方法状态（base + 3 个训练方法）× 2 个环境。

- Base 行使用一次固定 held-out 评测：Sokoban seeds `[10129, 10256]`，Navigation `base` tasks `[30, 59]`。
- 训练方法行的成功率、成功平均回合和 Ratio P95，表示训练 seeds `{0,1,2}` 的**计划聚合中央值**。
- 训练方法行的 GPU·h 表示**单环境、单训练 seed、401 updates** 的预计占卡时间，不是三个 seeds 的合计。
- 峰值显存表示同配置单卡运行的规划中央值；最终报告取实测运行中的最大值，并附卡型与配置。
- 成功平均回合只统计成功 episode。Sokoban 上限为 5，Navigation 上限为 10。

## 2. 八行保守预测中央值

| 方法 | 环境 | 成功率 | 峰值显存 (MiB) | GPU·h / seed | 成功平均回合 | Ratio P95 |
|---|---|---:|---:|---:|---:|---:|
| Base Qwen2.5-VL-3B | Sokoban | 15% | 42,000 | 0.8 | 3.8 | — |
| concat GRPO | Sokoban | 45% | 46,000 | 12.5 | 3.2 | 0.98 |
| fixed no-concat GAE | Sokoban | 42% | 47,500 | 14.2 | 3.5 | 0.97 |
| no-concat episode GRPO | Sokoban | **48%** | 45,500 | 12.8 | **3.0** | 0.98 |
| Base Qwen2.5-VL-3B | Navigation | 8% | 42,000 | 1.2 | 8.2 | — |
| concat GRPO | Navigation | 28% | 46,000 | 18.5 | 6.8 | 0.97 |
| fixed no-concat GAE | Navigation | 25% | 47,500 | 20.8 | 7.2 | 0.96 |
| no-concat episode GRPO | Navigation | **32%** | 45,500 | 19.2 | **6.5** | 0.97 |

这些圆整数值是实验前假设，不表示 episode GRPO 一定优于 baseline，也不预设差异具有统计显著性。

## 3. 预测依据与自洽检查

### 3.1 成功率与回合数

中央值遵循以下规划假设：

1. 结构化视觉任务经过 RL 后训练后，成功率相对 base 提升，但 Navigation 因部分可观测与渲染开销保持更低的绝对水平。
2. concat GRPO 提供完整轨迹统计；修复后的 no-concat GAE 恢复有效 critic 监督；episode GRPO 在短上下文下恢复 episode 统计单元。
3. 训练后成功 episode 的路径效率改善，但回合数始终受环境上限约束：Sokoban `3.0–3.5 ≤ 5`，Navigation `6.5–7.2 ≤ 10`。

最终结果必须报告三个训练 seeds 的均值、离散程度，以及每个 checkpoint 在同一 held-out 任务集上的结果。只报告 seed 0 不构成 confirmatory 结论。

### 3.2 峰值显存

规划中央值包含模型权重、训练激活、LoRA/优化器状态、rollout 引擎缓存、视觉输入和运行时开销。它们是租卡前的容量边界，不是精确的组件内存分解。

- Base inference：42,000 MiB。
- concat GRPO：46,000 MiB。
- fixed no-concat GAE：47,500 MiB，critic-bearing 路径是 48 GiB 卡的主要风险项。
- no-concat episode GRPO：45,500 MiB，不持有 critic。

48 GiB 设备的名义容量为 49,152 MiB；因此必须先跑真实 smoke。任何 OOM 或峰值接近卡上限的情况都优先调整 batch/rollout 配置或升级卡型，而不是把预测当作保证。

### 3.3 单 run GPU·h 与隐含吞吐

GPU·h 是端到端占卡时间，包含模型加载、warm-up、rollout、训练、validation、环境渲染、写盘和正常关闭。为避免旧文档中“episodes/min 与总时长不相加”的问题，中央值对应的隐含吞吐如下：

| 单 run | GPU·h | 对应时长 | 隐含吞吐 |
|---|---:|---:|---:|
| Sokoban base，128 episodes | 0.8 | 48 min | 约 2.67 episodes/min |
| Navigation base，30 episodes | 1.2 | 72 min | 约 0.42 episodes/min |
| Sokoban concat，401 updates | 12.5 | 750 min | 约 1.87 min/update |
| Sokoban GAE，401 updates | 14.2 | 852 min | 约 2.12 min/update |
| Sokoban episode GRPO，401 updates | 12.8 | 768 min | 约 1.92 min/update |
| Navigation concat，401 updates | 18.5 | 1,110 min | 约 2.77 min/update |
| Navigation GAE，401 updates | 20.8 | 1,248 min | 约 3.11 min/update |
| Navigation episode GRPO，401 updates | 19.2 | 1,152 min | 约 2.87 min/update |

真实运行可因卡型、CPU、数据盘、Unity 渲染和重试出现较大偏差。因此 GPU·h 的首要作用是预算和异常报警，而不是承诺完成时间。

### 3.4 Rollout/train Ratio P95

训练方法的中央值为 `0.96–0.98`。正式门控检查的是 P95 相对 1 的**绝对偏差**：

```text
|ratio_p95 - 1.0| <= 0.10
|ratio_p99 - 1.0| <= 0.20
mean_abs_logprob_delta <= 0.05
pre_update_clip_fraction <= 0.01
```

因此不能把门槛简写成单边的 `P95 < 1.1`。预测通过不代表实际一定通过；真实 gate 失败时必须终止首次更新并保留失败证据。

## 4. 总预算推导

所有 screening 时长按 401-step 中央值线性折算到 50 steps，仅作为租卡预算。

### 4.1 公共前置阶段

```text
smoke                                      = 0.5 GPU·h
base evaluation                            = 0.8 + 1.2 = 2.0 GPU·h
core screening                             = 98.0 × 50 / 401 ≈ 12.2 GPU·h
episode screening                          = 9 × 12.8 × 50 / 401 ≈ 14.4 GPU·h
visual ablation                            = 约 3–6 GPU·h（1–2 个环境）
```

其中 `98.0 = 12.5 + 14.2 + 12.8 + 18.5 + 20.8 + 19.2`，覆盖三个方法和两个环境的单 seed 401-step 中央值。

### 4.2 仅确认 screening 获胜方法

若获胜方法为 no-concat episode GRPO：

```text
confirmatory = 3 seeds × (12.8 + 19.2) = 96.0 GPU·h
independent final test = 3 checkpoints × (0.8 + 1.2) = 6.0 GPU·h
总计 = 0.5 + 2.0 + 12.2 + 14.4 + 96.0 + 6.0 + (3–6)
     ≈ 134–137 GPU·h
```

### 4.3 完整确认三种方法

```text
confirmatory = 3 seeds × [(12.5 + 18.5)
                        + (14.2 + 20.8)
                        + (12.8 + 19.2)]
             = 294.0 GPU·h
independent final test = 3 methods × 3 checkpoints × (0.8 + 1.2)
                       = 18.0 GPU·h
总计 ≈ 344–347 GPU·h
```

Checkpoint export/LoRA merge time is not guessed here; record and add its
measured GPU·h (if any) separately. The final-test term is required because
each frozen training-seed checkpoint receives its own held-out evaluation.

AutoDL 金额使用实际租用时的人民币单价计算：

```text
预计金额（人民币） = 预计 GPU·h × 实例实时单价（元/小时）
```

下载、环境安装和故障重跑是否占用计费实例，应另留余量；仓库不固定历史美元报价。

## 5. 规划容差与调查阈值

下表是工程规划容差，不是置信区间：

| 指标 | 规划容差 | 超出后优先检查 |
|---|---:|---|
| 成功率 | ±10 个百分点 | reward、checkpoint、评测 seed、环境服务 |
| 峰值显存 | ±10% | batch、rollout cache、offload、卡型 |
| 单 run GPU·h | ±30% | 吞吐、重试、Unity、I/O、validation 频率 |
| 成功平均回合 | ±1.5 turns | 成功定义、路径难度、episode 截断 |
| Ratio P95 | ±0.05 | processor、模板、模型版本、position ID |

出现下列情况应停止后续昂贵阶段并调查：

- critic-bearing smoke 在 48 GiB 卡 OOM；
- parity 任一硬门槛失败；
- 单 run GPU·h 超过中央值两倍且没有可解释的重试或硬件差异；
- 成功率低于中央值的一半；
- 数据、manifest、GPU 采样或原始 rollout 不完整。

## 6. 三训练 seed 的最终统计协议

每个训练方法和环境使用 seeds `{0,1,2}`，每个 seed 训练 401 updates，并在完全相同的 held-out 集合上评测。最终表至少报告：

- 三个 seed 的成功率均值与标准差；
- 每个 seed 的二项成功率区间或聚合区间，并明确计算口径；
- 成功 episode 平均回合的均值、标准差和样本量；
- 每个 run 的 GPU·h、三个 seeds 合计 GPU·h、峰值显存最大值；
- 每个 seed 的 Ratio P95/P99 和 gate 状态；
- 与 concat baseline 的差值，但不在显著性不足时使用“显著提升”。

原始 analyzer 结果保持 per-run 粒度；README 与简历只引用聚合表，不把 seed 0 screening 行冒充最终结果。

## 7. 实测替换协议

### 7.1 提取单个运行

```bash
python -m vagen.analysis.analyze_rollouts \
  --run exps/vlm_agent_rl/<run_dir> \
  --output-dir results/gpu/<run_name>
```

只有同时满足以下条件才可标记 `complete`：

- manifest 中包含顶层仓库与 verl commit，且正式运行工作树干净；
- 预期 step 和 validation episode 数量匹配；
- training run 的 parity gate 通过且没有历史失败尝试；
- GPU summary 返回码为 0、采样非空、设备数一致；
- 命令、解析配置和原始行为证据存在。

### 7.2 聚合与替换

1. 保留 `results/main_results_predicted.csv`，用于比较预测误差。
2. 将三个训练 seeds 的原始行保存到 `results/gpu/raw_runs/`。
3. 按 `(Method, Environment)` 聚合 seeds `{0,1,2}`。
4. 将聚合后的实测值写入 `results/main_results.csv`，更新 `Commit` 和 `Evidence`。
5. README 与完成态简历只引用 `results/main_results.csv`。
6. 对偏差超过规划容差的项目记录原因，不为了贴合预测而筛选 seed 或修改口径。

## 8. 预测来源边界

这些中央值综合了模型规模、LoRA/offload 配置、环境回合上限、validation 频率和同类多模态 RL 运行的工程经验。它们没有来自本仓库 CUDA 实测，也不是对任何论文结果的直接复现。可审计证据只有在真实 run 产生 manifest、原始 rollout、parity 和 GPU summary 后成立。
