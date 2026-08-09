# 项目当前状态

> **这是仓库中唯一的当前状态事实源。** `docs/archive/status-reports/` 下的历史报告均已废弃，只用于保留审计过程，不得据此声称项目已完成。

**状态日期：** 2026-08-09
**当前阶段：** 非 GPU 验收与远端交付已完成，仅待外部 CUDA 实验
**结论：** 单卡正式路径未发现已知非 GPU 阻塞。本文件所在顶层提交与 verl `6c705af7` 推送后，经远端 fresh-clone 复验即可进入 GPU smoke。

## 1. 已完成且有静态/CPU 证据的部分

- 三路受控比较已定义：concat GRPO、修复后的 no-concat GAE、no-concat episode GRPO。
- 轨迹重构、奖励归约、策略权重、log-probability parity、实验矩阵和分析工具已有实现与针对性测试。
- 稀疏 critic mask 的 20 步实验与 Sokoban 奖励长度偏差的 20-seed 原始数据已提交。
- Sokoban 数据划分改为棋盘级：train `[1,10000]` 包含 3,902 个棋盘；validation 与 final test 各枚举 128 个唯一棋盘，三份集合两两互斥。requested seed 不是任务身份，因此 validation/test 均引用 [`experiments/sokoban_board_split.json`](experiments/sokoban_board_split.json) 的显式列表。
- Navigation 已改为 `base_train[0,1199]` 训练、`base[0,29]` validation、`base[30,59]` final test。
- 评测上下文协议由训练方法推出，写入 evaluation manifest 并参与 resume 身份；聚合器在训练/评测协议不一致时拒绝发布。
- GPU 保守预测、三训练 seed 统计计划和预算公式已统一到 [PREDICTED_METRICS.md](PREDICTED_METRICS.md)。

“已有实现”不等于“完整交付已验收”。下面的命令结果决定当前能否进入 GPU。

## 2. 最终非 GPU 验收快照

下列结果来自 `git clone --recurse-submodules --branch main` 得到的干净远端副本，不依赖原工作目录的 editable install。

| Gate | 命令 | 最近结果 | 状态 |
|---|---|---:|---|
| CPU smoke | `conda run -n vagen bash scripts/run_smoke.sh` | 159 passed；1 条依赖 warning；无 NVIDIA GPU，GPU smoke 按设计跳过 | 通过 |
| 完整 CPU regression | `conda run -n vagen python -m pytest vagen/tests verl/tests/trainer/ppo -q` | 324 passed，0 failed；4 条上游/依赖 warning | 通过 |
| CI focused Ruff | 使用 `.github/workflows/cpu-tests.yml` 的 `ruff check` 文件清单 | All checks passed | 通过 |
| 配置与入口 | matrix contract + `DRY_RUN=1` 六组合 | 6 个分区有效（Sokoban validation/evaluation 均为 `explicit_list`）；六组合解析成功 | 通过 |
| 棋盘级划分 | `python -m vagen.analysis.sokoban_board_split verify --split experiments/sokoban_board_split.json --sample 8` | `valid: true`，无问题 | 通过 |
| 远端交付 | fresh clone 顶层 / verl SHA | 本文件所在提交 / `6c705af7`；两级工作树干净 | 通过 |
| GPU 实测 | 外部 Linux NVIDIA 实例 | 未运行 | 待执行 |

上述测试数量只描述这个时间点，不写入最终简历。最终完成态应引用 fresh clone 上的最后一次验收记录，而不是历史数字。

## 3. 剩余工作

剩余工作全部依赖外部 NVIDIA GPU：

1. 在 AutoDL 上运行 GPU smoke，确认 vLLM + LoRA、SGLang 评测、Navigation renderer、parity 与峰值显存。
2. Smoke 通过后按实验漏斗执行 base、screening、confirmatory、独立 final test 和视觉依赖评测。
3. 用实测聚合结果替换 `results/main_results.csv`，并按替换清单生成可投递简历。

Padding sentinel、Actor/Critic 实际 optimizer 路径、LoRA adapter 导出 fail-closed、独立 final-test、结果发布门控、GPU 清单和预算口径均已在本地实现并通过回归测试。

## 4. 非 GPU 最终验收命令

应从 clean worktree 和正确子模块 commit 执行：

```bash
conda run -n vagen bash scripts/run_smoke.sh

conda run -n vagen python -m pytest \
  vagen/tests \
  verl/tests/trainer/ppo \
  -q

bash scripts/run_experiment_matrix.sh describe
DRY_RUN=1 bash scripts/run_experiment_matrix.sh dry-run

bash -n scripts/run_training_method.sh
bash -n scripts/run_visual_eval.sh
bash -n scripts/run_experiment_matrix.sh
bash -n scripts/autodl_bootstrap.sh
```

Ruff 必须使用 [`.github/workflows/cpu-tests.yml`](.github/workflows/cpu-tests.yml) 中相同的 focused file list，避免本地与 CI 口径不同。

## 5. Fresh-clone 交付验收

在临时目录执行，不复用当前 editable install：

```bash
git clone --recurse-submodules <repository-url> <temporary-directory>
cd <temporary-directory>
git submodule status
git status --short
```

验收要求：

- 顶层和 `verl/` 都指向已推送的 commit；
- fresh clone 为 clean；
- `dp_actor.py` 的 policy-weight 依赖可以导入并执行；
- critic worker 保留可选 `value_mask`；
- CPU smoke 与 CI regression 在 fresh clone 环境通过。

## 6. GPU 阶段定义

GPU 阶段按以下顺序进行：

1. GPU smoke；
2. 两环境 base evaluation；
3. 3 方法 × 2 环境 core screening；
4. 3 × 3 episode objective screening；
5. screening 获胜方法的 2 环境 × 3 training seeds × 401 updates confirmatory；
6. validation 选定 checkpoint 后的一次性 final test；
7. 使用各环境 checkpoint 的 none/remove/shuffle_tiles 视觉依赖评测；
8. 三训练 seed 聚合、预测误差复盘和完成态简历替换。

规划计算量：获胜方法路线约 134–137 GPU·h（含 6 次独立 final test）；完整确认三种方法约 344–347 GPU·h（含 18 次 final test）。Checkpoint export/LoRA merge 按实测另加，金额按 AutoDL 实时人民币小时价计算。

## 7. 文档与数据事实源

| 内容 | 唯一事实源 |
|---|---|
| 当前是否完成 | 本文件 |
| 方法、环境、seed、steps、gate | `experiments/matrix.yaml` |
| 正式实验与统计协议 | `EXPERIMENTS.md` |
| GPU 保守预测与预算 | `PREDICTED_METRICS.md`、`results/main_results_predicted.csv` |
| GPU 实测聚合 | `results/main_results.csv`（当前仍为空） |
| CPU 原始证据 | `results/cpu/20260808-mac-arm64/` |
| 完成态简历模板 | `RESUME_PROJECT_CN.md` |
| 历史审计报告 | `docs/archive/status-reports/`（SUPERSEDED） |

## 8. 状态更新规则

- 不手工把 `incomplete` 或 `pending` 文本替换为 `complete`。
- 每次状态变化同时记录顶层 commit、verl commit、命令和实际输出。
- GPU 预测永远保留并显式标记“保守预测”；实测结果写入独立文件。
- fresh clone 的非 GPU gate 已于 2026-08-08 全绿；若实现或依赖变化，必须重新执行。
- 只有三训练 seed GPU 结果、final test、视觉消融和证据链全部完成，才可启用 `RESUME_PROJECT_CN.md` 的完成态数字。
