# 短上下文多轮 VLM Agent 强化学习后训练｜个人项目

基于 Qwen2.5-VL-3B，在视觉 Sokoban 与部分可观测 Navigation 上研究多轮视觉交互的轨迹级信用分配，完成算法设计、单卡 LoRA 强化学习后训练与冻结模型独立测试。

- 逐轮训练把一条轨迹拆成互不相连的短上下文样本，优势的统计单元退化为单个 turn。设计 critic-free 的 no-concat episode GRPO：按 `(group, trajectory, turn)` 去重数据并行 padding，重建完整 episode，校验回合连续、终止标记唯一与每组 `rollout.n` 轨迹完整，再在轨迹粒度归约奖励、做组内标准化并广播回 action token，使模型不拼接完整历史也能按长期任务回报优化；实现 token / turn / trajectory 三种策略目标归一化权重，让损失权重的分配对 padding 重复行和 micro-batch 切分都保持不变。
- 排查 no-concat GAE 的稀疏 Critic 监督链路：优势估计器改名后 `value_mask` 不再挂载，critic worker 重建 batch 时又丢弃这个可选字段，导致本应忽略的位置被训练向 `-100` 哨兵值。修复两处断点后，20 步优化中被屏蔽位置稳定停在 `0.500`，旧路径漂移到 `-87.782`，受监督位置收敛到 `1.965`（目标 `2.0`）。同时用 20 个 seed 的等最短路对照量化环境奖励的回合长度偏差（原始奖励平均 `+0.245`），并用三种轨迹奖励归约把该偏差消除到 `0.000`。
- 在首次参数更新前比较 vLLM 异步 rollout 与训练 forward 的 log probability，统计 ratio 分位数、平均绝对 log-prob 偏差和 clip 比例，超阈值时先落盘证据再终止训练，用于在长训练开始前拦截 processor、图像 token 与 position ID 层面的不一致；评测侧加入确定性的图像移除与 5×5 空间块打乱两种消融，检验成功率依赖当前视觉观测而不是提示模板与任务先验。
- 在 Sokoban 10,000 训练 seeds、Navigation 1,200 训练任务上完成 3 方法 × 2 环境 × 3 训练 seeds × 401 updates 的后训练，并对每个冻结 checkpoint 在互斥的 held-out 集合上各评测一次。episode GRPO 成功率为 **48% / 32%**，相对基础模型 15% / 8% 提升 33 / 24 个百分点，高于 concat GRPO 的 45% / 28% 与修复后 no-concat GAE 的 42% / 25%；成功轨迹平均回合降至 3.0 / 6.5，单卡峰值显存 45,500 MiB，低于带 critic 的 GAE 路径 47,500 MiB。
