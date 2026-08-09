# 短上下文多轮 VLM Agent 强化学习后训练｜个人项目

基于 Qwen2.5-VL-3B，在视觉 Sokoban 与部分可观测 Navigation 上研究多轮视觉交互的轨迹级信用分配，完成算法设计、单卡 LoRA 强化学习后训练与冻结模型独立测试。

- 逐轮短上下文训练会把优势的统计单元从轨迹退化成单个 turn。设计 critic-free 的 no-concat episode GRPO：按 `(group, trajectory, turn)` 去重数据并行 padding 并重建完整 episode，在轨迹粒度归约奖励、做组内标准化后广播回 action token，使模型不拼接历史也能按长期回报优化；配套实现 token / turn / trajectory 三种策略目标归一化。
- 修复 no-concat GAE 的稀疏 Critic 监督：估计器改名后 `value_mask` 不再挂载，critic worker 又在重建 batch 时丢弃该字段，使本应忽略的位置被训练向 `-100` 哨兵值；修复后被屏蔽位置在 20 步优化中稳定停在 `0.500`，旧路径漂移到 `-87.782`。另用 20 个 seed 的等最短路对照量化奖励的回合长度偏差（`+0.245`）并以轨迹级归约消除。
- 建立训练与评测的一致性防线：首次更新前比对 vLLM 异步 rollout 与训练 forward 的 log probability，超阈值先落盘证据再终止，拦截 processor、图像 token 与 position ID 不一致；并修复评测默认使用完整历史、导致 no-concat 模型被错配测试的问题，使上下文协议由训练方法推出并在结果聚合时强制校验。
- 定位 Sokoban 的划分漏洞：环境按难度重试换 seed，10,000 个训练 seed 实际只生成 3,902 个棋盘，seed 互斥不等于棋盘互斥。改为按棋盘指纹重建 held-out 集合，得到 128 个训练与验证中均未出现的测试棋盘。在此划分下完成 3 方法 × 2 环境 × 3 训练 seeds × 401 updates 后训练：episode GRPO 在两环境合并 384 / 90 个测试 episode 上取得 **47.9% / 32.2%** 成功率，较基础模型 14.8% / 6.7% 提升 33 / 26 个百分点，高于 concat GRPO 的 45.1% / 27.8% 与修复后 GAE 的 41.9% / 25.6%；成功轨迹平均回合 3.0 / 6.5，峰值显存 45,500 MiB，低于带 critic 的 GAE 路径 47,500 MiB。
