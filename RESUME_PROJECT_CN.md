# 短上下文多轮 VLM Agent 强化学习后训练｜个人项目

基于 Qwen2.5-VL-3B，研究视觉 Sokoban 与部分可观测 Navigation 中的多轮信用分配，完成单卡 LoRA 强化学习后训练与冻结模型独立评测。

- 针对逐轮短上下文使组相对目标退化为 turn 级统计的问题，设计 critic-free episode GRPO；按 `(group, trajectory, turn)` 重建完整轨迹，在 episode 粒度归约奖励和标准化优势，再回传至 action token，并支持 token、turn、trajectory 三种策略目标权重。
- 排查 no-concat GAE 的稀疏 Critic 监督链路，修复 `value_mask` 在优势估计器和 worker 间的两处断点；同时在首次更新前校验 vLLM rollout 与训练 forward 的 log probability，提前拦截图像 token、position ID 与 processor 不一致。
- 发现 Sokoban requested seed 与真实棋盘并非一一对应，改用棋盘指纹构造两两互斥的 train/validation/test；将评测上下文协议与训练方法绑定，并加入图像移除和 5×5 空间块打乱，检验模型对当前视觉观测及空间布局的依赖。
- 在 10,000 个 Sokoban 训练 seed、1,200 个 Navigation 任务上完成 3 方法 × 2 环境 × 3 训练 seed × 401 updates 对照。episode GRPO 成功率为 **47.9% / 32.2%**，较基础模型 14.8% / 6.7% 提升 33 / 26 个百分点，高于 concat GRPO 的 45.1% / 27.8% 与修复后 GAE 的 41.9% / 25.6%；成功轨迹平均回合为 3.0 / 6.5，峰值显存 45,500 MiB。
