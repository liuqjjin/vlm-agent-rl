# 多轮视觉 Agent 的轨迹级强化学习后训练｜个人项目

**技术栈：** Qwen2.5-VL-3B｜PyTorch、LoRA、GRPO / PPO / GAE｜VAGEN / verl、Ray / FSDP｜vLLM、SGLang

基于 VAGEN / verl 扩展多轮视觉 Agent 的强化学习训练，在 Sokoban 与部分可观测 Navigation 上研究短上下文 rollout 的轨迹级信用分配。

- no-concat 将轨迹拆成逐轮短上下文后，常规组相对目标会退化到 turn 粒度。开发 critic-free episode GRPO：按 `(group, trajectory, turn)` 去重并还原轨迹，在 episode 粒度归约奖励和计算优势，再回传至 action token；补充 token / turn / trajectory 三种损失权重，处理轨迹长度、padding 与 micro-batch 切分带来的偏差。
- 沿 trainer → worker → loss 链路定位 no-concat GAE 的稀疏 Critic 监督断点，修复 `value_mask` 在估计器分发和 batch 重建时丢失的问题；在首次更新前对齐 vLLM rollout 与训练 forward 的 log-prob，用 ratio 分位数、log-prob 偏差和 clip fraction 拦截图像 token、position ID 与 processor 不一致。
- 复核 Sokoban 任务生成，发现 requested seed 与实际棋盘并非一一对应，改用棋盘指纹生成两两互斥的训练、验证和测试集；冻结 checkpoint 后独立评测，并以图像移除和 5×5 空间块打乱检验视觉信息与空间布局依赖。
- 在 10,000 个 Sokoban 训练 seed、1,200 个 Navigation 任务上完成 3 方法 × 2 环境 × 3 训练 seed × 401 updates 对照。episode GRPO 成功率为 **47.9% / 32.2%**，较基础模型的 14.8% / 6.7% 提高 **33.1 / 25.5 个百分点**，高于 concat GRPO 的 45.1% / 27.8% 和 no-concat GAE 的 41.9% / 25.6%；峰值显存 45,500 MiB，比带 Critic 的 GAE 少 2,000 MiB。
