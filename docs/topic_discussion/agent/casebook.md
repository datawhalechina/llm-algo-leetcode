# 智能体任务判断手册

使用本表前先固定任务集合、成功条件、最大步数、模型版本、工具版本、沙盒规则和评测协议。每条记录至少包含任务 ID、轨迹 ID、步骤、动作、工具参数、环境观察、状态变化、耗时和终止原因。

从可观察现象开始，先采集“最小证据”，再进入相应机制；不要用最终回答文本猜测工具是否真实执行。

| 现象或目标 | 先检查什么 | 最小证据 | 机制入口 | 验证或转交入口 |
|:---|:---|:---|:---|:---|
| 最终回答合理，但外部状态没有变化 | 工具是否执行、返回状态和副作用 | 工具调用、退出状态、执行日志与状态差异 | [00 工作负载与执行链](./00_agent_workload_and_trace.md) | [02 沙盒与执行环境](./02_agent_sandbox_environment.md) |
| 工具名称正确但参数反复无效 | 工具契约、参数类型和观察反馈 | 工具 schema、实际参数、校验错误与重试动作 | [02 沙盒与执行环境](./02_agent_sandbox_environment.md) | [01 轨迹数据与验证](./01_agent_trajectory_data.md) |
| 任务陷入重复调用或无效循环 | 状态更新、终止条件和上下文内容 | 连续步骤、观察摘要、重复动作与步数上限 | [00 工作负载与执行链](./00_agent_workload_and_trace.md) | [推理优化](../inference_optimization/intro.md) |
| 相同任务复测结果不同 | 模型采样、工具版本、环境状态和重置策略 | 随机种子、模型与工具版本、环境快照和初始状态 | [02 沙盒与执行环境](./02_agent_sandbox_environment.md) | [部署与异构](../deployment_heterogeneous/intro.md) |
| 成功轨迹很多，但训练后任务能力没有提升 | 轨迹覆盖、失败样本和训练/评测隔离 | 数据来源、去重结果、失败类型与冻结评测 | [01 轨迹数据与验证](./01_agent_trajectory_data.md) | [数据工程](../data_engineering/intro.md) |
| 完成率提高但延迟或成本失控 | 模型、工具、沙盒和重试的暴露时间 | 同一任务的阶段时间、词元、工具次数与资源用量 | [00 工作负载与执行链](./00_agent_workload_and_trace.md) | [性能优化](../performance_optimization/intro.md) |
| 调度吞吐提高但单任务超时增多 | 就绪条件、公平性和长工具等待 | 队列事件、任务年龄、工具等待与超时分布 | [推理优化](../inference_optimization/intro.md) | [Part 02 · 70 服务调度](../../02_PyTorch_Algorithms/70_Serving_Scheduler_Benchmark.md) |
| 奖励上升但任务成功率没有提高 | 奖励分量与环境终态是否一致 | 奖励分解、终态验证、失败切片和独立评测 | [后训练优化](../post_training_optimization/intro.md) | [Part 02 · PT-REWARD-SIGNAL](../../02_PyTorch_Algorithms/PT-REWARD-SIGNAL_Reward_Model_Value_and_Advantage.md) |
| 旧策略轨迹进入当前更新后结果不稳定 | 策略版本、轨迹新鲜度和消费状态 | 策略哈希、生成版本、奖励版本与消费记录 | [后训练优化 · 轨迹生命周期](../post_training_optimization/08_trajectory_lifecycle_and_environment_interaction.md) | [Part 02 · PT-ROLLOUT-LIFECYCLE](../../02_PyTorch_Algorithms/PT-ROLLOUT-LIFECYCLE_Rollout_and_Policy_Freshness.md) |
| 需要比较 MCP、函数调用或业务 API | 能力、参数约束、失败状态和运行成本 | 同一任务上的工具契约、成功率、延迟与错误分布 | [00 工作负载与执行链](./00_agent_workload_and_trace.md) | [Part 05 综合项目](../../05_Integrated_Projects_and_Decisions/intro.md) |
