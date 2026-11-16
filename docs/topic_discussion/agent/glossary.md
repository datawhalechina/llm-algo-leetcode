# 智能体基础设施专题术语表

本表解释多步任务、工具执行和训练反馈中容易混淆的概念。全教程统一写法见[术语表与中英文写作规范](../../terminology.md)。

| 中文主称 | 英文 / 缩写 | 在本专题中的含义 | 容易混淆的概念 | 学习入口 |
|:---|:---|:---|:---|:---|
| 智能体任务 | agent task | 具有目标、初始状态、工具集合、成功条件和终止条件的多步工作负载 | 单次文本生成 | [00 工作负载与执行链](./00_agent_workload_and_trace.md) |
| 动作 | action | 模型在某一步选择的回复、工具调用或终止操作 | 工具已经执行 | [00 工作负载与执行链](./00_agent_workload_and_trace.md) |
| 观察 | observation | 工具或环境执行动作后返回的结构化反馈 | 模型对结果的猜测 | [01 轨迹数据与验证](./01_agent_trajectory_data.md) |
| 轨迹 | trajectory | 从初始状态开始，按顺序保存动作、观察、状态变化和终态的记录 | 只保存最终回答 | [01 轨迹数据与验证](./01_agent_trajectory_data.md) |
| 工具契约 | tool contract | 工具名称、参数 schema、返回格式、失败状态和副作用约束 | 具体接入协议 | [02 沙盒与执行环境](./02_agent_sandbox_environment.md) |
| 沙盒 | sandbox | 限制权限、网络、文件、资源和执行时间的受控环境 | 普通 Python 虚拟环境 | [02 沙盒与执行环境](./02_agent_sandbox_environment.md) |
| 环境快照 | environment snapshot | 能够复现任务初始状态、依赖和可见资源的记录 | 只有包版本清单 | [02 沙盒与执行环境](./02_agent_sandbox_environment.md) |
| 工具副作用 | tool side effect | 工具调用对文件、数据库、服务或外部状态产生的变化 | 工具返回的文本 | [02 沙盒与执行环境](./02_agent_sandbox_environment.md) |
| 任务成功率 | task success rate | 在冻结任务和评分协议下达到终态成功条件的任务比例 | 奖励均值或回答质量 | [智能体任务问题链](./walkthrough.md) |
| 轨迹新鲜度 | trajectory freshness | 轨迹策略版本、环境版本、奖励状态和消费状态是否仍满足当前更新条件 | 轨迹生成时间越新越好 | [后训练优化 · 轨迹生命周期](../post_training_optimization/08_trajectory_lifecycle_and_environment_interaction.md) |
| Agentic RL | agentic reinforcement learning | 策略在工具或环境中执行多步动作，并依据环境反馈优化任务行为 | 一种固定的策略优化算法 | [Part 05 · Agentic RL 扩展](../../05_Integrated_Projects_and_Decisions/02_Post_Training_Delivery_Project.md#小模型-agentic-rl-扩展) |
| MCP | Model Context Protocol | 工具与上下文的一种标准化接入协议 | 工具能力本身或 Agent 框架 | [00 工作负载与执行链](./00_agent_workload_and_trace.md) |

MCP、API、JSON、CPU、GPU 和稳定框架名称可保留原文；普通叙述使用中文主称。工具接入协议、编排框架和策略更新算法属于不同层次，不能互相替代。
