# 性能分析正文

性能分析的目标不是找到一张“最热”的图，而是把性能问题从现象推进到可验证的行动：

`问题现象 → 可测指标 → 待验证假设 → 工具证据 → 瓶颈归因 → 对照实验 → 行动决策`

本文负责组织判断框架。具体 Notebook 负责代码和工具操作，项目节负责固定 workload、保存结果和形成结论。

## 先定义问题和测量口径

“变慢”可能指单步时间、端到端延迟、吞吐下降、峰值显存上升或多卡扩展效率下降。不同指标对应不同测量方法，不能用一个 profiler 数字替代全部结论。

| 现象 | 先固定的指标 | 第一条假设 | 下一步 |
|:---|:---|:---|:---|
| 训练 step 变慢 | step time、samples/s、tokens/s | 算子热点、输入管线或同步增加 | 先做固定 workload 计时，再看 trace |
| 推理请求变慢 | TTFT、TPOT、端到端延迟、吞吐 | prefill、decode、排队或 KV Cache 受限 | 分解请求阶段和并发条件 |
| 显存峰值上升 | allocated、reserved、峰值时刻 | activation、workspace、cache 或碎片 | 看 memory timeline / snapshot |
| 多卡扩展不佳 | 单卡基线、扩展效率、collective 时间 | 通信等待、负载不均或同步 | 对比通信 trace 和计算区间 |

开始采集前必须记录模型、输入形状、batch、序列长度、并发、warmup、迭代次数、dtype、硬件、软件版本和随机种子。否则 before / after 的差异无法归因。

## 工具选择服从问题粒度

| 观察目标 | 首选工具 | 主要证据 | 何时升级 |
|:---|:---|:---|:---|
| 总耗时、吞吐和显存 | `time.perf_counter()`、`torch.cuda.Event`、CUDA memory API | 固定 workload 的聚合指标 | 阶段关系无法解释时进入 profiler |
| 算子热点和训练阶段 | `torch.profiler`、Chrome Trace、TensorBoard | CPU / CUDA 时间、调用次数、shape、memory | 需要 stream、搬运和系统重叠时进入 Nsight Systems |
| GPU 利用率和进程状态 | `nvidia-smi`、框架指标 | 进程显存、利用率、功耗和温度 | 需要 kernel 级指标时进入 Nsight Compute |
| CPU-GPU 重叠和同步 | Nsight Systems | stream、launch、同步、搬运和阶段关系 | 需要具体 kernel 访存时进入 Nsight Compute |
| kernel 计算和访存 | Nsight Compute | occupancy、带宽、Tensor Core、指令和 launch 配置 | 只在热点已定位后使用 |
| 多卡通信 | `torch.profiler`、NCCL trace / debug log、Nsight Systems | collective 时间、等待和 overlap | 需要集群级归因时进入分布式工具 |

工具等级不是证据等级。更高级的工具只会提供更多观察维度，不能修复不一致的 workload 或缺少 baseline 的实验设计。

## 时间、内存和通信要分开归因

同一段时间线可能同时包含计算、内存访问和通信，但它们不是同一个问题：

- 算子时间高，可能是计算量、shape、kernel 选择或硬件利用率问题；
- memory timeline 出现峰值，可能是 activation、临时 workspace、cache 或 allocator 保留；
- 多卡时间变长，可能是在等待 collective，而不是某个算子本身变慢。

应先标记观察事实，再写假设。例如“`aten::matmul` 占 CUDA 时间较高”是观察；“矩阵乘是瓶颈”仍需要 shape、带宽 / 计算利用率和对照实验支持。一个 trace 中出现重叠或排序变化，也不能直接证明某项优化减少了端到端时间。

## 从证据到对照实验

每个优化候选都应有对应的对照：

| 步骤 | 要回答的问题 | 输出 |
|:---|:---|:---|
| baseline | 原始 workload 的稳定基线是什么 | 聚合指标和环境快照 |
| 假设 | 哪个阶段、对象或等待被认为是瓶颈 | 可证伪的描述 |
| 定位 | 工具是否观察到对应热点或状态变化 | trace、表格或 snapshot |
| candidate | 只改变目标变量后，指标如何变化 | 同口径对照结果 |
| 回归 | 质量、稳定性和其他阶段是否退化 | 质量指标、重复结果和失败信息 |

只有“热点变化”和“目标指标改善”同时出现，才可以把归因升级为较强结论。若只看到 profile 形状变化，应保留为待验证假设。

## CPU、GPU 与系统级证据边界

| 环境 | 可以验证 | 不能单独证明 |
|:---|:---|:---|
| CPU | 计时接口、阶段标记、数据结构、聚合逻辑和简单算子关系 | CUDA kernel、GPU 显存、Tensor Core、GPU overlap |
| 单 GPU | CUDA Event、峰值显存、原生 PyTorch trace 和固定 workload | 多卡通信扩展、集群调度和所有 backend 行为 |
| GPU + 目标 backend | serving 阶段、请求指标、实际 kernel 路径和部署资源 | 其他硬件、版本或 workload 下的普遍结论 |
| 多 GPU / 系统工具 | 通信、同步、stream overlap、拓扑和 kernel 细节 | 没有单卡基线时的“扩展效率原因” |

`torch.profiler` 导出的 trace 是观察证据，不是自动生成的因果结论。`nvidia-smi` 适合观察进程级状态，也不能替代 kernel 或请求级测量。

## 项目分工

- `73` 建立训练性能和显存 baseline，固定 workload、环境和重复测量口径；
- `76` 比较 checkpoint、offload、hybrid 等训练侧显存策略及其时间代价；
- `75` 根据候选报告做预算敏感性和 accept / tune / reject 决策；
- `74` 使用真实 profiler trace 对训练侧方案做端到端证据收口；
- `79–81` 扩展到多卡并行、通信等待和分布式推理，不把单卡 trace 当成分布式结论。

这里的 profiling 是证据方法，不取代显存、推理或通信专题的策略设计。它负责确认“问题在哪里、代价是什么、证据够不够”。

## 最小决策模板

每次分析记录：

`现象 → 指标 → 假设 → 工具与配置 → 证据 → baseline / candidate → 质量与回归 → 行动`

- 证据不足：`inspect`，继续取证；
- 已定位但方案收益或代价不稳定：`optimize` / `tune`；
- 固定 workload 下收益稳定且无关键回归：`validate`；
- 方案没有收益、造成回归或无法复现：`revert`。

最终报告必须绑定模型、硬件、软件版本和 workload。没有这些条件，只能描述一次观察，不能宣称完成了通用性能优化。
