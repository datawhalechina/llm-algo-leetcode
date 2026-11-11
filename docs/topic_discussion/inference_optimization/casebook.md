# 推理优化判断手册

先声明本轮改变的是单项机制还是配置组合，再固定其余请求、生成和测量条件。记录模型与分词器、输入、到达方式、实际输出长度、停止原因、失败和运行环境；根据现象补充阶段时间、资源与执行路径，组合收益不拆分归因。

从首个检查动作区分计算、状态和等待，再选择机制练习或真实实验。普通解码没有草稿时，不要求接受率；MoE 负载账本也不直接替代 GPU 时间。

| 现象或目标 | 先检查什么 | 最小证据 | 机制入口 | 验证或转交入口 |
|:---|:---|:---|:---|:---|
| 长输入下首词元明显变慢 | 对照排队、预填充与首次返回时间，区分请求等待和层内计算 | 到达／接纳事件、输入长度、阶段时间、层内轨迹 | [01 请求指标](./01_request_path_and_metrics.md)、[02 模型前向](./02_prefill_and_attention_kernel.md) | 计算路径选 [INF-EXECUTION-DISPATCH](../../02_PyTorch_Algorithms/INF-EXECUTION-DISPATCH_Inference_Fusion_and_Backend_Dispatch.md)；真实请求基线选 [Part 02 · 66](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.md) |
| 普通解码每词元等待较长 | 分开本轮前向时间、KV 读取与两轮之间的等待 | 上下文长度、活跃批次、层内时间与客户端间隔 | [02 计算执行](./02_prefill_and_attention_kernel.md)、[05 服务调度](./05_serving_scheduling_and_pd.md) | [INF-ITERATION-BUDGET](../../02_PyTorch_Algorithms/INF-ITERATION-BUDGET_Serving_Iteration_Cost_and_Batch_Bounds.md) 单卡成本实验；服务等待进入 [Part 02 · 70](../../02_PyTorch_Algorithms/70_Serving_Scheduler_Benchmark.md) |
| 投机接受率高，但完整生成没有加速 | 比较每轮提交词元数与草稿、验证、修正的总成本 | 连续接受长度、有效推进量、阶段耗时、实际输出长度与质量 | [03 解码策略](./03_decoding_strategies.md) | [INF-PROPOSAL](../../02_PyTorch_Algorithms/INF-PROPOSAL_Dynamic_Proposal_and_Verification_Budget.md) 机制实验；[Part 02 · 68](../../02_PyTorch_Algorithms/68_Speculative_Decoding_Benchmark.md) 后端复测 |
| 投机生成的确定性结果或采样行为异常 | 先核对贪心／随机规则，再检查接受前缀、拒绝残差与全部接受后的追加分支 | 同一上下文下的目标／草稿概率、首拒绝位置、修正与停止状态 | [03 解码策略](./03_decoding_strategies.md) | [Part 02 · 23](../../02_PyTorch_Algorithms/23_Speculative_Decoding.md) CPU 状态与数值练习；随机采样不要求逐次文本完全相同 |
| MoE 分组后输出与参考不一致 | 对照排序前后的词元、专家与权重对应关系，再核对重复词元的累加 | 分配三元关系、专家区间、加权输出和误差 | [02 专家执行](./02_prefill_and_attention_kernel.md) | [INF-MOE-EXECUTION](../../02_PyTorch_Algorithms/INF-MOE-EXECUTION_MoE_Inference_Dispatch_and_Expert_Execution.md) CPU 分组与恢复测试 |
| MoE 激活专家较少，但完整调用仍慢 | 分开重排、专家计算和合并；观察各专家矩阵行数与实际调用 | 活跃专家、分配数、路径事件、完整调用时间、计划构造成本 | [02 模型前向](./02_prefill_and_attention_kernel.md) | [INF-MOE-EXECUTION](../../02_PyTorch_Algorithms/INF-MOE-EXECUTION_MoE_Inference_Dispatch_and_Expert_Execution.md) 单卡实验入口；实测状态见[资产表](./experiment_assets.md#moe-execution) |
| 并发增加后 TPOT 或尾延迟上升 | 对照请求到达、准入等待和迭代组成，区分过载与单步变慢 | 到达事件、队列、每轮词元、阶段配额、超时与吞吐 | [05 服务预算与调度](./05_serving_scheduling_and_pd.md) | [连续批处理](../../02_PyTorch_Algorithms/INF-CONTINUOUS-BATCHING_Continuous_Batching.md) 机制实验；[Part 02 · 70](../../02_PyTorch_Algorithms/70_Serving_Scheduler_Benchmark.md) 服务复测 |
| 长输入加入后，已有解码请求出现停顿 | 对照整段与分块预填充时的迭代长度、长请求等待与解码间隔 | 同一到达流、预填充块、短请求 TTFT、生成间隔、吞吐 | [05 阶段调度](./05_serving_scheduling_and_pd.md) | [Part 02 · 38](../../02_PyTorch_Algorithms/38_Prefill_Decode_Scheduling.md) 配对实验 |
| KV 持续增长，上下文或并发无法再增加 | 比较逻辑状态增长、物理页占用与结束请求回收 | 缓存长度、KV 头数、块表、引用、抢占与 OOM 位置 | [04 KV 生命周期](./04_kv_cache_lifecycle_and_reuse.md) | [Part 02 · 37](../../02_PyTorch_Algorithms/37_KV_Cache_Lifecycle_and_Capacity_Control.md) 容量与状态实验 |
| 重复前缀很多，但 TTFT 没有下降 | 先确认可复用的连续前缀与实际命中，再比较未命中后缀成本 | 模型与状态身份、命中词元、缓存占用和阶段时间 | [04 前缀复用](./04_kv_cache_lifecycle_and_reuse.md) | [Part 02 · 69](../../02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark.md) 真实服务缓存实验 |
| 文件可以加载，但低精度收益不稳定 | 对照产物格式、实际数据类型和内核，检查回退与转换 | 产物身份、加载日志、轨迹、延迟、峰值与质量 | [06 量化部署](./06_quantized_inference_and_deployment.md) | [Part 02 · 82](../../02_PyTorch_Algorithms/82_Quantization_Artifact_Evaluation_Project.md) 产物检查；[Part 02 · 83](../../02_PyTorch_Algorithms/83_Quantized_Serving_Profile_Benchmark.md) 服务复测 |
| 显存降低，但延迟或回答质量退化 | 对照量化对象、实际长度、停止与有效输出，不只看文件大小 | 相同任务集、TTFT、TPOT、资源、格式与完整性 | [06 低精度执行](./06_quantized_inference_and_deployment.md) | [量化与压缩](../quantization_compression/intro.md)、[07 比较方法](./07_benchmark_and_decision.md) |
| 单卡不满足容量或吞吐 | 先区分请求容量与模型／单请求状态容量，再选择副本或内部切分 | 单实例基线、请求负载、状态容量与拓扑条件 | [08 分布式推理](./08_distributed_inference_and_parallelism.md) | [通信与并行](../communication_parallelism/intro.md)、[Part 02 · 81](../../02_PyTorch_Algorithms/81_Distributed_Inference_Project.md) |
| 多卡 MoE 的最慢参与者拖慢迭代 | 对照专家放置、各设备负载与实际跨卡交换，而非只看平均负载 | 每专家／每设备分配、通信字节、暴露等待与最慢设备时间 | [08 专家并行](./08_distributed_inference_and_parallelism.md) | [Part 02 · 46](../../02_PyTorch_Algorithms/46_MoE_Expert_Parallel.md) 机制练习；[Part 02 · 80](../../02_PyTorch_Algorithms/80_MoE_Expert_Parallel_Benchmark.md) 并行项目 |
| 重复运行的收益方向不稳定 | 对照测量顺序、缓存状态、输入差异和设备负载 | 各轮原始样本、正反顺序、失败与启动／稳态划分 | [07 基准与决策](./07_benchmark_and_decision.md) | [性能分析](../performance_analysis/intro.md)、[性能优化](../performance_optimization/intro.md) |
