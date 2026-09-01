# 显存优化（Memory Optimization）

> 专题类型：主学习路线　主服务目标：显存预算与资源取舍

## 页面导语

本专题研究训练和推理中的显存对象、生命周期与预算取舍，回答显存被什么占用、压力出现在哪个阶段、优化代价转移到哪里，以及当前方案是否值得采用。

训练侧关注参数、梯度、optimizer state、activation 和临时张量；推理侧关注权重、KV Cache、请求并发和临时 attention 空间。两者共享 dtype、内存层级、带宽和 Profiling 基础，但项目证据不能混用。

## 适合谁

适合希望理解训练或推理显存占用，并在有限硬件上做资源取舍的学习者。主线从显存对象、账本和单机训练策略开始；CPU 可完成机制学习，GPU、backend 和多卡只在进入相应项目时需要。

## 如何开始

推荐从 Part02 的 [2.5 反向传播与显存优化](../../02_PyTorch_Algorithms/2_5.md) 进入；需要补通用训练计算图时先看 [Part00 07 Autograd](../../00_Prerequisites/07_PyTorch_Autograd_and_Backward.md)，需要补 GPU 内存层级时回看 [03 GPU Architecture and Memory](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.md)。

## 主学习线与分级

Task0–3 建立单机显存的共同语言、策略和训练侧证据；Task4–5 进入推理显存与量化扩展；Task6 进入分布式显存和 Profiling 收口。每个 Task 都按“机制 → 策略 → 验证出口”组织，但扩展内容不要求全部作为主线前置。

整体学习顺序是：

```text
显存对象与生命周期
  ↓
dtype、模型规模与显存账本
  ↓
单机训练显存策略
  ↓
训练测量、架构影响与预算决策
  ↓
推理 KV Cache 与服务显存
  ↓
量化容量扩展
  ↓
分布式显存与 Profiling
```

| Task | 学习内容 | 主学习线 / 项目入口 | 学习顺序 | 专题正文 |
|:---|:---|:---|:---|:---|
| Task0 | 显存对象与生命周期 | [Part 00 · 07 Autograd](../../00_Prerequisites/07_PyTorch_Autograd_and_Backward.md) → [Part 02 · 17 Autograd Basics](../../02_PyTorch_Algorithms/17_Autograd_Basics.md) → [Part 02 · 18 Activation / Loss Backward](../../02_PyTorch_Algorithms/18_Activation_and_Loss_Backward.md)；CPU 检查 saved tensors、梯度和 activation 生命周期 | 计算图 → backward → 状态驻留与释放 | [02 训练侧显存压力](./02_training_memory_pressure.md) |
| Task1 | dtype、模型规模、架构与显存账本 | [01 数据类型](../../01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.md) → [02 参数量与 FLOPs](../../01_Hardware_Math_and_Systems/02_LLM_Params_and_FLOPs.md) → [03 GPU 架构与显存](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.md) → [04 Attention](../../02_PyTorch_Algorithms/04_Attention_MHA_GQA.md) → [05 LLaMA3 Block](../../02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.md) → [06 显存计算与 ZeRO](../../01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.md)；CPU 输出参数、dtype、activation 和 optimizer state 账本 | dtype → 参数规模 → Attention / Block → GPU 条件 → 显存账本 | [01 显存账本与指标](./01_vram_ledger_and_metrics.md)；04 / 05 只建立结构到显存的映射，深入架构优化属于扩展 |
| Task2 | 单机训练显存策略 | [12 梯度累积](../../02_PyTorch_Algorithms/12_Gradient_Accumulation.md) → [19 激活检查点](../../02_PyTorch_Algorithms/19_Activation_Checkpointing_and_Activation_Offload.md) → [42 激活 Offload](../../02_PyTorch_Algorithms/42_Activation_Offload.md)；CPU 检查逻辑、梯度对齐和状态变化 | micro-step → 重算 → CPU-GPU 搬运 | [03 Checkpoint 与 Offload](./03_checkpointing_and_offload.md) |
| Task3 | 训练侧测量与预算决策 | [73 训练性能分析](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.md) → [76 策略对比](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.md) → [75 预算压缩](../../02_PyTorch_Algorithms/75_Memory_Budget_Compression_Project.md)；固定 workload、指标和质量门槛后输出 accept / tune / reject；架构扩展 [08](../../02_PyTorch_Algorithms/08_Architecture_Tricks.md) → [06](../../02_PyTorch_Algorithms/06_MoE_Router.md) → [07](../../02_PyTorch_Algorithms/07_MoE_Load_Balancing_Loss.md) → [61](../../02_PyTorch_Algorithms/61_Model_Architecture_Exploration.md) | 测量口径 → baseline → 策略比较 → 预算敏感性；架构扩展用于比较结构变化 | [06 基准测试与权衡决策](./06_benchmark_and_tradeoff_decision.md) |
| Task4 | 推理侧显存与 KV Cache | [11 KV Cache 增长](../../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md) → [22 PagedAttention](../../02_PyTorch_Algorithms/22_vLLM_PagedAttention.md) → [34 Prefix Cache](../../02_PyTorch_Algorithms/34_Prefix_Caching_and_Chunked_Prefill.md) → [41 KV Cache 量化](../../02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.md)；项目 [66 推理 baseline](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.md)、[69 Prefix Cache](../../02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark.md)；扩展 [71 MLA / KV Cache](../../02_PyTorch_Algorithms/71_MLA_KV_Cache_Architecture_Benchmark.md) | 增长 → 分页 → 复用 → cache dtype → 容量验证 → 架构扩展 | [04 推理 Cache 与显存预算](./04_inference_cache_and_memory_budget.md) |
| Task5 | 量化与显存容量扩展 | [21 量化理论](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.md) → [25 W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.md) → [40 GPTQ / AWQ](../../02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.md) → 项目 [67 量化部署](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.md) | 表示 → 权重格式 → 量化算法 → backend → 显存 / 质量验证 | [05 量化作为显存工具](./05_quantization_as_a_memory_tool.md) |
| Task6 | 分布式显存与 Profiling 收口 | 分布式：[27 ZeRO](../../02_PyTorch_Algorithms/27_ZeRO_Optimizer_Sim.md) → [28 Pipeline](../../02_PyTorch_Algorithms/28_Pipeline_Parallelism_MicroBatch.md) → [29 Tensor Parallel](../../02_PyTorch_Algorithms/29_Tensor_Parallelism_Sim.md) → [79 分布式并行](../../02_PyTorch_Algorithms/79_Distributed_Parallel_Benchmark.md) / [80 MoE 专家并行](../../02_PyTorch_Algorithms/80_MoE_Expert_Parallel_Benchmark.md) / [81 分布式推理](../../02_PyTorch_Algorithms/81_Distributed_Inference_Project.md)；Profiling：[13](../../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md) → [74](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.md) | 切分与通信、trace 归因、端到端决策；两条分支并列，不要求先学分布式再学 Profiling | [06 基准测试与权衡决策](./06_benchmark_and_tradeoff_decision.md) |

Task1 建立显存账本，Task3 将 Task0–2 的机制转化为训练测量和预算决策；`61` 是架构扩展项目，`71` 属于推理显存分支。

### 核心与扩展分级

核心路径是 Task0–3：建立显存对象、账本和训练侧策略，再用 73、76、75 完成测量、比较和预算决策。Task4–6 是按需进入的高级分支，分别进入推理 KV Cache、量化、分布式和 Profiling；`08 / 06 / 07 / 61` 是架构扩展，不是共同前置。

### CPU / GPU 证据边界

| Task | CPU 可以完成 | GPU 或真实 backend 才能确认 |
|:---|:---|:---|
| Task0 | backward、activation 生命周期、梯度正确性 | 本 Task 不要求 GPU |
| Task1 | dtype、参数、Attention / Block 规模，以及梯度、optimizer state 和 activation 账本 | 实际峰值、allocator reserved、带宽和 OOM 边界 |
| Task2 | accumulation、checkpoint、offload 的逻辑和梯度对齐 | 显存节省、重算 / 搬运代价、吞吐和 OOM |
| Task3 | workload、warmup、同步、重复运行、峰值显存和策略约束的测量逻辑 | 73 / 76 / 75 的真实 GPU baseline、策略比较和预算决策 |
| Task4 | KV Cache shape、容量估算、分页、prefix sharing 和 KV Cache dtype 的逻辑 | backend 的 cache 命中、并发、TTFT / TPOT 和服务显存 |
| Task5 | 量化误差、字节数和预算决策 | 量化格式加载、kernel、真实显存、吞吐和任务质量 |
| Task6 | ZeRO、pipeline、tensor、expert parallel 的切分模拟；指标聚合和决策逻辑 | 多卡显存分摊、通信时间、拓扑影响、profiler trace 以及 kernel / 重算 / 搬运归因 |

CPU 运行可以使用 GPU 机器，但 `device='cpu'` 的结果仍只能归入 CPU 证据；不能因为运行环境有 GPU，就把 CPU 计算写成 GPU 实测。

## 项目验证与报告

73、76、75 必须使用匹配的模型、dtype、batch、seq_len 和 workload；74 作为 Profiling 收口，不能把不同条件下的数字直接横向比较。

| 项目 | 实验职责 | 关键约束 |
|---|---|---|
| 73 | 建立训练 baseline，记录 step time、吞吐、显存、loss / eval loss 和 OOM | 固定模型、dtype、optimizer、warmup、iters、seed；BF16 / 长序列属于扩展 workload |
| 76 | 在相同 workload 下比较 `baseline`、`checkpoint`、`offload`、`hybrid` | 同时检查显存、吞吐、质量和状态；OOM 或超过质量门槛不能进入可行集合 |
| 75 | 读取 76 的 JSON，改变显存上限或吞吐下限，观察预算决策是否稳定 | 不重新训练；输出 `accept / tune / reject`，9600 / 11200 MB 只代表当前设备边界 |
| 74 | 用真实 profiler trace 解释重算、搬运、optimizer step 和端到端代价 | 没有 trace 时只能报告证据缺口，不能写成完整 Profiling 结论 |

当前设备的显存空间有限，FP32 的更长序列可能直接 OOM；这不是实验失败，而是需要记录的容量边界。若要研究更高 activation 压力，应改用 BF16、LoRA / QLoRA、分块 loss 或 activation-only benchmark，并在报告中明确标记为扩展 workload。所有结果都应保留配置和 JSON 文件，避免只复制终端中的单次数字。

## 跨专题复用与边界

### 复用规则与证据等级

同一 Notebook 在不同路线中只切换观察目标：显存路线关注对象账本、峰值和容量，推理路线关注 KV Cache、TTFT / TPOT 和并发，算子与编译路线关注 kernel、访存和融合，训练微调路线关注 loss、梯度和稳定性。Notebook 只保留一份权威内容，路线正文负责提出不同问题；不同模型、设备、dtype 和 workload 的结果不能直接合并。

### 项目报告与数据索引

| 项目 | 主要证据 | 当前报告 |
|:---|:---|:---|
| 73 | 固定 workload 下的训练 baseline、step time、吞吐、峰值显存、loss 和 OOM | [73 GPU 训练报告](../../benchmarks/results/73_real_gpu_training.json) |
| 76 | baseline / checkpoint / offload / hybrid 的显存、吞吐和质量对比 | [76 显存策略报告](../../benchmarks/results/76_real_gpu_memory.json) |
| 75 | 读取 76 报告后的显存预算与吞吐门槛敏感性 | [75 预算决策报告](../../benchmarks/results/75_memory_budget_decision_9600.json) |
| 74 | profiler trace 汇总，以及重算、搬运和端到端时间归因 | [74 性能分析报告](../../benchmarks/results/74_profiling_optimization.json)；[GPU 性能分析报告](../../benchmarks/results/74_real_gpu_profile.json) |

这些报告只代表各自记录的模型、设备、dtype 和 workload。扩展实验应新增报告文件，不覆盖主线报告；没有匹配的 baseline 或真实 trace 时，只能标记为待验证。

### 共享项目的观察边界

| 项目 | 主责路线 | 本专题复用的观察目标 |
|:---|:---|:---|
| 66 | 推理优化 | 只复用 KV Cache、峰值显存和并发容量，不把推理吞吐结论改写成显存结论 |
| 67 | 推理优化 / 量化与压缩 | 关注权重或 KV Cache 是否压进预算；格式、kernel 和服务收益仍以对应专题为主 |
| 71 | 推理优化 | 关注 MLA 对 KV Cache 表示和容量的影响，不替代训练侧显存项目 |
| 74 | 显存优化 | 复用 Profiling 方法检查重算、搬运和 kernel 代价；Profiling 专题负责方法解释，不重新选择显存策略 |
| 79–81 | 通信与并行 | 关注切分后的显存分摊和通信代价；多卡服务与扩展效率仍由并行专题负责 |

| 等级 | 环境 | 可以形成的结论 |
|:---|:---|:---|
| 机制验证 | `CPU-first` | 公式、shape、梯度、生命周期和决策逻辑 |
| 单 GPU 项目 | `GPU required` | 峰值显存、吞吐、OOM 边界和固定 workload 下的策略比较 |
| 高级扩展 | GPU、backend 或多卡 | profiler trace、服务并发、通信和部署结论 |

不要把“代码运行成功”写成“显存优化成功”。CPU 或 toy 结果只能说明机制；单次 GPU 运行只能说明当前环境观察；稳定决策至少需要固定 workload、baseline / candidate、质量门槛和报告文件。

### 进一步阅读

Part00 / Part01 只提供共享机制和测量语言，不需要重复学习全部内容。需要理解前置映射和阅读顺序，查看[显存优化与性能调优深入阅读](./walkthrough.md)；需要判断表、预算门槛和策略分流，查看[显存优化与性能调优正文](./casebook.md)。

如果问题首先表现为请求链路速度，进入[推理优化](../inference_optimization/intro.md)；如果重点是低比特压缩，进入[量化与压缩](../quantization/intro.md)；如果需要 profiler 证据，进入[性能分析](../profiling/intro.md)；如果涉及多卡切分和通信，进入[通信与并行](../communication_parallel/intro.md)。

## 环境与证据边界

基础机制可以 CPU-first；真实训练、显存峰值和策略对比需要 NVIDIA GPU。运行前确认 PyTorch CUDA 可用，并按 Notebook 输出保存 JSON。项目运行顺序、GPU 检查、结果文件和 74 profiling 要求见[73–76 显存优化项目验证清单](../../verification/memory_projects.md)。
