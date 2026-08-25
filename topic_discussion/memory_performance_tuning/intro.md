# 显存优化专题

## 专题定位

本专题研究训练和推理中的显存对象、生命周期与预算取舍，最终通过固定 workload、真实 GPU 测量和 profiling 形成可复现的优化决策。

它不是单独讲某个技巧，而是回答四个问题：显存被什么占用、压力出现在哪个阶段、优化把代价转移到了哪里、当前方案是否值得采用。

训练侧重点是参数、梯度、optimizer state、activation 和临时张量；推理侧重点是权重、KV Cache、请求并发和临时 attention 空间。两者共享 dtype、内存层级、带宽和 profiling 基础，但项目证据不能混用。

## 如何开始

推荐从 Part02 的 [2.5 反向传播与显存优化](../../02_PyTorch_Algorithms/2_5.md) 进入；需要补通用训练计算图时先看 [Part00 07 Autograd](../../00_Prerequisites/07_PyTorch_Autograd_and_Backward.ipynb)，需要补 GPU 内存层级时回看 [03 GPU Architecture and Memory](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.ipynb)。

核心训练项目链为：

```text
73 训练基线 → 76 策略比较 → 75 预算决策 → 74 Profiling 收口
```

## 73–76 实验注意事项

这四节必须按同一条 workload 口径串起来，不能把不同模型、序列长度或精度的结果直接横向比较。

- **73 先建基线**：固定模型、`batch_size`、`seq_len`、dtype、optimizer、warmup、iters 和 seed；先记录 step time、吞吐、`peak_memory`、`peak_reserved`、loss / eval loss 与 OOM 状态。当前 12GB 设备的主线是 Qwen2.5-0.5B、batch 1、seq_len 768、FP32、AdamW；BF16 / seq_len 1024 是容量扩展，不替代主线基线。
- **76 再比策略**：在与 73 相同的 workload 下比较 `baseline`、`checkpoint`、`offload`、`hybrid`。同时查看显存、吞吐、质量和状态，不能只看峰值显存；策略只要 OOM 或超过质量门槛，就不能进入可行集合。
- **75 做预算敏感性**：直接读取 76 的 JSON，不重新训练；至少改变显存上限或最低吞吐，观察可行策略、最佳候选和 `accept / tune / reject` 是否稳定。当前设备可用于说明 9600 与 11200 MB 预算下的边界，但不能据此推断更大模型的普遍收益。
- **74 最后看原因**：74 需要在相同主线 workload 上采集真实 profiler trace，检查 checkpoint 重算、offload 搬运和 optimizer step 的时间代价。没有 trace 时只能报告证据缺口，不能把 73–76 的汇总表写成 profiling 结论。

当前设备的显存空间有限，FP32 的更长序列可能直接 OOM；这不是实验失败，而是需要记录的容量边界。若要研究更高 activation 压力，应改用 BF16、LoRA / QLoRA、分块 loss 或 activation-only benchmark，并在报告中明确标记为扩展 workload。所有结果都应保留配置和 JSON 文件，避免只复制终端中的单次数字。

没有 GPU 时，可以完成 Task0–2 的机制与报告结构；峰值显存、吞吐、OOM 边界和策略收益需要 GPU。推理缓存和量化属于按需进入的分支，不是训练主线的硬性前置。

## Task0–6 路线

Task0–3 是训练侧主线，Task4–5 是推理显存与量化分支，Task6 负责证据收口。专题正文 `01–06` 用来解释和串联，不替代 Notebook 或项目报告。

### Task0 → Task1 → Task2：从机制到策略

这三步不是把几门课简单串在一起，而是分别回答三个问题：

```text
Task0：为什么 backward 可能需要保存 activation？
  ↓
Task1：这些状态分别占用什么资源，如何测量？
  ↓
Task2：确认压力来源后，哪种策略值得比较？
```

Task1 是共享的资源与证据层，不是额外的硬件课程。它只要求学习者能够区分参数、梯度、optimizer state、activation、KV Cache 和临时张量，并理解 dtype、容量、带宽和 profiling 指标之间的关系；ZeRO、分布式显存和复杂通信属于后续扩展。Task1 本身不输出“某策略一定省了多少显存”的结论，真实收益要交给 73–76 的固定 workload 项目验证。

| Task | 目标 | 核心入口 | 扩展入口 | 主要边界 |
|:---|:---|:---|:---|:---|
| Task0 | 理解训练计算图为什么可能保存 activation | [07 Autograd and Backward](../../00_Prerequisites/07_PyTorch_Autograd_and_Backward.ipynb) → [18 Activation / Loss Backward](../../02_PyTorch_Algorithms/18_Activation_and_Loss_Backward.ipynb) | [17 Attention Backward](../../02_PyTorch_Algorithms/17_Autograd_Basics.ipynb) | 只讲训练机制，不讨论 KV Cache，也不输出真实 GPU 收益；17 是 Attention backward 的进阶扩展，不是通用 Autograd 入门 |
| Task1 | 建立训练与推理共享的资源和测量语言 | [01 Data Types](../../01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.ipynb) → [02 Params / FLOPs](../../01_Hardware_Math_and_Systems/02_LLM_Params_and_FLOPs.ipynb) → [03 GPU Memory](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.ipynb) → [06 VRAM / ZeRO](../../01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.ipynb) | 共享支撑：[04 Attention](../../01_Hardware_Math_and_Systems/04_Attention_Memory_Optimization.ipynb)、[12 Mixed Precision](../../01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.ipynb)、[14 FlashAttention](../../01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.ipynb)；证据出口：[13 Profiling](../../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.ipynb) | 负责对象、规模、容量和硬件代价，不直接决定策略 |
| Task2 | 比较训练侧 accumulation、checkpoint 和 offload 的机制 | [12 Gradient Accumulation](../../02_PyTorch_Algorithms/12_Gradient_Accumulation.ipynb) → [19 Activation Checkpointing](../../02_PyTorch_Algorithms/19_Activation_Checkpointing_and_Activation_Offload.ipynb) | [42 Activation Offload](../../02_PyTorch_Algorithms/42_Activation_Offload.ipynb) | 只建立训练策略假设，不替代真实 benchmark |
| Task3 | 在固定 workload 下验证训练侧策略 | [73 基线](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.ipynb) → [76 策略比较](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.ipynb) → [75 预算决策](../../02_PyTorch_Algorithms/75_Memory_Budget_Compression_Project.ipynb) | 高压力 workload、offload / hybrid 和严格预算 | 只负责训练侧实测与预算决策，不覆盖推理 backend 或量化部署 |
| Task4 | 理解推理侧 KV Cache 和缓存管理 | [14 FlashAttention Memory Model](../../01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.ipynb) → [11 KV Cache](../../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.ipynb) → [22 PagedAttention](../../02_PyTorch_Algorithms/22_vLLM_PagedAttention.ipynb) → [34 Prefix Caching](../../02_PyTorch_Algorithms/34_Prefix_Caching_and_Chunked_Prefill.ipynb) | [24 RadixAttention](../../02_PyTorch_Algorithms/24_SGLang_RadixAttention.ipynb)、[37 KV Cache Scheduling](../../02_PyTorch_Algorithms/37_KV_Cache_Scheduling.ipynb)、[66 backend](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb) | 不解释训练 activation；真实 backend、并发和多方案比较属于扩展 |
| Task5 | 把量化作为显存压缩手段进行评估 | [21 Quantization Theory](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.ipynb) → [25 W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.ipynb) | [40 GPTQ / AWQ](../../02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.ipynb)、[41 FP8 / KV Cache](../../02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.ipynb)、[67 Deployment](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb) | 权重文件变小不等于端到端收益，速度和质量仍需验证 |
| Task6 | 汇总显存、时间、质量和 profiler 证据 | [74 Profiling Driven Optimization](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.ipynb) | [43 Unified Memory](../../02_PyTorch_Algorithms/43_Unified_Memory_Management.ipynb)、[44 Auto Tuning](../../02_PyTorch_Algorithms/44_Auto_Tuning_Framework.ipynb)、[45 Memory Cut Planning](../../02_PyTorch_Algorithms/45_Memory_Cut_Planning.ipynb) | 没有真实 trace 时只能报告证据缺口，不能写成完整优化结论 |

Task1 的主线是 `dtype → 参数规模 → GPU 硬件代价 → 显存状态账本`；`04 / 12 / 14` 是共享支撑，`13 Profiling` 是后续测量出口，不属于 Task1 的核心顺序。显存路线关注对象驻留、容量和带宽代价，其他路线在同一 Notebook 上切换观察目标，完整差异由各专题正文说明。

## 证据与环境等级

| 等级 | 环境 | 可以形成的结论 |
|:---|:---|:---|
| 机制验证 | `CPU-first` | 公式、shape、梯度、生命周期和决策逻辑 |
| 单 GPU 项目 | `GPU required` | 峰值显存、吞吐、OOM 边界和固定 workload 下的策略比较 |
| 高级扩展 | GPU、backend 或多卡 | profiler trace、服务并发、通信和部署结论 |

不要把“代码运行成功”写成“显存优化成功”。CPU 或 toy 结果只能说明机制；单次 GPU 运行只能说明当前环境观察；稳定决策至少需要固定 workload、baseline / candidate、质量门槛和报告文件。

## 共享小节如何使用

Part00 / Part01 只提供共享机制和测量语言，不需要重新学习全部内容。详细的前置映射、14–16 的分支关系以及 01、03、06、11、12、13、14 的阅读问题，见[深入阅读](./walkthrough.md)。

同一 Notebook 在不同路线中只切换观察目标：显存路线关注对象账本和峰值，推理路线关注 KV Cache、TTFT / TPOT 和并发，算子与编译路线关注 kernel、访存和融合，训练微调路线关注 loss、梯度和稳定性。Notebook 只保留一份权威内容，路线正文负责提出不同问题。

## 项目产出与延伸入口

训练侧样板项目为 [73](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.ipynb) → [76](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.ipynb) → [75](../../02_PyTorch_Algorithms/75_Memory_Budget_Compression_Project.ipynb) → [74](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.ipynb)。

- 需要判断表、预算门槛和策略分流：阅读[显存优化与性能调优正文](./casebook.md)。
- 需要理解前置小节如何衔接：阅读[显存优化与性能调优深入阅读](./walkthrough.md)。
- 需要研究请求链路速度：进入[推理优化专题](../inference_optimization/intro.md)。
- 需要研究低比特压缩：进入[量化与压缩专题](../quantization/intro.md)。
- 需要研究 profiler 证据：进入[Profiling 专题](../profiling/intro.md)。
- 需要研究多卡切分和通信：进入[通信与并行专题](../communication_parallel/intro.md)。

分布式显存与系统级预算属于高级扩展，连接 Part01 的通信、异构调度和并行策略，以及 79–81 项目，不作为 Task0–6 的单机主线前置。

## 环境与验证

基础机制可以 CPU-first；真实训练、显存峰值和策略对比需要 NVIDIA GPU。运行前确认 PyTorch CUDA 可用，并按 Notebook 输出保存 JSON。项目运行顺序、GPU 检查、结果文件和 74 profiling 要求见[73–76 显存优化项目验证清单](../../docs/verification/memory_projects.md)。
