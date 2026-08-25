# 显存优化专题

## 专题定位

本专题用于串起显存优化主线：先看训练侧为什么会 OOM，再看推理侧为什么会被 KV cache 顶住预算，最后把 checkpointing、offload、量化和 benchmark 一起收回端到端 trade-off 结论。这里重点关注 `peak memory / VRAM ledger / trade-off`；如果问题先表现为请求链路变慢，应优先转到推理优化专题。

## Infra 层定位

显存优化主要涉及 `L1 硬件与内存层`、`L2 系统软件与访存层`、`L3 框架与运行时`，并延伸到 L4 的 KV Cache、量化和 Serving 调度。评估策略时，至少要同时检查容量、带宽、计算重算、CPU-GPU 搬运和端到端性能，不能只比较峰值显存。

## 同一内容的显存目标

本专题把 checkpointing、offload、paging、KV Cache 和量化看成“资源预算策略”：核心问题是减少哪类状态的驻留，以及代价转移到了重算、带宽、通信还是质量。量化在这里首先是显存工具，要先回答模型是否装得下、上下文或并发是否能提高，再用实际 workload 验证速度和质量。

与推理专题共享同一来源 Notebook 时，所有学习者先理解共同机制，再按目标选择指标：显存目标关注状态账本、peak memory、带宽、并发容量和 OOM 边界，最终输出是显存上限、吞吐下限、质量下限下的 `accept / tune / reject` 决策。

## 推荐入口

推荐从 Part 02 的 [2.5 反向传播与显存优化](../../02_PyTorch_Algorithms/2_5.md) 开始；如果还不熟悉训练状态、显存层级或 dtype，再回补 Task 1 的 Part 01 基础。这样可以先进入训练侧问题，再按需要补齐硬件和账本，而不必把整个 Part 01 顺读一遍。

## 前置阅读

- Part 00：Autograd、反向传播和训练循环基础。
- Part 01：数据类型、参数规模、GPU 架构和显存账本基础。
- Part 02：优先完成 2.5，再进入 73、76 和 75 的项目决策链。

## 主学习线

`Task1-6` 是学习路线，指向 `Part 00 / Part 01 / Part 02` 的具体小节；最后一列的 `01-06` 是专题正文页，只负责解释和串联。

| Task | 学习内容 | 主学习线 | 专题正文 |
|:---|:---|:---|:---|
| Task1 | 显存对象与硬件底座 | `Part 01:01 -> 02 -> 03 -> 06`；共享支撑：`04 / 12 / 14` | [01 VRAM Ledger and Metrics](./01_vram_ledger_and_metrics.md) |
| Task2 | 训练侧显存优化 | `Part 02:12 -> 19 -> 42` | [02 Training Memory Pressure](./02_training_memory_pressure.md) |
| Task3 | 训练侧验证与调优 | `73 -> 76 -> 75 -> 74` | [03 Checkpointing and Offload](./03_checkpointing_and_offload.md) |
| Task4 | 推理侧显存优化 | `Part 01:11 -> 22 -> 24 -> 34 -> 37` | [04 Inference Cache and Memory Budget](./04_inference_cache_and_memory_budget.md) |
| Task5 | 量化作为显存手段 | `Part 01:21 -> 25 -> 26 -> 40 -> 41 -> 67` | [05 Quantization as a Memory Tool](./05_quantization_as_a_memory_tool.md) |
| Task6 | 显存管理、自动调优与 trade-off 收口 | `43 -> 44 -> 45 -> 74 -> 75 -> 76 -> 67` | [06 Benchmark and Trade-off Decision](./06_benchmark_and_tradeoff_decision.md) |

### Task 1 的边界

Task 1 只负责回答三个问题：显存对象是什么、每个对象大约占多少空间、硬件层级为什么会改变访存代价。

- **主线基础**：[01 数据类型与精度](../../01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.md) → [02 参数量与 FLOPs](../../01_Hardware_Math_and_Systems/02_LLM_Params_and_FLOPs.md) → [03 GPU 架构与显存](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.md) → [06 显存计算与 ZeRO](../../01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.md)。
- **共享支撑**：[04 Attention 显存优化](../../01_Hardware_Math_and_Systems/04_Attention_Memory_Optimization.md) 解释推理侧 KV Cache，[12 Tensor Core 与混合精度](../../01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.md)解释 dtype、带宽和吞吐的关系，[14 FlashAttention 显存模型](../../01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.md)解释 Attention 工作集和片上复用。
- **暂不放入 Task 1**：[13 Profiling 与瓶颈分析](../../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)负责测量和定位，属于 Task 3/74 的证据链；[11 KV Cache 与显存增长](../../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md)主要服务推理侧 Task 4。

Task 1 的 CPU 代码只能验证公式、数量级和对象之间的关系，不能证明真实 GPU 峰值显存、HBM 带宽或吞吐收益；这些结论留给 73–76 和 74。

这里的 `04 / 12 / 14` 是共享支撑，不是显存优化路线独占内容：显存路线关注容量、驻留对象和带宽代价，推理路线关注 KV Cache、prefill/decode 和服务吞吐，算子与编译路线关注 tile、融合和 kernel 访存。Notebook 只说明本节机制与证据边界，完整的路线差异由本专题正文组织。

Task 1 完成后，学习者应能把“一个对象占多少空间、总共有多少对象、硬件如何改变访问代价”说清楚；但还不能据此决定 checkpoint、offload 或 paging 是否值得。后者进入 Task 2、Task 4 和真实项目验证。

## 正文与跳转

先按上面的 `Task1-6` 走 notebook 主线；遇到“训练侧和推理侧的预算问题怎么区分”“为什么峰值降了但系统未必更好”时，再回来看对应的专题正文 `01-06`。想看汇总版就进 [显存优化与性能调优正文](./casebook.md)，想按完整故事线走一遍就进 [显存优化与性能调优深入阅读](./walkthrough.md)。

如果问题已经跨到别的专题：
[反向传播与训练机制专题](../backpropagation_training_mechanism/intro.md) 负责训练侧 activation 与 backward 基础，[推理优化专题](../inference_optimization/intro.md) 负责请求链路速度问题，[量化与压缩专题](../quantization/intro.md) 负责低比特压缩路线，[Profiling 专题](../profiling/intro.md) 负责证据链和瓶颈定位，[通信与并行专题](../communication_parallel/intro.md) 负责多卡切分与参数分摊边界。

## 项目结论

推荐顺序为 [73 训练性能分析](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.md) → [76 Activation / Checkpoint / Offload 对比](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.md) → [75 显存预算压缩](../../02_PyTorch_Algorithms/75_Memory_Budget_Compression_Project.md) → [74 Profiling 驱动的端到端优化](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.md)。73 负责建立测量基线，76 负责比较具体策略，75 负责形成预算决策，74 负责用真实 profiler 证据解释收益和代价。

## 73–76 实验注意事项

这四节必须按同一条 workload 口径串起来，不能把不同模型、序列长度或精度的结果直接横向比较。

- **73 先建基线**：固定模型、`batch_size`、`seq_len`、dtype、optimizer、warmup、iters 和 seed；先记录 step time、吞吐、`peak_memory`、`peak_reserved`、loss / eval loss 与 OOM 状态。当前 12GB 设备的主线是 Qwen2.5-0.5B、batch 1、seq_len 768、FP32、AdamW；BF16 / seq_len 1024 是容量扩展，不替代主线基线。
- **76 再比策略**：在与 73 相同的 workload 下比较 `baseline`、`checkpoint`、`offload`、`hybrid`。同时查看显存、吞吐、质量和状态，不能只看峰值显存；策略只要 OOM 或超过质量门槛，就不能进入可行集合。
- **75 做预算敏感性**：直接读取 76 的 JSON，不重新训练；至少改变显存上限或最低吞吐，观察可行策略、最佳候选和 `accept / tune / reject` 是否稳定。当前设备可用于说明 9600 与 11200 MB 预算下的边界，但不能据此推断更大模型的普遍收益。
- **74 最后看原因**：74 需要在相同主线 workload 上采集真实 profiler trace，检查 checkpoint 重算、offload 搬运和 optimizer step 的时间代价。没有 trace 时只能报告证据缺口，不能把 73–76 的汇总表写成 profiling 结论。

当前设备的显存空间有限，FP32 的更长序列可能直接 OOM；这不是实验失败，而是需要记录的容量边界。若要研究更高 activation 压力，应改用 BF16、LoRA / QLoRA、分块 loss 或 activation-only benchmark，并在报告中明确标记为扩展 workload。所有结果都应保留配置和 JSON 文件，避免只复制终端中的单次数字。

## 环境与验证

基础机制可 CPU-first；真实训练、显存峰值和策略对比需要 NVIDIA GPU。运行前确认 PyTorch CUDA 可用，并按 Notebook 输出保存 JSON 结果，不能只根据单次峰值变化判定优化成功。
