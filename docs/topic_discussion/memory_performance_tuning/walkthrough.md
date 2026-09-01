# 显存优化与性能调优深入阅读

假设你接手的是同一个系统：训练阶段在中后段开始 OOM，勉强把训练跑完以后，服务侧又发现长上下文和多轮对话会把 KV cache 顶高，最终不是 batch 上不去，就是延迟和吞吐一起变差。

这条线最重要的是按暴露顺序判断：训练先在哪一侧失控，推理又在哪一侧顶住预算，最后哪些方案只是止血，哪些方案真的值得保留。

主项目线分成“训练侧决策”和“最终收口”两段：`73` 建立训练基线，`76` 比较 checkpoint / offload / hybrid，`75` 形成训练侧预算决策，`74` 再用 profiling 对显存优化方案做端到端最终验证。Task 4 和 Task 5 的推理、量化内容是扩展分支，不是所有学习者的硬性前置。

## 起点：先建立显存问题的共同语言

这条路线不是从“选哪个省显存技巧”开始，而是先确认系统中有哪些显存对象，以及它们在什么时候存在。

Task0 先从训练计算图开始：forward 产生的部分中间结果可能要留到 backward 使用，因此 activation 会在训练期间形成动态显存压力。Part00 `07 Autograd and Backward` 负责建立通用的计算图、梯度和生命周期语言，`18 Activation and Loss Backward` 进一步观察激活函数与 loss 的局部梯度；`17 Attention Backward` 是需要手写 Attention 反向时再进入的进阶扩展。这里得到的是机制判断，不是某个 GPU 上的显存节省比例。

Task1 再把对象放回硬件和运行时环境：dtype 决定对象的字节数，模型规模决定对象数量，GPU 内存层级决定容量和带宽，`06 VRAM / ZeRO` 再把参数、梯度和 optimizer state 放入训练状态账本。主线是 `01 → 02 → 03 → 06`；`04 Attention`、`12 Tensor Core / Mixed Precision`、`14 FlashAttention Memory Model` 是共享支撑，`13 Profiling` 是后续测量出口，`20 显存账本`用于补充对象和指标的统一视角。

到这里，学习者应该能够提出一个可测量的问题：当前压力来自哪个对象、发生在生命周期的哪个阶段、属于容量不足还是带宽 / 搬运问题。只有完成这一步，后面的 checkpoint、offload、KV Cache 或量化比较才有明确的假设。

因此 Task1 的位置是一个桥梁，而不是额外的学习负担：Task0 解释“为什么会保存状态”，Task1 统一“状态如何进入资源账本和测量指标”，Task2 才比较“如何用重算、累积或搬运换显存”。如果还不能指出主要对象，就不应直接根据技巧名称选择策略。

## 第一段：训练中后段开始 OOM

故事可以从一个常见症状开始：某个 batch、序列长度或训练阶段触发 OOM。第一反应往往是缩 batch，但这通常只是止血动作，不是判断结论。更稳的做法是先沿训练侧显存链路排一遍：

- Part 02 [12 Gradient Accumulation](../../02_PyTorch_Algorithms/12_Gradient_Accumulation.md)
- Part 02 [19 Activation Checkpointing and Activation Offload](../../02_PyTorch_Algorithms/19_Activation_Checkpointing_and_Activation_Offload.md)
- Part 02 [42 Activation Offload](../../02_PyTorch_Algorithms/42_Activation_Offload.md)
- Part 02 [73 Training Performance Analysis](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.md)
- Part 02 [76 Activation / Checkpoint / Offload Benchmark](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.md)
- Part 02 [75 Memory Budget Compression Project](../../02_PyTorch_Algorithms/75_Memory_Budget_Compression_Project.md)

这一段真正要拆开的，是 `effective batch`、`activation`、`checkpointing` 和 `offload`。核心路径先完成 `73 -> 76 checkpoint -> 75`；`offload / hybrid` 和更高压力 workload 属于训练侧扩展。很多时候 batch 只是把 activation、重算和搬运成本一并放大了，而不是唯一矛盾本身。

这里可以回看 Part00 的四个共享入口：`05 Tensor` 用来确认 dtype、device 和输入对象，`06 Layout` 用来排查 reshape / contiguous 是否引入复制，`12 Training Interface` 用来固定 batch、padding 和 seq_len，`19 Debugging` 用来区分 OOM、NaN 与 device mismatch。它们提供排查语言，但不替代 73 / 76 的真实测量。

如果压力来自 block 内部，再补看 `14 Activation Functions`、`15 Normalization` 和 `16 Attention`：先判断哪些中间结果会参与 backward，再区分训练 attention 的临时 score 与自回归 decoding step 之间保留的 KV Cache。前者主要进入训练侧 73 / 76，后者转入推理侧 22 / 34 / 66。

## 第二段：另一条分支——推理侧还是装不下

推理侧不要求先完成训练项目；只要问题对象从 activation / optimizer state 转为权重、KV Cache 或请求临时空间，就可以从 Task1 直接进入这条分支。典型现象是模型能加载，但只要上下文拉长、并发上去，显存就被 KV Cache 顶满。这时要切到推理侧显存链路：

- Part 02 [22 vLLM PagedAttention](../../02_PyTorch_Algorithms/22_vLLM_PagedAttention.md)
- Part 02 [34 Prefix Caching and Chunked Prefill](../../02_PyTorch_Algorithms/34_Prefix_Caching_and_Chunked_Prefill.md)
- Part 02 [66 Inference Performance Comparison](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.md)

核心路径先看 `22 -> 34`，再用 `66` 完成单 backend 最小验证；`24 RadixAttention`、`37 KV Cache Scheduling`、`41 KV Cache Quantization` 和 `67` 的真实量化 backend 部署属于扩展路径。

这里的核心不是“为什么慢”，而是“为什么装不下”。要先分清 cache 增长是不是请求形态的自然结果，prefix reuse 和 paging 是否足够，KV cache quantization 是否值得引入。

量化是另一条显存扩展分支：核心先看 [21 量化理论](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.md)、[25 W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.md) 和 [67 量化推理与部署](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.md) 的本地加载；再按需进入 [40 GPTQ / AWQ](../../02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.md) 和 [41 FP8 / KV Cache Quantization](../../02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.md) 的真实 backend 扩展。这里要判断的是：量化省下来的显存是否换来了更长上下文、更大 batch 或更高并发，而不是只看权重文件变小。

## 第三段：账本和实测开始打架

走到这一步，团队通常会碰到第三类问题：理论上算出来应该够，实测却还是很紧；或者峰值显存降下来了，但 benchmark 没好多少。这时就不能只看局部收益，而要把账本和实测证据对齐：

- Part 01 [06 VRAM Calculation and ZeRO](../../01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.md)
- Part 01 [13 Profiling and Bottleneck Analysis](../../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
- Part 02 [73 Training Performance Analysis](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.md)
- Part 02 [76 Activation / Checkpoint / Offload Benchmark](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.md)
- Part 02 [75 Memory Budget Compression Project](../../02_PyTorch_Algorithms/75_Memory_Budget_Compression_Project.md)
- Part 02 [74 Profiling Driven End-to-End Optimization](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.md)

这一段真正要回答的是：理论账本里有没有漏掉临时 buffer、碎片或流程开销；训练侧峰值下降是不是只是把时间转移到了别处；推理侧 cache 压缩是不是只是把显存问题换成了延迟问题。`74` 不替代 `75` 的训练侧预算决策，而是负责最后的 profiling 和端到端验证。

Part00 的 `09 Module` 和 `10 State Dict` 也要放回这里理解：前者帮助确认哪些状态属于模型对象，后者帮助区分“保存以便恢复”和“运行时减少驻留”。如果这两个边界没有先分清，显存账本很容易把 checkpoint 文件、参数状态和 activation 混成一类。

## 第四段：把候选方案放回同一张对比表

到了真正做决策的时候，不能只写“试过 checkpointing、offload、KV quant”，而要把它们放回同一张 baseline / candidate 对比表里。训练侧先由 `75` 输出 `accept / tune / reject`，再由 `74` 检查候选方案在端到端 workload 下是否仍然成立：`baseline` 保留原始配置，`candidate A` 先用 checkpointing 止血，`candidate B` 再引入 offload，推理扩展中再比较 paging / prefix reuse / KV quant。真正的判断不该停在“省了显存”。

## 最终结论长什么样

把这条故事走完以后，一个更像真实交付的结论通常不是“我们用了某个省显存技巧”，而是：训练中后段 OOM 的主要矛盾在 activation 与 effective batch，训练侧方案经过 `73 -> 76 -> 75` 做出预算判断，随后由 `74` 验证端到端代价；如果继续进入推理侧，再说明 KV Cache、量化或 backend 配置是否改变了可部署边界。
