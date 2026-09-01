# 推理优化深入阅读

假设你接手的是同一个在线服务：模型已经能跑，但长 prompt 下首 token 明显变慢；并发一起来，生成速度又开始恶化；为了继续把 batch 和上下文往上推，团队开始尝试 cache 策略和量化，结果显存虽然省了，服务体验却不一定更好。

这条线最重要的是按暴露顺序判断：问题先出在哪一段，压下去以后瓶颈又转到哪里，最后哪些候选方案真的值得保留。

## 第一段：先确认是不是 prefill 问题

第一阶段最常见的症状是：短 prompt 还可以，一旦 prompt 拉长，`TTFT` 明显升高，但 decode 阶段单 token 速度还没完全坏掉。不要笼统说“模型太慢”，而是先把问题压到 prefill：

- Part 01 [03 GPU Architecture and Memory](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.md)
- Part 01 [14 FlashAttention Memory Model](../../01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.md)
- Part 01 [24 SRAM Optimization Techniques](../../01_Hardware_Math_and_Systems/24_SRAM_Optimization_Techniques.md)
- Part 02 [20 FlashAttention Sim](../../02_PyTorch_Algorithms/20_FlashAttention_Sim.md)
- Part 02 [34 Prefix Caching and Chunked Prefill](../../02_PyTorch_Algorithms/34_Prefix_Caching_and_Chunked_Prefill.md)

这里真正想确认的是：`TTFT` 升高是不是主要来自 attention 和 prefill，FlashAttention 或 chunked prefill 有没有空间，重复前缀是不是让无效 prefill 变多。第一个关键判断是：**首 token 慢，不等于生成慢。**

## 第二段：prefill 压下去以后，瓶颈转到 decode

团队把 prefill 调过一轮以后，经常会看到第二个问题：`TTFT` 下来了，但并发一高，`TPOT` 又开始恶化，generated tokens/s 还是不高。这时问题已经从 prefill 转移到了 decode loop 和请求组织：

- Part 02 [21 Decoding Strategies](../../02_PyTorch_Algorithms/21_Decoding_Strategies.md)
- Part 02 [23 Speculative Decoding](../../02_PyTorch_Algorithms/23_Speculative_Decoding.md)
- Part 02 [35 Multi Token Decoding](../../02_PyTorch_Algorithms/35_Multi_Token_Decoding.md)
- Part 02 [36 Decode Scheduling](../../02_PyTorch_Algorithms/36_Decode_Scheduling.md)

这里最容易出现的误判是：只要引入 speculative decoding，吞吐就一定变高。真实情况更依赖 workload；如果请求短、并发低，复杂策略不一定值得，如果调度没排顺，单独换策略也未必真能把 TPOT 压下来。

## 第三段：吞吐还想继续上推，cache 开始决定边界

走到这一步，服务通常已经不是“完全跑不动”，而是开始被 KV cache 和调度边界卡住：batch 想继续往上推，但 `peak memory` 顶住了；cache 一边跑一边涨，decode 稳定性和并发一起变差。这时要把注意力切到 cache 管理和请求调度：

- Part 01 [11 KV Cache and Memory Growth](../../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md)
- Part 02 [22 vLLM PagedAttention](../../02_PyTorch_Algorithms/22_vLLM_PagedAttention.md)
- Part 02 [24 SGLang RadixAttention](../../02_PyTorch_Algorithms/24_SGLang_RadixAttention.md)
- Part 02 [34 Prefix Caching and Chunked Prefill](../../02_PyTorch_Algorithms/34_Prefix_Caching_and_Chunked_Prefill.md)
- Part 02 [37 KV Cache Scheduling](../../02_PyTorch_Algorithms/37_KV_Cache_Scheduling.md)

这里要分清：在推理优化里，我们先问的是 **cache 怎样影响吞吐、并发和服务稳定性**；如果核心问题已经变成“装不下”，再转去显存专题。

## 第四段：量化进入候选集，但不自动代表服务更好

当 cache、batch 和部署成本一起成为约束时，团队通常会开始引入量化。这里最危险的误判是：只要显存降了，就默认服务更好。这一步应沿量化部署线来判断：

- Part 01 [21 Quantization Theory and INT4 INT8](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.md)
- Part 02 [25 Quantization W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.md)
- Part 02 [40 GPTQ and AWQ Weight Quantization](../../02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.md)
- Part 02 [41 FP8 and KV Cache Quantization](../../02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.md)
- Part 02 [67 Quantized Inference and Deployment](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.md)

这里真正要比较的是：这些方案改的是权重、KV cache 还是运行态张量；它们对 `TTFT / TPOT / throughput / peak memory` 的影响是否一致；当前服务目标更重在线交互还是更重离线吞吐。

## 第五段：最后回到同一 workload 做结论

前面几段都还是局部判断，真正做结论时，必须回到同一个 benchmark 框架里。核心收口页是 66；68、69、70 是按 Decode、Cache 和 Serving 目标选择的扩展项目：

- **核心项目：** Part 02 [66 Inference Performance Comparison](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.md)
- **扩展项目：** Part 02 [68 Speculative Decoding Benchmark](../../02_PyTorch_Algorithms/68_Speculative_Decoding_Benchmark.md)
- **主题项目：** Part 02 [69 Prefix Caching Benchmark](../../02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark.md)
- **扩展项目：** Part 02 [70 Serving Scheduler Benchmark](../../02_PyTorch_Algorithms/70_Serving_Scheduler_Benchmark.md)

真正的 benchmark 收口不是“这个方法更先进”，而是 `accept / tune / reject`：它是否适合当前 workload 和服务目标。把这条故事走完以后，一个更像真实交付的结论通常是：长 prompt 下先解决 prefill，随后瓶颈转到 decode 与调度，再往后是 cache 和量化共同决定服务边界，最终被接受的不是某个单点技巧，而是一组在同一 workload 下同时站得住的链路优化组合。
