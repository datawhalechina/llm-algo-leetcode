# 18. CUDA Graph and JIT Compile | CUDA Graph 与 JIT 编译：启动开销消除与多流调度

本节承接 17 节的 CUDA Streams 与异步传输，继续往前一步：当计算图的结构已经相对固定时，怎样进一步减少每一步训练或推理的启动开销。

## 这一节解决什么问题

在很多大模型推理和训练场景里，真正拖慢吞吐量的并不只是 kernel 的计算本身，还包括大量重复的 kernel launch、Python 调度和运行时开销。CUDA Graph 要解决的，就是把一段稳定的计算流程“录下来”，然后重复回放，从而减少启动损耗。

## 为什么有了 Streams 还需要 Graph

- **Streams** 解决的是“不同任务之间如何并发、如何重叠”。
- **CUDA Graph** 解决的是“同一段稳定流程如何减少启动开销”。

换句话说，Streams 负责调度，Graph 负责压缩调度成本。两者不是替代关系，而是互补关系。

## 与 Triton 的关系

在 Part 3 里，我们已经学会了如何写 Triton kernel。进入 Part 4 后，一个常见的工程问题是：

- 单次 Triton kernel 已经够快了，为什么整条推理链还是不够快？
- 如果一个推理步骤里要连续执行多个固定 kernel，是否可以把它们一起包装成 Graph？
- vLLM / serving 场景里，如何把 Triton kernel 放进 CUDA Graph 里，进一步减少 launch overhead？

本节将围绕这些问题，建立 Graph、JIT 与 Triton kernel 之间的工程联系。

## 一个最小的工程场景

可以把 CUDA Graph 理解为“把一段稳定的推理步骤录下来，然后重复播放”：

```python
# 伪代码：固定形状的推理步骤
for step in range(warmup):
    y = triton_kernel(x)
    z = other_cuda_kernel(y)

# 录制 Graph
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    y = triton_kernel(x_static)
    z = other_cuda_kernel(y)

# 回放 Graph
for _ in range(num_iters):
    graph.replay()
```

这个场景的重点不是“Graph 能替代 Triton”，而是：

- Triton 负责高性能算子本身。
- CUDA Graph 负责减少每次重复执行这些算子的启动开销。
- 在 vLLM、serving 和固定结构的 decode loop 里，这种组合非常常见。

## 学习目标

1. 理解 CUDA Graph 解决的核心瓶颈。
2. 理解 Streams 与 Graph 的分工。
3. 理解 Graph 如何与 Triton / CUDA kernel 配合使用。
4. 为后续推理服务和高吞吐部署建立概念基础。

## 导航

- [Part 4 导学](./intro.md)
- [上一组 4.2 系统级性能优化](./4_2.md)
- [下一组 4.3 分布式训练工程](./4_3.md)

## 当前状态

- 已补充最小 Graph capture 场景。
- 后续还可以继续补充 CUDA Graph capture、JIT compilation 和 launch overhead 优化细节。
- 当前作为系统优化与推理服务之间的桥接页，重点是建立“Streams 之后为什么需要 Graph”的直觉。
