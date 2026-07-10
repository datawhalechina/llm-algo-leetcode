# 30. Dynamic Shape Handling | 动态 Shape 处理

**难度：** Medium-Hard | **环境：** CPU-first | **标签：** `动态 Shape`, `推理服务` | **目标人群：** 动态 batching 学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/30_Dynamic_Shape_Handling.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


这一页讲的是输入长度不固定时，为什么 batching、缓存和执行路径会同时变复杂。

**关键词：** `dynamic shape`, `batching`, `padding`
## 前置阅读

**导语：** 这一页先把动态 batching、缓存复用和执行路径的关系接上，再看输入长度变化为什么会让系统更难稳定优化。

- [07. CPU and GPU Heterogeneous Scheduling | CPU 与 GPU 异构调度](./07_CPU_GPU_Heterogeneous_Scheduling.md)
- [08. Programming Models and CUDA/Triton | 编程模型演进](./08_Programming_Models_CUDA_Triton.md)
- [29. CUDA Stream Advanced Scheduling | CUDA Stream 高级调度](./29_CUDA_Stream_Advanced_Scheduling.md)

## 相关阅读

**导语：** 如果想继续把动态输入和后续的调度、虚拟化、成本判断串起来，可以接着看这些页。

- [19. Operator Fusion Introduction | 算子融合导论](./19_Operator_Fusion_Introduction.md)
- [15. CUDA Custom Kernel Intro | CUDA 自定义算子入门](../04_CUDA_and_System_Optimization/15_CUDA_Custom_Kernel_Intro.md)
- [18. CUDA Graph and JIT Compile | CUDA Graph 与 JIT 编译](../04_CUDA_and_System_Optimization/18_CUDA_Graph_and_JIT_Compile.md)

## Q1：动态 shape 为什么会让复用变差？

<details><summary>点击展开查看解析</summary>

固定 shape 时，batch 形状、缓存布局、kernel 形态和 graph 回放都更容易复用。

一旦输入长度不固定，这些复用都会变差：不同请求要重新分桶，padding 会引入无效计算，graph / kernel 复用率下降，执行路径也更容易分叉。

所以动态 shape 的难点，不是长度变化本身，而是它让原本能重复利用的优化结果变得不稳定。
</details>
### Q1小验证：固定假设为什么会失效

先记住固定 shape 下哪些结果能复用，再看动态输入会打断哪些复用。

```python
def reuse_gap(shape_fixed=True, cached_batches=4, dynamic_lengths=1):
    # 复用是否成立，不是看一句 fixed / dynamic，而是看缓存和 kernel 能复用多少次。
    if not shape_fixed:
        return {'reuse_ratio': round(cached_batches / (cached_batches + dynamic_lengths), 2), 'stable': False}
    return {'reuse_ratio': 1.0, 'stable': True}

for case in [(True, 4, 1), (False, 4, 1), (False, 2, 4)]:
    print(case, '->', reuse_gap(*case))
print('dynamic shape hurts when cached shapes stop being reusable')

```

## Q2：padding 和 bucketing 分别在减少什么浪费？

<details><summary>点击展开查看解析</summary>

- **padding** 减少的是形状不一致带来的拼 batch 难题，它用填充把不同长度的输入拼成更规则的张量；
- **bucketing** 减少的是无效 padding，它把相近长度的请求放到同一桶里，让浪费更少。

两者的目标都不是“消灭长度差异”，而是在吞吐、显存浪费和 kernel 稳定性之间找折中。

如果没有这层处理，动态 batching 很容易同时损失计算效率和资源利用率。
</details>
### Q2小验证：什么时候更适合 bucketing

长度分布越分散，bucketing 对减少 padding 浪费越有价值。

```python
def padding_overhead(lengths, bucket_size=None):
    # padding 的代价在于把短序列补到同一长度时浪费了多少 token。
    if bucket_size is None:
        max_len = max(lengths)
        wasted = sum(max_len - l for l in lengths)
    else:
        wasted = sum((bucket_size - (l % bucket_size)) % bucket_size for l in lengths)
    return wasted

batches = [[64, 128, 256], [240, 248, 256], [512, 520, 800]]
for lengths in batches:
    print(lengths, '-> waste no bucket:', padding_overhead(lengths), 'bucket128:', padding_overhead(lengths, 128))
print('bucketing helps when the no-bucket waste is much larger than the bucket padding')

```

## Q3：动态 shape 为什么会让性能更容易抖动？

<details><summary>点击展开查看解析</summary>

动态 shape 会同时影响两层决策：

1. **执行路径**：不同长度可能触发不同 kernel、不同 graph 或不同调度分支。
2. **缓存策略**：原本在固定形状下可复用的编译结果、块大小、缓存布局和调度参数，在动态场景里可能要按形状重新选择。

所以动态 shape 的本质，是把“编译期可确定的执行链”变成“依赖 runtime 形状判断的执行链”，性能也就更容易抖动。
</details>
### Q3小验证：为什么性能会抖动

长度变化导致路径和缓存都跟着变时，性能就更容易抖。

```python
def path_stability(lengths):
    # 形状一变就换路径，说明 runtime 选择带来的抖动会更大。
    unique_paths = len(set('short' if l < 256 else 'long' if l < 1024 else 'very_long' for l in lengths))
    return {'unique_paths': unique_paths, 'stable': unique_paths == 1}

for lengths in [[128, 128, 128], [128, 512, 128], [128, 512, 2048]]:
    print(lengths, '->', path_stability(lengths))
print('more path variety means more performance jitter')

```

## ⚠️ 常见误区

- 动态 shape 不是“只是输入长度不同”。
- padding 能解决对齐，但不等于没有浪费。
- bucketing 不是万能，只是减少浪费的办法之一。
- 动态输入场景里，执行路径和缓存复用都可能变差。