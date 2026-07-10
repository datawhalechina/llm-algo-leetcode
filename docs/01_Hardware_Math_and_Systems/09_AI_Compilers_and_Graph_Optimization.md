# 09. AI Compilers and Graph Optimization | AI 编译器与计算图优化

**难度：** Hard | **环境：** CPU-first | **标签：** `系统架构`, `AI Compiler` | **目标人群：** 编译优化入门者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/09_AI_Compilers_and_Graph_Optimization.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


这一页把“图怎么被优化成更适合执行的形式”讲清楚，重点是理解编译器为什么会成为推理和部署链路里的关键一环。

**关键词：** `AI Compiler`, `Graph Optimization`, `fusion`
## 前置阅读
**导语：** 这一页先把图优化、fusion 和 lowering 的关系接上，再理解编译器为什么会成为推理和部署链路里的关键一环。
- [Group 1D: Heterogeneous Scheduling and Operator Programming | 1D: 异构调度与算子编程](./1D.md)
- [Group 1E: Compiler Optimization and Hardware Ecosystem | 1E: 编译优化与硬件生态](./1E.md)
- [04. Attention MHA GQA | 多头注意力](../02_PyTorch_Algorithms/04_Attention_MHA_GQA.md)
## 相关阅读
**导语：** 如果想继续把图优化和执行后端的关系补完整，可以接着看这些页。
- [04. Triton 矩阵乘法 (GEMM) 与自动调优 (Autotune)](../03_Triton_Kernels/04_Triton_GEMM_Tutorial.md)
- [05. Triton 性能调优与基准测试 (Autotune & Profiling)](../03_Triton_Kernels/05_Triton_Autotune_and_Profiling.md)
- [18. CUDA Graph and JIT Compile | CUDA Graph 与 JIT 编译](../04_CUDA_and_System_Optimization/18_CUDA_Graph_and_JIT_Compile.md)
## Q1：AI 编译器到底在压缩什么成本向量？

<details><summary>点击展开查看解析</summary>

AI 编译器优化的对象不是 Python 语法，而是计算图在 lowering 过程中形成的成本向量。

它关心的不是“这段代码长什么样”，而是这张图在落到具体 backend 前会产生多少 launch、多少中间张量、多少布局重写、多少同步点。

所以编译器不是把代码“翻译一下”，而是在同一张图上做代价最小化的执行路径重写。
</details>
### Q1小验证：图优化先改什么

先问自己是算子形状、执行顺序，还是中间张量在拖慢系统。

```python
def graph_cost(num_ops, fused_groups, layout_transforms=0):
    # 编译器优化的核心，是减少中间张量和 kernel launch，而不是简单重命名算子。
    launches = fused_groups
    intermediates = max(num_ops - fused_groups, 0)
    return {'launches': launches, 'intermediates': intermediates, 'cost': launches + intermediates + layout_transforms}

plans = [
    ('naive', graph_cost(4, 4, 0)),
    ('part_fused', graph_cost(4, 2, 1)),
    ('more_fused', graph_cost(4, 1, 1)),
]
for name, stats in plans:
    print(name, '->', stats)
print('graph optimization pays off by cutting launches and intermediates together')

```

## Q2：为什么算子融合必须同时看复用、布局和副作用？

<details><summary>点击展开查看解析</summary>

图优化和算子融合之所以经常绑在一起，是因为它们都在处理同一条数据路径：能不能让中间结果不落地、能不能把连续计算收进同一个执行单元、能不能在不破坏语义的前提下复用布局。

当两个算子共享中间张量时，编译器才有融合空间；但是否真能 fuse，还要再看副作用、register pressure 和 layout 是否兼容。

所以融合不是“把两个名字拼起来”，而是同时满足复用、正确性和局部资源约束。
</details>
### Q2小验证：为什么融合能省掉一次搬运

如果两个算子共享同一批中间结果，就要优先考虑是否能在图层合并。

```python
def fusion_score(shared_tensor=True, no_side_effect=True, layout_match=True):
    # 融合不是‘能不能拼起来’，而是收益、正确性和布局是否同时成立。
    score = 0
    score += 2 if shared_tensor else 0
    score += 2 if layout_match else -1
    score += 2 if no_side_effect else -3
    return score

cases = [
    ('ideal', True, True, True),
    ('layout_bad', True, True, False),
    ('side_effect', True, False, True),
]
for name, shared, clean, layout in cases:
    print(name, '-> fusion score', fusion_score(shared, clean, layout))
print('fusion is a three-way condition: reuse, correctness, layout')

```

## Q3：为什么同一张图在不同 backend 上会得到不同的最优解？

<details><summary>点击展开查看解析</summary>

同一张图在不同 backend 上会得到不同的最优解，不是因为图的语义变了，而是因为目标硬件允许的布局、调度和 kernel 形态不同。

编译器在 lowering 的时候，不只是把算子往下放，还要同时满足静态形状、内存布局、算子可用性和性能上限这些约束；一旦某个约束不满足，图优化策略就会变。

所以图优化的最后一步，其实是把“图上的理想解”压回“backend 上可执行的约束解”。
</details>
### Q3小验证：同一张图为什么会跑出不同速度

先想 backend 和硬件支持差异，再看图本身。

```python
def backend_choice(constraints):
    # 后端选择不是按名字排序，而是按约束是否匹配。
    score = {
        'CPU': constraints['portability'] + constraints['debuggability'] + constraints['small_batch'],
        'CUDA': constraints['performance'] + constraints['ecosystem'] + constraints['static_shape'],
        'Triton': constraints['performance'] + constraints['flexibility'] + constraints['static_shape'],
    }
    backend = max(score, key=score.get)
    return backend, score

cases = {
    'research': {'performance': 2, 'flexibility': 2, 'ecosystem': 1, 'portability': 0, 'debuggability': 1, 'small_batch': 0, 'static_shape': 1},
    'production': {'performance': 2, 'flexibility': 1, 'ecosystem': 2, 'portability': 1, 'debuggability': 1, 'small_batch': 0, 'static_shape': 2},
    'debug_first': {'performance': 0, 'flexibility': 1, 'ecosystem': 0, 'portability': 2, 'debuggability': 2, 'small_batch': 2, 'static_shape': 0},
}
for name, constraints in cases.items():
    backend, score = backend_choice(constraints)
    print(name, '->', backend, score)
print('backend choice is a constraint-matching problem, not a prestige ranking')

```

## ⚠️ 常见误区

- AI 编译器不是“把 Python 变成 C++”这么简单。
- 图优化的关键不只是算子名，而是执行路径和数据流。
- 融合不是为了让名字更高级，而是减少中间结果和搬运。
- backend 差异会直接影响优化结果。