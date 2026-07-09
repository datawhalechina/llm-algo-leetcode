# 32. TVM MLIR Deep Practice | TVM / MLIR 深度实践

**难度：** Hard | **环境：** CPU-first | **标签：** `AI 编译器`, `TVM` | **目标人群：** 编译器后端学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/32_TVM_MLIR_Deep_Practice.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


这一页讲的是为什么“从图到可执行 kernel”不是简单翻译，而是一整条优化和 lowering 的链路。

**关键词：** `TVM`, `MLIR`, `Relay`, `autotune`, `backend`
## 前置

**导语：** 这一页先把图优化、算子融合和 lowering 的直觉接上，再看 TVM / MLIR 为什么会在后端链路里占核心位置。

## 相关阅读

**导语：** 如果想继续把编译后端、调度和成本模型串起来，可以接着看这些页。

## Q1：TVM / MLIR 这类编译器后端在做什么？

<details><summary>点击展开查看解析</summary>

后端要做的不是“把高层代码换个语法”，而是把高层表示逐步 lowering 到能在目标硬件上高效运行的形式。

这个过程通常会涉及：
- 图或 IR 的转换；
- 算子拆分与融合；
- 调度与布局优化；
- 面向目标 backend 的 codegen。

所以编译器后端本质上是在把“可表达”变成“可高效执行”。
</details>
### Q1小验证：lowering 不是翻译

先把“变成能跑”与“变得更快跑”分开理解。

```python
ir = {'op': 'matmul', 'dtype': 'fp16', 'shape': 'B x N x K'}

def specialize_shape(graph):
    graph = dict(graph)
    graph['shape'] = graph['shape'].replace('B', 'batch')
    return graph

def legalize_ops(graph):
    graph = dict(graph)
    graph['legalized'] = True
    return graph

def schedule(graph):
    graph = dict(graph)
    graph['schedule'] = 'tile=128, warp=4'
    return graph

def codegen(graph):
    graph = dict(graph)
    graph['backend'] = 'cuda'
    return graph

pipeline = [specialize_shape, legalize_ops, schedule, codegen]
for step in pipeline:
    ir = step(ir)
    print(step.__name__, '->', sorted(ir.keys()))
```

## Q2：为什么 autotune 和 backend 选择这么重要？

<details><summary>点击展开查看解析</summary>

同一个高层表示，在不同硬件、不同 tile、不同 schedule 下，性能可能差别很大。

autotune 的作用，是在可行搜索空间里找更适合目标硬件的参数组合；backend 选择则决定最终会落到哪条执行路径上。

所以编译器后端不是“一次写完就完”，而是需要根据硬件不断调优和选择。
</details>
### Q2小验证：为什么参数搜索有价值

不同 tile 和 schedule 会直接改变性能。

```python
def tune(tile, schedule):
    return tile * schedule

for tile, schedule in [(64, 1), (128, 2), (256, 3)]:
    print(tile, schedule, '->', tune(tile, schedule))
```

## Q3：为什么编译器和 kernel 实现最后会连到一起？

<details><summary>点击展开查看解析</summary>

编译器决定“怎么拆、怎么排、怎么调”，kernel 决定“怎么真的跑”。

一旦进入目标硬件执行层，编译器的调度结果就会具体反映到 kernel 的内存访问、并行组织和算子组合上。反过来，kernel 能力也会限制编译器能把图优化到什么程度。

所以 TVM / MLIR 不是和 Triton、CUDA 分开的知识，而是和它们组成一条从高层到低层的连续链路。
</details>
### Q3小验证：编译器和 kernel 谁更靠近硬件

答案不是二选一，而是两者互相约束。

```python
compiler_passes = [
    ('fusion', {'shared_memory': True}),
    ('vectorize', {'mma': True}),
    ('layout_opt', {'contiguous_load': True}),
]

hardware = {
    'shared_memory': True,
    'mma': False,
    'contiguous_load': True,
}

for name, need in compiler_passes:
    ok = all(hardware.get(k, False) >= v for k, v in need.items())
    print(name, '->', 'keep' if ok else 'drop')
```

## Q4：为什么 MLIR 更强调多层 IR 和 pass pipeline，而不是一步到位的 lower？

<details><summary>点击展开查看解析</summary>

更深的后端优化，往往不是靠一个“万能 lowering”完成的，而是靠一串可组合、可局部验证的 passes 逐层收敛。

这样做有三个好处：
- **可控**：每一层 IR 只处理自己能表达清楚的问题，避免把所有硬件细节一次塞进去；
- **可复用**：同一套上层表示可以映射到不同硬件后端；
- **可优化**：不同 pass 可以分别针对布局、合法化、调度和生成做局部优化。

所以 MLIR 的价值，不只是“中间表示更多”，而是让 lowering 过程本身变成一条可维护的优化链路。
</details>
### Q4小验证：为什么分层比一步到位更稳

```python
passes = [
    ('shape_specialize', 1),
    ('legalize_ops', 2),
    ('layout_transform', 2),
    ('scheduling', 3),
    ('codegen', 4),
]

print('pass pipeline:')
for name, cost in passes:
    print(f'- {name}: local cost {cost}')
print('total stages:', len(passes))

```

## ⚠️ 常见误区

- 后端不是简单代码生成。
- autotune 不是可有可无，而是决定性能的重要部分。
- 编译器和 kernel 不是两条独立链路。
- 图到 kernel 的过程本质上是在做层层约束下的优化。