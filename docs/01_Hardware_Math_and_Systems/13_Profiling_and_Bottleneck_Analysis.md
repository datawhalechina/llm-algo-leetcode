# 13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位

**难度：** Medium | **环境：** GPU optional | **标签：** `系统性能`, `Profiling`, `瓶颈分析` | **目标人群：** 系统性能入门者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

这一页把“看懂硬件”和“会定位问题”接起来，先建立先测量、再优化的判断习惯。

这一页在整个教程的纵向主线里属于 `Part 01` 的性能取证基础页，主服务 `监督微调路线`，也给 `推理优化路线` 补一层证据链视角。学完这里，后面再看 `60 / 62 / 66 / 73` 时，你会更容易把 step time、数据搬运、同步等待和 kernel 瓶颈放进同一条 profiling 证据链里；如果这里没学明白，后面很容易只剩经验判断，而不知道瓶颈到底卡在算子、调度还是通信。按专题归类，这一页主要属于 `Profiling 专题`，并同时支撑训练与推理两条路线的定位能力。

**关键词：** `profiling`, `bottleneck`, `latency`

## 证据边界与显存路线映射
本节是 **GPU optional**：CPU 模拟可以验证计时、阶段拆分、热点排序和瓶颈分类的流程；真实 GPU profiler 才能验证 CUDA kernel 时间、显存生命周期、同步等待和带宽利用率；通信开销需要在包含多卡通信的 workload 中单独验证。模拟热点只能作为待验证假设。

对应显存优化路线的 Task3（训练侧性能验证）和 Task6（profiling 收口），直接支撑 73 的 baseline 设计与 74 的 trace 解读；本节不替代 73 / 76 的真实 benchmark，也不单独决定 accept / tune / reject。

---

## 前置阅读

**导语：** 先从单卡硬件与编程模型入手，再看 profiling 的瓶颈定位会更顺；如果你正在走 `监督微调路线`，这里会直接服务训练实验、项目页和训练性能分析链。

- [Group 1B: Single-GPU Hardware and Memory Optimization | 1B: 单卡硬件与访存优化](./1B.md)
- [Group 1D: Heterogeneous Scheduling and Operator Programming | 1D: 异构调度与算子编程](./1D.md)

## 相关阅读

**导语：** 把 profiling 放到推理和训练实战里看，更容易验证结论；如果你正在走 `监督微调路线`，它会直接服务 `04 / 60 / 62 / 73`，把“感觉慢”变成“证据显示慢在哪里”。

- [66. Inference Performance Comparison | 推理性能对比实验](../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.md)
- [73. Training Performance Analysis | 训练性能分析](../02_PyTorch_Algorithms/73_Training_Performance_Analysis.md)
- [60. LoRA Fine-Tuning Project | LoRA 微调项目](../02_PyTorch_Algorithms/60_LoRA_Fine_Tuning_Project.md)
---
## 常用工具链

**导语：** profiling 不只是“看图”，更重要的是按层级选工具：先看时间线，再看 kernel，再回到代码。

- `torch.profiler`：先看 Python / operator / CPU-GPU 时间线，回答“时间花在哪”。
- `Nsight Systems`：看整条时间线、stream overlap、通信与计算是否重叠，回答“调度是不是问题”。
- `Nsight Compute`：看单个 kernel 的 occupancy、memory throughput、warp stall 和 instruction mix，回答“kernel 为什么慢”。
- `nvprof`：可以作为历史工具了解，但当前更推荐前面两类工具。

工具的顺序通常是：先用 `torch.profiler` 找大方向，再用 `Nsight Systems` 看系统级重叠，最后用 `Nsight Compute` 钻到 kernel 细节。

## Q1：Profiling 的第一步是什么，为什么不能一上来就改代码？

<details>
<summary>点击展开查看解析</summary>

Profiling 的第一步不是找“感觉上最慢的地方”，而是先把时间花在哪些阶段测出来。

通常要先回答三个问题：

1. **时间主要耗在前向、反向，还是数据搬运？**
2. **瓶颈是单步延迟高，还是整体吞吐低？**
3. **问题更像计算瓶颈、显存瓶颈，还是调度瓶颈？**

如果不先测量，优化经常会变成“改得很努力，但方向不对”。很多时候，真正拖慢系统的不是你最先注意到的那个算子，而是数据预处理、同步点、或者某个意外的内存拷贝。

所以 profiling 的价值不是生成一堆图，而是帮你把注意力从“猜测”转到“证据”。
</details>
### Q1小验证：先问自己哪类信息最重要

看到一个慢任务时，先判断它更像是延迟问题还是吞吐问题。

```python
def summarize_profile(phases):
    total = sum(phases.values())
    ranked = sorted(phases.items(), key=lambda kv: kv[1], reverse=True)
    shares = [(name, round(cost / total, 2)) for name, cost in ranked]
    return total, ranked[0], shares

profile = {'forward': 32, 'backward': 18, 'h2d_copy': 10, 'sync': 5}
total, dominant, shares = summarize_profile(profile)
print('total_ms =', total)
print('dominant_stage =', dominant)
print('shares =', shares)

```

## Q2：怎么区分算力瓶颈、访存瓶颈和调度瓶颈？

<details>
<summary>点击展开查看解析</summary>

可以先用三类信号去分：

- **算力瓶颈**：计算单元忙，算子本身 FLOPs 高，且优化重点常在减少冗余计算或提高矩阵利用率。
- **访存瓶颈**：算子算得不算多，但搬运的数据很多，带宽指标接近上限，常见于 Attention、缓存访问和大张量读写。
- **调度瓶颈**：单个算子不一定重，但 kernel 太碎、同步太多、launch 开销和流水线利用率不好，常见于 Python 侧碎操作、CPU/GPU 频繁切换、stream 使用不当。

这三类瓶颈往往会叠在一起，但 profiling 的作用就是帮你判断“主矛盾是什么”。一旦主矛盾判错，后续优化方向就会偏离。
</details>
### Q2小验证：瓶颈类型先分哪一类

先别急着改实现，先判断问题更像哪一类瓶颈。

```python
def bottleneck_score(flop_util, bw_util, launch_count):
    # 越接近 1.0 表示越接近上限；launch_count 越大越偏调度问题。
    compute_gap = 1 - flop_util
    memory_gap = 1 - bw_util
    launch_gap = min(launch_count / 100.0, 1.0)
    return {
        'compute': round(compute_gap, 2),
        'memory': round(memory_gap, 2),
        'scheduling': round(launch_gap, 2),
    }

case = bottleneck_score(flop_util=0.42, bw_util=0.91, launch_count=18)
print(case)
print('main bottleneck =', min(case, key=case.get))

```

## Q3：profiling 结果如何反过来指导优化？

<details>
<summary>点击展开查看解析</summary>

profiling 的终点不是“看完图”，而是把结果变成下一步动作。

- 如果是**访存瓶颈**，通常要考虑减少搬运、合并 kernel、提高数据复用，或者换更好的 attention / cache 方案。
- 如果是**算力瓶颈**，通常要考虑更合适的矩阵实现、混合精度、Tensor Core 利用率或者更高效的 kernel 设计。
- 如果是**调度瓶颈**，通常要考虑减少 Python 开销、减少同步、合并小操作、优化 stream 组织。

这也是为什么 profiling 和 Triton、编译优化、系统调度会连在一起：它们不是并列知识点，而是“测量 -> 诊断 -> 重写”的连续链路。
</details>
### Q3小验证：看到结果后先选哪个方向

把瓶颈先分清，再决定是改算子、改内存还是改调度。

```python
def recommend_actions(summary):
    actions = []
    if summary['compute'] > 0.4:
        actions.append('improve kernel efficiency')
    if summary['memory'] > 0.4:
        actions.append('reduce memory traffic')
    if summary['scheduling'] > 0.1:
        actions.append('merge launches / reduce sync')
    return actions or ['keep profiling']

for case in [
    {'name': 'attention', 'flop_util': 0.38, 'bw_util': 0.96, 'launch_count': 12},
    {'name': 'kernel', 'flop_util': 0.84, 'bw_util': 0.41, 'launch_count': 8},
    {'name': 'python_overhead', 'flop_util': 0.92, 'bw_util': 0.88, 'launch_count': 26},
]:
    summary = bottleneck_score(case['flop_util'], case['bw_util'], case['launch_count'])
    print(case['name'], '->', recommend_actions(summary))

```

## ⚠️ 常见误区

- Profiling 不是为了“找一个看起来最慢的点”，而是为了确认主瓶颈。
- 高延迟不一定等于高算力消耗，很多时候是同步点或搬运在拖慢系统。
- 优化前不测量，优化后不复测，通常都会让结论失真。
- profiling 和优化不是两件事，它们是同一条链路上的前后两步。