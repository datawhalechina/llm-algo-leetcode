# 05 Triton Autotune and Profiling

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/03_CUDA_and_Triton_Kernels/05_Triton_Autotune_and_Profiling.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 05. Triton 性能调优与基准测试 (Autotune & Profiling)

**难度：** Medium | **标签：** `Triton`, `Profiling`, `Autotuning` | **目标人群：** 核心 Infra 与算子开发

在工业界，实现结果正确的算子只是第一步。真正的核心竞争力在于**如何证明算子的性能优势**，以及**如何压榨出硬件的极限性能**。
不同大小的张量、不同的 GPU 架构（A100 vs H100）对最佳的 `BLOCK_SIZE` 和 `num_warps` (线程束数量) 的要求是不同的。Triton 提供了 `@triton.autotune` 装饰器来实现**启发式搜索**，以及 `triton.testing.perf_report` 来绘制专业的性能吞吐量曲线图。
本节我们将以一个 Element-wise 操作为例，展示如何自动化搜索最优配置并生成 Profiling 报告。


### Step 1: 调优与测速的核心概念

> **Auto-Tuning (自动调优)：**
> 我们可以提前配置多个候选的字典 `triton.Config`，例如 `BLOCK_SIZE=1024, num_warps=4` 或 `BLOCK_SIZE=4096, num_warps=8`。
> 在第一次运行算子时，Triton 会在后台执行所有配置（预热），记录最优配置，并在后续调用中自动使用该配置。这被称为 JIT 时的启发式搜索。

> **Profiling (性能基准分析)：**
> 算子实现后，我们需要绘制一条横轴为 `N` (数据量大小)，纵轴为 `GB/s` (显存带宽吞吐) 或 `TFLOPs` (计算吞吐) 的折线图。
> 通过 `@triton.testing.perf_report` 装饰器，我们可以优雅地对比 `PyTorch 原生` 和 `Triton 算子` 在不同数据规模下的性能差异。


### Step 2: 吞吐量 的物理意义
在优化算子时，我们需要衡量它离硬件物理极限还有多远。对于 Memory Bound 算子，我们的评价指标是带宽 (GB/s)，即算法处理数据的字节数除以耗时。对于 Compute Bound 算子 (如 GEMM)，指标是算力 (TFLOPS)。通过 `triton.testing.perf_report`，我们可以可视化展示不同尺寸下的性能。


### Step 3: Profiling 代码框架
定义一个 `triton.testing.Benchmark` 实例，指明 X 轴测试变量的区间范围、图表的标题等。然后编写一个 `benchmark` 函数，在内部使用 `do_bench` 获得精确的毫秒级执行时间，最后转换换算并返回。


###  Step 4: 动手实战

**要求**：请补全下方 `vector_add_autotune_kernel` 的 `@triton.autotune` 配置，并运行性能基准测试查看吞吐量图表。


```python
import torch
import triton
import triton.language as tl

# ==========================================
# TODO 1: 添加 triton.autotune 装饰器
# 提示: 提供一个 configs 列表，包含多个 triton.Config({'BLOCK_SIZE': 1024}, num_warps=4) 等组合
# 目标是让编译器在运行时找出对于当前 N 最好的 BLOCK_SIZE 和 num_warps
# ==========================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
    ],
    key=['n_elements'], # 当 n_elements 发生显著变化时重新 autotune
)
@triton.jit
def vector_add_autotune_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)

def add_triton(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    # grid 使用一个 lambda 函数，利用 meta 字典获取当前自动调优选中的 BLOCK_SIZE
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    vector_add_autotune_kernel[grid](x, y, output, n_elements)
    return output

# ==========================================
# 性能测试配置
# ==========================================
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # x 轴作为参数名 (传递给测试函数的参数)
        x_vals=[2**i for i in range(12, 28, 2)],  # x 轴的取值范围 (向量大小从 4K 到 128M)
        x_log=True,  # x 轴使用对数刻度
        line_arg='provider',  # y 轴的多条折线，代表不同的提供商 (PyTorch vs Triton)
        line_vals=['torch', 'triton'],  # 提供商的具体名称
        line_names=['PyTorch', 'Triton (Autotuned)'],  # 图例上的名字
        styles=[('blue', '-'), ('green', '-')],  # 线条样式
        ylabel='GB/s',  # y 轴标签 (带宽吞吐)
        plot_name='vector-add-performance',  # 图像的名称前缀
        args={},  # 传给测试函数的额外固定参数
    )
)
def benchmark(size, provider):
    x = torch.randn(size, device='cuda', dtype=torch.float32)
    y = torch.randn(size, device='cuda', dtype=torch.float32)
    
    quantiles = [0.5, 0.2, 0.8] # 记录中位数，20%分位数，80%分位数
    
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add_triton(x, y), quantiles=quantiles)
        
    # 计算带宽吞吐 (GB/s): 读取 2 个向量，写入 1 个向量
    gbps = lambda ms: 3 * x.numel() * x.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

```

```python
# 运行基准测试并打印结果
# 请在带有 NVIDIA GPU 的机器上运行
import torch

if not torch.cuda.is_available():
    print("⏭️ 忽略测试：此环境没有 NVIDIA GPU，无法运行 Triton 基准测试。")
else:
    print("🚀 开始运行性能分析 (Profiling)... 这可能需要十几秒钟。")
    # 运行 benchmark 并打印结果 (不保存图片，直接打印 pandas dataframe 格式)
    benchmark.run(print_data=True, show_plots=False)
    print("\n✅ Autotune 和 Profiling 测试完成！")
    print("💡 在面试中，提及使用 do_bench 替代 time.time 规避 CUDA 异步，并搜索最佳 num_warps，是极佳的加分项。")

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

::: details 💡 点击查看官方解析与参考代码

### 📝 Autotune 参考实现解析

1. **`@triton.autotune`**: 我们提供了从 `512` 到 `8192` 不同的 `BLOCK_SIZE` 配置，并相应搭配了不同的 `num_warps`。通常，更大的 Block 需要更多的 Warps 来隐藏内存延迟。
2. **`key=['N']`**: 这是一个很重要的参数。由于每次运算的数据量 `N` 可能不同，Triton 会对每一个不同的 `N` 进行一次 Autotune 调优，并将结果缓存。如果下次遇到相同的 `N`，它会直接使用上次调优出来的最佳配置，而不必重新穷举。
3. **消除硬编码**: 在 Python 的启动函数中，原先我们硬编码的 `BLOCK_SIZE` 被移除。`grid` 函数现在接受一个 `meta` 字典，Triton 会自动将当前尝试的配置塞入 `meta` 中，从而让启动代码变得完全动态。

```python
# ==========================================
# 💡 参考答案
# ==========================================

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def vector_add_kernel_autotune(
    x_ptr, y_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_vector_add_autotune(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    N = x.numel()
    out = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    
    vector_add_kernel_autotune[grid](
        x, y, out, N
    )
    return out

```

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
