# 05. Triton Autotune and Profiling | Triton 性能调优与基准测试 (Autotune & Profiling)

**难度：** Medium | **标签：** `Triton`, `Profiling`, `Autotuning` | **目标人群：** 核心 Infra 与算子开发

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/03_CUDA_and_Triton_Kernels/05_Triton_Autotune_and_Profiling.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


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
```


```python
# ==========================================
# TODO 1: 添加 triton.autotune 装饰器
# 提示: 
# 1. 使用 @triton.autotune 装饰器
# 2. 提供 configs 列表，包含至少 3 个不同的 triton.Config 配置 (探索不同的 BLOCK_SIZE 和 num_warps 组合)
# 3. 设置 key 为 ['n_elements']，以便对不同的输入大小缓存最优配置
# ==========================================
# @triton.autotune(???)

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
    
    # ==========================================
    # TODO 2: 动态计算 grid
    # ==========================================
    # grid = ???
    pass
    
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
        try:
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: add_triton(x, y), quantiles=quantiles)
        except Exception as e:
            ms, min_ms, max_ms = float('inf'), float('inf'), float('inf')
        
    # 计算带宽吞吐 (GB/s): 读取 2 个向量，写入 1 个向量
    gbps = lambda ms: 3 * x.numel() * x.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)
```


```python
# ==========================================
# 验证正确性测试
# ==========================================
def test_autotune_correctness():
    if not torch.cuda.is_available():
        print("⏭️  忽略测试：无 GPU")
        return
    
    print("\n--- 测试开始 ---")
    try:
        x = torch.randn(10000, device='cuda')
        y = torch.randn(10000, device='cuda')
        z = add_triton(x, y)
        assert torch.allclose(x + y, z), "❌ Autotune 算子输出不正确"
        print("✅ Autotune 正确性测试通过")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise e

test_autotune_correctness()
```


```python
# 运行基准测试并打印结果
# 请在带有 NVIDIA GPU 的机器上运行
import torch

if not torch.cuda.is_available():
    print("⏭️ 忽略测试：此环境没有 NVIDIA GPU，无法运行 Triton 基准测试。")
else:
    print("开始运行性能分析 (Profiling)... 这可能需要十几秒钟。")
    # 运行 benchmark 并打印结果 (不保存图片，直接打印 pandas dataframe 格式)
    benchmark.run(print_data=True, show_plots=False)

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---
## 参考代码与解析

### 代码

```python
import torch
import triton
import triton.language as tl

# ==========================================
# TODO 1: 添加 triton.autotune 装饰器
# ==========================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def vector_add_autotune_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)

def add_triton(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    out = torch.empty_like(x)
    
    # ==========================================
    # TODO 2: 动态计算 grid
    # ==========================================
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    vector_add_autotune_kernel[grid](
        x, y, out, n_elements
    )
    return out
```

### 解析

**1. TODO 1: 添加 triton.autotune 装饰器**
- **实现方式**：使用 `@triton.autotune` 装饰器，提供多个 `triton.Config` 配置组合
- **关键点**：每个配置指定不同的 `BLOCK_SIZE` 和 `num_warps` 组合，让 Triton 在运行时自动选择最优配置
- **技术细节**：
  - `configs` 列表包含从 512 到 8192 的不同 BLOCK_SIZE，对应不同的 num_warps（2 到 16）
  - 通常更大的 BLOCK_SIZE 需要更多的 warps 来隐藏内存延迟
  - `key=['n_elements']` 指定 Triton 根据输入数据量 `n_elements` 进行调优缓存，相同数据量会复用已调优的最佳配置

**2. TODO 2: 动态计算 grid**
- **实现方式**：`grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)`
- **关键点**：使用 lambda 函数接收 `meta` 字典，Triton 会自动将当前配置注入其中
- **技术细节**：消除了硬编码的 BLOCK_SIZE，使启动代码完全动态化，autotune 可以自由尝试不同配置

**工程优化要点**
- **Autotune 原理**：Triton 在首次运行时会预热所有配置，测量每个配置的性能，并缓存最优结果。后续调用直接使用缓存的最佳配置，避免重复搜索。
- **Profiling 最佳实践**：使用 `triton.testing.do_bench` 而非 `time.time()`，因为 CUDA 操作是异步的，`do_bench` 会正确处理 GPU 同步并返回准确的执行时间。
- **性能指标选择**：对于 Memory Bound 算子（如向量加法），使用带宽 (GB/s) 作为评价指标；对于 Compute Bound 算子（如矩阵乘法），使用算力 (TFLOPS)。
- **配置空间设计**：BLOCK_SIZE 通常选择 2 的幂次方（便于硬件对齐），num_warps 根据 BLOCK_SIZE 调整（更大的块需要更多并行度）。
- **缓存键设计**：`key` 参数应包含影响性能的关键维度（如数据量、矩阵形状），确保不同场景使用合适的配置。