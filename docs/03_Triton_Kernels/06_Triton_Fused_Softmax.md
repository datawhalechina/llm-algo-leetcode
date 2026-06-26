# 06. Triton Fused Softmax | Triton 进阶：跨线程归约与数值稳定 (Safe Softmax)

**难度：** Hard | **标签：** `Triton`, `Reduction`, `Attention` | **目标人群：** 核心 Infra 与算子开发

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/03_Triton_Kernels/06_Triton_Fused_Softmax.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


在纯 Python 模拟 FlashAttention 时，我们探讨过 Softmax。标准的 Softmax 是 Memory Bound 的痛点，因为它需要读取整个行三次：寻找最大值、计算指数和、计算除法。
通过 Triton 的 SRAM，我们可以将**一整行 (Row)** 加载到片上缓存，在 SRAM 内部完成 `max`, `exp`, `sum` 以及除法运算，最终只写回显存一次！
本节我们将实现一个高效率、数值稳定的 Fused Softmax 算子。

## 前置

**导语：** 这一节会把 Softmax 这类归约算子直接接到 Triton 的行级并行模型上。

- [Part 1: 1B 单卡硬件与访存优化](../01_Hardware_Math_and_Systems/1B.md)
- [Part 1: 1D 异构调度与算子编程](../01_Hardware_Math_and_Systems/1D.md)
- [Part 1: 19 算子融合导论](../01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.md)

### Step 1: Softmax 数值稳定性与 Triton 归约

> **数值稳定性 (Safe Softmax)：**
> 如果直接计算 $e^x$，当 $x$ 较大时（如 50），$e^{50}$ 会导致浮点数溢出 (NaN)。
> 解决方案：让一行的每一个元素都减去该行的最大值 $m$。
> $Soft\max(x_i) = \frac{e^{x_i - m}}{\sum e^{x_j - m}}$，这在数学上等价，但在计算机浮点表示中更加安全。

> **Triton 的行级并行：**
> 处理形状为 `(M, N)` 的矩阵时，通常分配**一个 Program (线程块) 专门处理矩阵的一行**。
> - `pid = tl.program_id(0)` 获取行号。
> - 计算该行在内存中的起始指针。
> - 将该行全部 Load 进 SRAM。本节采用**单行一次性装载**的简化版假设，即 `N <= BLOCK_SIZE`；如果 `N` 更大，后续可以扩展成分块循环归约。
> - `tl.max(x, axis=0)` 和 `tl.sum(x, axis=0)` 在这里表示对整行做一次归约；对一维向量来说，`axis=0` 就是“沿这一维全部归约”。

### Step 2: 跨线程安全 Softmax 归约算法
朴素的 Softmax 公式是 $e^{x_i} / \sum e^{x_j}$，但这很容易导致指数爆炸（例如 $e^{100}$ 会变成 NaN）。安全版本（Safe Softmax）是先求出这一行的最大值 $M$，然后计算 $e^{x_i - M}$。由于 Triton 的 `tl.max` 和 `tl.sum` 会被编译成块内并行归约，我们可以借助它们完成这一数值稳定的变换，而不是手写串行扫描。

### Step 3: 代码实现框架
内部分配指针时，我们需要指向二维张量的特定一行。加载后得到的是一维向量，因此 `x_max = tl.\max(x, axis=0)` 的 `axis=0` 表示对整行做全归约。将其从原数据中减去：`num = tl.exp(x - x_max)`，再求和得到 `denom = tl.sum(num, axis=0)`，最后相除即为最终分布。

###  Step 4: 动手实战

**要求**：请补全下方 `fused_softmax_kernel`。假设输入矩阵大小为 `(M, N)`，为了教学简化，我们采用**单行一次性装载**的版本，也就是假设 `N <= BLOCK_SIZE`，让每一行都能一次性被装入 SRAM。后续如果要处理更大的 `N`，可以在这个骨架上再扩展成分块循环版本。


```python
try:
    import triton
except ModuleNotFoundError:
    try:
        import google.colab  # type: ignore
    except Exception:
        raise
    import subprocess, sys
    print('Installing Triton for Part 3...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'triton'])
    import triton

import torch
import triton
import triton.language as tl
```


```python

@triton.jit
def fused_softmax_kernel(
    output_ptr, input_ptr, input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # 1. 获取当前 program 处理的是哪一行
    row_idx = tl.program_id(0)
    
    # 2. 定位到当前行的起始指针
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    # 3. 构造这一行的连续索引和掩码 (防越界)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    
    # 4. 加载一整行到 SRAM 中
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # ==========================================
    # TODO 1: 寻找当前行的最大值 (安全 Softmax 第一步)
    # ==========================================
    # row_max = ???
    pass
    
    # 减去最大值，避免 exp() 溢出
    # safe_row = ???
    
    # ==========================================
    # TODO 2: 计算指数 (Numerator)
    # ==========================================
    # numerator = ???
    pass
    
    # ==========================================
    # TODO 3: 求和 (Denominator)
    # ==========================================
    # denominator = ???
    pass
    
    # ==========================================
    # TODO 4: 计算最终输出
    # ==========================================
    # softmax_output = ???
    pass
    
    # 定位输出指针，写回
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
        
    grid = (M,)
    
    fused_softmax_kernel[grid](
        y, x,
        x.stride(0), y.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    return y
raise NotImplementedError("请先完成 TODO 1-4")

```


```python
# 测试你的实现
def test_fused_softmax():
    if not torch.cuda.is_available():
        print("⏭️ 无 GPU，完成结构检查；运行级验证需要 GPU。")
        assert "fused_softmax_kernel" in globals(), "缺少 fused_softmax_kernel"
        assert "triton_softmax" in globals(), "缺少 triton_softmax"
        print("✅ Triton Fused Softmax 结构检查通过")
        return True
        
    try:
        torch.manual_seed(42)
        test_cases = [
            (8192, 1000),
            (7, 257),
        ]
        
        for M, N in test_cases:
            x = torch.randn(M, N, device='cuda')
            y_ref = torch.softmax(x, dim=1)
            y_tri = triton_softmax(x)
            diff = torch.max(torch.abs(y_ref - y_tri))
            print(f"[{M}x{N}] 最大误差: {diff.item():.6e}")
            assert diff < 1e-5, f"Triton Softmax 计算结果不准确！(M={M}, N={N})"
        
        print("✅ 跨线程归约的数值稳定 Softmax 算子实现成功！")
        
        print()
        print("--- 性能基准测试 (Benchmark) ---")
        quantiles = [0.5, 0.2, 0.8]
        M, N = 8192, 1000
        x = torch.randn(M, N, device='cuda')
        ms_pt, min_ms_pt, max_ms_pt = triton.testing.do_bench(lambda: torch.softmax(x, dim=1), quantiles=quantiles)
        ms_tr, min_ms_tr, max_ms_tr = triton.testing.do_bench(lambda: triton_softmax(x), quantiles=quantiles)
        print(f"PyTorch Time: {ms_pt:.4f} ms")
        print(f"Triton Time:  {ms_tr:.4f} ms")
        print(f"Speedup:      {ms_pt / ms_tr:.2f}x")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise

test_fused_softmax()

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

@triton.jit
def fused_softmax_kernel(
    output_ptr, input_ptr, input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # 1. 获取当前 program 处理的是哪一行
    row_idx = tl.program_id(0)
    
    # 2. 定位到当前行的起始指针
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    # 3. 构造这一行的连续索引和掩码 (防越界)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    
    # 4. 加载一整行到 SRAM 中
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # ==========================================
    # TODO 1: 寻找当前行的最大值 (安全 Softmax 第一步)
    # ==========================================
    row_max = tl.max(row, axis=0)
    
    # 减去最大值，避免 exp() 溢出
    safe_row = row - row_max
    
    # ==========================================
    # TODO 2: 计算指数 (Numerator)
    # ==========================================
    numerator = tl.exp(safe_row)
    
    # ==========================================
    # TODO 3: 求和 (Denominator)
    # ==========================================
    denominator = tl.sum(numerator, axis=0)
    
    # ==========================================
    # TODO 4: 计算最终输出
    # ==========================================
    softmax_output = numerator / denominator
    
    # 定位输出指针，写回
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
        
    grid = (M,)
    
    fused_softmax_kernel[grid](
        y, x,
        x.stride(0), y.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    return y
```

### 解析

**1. TODO 1: 寻找当前行的最大值**
- **实现方式**：`row_max = tl.max(row, axis=0)`
- **关键点**：`row` 在这里是一维向量，所以 `axis=0` 就是对整行做一次全归约；Triton 会把它降低成块内并行归约，而不是串行扫描。
- **技术细节**：这是 Safe Softmax 的第一步，通过减去最大值防止 `exp()` 计算时的数值溢出。这里使用 `tl.load(..., other=-float('inf'))` 填充越界位置，因此这些位置不会影响最大值的计算。

**2. TODO 2: 计算指数**
- **实现方式**：`numerator = tl.exp(safe_row)`，其中 `safe_row = row - row_max`
- **关键点**：对减去最大值后的数据计算指数，确保数值稳定性
- **技术细节**：数学上，
  `Softmax(x_i) = e^{x_i} / Σe^{x_j}`
  `= e^{x_i-m} / Σe^{x_j-m}`，
  所以减去最大值不会改变结果，只会避免大数值指数运算。越界位置的 `-inf - row_max` 仍为 `-inf`，其 `exp(-inf) = 0`。

**3. TODO 3: 求和计算分母**
- **实现方式**：`denominator = tl.sum(numerator, axis=0)`
- **关键点**：在 SRAM 内对指数值进行归约求和
- **技术细节**：越界位置的 `exp(-inf) = 0` 不会影响求和结果，这是 mask 处理的精妙之处——通过填充 `-inf` 而非 `0`，使得在 `max` 和 `sum` 两个阶段都能正确处理边界。

**4. TODO 4: 计算最终输出**
- **实现方式**：`softmax_output = numerator / denominator`
- **关键点**：逐元素除法得到归一化的 Softmax 概率分布
- **技术细节**：最终写回时使用 `mask` 过滤，确保只有有效位置被写入输出张量。

**工程优化要点**
- **内存访问优化**：整行数据只从 HBM 读取一次，所有计算（max、exp、sum、div）都在 SRAM 内完成，最后只写回一次，相比朴素实现减少了 2 次 HBM 往返。
- **数值稳定性**：Safe Softmax 通过减去最大值避免指数溢出，这是工业级实现的标准做法。
- **并行策略**：每个 Triton Program 处理一行，行间通常可并行，通常无需额外同步。
- **教学范围**：这里讨论的是单行 Softmax 的数值稳定实现，不是分块版 Softmax；如果 `N > BLOCK_SIZE`，应把它作为后续扩展而不是本节主线。
- **动态 num_warps 调整**：根据 BLOCK_SIZE 动态调整 warp 数量，在答案实现中针对大块尺寸（≥2048）增加并行度以提升性能。
- **Mask 处理技巧**：使用 `-inf` 作为越界填充值，利用其数学性质（`max` 时被忽略，`exp` 后为 0）优雅地处理边界情况，避免额外的条件分支。