# 06. Triton Fused Softmax | Triton 进阶：跨线程归约与数值稳定 (Safe Softmax)

**难度：** Hard | **标签：** `Triton`, `Reduction`, `Attention` | **目标人群：** 核心 Infra 与算子开发

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/03_CUDA_and_Triton_Kernels/06_Triton_Fused_Softmax.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


在纯 Python 模拟 FlashAttention 时，我们探讨过 Softmax。标准的 Softmax 是 Memory Bound 的痛点，因为它需要读取整个行三次：寻找最大值、计算指数和、计算除法。
通过 Triton 的 SRAM，我们可以将**一整行 (Row)** 加载到片上缓存，在 SRAM 内部完成 `max`, `exp`, `sum` 以及除法运算，最终只写回显存一次！
本节我们将实现一个高效率、数值稳定的 Fused Softmax 算子。

### Step 1: Softmax 数值稳定性与 Triton 归约

> **数值稳定性 (Safe Softmax)：**
> 如果直接计算 $e^x$，当 $x$ 较大时（如 50），$e^{50}$ 会导致浮点数溢出 (NaN)。
> 解决方案：让一行的每一个元素都减去该行的最大值 $m$。
> $Soft\max(x_i) = \frac{e^{x_i - m}}{\sum e^{x_j - m}}$，这在数学上完全等价，但在计算机浮点表示中更加安全。

> **Triton 的行级并行：**
> 处理形状为 `(M, N)` 的矩阵时，通常分配**一个 Program (线程块) 专门处理矩阵的一行**。
> - `pid = tl.program_id(0)` 获取行号。
> - 计算该行在内存中的起始指针。
> - 将该行全部 Load 进 SRAM（若行极长，需要切块循环归约，为简化教学本节假设 N 小于 BLOCK_SIZE 使得一行能完全载入 SRAM）。
> - 使用 `tl.\max(x, axis=0)` 和 `tl.sum(x, axis=0)` 在 SRAM 内进行高效归约 (Reduction)。

### Step 2: 跨线程安全 Softmax 归约算法
朴素的 Softmax 公式是 $e^{x_i} / \sum e^{x_j}$，但这很容易导致指数爆炸（例如 $e^{100}$ 会变成 NaN）。安全版本（Safe Softmax）是先求出这一行的最大值 $M$，然后计算 $e^{x_i - M}$。由于 Triton 只能在块内做 `tl.max` 和 `tl.sum`，我们需要巧妙地应用局部跨线程归约来处理这一数学变换。

### Step 3: 代码实现框架
内部分配指针时，我们需要指向二维张量的特定一行。接着使用 `x_max = tl.\max(x, axis=0)` 提取行最大值。将其从原数据中减去：`num = tl.exp(x - x_max)`，再求和得到 `denom = tl.sum(num, axis=0)`，最后相除即为最终分布。

###  Step 4: 动手实战

**要求**：请补全下方 `fused_softmax_kernel`。假设输入矩阵大小为 `(M, N)`，为了教学简化，我们假设 `N` 足够小，可以一次性被 `BLOCK_SIZE` 装入 SRAM 中（这是 FlashAttention 分块大小的常见情况）。


```python
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
    # 注意乘上 row_stride (每一行的步长)，因为在物理内存中，二维张量是展平的一维数组
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    # 3. 构造这一行的连续索引和掩码 (防越界)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    
    # 4. 加载一整行到 SRAM 中。越界的部分补一个极小的负数 (比如 -float('inf'))，防止影响求 max
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # ==========================================
    # TODO 1: 寻找当前行的最大值 (安全 Softmax 第一步)
    # ==========================================
    # row_max = ???
    row_max = tl.zeros((1,), dtype=tl.float32)  # 占位初始化
    
    safe_row = row  # 占位初始化
    
    # ==========================================
    # TODO 2: 计算指数 (Numerator)
    # ==========================================
    # numerator = ???
    numerator = safe_row  # 占位初始化
    
    # ==========================================
    # TODO 3: 求和 (Denominator)。注意只有 mask 内的值才能参与计算
    # ==========================================
    # denominator = ???
    denominator = tl.zeros((1,), dtype=tl.float32) + 1.0  # 占位初始化（避免除零）
    
    # ==========================================
    # TODO 4: 计算最终输出，并存回显存
    # ==========================================
    # softmax_output = ???
    softmax_output = numerator  # 占位初始化
    
    # 定位输出指针，写回
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

def triton_softmax(x: torch.Tensor):
    M, N = x.shape
    # 分配输出内存
    y = torch.empty_like(x)
    
    # 寻找大于 N 的最小 2 的幂次方作为 BLOCK_SIZE，保证一行能一次性塞入
    # 如果 N 太大，这里会抛出内存限制异常，此时需要分块循环
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    # 启动 M 个 program
    grid = (M, )
    
    fused_softmax_kernel[grid](
        y, x,
        x.stride(0), y.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y
```


```python
# 测试你的实现
def test_fused_softmax():
    if not torch.cuda.is_available():
        print("⏭️ 忽略测试：无 GPU。")
        return
        
    try:
        torch.manual_seed(42)
        # 测试数据: 8192 行，每行 1000 个元素
        M, N = 8192, 1000
        x = torch.randn(M, N, device='cuda')
        
        # PyTorch 原生 Softmax
        y_ref = torch.softmax(x, axis=1)
        
        # Triton 实现
        y_tri = triton_softmax(x)
        
        diff = torch.max(torch.abs(y_ref - y_tri))
        print(f"最大误差: {diff.item():.6e}")
        assert diff < 1e-5, "Triton Softmax 计算结果不准确！"
        
        print("✅ 跨线程归约的数值稳定 Softmax 算子实现成功！")
        
    
        print("\n--- 性能基准测试 (Benchmark) ---")
        quantiles = [0.5, 0.2, 0.8]
        ms_pt, min_ms_pt, max_ms_pt = triton.testing.do_bench(lambda: torch.softmax(x, axis=1), quantiles=quantiles)
        ms_tr, min_ms_tr, max_ms_tr = triton.testing.do_bench(lambda: triton_softmax(x), quantiles=quantiles)
        print(f"PyTorch Time: {ms_pt:.4f} ms")
        print(f"Triton Time:  {ms_tr:.4f} ms")
        print(f"Speedup:      {ms_pt / ms_tr:.2f}x")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

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
- **关键点**：使用 Triton 的归约操作在 SRAM 内高效计算行最大值
- **技术细节**：这是 Safe Softmax 的第一步，通过减去最大值防止 `exp()` 计算时的数值溢出。由于越界位置在 `tl.load` 时填充了 `-inf`，它们不会影响最大值的计算。

**2. TODO 2: 计算指数**
- **实现方式**：`numerator = tl.exp(safe_row)`，其中 `safe_row = row - row_max`
- **关键点**：对减去最大值后的数据计算指数，确保数值稳定性
- **技术细节**：$e^{x_i - m}$ 在数学上等价于 $e^{x_i} / e^m$，但前者避免了大数值的指数运算。越界位置的 `-inf - row_max` 仍为 `-inf`，其 `exp(-inf) = 0`。

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
- **并行策略**：每个 Triton Program 处理一行，行间完全并行，无需同步。
- **动态 num_warps 调整**：根据 BLOCK_SIZE 动态调整 warp 数量，在答案实现中针对大块尺寸（≥2048）增加并行度以提升性能。
- **Mask 处理技巧**：使用 `-inf` 作为越界填充值，利用其数学性质（`max` 时被忽略，`exp` 后为 0）优雅地处理边界情况，避免额外的条件分支。