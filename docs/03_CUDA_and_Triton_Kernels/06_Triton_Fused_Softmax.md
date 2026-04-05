# 06 Triton Fused Softmax

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/03_CUDA_and_Triton_Kernels/06_Triton_Fused_Softmax.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 06. Triton 进阶：跨线程归约与数值稳定 (Safe Softmax)

**难度：** Hard | **标签：** `Triton`, `Reduction`, `Attention` | **目标人群：** 核心 Infra 与算子开发

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
    row_max = tl.max(row, axis=0)
    
    # 减去最大值，避免 exp() 溢出
    safe_row = row - row_max
    
    # ==========================================
    # TODO 2: 计算指数 (Numerator)
    # ==========================================
    numerator = tl.exp(safe_row)
    
    # ==========================================
    # TODO 3: 求和 (Denominator)。注意只有 mask 内的值才能参与计算
    # 在 Load 时 other 补的是 -inf，其 exp(-inf) = 0，不影响 sum
    # ==========================================
    denominator = tl.sum(numerator, axis=0)
    
    # ==========================================
    # TODO 4: 计算最终输出，并存回显存
    # ==========================================
    softmax_output = numerator / denominator
    
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
        
    
        print("\n--- ⚡ 性能基准测试 (Benchmark) ---")
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

::: details 💡 点击查看官方解析与参考代码

### Fused Softmax 参考实现解析

1. **并行策略**: 分配每个 Triton Block 处理矩阵的一行，这意味着 `row_max`, `sum` 都只需要沿 `axis=0` 执行局部归约即可。
2. **处理溢出 (`Safe Softmax`)**: 寻找每行的最大值并将其从各个元素中减去，再求 `exp()`，这是为了防止浮点数溢出，在深度学习中这是一个非常基础但关键的技巧。
3. **Mask 的精妙处理**: 越界的元素我们 `load` 了 `-inf`。`-inf` 减去 `row_max` 依然是负无穷；而 $e^{-\infty} = 0$，所以在 `tl.sum` 的阶段自然加的是 0，完全不会干扰分母计算。计算最终结果时越界位置也会得到 $0/1 = 0$，最终写回时被 `mask` 过滤掉。

```python
# ==========================================
# 💡 参考答案
# ==========================================

import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    input_row_start_ptr = input_ptr + row_idx * input_row_stride
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = input_row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # 1. 求最大值以确保数值稳定
    row_max = tl.max(row, axis=0)
    safe_row = row - row_max
    
    # 2. 计算指数
    numerator = tl.exp(safe_row)
    
    # 3. 计算分母（越界部分由于 load 了 -inf，exp 后是 0，不影响 sum）
    denominator = tl.sum(numerator, axis=0)
    
    # 4. 最终归一化与写回
    softmax_output = numerator / denominator
    
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
    
    softmax_kernel[grid](
        x, y,
        x.stride(0), y.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    return y

```

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
