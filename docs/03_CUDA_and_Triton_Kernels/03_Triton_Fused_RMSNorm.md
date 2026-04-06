# 03 Triton Fused RMSNorm

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/03_CUDA_and_Triton_Kernels/03_Triton_Fused_RMSNorm.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 006. Triton 算子开发实战：Fused RMSNorm

**难度：** Hard | **标签：** `推理优化`, `Triton`, `Memory Bound` | **目标人群：** 核心 Infra 与算子开发

本节我们将步入大模型异构计算的深水区。我们将使用 OpenAI 的 **Triton** 语言，亲手编写一个在 GPU 上运行的 Fused RMSNorm 算子，并用 `do_bench` 证明它比 PyTorch 的原生实现更快！

> **相关阅读**:
> 本节使用 Triton 实现了底层的极致显存与计算优化。
> 如果你对该算子的数学公式推导和纯 PyTorch 高层结构还不熟悉，建议先复习 PyTorch 篇：
>  [`../02_PyTorch_Algorithms/01_RMSNorm_Tutorial.ipynb`](../02_PyTorch_Algorithms/01_RMSNorm_Tutorial.ipynb)


### Step 1: 核心思想与痛点

> **PyTorch 原生 RMSNorm 的性能瓶颈在哪里？**
> 在 PyTorch 中，`x^2`、`mean`、`rsqrt` 和 `x * weight` 是多个独立的算子。每次计算，GPU 都需要把中间结果写回到显存（HBM/VRAM），然后再读出来进行下一步计算。
> 这种频繁的读写导致它严重受限于**内存带宽 (Memory Bound)**，而不是计算能力 (Compute Bound)。
>
> **Triton Fused Kernel 的本质：**
> “算子融合 (Operator Fusion)”。我们写一个底层的 GPU Kernel，让一个线程块（Block）把一行数据一次性读到超高速的片上内存（SRAM）中，在 SRAM 里算完均方根和缩放，最后只把结果写回一次显存。这极大地节约了访存时间。


### Step 2: 归约 代码框架
内核沿着隐藏层维度（特征维度）分配指针。读取一整行的特征数据后，利用 `tl.sum(x * x, axis=0)` 快速求出平方和。接着除以维度长度得到方差，算出 `rsqrt`，再用它将原特征标准化。最后乘以来自 HBM 的可学习参数 `weight` 块，并写回结果。


###  Step 3: 核心公式与 Triton 编程模型

**RMSNorm 公式回顾：**
$$ y = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma $$

**Triton 编程范式：**
1. `tl.program_id(0)`: 获取当前处理的行号（Row Index）。在大模型中，通常一行代表一个 Token 的特征向量（维度为 $d$）。
2. `tl.arange(0, BLOCK_SIZE)`: 生成列的偏移量（Offsets）。
3. `tl.load(ptr, mask)`: 从显存（HBM）加载数据到片上内存（SRAM）。
4. `tl.sum(x, axis=0)`: 在 SRAM 内进行高效的规约计算（Reduction）。
5. `tl.store(ptr, value, mask)`: 将计算结果写回显存。


###  Step 4: 动手实战

**要求**：请补全下方 `_rmsnorm_fwd_fused` 的 Triton Kernel。
**注意：**
1. Triton 的并行粒度是我们自己控制的。在这个例子中，我们分配 1 个 Program 负责计算 1 行（即 1 个 Token 的特征）。


```python
import torch
import triton
import triton.language as tl

@triton.jit
def _rmsnorm_fwd_fused(
    X_ptr,        # 输入矩阵 X 的指针 [M, N]
    Y_ptr,        # 输出矩阵 Y 的指针 [M, N]
    W_ptr,        # 权重向量 W 的指针 [N]
    stride_x_row, # X 矩阵行之间的步长 (Stride)
    stride_y_row, # Y 矩阵行之间的步长
    N,            # 每一行的元素个数 (特征维度 d)
    eps,          # RMSNorm 的 epsilon
    BLOCK_SIZE: tl.constexpr, # Triton 的块大小 (必须是 2 的幂)
):
    # 1. 确定当前 Program 处理的是哪一行
    row_idx = tl.program_id(0)
    
    # 2. 定位到当前行的起始地址
    row_start_ptr = X_ptr + row_idx * stride_x_row
    y_row_start_ptr = Y_ptr + row_idx * stride_y_row
    
    # 3. 生成列偏移量 [0, 1, ..., BLOCK_SIZE-1]
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # 因为 N 可能不是 2 的幂，所以需要 mask 来防止越界访问
    mask = col_offsets < N
    
    # ==========================================
    # TODO 1: 从显存加载当前行的数据 x，和权重 w 到 SRAM
    # 提示: tl.load(指针 + 偏移, mask=mask, other=0.0)
    # ==========================================
    # x = ???
    # w = ???
    
    # ==========================================
    # TODO 2: 在 SRAM 中计算平方和与均方根 (RMS)
    # 1. x_sq = x * x
    # 2. 强制转换 x_sq 为 float32 (tl.math.cast 或是转义) 防溢出
    # 3. 使用 tl.sum 求和
    # 4. 计算 rsqrt: tl.math.rsqrt( sum / N + eps )
    # ==========================================
    # x_sq = ???
    # ...
    # rsqrt = ???
    
    # ==========================================
    # TODO 3: 归一化并乘以权重，最后写回显存
    # y = x * rsqrt * w
    # tl.store(y_row_start_ptr + col_offsets, y, mask=mask)
    # ==========================================
    # y = ???
    # ???
    pass

def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    # 确保输入是连续的
    x = x.contiguous()
    y = torch.empty_like(x)
    
    # 展平前 N-1 维，将矩阵视为 [M, N]
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    
    # 寻找大于 N 的最小的 2 的幂作为 BLOCK_SIZE
    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(N)
    assert BLOCK_SIZE <= MAX_FUSED_SIZE, "特征维度过大，请使用分块计算版本！"
    
    # 启动 Triton Kernel
    grid = (M,)  # 分配 M 个线程块，每个处理一行
    _rmsnorm_fwd_fused[grid](
        x, y, weight,
        x.stride(0) if x.ndim > 1 else 0,
        y.stride(0) if y.ndim > 1 else 0,
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y

```

```python
# 运行此单元格以测试你的实现
def test_triton_rmsnorm():
    try:
    if not torch.cuda.is_available():
        print("⏭️ 忽略测试：无 GPU。本节编译和运行 Triton/CUDA 算子必须依赖 NVIDIA GPU。")
        return
        
        # 1. 功能性测试 (Correctness)
        print("1. 测试数值正确性...")
        torch.manual_seed(0)
        M, N = 4096, 4096
        x = torch.randn(M, N, device='cuda', dtype=torch.float16)
        weight = torch.ones(N, device='cuda', dtype=torch.float16)
        eps = 1e-6
        
        # PyTorch Native
        x_f32 = x.float()
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        x_norm = x_f32 * torch.rsqrt(variance + eps)
        out_torch = (weight.float() * x_norm).half()
        
        # Triton
        out_triton = triton_rmsnorm(x, weight, eps)
        
        # 检查容差
        assert torch.allclose(out_torch, out_triton, atol=1e-3, rtol=1e-3), "Triton 算子输出与 PyTorch 不一致！"
        print("✅ 数值正确！")
        
        # 2. 性能基准测试 (Performance Benchmark)
        print("\n2. 性能基准测试 (Performance Benchmark)...")
        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=['N'],
                x_vals=[1024 * i for i in range(2, 10)],
                line_arg='provider',
                line_vals=['pytorch', 'triton'],
                line_names=['PyTorch Native', 'Triton Fused'],
                styles=[('blue', '-'), ('green', '-')],
                ylabel='GB/s',
                plot_name='RMSNorm Performance',
                args={'M': 4096},
            )
        )
        def benchmark(M, N, provider):
            x = torch.randn(M, N, device='cuda', dtype=torch.float16)
            weight = torch.ones(N, device='cuda', dtype=torch.float16)
            eps = 1e-6
            quantiles = [0.5, 0.2, 0.8]
            
            if provider == 'pytorch':
                def y_fwd():
                    variance = x.float().pow(2).mean(-1, keepdim=True)
                    return (x.float() * torch.rsqrt(variance + eps) * weight.float()).half()
                ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles)
            if provider == 'triton':
                ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_rmsnorm(x, weight, eps), quantiles=quantiles)
            
            gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6 # 读写两次显存
            return gbps(ms), gbps(max_ms), gbps(min_ms)
        
        print("正在运行 Benchmark，请稍候...")
        benchmark.run(print_data=True, show_plots=False)
        print("\n✅ All Tests Passed! 恭喜你，工业级 Triton 加速算子开发成功！")
        
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
    except RuntimeError as e:
        print(f"❌ 运行时错误 (CUDA/Triton 异常): {e}\n你可能需要一张拥有 CUDA 核心的 NVIDIA 显卡来运行它。")
    except Exception as e:
        print(f"❌ 发生未知异常: {e}")

test_triton_rmsnorm()

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

::: details 💡 点击查看官方解析与参考代码

### 📝 RMSNorm 参考实现解析

1. **并行策略**: 按行 (Row) 分配给不同的 Triton Block。`M` 行启动 `M` 个 Block。
2. **加载数据**: 使用 `x.to(tl.float32)` 将可能为 FP16/BF16 的数据转成 FP32，**这极其重要，因为平方求和在 FP16 下极易溢出。**
3. **计算 RMS**: 
   - 使用 `tl.sum` 求整行的平方和。
   - `tl.math.rsqrt` 用于计算平方根的倒数（即 $1/\sqrt{x}$），这比先开方再除法在硬件指令上更快。
4. **输出**: 将计算完的 `y` 强转回原本的 `dtype`（如 FP16）并利用 `tl.store` 写回显存。

```python
# ==========================================
# 💡 参考答案
# ==========================================

import torch
import triton
import triton.language as tl

@triton.jit
def _rmsnorm_kernel(
    x_ptr, y_ptr, w_ptr,
    stride_x_row, stride_y_row,
    N, eps,
    BLOCK_N: tl.constexpr
):
    # 1. 定位当前行
    row_idx = tl.program_id(0)
    x_row_start_ptr = x_ptr + row_idx * stride_x_row
    y_row_start_ptr = y_ptr + row_idx * stride_y_row
    
    col_offsets = tl.arange(0, BLOCK_N)
    mask = col_offsets < N
    
    # ==========================================
    # 1. 从显存加载数据到 SRAM
    # ==========================================
    x = tl.load(x_row_start_ptr + col_offsets, mask=mask, other=0.0)
    w = tl.load(w_ptr + col_offsets, mask=mask, other=0.0)
    
    # ==========================================
    # 2. 计算均方根倒数 (rsqrt)
    # 关键：转为 float32 累加防止溢出
    # ==========================================
    x_f32 = x.to(tl.float32)
    x_sq = x_f32 * x_f32
    sum_sq = tl.sum(x_sq, axis=0)
    rsqrt = tl.math.rsqrt((sum_sq / N) + eps)
    
    # ==========================================
    # 3. 归一化并乘权重，转回原类型后写入显存
    # ==========================================
    y = x_f32 * rsqrt * w
    y_out = y.to(x.dtype)
    tl.store(y_row_start_ptr + col_offsets, y_out, mask=mask)

def triton_rmsnorm(x, weight, eps=1e-5):
    M, N = x.shape
    y = torch.empty_like(x)
    
    # 选择 BLOCK_N (N 的下一个 2 的幂)
    BLOCK_N = triton.next_power_of_2(N)
    
    # grid 维度设置：M 行，启动 M 个 block
    grid = (M,)
    
    _rmsnorm_kernel[grid](
        x, y, weight,
        x.stride(0), y.stride(0),
        N, eps,
        BLOCK_N=BLOCK_N,
    )
    return y

```

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
