# 04. Triton GEMM Tutorial | Triton 矩阵乘法 (GEMM) 与自动调优 (Autotune)

**难度：** Hard | **标签：** `Triton`, `GEMM`, `Compute Bound`, `Autotuning` | **目标人群：** 核心 Infra 与算子开发

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/03_CUDA_and_Triton_Kernels/04_Triton_GEMM_Tutorial.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


矩阵乘法 (GEMM: General Matrix Multiply) 是深度学习中最核心的计算操作，也是充分利用 GPU 算力的关键。本节我们将利用二维的 Thread Block，实现一个 $C = A \times B$ 的矩阵乘法 Kernel。并且，我们将引入 `@triton.autotune`，探索不同的 `BLOCK_M, BLOCK_N, BLOCK_K` 以及 Pipeline 级数 (`num_stages`) 对最终 **TFLOPs (每秒万亿次浮点运算)** 的影响。

### Step 1: 核心思想与痛点

> **2D 分块 (2D Tiling)**
> 计算 $C_{M \times N} = A_{M \times K} \times B_{K \times N}$。
> 我们不再给每一个元素分配一个线程，而是将 $C$ 矩阵划分为大小为 `BLOCK_M x BLOCK_N` 的小块。
> 每个 Triton Program 负责计算 $C$ 中的一个小块。
> 为了算出这一个小块，这个 Program 需要在 $K$ 的维度上循环遍历 $A$ 的一小块（`BLOCK_M x BLOCK_K`）和 $B$ 的一小块（`BLOCK_K x BLOCK_N`），进行**点积累加**。

> **Tensor Core 与 TFLOPs**
> 在 Triton 中，`accumulator += tl.dot(a_block, b_block)` 会被底层映射为专门调用 NVIDIA **Tensor Core (张量核心)** 的指令。
> 但是，到底切多大块最好？
> - 块太大：SRAM 塞不下，或者需要的寄存器 (Registers) 太多导致 Spilling (溢出到显存)。
> - 块太小：无法充分利用 Tensor Core 计算阵列。
> 因此，必须使用 `@triton.autotune` 搜索最佳配置。

### Step 2: 2D 分块与 SRAM 调度原理
相比于逐元素的加法，GEMM 是极度 Compute Bound 的。为了最大化利用 GPU Tensor Core，我们不仅要在网格（Grid）上按二维划分为 M_BLOCK 和 N_BLOCK，还要在内核内部循环遍历 K 维度。每次加载一小块 (M_BLOCK, K_BLOCK) 和 (K_BLOCK, N_BLOCK) 的矩阵到 SRAM，执行点积累加到相同的累加寄存器中。这保证了数据复用率最高。

### Step 3: 自动调优代码框架
Triton 的核心是装饰器 `@triton.autotune`。你可以在上方定义一组配置（如 `triton.Config`），列出不同的 `BLOCK_M`, `BLOCK_N`, `num_warps`, `num_stages` 的组合。Triton 编译器会在内核首次执行时，将所有配置暴力运行一遍，自动筛选出吞吐量最高的那组参数。

###  Step 4: 动手实战

**要求**：请补全下方 `gemm_kernel` 中双重循环和 `tl.dot` 的累加逻辑。并且观察 `@triton.autotune` 中提供的搜索空间。


```python
import torch
import triton
import triton.language as tl

# ==========================================
# Autotune: 搜索空间 (Search Space) 分析
# 这里列举了 8 种典型的配置组合，由 Triton 在运行时自动 Warmup 并选择最快的一个。
# num_stages 决定了软件流水线 (Software Pipelining) 的深度，用于隐藏内存加载延迟。
# num_warps 决定了分配给这个 Block 的线程束数量。
# ==========================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """计算矩阵乘法 C = A @ B."""
    # 1. L2 Cache 命中率优化：使用 Swizzle (交错) 映射策略分配 Program ID 到矩阵块
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 2. 定位当前 Program 处理的 A, B, C 块的数据指针
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # 3. 初始化累加器 accumulator，分配在极速的寄存器 (Register) 中
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 4. 循环 K 维度，每次加载 BLOCK_SIZE_K 大小的块到 SRAM
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # ==========================================
        # TODO 1: 加载 a 和 b 块，并用 0 填充越界的地方
        # ==========================================
        # a = ???
        # b = ???
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # ==========================================
        # TODO 2: 矩阵乘加 (底层调用 Tensor Core)
        # ==========================================
        # accumulator += ???
        accumulator += tl.dot(a, b)
        
        # ==========================================
        # TODO 3: 推进指针到下一个 K 块
        # ==========================================
        # a_ptrs += ???
        # b_ptrs += ???
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # 5. 写入 C 矩阵 (HBM)
    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(tl.float16), mask=c_mask)

def triton_gemm(a: torch.Tensor, b: torch.Tensor):
    # 检查维度和类型
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    
    # 预分配连续的输出张量
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    # 启动 1D Grid (包含所有计算块的扁平化数组)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c

```


```python
# 运行测试并进行 Benchmark 性能对比
def test_fused_gemm():
    if not torch.cuda.is_available():
        print("⏭️ 忽略测试：无 GPU。")
        return
        
    try:
        torch.manual_seed(42)
        # 测试大型矩阵以填满 GPU (A100 级别通常需要 4K x 4K 以上才能测出峰值)
        M, N, K = 4096, 4096, 4096
        a = torch.randn((M, K), device='cuda', dtype=torch.float16)
        b = torch.randn((K, N), device='cuda', dtype=torch.float16)
        
        # 1. 验证结果正确性
        triton_output = triton_gemm(a, b)
        torch_output = torch.matmul(a, b)
        
        # 矩阵乘法的绝对误差容忍度需要调高一点 (受浮点精度累加影响)
        diff = torch.max(torch.abs(triton_output - torch_output))
        print(f"最大误差: {diff.item():.6e}")
        assert diff < 1e-2, "Triton GEMM 结果不正确！"
        print("✅ Triton 分块 GEMM 逻辑正确！")
        
        # 2. Benchmark 对比 (吞吐量 TFLOPs)
        print("\n--- ⚡ 性能基准测试 (Benchmark: TFLOPs) ---")
        quantiles = [0.5, 0.2, 0.8]
        # 计算浮点运算次数 (2 * M * N * K)
        flops = 2 * M * N * K
        
        ms_pt, _, _ = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
        ms_tr, _, _ = triton.testing.do_bench(lambda: triton_gemm(a, b), quantiles=quantiles)
        
        # 转换为 TFLOPs (TeraFLOPs per second)
        tflops_pt = (flops / ms_pt) * 1e-9
        tflops_tr = (flops / ms_tr) * 1e-9
        
        print(f"PyTorch cuBLAS Time: {ms_pt:.4f} ms | Throughput: {tflops_pt:.1f} TFLOPs")
        print(f"Triton Autotune Time:{ms_tr:.4f} ms | Throughput: {tflops_tr:.1f} TFLOPs")
        print("✅ 所有测试通过！通过 @triton.autotune 搜索最佳的 BLOCK 切块与 Pipeline 配置，可以逼近 cuBLAS 的性能。")
        
    except NotImplementedError:
        print("请先完成 TODO 代码！")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

test_fused_gemm()

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
# ==========================================
# 💡 参考答案
# ==========================================

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """计算矩阵乘法 C = A @ B."""
    # 1. L2 Cache 命中率优化：使用 Swizzle (交错) 映射策略分配 Program ID 到矩阵块
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 2. 定位当前 Program 处理的 A, B, C 块的数据指针
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # 3. 初始化累加器 accumulator，分配在极速的寄存器 (Register) 中
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 4. 循环 K 维度，每次加载 BLOCK_SIZE_K 大小的块到 SRAM
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # ==========================================
        # TODO 1: 加载 a 和 b 块，并用 0 填充越界的地方
        # ==========================================
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # ==========================================
        # TODO 2: 矩阵乘加 (底层调用 Tensor Core)
        # ==========================================
        accumulator += tl.dot(a, b)
        
        # ==========================================
        # TODO 3: 推进指针到下一个 K 块
        # ==========================================
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # 5. 写入 C 矩阵 (HBM)
    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(tl.float16), mask=c_mask)

def triton_gemm(a: torch.Tensor, b: torch.Tensor):
    # 检查维度和类型
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    
    # 预分配连续的输出张量
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    # 启动 1D Grid (包含所有计算块的扁平化数组)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c

```

### 解析

**1. TODO 1: 加载 a 和 b 块到SRAM**
- **实现方式**：使用 `tl.load` 加载矩阵块，使用mask防止K维度越界
- **关键点**：mask机制确保不会读取超出K维度的数据，other=0.0填充越界位置
- **技术细节**：
  - `a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)` 加载A矩阵的一个块（BLOCK_M × BLOCK_K）
  - `b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)` 加载B矩阵的一个块（BLOCK_K × BLOCK_N）
  - mask计算剩余的K长度，防止最后一个块越界
  - 越界位置填充0.0，不会影响矩阵乘法的累加结果

**2. TODO 2: 矩阵乘加（调用Tensor Core）**
- **实现方式**：`accumulator += tl.dot(a, b)` 执行矩阵乘法并累加到accumulator
- **关键点**：`tl.dot` 是Triton最强大的函数之一，直接映射到GPU的Tensor Core硬件
- **技术细节**：
  - 对于FP16/BF16输入，`tl.dot` 编译后会生成Tensor Core的MMA（Matrix Multiply-Accumulate）指令
  - Tensor Core可以在单个周期内完成大规模的混合精度矩阵乘加计算
  - accumulator使用FP32精度累加，避免FP16累加时的精度损失
  - 在K维度上循环累加，实现完整的矩阵乘法：$C = \sum_{k} A[:, k] \times B[k, :]$

**3. TODO 3: 推进指针到下一个K块**
- **实现方式**：`a_ptrs += BLOCK_SIZE_K * stride_ak`，`b_ptrs += BLOCK_SIZE_K * stride_bk`
- **关键点**：指针按stride步进，移动到下一个K块的起始位置
- **技术细节**：
  - A矩阵沿K维度移动：每次前进BLOCK_SIZE_K列
  - B矩阵沿K维度移动：每次前进BLOCK_SIZE_K行
  - stride_ak和stride_bk是矩阵在内存中的步长，确保正确的内存访问模式
  - 循环结束后，完成了整个K维度的遍历和累加

**工程优化要点**
- **Swizzle映射优化**：使用GROUP_SIZE_M参数实现Program ID的交错映射，提高L2 Cache命中率。相邻的program处理相邻的矩阵块，增加数据复用
- **Autotune搜索空间**：提供8种不同的BLOCK配置，Triton在首次运行时自动测试所有配置并选择最快的。搜索空间包括BLOCK_SIZE（影响数据复用）、num_stages（软件流水线深度）、num_warps（线程束数量）
- **混合精度策略**：输入使用FP16节省带宽，累加器使用FP32保证精度，输出转回FP16节省存储。这是Tensor Core的标准使用模式
- **内存访问模式**：2D Tiling确保每个数据块被多次复用。A的每一行被复用N/BLOCK_N次，B的每一列被复用M/BLOCK_M次，最大化算术强度（计算量/访存量）
- **Tensor Core利用率**：BLOCK_SIZE的选择直接影响Tensor Core的利用率。过小无法喂饱计算单元，过大导致寄存器溢出。Autotune自动找到最佳平衡点
- **软件流水线**：num_stages控制流水线深度，通过预取下一个块的数据来隐藏内存延迟。更深的流水线可以提高吞吐，但需要更多SRAM
- **工业实践**：cuBLAS经过多年优化，性能极致。Triton通过Autotune可以达到cuBLAS 80-95%的性能，对于自定义GEMM变体（如稀疏矩阵乘、融合激活函数）具有更大优势