# 04 Triton GEMM Tutorial

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/03_CUDA_and_Triton_Kernels/04_Triton_GEMM_Tutorial.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 04. GPU 编程的皇冠明珠：矩阵乘法 (GEMM) 与分块搜索空间 (Autotune)

**难度：** Hard | **标签：** `Triton`, `GEMM`, `Compute Bound`, `Autotuning` | **目标人群：** 核心 Infra 与算子开发

如果说逐元素操作（如 SwiGLU、RMSNorm）是为了突破**显存墙 (Memory Bound)**，那么矩阵乘法 (GEMM: General Matrix Multiply) 则是为了榨干 GPU 恐怖的**算力墙 (Compute Bound)**。
本节我们将利用二维的 Thread Block，实现一个 $C = A \times B$ 的矩阵乘法 Kernel。并且，我们将引入 `@triton.autotune`，探索不同的 `BLOCK_M, BLOCK_N, BLOCK_K` 以及 Pipeline 级数 (`num_stages`) 对最终 **TFLOPs (每秒万亿次浮点运算)** 的巨大影响！


### Step 1: 核心思想与痛点

> **2D 分块 (2D Tiling)**
> 计算 $C_{M \times N} = A_{M \times K} \times B_{K \times N}$。
> 我们不再给每一个元素分配一个线程，而是将 $C$ 矩阵划分为大小为 `BLOCK_M x BLOCK_N` 的小块。
> 每个 Triton Program 负责计算 $C$ 中的一个小块。
> 为了算出这一个小块，这个 Program 需要在 $K$ 的维度上循环遍历 $A$ 的一小块（`BLOCK_M x BLOCK_K`）和 $B$ 的一小块（`BLOCK_K x BLOCK_N`），进行**点积累加**。

> **Tensor Core 与 TFLOPs 瓶颈**
> 在 Triton 中，`accumulator += tl.dot(a_block, b_block)` 会被底层映射为专门调用 NVIDIA **Tensor Core (张量核心)** 的指令。
> 但是，到底切多大块最好？
> - 块太大：SRAM 塞不下，或者需要的寄存器 (Registers) 太多导致 Spilling (溢出到显存)。
> - 块太小：无法喂饱庞大的 Tensor Core 计算阵列。
> 因此，必须使用 `@triton.autotune` 穷举搜索最佳配置！


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
        print("💡 在工业界，利用 @triton.autotune 穷举搜索最佳的 BLOCK 切块与 Pipeline，是算子工程师逼近 cuBLAS 硬件极限算力的必经之路。")
        
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

::: details 💡 点击查看官方解析与参考代码

### 📝 GEMM 参考实现解析

1. **掩码控制**: 在加载内存块 `a` 和 `b` 时，必须计算剩余的 `K` 长度防止越界。超出边界的地方补 `other=0.0`，这样在做矩阵乘时不会改变累加器的值。
2. **矩阵乘核心 (`tl.dot`)**: `tl.dot(a, b)` 是 Triton 最强大的函数之一，如果 `a` 和 `b` 是 FP16 或 BF16 类型，这行代码在编译后会被映射为底层 GPU 的 Tensor Core (MMA 指令)，可以在单个周期内完成巨大的混合精度矩阵乘加计算。
3. **指针步进**: 在每次计算完 `BLOCK_SIZE_K` 的循环后，`a_ptrs` 和 `b_ptrs` 会根据各自的 `stride` 前进，以加载下一批数据块，如此反复累加完成内积运算。

```python
# ==========================================
# 💡 参考答案
# ==========================================

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 1. 计算掩码并加载数据
        # 提示：由于 M 和 N 在计算 offs_am 和 offs_bn 时已处理过取模，这里主要防止 K 维度越界
        mask_a = offs_k[None, :] < K - k * BLOCK_SIZE_K
        mask_b = offs_k[:, None] < K - k * BLOCK_SIZE_K
        
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        
        # 2. 矩阵乘法
        # 这里底层会自动映射到 Tensor Core 指令 (如 mma.sync)
        accumulator += tl.dot(a, b)
        
        # 3. 移动指针到下一个 K 块
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

```

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
