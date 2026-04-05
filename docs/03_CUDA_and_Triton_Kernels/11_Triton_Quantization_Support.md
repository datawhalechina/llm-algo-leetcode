# 11 Triton Quantization Support

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/03_CUDA_and_Triton_Kernels/11_Triton_Quantization_Support.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 11. Triton 量化算子：W8A16 权重量化融合矩阵乘法 (Quantization GEMM)

**难度：** Hard | **标签：** `Triton`, `Quantization`, `GPTQ` | **目标人群：** 核心 Infra 与算子开发

在模型部署中，**Weight-Only 量化** (例如 W8A16 或 W4A16) 是最普遍的显存优化手段。由于激活值 (Activation) 依然保持 FP16，所以传统的 PyTorch 需要在计算前显式地把权重反量化回 FP16，这不仅慢，还抵消了显存带来的带宽优势。

以 GPTQ/AWQ 为代表的现代量化框架，底层的核心技术是 **On-the-fly Dequantization (即时反量化)**。
本节我们将编写一个 Triton 算子：在 SRAM 中读入 INT8 的权重和 FP16 的缩放因子 (Scales)，在寄存器里动态反量化为 FP16 后，立即与激活值相乘。


> **相关阅读**:  
> 如果你对量化的数学公式推导和纯 PyTorch 实现还不熟悉，建议先复习 PyTorch 篇：
> [`../02_PyTorch_Algorithms/16_Quantization_W8A16.ipynb`](../02_PyTorch_Algorithms/16_Quantization_W8A16.ipynb)

### Step 1: 融合反量化矩阵乘法的核心思想

> **计算公式：**
> 输入特征矩阵 $X$ (FP16)，量化权重矩阵 $W_{int8}$ (INT8)，每列的缩放比例 $S$ (FP16)。
> $Y = X \times (W_{int8} \times S)$
> 注意，我们不生成庞大的 $W_{fp16}$，而是将融合后的计算直接放入 `tl.dot()` 中。

> **为什么要融合？**
> - **传统方式**：先把整个 $W_{int8}$ 反量化为 $W_{fp16}$ (需要额外的 HBM 读写)，然后做标准的 FP16 GEMM。
> - **融合方式**：在 Triton 内核的 SRAM 中，每次只加载一小块 $W_{int8}$，立即在寄存器里转成 FP16 并乘以 Scale，然后直接参与 `tl.dot` 累加。全程不产生额外的 HBM 访问。

### Step 2: SRAM 内反量化的执行流程

在 Triton 分块 GEMM 的最内层循环中，每一轮迭代的操作如下：

1. **加载 X 块** (FP16)：从 HBM 读入一小块输入特征矩阵到 SRAM。
2. **加载 W 块** (INT8)：从 HBM 读入一小块量化权重到 SRAM。注意此时读取的数据量只有 FP16 的一半。
3. **类型转换与缩放**：在 SRAM/寄存器中，利用 `w.to(tl.float16)` 将 INT8 转为浮点型，再乘以对应列的缩放因子 $S$，得到 $W_{fp16}$。
4. **矩阵乘累加**：执行标准的 `tl.dot(X, W_fp16)` 并累加到结果中。

关键点：反量化操作发生在 SRAM 内部，不产生额外的 HBM 读写开销。

### Step 3: 内核函数签名与数据布局

```
w8a16_gemm_kernel(x_ptr, w_int8_ptr, scales_ptr, y_ptr, M, N, K, ...)
```

- **x_ptr**: 输入特征矩阵，形状 `(M, K)`，数据类型 FP16
- **w_int8_ptr**: 量化权重矩阵，形状 `(K, N)`，数据类型 INT8
- **scales_ptr**: 每列的缩放因子，形状 `(N,)`，数据类型 FP16
- **y_ptr**: 输出矩阵，形状 `(M, N)`，数据类型 FP16

Grid 划分为 2D：`(ceil(M/BLOCK_M), ceil(N/BLOCK_N))`，每个 Triton Program 负责输出矩阵中一个 `(BLOCK_M, BLOCK_N)` 大小的子块。

###  Step 4: 动手实战

**要求**：请补全下方 `w8a16_gemm_kernel`。我们需要将 INT8 的权重即时转为 FP16 并完成矩阵乘法。为了简化，我们使用按列量化 (Per-channel Quantization)。


```python
import torch
import triton
import triton.language as tl

@triton.jit
def w8a16_gemm_kernel(
    x_ptr, w_int8_ptr, scales_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 1. 确定当前处理的输出块 (Block_M, Block_N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 获取输出块对应的 M 维度和 N 维度的指针偏移
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # 提前计算指向 scale 数组的偏移 (因为 scale 长度为 N)
    scale_ptrs = scales_ptr + offs_n
    
    # 初始化累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # 2. 沿着 K 维度进行循环归约
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        
        # 计算 X 和 W 的数据指针
        # X: (BLOCK_M, BLOCK_K)
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        # W: (BLOCK_K, BLOCK_N)
        w_ptrs = w_int8_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
        
        # 加载数据 (假设矩阵维度都是 block 的整数倍，省略掩码)
        x = tl.load(x_ptrs)
        w_int8 = tl.load(w_ptrs)
        
        # ==========================================
        # TODO 1: 在 SRAM 中进行动态反量化
        # 1. 将 w_int8 转换为 tl.float16 (或与 x 相同的类型)
        # 2. 加载对应的 scales 缩放因子
        # 3. 将权重与缩放因子相乘。注意 scales 是一维的 (BLOCK_N,)，需要利用广播机制乘以 W 的每一列
        # ==========================================
        # w_fp16 = ???
        w_fp16 = w_int8.to(x.dtype)
        scales = tl.load(scale_ptrs)
        w_fp16 = w_fp16 * scales[None, :]
        
        # ==========================================
        # TODO 2: 执行点积并累加
        # ==========================================
        # acc = ???
        acc += tl.dot(x, w_fp16)
        
    # 3. 写回显存 (转换为 FP16)
    y_ptrs = y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    tl.store(y_ptrs, acc.to(tl.float16))

def triton_w8a16_gemm(x: torch.Tensor, w_int8: torch.Tensor, scales: torch.Tensor):
    M, K = x.shape
    _, N = w_int8.shape
    
    y = torch.empty((M, N), device=x.device, dtype=torch.float16)
    
    BLOCK_M = 16
    BLOCK_N = 64
    BLOCK_K = 64
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    w8a16_gemm_kernel[grid](
        x, w_int8, scales, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w_int8.stride(0), w_int8.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return y

```

```python
# 测试你的实现
def test_w8a16_gemm():
    if not torch.cuda.is_available():
        print("⏭️ 忽略测试：无 GPU。")
        return
        
    try:
        torch.manual_seed(42)
        M, N, K = 32, 256, 128
        
        # 构造输入
        x = torch.randn(M, K, device='cuda', dtype=torch.float16)
        
        # 模拟 INT8 权重 (数值范围 -128 到 127)
        w_int8 = torch.randint(-128, 127, (K, N), device='cuda', dtype=torch.int8)
        
        # 模拟 FP16 缩放比例 (每列一个 scale)
        scales = torch.randn(N, device='cuda', dtype=torch.float16) * 0.01
        
        # 1. PyTorch 原生参考计算 (需要显式反量化，占用大量额外显存)
        w_fp16_ref = w_int8.to(torch.float16) * scales.unsqueeze(0)
        y_ref = x @ w_fp16_ref
        
        # 2. Triton 融合计算
        y_tri = triton_w8a16_gemm(x, w_int8, scales)
        
        # 3. 验证结果
        diff = torch.max(torch.abs(y_ref - y_tri))
        print(f"最大误差: {diff.item():.6e}")
        assert diff < 1e-3, "Triton W8A16 量化 GEMM 结果不正确！"
        
        print("✅ 完美！你实现了 GPTQ/AWQ 等量化框架最核心的即时反量化算子！")
        print("💡 在面试中，解释清楚为何 'On-the-fly' 转换能绕开 HBM 带宽瓶颈，将极大展现你的架构深度。")
        
    
        print("\n--- ⚡ 性能基准测试 (Benchmark) ---")
        # 典型的 LLM Linear 层尺寸
        M, N, K = 4096, 4096, 4096
        
        x_l = torch.randn(M, K, device='cuda', dtype=torch.float16)
        w_int8_l = torch.randint(-128, 127, (K, N), device='cuda', dtype=torch.int8)
        scales_l = torch.randn(N, device='cuda', dtype=torch.float16) * 0.01
        
        # 为了公平对比，PyTorch 需要预先反量化权重
        w_fp16_l = w_int8_l.to(torch.float16) * scales_l.unsqueeze(0)
        
        quantiles = [0.5, 0.2, 0.8]
        
        # PyTorch 执行纯 FP16 的 GEMM
        ms_pt, _, _ = triton.testing.do_bench(lambda: x_l @ w_fp16_l, quantiles=quantiles)
        
        # Triton 执行即时反量化并乘加的 GEMM
        ms_tr, _, _ = triton.testing.do_bench(lambda: triton_w8a16_gemm(x_l, w_int8_l, scales_l), quantiles=quantiles)
        
        print(f"PyTorch FP16xFP16 GEMM Time:     {ms_pt:.4f} ms")
        print(f"Triton W8A16 On-the-fly GEMM:    {ms_tr:.4f} ms")
        print(f"Speedup vs Standard FP16:        {ms_pt / ms_tr:.2f}x")
        print("💡 结论：在 Memory Bound 极强的 Linear 层中，读取 1/2 体积的 INT8 权重，即使需要加上额外的 SRAM 内反量化计算，整体端到端速度依然可能更快！")
    except NotImplementedError:
        print("请先完成 TODO 代码！")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

test_w8a16_gemm()

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

::: details 💡 点击查看官方解析与参考代码

### 💡 参考解答：Triton 量化算子 W8A16 融合 GEMM

在这个量化算子的实现中，最核心的操作是**即时反量化 (On-the-fly Dequantization)**。它的优势在于将昂贵的 HBM 显存读写转换为了极速的 SRAM/寄存器内计算：
1. **轻量级加载**：通过 `tl.load(w_ptrs)` 加载 INT8 格式的权重，这使得这一步的显存读取量仅为原本 FP16 的一半，极大地缓解了 Memory Bound 的瓶颈。
2. **SRAM 内计算**：`w_int8.to(x.dtype)` 以及随后的 `w_fp16 * scales[None, :]` 这两步操作，完全发生在高速的流处理器寄存器中。虽然增加了浮点类型转换和乘法的指令开销，但在 IO 瓶颈的大模型推理场景下是完全值得的。
3. **融合乘加**：反量化后的结果直接投入 `tl.dot(x, w_fp16)` 参与矩阵乘，从始至终没有任何反量化后的 FP16 权重数组被写回到主存。

```python
import torch
import triton
import triton.language as tl

@triton.jit
def w8a16_gemm_kernel(
    x_ptr, w_int8_ptr, scales_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 1. 确定当前处理的输出块 (Block_M, Block_N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 获取输出块对应的 M 维度和 N 维度的指针偏移
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # 提前计算指向 scale 数组的偏移 (因为 scale 长度为 N)
    scale_ptrs = scales_ptr + offs_n
    
    # 初始化累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # 2. 沿着 K 维度进行循环归约
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        
        # 计算 X 和 W 的数据指针
        # X: (BLOCK_M, BLOCK_K)
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        # W: (BLOCK_K, BLOCK_N)
        w_ptrs = w_int8_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
        
        # 加载数据
        x = tl.load(x_ptrs)
        w_int8 = tl.load(w_ptrs)
        
        # TODO 1: 在 SRAM 中进行动态反量化
        w_fp16 = w_int8.to(x.dtype)
        scales = tl.load(scale_ptrs)
        w_fp16 = w_fp16 * scales[None, :]
        
        # TODO 2: 执行点积并累加
        acc += tl.dot(x, w_fp16)
        
    # 3. 写回显存 (转换为 FP16)
    y_ptrs = y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    tl.store(y_ptrs, acc.to(tl.float16))

def triton_w8a16_gemm(x: torch.Tensor, w_int8: torch.Tensor, scales: torch.Tensor):
    M, K = x.shape
    _, N = w_int8.shape
    
    y = torch.empty((M, N), device=x.device, dtype=torch.float16)
    
    BLOCK_M = 16
    BLOCK_N = 64
    BLOCK_K = 64
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    w8a16_gemm_kernel[grid](
        x, w_int8, scales, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w_int8.stride(0), w_int8.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return y
```

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
