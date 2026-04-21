# 11. Triton Quantization Support | Triton 量化算子：W8A16 权重量化融合矩阵乘法 (Quantization GEMM)

**难度：** Hard | **标签：** `Triton`, `Quantization`, `GPTQ` | **目标人群：** 核心 Infra 与算子开发

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/03_CUDA_and_Triton_Kernels/11_Triton_Quantization_Support.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


在模型部署中，**Weight-Only 量化** (例如 W8A16 或 W4A16) 是最普遍的显存优化手段。由于激活值 (Activation) 依然保持 FP16，所以传统的 PyTorch 需要在计算前显式地把权重反量化回 FP16，这不仅慢，还抵消了显存带来的带宽优势。

以 GPTQ/AWQ 为代表的现代量化框架，底层的核心技术是 **On-the-fly Dequantization (即时反量化)**。
本节我们将编写一个 Triton 算子：在 SRAM 中读入 INT8 的权重和 FP16 的缩放因子 (Scales)，在寄存器里动态反量化为 FP16 后，立即与激活值相乘。


> **相关阅读**:  
> 如果你对量化的数学公式推导和纯 PyTorch 实现还不熟悉，建议先复习 PyTorch 篇：
> [`../02_PyTorch_Algorithms/20_Quantization_W8A16.ipynb`](../02_PyTorch_Algorithms/20_Quantization_W8A16.md)
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
```


```python
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
        
        # ==========================================
        # TODO 1: 在 SRAM 中进行动态反量化
        # 提示: 将 w_int8 转换为浮点类型，加载 scales，使用广播机制相乘
        # ==========================================
        # w_fp16 = ???
        # scales = ???
        # w_fp16 = ???
        
        # ==========================================
        # TODO 2: 执行点积并累加
        # 提示: 使用 tl.dot 计算矩阵乘法并累加到 acc
        # ==========================================
        # acc += ???
        pass
        
    raise NotImplementedError("请完成 TODO 1-2")

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
        
        print("✅ W8A16 即时反量化 GEMM 验证通过。")
        
    
        print("\n--- 性能基准测试 (Benchmark) ---")
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
        print(" 在 Memory Bound 场景下，读取 INT8 权重（一半带宽）+ SRAM 内反量化的总开销可能低于读取完整 FP16 权重。")
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
## 参考代码与解析
### 代码

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

### 解析

**1. TODO 1: 在 SRAM 中进行动态反量化**
- **实现方式**：
  ```python
  w_fp16 = w_int8.to(x.dtype)
  scales = tl.load(scale_ptrs)
  w_fp16 = w_fp16 * scales[None, :]
  ```
- **关键点**：这是 W8A16 量化算子的核心，实现了即时反量化（On-the-fly Dequantization）
- **技术细节**：
  - `w_int8.to(x.dtype)`：将 INT8 权重转换为 FP16（与输入 x 的数据类型一致）
  - 类型转换发生在 SRAM/寄存器中，不产生额外的 HBM 访问
  - `tl.load(scale_ptrs)`：加载当前块对应的缩放因子，形状为 `(BLOCK_N,)`
  - `scales[None, :]`：将一维 scales 扩展为 `(1, BLOCK_N)`，用于广播
  - `w_fp16 * scales[None, :]`：对权重矩阵的每一列应用对应的缩放因子
  - 广播机制：`(BLOCK_K, BLOCK_N) * (1, BLOCK_N)` → `(BLOCK_K, BLOCK_N)`
  - 整个反量化过程在 SRAM 内完成，避免了传统方法中将完整 FP16 权重写回 HBM 的开销

**2. TODO 2: 执行点积并累加**
- **实现方式**：
  ```python
  acc += tl.dot(x, w_fp16)
  ```
- **关键点**：使用 Triton 的高性能矩阵乘法原语，直接对反量化后的权重进行计算
- **技术细节**：
  - `tl.dot(x, w_fp16)`：计算 `(BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N)` → `(BLOCK_M, BLOCK_N)`
  - `acc` 使用 FP32 累加器，避免精度损失
  - 反量化后的 `w_fp16` 直接参与矩阵乘法，无需写回 HBM
  - 循环归约：沿 K 维度分块计算，每次迭代累加一个子块的结果
  - 最终写回时转换为 FP16：`acc.to(tl.float16)`

**工程优化要点**

- **显存带宽优化**：读取 INT8 权重只需 FP16 的一半带宽，在 Memory Bound 场景下显著提升性能
- **即时反量化**：反量化操作在 SRAM/寄存器中完成，避免了传统方法中生成完整 FP16 权重矩阵的 HBM 开销
- **计算与访存重叠**：类型转换和缩放操作的计算开销被 HBM 访存延迟隐藏
- **Per-channel 量化**：每列使用独立的缩放因子，保持较高的量化精度
- **FP32 累加器**：使用 FP32 进行中间累加，避免 FP16 的精度损失和数值溢出
- **分块计算**：使用 2D Grid 并行处理输出矩阵的不同块，充分利用 GPU 并行性
- **工业应用**：该算子是 GPTQ、AWQ 等量化框架的核心组件，广泛应用于大模型推理加速
- **适用场景**：
  - Memory Bound 的 Linear 层（如 LLM 的 FFN 和 Attention 投影层）
  - 显存受限的部署环境（边缘设备、多租户推理服务）
  - 需要在推理速度和模型精度之间取得平衡的场景
- **性能收益**：
  - 显存占用减半（INT8 vs FP16）
  - 在 Memory Bound 场景下可获得 1.5-2x 的加速
  - 相比预先反量化的方法，避免了额外的显存分配和数据传输