# 10. Triton Quantization | Triton 量化算子：W8A16 权重量化融合矩阵乘法 (Quantization GEMM)

**难度：** Hard | **标签：** `Triton`, `Quantization`, `GPTQ` | **目标人群：** 核心 Infra 与算子开发

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/03_Triton_Kernels/10_Triton_Quantization.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


在模型部署中，**Weight-Only 量化** (例如 W8A16 或 W4A16) 是最普遍的显存优化手段。由于激活值 (Activation) 依然保持 FP16，所以传统的 PyTorch 需要在计算前显式地把权重反量化回 FP16，这不仅慢，还抵消了显存带来的带宽优势。

以 GPTQ/AWQ 为代表的现代量化框架，底层的关键技术之一是 **On-the-fly Dequantization (即时反量化)**。
本节我们将编写一个 Triton 算子：在 SRAM 中读入 INT8 的权重和 FP16 的缩放因子 (Scales)，在寄存器里动态反量化为 FP16 后，立即与激活值相乘。

## 本节定位：推理优化的第二条路径

在上一节（09. PagedAttention）中，我们解决了 **KV Cache 碎片化** 问题，让显存能够更高效地复用。

本节解决另一个核心问题：**权重体积**。

### 为什么需要量化？

| 模型规模 | FP16 权重大小 | A100 80GB 单卡 | 需要张量并行 |
|---------|-------------|---------------|------------|
| 7B | 14GB | 可单卡跑 | 不需要 |
| 13B | 26GB | 可单卡跑 | 不需要 |
| 70B | 140GB | 需要 2 卡 | 需要 |
| 70B（INT8） | 70GB | 可单卡跑 | **不需要！** |

**量化带来的核心收益：**
1. **显存减半**：INT8 权重体积是 FP16 的一半
2. **带宽减半**：Memory Bound 场景下推理吞吐提升
3. **部署门槛降低**：大模型更容易在单卡上跑起来

### 与 Attention 主线的关系

量化可以应用在任何 Linear 层上，包括：
- Q/K/V 投影层（Attention 的一部分）
- FFN 的 Gate/Up/Down 投影层（MLP 的一部分）

所以量化不是 Attention 的“后继”，而是 **与 Attention 优化并列的另一条推理优化路径**。

如果说 07-09 是在“算得更快”，那 10-11 就是在“省得更多”。

## 与 20 节的关系

**W8A16 Quantization：**
- 20 节：先在 PyTorch 层讲清量化公式、误差和数据流
- 10 节：把同一套 W8A16 思路落到 Triton 融合 GEMM 上
- 核心区别：10 节关注的是即时反量化 + 矩阵乘法融合，而不是重新发明量化定义

## 前置

**导语：** 这一节把量化权重的反量化和 GEMM 融合在一起，目标是减少额外的 HBM 往返。

- [Part 1: 1B 单卡硬件与访存优化](../01_Hardware_Math_and_Systems/1B.md)
- [Part 1: 1D 异构调度与算子编程](../01_Hardware_Math_and_Systems/1D.md)
- [Part 1: 19 算子融合导论](../01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.md)

## 相关阅读
**导语：** 如果你想先把量化公式和 PyTorch 版本过一遍，可以继续看这页；不影响继续读本节，但会更容易理解即时反量化。
- [Part 2: 20 Quantization W8A16](../02_PyTorch_Algorithms/20_Quantization_W8A16.md)

### Step 1: 融合反量化矩阵乘法的主要思想

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
### 补充说明：性能分析与 Autotune

本节先给出一组足够清晰的 baseline 参数，便于把‘即时反量化 + GEMM 融合’这个主线讲透。
如果要做更细粒度的性能分析，可以继续用前面章节的工程习惯，结合 Nsight / nvprof / Triton benchmark 对比：
- 反量化版本 vs 显式反量化版本
- INT8 权重读带宽 vs FP16 权重读带宽
- 总时延、HBM 访问量和算子占比

`autotune` 的作用不是改变量化公式，而是替不同的 `M/N/K` 组合挑出更合适的 tile 配置。

###  Step 4: 动手实战

**要求**：请补全下方 `w8a16_gemm_kernel`。我们需要将 INT8 的权重即时转为 FP16 并完成矩阵乘法。为了简化，这里使用按列量化 (Per-channel Quantization)。


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
        
def _question_placeholder_10():
    raise NotImplementedError("请完成 TODO 1-2")

_question_placeholder_10()

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
        
    
        print("\n--- 性能观察（基于当前环境）---")
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
        print(f"当前环境观测比 (PyTorch / Triton): {ms_pt / ms_tr:.2f}x")
        print(" 说明：W8A16 的收益更依赖大规模 Memory Bound 场景；在当前矩阵规模和硬件组合下，反量化开销可能抵消部分带宽优势。")
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
- **关键点**：先把 INT8 权重转回 FP16，再乘上每列对应的 scale，完成即时反量化。
- **技术细节**：
  - `w_int8.to(x.dtype)` 只负责类型转换，不会改变数值分布。
  - `scales[None, :]` 通过广播与权重矩阵逐列相乘。
  - 反量化在 SRAM / 寄存器内完成，不需要先写回完整 FP16 权重到 HBM。

**2. TODO 2: 执行点积并累加**
- **实现方式**：
  ```python
  acc += tl.dot(x, w_fp16)
  ```
- **关键点**：把反量化后的权重直接送入 `tl.dot`，在同一轮 kernel 内完成 GEMM 累加。
- **技术细节**：
  - `acc` 使用更高精度的累加器保存中间结果，避免 FP16 的数值误差。
  - `tl.dot` 是这个算子的主计算路径，尽量让反量化和乘加都留在片上完成。
  - 当前实现的收益重点不是减少计算量，而是降低 HBM 访存压力。

**工程优化要点**
- **显存带宽优化**：读取 INT8 权重只需 FP16 的一半带宽，在 Memory Bound 场景下更容易体现收益。
- **即时反量化**：避免先生成完整 FP16 权重矩阵，再把它搬到显存里做第二次计算。
- **FP32 累加器**：保留中间累加精度，减少量化和矩阵乘法叠加带来的误差。
- **Per-channel 量化**：每列独立 scale，通常比 per-tensor 更稳，也更接近工程实践。
- **适用场景**：LLM 的 Linear / Projection 层，尤其是显存受限或带宽受限的推理环境。

**扩展说明**
- **Group-wise 量化**：当 `N` 很大时，可让多个列共享同一个 scale，减少 scale 的存储与加载开销。
- **对称 / 非对称量化**：当前实现是对称量化；如果引入 `zero_point`，则可以扩展为非对称量化。

### 下一步：从显存压缩到多租户服务

量化解决了**模型体积**的问题，让 70B 模型也能跑在单卡上。

但在真实的推理服务中，还有另一个瓶颈：多个用户同时使用不同 LoRA 权重时，如果逐个串行加载，GPU 的利用率会非常低。

下一节，我们将学习 Multi-LoRA：通过 Token 级动态路由，让一个 Batch 同时服务多个 LoRA 请求，榨干 GPU 的推理吞吐量。
