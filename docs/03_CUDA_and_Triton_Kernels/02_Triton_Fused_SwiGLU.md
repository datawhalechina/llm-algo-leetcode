# 02. Triton Fused SwiGLU | Triton 算子开发：融合门控激活函数 (Fused SwiGLU)

**难度：** Medium | **标签：** `Triton`, `Memory Bound`, `Operator Fusion` | **目标人群：** 核心 Infra 与算子开发

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/03_CUDA_and_Triton_Kernels/02_Triton_Fused_SwiGLU.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


SwiGLU 激活函数由两个线性层输出的非线性组合构成：$SwiGLU(x, W, V) = Swish(xW) \otimes xV$。
由于大模型的非线性层通常是纯粹的 Element-wise 操作（没有归约 Reduce），这使得它们极度 **Memory Bound (访存受限)**。在原生 PyTorch 中，多次访存会严重拖慢整体运算速度。
通过 Triton 编写 **算子融合 (Operator Fusion)** 内核，我们将多次读取（Read）和写入（Write）合并成一次，极大提升执行效率。

> **相关阅读**:
> 本节使用 Triton 实现了底层的极致显存与计算优化。
> 如果你对该算子的数学公式推导和纯 PyTorch 高层结构还不熟悉，建议先复习 PyTorch 篇：
>  [`../02_PyTorch_Algorithms/02_SwiGLU_Activation.ipynb`](../02_PyTorch_Algorithms/02_SwiGLU_Activation.md)

### Step 1: 算子融合的本质

> **PyTorch 连续计算的代价：**
> 假设输入是 $X, Y$，需要计算 $SwiGLU = X \cdot \sigma(X) \cdot Y$。
> 1. 计算 $\sigma(X)$：读取 $X$，计算 Sigmoid，写入显存。
> 2. 计算 $X \cdot \sigma(X)$：读取 $X$，读取 $\sigma(X)$，相乘，写入显存。
> 3. 计算最终结果：读取前面的结果，读取 $Y$，相乘，写入显存。
> **结果：极高的显存带宽消耗 (HBM Read/Write)。**

> **Triton 融合内核的机制：**
> 1. 将数据分块 (Block-wise) 读入超高速的 SRAM（片上缓存）。
> 2. 在 SRAM 内完成所有乘法和激活函数计算（在 SRAM 内不需要 HBM 读写代价）。
> 3. 直接将最终结果写回显存。
> **结果：只需要一次 HBM Read 和一次 HBM Write，速度翻倍！**

### Step 2: 算子融合如何打破 Memory Bound
如果使用纯 PyTorch：执行 `F.silu(x) * y` 需要读两次 HBM，写两次 HBM。但在 GPU 架构中，HBM 带宽是非常昂贵的。通过 Triton 算子融合，我们将 `x` 和 `y` 各自从 HBM 读入 SRAM 一次，在寄存器极速完成激活和乘法，直接把结果写回 HBM，从而省去了所有中间结果的存取开销。

### Step 3: 算子融合代码框架
我们设计一个接收三个指针（输入 X 的指针、输入 Y 的指针、输出指针）的内核。在每个 Program 内，并行地读取 `BLOCK` 长度的 `x` 块和 `y` 块，在 Triton 内执行 `x * tl.sigmoid(x) * y`，然后覆盖写入输出地址。

###  Step 4: 动手实战

**要求**：请补全下方 `fused_swiglu_kernel` 的 Triton 内核实现。假设输入 $X, Y$ 的形状是一维展开的长度 `N`。


```python
import torch
import triton
import triton.language as tl
```


```python
@triton.jit
def fused_swiglu_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    使用 Triton 融合 SwiGLU 激活函数的计算。
    公式: SwiGLU(x, y) = x * sigmoid(beta * x) * y (这里简化 beta=1)
    """
    # 1. 确定当前 program 处理的块 (Block) ID 和起始偏移
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # 2. 生成当前块处理的连续索引
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 3. 创建掩码，防止越界访问 (通常 n_elements 可能不是 BLOCK_SIZE 的整数倍)
    mask = offsets < n_elements
    
    # ==========================================
    # TODO 1: 从 x_ptr 和 y_ptr 中加载对应的数据到 SRAM
    # ==========================================
    # x = ???
    # y = ???
    pass
    
    # ==========================================
    # TODO 2: 在 SRAM 中进行核心算术运算
    # ==========================================
    # sig_x = ???
    # swish_x = ???
    # out = ???
    pass
    
    # ==========================================
    # TODO 3: 将最终结果存回 out_ptr
    # ==========================================
    # ???
    pass

def triton_fused_swiglu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """封装 Triton 内核的 PyTorch 调用接口"""
    assert x.is_cuda and y.is_cuda, "输入张量必须在 GPU 上"
    assert x.is_contiguous() and y.is_contiguous(), "输入张量必须在内存中连续"
    assert x.shape == y.shape, "X 和 Y 的形状必须一致"
    
    # 展开为一维
    n_elements = x.numel()
    
    # 分配输出内存 (必须预先分配并保证在 GPU 上且连续)
    out = torch.empty_like(x)
    
    # 设置网格维度 (Grid) 和块大小 (Block Size)
    # 对于简单的逐元素操作，通常 BLOCK_SIZE 设为 1024 或更大的 2 的幂次方
    BLOCK_SIZE = 1024
    
    # 计算需要启动的线程块数量，向上取整：ceil(N / BLOCK_SIZE)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # 启动内核
    fused_swiglu_kernel[grid](
        x, y, out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

```


```python
# 测试你的实现 (请在拥有 NVIDIA GPU 的环境下运行)
import torch.nn.functional as F

def test_fused_swiglu():
    if not torch.cuda.is_available():
        print("⏭️ 忽略测试：此环境没有 NVIDIA GPU，无法运行 Triton 内核。")
        return
        
    try:
        torch.manual_seed(42)
        # 测试数据大小不规整的情况，验证边界掩码 (Mask) 是否正确
        n = 1024 * 3 + 123 
        
        # 为了更贴近实际，使用半精度浮点数
        x = torch.randn(n, device='cuda', dtype=torch.float16)
        y = torch.randn(n, device='cuda', dtype=torch.float16)
        
        # 1. PyTorch 官方基准计算
        # 注意: SiLU 就是 beta=1 时的 Swish 函数
        out_ref = F.silu(x) * y
        
        # 2. Triton 算子计算
        out_triton = triton_fused_swiglu(x, y)
        
        # 3. 验证结果
        diff = torch.max(torch.abs(out_ref - out_triton))
        print(f"最大误差: {diff.item():.6e}")
        assert diff < 5e-3, "Triton 算子计算结果不正确！"
        
        print("✅ Triton 融合算子测试通过！")
        print(" 算子融合 (Operator Fusion) 可大幅降低 GPU 访存开销。")
        
        # 边界测试
        print("\n---  边界情况测试 ---")
        
        # 测试1: 单元素
        x1 = torch.tensor([1.0], device='cuda', dtype=torch.float16)
        y1 = torch.tensor([2.0], device='cuda', dtype=torch.float16)
        out1 = triton_fused_swiglu(x1, y1)
        ref1 = F.silu(x1) * y1
        assert torch.allclose(out1, ref1, rtol=5e-3), "单元素测试失败"
        print("✅ 单元素向量测试通过")
        
        # 测试2: 小向量（小于BLOCK_SIZE）
        x2 = torch.randn(100, device='cuda', dtype=torch.float16)
        y2 = torch.randn(100, device='cuda', dtype=torch.float16)
        out2 = triton_fused_swiglu(x2, y2)
        ref2 = F.silu(x2) * y2
        assert torch.allclose(out2, ref2, rtol=5e-3), "小向量测试失败"
        print("✅ 小向量测试通过")
        
        # 测试3: 恰好对齐BLOCK_SIZE
        x3 = torch.randn(1024, device='cuda', dtype=torch.float16)
        y3 = torch.randn(1024, device='cuda', dtype=torch.float16)
        out3 = triton_fused_swiglu(x3, y3)
        ref3 = F.silu(x3) * y3
        assert torch.allclose(out3, ref3, rtol=5e-3), "对齐测试失败"
        print("✅ BLOCK_SIZE对齐测试通过")
    
        print("\n--- 性能基准测试 (Benchmark) ---")
        quantiles = [0.5, 0.2, 0.8]
        ms_pt, min_ms_pt, max_ms_pt = triton.testing.do_bench(lambda: F.silu(x) * y, quantiles=quantiles)
        ms_tr, min_ms_tr, max_ms_tr = triton.testing.do_bench(lambda: triton_fused_swiglu(x, y), quantiles=quantiles)
        print(f"PyTorch Time (Multiple Reads/Writes): {ms_pt:.4f} ms")
        print(f"Triton Time (Fused 1 Read/Write):     {ms_tr:.4f} ms")
        print(f"Speedup:                              {ms_pt / ms_tr:.2f}x")
    except NotImplementedError:
        print("请先完成 TODO 代码！")
    except NameError as e:
        print(f"命名错误：可能有些变量没有定义。{e}")
    except triton.compiler.errors.CompilationError as e:
        print(f"❌ Triton 编译失败，请检查语法:\n{e}")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

test_fused_swiglu()
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
def fused_swiglu_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # TODO 1: 从 x_ptr 和 y_ptr 加载数据到 SRAM
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # TODO 2: 在 SRAM 中进行核心算术运算
    # 注意: tl.sigmoid 只支持 fp32/fp64，需要先转换
    x_fp32 = x.to(tl.float32)
    sig_x = tl.sigmoid(x_fp32)
    swish_x = x * sig_x.to(x.dtype)
    out = swish_x * y
    
    # TODO 3: 存储结果到 HBM
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_fused_swiglu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """封装 Triton 内核的 PyTorch 调用接口"""
    assert x.is_cuda and y.is_cuda, "输入张量必须在 GPU 上"
    assert x.is_contiguous() and y.is_contiguous(), "输入张量必须在内存中连续"
    assert x.shape == y.shape, "X 和 Y 的形状必须一致"
    
    # 展开为一维
    n_elements = x.numel()
    
    # 分配输出内存
    out = torch.empty_like(x)
    
    # 设置块大小
    BLOCK_SIZE = 1024
    
    # 计算需要启动的线程块数量
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # 启动内核
    fused_swiglu_kernel[grid](
        x, y, out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

```

### 解析

**1. TODO 1: 加载数据到SRAM**
- **实现方式**：`x = tl.load(x_ptr + offsets, mask=mask)`，`y = tl.load(y_ptr + offsets, mask=mask)`
- **关键点**：从HBM加载数据到SRAM，使用mask保护越界访问。这是算子融合的第一步
- **技术细节**：两个输入向量的对应块同时加载到片上内存，为后续融合计算做准备。SRAM的带宽是HBM的10-100倍

**2. TODO 2: 在SRAM中进行核心算术运算**
- **实现方式**：`x_fp32 = x.to(tl.float32)`，`sig_x = tl.sigmoid(x_fp32)`，`swish_x = x * sig_x.to(x.dtype)`，`out = swish_x * y`
- **关键点**：在SRAM中完成所有计算，避免中间结果写回HBM。这是算子融合的核心收益
- **技术细节**：
  - Triton的`tl.sigmoid`函数只支持fp32/fp64，不支持fp16。需要先将输入转换为fp32，计算后再转回原始精度
  - Swish(x) = x * sigmoid(x)，也称为SiLU激活函数
  - SwiGLU = Swish(x) * y，门控机制允许网络动态控制信息流
  - 类型转换在SRAM中进行，开销很小，不会影响融合算子的性能优势

**3. TODO 3: 存储结果到HBM**
- **实现方式**：`tl.store(out_ptr + offsets, out, mask=mask)`
- **关键点**：将最终结果一次性写回HBM，完成算子融合
- **技术细节**：相比PyTorch的多次读写（读x、写sigmoid(x)、读x、读sigmoid(x)、写swish(x)、读swish(x)、读y、写结果），融合后只需2次读（x、y）和1次写（结果），减少67%的内存访问

**工程优化要点**
- **算子融合收益**：将原本需要5次HBM访问（2读+1写+1读+1写）优化为3次（2读+1写），减少40%的内存访问。对于更复杂的融合可节省更多
- **Memory Bound突破**：Element-wise操作的计算速度远快于内存带宽，算子融合可实现2-3倍加速。在A100上，HBM带宽约2TB/s，而计算吞吐可达312 TFLOPS
- **SRAM利用**：片上内存（SRAM）的带宽是HBM的10-100倍，延迟也低得多。在SRAM中完成计算是性能优化的关键
- **适用场景**：激活函数（SwiGLU、GELU、ReLU）、归一化（LayerNorm、RMSNorm）、element-wise操作等Memory Bound算子都适合融合
- **Kernel融合策略**：通常将连续的element-wise操作融合在一起。过度融合会增加寄存器压力，需要权衡
- **工业实践**：vLLM、TensorRT-LLM、FlashAttention等高性能推理引擎大量使用算子融合。LLaMA等模型的FFN层使用SwiGLU，融合实现可显著提升推理速度