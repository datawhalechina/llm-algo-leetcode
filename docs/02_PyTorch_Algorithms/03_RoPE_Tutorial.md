# 03. RoPE Tutorial | 旋转位置编码 (RoPE)

**难度：** Medium | **标签：** `基础架构`, `PyTorch` | **目标人群：** 模型微调与工程部署

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/03_RoPE_Tutorial.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本节我们将解析大模型当前最主流的位置编码方式：**RoPE (Rotary Position Embedding)**，并亲手用复数形式（Complex Tensor）实现它。这是 LLaMA, Qwen, DeepSeek 的标配！


> **相关阅读**:
> 本节使用纯 PyTorch 实现了算法逻辑与数学推导。
> 如果你想学习工业界如何打破该算子的 Memory Bound (访存瓶颈)，请前往 Triton 篇：
>  [`../03_CUDA_and_Triton_Kernels/07_Triton_Fused_RoPE.ipynb`](../03_CUDA_and_Triton_Kernels/07_Triton_Fused_RoPE.md)

### Step 1: 核心思想与痛点

> **为什么需要 RoPE？**
> 原生的 Transformer 使用绝对位置编码（如正弦波或可学习参数），导致模型很难泛化到比训练集更长的序列。我们希望模型能在计算 Attention 时感知到 Token 之间的**相对距离**。
> **RoPE 的本质：**
> “借用复数的旋转”。通过将 Query 和 Key 的向量映射到复数空间并旋转特定角度，在计算内积（Dot-product）时，结果自然就带有了相对位置信息 $(m-n)$。

### Step 2: 代码实现框架
在 PyTorch 中，最高效的 RoPE 实现方式之一是利用复数乘法。我们将最后一维切分为两半并组合成复数形式，再乘以预先计算好的复数旋转矩阵 $e^{im\theta}$。完成旋转后，再使用 `torch.view_as_real` 恢复为实数表示。

###  Step 3: 核心公式与张量维度

1. **预计算旋转角 (Precompute Frequencies):**
   频率计算公式：$\Theta = 10000^{-2i/d}$，其中 $i$ 是维度索引，$d$ 是 Head Dimension。
   生成复数形式的极坐标：$e^{i m \Theta} = \cos(m \Theta) + i \sin(m \Theta)$
   
2. **应用旋转 (Apply Rotary Embedding):**
   将输入的 Query 或 Key 视为复数：`x = x_real + i * x_imag`
   利用复数乘法直接完成旋转矩阵的运算：$x_{rotated} = x \times e^{i m \Theta}$

###  Step 4: 动手实战

**要求**：请补全下方 `precompute_freqs_cis` 和 `apply_rotary_emb` 函数。
提示：可以使用 `torch.view_as_complex` 和 `torch.view_as_real` 这两个核心函数！


```python
import torch
```


```python
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    计算复数指数频率张量 (cis = cos + i * sin)
    """
    # ==========================================
    # TODO 1: 用极坐标生成复数张量 (提示: torch.polar)
    # ==========================================
    # freqs = ???
    # t = ???
    # freqs_cis = ???
                 
    freqs_cis = torch.ones((end, dim // 2), dtype=torch.complex64)  # 占位初始化（错误实现，供测试框架捕获）                                                                                                             
    return freqs_cis   
    

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将旋转位置编码应用到 Query 和 Key 上
    """
    # ==========================================
    # TODO 2: 将 xq, xk 从实数张量转为复数张量
    # 提示: 
    # ==========================================
    # xq_ = ???
    # xk_ = ???
                                                                                                                                                                     
    xq_ = torch.view_as_complex(torch.zeros(*xq.shape[:-1], xq.shape[-1] // 2, 2, dtype=xq.dtype, device=xq.device)) # 占位初始化     
    xk_ = torch.view_as_complex(torch.zeros(*xk.shape[:-1], xk.shape[-1] // 2, 2, dtype=xk.dtype, device=xk.device)) # 占位初始化        
    
    
    # ==========================================
    # TODO 3: 进行复数乘法，并转回实数张量
    # 提示: 
    # ==========================================
    # xq_out = ???
    # xk_out = ???

    xq_out = torch.zeros_like(xq)  # 占位初始化                                                                                                                                                               
    xk_out = torch.zeros_like(xk)  # 占位初始化                                                                                                                                                               
                 
    return xq_out.type_as(xq), xk_out.type_as(xk)      

```


```python
# 运行此单元格以测试你的实现
def test_rope():
    try:
        print("=" * 60)
        print("开始测试 RoPE 旋转位置编码")
        print("=" * 60)

        batch_size, seq_len, num_heads, head_dim = 2, 16, 4, 64

        # Test 1: 形状测试
        print("\n【Test 1】形状测试")
        xq = torch.randn(batch_size, seq_len, num_heads, head_dim)
        xk = torch.randn(batch_size, seq_len, num_heads, head_dim)

        freqs_cis = precompute_freqs_cis(head_dim, seq_len)
        xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)

        assert xq_out.shape == xq.shape, f"Query 输出形状错误: 期望 {xq.shape}, 实际 {xq_out.shape}"
        assert xk_out.shape == xk.shape, f"Key 输出形状错误: 期望 {xk.shape}, 实际 {xk_out.shape}"
        assert freqs_cis.shape == (seq_len, head_dim // 2), f"频率张量形状错误"
        
        # 🚀 核心修复：防止占位符作弊，输出绝不能等于输入
        assert not torch.allclose(xq, xq_out, atol=1e-5), "TODO 3 未完成: 输出与输入完全相同，RoPE 旋转未生效！"
        
        print("  ✅ 输出形状测试通过")
        print("  ✅ 频率张量形状测试通过")

        # Test 2: 数值范围测试
        print("\n【Test 2】数值范围测试")
        norm_before = torch.norm(xq, dim=-1)
        norm_after = torch.norm(xq_out, dim=-1)
        assert torch.allclose(norm_before, norm_after, rtol=1e-4, atol=1e-5), "RoPE 改变了向量模长！"
        print("  ✅ 向量模长保持不变（旋转不变性）")

        assert not torch.isnan(xq_out).any(), "输出包含 NaN！"
        assert not torch.isinf(xq_out).any(), "输出包含 Inf！"
        print("  ✅ 无 NaN/Inf 数值异常")

        # Test 3: 相对位置编码验证
        print("\n【Test 3】相对位置编码验证")
        pos0 = xq_out[:, 0, :, :]
        pos1 = xq_out[:, 1, :, :]
        assert not torch.allclose(pos0, pos1, rtol=1e-3), "不同位置的输出相同，位置编码失败！"
        print("  ✅ 位置编码生效（不同位置输出不同）")

        # Test 4: 精度稳定性测试
        print("\n【Test 4】精度稳定性测试")
        xq_fp16 = torch.randn(1, 8, 2, head_dim, dtype=torch.float16)
        xk_fp16 = torch.randn(1, 8, 2, head_dim, dtype=torch.float16)
        freqs_fp16 = precompute_freqs_cis(head_dim, 8)

        xq_out_fp16, xk_out_fp16 = apply_rotary_emb(xq_fp16, xk_fp16, freqs_fp16)

        assert xq_out_fp16.dtype == torch.float16, "输出类型错误！"
        assert not torch.isnan(xq_out_fp16).any(), "FP16 输入导致 NaN！"
        print("  ✅ FP16 输入处理正确")
        print("  ✅ 精度提升机制工作正常")

        print("\n" + "=" * 60)
        print(" RoPE 算子实现通过测试。")
        print("   所有测试用例均已通过")
        print("=" * 60)

    except NotImplementedError:
        print("\n❌ 测试失败: 请先完成 TODO 部分的代码！")
    except TypeError as e:
        print(f"\n❌ 测试失败: 代码可能未完成")
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        raise e  # 将错误抛给测试脚本
    except Exception as e:
        print(f"\n❌ 发生未知异常: {type(e).__name__}: {e}")
        raise e  # 将错误抛给测试脚本

test_rope()

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
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # TODO 1: 计算逆频率并生成复数张量
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO 2: 转换为复数张量（注意精度提升）
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    # TODO 3: 复数乘法并转回实数
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

### 解析

**1. TODO 1 (预计算旋转频率与极坐标复数生成)**

- **逆频率计算：** 使用公式 $\Theta = 10000^{-2i/d}$ 计算每个维度的旋转频率。代码中  步长为 2，对应复数的实部和虚部配对，除以  后取负指数。
- **位置编码矩阵：** 通过  生成  的角度矩阵，其中  是位置索引 。
- **极坐标复数：**  生成复数 ^{i\theta}$，这里  全为 1（模长）， 是预计算的角度矩阵。这是 RoPE 的核心数学表示。
- **工程细节：** 为什么代码用  而公式是 hmtBc2i/d$？因为 PyTorch 复数将最后一维按  成对存储，步长 2 正好对应公式中的  指数。

**2. TODO 2 (实数张量转复数张量与精度提升)**

- **精度提升的必要性（Critical）：** 在执行  之前必须先调用  将张量提升到 FP32。这是因为复数乘法在 FP16/BF16 下极易发散或产生 NaN，导致训练崩溃。这是 RoPE 实现中最容易踩的坑，LLaMA 等开源模型的源码中都强制使用 FP32 进行旋转计算。
- **维度重塑：**  将最后一维  拆分为 ，其中 2 对应实部和虚部。
- **复数转换：**  将形状  的实数张量解释为复数张量 ，每两个相邻元素组成一个复数。

**3. TODO 3 (复数乘法旋转与实数还原)**

- **广播机制：**  将  的形状从  扩展为 ，以便与  的形状  进行广播。
- **复数乘法：**  完成旋转操作，这是 RoPE 的核心计算。复数乘法 (c+di) = (ac-bd) + (ad+bc)i$ 自动实现了旋转矩阵的效果。
- **实数还原：**  将复数张量转回实数表示，在最后增加一个大小为 2 的维度 。
- **维度展平：**  将最后两个维度  合并回 ，恢复原始形状。
- **类型恢复：**  将结果转回输入的原始精度（如 FP16），因为前面为了数值稳定性提升到了 FP32。

**进阶思考：RoPE 的上下文外推 (Context Extension)**

- **问题背景：** 模型在 4K 序列长度上训练，如何在推理时支持 16K 甚至 128K？直接外推会导致性能急剧下降。
- **解决方案：** 工业界提出了多种 RoPE Scaling 技术：
  - **线性插值 (Linear Scaling)：** 将位置索引  除以缩放因子，相当于压缩位置空间。
  - **NTK-aware Scaling：** 动态调整基频 （如从 10000 增大到 100000），降低高频分量的旋转速度。
  - **YaRN：** 结合低频外推和高频插值，在不同维度使用不同的缩放策略。
- **工程实践：** LLaMA 2 使用线性插值支持 32K 上下文，Qwen 使用动态 NTK 支持 128K，这些技术使得 RoPE 成为当前大模型位置编码的事实标准。