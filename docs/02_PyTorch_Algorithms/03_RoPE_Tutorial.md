# 03. RoPE Tutorial | 旋转位置编码教程

**难度：** Medium | **环境：** CPU-first | **标签：** `基础架构`, `位置编码`, `PyTorch` | **目标人群：** 模型微调与工程部署

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/03_RoPE_Tutorial.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本节我们将解析大模型当前最主流的位置编码方式：**RoPE (Rotary Position Embedding)**，并亲手用复数形式（Complex Tensor）实现它。这是 LLaMA, Qwen, DeepSeek 的标配！
可以先把 RoPE 记成一件事：它不是给 token 加一个独立的位置向量，而是通过旋转 Query/Key 让位置信息进入注意力点积里。

**关键词：** `RoPE`, `positional encoding`, `complex tensor`
## 前置阅读

**导语：** 如果还没把张量变换和注意力直觉理顺，先看下面几页再进入 RoPE 会更顺。

- [P0: 05. PyTorch Tensor Fundamentals | PyTorch 张量基础操作](../00_Prerequisites/05_PyTorch_Tensor_Fundamentals.md)
- [P0: 16. Attention Mechanism Intro | 注意力机制导论](../00_Prerequisites/16_Attention_Mechanism_Intro.md)

## 相关阅读

**导语：** 本节先把 RoPE 的旋转位置编码数学推导讲清楚；如果想看它和 Attention 融合后在实现层怎么落地，再看硬件直觉和算子融合页面。

- [P1: 03. GPU Architecture and Memory | GPU 物理架构与内存层级](../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.md)
- [P1: 19. Operator Fusion Introduction | 算子融合导论](../01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.md)

### Step 1: 核心思想与痛点

本节说明 RoPE 的设计动机与核心思想。

> **为什么需要 RoPE？**
> 原生的 Transformer 使用绝对位置编码（如正弦波或可学习参数），导致模型很难泛化到比训练集更长的序列。我们希望模型能在计算 Attention 时感知到 Token 之间的**相对距离**。
> **RoPE 的本质：**
> “借用复数的旋转”。通过将 Query 和 Key 的向量映射到复数空间并旋转特定角度，在计算内积（Dot-product）时，结果自然就带有了相对位置信息 $(m-n)$。其中，$m$ 是 Query 的位置，$n$ 是 Key 的位置，两者之差 $m−n$ 就是它们之间的相对距离——RoPE 通过复数旋转让 Attention 内积的结果只依赖于这个差值，从而让 Attention 内积的结果依赖于 Token 间的相对位置。

### Step 2: 代码实现框架

在 PyTorch 中，最高效的 RoPE 实现方式之一是利用复数乘法。我们将最后一维切分为两半并组合成复数形式，再乘以预先计算好的复数旋转矩阵 $e^{im\theta_{j}}$。完成旋转后，再使用 `torch.view_as_real` 恢复为实数表示。


因此实现时的主线其实很固定：先算出 freqs_cis （即预计算的复数旋转因子），再把它和 xq / xk（即 Attention 中的 Query 和 Key 投影张量）做广播对齐，最后完成复数旋转并回到原始实数形状。这样学习者在写 TODO 1/2/3 时，就能清楚地知道每段代码对应实现流程中的哪个环节。

###  Step 3: 核心公式与张量维度

这一节把频率、位置和维度对齐关系摆清楚，方便把数学公式和代码里的广播一步一步对上。

1. **预计算旋转角 (Precompute Frequencies)**
   
   频率计算公式：
   $\theta_{j} = 10000^{-2j/d}$。其中，$j=0,1,2,…,d/2​−1$ ，是维度索引，用于遍历每一组维度对（最后一维两两一组）。$d$ 是 Head Dimension，即每个注意力头的维度。
   
   **关于基数 10000 的选择说明**：基数 10000 直接继承自 Transformer 原始论文（Vaswani et al., 2017）中的正弦位置编码设计。其数学意义在于：当 $j=0$ 时，频率$\theta_{0}=1$ 对应最快的旋转速度（$m$ 每增加 1，角度旋转 1 弧度），用于捕获相邻 Token 间的细微相对位置。当 $j=d/2−1$ 时，频率 $\theta_{d/2-1}≈1/10000$，对应最慢的旋转速度，负责感知长距离的绝对位置。
   
   这一设计使得不同维度拥有指数级分布的频率，让模型像多波段接收器一样，同时兼顾局部和全局的位置感知。10000是原论文作者通过实验验证的有效平衡点——过大（如 $10^{6}$）会导致高维度信号变化过缓，浪费编码能力；过小（如 100）则会限制模型的长程依赖能力。

   生成复数形式的极坐标：
   $e^{i m \theta_{j}} = \cos(m \theta_{j}) + i \sin(m \theta_{j})$。其中，$m$指的是Token 在序列中的绝对位置索引，$m=0,1,2,…,L−1$（$L$为序列长度）
   
2. **应用旋转 (Apply Rotary Embedding)**
   
   将输入的 Query 或 Key 视为复数，具体做法是将最后一维切分为等长的两半，前一半作为实部，后一半作为虚部：
   $x = x_{real} + i \cdot x_{imag}$

   利用复数乘法直接完成旋转矩阵的运算：
   $x_{rotated} = x \times e^{i m \theta_{j}}$

3. **实现提示** 
   
   `reshape_for_broadcast` 的作用是将 freqs_cis（形状为 [seq_len, d/2]）变形为 [1, seq_len, 1, d/2]，使其与 xq/xk（形状为[batch, seq_len, heads, d]）在 batch 和 head 维度上广播对齐（其中，batch	批次大小，seq_len指序列长度）。先对齐维度，再做复数乘法，旋转位置编码才会同时正确作用到 batch 和 head 维度。
###  Step 4: 动手实战

这里开始把频率预计算、复数转换和旋转还原落到最小可运行代码里，重点看每一步张量形状怎么变。

**要求**：请补全下方 `precompute_freqs_cis` 和 `apply_rotary_emb` 函数。
提示：可以使用 `torch.view_as_complex` 和 `torch.view_as_real` 这两个核心函数！


```python
import torch
```


```python
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    预计算复数旋转因子 freqs_cis。
    
    Args:
        dim: head_dim，必须为偶数
        end: 序列长度
        theta: 基数，默认 10000
    
    Returns:
        freqs_cis: 形状为 [end, dim//2] 的复数张量
    """
    # ==========================================
    # TODO 1: 用极坐标生成复数张量 (提示: torch.polar)
    # 1. 计算逆频率向量 inv_freq = 1/(theta ** (2j/d))
    #   torch.arange(0, dim, 2) 步长为 2，对应公式中的 2j
    # 2. 生成位置索引 t = [0, 1, ..., end-1]
    # 3. 计算角度矩阵 angles = outer(t, inv_freq)
    # 4. 用 torch.polar 生成复数 e^{i * angles}
    # ==========================================
    #inv_freq = ??
    #t = ??
    #angles = ??
    #freqs_cis = ??
    return freqs_cis

# 将频率张量扩展到可广播形状，供 Step 3 的复数乘法使用
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    将 freqs_cis 变形为与 x 广播对齐的形状。
    
    假设 x 的形状为 [batch, seq_len, heads, head_dim//2]（复数形式），
    将 freqs_cis 从 [seq_len, head_dim//2] 变形为 [1, seq_len, 1, head_dim//2]。
    """
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

    Args:
        xq: [batch, seq_len, heads, head_dim]
        xk: [batch, seq_len, heads, head_dim]
        freqs_cis: [seq_len, head_dim//2]，预计算的旋转因子
    Returns:
        旋转后的 xq, xk，形状与输入一致
    """
    # ==========================================
    # TODO 2: 将 xq, xk 从实数张量转为复数张量
    # 提示: 先把最后一维拆成两个一组，再转成复数
    #   1. 提升精度到 FP32: .float()
    #   2. 将最后一维 head_dim 拆分为 (-1, 2)，其中 2 对应实部和虚部
    #   3. 用 torch.view_as_complex 转为复数
    # 提示：reshape(*xq.shape[:-1], -1, 2) 保留前面所有维度，最后变为 (..., -1, 2)
    # ==========================================
    # xq_complex = ???
    # xk_complex = ???
    
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_complex)
    # 确保类型一致
    freqs_cis = freqs_cis.to(xq_complex.dtype)
    

    # ==========================================
    # TODO 3: 进行复数乘法，并转回实数张量
    # 步骤：
    #   1. 复数乘法: xq_complex * freqs_cis（自动广播）
    #   2. 用 torch.view_as_real 转回实数，形状变为 (..., 2)
    #   3. 用 .flatten(-2) 将最后两维合并回 head_dim
    #   4. 用 .type_as(xq) 恢复为输入的数据类型
    # ==========================================
    # xq_out = ??
    # xk_out = ??
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
        
        # 核心修复：防止占位符作弊，输出绝不能等于输入
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
        raise
    except (AttributeError, NameError, TypeError) as e:
        print(f"\n❌ 测试失败: 代码可能未完成")
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        raise
    except Exception as e:
        print(f"\n❌ 发生未知异常: {type(e).__name__}: {e}")
        raise

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
    """
    预计算复数旋转因子 freqs_cis。
    
    Args:
        dim: head_dim，必须为偶数
        end: 序列长度
        theta: 基数，默认 10000
    
    Returns:
        freqs_cis: 形状为 [end, dim//2] 的复数张量
    """
    
    # 先按维度间隔计算逆频率，再把位置和频率组合成复数旋转角。
    # TODO 1: 计算逆频率并生成复数张量
    assert dim % 2 == 0, f"Head dimension must be even for RoPE, got {dim}"
    # 生成逆频率向量：对应公式中的 theta_j = 10000^{-2j/d}
    # torch.arange(0, dim, 2) 步长为 2，对应公式中的 j 索引
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    t = torch.arange(end, device=inv_freq.device, dtype=torch.float32)
    
    # 位置与频率的外积：angles[m, j] = m * theta_j
    angles  = torch.outer(t, inv_freq)
    
    # 生成复数旋转因子 e^{i * angles}
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    将 freqs_cis 变形为与 x 广播对齐的形状。
    
    假设 x 的形状为 [batch, seq_len, heads, head_dim//2]（复数形式），
    将 freqs_cis 从 [seq_len, head_dim//2] 变形为 [1, seq_len, 1, head_dim//2]。
    """
    ndim = x.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对 Query 和 Key 应用旋转位置编码。

    Args:
        xq: [batch, seq_len, heads, head_dim]
        xk: [batch, seq_len, heads, head_dim]
        freqs_cis: [seq_len, head_dim//2]，预计算的旋转因子
    
    Returns:
        旋转后的 xq, xk，形状与输入一致
    """
    # 先把最后一维两两配对，再提升到 FP32 后解释成复数。
    # TODO 2: 转换为复数张量（注意精度提升）
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_complex)
    # 确保类型一致
    freqs_cis = freqs_cis.to(xq_complex.dtype)
    
    # TODO 3: 复数乘法并转回实数
    # 复数乘法 (a+bi)*(c+di) 自动实现旋转矩阵的效果
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(-2)
    
    # 恢复为输入的 dtype（如 BF16）
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

### 答案与直觉

- **本节要解决什么：** 把相对位置信息写进 Query/Key，让注意力直接感知 token 间距离。
- **为什么这样做：** 复数乘法天然对应二维旋转，和 RoPE 的几何直觉完全一致。
- **带走的直觉：** 先算频率、再做广播、最后旋转还原，是 RoPE 的固定实现路径。

**1. TODO 1 (预计算旋转频率与极坐标复数生成)**

- **逆频率计算：** 使用公式 $\text{inv\_freq}_j = \theta ^{-2j/d}$ （$j=0,1,\ldots,d/2-1$ 为维度索引）计算每个维度的旋转频率。代码中用 `torch.arange(0, dim, 2)` 以步长 2 取偶数维索引，对应复数的实部和虚部配对，并作为 $\theta$ 的负指数，即 $\theta^{-2j/d}$。
- **位置编码矩阵：** 通过 `torch.outer(t, inv_freq)` 生成位置 `t` 与频率 `inv_freq` 的角度矩阵，其中 `t` 是位置索引。
- **极坐标复数：** `torch.polar(torch.ones_like(angles), angles)` 生成复数 $e^{i\theta}$，这里 `torch.ones_like(angles)` 全为 1（模长），`angles` 是预计算的角度矩阵。这是 RoPE 的核心数学表示。
- **工程细节：** 为什么代码用 `torch.arange(0, dim, 2)` 而公式是 $\theta^{-2j/d}$？因为实现里最后一维是按实部/虚部两两配对的，步长 2 正好在枚举每一对里的偶数位置，也就对应了公式里的 $2j$。

**2. TODO 2 (实数张量转复数张量与精度提升)**

- **精度提升的必要性：** 在执行 `torch.view_as_complex` 之前必须先调用 `.float()` 将张量提升到 FP32。这是因为复数乘法在 FP16/BF16 下极易发散或产生 NaN，导致训练崩溃。这是 RoPE 实现中最容易踩的坑，LLaMA 等开源模型的源码中都强制使用 FP32 进行旋转计算。
- **维度重塑：** 将最后一维 `head_dim` 拆分为 `(-1, 2)`，其中 `2` 对应实部和虚部。
- **复数转换：** 将形状 `(..., head_dim)` 的实数张量解释为复数张量 `(..., head_dim // 2)`，每两个相邻元素组成一个复数。

**3. TODO 3 (复数乘法旋转与实数还原)**

- **广播机制：** 将 `freqs_cis` 的形状从 `(seq_len, head_dim // 2)` 扩展为 `(1, seq_len, 1, head_dim // 2)`，使其与 `xq/xk` 的 `(batch, seq_len, heads, head_dim)` 在 `batch` 和 `heads` 维度上广播对齐。
- **复数乘法：** 完成旋转操作，这是 RoPE 的核心计算。复数乘法 $(a+bi)(c+di) = (ac-bd) + (ad+bc)i$ 自动实现了旋转矩阵的效果。
- **实数还原：** 将复数张量转回实数表示，在最后增加一个大小为 2 的维度。
- **维度恢复：** 使用 `.flatten(-2)` 将最后两个维度 (..., 2) 合并回 `head_dim`，恢复原始形状。

**进阶思考：RoPE 的上下文外推 (Context Extension)**

- **问题背景：** 模型在 4K 序列长度上训练，如何在推理时支持 16K 甚至 128K？直接外推会导致性能急剧下降。
- **解决方案：** 工业界提出了多种 RoPE Scaling 技术：
  - **线性插值：** 将位置索引 $m$ 除以缩放因子，相当于压缩位置空间。
  - **NTK-aware Scaling：** 动态调整基频 （如从 10000 增大到 100000），降低高频分量的旋转速度。
  - **YaRN：** 结合低频外推和高频插值，在不同维度使用不同的缩放策略。
- **工程实践：** LLaMA 2 使用线性插值支持 32K 上下文，Qwen 使用动态 NTK 支持 128K，这些技术使得 RoPE 成为当前大模型位置编码的事实标准。