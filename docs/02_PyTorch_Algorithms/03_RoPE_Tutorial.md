# 03 RoPE Tutorial

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/03_RoPE_Tutorial.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 03. 旋转位置编码 (RoPE)

**难度：** Medium | **标签：** `基础架构`, `PyTorch` | **目标人群：** 模型微调与工程部署

本节我们将解析大模型当前最主流的位置编码方式：**RoPE (Rotary Position Embedding)**，并亲手用复数形式（Complex Tensor）实现它。这是 LLaMA, Qwen, DeepSeek 的标配！


> **相关阅读**:
> 本节使用纯 PyTorch 实现了算法逻辑与数学推导。
> 如果你想学习工业界如何打破该算子的 Memory Bound (访存瓶颈)，请前往 Triton 篇：
>  [`../03_Triton_Kernels/07_Triton_Fused_RoPE.ipynb`](../03_Triton_Kernels/07_Triton_Fused_RoPE.ipynb)


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
提示：可以使用 `torch.view_as_complex` 和 `torch.view_as_real` 这两个神器！


```python
import torch

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    计算复数指数频率张量 (cis = cos + i * sin)
    """
    # 1. 计算逆频率 1.0 / (theta ** (2i/dim))
    # freqs = ???
    
    # 2. 生成从 0 到 end-1 的位置张量 t
    # t = ???
    
    # 3. 外积生成 [seq_len, dim/2] 的角度矩阵
    # freqs = torch.outer(t, freqs)
    
    # ==========================================
    # TODO 1: 用极坐标生成复数张量 (提示: torch.polar)
    # ==========================================
    # freqs_cis = ???
    # return freqs_cis
    pass

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
    # TODO 2: 将 xq, xk 从 [..., dim] 转为复数张量 [..., dim//2]
    # 提示: 先 reshape 成 [..., -1, 2]，再用 torch.view_as_complex
    # ==========================================
    # xq_ = ???
    # xk_ = ???
    
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    # ==========================================
    # TODO 3: 进行复数乘法，并转回实数张量
    # 提示: 相乘后用 torch.view_as_real，最后 flatten(3) 恢复维度
    # ==========================================
    # xq_out = ???
    # xk_out = ???
    
    # return xq_out.type_as(xq), xk_out.type_as(xk)
    pass

```

```python
# 运行此单元格以测试你的实现
def test_rope():
    try:
        batch_size, seq_len, num_heads, head_dim = 2, 16, 4, 64
        xq = torch.randn(batch_size, seq_len, num_heads, head_dim)
        xk = torch.randn(batch_size, seq_len, num_heads, head_dim)
        
        freqs_cis = precompute_freqs_cis(head_dim, seq_len)
        xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)
        
        assert xq_out.shape == xq.shape, f"Shape mismatch! Expected {xq.shape}, got {xq_out.shape}"
        assert xk_out.shape == xk.shape
        print("\n✅ All Tests Passed! 恭喜你，RoPE 算子复数版实现成功！")
        
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
    except TypeError as e:
        print("代码可能还未填完，导致返回 None")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")

test_rope()

```

<details>
<summary>点击查看代码运行输出</summary>

```text

❌ 测试失败: name 'xq_' is not defined

```
</details>

```python

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

::: details 💡 点击查看官方解析与参考代码

RoPE (旋转位置编码) 巧妙地将位置信息注入到内积操作中。实现的核心在于利用 PyTorch 的复数数据类型。先将倒数维切成两半作为实部和虚部 (通过 reshape(-1, 2) 和 view_as_complex)，然后通过 torch.polar 构造指数复数 ^{i\theta}$。完成复数乘法旋转后，再用 view_as_real 和 flatten(3) 转回实数张量。

```python
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
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
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
