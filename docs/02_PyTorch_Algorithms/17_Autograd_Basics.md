# 17. Autograd Basics | Attention 反向传播与自定义 Autograd

**难度：** Medium | **环境：** CPU-first | **标签：** `显存优化`, `Autograd`, `反向传播` | **目标人群：** 显存优化学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/17_Autograd_Basics.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

Attention 的前向计算把 $Q$、$K$、$V$ 组合成输出；反向传播则沿着相反的依赖关系，把输出梯度传回这三个输入。本节不再重复通用的损失—激活反向，而是用 Attention 这个中间状态昂贵的案例，说明反向阶段为什么会产生显存压力，并把它与后续的分块计算联系起来。

学习时可以先沿着 $dO \rightarrow dV、dP \rightarrow dS \rightarrow dQ、dK$ 的方向观察依赖关系，再把每个中间量的形状、保存状态和显存代价对应起来。

**关键词：** `Autograd`, `backward`, `gradcheck`

---
## 前置阅读

**导语：** 先理解 PyTorch 如何记录计算图，并复习 Attention 的 Q/K/V 前向路径，再进入手写 Attention backward。训练循环属于辅助背景。

- [P0: 07. PyTorch Autograd and Backward | PyTorch 自动求导与反向传播](../00_Prerequisites/07_PyTorch_Autograd_and_Backward.md)
- [04. Attention MHA/GQA | 多头注意力与 KV Cache](../02_PyTorch_Algorithms/04_Attention_MHA_GQA.md)
- [P0: 13. Simple Neural Network Training | 简单神经网络训练循环](../00_Prerequisites/13_Simple_Neural_Network_Training.md)

---

### Step 1: Attention backward 的整体梯度链

把 Attention 看成一条依赖链：前向阶段由 $Q、K、V$ 形成 $S$、$P$ 和 $O$；反向阶段从输出梯度 $dO$ 出发，先分成 $dV$ 与 $dP$，再经过 $dS$ 回到 $dQ$ 和 $dK$。其中 $dO$ 是这条反向链的起点。

先把每个量的形状和流向对齐，再把梯度依赖、保存状态和显存账本放到同一条链上观察。

> **张量形状说明：**
> - $Q, K, V \in \mathbb{R}^{B \times N \times d}$（batch $B$、序列长度 $N$、特征维数 $d$）
> - $S, P \in \mathbb{R}^{B \times N \times N}$
> - $O \in \mathbb{R}^{B \times N \times d}$

![Autograd：前向记录与反向梯度流](../public/02_PyTorch_Algorithms/17_autograd_attention_backward.svg)
### Step 2: 链式法则

先把图中的 $dO$ 理解成损失函数传给 Attention 输出 $O$ 的梯度。因为 $O = P V$，它会分成两条支路：一条直接求 $dV$；另一条先求 $dP$，再穿过 Softmax 得到 $dS$，最后回传到 $Q$ 和 $K$。

沿着图中的两条支路，依次推导下面四个结果。

**1. 求 $dV$**
因为 $O = P V$，根据矩阵乘法求导法则：
$$ dV = P^T \cdot dO $$

**2. 求 $dP$（关键衔接）**
同样因为 $O = P V$，对 $P$ 求导可得：
$$ dP = dO \cdot V^T $$

**3. 计算 Softmax 的梯度**
我们需要从 $dP$ 求得 $dS$。Softmax 的雅可比矩阵非常特殊：
已知 $P_i = \frac{e^{S_i}}{\sum e^{S_j}}$，应用链式法则后，可以按行写成下面的形式：
$$ dS = P \odot (dP - \text{row\_sum}(P \odot dP)) $$
(*注：$\odot$ 表示 Element-wise 逐元素乘法。后面的加和项是通过广播机制实现的*)

**4. 求 $dQ$ 和 $dK$**
此时我们已经拿到了 $dS$。本节统一使用缩放点积定义 $S = \frac{Q K^T}{\sqrt{d}}$，因此：
$$ dQ = dS \cdot K \cdot \frac{1}{\sqrt{d}} $$
$$ dK = dS^T \cdot Q \cdot \frac{1}{\sqrt{d}} $$

![Attention backward：梯度沿计算依赖逆向传播](../public/02_PyTorch_Algorithms/17_autograd_gradient_chain.svg)


### Step 3: saved tensors 与反向显存代价

反向计算需要前向阶段留下部分中间状态。显式 Attention 路径通常要保留 $Q、K、V$ 以及归一化后的 $P$，再用 `numel × element_size` 估算它们的理论容量；其中 $P$ 含有两个序列维度，因此序列变长时，保存容量和 HBM 读写都会受到放大。这个账本用于解释压力来源，不等同于 CUDA 峰值。若反向仍需要归一化结果，就要思考能否不保存完整的 $P$，而是在分块计算中保留足够的统计量。FlashAttention 通过分块计算、online softmax 和片上数据复用，改变了中间状态的保存与搬运方式；实际显存和速度收益仍需结合 GPU、序列长度、dtype 和 workload 测量。

| 前向状态或策略 | 反向用途 | 需要关注的代价 | 后续机制 |
| --- | --- | --- | --- |
| $Q / K / V$ | 回到输入投影的梯度 | 保存量随 batch、序列长度和 head 维度变化 | checkpoint 可选择重算部分状态 |
| $P$ | 计算 $dV$、$dP$ 并穿过 Softmax | 形状含 $N×N$，可能增加激活显存和 HBM 读写 | FlashAttention 避免显式保存完整 $P$ |
| 只保存边界或重算 | 在反向阶段恢复中间结果 | 用额外计算换保存空间 | activation checkpointing / online softmax |
| 理论账本 | 统计各状态的 numel 与 dtype 字节数 | 只能说明保存对象的容量下界 | 真实峰值需由 profiler 或项目 benchmark 验证 |

### Step 4: 实现并验证 Attention backward

现在把前面三步的公式和状态关系写成 `torch.autograd.Function`，再用数值和自动求导结果检查实现。下面的表格把实现范围、关键机制和验证方式放在一起，代码单元中的 TODO 与这些行一一对应。



| 实现部分 | 机制 | 验证方式 |
| --- | --- | --- |
| 输入范围 | 无 mask、无 dropout、单头、等长 Q/K/V 的缩放点积 Attention | 确认输入形状与 dtype 满足测试条件 |
| `forward` | 计算 `S → P → O`，保存 backward 需要的状态 | 检查输出形状和有限值 |
| `backward` | 依次计算 `dV → dP → dS → dQ/dK`；Softmax 按行计算修正项 | 与 PyTorch 自动求导对照 |
| `ctx.saved_tensors` | 读取前向保存的 `Q/K/V/P` | 检查状态与梯度形状对应 |
| `gradcheck` | 用数值梯度检验手写梯度 | 通过容差检查 |


```python
import torch
import torch.nn.functional as F
import math
```


```python
class CustomAttention(torch.autograd.Function):
    """无 mask、无 dropout 的教学版缩放点积 Attention。

    输入和输出均使用 [B, N, d]；本题只实现单头、等长 Q/K/V 的路径。
    """
    @staticmethod
    def forward(ctx, q, k, v):
        """计算缩放点积 Attention。

        Args:
            q, k, v: [B, N, d] 张量，device 和 dtype 必须一致。

        Returns:
            [B, N, d] 的 attention 输出。
        """
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError('q、k、v 必须是 [batch, seq_len, head_dim] 三维张量')
        if q.shape[:2] != k.shape[:2] or k.shape[:2] != v.shape[:2]:
            raise ValueError('q、k、v 的 batch 和序列长度必须一致')
        if q.size(-1) != k.size(-1) or v.size(-1) != q.size(-1):
            raise ValueError('q、k、v 的 head_dim 必须一致')
        if q.device != k.device or k.device != v.device:
            raise ValueError('q、k、v 必须位于同一 device')
        if q.dtype != k.dtype or k.dtype != v.dtype:
            raise TypeError('q、k、v 必须使用相同 dtype')
        # 1. 缩放点积
        d_k = q.size(-1)
        scale = 1.0 / math.sqrt(d_k)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # 2. Softmax 获取概率 P
        p = F.softmax(scores, dim=-1)
        
        # 3. 乘上 V 得到输出
        out = torch.matmul(p, v)
        
        # 保存反向传播需要用到的张量
        ctx.save_for_backward(q, k, v, p)
        ctx.scale = scale
        
        return out

    @staticmethod
    def backward(ctx, dout):
        """根据上游输出梯度返回 q、k、v 的梯度。

        Args:
            dout: 与 forward 输出同形状的上游梯度。

        Returns:
            dq、dk、dv，形状分别与 q、k、v 相同。
        """
        # 这些张量在 forward 中由 ctx.save_for_backward 保存，backward 会重新读取它们。
        q, k, v, p = ctx.saved_tensors
        scale = ctx.scale
        if dout.shape != q.shape:
            raise ValueError('dout 必须与 Attention 输出具有相同形状')
        
        # ==========================================
        # TODO 1: 求 dV
        # 提示：由 out = P @ V，使用 P.transpose(-2, -1) @ dout。
        # ==========================================
        # dv = ???
        
        # ==========================================
        # TODO 2: 求 dP
        # 提示：由 out = P @ V，使用 dout @ V.transpose(-2, -1)。
        # ==========================================
        # dp = ???
        
        # ==========================================
        # TODO 3: 穿过 Softmax 求 dS
        # 提示：对最后一维求 row_sum，避免显式构造 Softmax 雅可比矩阵。
        # ==========================================
        # dp_mul_p = ???
        # row_sum = ???
        # ds = ???
        
        # ==========================================
        # TODO 4: 求 dQ 和 dK（注意 scale 会同时传回两条输入支路）
        # 提示：由 S = Q @ K.transpose(-2, -1) * scale 回传到 q / k。
        # ==========================================
        # dq = ???
        # dk = ???
        
        return dq, dk, dv

```

### 测试

运行下面的测试单元，确认手写 `backward` 和 PyTorch 自动求导保持一致。

```python
# 运行此单元格以测试你的实现
def _legacy_test_attention_backward():
    """验证前向数值、输入梯度形状、梯度数值和输入校验。"""
    torch.manual_seed(42)
    B, N, d = 2, 8, 16

    # 使用 float64，便于 gradcheck 用有限差分检查手写 backward。
    q = torch.randn(B, N, d, dtype=torch.float64, requires_grad=True)
    k = torch.randn(B, N, d, dtype=torch.float64, requires_grad=True)
    v = torch.randn(B, N, d, dtype=torch.float64, requires_grad=True)

    print("1. 测试前向传播和输出形状...")
    custom_out = CustomAttention.apply(q, k, v)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    ref_out = torch.matmul(F.softmax(scores, dim=-1), v)
    assert custom_out.shape == (B, N, d), "Attention 输出形状不一致！"
    assert torch.isfinite(custom_out).all(), "前向输出包含 NaN 或 Inf！"
    assert torch.allclose(custom_out, ref_out), "前向传播结果不一致！"

    print("\n2. 进行梯度数值检验 (Gradcheck)...")
    assert torch.autograd.gradcheck(CustomAttention.apply, (q, k, v), eps=1e-6, atol=1e-4)

    # 用同一组输入分别回传，检查 q/k/v 的梯度形状、有限性和数值。
    q_ref, k_ref, v_ref = [t.detach().clone().requires_grad_() for t in (q, k, v)]
    ref_scores = torch.matmul(q_ref, k_ref.transpose(-2, -1)) / math.sqrt(d)
    torch.matmul(F.softmax(ref_scores, dim=-1), v_ref).sum().backward()
    q_custom, k_custom, v_custom = [t.detach().clone().requires_grad_() for t in (q, k, v)]
    CustomAttention.apply(q_custom, k_custom, v_custom).sum().backward()
    for custom_grad, reference_grad, tensor in zip((q_custom.grad, k_custom.grad, v_custom.grad), (q_ref.grad, k_ref.grad, v_ref.grad), (q_custom, k_custom, v_custom)):
        assert custom_grad.shape == tensor.shape, "梯度形状与输入不一致！"
        assert torch.isfinite(custom_grad).all(), "梯度包含 NaN 或 Inf！"
        assert torch.allclose(custom_grad, reference_grad, atol=1e-5), "手写梯度与 PyTorch 结果不一致！"

    # 输入校验应报告具体原因，而不是把错误伪装成 TODO 未完成。
    bad_k = torch.randn(B, N + 1, d, dtype=torch.float64)
    try:
        CustomAttention.apply(q.detach(), bad_k, v.detach())
    except ValueError as error:
        assert "batch 和序列长度" in str(error)
    else:
        raise AssertionError("不匹配的序列长度应该触发 ValueError")

    print("✅ All Tests Passed! Attention 反向传播实现通过测试。")

# 机制测试入口：把前向、梯度、gradcheck 和输入契约分别验证。
def _build_attention_backward_case():
    torch.manual_seed(42)
    q = torch.randn(2, 8, 16, dtype=torch.double, requires_grad=True)
    k = torch.randn(2, 8, 16, dtype=torch.double, requires_grad=True)
    v = torch.randn(2, 8, 16, dtype=torch.double, requires_grad=True)
    return q, k, v

def _assert_attention_forward(q, k, v):
    output = CustomAttention.apply(q, k, v)
    scores = (q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
    expected = torch.softmax(scores, dim=-1) @ v
    assert torch.allclose(output, expected, atol=1e-8), "forward 数值不一致"

def _assert_attention_gradients(q, k, v):
    output = CustomAttention.apply(q, k, v)
    output.sum().backward()
    assert q.grad is not None and k.grad is not None and v.grad is not None, "输入梯度缺失"
    assert q.grad.shape == q.shape and k.grad.shape == k.shape and v.grad.shape == v.shape, "梯度形状错误"

def _assert_attention_gradcheck():
    torch.manual_seed(7)
    q = torch.randn(1, 3, 4, dtype=torch.double, requires_grad=True)
    k = torch.randn(1, 3, 4, dtype=torch.double, requires_grad=True)
    v = torch.randn(1, 3, 4, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(CustomAttention.apply, (q, k, v), eps=1e-6, atol=1e-4), "gradcheck 未通过"

def _assert_attention_input_contract():
    q = torch.randn(2, 3, 4)
    try:
        CustomAttention.apply(q, q[:, :2], q)
    except ValueError:
        return
    raise AssertionError("不匹配的序列长度应该触发 ValueError")

def test_attention_backward():
    try:
        q, k, v = _build_attention_backward_case()
        _assert_attention_forward(q, k, v)
        _assert_attention_gradients(q, k, v)
        _assert_attention_gradcheck()
        _assert_attention_input_contract()
        print("✅ Attention 反向传播的前向、梯度和输入契约测试通过。")
    except (AttributeError, NameError, TypeError, ValueError) as e:
        raise NotImplementedError("请先完成 TODO 部分。") from e
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
        raise NotImplementedError("请先完成 TODO 部分。") from e

test_attention_backward()

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
class CustomAttention(torch.autograd.Function):
    """无 mask、无 dropout 的教学版缩放点积 Attention。

    输入和输出均使用 [B, N, d]；本题只实现单头、等长 Q/K/V 的路径。
    """
    @staticmethod
    def forward(ctx, q, k, v):
        """计算缩放点积 Attention。

        Args:
            q, k, v: [B, N, d] 张量，device 和 dtype 必须一致。

        Returns:
            [B, N, d] 的 attention 输出。
        """
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError('q、k、v 必须是 [batch, seq_len, head_dim] 三维张量')
        if q.shape[:2] != k.shape[:2] or k.shape[:2] != v.shape[:2]:
            raise ValueError('q、k、v 的 batch 和序列长度必须一致')
        if q.size(-1) != k.size(-1) or v.size(-1) != q.size(-1):
            raise ValueError('q、k、v 的 head_dim 必须一致')
        if q.device != k.device or k.device != v.device:
            raise ValueError('q、k、v 必须位于同一 device')
        if q.dtype != k.dtype or k.dtype != v.dtype:
            raise TypeError('q、k、v 必须使用相同 dtype')
        d_k = q.size(-1)
        scale = 1.0 / math.sqrt(d_k)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        p = F.softmax(scores, dim=-1)
        out = torch.matmul(p, v)
        
        ctx.save_for_backward(q, k, v, p)
        ctx.scale = scale
        
        return out

    @staticmethod
    def backward(ctx, dout):
        """根据上游输出梯度返回 q、k、v 的梯度。

        Args:
            dout: 与 forward 输出同形状的上游梯度。

        Returns:
            dq、dk、dv，形状分别与 q、k、v 相同。
        """
        q, k, v, p = ctx.saved_tensors
        scale = ctx.scale
        
        # TODO 1: 求 dV
        # 提示：由 out = P @ V，使用 P.transpose(-2, -1) @ dout。
        dv = torch.matmul(p.transpose(-2, -1), dout)
        
        # TODO 2: 求 dP
        # 提示：由 out = P @ V，使用 dout @ V.transpose(-2, -1)。
        dp = torch.matmul(dout, v.transpose(-2, -1))
        
        # TODO 3: 穿过 Softmax 求 dS
        # 提示：对最后一维求 row_sum，避免显式构造 Softmax 雅可比矩阵。
        dp_mul_p = dp * p
        row_sum = dp_mul_p.sum(dim=-1, keepdim=True)
        ds = p * (dp - row_sum)
        
        # TODO 4: 求 dQ 和 dK
        # 提示：由 S = Q @ K.transpose(-2, -1) * scale 回传到 q / k。
        dq = torch.matmul(ds, k) * scale
        dk = torch.matmul(ds.transpose(-2, -1), q) * scale
        
        return dq, dk, dv

```

### 解析

**1. TODO 1: 求 dV**

- **实现方式**：`dv = torch.matmul(p.transpose(-2, -1), dout)`
- **数学原理**：输出 `out = P V`，所以对 `V` 的梯度就是把上游梯度乘回去。
- **工程意义**：这是 Attention 反向里最直接的一步，先把 Value 方向的梯度拿到。

**2. TODO 2: 求 dP**

- **实现方式**：`dp = torch.matmul(dout, v.transpose(-2, -1))`
- **数学原理**：同样由 `out = P V` 得到，对 `P` 的梯度可以直接通过矩阵乘法反推。
- **工程意义**：这一步把输出梯度重新映射回注意力概率矩阵。

**3. TODO 3: 穿过 Softmax 求 dS**

- **实现方式**：先算 `dp_mul_p = dp * p`，再对行求和得到 `row_sum`，最后得到 `ds = p * (dp - row_sum)`。
- **数学原理**：Softmax 的反向可以化成一个稳定的逐行修正项，不需要显式构造完整雅可比矩阵。
- **工程意义**：这一步决定了 Softmax 输出梯度如何回到打分矩阵，是实现中需要重点检查的环节。

**4. TODO 4: 求 dQ 和 dK**

- **实现方式**：`dq = torch.matmul(ds, k) * scale`，`dk = torch.matmul(ds.transpose(-2, -1), q) * scale`
- **数学原理**：因为 `scores = QK^T / sqrt(d)`，所以最后回到 `Q` 和 `K` 时还要乘上缩放因子。
- **工程意义**：这一步把 Attention 的反向梯度真正落回输入表示。

**进阶思考**

- 如果不保存 `P`，反向传播还能怎么做？
- 为什么这会自然引出 FlashAttention 的重计算思想？
- 你能把这条链路和第 20 节的在线 Softmax 对上吗？

---

## 相关阅读

**导语：** 手写 Attention backward 后，可以回到 PyTorch 的自动求导接口，再沿激活、损失和性能分析继续深入。

- [PyTorch Autograd 官方文档](https://pytorch.org/docs/stable/autograd.html)
- [PyTorch 自定义 `autograd.Function` 文档](https://pytorch.org/docs/stable/notes/extending.html)
- [18. 激活函数与损失反向传播](../02_PyTorch_Algorithms/18_Activation_and_Loss_Backward.md)
- [20. FlashAttention 模拟](../02_PyTorch_Algorithms/20_FlashAttention_Sim.md)
- [P1: 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
  
---