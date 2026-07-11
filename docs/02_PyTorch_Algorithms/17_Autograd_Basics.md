# 17. Autograd Basics | 自动微分基础

**难度：** Medium | **环境：** CPU-first | **标签：** `Autograd`, `Backward`, `梯度` | **目标人群：** 底层算子开发与算法基础训练

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/17_Autograd_Basics.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


先把 Attention 前向、链式法则和 PyTorch Autograd 的回传关系串起来，再去看激活函数和损失函数的反向，推导会更顺。

**关键词：** `Autograd`, `backward`, `gradcheck`
## 前置阅读

**导语：** 先把 Autograd 的反向传播路径补齐，再回到自定义反传实现会更顺。

- [07. PyTorch Autograd and Backward | PyTorch 自动求导与反向传播](../00_Prerequisites/07_PyTorch_Autograd_and_Backward.md)
- [13. Simple Neural Network Training | 简单神经网络训练循环](../00_Prerequisites/13_Simple_Neural_Network_Training.md)


## 相关阅读

**导语：** 自定义反传之后，最自然的延伸就是激活函数和损失函数的反向推导。

- [13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
- [18. Activation and Loss Backward | 激活函数与损失反向传播](../02_PyTorch_Algorithms/18_Activation_and_Loss_Backward.md)

### Step 1: 前向传播回顾与变量定义

为了不打断思路，我们先简洁回顾一下 04 节的单头 Attention 前向公式（省略缩放因子 $\sqrt{d}$ 简化推导，后文代码中会加回）：

1. **打分矩阵**：$S = Q K^T$
2. **概率矩阵**：$P = \text{Softmax}(S, \text{dim}=-1)$
3. **最终输出**：$O = P V$

> **张量形状说明：**
> - $Q, K, V \in \mathbb{R}^{N \times d}$ (序列长度 $N$，特征维数 $d$)
> - $S, P \in \mathbb{R}^{N \times N}$
> - $O \in \mathbb{R}^{N \times d}$
### Step 2: 链式法则逆流而上 (微积分时间)

假设下游的损失函数已经帮我们算好了输出张量 $O$ 的梯度 $\nabla O$（通常简写为 $dO$）。我们的任务是求出 $dQ, dK, dV$。

**1. 求 $dV$（最简单的）**
因为 $O = P V$，根据矩阵乘法求导法则：
$$ dV = P^T \cdot dO $$

**2. 求 $dP$（关键衔接）**
同样因为 $O = P V$，对 $P$ 求导可得：
$$ dP = dO \cdot V^T $$

**3. 跨越 Softmax (核心难点)**
我们需要从 $dP$ 求得 $dS$。Softmax 的雅可比矩阵非常特殊：
已知 $P_i = \frac{e^{S_i}}{\sum e^{S_j}}$，其对于 $S$ 的导数在应用链式法则后，会化简为一个非常优美的形式：
$$ dS = P \odot (dP - \text{row\_sum}(P \odot dP)) $$
(*注：$\odot$ 表示 Element-wise 逐元素乘法。后面的加和项是通过广播机制实现的*)

**4. 求 $dQ$ 和 $dK$**
此时我们已经拿到了 $dS$。因为 $S = Q K^T$（如果带缩放因子则是 $S = \frac{Q K^T}{\sqrt{d}}$）：
$$ dQ = \frac{dS \cdot K}{\sqrt{d}} $$
$$ dK = \frac{dS^T \cdot Q}{\sqrt{d}} $$
### Step 3: 手撕 PyTorch Autograd Function

现在，把你刚才看到的微积分公式，转化为能够实际运行的代码。我们将继承 `torch.autograd.Function`。

**要求**：完成 `backward` 函数中 TODO 的数学推导代码。你可以使用 `ctx.saved_tensors` 来获取前向传播时保存的 $Q, K, V, P$ 等变量。
这一节的实现顺序就是先求 `dV / dP`，再穿过 Softmax 得到 `dS`，最后回到 `dQ / dK`。

```python
import torch
import torch.nn.functional as F
import math
```


```python
class CustomAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
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
        # 提取前向保存的张量
        q, k, v, p = ctx.saved_tensors
        scale = ctx.scale
        
        # ==========================================
        # TODO 1: 求 dV
        # ==========================================
        # dv = ???
        
        # ==========================================
        # TODO 2: 求 dP
        # ==========================================
        # dp = ???
        
        # ==========================================
        # TODO 3: 穿过 Softmax 求 dS
        # ==========================================
        # dp_mul_p = ???
        # row_sum = ???
        # ds = ???
        
        # ==========================================
        # TODO 4: 求 dQ 和 dK (别忘了乘以 scale 缩放因子)
        # ==========================================
        # dq = ???
        # dk = ???
        
        return dq, dk, dv

```


```python
# 运行此单元格以测试你的实现
def test_attention_backward():
    try:
        torch.manual_seed(42)
        B, N, d = 2, 8, 16
        
        # 随机初始化张量，必须要求梯度
        q = torch.randn(B, N, d, dtype=torch.float64, requires_grad=True)
        k = torch.randn(B, N, d, dtype=torch.float64, requires_grad=True)
        v = torch.randn(B, N, d, dtype=torch.float64, requires_grad=True)
        
        print("1. 测试前向传播是否正常...")
        custom_out = CustomAttention.apply(q, k, v)
        
        # 原生 PyTorch 实现
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        ref_out = torch.matmul(F.softmax(scores, dim=-1), v)
        
        assert torch.allclose(custom_out, ref_out), "前向传播结果不一致！"
        
        print("\n2. 进行梯度数值检验 (Gradcheck)...")
        test_passed = torch.autograd.gradcheck(CustomAttention.apply, (q, k, v), eps=1e-6, atol=1e-4)
        
        if test_passed:
            print("✅ All Tests Passed! Attention 反向传播实现通过测试。")
            
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError, RuntimeError) as e:
        if isinstance(e, AttributeError):
            print("代码未完成，无法找到必要的属性")
        elif isinstance(e, NameError):
            print("代码可能未完成，导致了变量未定义")
        elif isinstance(e, TypeError):
            print("代码可能未完成，导致了类型错误")
        elif isinstance(e, ValueError):
            print("代码可能未完成，导致了张量维度错误")
        elif isinstance(e, AssertionError):
            print(f"代码可能未完成，导致了断言失败: {e}")
        else:
            print("代码可能未完成，导致了 gradcheck 或反向传播异常")
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e
    except Exception as e:
        print(f"❌ 发生异常: {e}")
        raise

test_attention_backward()

```

### Step 4: 工业界的现实与破局（预告）

看看你刚才写的 `ctx.save_for_backward(q, k, v, p)`。这行代码在反向传播被调用前，会**一直把 $P$ 锁在显存里**。

如果现在的上下文是 $128K$（如 GPT-4），$P$ 的大小就是 $128K \times 128K$。即便在 FP16 精度下，**单单存这一个 $P$ 矩阵，一个 Batch 就需要占用约 32 GB 的显存！** 稍微开大点 Batch Size，连 80G 的 A100 都会触发 OOM。

> **思考题**：如果你是底层算法工程师，怎么解决这个问题？
> **答案预告**：不存 $P$！我们在反向传播需要 $P$ 的时候，**拿 $Q$ 和 $K$ 现场重算一次 $P$（Recomputation）！** 通过巧妙的 SRAM 分块加载机制，虽然计算量变大了，但因为避免了把庞大的 $P$ 写入又读出非常缓慢的 HBM，最终不但不 OOM，**速度反而变快了 3 倍！**

这就是下一节业界广泛使用的 **FlashAttention** 所做的事。
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
    @staticmethod
    def forward(ctx, q, k, v):
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
        q, k, v, p = ctx.saved_tensors
        scale = ctx.scale
        
        # TODO 1: 求 dV
        dv = torch.matmul(p.transpose(-2, -1), dout)
        
        # TODO 2: 求 dP
        dp = torch.matmul(dout, v.transpose(-2, -1))
        
        # TODO 3: 穿过 Softmax 求 dS
        dp_mul_p = dp * p
        row_sum = dp_mul_p.sum(dim=-1, keepdim=True)
        ds = p * (dp - row_sum)
        
        # TODO 4: 求 dQ 和 dK
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
- **工程意义**：这是 Attention 反向里最关键的一步，也是很多讲解容易卡住的地方。

**4. TODO 4: 求 dQ 和 dK**

- **实现方式**：`dq = torch.matmul(ds, k) * scale`，`dk = torch.matmul(ds.transpose(-2, -1), q) * scale`
- **数学原理**：因为 `scores = QK^T / sqrt(d)`，所以最后回到 `Q` 和 `K` 时还要乘上缩放因子。
- **工程意义**：这一步把 Attention 的反向梯度真正落回输入表示。

**进阶思考**

- 如果不保存 `P`，反向传播还能怎么做？
- 为什么这会自然引出 FlashAttention 的重计算思想？
- 你能把这条链路和第 20 节的在线 Softmax 对上吗？
