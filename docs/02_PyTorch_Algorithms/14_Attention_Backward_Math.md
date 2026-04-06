# 14 Attention Backward Math

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/14_Attention_Backward_Math.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 14. 注意力机制反向传播推导与自定义 Autograd (Attention Backward)

**难度：** Hard | **标签：** `微积分`, `PyTorch`, `Autograd` | **目标人群：** 底层算子开发、高阶算法面试

在第 04 节，我们完成了多头注意力机制（MHA）的前向传播搭建。然而，**“大模型训练”与“推理”核心挑战之一在于反向传播 (Backward Pass)。**

为什么各大厂商都在持续优化 FlashAttention？因为在训练时，为了计算反向传播的梯度，框架必须在显存中保存尺寸为 $N \times N$ (序列长度的平方) 的庞大注意力分数矩阵（Attention Probabilities）。当序列长度达到 128K 时，仅这一个中间变量就会极易导致 GPU 显存溢出 (OOM)。

在下一节正式进入 FlashAttention 之前，我们必须跨过这座高山：**彻底搞懂 Attention 反向传播的微积分推导，并抛弃 PyTorch 原生的 `.backward()`，利用 `torch.autograd.Function` 写出它的反向梯度计算代码！**

这是底层架构岗位的常见考核点。

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
我们需要从 $dP$ 求得 $dS$。Softmax 的雅可比矩阵极其特殊：
已知 $P_i = \frac{e^{S_i}}{\sum e^{S_j}}$，其对于 $S$ 的导数在应用链式法则后，会化简为一个极其优美的形式：
$$ dS = P \odot (dP - \text{row\_sum}(P \odot dP)) $$
(*注：$\odot$ 表示 Element-wise 逐元素乘法。后面的加和项是通过广播机制实现的*)

**4. 求 $dQ$ 和 $dK$**
此时我们已经拿到了 $dS$。因为 $S = Q K^T$（如果带缩放因子则是 $S = \frac{Q K^T}{\sqrt{d}}$）：
$$ dQ = \frac{dS \cdot K}{\sqrt{d}} $$
$$ dK = \frac{dS^T \cdot Q}{\sqrt{d}} $$

### Step 3: 手撕 PyTorch Autograd Function

现在，把你刚才看到的微积分公式，转化为能够实际运行的代码。我们将继承 `torch.autograd.Function`。

**要求**：完成 `backward` 函数中 TODO 的数学推导代码。你可以使用 `ctx.saved_tensors` 来获取前向传播时保存的 $Q, K, V, P$ 等变量。

```python
import torch
import torch.nn.functional as F
import math

class CustomAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        # 1. 缩放点积
        d_k = q.size(-1)
        scale = 1.0 / math.sqrt(d_k)
        
        # S = QK^T * scale.  Shape: [batch, N, N]
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
        # 公式: dV = P^T * dO
        # ==========================================
        # dv = ???
        
        # ==========================================
        # TODO 2: 求 dP
        # 公式: dP = dO * V^T
        # ==========================================
        # dp = ???
        
        # ==========================================
        # TODO 3: 穿过 Softmax 求 dS
        # 公式: dS = P * (dP - row_sum(P * dP))
        # 提示: 使用 .sum(dim=-1, keepdim=True) 计算 row_sum
        # ==========================================
        # dp_mul_p = ???
        # row_sum = ???
        # ds = ???
        
        # ==========================================
        # TODO 4: 求 dQ 和 dK (别忘了乘以 scale 缩放因子)
        # dQ = (dS * K) * scale
        # dK = (dS^T * Q) * scale
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
        
        print("\n2. 进行黑魔法级别的梯度数值检验 (Gradcheck)...")
        # torch.autograd.gradcheck 会用极限逼近法计算数值梯度，并与你的解析梯度进行极其严苛的对比
        test_passed = torch.autograd.gradcheck(CustomAttention.apply, (q, k, v), eps=1e-6, atol=1e-4)
        
        if test_passed:
            print("✅ All Tests Passed! 你成功手推了困扰无数人的 Attention 反向传播微积分！")
            
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
    except Exception as e:
        print(f"❌ 发生异常 (很可能是梯度公式写错了): {e}")

test_attention_backward()

```

### Step 4: 工业界的现实与破局（预告）

看看你刚才写的 `ctx.save_for_backward(q, k, v, p)`。这行代码在反向传播被调用前，会**一直把 $P$ 锁在显存里**。

如果现在的上下文是 $128K$（如 GPT-4），$P$ 的大小就是 $128K \times 128K$。即便在 FP16 精度下，**单单存这一个 $P$ 矩阵，一个 Batch 就需要占用约 32 GB 的显存！** 稍微开大点 Batch Size，连 80G 的 A100 都会触发 OOM。

> **思考题**：如果你是底层算法工程师，怎么解决这个问题？
> **答案预告**：不存 $P$！我们在反向传播需要 $P$ 的时候，**拿 $Q$ 和 $K$ 现场重算一次 $P$（Recomputation）！** 通过巧妙的 SRAM 分块加载机制，虽然计算量变大了，但因为避免了把庞大的 $P$ 写入又读出极其缓慢的 HBM，最终不但不 OOM，**速度反而变快了 3 倍！**

这就是下一节业界广泛使用的 **FlashAttention** 所做的事。

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

::: details 💡 点击查看官方解析与参考代码

Attention 梯度的核心在于处理 Softmax 的雅可比矩阵。对于 $dP$，通过矩阵转置操作可以轻松由链式法则获得；随后通过提取 $P \odot dP$ 并对行求和，构造出 Softmax 的反向传播梯度 $dS$。这也是理解重计算（Recomputation）技术以节省大规模训练显存的理论基础。

```python
import torch
import torch.nn.functional as F
import math

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
        
        # 1. dV = P^T * dO
        dv = torch.matmul(p.transpose(-2, -1), dout)
        
        # 2. dP = dO * V^T
        dp = torch.matmul(dout, v.transpose(-2, -1))
        
        # 3. dS = P * (dP - row_sum(P * dP))
        dp_mul_p = dp * p
        row_sum = dp_mul_p.sum(dim=-1, keepdim=True)
        ds = p * (dp - row_sum)
        
        # 4. dQ 和 dK
        dq = torch.matmul(ds, k) * scale
        dk = torch.matmul(ds.transpose(-2, -1), q) * scale
        
        return dq, dk, dv

```

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
