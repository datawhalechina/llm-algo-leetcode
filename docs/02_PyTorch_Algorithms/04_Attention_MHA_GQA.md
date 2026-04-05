# 04 Attention MHA GQA

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/04_Attention_MHA_GQA.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 04. 注意力机制与 KV Cache (MHA / GQA / MQA)

**难度：** Medium | **标签：** `基础架构`, `PyTorch`, `推理优化` | **目标人群：** 模型微调与工程部署

欢迎来到 LLM-LeetCode！本节我们将深入解析大语言模型的核心组件：**注意力机制**，并实现支持 KV Cache 和 GQA (Grouped-Query Attention) 的代码。


### Step 1: 核心思想与痛点

> **为什么需要 MQA 和 GQA？**
> * **MHA (Multi-Head Attention)**: 标准的多头注意力。每个 Query 头都有自己专属的 Key 和 Value 头。在推理（Decode阶段）时，读取巨量的 KV Cache 会遇到**内存带宽瓶颈 (Memory-bound)**。
> * **MQA (Multi-Query Attention)**: 所有的 Query 头共享**同一个** Key 和 Value 头。极大地减少了 KV Cache 的显存占用和访存量，但模型能力会有所下降。
> * **GQA (Grouped-Query Attention)**: LLaMA-2/3 采用的折中方案。将 Query 头分组，每组共享一个 Key 和 Value 头。在**性能和显存之间取得了准确的平衡**。

> **什么是 KV Cache？**
> 在自回归生成中，每次生成第 $N$ 个 Token，我们都需要前面 $N-1$ 个 Token 的信息。为了避免重复计算前面的特征，我们把前 $N-1$ 步算好的 Key 和 Value 张量**缓存（Cache）**起来，在当前步直接拼接参与计算。


###  Step 2: 核心公式与张量维度

**注意力计算公式：**
$$ \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

**张量维度追踪 (Shape Tracking) - 算法工程师的灵魂：**
假设 `Batch=B`, `Seq_len=S`, `Num_Heads=H`, `Head_Dim=D`
1. 线性投影后：`Q` 形状为 `[B, S, H * D]`
2. 切分多头后：转置为 `[B, H, S, D]`
3. 注意力分数计算：`Q @ K^T` -> `[B, H, S, D] @ [B, H, D, S]` -> `[B, H, S, S]`
4. 乘以 Value：`Scores @ V` -> `[B, H, S, S] @ [B, H, S, D]` -> `[B, H, S, D]`
5. 最后合并多头：转置回 `[B, S, H, D]` 并 `view` 成 `[B, S, H * D]`。


###  Step 3: 工业界源码映射

在真实的工业界代码中，这段逻辑在哪里？
* **HuggingFace LLaMA**: `transformers/models/llama/modeling_llama.py` 中的 `LlamaAttention` 类。
* **vLLM (推理框架)**: 核心关注它的 PagedAttention 实现，用来解决这里 KV Cache 的显存碎片化问题。


###  Step 4: 动手实战

**要求**：请补全下方 `GroupedQueryAttention` 的 `forward` 函数中的 `TODO` 部分，实现：
1. 张量的多头切分与 Reshape
2. KV Cache 的拼接逻辑
3. 注意力分数的计算


```python
import torch
import torch.nn as nn
import math

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    将 KV 头复制 n_rep 次，以匹配 Query 头的数量 (GQA/MQA 需要)
    """
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)

class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_kv_heads: int = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.head_dim = hidden_dim // num_heads
        
        # 定义投影矩阵
        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)

    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: torch.Tensor = None, 
        kv_cache: tuple[torch.Tensor, torch.Tensor] = None
    ):
        batch_size, seq_len, _ = x.shape
        
        # 1. 线性投影
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        # ==========================================
        # TODO 1: Reshape xq, xk, xv 以适配多头注意力计算
        # 目标形状: [batch_size, num_heads (或 kv_heads), seq_len, head_dim]
        # 提示: 使用 .view() 和 .transpose()
        # ==========================================
        # xq = ???
        # xk = ???
        # xv = ???
        
        
        # ==========================================
        # TODO 2: 处理 KV Cache
        # 如果 kv_cache 不为空，将之前的 k_cache 和当前的 xk 在 seq_len 维度(dim=2)拼接
        # ==========================================
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            # xk = ???
            # xv = ???
            
        new_kv_cache = (xk, xv)
        
        # 通过 repeat_kv 把 GQA 的 KV 头数扩充到和 Query 数量一致
        xk = repeat_kv(xk, self.num_queries_per_kv)
        xv = repeat_kv(xv, self.num_queries_per_kv)
        
        # ==========================================
        # TODO 3: 计算注意力分数 (Scaled Dot-Product)
        # 1. scores = Q @ K^T / sqrt(d)
        # 2. probs = softmax(scores + mask)
        # 3. output = probs @ V
        # ==========================================
        # scores = ???
        
        if attention_mask is not None:
            scores = scores + attention_mask
            
        # probs = ???
        # output = ???
        
        
        # ==========================================
        # TODO 4: 恢复形状并输出
        # [B, H, S, D] -> [B, S, H*D]
        # ==========================================
        # output = ???
        
        # return self.o_proj(output), new_kv_cache
        pass

```

```python
# 运行此单元格以测试你的实现
def test_mha_mqa_gqa():
    try:
        batch_size, seq_len, hidden_dim, num_heads = 2, 16, 128, 4
        
        # 1. 测试 MHA
        print("Testing MHA (Multi-Head Attention)...")
        mha = GroupedQueryAttention(hidden_dim, num_heads, num_kv_heads=num_heads)
        x = torch.randn(batch_size, seq_len, hidden_dim)
        out, _ = mha(x)
        assert out.shape == (batch_size, seq_len, hidden_dim), "MHA 输出形状错误!"
        
        # 2. 测试 GQA
        print("Testing GQA (Grouped-Query Attention)...")
        gqa = GroupedQueryAttention(hidden_dim, num_heads, num_kv_heads=2)
        out, _ = gqa(x)
        assert out.shape == (batch_size, seq_len, hidden_dim), "GQA 输出形状错误!"
        
        # 3. 测试 KV Cache
        print("Testing KV Cache Autoregressive Decoding...")
        prefill_len = 5
        x_prefill = torch.randn(batch_size, prefill_len, hidden_dim)
        _, kv_cache = mha(x_prefill)
        
        x_decode = torch.randn(batch_size, 1, hidden_dim)
        out_decode, new_kv_cache = mha(x_decode, kv_cache=kv_cache)
        assert new_kv_cache[0].shape == (batch_size, num_heads, prefill_len + 1, hidden_dim // num_heads), "KV Cache 更新错误!"
        
        print("\n✅ All Tests Passed! 恭喜你，大模型核心 Attention 算子实现成功！")
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
    except Exception as e:
        print(f"\n❌ 测试失败，请检查张量维度: {e}")

test_mha_mqa_gqa()

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

::: details 💡 点击查看官方解析与参考代码

Attention 算子的核心是张量维度的追踪与变换。首先要用 view + transpose 将 QKV 转为多头形式 [batch_size, num_heads, seq_len, head_dim]。如果是自回归推理，将历史的 KV cache 与当前的 xk, xv 在 seq_len 维度进行拼接。接着计算 Attention Scores 时，只需用 xq @ xk.transpose(2, 3) 进行最后两维的矩阵乘法，得到注意力权重后再去乘 xv。对于 GQA 支持，借助外部传入的重复函数 repeat_kv，先将少量 KV 头扩充到 Query 头的数量再进行运算。

```python
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)

class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_kv_heads: int = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)

    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: torch.Tensor = None, 
        kv_cache: tuple[torch.Tensor, torch.Tensor] = None
    ):
        batch_size, seq_len, _ = x.shape
        
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        xq = xq.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            xk = torch.cat([k_cache, xk], dim=2)
            xv = torch.cat([v_cache, xv], dim=2)
            
        new_kv_cache = (xk, xv)
        
        xk = repeat_kv(xk, self.num_queries_per_kv)
        xv = repeat_kv(xv, self.num_queries_per_kv)
        
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            scores = scores + attention_mask
            
        probs = torch.nn.functional.softmax(scores, dim=-1)
        output = torch.matmul(probs, xv)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.o_proj(output), new_kv_cache
```

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
