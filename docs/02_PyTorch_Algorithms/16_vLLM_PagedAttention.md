# 16 vLLM PagedAttention

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/16_vLLM_PagedAttention.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 16. 经典推理框架: 模拟 Continuous Batching 与 PagedAttention

**难度：** Hard | **标签：** `推理架构`, `vLLM` | **目标人群：** 核心 Infra 与算子开发

本节我们将揭秘工业界大模型推理框架（如 **vLLM**）的两大杀手锏技术：**Continuous Batching (连续批处理/动态批处理)** 和 **PagedAttention (分页注意力池)**。
这是目前算法面经里含金量最高，但资料最匮乏的部分！

> **相关阅读**:
> 本节使用纯 PyTorch 实现了算法逻辑与数学推导。
> 如果你想学习工业界如何打破该算子的 Memory Bound (访存瓶颈)，请前往 Triton 篇：
>  [`../03_Triton_Kernels/09_Triton_KV_Cache_and_PagedAttention.ipynb`](../03_Triton_Kernels/09_Triton_KV_Cache_and_PagedAttention.ipynb)



### Step 1: 核心思想与痛点

> **痛点 1：Static Batching 的低效**
> 在传统的 PyTorch 推理中，Batch 内的不同请求长度不一。如果 Request A 生成了 10 个 Token 就结束了，而 Request B 需要生成 100 个，那么 A 生成完后 GPU 只能干等 B（即用 Padding 填充计算），导致算力极度浪费。
> **解法：Continuous Batching (Orca/vLLM)**
> 打破 Static Batch 的概念，在 `Step` (Iteration) 粒度上动态重组。A 结束了，立刻把队列里的 Request C 塞进来接着算。

> **痛点 2：KV Cache 的显存碎片化**
> KV Cache 的显存大小是**不可预知的**（你不知道模型最终会生成多长的回复）。如果我们提前按 `max_len` 分配整块显存，会造成严重的内部碎片（超过 60% 浪费）。
> **解法：PagedAttention (vLLM)**
> 借鉴操作系统的虚拟内存管理。把显存切分成固定大小的 **Block** (比如 1个Block存16个Token)。在生成时，按需分配物理 Block，并通过 `Block Table` (块表) 记录虚拟 Token 序列到物理块的映射。


### Step 2: 代码实现框架
系统需要维护一个 `BlockTable`，它是一个二维字典或矩阵，记录了每个序列的逻辑 Block 对应着显存池（K_Cache 和 V_Cache 池）中的哪个物理 Block ID。在解码时，通过查询这个表，将散落的物理 Block 重新聚集起来，与当前的 Query 向量进行 Attention 点积。


###  Step 3: PagedAttention 模拟机制

为了让你在不写几千行 C++ 的情况下弄懂 PagedAttention，我们将用纯 Python 模拟它的核心数据结构：

1. **Physical Block Pool (物理块池)**：一个预先分配好的大张量，形状为 `[num_blocks, block_size, hidden_dim]`。
2. **Block Table (块表)**：每个 Request 都有一个专属的块表，它是一个整数列表（`List[int]`），记录了这个 Request 的第 $i$ 个逻辑块存在物理池的哪个索引里。
3. **KV Cache Manager**：负责在 Token 生成时，“按需”分配新的物理块索引。


###  Step 4: 动手实战

**要求**：请补全下方 `KVCacheManager`，实现一个极简版的 vLLM 内存管理器。


```python
import torch
from typing import List

class Request:
    def __init__(self, request_id: int, prompt_len: int):
        self.request_id = request_id
        self.seq_len = prompt_len
        # 记录此请求占据的物理 Block 索引
        self.block_table: List[int] = []

class KVCacheManager:
    def __init__(self, num_blocks: int, block_size: int, head_dim: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.head_dim = head_dim
        
        # 模拟预分配一块大显存池 (vLLM 会在 GPU 上分配几 GB)
        # 形状: [num_blocks, block_size, head_dim]
        self.physical_kv_cache = torch.zeros(num_blocks, block_size, head_dim)
        
        # 跟踪哪些物理块被占用了
        self.free_blocks: List[int] = list(range(num_blocks))

    def allocate_for_prefill(self, req: Request):
        """
        请求刚进来时 (Prefill阶段)，为它的 Prompt 长度分配所需的全部 Block
        """
        # ==========================================
        # TODO 1: 计算需要的 block 数量
        # 提示: 向上取整 (seq_len / block_size)
        # ==========================================
        # needed_blocks = ???
        
        # ==========================================
        # TODO 2: 从 free_blocks 中弹出对应数量的 block 索引，
        # 并追加到请求的 block_table 中
        # 如果 free_blocks 不够了，抛出 RuntimeError("OOM")
        # ==========================================
        # ???
        pass

    def allocate_for_decode(self, req: Request):
        """
        自回归生成时 (Decode阶段)，检查序列长度。
        如果当前最后一个 Block 满了，则按需分配 1 个新 Block。
        """
        req.seq_len += 1  # 长度加 1
        
        # ==========================================
        # TODO 3: 判断是否刚好需要跨入新的一块 Block？
        # 条件：加 1 后的 seq_len 除以 block_size 余数是多少？
        # ==========================================
        # is_new_block_needed = ???
        
        # 如果需要，尝试分配 1 个新的物理 Block 放入块表
        # if is_new_block_needed:
        #    if not self.free_blocks: ...
        #    req.block_table.append(???)
        pass

    def get_physical_cache(self, req: Request) -> torch.Tensor:
        """
        (模拟 PagedAttention 底层加载逻辑)
        根据块表，把不连续的物理块“拼凑”成逻辑上连续的 KV Cache (仅作验证用途)
        """
        # ==========================================
        # TODO 4: 根据 req.block_table 的索引，
        # 从 self.physical_kv_cache 中取出块，并用 torch.cat 在 seq_len (dim=0) 维度拼接
        # ==========================================
        # blocks = ???
        # cat_blocks = ???
        
        # 最后，只截取真实 seq_len 长度返回 (因为最后一个块可能没填满)
        # return cat_blocks[:req.seq_len]
        pass

```

```python
# 运行此单元格以测试你的实现
def test_paged_attention_manager():
    try:
        manager = KVCacheManager(num_blocks=10, block_size=4, head_dim=64)
        print("初始化内存池...")
        
        # 1. 模拟一个 Request (Prompt 长度为 6)
        req1 = Request(request_id=1, prompt_len=6)
        
        manager.allocate_for_prefill(req1)
        assert len(req1.block_table) == 2, "长度 6 的请求应分配 2 个 Block！"
        assert len(manager.free_blocks) == 8, "池中应该剩下 8 个空闲块！"
        print(f"✅ Prefill 测试通过！Req1 分配的块表: {req1.block_table}")
        
        # 2. 模拟 Decode 阶段生成 Token (产生第 7 个 token，不需要新块)
        manager.allocate_for_decode(req1)
        assert len(req1.block_table) == 2, "生成第 7 个 token 时不应该分配新块！"
        
        # 3. 产生第 8，再产生第 9 个 Token (跨过 Block 边界，需要新块)
        manager.allocate_for_decode(req1) # 长度变为 8
        manager.allocate_for_decode(req1) # 长度变为 9，触发新分配
        
        assert len(req1.block_table) == 3, "生成第 9 个 token 时应当分配了第 3 个新块！"
        assert len(manager.free_blocks) == 7, "池中应该剩下 7 个空闲块！"
        print(f"✅ Decode 动态分配测试通过！Req1 最新块表: {req1.block_table}")
        
        # 4. 模拟底层 PagedAttention 组装验证
        # 手动往第一块里写点假数据
        manager.physical_kv_cache[req1.block_table[0], 0, 0] = 999.0
        
        cache = manager.get_physical_cache(req1)
        assert cache.shape == (9, 64), f"拼装出来的连续 Cache 形状不对，应为 (9, 64)，实为 {cache.shape}"
        assert cache[0, 0] == 999.0, "数据未正确映射！"
        
        print("\n✅ All Tests Passed! 恭喜你，你已经掌握了 vLLM / PagedAttention 破局大模型推理瓶颈的核心奥秘！")
        
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
    except RuntimeError as e:
        print(f"❌ 运行时错误: {e}")
    except TypeError as e:
        print("代码可能未完成，导致变量为 NoneType。")
    except Exception as e:
        print(f"❌ 发生未知异常: {e}")

test_paged_attention_manager()

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

::: details 💡 点击查看官方解析与参考代码

PagedAttention 的灵感来源于操作系统中的虚拟内存管理，利用分页存储机制管理 KV Cache。其核心是在显存中按块分配缓存，从而能够显著提升显存利用率和并发度，是 vLLM 框架的基石。

```python
def paged_attention_sim(query, key_cache, value_cache, block_tables, context_lens, block_size):
    batch_size, num_heads, head_size = query.shape
    out = torch.zeros_like(query)
    
    for i in range(batch_size):
        ctx_len = context_lens[i].item()
        num_blocks = (ctx_len + block_size - 1) // block_size
        blocks = block_tables[i, :num_blocks]
        
        K_i = key_cache[blocks].reshape(-1, num_heads, head_size)[:ctx_len]
        V_i = value_cache[blocks].reshape(-1, num_heads, head_size)[:ctx_len]
        
        q_i = query[i].unsqueeze(0)
        
        scores = torch.matmul(q_i, K_i.transpose(-2, -1)) / math.sqrt(head_size)
        attn_weights = F.softmax(scores, dim=-1)
        
        out[i] = torch.matmul(attn_weights, V_i).squeeze(0)
        
    return out
```

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
