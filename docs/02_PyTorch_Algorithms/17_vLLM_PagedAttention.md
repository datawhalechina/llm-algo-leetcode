# 17. vLLM PagedAttention | 经典推理框架: 模拟 Continuous Batching 与 PagedAttention

**难度：** Hard | **标签：** `推理架构`, `vLLM` | **目标人群：** 核心 Infra 与算子开发

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/17_vLLM_PagedAttention.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本节我们将揭秘工业界大模型推理框架（如 **vLLM**）的两大杀手锏技术：**Continuous Batching (连续批处理/动态批处理)** 和 **PagedAttention (分页注意力池)**。
这是目前算法面经里含金量最高，但资料最匮乏的部分！

> **相关阅读**:
> 本节使用纯 PyTorch 实现了算法逻辑与数学推导。
> 如果你想学习工业界如何打破该算子的 Memory Bound (访存瓶颈)，请前往 Triton 篇：
>  [`../03_CUDA_and_Triton_Kernels/10_Triton_KV_Cache_and_PagedAttention.ipynb`](../03_CUDA_and_Triton_Kernels/10_Triton_KV_Cache_and_PagedAttention.md)


### Step 1: 核心思想与痛点

> **痛点 1：Static Batching 的低效**
> 在传统的 PyTorch 推理中，Batch 内的不同请求长度不一。如果 Request A 生成了 10 个 Token 就结束了，而 Request B 需要生成 100 个，那么 A 生成完后 GPU 只能干等 B（即用 Padding 填充计算），导致算力非常浪费。
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
```


```python
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
        
        print("\n✅ All Tests Passed! PagedAttention 内存管理逻辑验证通过。")
        
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
## 参考代码与解析

### 代码

```python
class Request:
    def __init__(self, request_id: int, prompt_len: int):
        self.request_id = request_id
        self.seq_len = prompt_len
        self.block_table: List[int] = []

class KVCacheManager:
    def __init__(self, num_blocks: int, block_size: int, head_dim: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.head_dim = head_dim
        
        # TODO 1: 模拟预分配一块大显存池
        self.physical_kv_cache = torch.zeros(num_blocks, block_size, head_dim)
        
        # 跟踪哪些物理块被占用了
        self.free_blocks: List[int] = list(range(num_blocks))

    def allocate_for_prefill(self, req: Request):
        """
        请求刚进来时 (Prefill阶段)，为它的 Prompt 长度分配所需的全部 Block
        """
        # TODO 2: 计算需要的 block 数量（向上取整）
        needed_blocks = (req.seq_len + self.block_size - 1) // self.block_size
        
        # TODO 3: 从 free_blocks 中弹出对应数量的 block 索引
        if len(self.free_blocks) < needed_blocks:
            raise RuntimeError("OOM")
        
        for _ in range(needed_blocks):
            block_id = self.free_blocks.pop(0)
            req.block_table.append(block_id)

    def allocate_for_decode(self, req: Request):
        """
        自回归生成时 (Decode阶段)，检查序列长度。
        如果当前最后一个 Block 满了，则按需分配 1 个新 Block。
        """
        req.seq_len += 1
        
        # TODO 4: 判断是否需要新的 Block
        is_new_block_needed = (req.seq_len % self.block_size) == 1
        
        if is_new_block_needed:
            if not self.free_blocks:
                raise RuntimeError("OOM")
            block_id = self.free_blocks.pop(0)
            req.block_table.append(block_id)

    def get_physical_cache(self, req: Request) -> torch.Tensor:
        """
        根据块表，把不连续的物理块"拼凑"成逻辑上连续的 KV Cache
        """
        # TODO 5: 根据 req.block_table 的索引，从物理池中提取对应的块
        blocks = [self.physical_kv_cache[block_id] for block_id in req.block_table]
        cat_blocks = torch.cat(blocks, dim=0)
        
        # 只截取真实 seq_len 长度返回
        return cat_blocks[:req.seq_len]
```

### 解析

**1. TODO 1: 初始化物理块池**
- **实现方式**：`self.physical_kv_cache = torch.zeros(num_blocks, block_size, head_dim)`
- **关键点**：预分配固定大小的显存池，避免动态分配的碎片化
- **技术细节**：形状为 `[num_blocks, block_size, head_dim]`，每个 block 存储固定数量的 token

**2. TODO 2: 计算 Prefill 阶段需要的 block 数量**
- **实现方式**：`needed_blocks = (req.seq_len + self.block_size - 1) // self.block_size`
- **关键点**：向上取整，确保能容纳所有 token
- **技术细节**：使用 `(a + b - 1) // b` 实现向上取整，避免浮点运算

**3. TODO 3: 分配物理块**
- **实现方式**：从 `free_blocks` 中弹出 `needed_blocks` 个索引，追加到 `req.block_table`
- **关键点**：如果空闲块不足，抛出 OOM 异常
- **技术细节**：使用 `pop(0)` 从队列头部取出，模拟 FIFO 分配策略

**4. TODO 4: Decode 阶段按需分配**
- **实现方式**：`is_new_block_needed = (req.seq_len % self.block_size) == 1`
- **关键点**：只有当序列长度刚好跨越 block 边界时才分配新块
- **技术细节**：`seq_len % block_size == 1` 表示刚进入新块的第一个位置

**5. TODO 5: 拼装物理块**
- **实现方式**：`blocks = [self.physical_kv_cache[block_id] for block_id in req.block_table]`，`cat_blocks = torch.cat(blocks, dim=0)`
- **关键点**：根据块表索引，将不连续的物理块拼接成逻辑上连续的 KV Cache
- **技术细节**：最后截取 `[:req.seq_len]` 因为最后一个块可能未填满

**工程优化要点**
- **显存利用率**：PagedAttention 将显存利用率从 40% 提升到 90%+，减少内部碎片
- **动态批处理**：配合 Continuous Batching，实现请求级别的动态调度
- **块大小权衡**：block_size 太小增加管理开销，太大增加内部碎片，通常选择 16-32
- **共享机制**：vLLM 支持多个请求共享相同的 Prompt 块（如系统提示词），进一步节省显存
- **工业实现**：真实的 vLLM 使用 CUDA kernel 实现 PagedAttention，支持多头注意力和批处理