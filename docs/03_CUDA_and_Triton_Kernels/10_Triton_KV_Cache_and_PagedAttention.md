# 10. Triton KV Cache and PagedAttention | Triton 进阶：PagedAttention 的底层实现 (KV Cache 间接寻址)

**难度：** Hard | **标签：** `Triton`, `PagedAttention`, `vLLM` | **目标人群：** 核心 Infra 与算子开发

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/03_CUDA_and_Triton_Kernels/10_Triton_KV_Cache_and_PagedAttention.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


在 `02_PyTorch_Algorithms` 章节中，我们用纯 Python 模拟了 PagedAttention 分页管理 KV Cache 的核心思想。
但在真实的 GPU 硬件上，如何高效地实现**间接寻址 (Indirect Memory Access)**？如何根据 `Block Table` (块映射表) 动态去物理内存池 (Block Pool) 里提取不连续的 K 和 V 块，并在 SRAM 内完成 Online Softmax 归约？
本节我们将用 Triton 编写一个极简版的 PagedAttention 解码阶段 (Decoding) 前向内核。这也是 vLLM 推理引擎的最核心基石。


> **相关阅读**:
> 本节使用 Triton 实现了底层的极致显存与计算优化。
> 如果你对该算子的数学公式推导和纯 PyTorch 高层结构还不熟悉，建议先复习 PyTorch 篇：
>  [`../02_PyTorch_Algorithms/17_vLLM_PagedAttention.ipynb`](../02_PyTorch_Algorithms/17_vLLM_PagedAttention.md)

### Step 1: Paged KV Cache 的物理存储与逻辑映射

> **物理存储布局 (Physical Block Pool)：**
> KV Cache 被提前分配为一个巨大的连续张量，形状通常为 `[num_blocks, block_size, num_heads, head_dim]`。
> 每一个物理块 (Block) 包含了 `block_size` 个 Token 的特征。

> **逻辑映射 (Block Table)：**
> 每个请求 (Sequence) 拥有一张表 `block_tables[batch_idx, logical_block_idx]`，记录了该请求的第几个逻辑块映射到了物理内存池的哪个块索引。

> **Triton 解码内核的设计：**
> - **Grid 分配**：使用二维网格 `(batch_size, num_heads)`，一个 Program 处理一个序列的一个 Head 的查询 (Query)。
> - 由于是解码阶段，Query 只有一个 Token，形状为 `(head_dim, )`。
> - **循环归约**：在 Program 内部，根据上下文长度 `context_len` 循环遍历所有的逻辑块。
> - 在每次循环中，查表得到物理块索引，从池子中 Load 对应的 K 和 V，计算注意力分数，并在 SRAM 中维护 Online Softmax 状态。

### Step 2: 物理分页存储映射理论
在大规模解码期间，如果为每个请求预分配巨大的连续显存存放 KV Cache，会产生极大的内部显存碎片。vLLM 提出的 PagedAttention 将显存切成固定大小的物理块。逻辑序列通过映射表（Block Table）寻找物理块地址，极大地提升了并发承载能力。

### Step 3: PagedAttention 内核代码框架
在传统的 Flash Attention 内核中加入了一层间接寻址逻辑。在内层遍历序列长度的循环时，计算对应的逻辑块编号 `logical_block_id`。查表 `physical_block_id = tl.load(block_table_ptr + logical_block_id)`，用该物理块 ID 结合 Stride 拼凑出读取 KV 缓存的真实物理地址。

###  Step 4: 动手实战

**要求**：请补全下方 `paged_attention_decoding_kernel`。你需要实现最核心的查表寻址，并计算注意力点积。


```python
import torch
import triton
import triton.language as tl
import math
```


```python
@triton.jit
def paged_attention_decoding_kernel(
    out_ptr, q_ptr, k_cache_ptr, v_cache_ptr,
    block_tables_ptr, context_lens_ptr,
    sm_scale,
    stride_k_block, stride_k_seq, stride_k_head, stride_k_dim,
    stride_bt_batch, stride_bt_block,
    BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    # 1. 获取当前处理的 Batch 和 Head 索引
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # 2. 获取当前序列的上下文长度
    context_len = tl.load(context_lens_ptr + batch_idx)
    num_logical_blocks = tl.cdiv(context_len, BLOCK_SIZE)
    
    # 3. 加载 Query (解码阶段 Q 只有一个 Token)
    q_offset = batch_idx * tl.num_programs(1) * HEAD_DIM + head_idx * HEAD_DIM + tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + q_offset)
    
    # 初始化 Online Softmax 状态和输出累加器
    m_i = -float('inf')
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    
    # 4. 循环遍历当前序列的所有逻辑块
    for logical_block_idx in range(num_logical_blocks):
        # ==========================================
        # TODO 1: 查表获取物理块索引 (Indirect Memory Access)
        # 提示: 计算 block_tables 的偏移量，使用 tl.load 读取物理块索引
        # ==========================================
        # bt_offset = ???
        # physical_block_idx = ???
        
        # 计算当前逻辑块内每个 Token 的全局实际索引，用于构造 Mask
        start_token_idx = logical_block_idx * BLOCK_SIZE
        token_offsets = tl.arange(0, BLOCK_SIZE)
        physical_token_idx = start_token_idx + token_offsets
        mask = physical_token_idx < context_len
        
        # ==========================================
        # TODO 2: 计算 K 和 V 在物理池中的偏移量，并加载到 SRAM
        # 提示: 使用 physical_block_idx 和各维度的 stride 计算偏移量
        # K_cache 形状: [num_blocks, block_size, num_heads, head_dim]
        # ==========================================
        # k_offset = ???
        # k = ???
        # v = ???
        
        # ==========================================
        # TODO 3: 计算注意力分数 qk，并屏蔽 (Mask) 无效 Token
        # 提示: 计算 Q 和 K 的点积，乘以缩放因子，使用 mask 屏蔽无效位置
        # ==========================================
        # qk = ???
        
    
        # ==========================================
        # TODO 4: Online Softmax 归约
        # ==========================================
        # m_block = ???
        # m_new = ???
        # alpha = ???
        # p = ???
        # l_new = ???
        
        # ==========================================
        # TODO 5: 累加 V
        # ==========================================
        # acc = ???
        
        # m_i = ???
        # l_i = ???
        pass
        
    raise NotImplementedError("请完成 TODO 1-3")

def triton_paged_attention_decode(q, k_cache, v_cache, block_tables, context_lens, block_size):
    batch_size, num_heads, head_dim = q.shape
    out = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(head_dim)
    
    grid = (batch_size, num_heads)
    
    paged_attention_decoding_kernel[grid](
        out, q, k_cache, v_cache, block_tables, context_lens, sm_scale,
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        block_tables.stride(0), block_tables.stride(1),
        BLOCK_SIZE=block_size, HEAD_DIM=head_dim
    )
    return out

```


```python
# 测试你的实现
def test_paged_attention():
    if not torch.cuda.is_available():
        print("⏭️ 忽略测试：无 GPU。")
        return
        
    try:
        torch.manual_seed(42)
        batch_size, num_heads, head_dim = 2, 4, 64
        block_size = 16
        num_blocks = 20 # 物理内存池大小
        max_blocks_per_seq = 5
        
        # 构造输入
        q = torch.randn(batch_size, num_heads, head_dim, device='cuda', dtype=torch.float16)
        k_cache = torch.randn(num_blocks, block_size, num_heads, head_dim, device='cuda', dtype=torch.float16)
        v_cache = torch.randn(num_blocks, block_size, num_heads, head_dim, device='cuda', dtype=torch.float16)
        
        # 模拟分配的上下文长度和 Block Tables
        context_lens = torch.tensor([10, 25], device='cuda', dtype=torch.int32)
        block_tables = torch.tensor([
            [5, 0, 0, 0, 0],    # 序列0：用了 1 个物理块 (块 5)
            [12, 3, 0, 0, 0]    # 序列1：用了 2 个物理块 (块 12, 3)
        ], device='cuda', dtype=torch.int32)
        
        # 1. 构建 PyTorch 参考实现 (利用高级索引还原连续 KV Tensor)
        out_ref = torch.zeros_like(q)
        sm_scale = 1.0 / math.sqrt(head_dim)
        
        for b in range(batch_size):
            seq_len = context_lens[b].item()
            num_logical = math.ceil(seq_len / block_size)
            # 还原连续的 K 和 V
            k_seq = []
            v_seq = []
            for i in range(num_logical):
                phys_idx = block_tables[b, i].item()
                k_seq.append(k_cache[phys_idx])
                v_seq.append(v_cache[phys_idx])
                
            # [seq_len, num_heads, head_dim]
            k_cont = torch.cat(k_seq, dim=0)[:seq_len]
            v_cont = torch.cat(v_seq, dim=0)[:seq_len]
            
            # 对每个 Head 做 Attention
            for h in range(num_heads):
                q_h = q[b, h].unsqueeze(0) # [1, head_dim]
                k_h = k_cont[:, h, :] # [seq_len, head_dim]
                v_h = v_cont[:, h, :]
                
                attn = torch.softmax((q_h @ k_h.T) * sm_scale, dim=-1)
                out_ref[b, h] = (attn @ v_h).squeeze(0)
                
        # 2. Triton PagedAttention
        out_tri = triton_paged_attention_decode(q, k_cache, v_cache, block_tables, context_lens, block_size)
        
        # 3. 对比验证
        diff = torch.max(torch.abs(out_ref - out_tri))
        print(f"最大误差: {diff.item():.6e}")
        assert diff < 2e-3, "Triton PagedAttention 计算结果不正确！"
        
        print("✅ PagedAttention 间接寻址与 Online Softmax 验证通过。")
        
    
        print("\n--- 性能基准测试 (Benchmark) ---")
        # 放大测试规模模拟真实的 Decoding 并发
        batch_size, num_heads, head_dim = 32, 32, 128
        block_size = 16
        num_blocks = 2048 # 物理内存池大小 (足够大)
        
        q_l = torch.randn(batch_size, num_heads, head_dim, device='cuda', dtype=torch.float16)
        k_cache_l = torch.randn(num_blocks, block_size, num_heads, head_dim, device='cuda', dtype=torch.float16)
        v_cache_l = torch.randn(num_blocks, block_size, num_heads, head_dim, device='cuda', dtype=torch.float16)
        
        # 假设每个请求的 context 长度随机在 100 到 1000 之间
        context_lens_l = torch.randint(100, 1000, (batch_size,), device='cuda', dtype=torch.int32)
        
        # 构造 block_tables (简单模拟连续分配，真实环境是碎片化的)
        max_logical_blocks = math.ceil(1000 / block_size)
        block_tables_l = torch.zeros((batch_size, max_logical_blocks), device='cuda', dtype=torch.int32)
        alloc_idx = 0
        for b in range(batch_size):
            num_l = math.ceil(context_lens_l[b].item() / block_size)
            for i in range(num_l):
                block_tables_l[b, i] = alloc_idx
                alloc_idx += 1
                
        # 由于在 Python 里写循环还原 Paged KV 太慢了，我们只测 Triton
        # vLLM 的核心优势就在于 C++/Triton 级别的 Paged 间接寻址
        quantiles = [0.5, 0.2, 0.8]
        ms_tr, _, _ = triton.testing.do_bench(lambda: triton_paged_attention_decode(q_l, k_cache_l, v_cache_l, block_tables_l, context_lens_l, block_size), quantiles=quantiles)
        print(f"Triton PagedAttention Time (Batch={batch_size}, AvgSeqLen=500): {ms_tr:.4f} ms")
    except NotImplementedError:
        print("请先完成 TODO 代码！")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

test_paged_attention()

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
import math

@triton.jit
def paged_attention_decoding_kernel(
    out_ptr, q_ptr, k_cache_ptr, v_cache_ptr,
    block_tables_ptr, context_lens_ptr,
    sm_scale,
    stride_k_block, stride_k_seq, stride_k_head, stride_k_dim,
    stride_bt_batch, stride_bt_block,
    BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    # 1. 获取当前处理的 Batch 和 Head 索引
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # 2. 获取当前序列的上下文长度
    context_len = tl.load(context_lens_ptr + batch_idx)
    num_logical_blocks = tl.cdiv(context_len, BLOCK_SIZE)
    
    # 3. 加载 Query (解码阶段 Q 只有一个 Token)
    q_offset = batch_idx * tl.num_programs(1) * HEAD_DIM + head_idx * HEAD_DIM + tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + q_offset)
    
    # 初始化 Online Softmax 状态和输出累加器
    m_i = -float('inf')
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    
    # 4. 循环遍历当前序列的所有逻辑块
    for logical_block_idx in range(num_logical_blocks):
        # TODO 1: 查表获取物理块索引
        bt_offset = batch_idx * stride_bt_batch + logical_block_idx * stride_bt_block
        physical_block_idx = tl.load(block_tables_ptr + bt_offset)
        
        # 计算当前逻辑块内每个 Token 的全局实际索引，用于构造 Mask
        start_token_idx = logical_block_idx * BLOCK_SIZE
        token_offsets = tl.arange(0, BLOCK_SIZE)
        physical_token_idx = start_token_idx + token_offsets
        mask = physical_token_idx < context_len
        
        # TODO 2: 计算 K 和 V 在物理池中的偏移量，并加载到 SRAM
        k_offset = physical_block_idx * stride_k_block + token_offsets[:, None] * stride_k_seq + head_idx * stride_k_head + tl.arange(0, HEAD_DIM)[None, :]
        
        k = tl.load(k_cache_ptr + k_offset, mask=mask[:, None], other=0.0)
        v = tl.load(v_cache_ptr + k_offset, mask=mask[:, None], other=0.0)
        
        # TODO 3: 计算注意力分数 qk，并屏蔽 (Mask) 无效 Token
        qk = tl.sum(q[None, :] * k, axis=1) * sm_scale
        qk = tl.where(mask, qk, -float('inf'))
        
        # TODO 4:Online Softmax 归约
        m_block = tl.max(qk, axis=0)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)
        l_new = l_i * alpha + tl.sum(p, axis=0)
        
        # TODO 5:累加 V
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        
        m_i = m_new
        l_i = l_new
        
    # 5. 最终归一化并写回
    acc = acc / l_i
    tl.store(out_ptr + q_offset, acc.to(out_ptr.dtype.element_ty))

def triton_paged_attention_decode(q, k_cache, v_cache, block_tables, context_lens, block_size):
    batch_size, num_heads, head_dim = q.shape
    out = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(head_dim)
    
    grid = (batch_size, num_heads)
    
    paged_attention_decoding_kernel[grid](
        out, q, k_cache, v_cache, block_tables, context_lens, sm_scale,
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        block_tables.stride(0), block_tables.stride(1),
        BLOCK_SIZE=block_size, HEAD_DIM=head_dim
    )
    return out

```

### 解析

**1. TODO 1: 查表获取物理块索引 (Indirect Memory Access)**
- **实现方式**：
  ```python
  bt_offset = batch_idx * stride_bt_batch + logical_block_idx * stride_bt_block
  physical_block_idx = tl.load(block_tables_ptr + bt_offset)
  ```
- **关键点**：这是 PagedAttention 的核心创新，通过间接寻址实现显存碎片化管理
- **技术细节**：
  - `block_tables` 是一个二维表，形状为 `[batch_size, max_logical_blocks]`
  - 每个序列有自己的映射表，记录逻辑块到物理块的映射关系
  - 使用 `stride_bt_batch` 定位到当前序列的映射表
  - 使用 `stride_bt_block` 定位到当前逻辑块的映射项
  - `tl.load` 读取物理块索引，这个索引指向 KV Cache 物理内存池中的实际位置

**2. TODO 2: 计算 K 和 V 在物理池中的偏移量，并加载到 SRAM**
- **实现方式**：
  ```python
  k_offset = physical_block_idx * stride_k_block + token_offsets[:, None] * stride_k_seq + head_idx * stride_k_head + tl.arange(0, HEAD_DIM)[None, :]
  k = tl.load(k_cache_ptr + k_offset, mask=mask[:, None], other=0.0)
  v = tl.load(v_cache_ptr + k_offset, mask=mask[:, None], other=0.0)
  ```
- **关键点**：使用物理块索引和多维 stride 精确定位 KV Cache 中的数据
- **技术细节**：
  - KV Cache 形状：`[num_blocks, block_size, num_heads, head_dim]`
  - `physical_block_idx * stride_k_block`：定位到物理块的起始位置
  - `token_offsets[:, None] * stride_k_seq`：定位到块内的 Token 位置（二维索引的行维度）
  - `head_idx * stride_k_head`：定位到当前 Head
  - `tl.arange(0, HEAD_DIM)[None, :]`：定位到特征维度（二维索引的列维度）
  - 使用 `mask[:, None]` 保护边界，防止读取超出 `context_len` 的无效数据
  - K 和 V 使用相同的偏移量，因为它们的存储布局一致

**3. TODO 3: 计算注意力分数 qk，并屏蔽 (Mask) 无效 Token**
- **实现方式**：
  ```python
  qk = tl.sum(q[None, :] * k, axis=1) * sm_scale
  qk = tl.where(mask, qk, -float('inf'))
  ```
- **关键点**：计算 Q 和 K 的点积，并使用 mask 屏蔽无效位置
- **技术细节**：
  - `q` 形状：`(HEAD_DIM,)`，是解码阶段的单个 Token Query
  - `k` 形状：`(BLOCK_SIZE, HEAD_DIM)`，是当前物理块的所有 Key
  - `q[None, :]` 扩展为 `(1, HEAD_DIM)`，与 `k` 广播相乘
  - `tl.sum(..., axis=1)` 对特征维度求和，得到 `(BLOCK_SIZE,)` 的注意力分数
  - `sm_scale = 1.0 / sqrt(head_dim)` 是标准的缩放因子，防止 Softmax 梯度消失
  - `tl.where(mask, qk, -float('inf'))` 将超出 `context_len` 的位置设为负无穷，确保 Softmax 后权重为 0

**4. TODO 4: Online Softmax 归约**
- **实现方式**（答案区代码中的注释部分）：
  ```python
  m_block = tl.max(qk, axis=0)
  m_new = tl.maximum(m_i, m_block)
  alpha = tl.exp(m_i - m_new)
  p = tl.exp(qk - m_new)
  l_new = l_i * alpha + tl.sum(p, axis=0)
  ```
- **关键点**：在循环中维护 Softmax 的最大值和归一化因子，避免数值溢出
- **技术细节**：
  - `m_i` 和 `l_i` 是累积的最大值和归一化因子
  - `m_block` 是当前块的最大注意力分数
  - `m_new` 是全局最大值，用于数值稳定的 Softmax
  - `alpha = exp(m_i - m_new)` 是修正因子，用于更新之前累积的结果
  - `p = exp(qk - m_new)` 是当前块的 Softmax 分子（未归一化）
  - `l_new` 是更新后的归一化因子

**5. TODO 5: 累加 V 并更新状态**
- **实现方式**：
  ```python
  acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
  m_i = m_new
  l_i = l_new
  ```
- **关键点**：使用修正因子更新累积的输出，并更新 Softmax 状态
- **技术细节**：
  - `acc * alpha` 修正之前累积的输出（因为最大值变化了）
  - `p[:, None] * v` 计算加权的 V，形状为 `(BLOCK_SIZE, HEAD_DIM)`
  - `tl.sum(..., axis=0)` 对 Token 维度求和，得到 `(HEAD_DIM,)` 的输出贡献
  - 更新 `m_i` 和 `l_i` 为新的状态，用于下一个块的计算

**工程优化要点**

- **显存碎片化管理**：PagedAttention 将 KV Cache 切分成固定大小的物理块，通过映射表实现逻辑地址到物理地址的转换，消除了传统连续分配导致的显存碎片
- **间接寻址**：通过 `block_tables` 查表实现动态路由，允许不同序列的 KV Cache 分散存储在物理内存池的任意位置
- **Online Softmax**：在循环中维护 Softmax 状态，避免存储完整的注意力矩阵，节省显存并提高计算效率
- **SRAM 内计算**：K、V 和注意力分数都在 SRAM 中计算，最小化 HBM 访问次数
- **Mask 保护**：使用 `mask` 确保只处理有效的 Token，防止越界访问和计算错误
- **数值稳定性**：使用 Safe Softmax（减去最大值）防止指数运算溢出
- **并发承载能力**：vLLM 论文表明，PagedAttention 可以将并发请求数提升 3 倍以上，因为消除了显存碎片导致的浪费
- **解码阶段优化**：本实现针对解码阶段（Query 只有一个 Token），避免了 Prefill 阶段的复杂性
- **工业应用**：该算子是 vLLM 推理引擎的核心组件，广泛应用于大规模 LLM 推理服务