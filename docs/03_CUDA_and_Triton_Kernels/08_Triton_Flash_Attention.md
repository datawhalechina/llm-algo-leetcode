# 08 Triton Flash Attention

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/03_CUDA_and_Triton_Kernels/08_Triton_Flash_Attention.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 08. Triton Flash Attention：编写真正的 Flash Attention 前向算子

**难度：** Hard | **标签：** `Triton`, `FlashAttention`, `Memory Bound` | **目标人群：** 核心 Infra 与算子开发

在 `02_PyTorch_Algorithms/13_FlashAttention_Sim` 和 `03_Triton_Kernels/06_Triton_Fused_Softmax` 中，我们已经完全掌握了 Flash Attention 的两大数学核心：**分块计算 (Tiling)** 和 **在线安全 Softmax 归约 (Online Safe Softmax)**。
本节我们将把这两者结合起来，利用 Triton 在 SRAM 中的极速读写，编写一个真正的、可运行在 GPU 上的 Flash Attention 前向计算内核。这是大模型推理与训练提速的基石算子。


> **相关阅读**:  
> 本节使用 Triton 实现了底层的显存与计算优化。
> 如果你对该算子的数学公式推导和纯 PyTorch 高层结构还不熟悉，建议先复习 PyTorch 篇：
> [`../02_PyTorch_Algorithms/13_FlashAttention_Sim.ipynb`](../02_PyTorch_Algorithms/13_FlashAttention_Sim.ipynb)

### Step 1: Flash Attention 内核的执行逻辑

> **任务分配 (Grid)：**
> 假设有 `batch` 个序列，每个序列有 `n_heads` 个注意力头。
> 我们的 Grid 划分为 `(batch * n_heads, num_blocks_q)`。
> 也就是说，**每个 Triton Program 负责计算一个序列、一个 Head 中的一小块 Q** 的最终输出。

> **SRAM 内的循环计算过程 (对于当前分配到的 Q Block)：**
> 1. 初始化累加器 `acc` (用于存最终的 O) 和 `m_i`, `l_i` (用于 Online Softmax)。
> 2. 将这块 Q 从 HBM 加载到 SRAM。
> 3. **内层循环遍历 K 和 V 的所有 Block：**
>    - 加载当前块的 K 和 V 到 SRAM。
>    - 计算注意力分数 $S = Q \times K^T$ (SRAM 内的矩阵乘)。
>    - 执行 Online Softmax 更新逻辑，计算出修正系数。
>    - 修正过去的累加器 `acc`，并累加新的 $P \times V$。
> 4. 循环结束后，将最终归一化的结果 `acc / l_i` 写回 HBM。


### Step 2: Online Softmax 与 Tiling 深度剖析
为了能够在 SRAM 中处理长序列点积，Flash Attention 发明了 Tiling 与 Online Softmax。在内层循环迭代块 $j$ 时，局部最大值 $m_{new} = \max(m_{old}, m_j)$ 会发生变化。我们需要乘上修正系数 $e^{m_{old} - m_{new}}$ 来弥补之前循环计算结果造成的偏差，使其在数学上严格等价于完整的全局 Softmax。


### Step 3: 内核基本框架
建立 1D Grid 架构，每个程序固定加载一个 $Q$ 块和累加器。遍历 $K$ 和 $V$ 的序列长度：加载 $K_j$ 计算局部点积，提取局部最大值，修正历史累加的 $L$ 和 $Acc$ 变量，累加当前的 $P_j \times V_j$，然后继续读取下一个块。最后归一化并写入 HBM。


###  Step 4: 动手实战

**要求**：请补全下方 `flash_attn_fwd_kernel`，补全最核心的 SRAM 内部矩阵乘法与 Softmax 状态更新逻辑。


```python
import torch
import triton
import triton.language as tl

@triton.jit
def flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, sm_scale,
    Out_ptr,
    seqlen_q, seqlen_k, head_dim,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # 1. 确定当前 Program 的 ID 和处理的 Q 块范围
    # 为了简化，我们只处理单 Batch 单 Head 的情况，所以 grid 只有一个维度：处理 Q 的哪个分块
    start_m = tl.program_id(0)
    
    # 初始化指向 Q 块的指针偏移
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)
    
    # 我们假设 head_dim 恰好等于 BLOCK_D，且 seqlen_q 是 BLOCK_M 的倍数 (省去边界检查的麻烦)
    q_ptrs = Q_ptr + (offs_m[:, None] * head_dim + offs_d[None, :])
    q = tl.load(q_ptrs)
    
    # 2. 初始化累加器和 Online Softmax 的状态
    acc = tl.zeros((BLOCK_M, head_dim), dtype=tl.float32)
    m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float('inf')
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    # 3. 内层循环：遍历 K 和 V 的所有块
    # 计算需要循环多少次
    num_n_blocks = tl.cdiv(seqlen_k, BLOCK_N)
    
    for start_n in range(0, num_n_blocks):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        
        # 加载 K 块和 V 块
        # 注意 K 需要转置，即原本形状是 (seqlen_k, head_dim)，取出来的块是 (BLOCK_N, head_dim)
        # 转置后可以直接和 Q 矩阵相乘
        k_ptrs = K_ptr + (offs_n[:, None] * head_dim + offs_d[None, :])
        v_ptrs = V_ptr + (offs_n[:, None] * head_dim + offs_d[None, :])
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        
        # ==========================================
        # TODO 1: 计算注意力分数 S = Q @ K^T * scale
        # 提示: 使用 tl.dot(A, B)，注意 K 的形状是 (BLOCK_N, head_dim)，所以需要转置 tl.trans(k)
        # ==========================================
        # qk = ???
        qk = tl.dot(q, tl.trans(k))
        qk *= sm_scale
        
        # ==========================================
        # TODO 2: 计算 Online Softmax 的新状态
        # 1. m_block = 当前块每行的最大值
        # 2. m_new = max(旧的 m_i, m_block)
        # ==========================================
        m_block = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_block)
        
        # ==========================================
        # TODO 3: 计算指数并更新 l_i 和 p
        # 1. p = tl.exp(qk - m_new[:, None])
        # 2. 修正过去的 l_i：l_new = l_i * exp(m_i - m_new) + 当前 p 的行和
        # ==========================================
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_new = l_i * alpha + tl.sum(p, axis=1)
        
        # ==========================================
        # TODO 4: 修正过去的输出结果 acc，并累加新的 p @ v
        # 注意：这里需要先把 acc 也乘以修正系数 alpha[:, None]
        # 然后使用 tl.dot 累加 p 和 v 的乘积
        # ==========================================
        # acc = ???
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        
        # 更新状态准备进入下一块的循环
        m_i = m_new
        l_i = l_new
        
    # 4. 循环结束，除以全局的 l_i 完成最终归一化
    acc = acc / l_i[:, None]
    
    # 5. 写回显存
    out_ptrs = Out_ptr + (offs_m[:, None] * head_dim + offs_d[None, :])
    tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty))

def triton_flash_attention(q, k, v, sm_scale):
    # 限制条件简化：仅支持 2D 张量，且能被 BLOCK 整除
    seqlen_q, head_dim = q.shape
    seqlen_k, _ = k.shape
    
    out = torch.empty_like(q)
    
    # 配置分块大小
    BLOCK_M = 128
    BLOCK_N = 128
    
    grid = (triton.cdiv(seqlen_q, BLOCK_M), )
    
    flash_attn_fwd_kernel[grid](
        q, k, v, sm_scale, out,
        seqlen_q, seqlen_k, head_dim,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
    return out

```

```python
# 测试你的实现
import math

def test_triton_flash_attention():
    if not torch.cuda.is_available():
        print("⏭️ 忽略测试：无 GPU。")
        return
        
    try:
        torch.manual_seed(42)
        # 构造规则形状的数据
        seqlen_q = 256
        seqlen_k = 256
        head_dim = 64
        
        q = torch.randn(seqlen_q, head_dim, device='cuda', dtype=torch.float16)
        k = torch.randn(seqlen_k, head_dim, device='cuda', dtype=torch.float16)
        v = torch.randn(seqlen_k, head_dim, device='cuda', dtype=torch.float16)
        sm_scale = 1.0 / math.sqrt(head_dim)
        
        # 1. PyTorch 原生计算基准
        attn = (q @ k.transpose(-2, -1)) * sm_scale
        attn = torch.nn.functional.softmax(attn, dim=-1)
        out_ref = attn @ v
        
        # 2. Triton 算子计算
        out_tri = triton_flash_attention(q, k, v, sm_scale)
        
        # 3. 验证误差
        diff = torch.max(torch.abs(out_ref - out_tri))
        print(f"最大误差: {diff.item():.6e}")
        assert diff < 1e-3, "Triton Flash Attention 结果不正确！"
        
        print("✅ 太震撼了！你成功用 Triton 编写了最核心的 Flash Attention 前向计算内核！")
        print("🔥 能够处理 SRAM 中块与块之间的局部最大值归约更新，标志着你已正式踏入高阶算子开发的大门。")
        
    
        print("\n--- ⚡ 性能基准测试 (Benchmark) ---")
        # 典型的 LLM 推理/训练尺寸
        seqlen_q = 4096
        seqlen_k = 4096
        head_dim = 128
        q_l = torch.randn(seqlen_q, head_dim, device='cuda', dtype=torch.float16)
        k_l = torch.randn(seqlen_k, head_dim, device='cuda', dtype=torch.float16)
        v_l = torch.randn(seqlen_k, head_dim, device='cuda', dtype=torch.float16)
        sm_scale = 1.0 / math.sqrt(head_dim)
        
        def torch_attn(q, k, v):
            attn = (q @ k.transpose(-2, -1)) * sm_scale
            attn = torch.nn.functional.softmax(attn, dim=-1)
            return attn @ v
            
        quantiles = [0.5, 0.2, 0.8]
        # 注意: 如果 seqlen = 4096, Pytorch 需要分配 4096x4096x2 bytes = 32MB 的中间矩阵
        # Flash Attention 只需要 O(N) 的显存，并且速度极快
        try:
            ms_pt, _, _ = triton.testing.do_bench(lambda: torch_attn(q_l, k_l, v_l), quantiles=quantiles)
            ms_tr, _, _ = triton.testing.do_bench(lambda: triton_flash_attention(q_l, k_l, v_l, sm_scale), quantiles=quantiles)
            print(f"PyTorch Time (O(N^2) memory): {ms_pt:.4f} ms")
            print(f"Triton Time (O(N) memory):    {ms_tr:.4f} ms")
            print(f"Speedup:                      {ms_pt / ms_tr:.2f}x")
        except torch.cuda.OutOfMemoryError:
            print("PyTorch OOM! 序列太长导致显存溢出，但 Triton Flash Attention 依然可以运行。")
    except NotImplementedError:
        print("请先完成 TODO 代码！")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

test_triton_flash_attention()

```

::: details 💡 点击查看官方解析与参考代码

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

### 💡 参考解答：Triton Flash Attention 前向算子

在这个实现中，我们完成了核心的 Flash Attention 前向计算内核。注意以下几个关键点：
1. **QK 点积与缩放**：使用 `tl.dot(q, tl.trans(k))` 计算注意力分数，并乘以 `sm_scale`。
2. **局部最大值更新**：计算当前块的最大值 `m_block = tl.max(qk, axis=1)`，然后用 `tl.maximum(m_i, m_block)` 更新全局最大值 `m_new`。
3. **分母与概率更新**：根据新的最大值计算概率 `p = tl.exp(qk - m_new[:, None])`。修正系数 `alpha = tl.exp(m_i - m_new)` 用于缩放之前的累加值，然后更新归一化分母 `l_new = l_i * alpha + tl.sum(p, axis=1)`。
4. **输出累加**：先用 `alpha` 修正之前的累加器结果 `acc = acc * alpha[:, None]`，然后再累加当前块的加权值 `acc += tl.dot(p.to(v.dtype), v)`。最后别忘了循环结束后除以全局的 `l_i` 进行归一化。

```python
import torch
import triton
import triton.language as tl

@triton.jit
def flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, sm_scale,
    Out_ptr,
    seqlen_q, seqlen_k, head_dim,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # 1. 确定当前 Program 的 ID 和处理的 Q 块范围
    start_m = tl.program_id(0)
    
    # 初始化指向 Q 块的指针偏移
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)
    
    q_ptrs = Q_ptr + (offs_m[:, None] * head_dim + offs_d[None, :])
    q = tl.load(q_ptrs)
    
    # 2. 初始化累加器和 Online Softmax 的状态
    acc = tl.zeros((BLOCK_M, head_dim), dtype=tl.float32)
    m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float('inf')
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    # 3. 内层循环：遍历 K 和 V 的所有块
    num_n_blocks = tl.cdiv(seqlen_k, BLOCK_N)
    
    for start_n in range(0, num_n_blocks):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        
        k_ptrs = K_ptr + (offs_n[:, None] * head_dim + offs_d[None, :])
        v_ptrs = V_ptr + (offs_n[:, None] * head_dim + offs_d[None, :])
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        
        # TODO 1: 计算注意力分数 S = Q @ K^T * scale
        qk = tl.dot(q, tl.trans(k))
        qk *= sm_scale
        
        # TODO 2: 计算 Online Softmax 的新状态
        m_block = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_block)
        
        # TODO 3: 计算指数并更新 l_i 和 p
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_new = l_i * alpha + tl.sum(p, axis=1)
        
        # TODO 4: 修正过去的输出结果 acc，并累加新的 p @ v
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        
        # 更新状态准备进入下一块的循环
        m_i = m_new
        l_i = l_new
        
    # 4. 循环结束，除以全局的 l_i 完成最终归一化
    acc = acc / l_i[:, None]
    
    # 5. 写回显存
    out_ptrs = Out_ptr + (offs_m[:, None] * head_dim + offs_d[None, :])
    tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty))

def triton_flash_attention(q, k, v, sm_scale):
    seqlen_q, head_dim = q.shape
    seqlen_k, _ = k.shape
    
    out = torch.empty_like(q)
    
    BLOCK_M = 128
    BLOCK_N = 128
    
    grid = (triton.cdiv(seqlen_q, BLOCK_M), )
    
    flash_attn_fwd_kernel[grid](
        q, k, v, sm_scale, out,
        seqlen_q, seqlen_k, head_dim,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
    return out
```

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
