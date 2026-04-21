# 15. FlashAttention Sim | 深入理解 FlashAttention：分块计算与 Online Softmax

**难度：** Hard | **标签：** `FlashAttention`, `Memory Bound`, `PyTorch` | **目标人群：** 核心 Infra 与算子开发

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/15_FlashAttention_Sim.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


在标准的自注意力 (Self-Attention) 机制中，时间复杂度和空间复杂度都是序列长度 $O(N^2)$。当序列变得极长（如 128k, 1M Token），庞大的 $N \times N$ 注意力分数矩阵 $(QK^T)$ 会直接导致显存溢出 (OOM)。

FlashAttention (Tri Dao et al., 2022) 带来了一场革命。它的核心思想不是减少计算量 (FLOPs 甚至略有增加)，而是通过 **Tiling (分块计算)** 和 **Online Softmax** 完全避免了将大规模的 $N \times N$ 中间结果写回到缓慢的 GPU 显存 (HBM) 中，从而将空间复杂度降为 $O(N)$，并大幅提升了实际运行速度。


本节我们将不用 Triton/CUDA，而是仅用 PyTorch 的循环，在数学逻辑上 1:1 模拟 FlashAttention 的前向计算过程，帮助你完全弄懂 Online Softmax 是如何工作的。


> **相关阅读**:
> 本节使用纯 PyTorch 实现了算法逻辑与数学推导。
> 如果你想学习工业界如何打破该算子的 Memory Bound (访存瓶颈)，请前往 Triton 篇：
>  [`../03_CUDA_and_Triton_Kernels/08_Triton_Flash_Attention.ipynb`](../03_CUDA_and_Triton_Kernels/08_Triton_Flash_Attention.md)

### Step 1: 核心理论与 Online Softmax

> **标准 Softmax 的痛点：**
> 1. 求每一行的最大值 $m = \max(x)$ (防溢出)。
> 2. 求每一行的指数和 $l = \sum e^{x - m}$。
> 3. 求最终结果 $y_i = \frac{e^{x_i - m}}{l}$。
> 这意味着在算出所有 $x$ 之前，你无法算出 $m$ 和 $l$，因此必须把所有的 $x$ 先存下来。在 Attention 中，$x$ 就是那个大规模的 $S = QK^T$ 矩阵！

> **Online Softmax 的机制：**
> 我们可以在只看到**部分数据**时，持续更新一个局部的最大值 $m_{new}$ 和局部的指数和 $l_{new}$。
> 当新来一个分块 (Block) 时，如果新块的最大值更大，我们可以用一个数学技巧，把之前算好的部分“修正”过来，而不需要重新算前面的块！
> 
> **更新公式：**
> - 新的局部最大值：$m_{new} = \max(m_{old}, m_{block})$
> - 修正旧的指数和：$l_{new} = l_{old} \cdot e^{m_{old} - m_{new}} + l_{block} \cdot e^{m_{block} - m_{new}}$
> - 修正旧的输出结果（乘积累加）：$O_{new} = O_{old} \cdot \frac{l_{old} \cdot e^{m_{old} - m_{new}}}{l_{new}} + \frac{e^{S_{block} - m_{new}} \cdot V_{block}}{l_{new}}$

### Step 2: Flash Attention 分块机制原理
由于标准的 Attention 需要 $O(N^2)$ 的显存来存储巨大的 Attention Score 矩阵 $S = QK^T$，当上下文变长时必定 OOM。Flash Attention 巧妙地在序列维度上对 Q, K, V 进行分块（Tiling）。通过外层循环遍历 Q 块，内层循环遍历 K 和 V 块，我们可以在保持数学上完全等价的前提下，将显存消耗降到 $O(N)$。

### Step 3: 代码实现框架
核心是三层嵌套的循环（或者是二维 Grid）。对于当前处理的一小块 $Q_{block}$，在内层遍历所有 $K_{block}$ 时，动态地更新局部最大值 $m$ 和局部指数和 $l$。这是在纯 PyTorch 中使用 `for` 循环来模拟底层 C++ 内存块调度的绝佳方式。

###  Step 4: 动手实战

**要求**：请补全下方 `flash_attention_forward_sim` 函数，实现分块 (Tiling) 的 QKV 乘法以及 Online Softmax 逻辑。


```python
import torch
import math
```


```python
def flash_attention_forward_sim(q, k, v, block_size=2):
    """
    纯 PyTorch 模拟 FlashAttention 前向传播。
    假设没有 Batch 和 Head 维度，q, k, v 的形状都是 (seq_len, dim)。
    """
    seq_len, dim = q.shape
    
    # 初始化输出 O，全局最大值 m，全局指数和 l
    out = torch.zeros((seq_len, dim), device=q.device)
    m = torch.full((seq_len, 1), -float('inf'), device=q.device)
    l = torch.zeros((seq_len, 1), device=q.device)
    
    scale = 1.0 / math.sqrt(dim)
    
    # 外层循环：遍历 Q 的分块 (通常按行划分目标输出)
    for i in range(0, seq_len, block_size):
        q_block = q[i:i+block_size] * scale
        
        # 内层循环：遍历 K, V 的分块 (计算这一块 Q 与所有 K 的注意力)
        for j in range(0, seq_len, block_size):
            k_block = k[j:j+block_size]
            v_block = v[j:j+block_size]
            
            # ==========================================
            # TODO 1: 计算当前块的未归一化分数 S_ij
            # ==========================================
            # S_ij = ???
            
            # ==========================================
            # TODO 2: 计算当前块的局部最大值 m_block，并求出新的全局最大值 m_new
            # ==========================================
            # m_block = ???
            # m_new = ???
            
            # ==========================================
            # TODO 3: 计算 P_ij = exp(S_ij - m_new)
            # ==========================================
            # P_ij = ???
            
            # ==========================================
            # TODO 4: 计算当前块的局部指数和 l_block，并更新全局指数和 l_new
            # ==========================================
            # l_block = ???
            # l_new = ???
            
            # ==========================================
            # TODO 5: 更新输出 O_i
            # ==========================================
            # out[i:i+block_size] = ???
            
            # 更新保存的全局状态
            # m[i:i+block_size] = m_new
            # l[i:i+block_size] = l_new
            pass
            
    return out

```


```python
# 测试你的实现
def test_flash_attention_sim():
    try:
        torch.manual_seed(42)
        seq_len, dim = 8, 4
        q = torch.randn(seq_len, dim)
        k = torch.randn(seq_len, dim)
        v = torch.randn(seq_len, dim)
        
        # 1. 计算标准 Attention 作为 Ground Truth
        scale = 1.0 / math.sqrt(dim)
        scores = (q @ k.transpose(-2, -1)) * scale
        attn = torch.nn.functional.softmax(scores, dim=-1)
        out_ref = attn @ v
        
        # 2. 计算模拟的 FlashAttention (使用分块 block_size=2)
        out_sim = flash_attention_forward_sim(q, k, v, block_size=2)
        
        # 3. 验证结果
        diff = torch.max(torch.abs(out_ref - out_sim))
        print(f"最大误差: {diff.item():.6e}")
        assert diff < 1e-5, "计算结果与标准 Attention 不一致！"
        
        print("✅ Online Softmax 与分块计算逻辑正确！")
        print("\n FlashAttention 分块计算逻辑验证通过。")
        
    except NotImplementedError:
        print("请先完成 TODO 代码！")
    except TypeError as e:
        print("代码可能未完成，导致了操作错误。")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

test_flash_attention_sim()

```

### Step 5: 工业界的演进 —— FlashAttention V1 vs V2 vs V3

了解了基础的 Online Softmax 和分块机制后，我们来看看业界是如何一步步充分发挥 GPU（如 A100 和 H100）硬件性能的。这是大厂高阶岗位的必考题。

> **FlashAttention-1 (2022)：打破显存墙**
> - **核心创新**：如我们刚才所写，通过 Tiling（分块）和 Recomputation（重计算），将空间复杂度从 $O(N^2)$ 降到 $O(N)$。
> - **局限**：在 Thread Block 内部，做了太多非矩阵乘法（Non-Matmul）的运算（如反复缩放中间变量）。同时，在 Batch Size 很小但序列极长的情况下，GPU 利用率（Occupancy）非常低。

> **FlashAttention-2 (2023)：算法级优化与多维并行**
> - **核心创新 1：减少 Non-Matmul FLOPs**。调整了内部循环逻辑，不再于每一步都除以局部最大的指数，而是推迟到最后统一进行缩放。这把宝贵的 CUDA Core 算力省下来留给了 Tensor Core。
> - **核心创新 2：Sequence Parallelism (序列级并行)**。V1 只是在 Batch 和 Head 维度上给不同的 Thread Block 分配任务。V2 允许在序列长度维度上进行切块分配，这让**长文本推理**时 GPU 能够处于满载状态。

> **FlashAttention-3 (2024)：绑定 Hopper (H100) 的极限压榨**
> - **核心创新 1：WGMMA 异步计算**。H100 引入了 Warp Group (4 个 Warp 为一组)，FA3 使用了底层架构特有指令，允许 Tensor Core 在后台异步执行计算，而不阻塞寄存器。
> - **核心创新 2：TMA (Tensor Memory Accelerator)**。H100 专有的硬件级内存搬运器。FA3 让 TMA 自动从全局显存抓取数据到共享内存 (SRAM)，完全解放了用于搬运数据的线程。
> - **核心创新 3：2-Stage to Ping-Pong Pipeline**。V1/V2 是两级流水线（Load -> Compute）。FA3 利用 TMA 实现了高效的软件流水线以掩盖延迟，实现了计算与访存的有效重叠。

---
** 思考题（进阶验证）**：
在 V1 的算法中，我们在内层循环每次更新块时，都会执行 `v_block = v_block * scale1 + v_i * scale2`。这个标量乘法是跑在 CUDA Core 上的，速度很慢。
如果我们要朝着 FlashAttention-2 的方向优化上面的纯 PyTorch 模拟代码，我们应该怎么在数学上修改这段 `Online Softmax`，使得 `v_block` 的缩放只在整个循环结束时发生一次？
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
def flash_attention_forward_sim(q, k, v, block_size=2):
    """
    纯 PyTorch 模拟 FlashAttention 前向传播。
    假设没有 Batch 和 Head 维度，q, k, v 的形状都是 (seq_len, dim)。
    """
    seq_len, dim = q.shape
    
    # TODO 1: 初始化输出 O，全局最大值 m，全局指数和 l
    out = torch.zeros((seq_len, dim), device=q.device)
    m = torch.full((seq_len, 1), -float('inf'), device=q.device)
    l = torch.zeros((seq_len, 1), device=q.device)
    
    scale = 1.0 / math.sqrt(dim)
    
    # 外层循环：遍历 Q 的分块
    for i in range(0, seq_len, block_size):
        q_block = q[i:i+block_size] * scale
        m_i = m[i:i+block_size]
        l_i = l[i:i+block_size]
        out_i = out[i:i+block_size]
        
        # 内层循环：遍历 K, V 的分块
        for j in range(0, seq_len, block_size):
            k_block = k[j:j+block_size]
            v_block = v[j:j+block_size]
            
            # TODO 2: 计算当前块的未归一化分数 S_ij
            S_ij = q_block @ k_block.transpose(-2, -1)
            
            # TODO 3: 计算当前块的局部最大值 m_block，并求出新的全局最大值 m_new
            m_block = torch.max(S_ij, dim=-1, keepdim=True)[0]
            m_new = torch.maximum(m_i, m_block)
            
            # TODO 4: 计算 P_ij = exp(S_ij - m_new)
            P_ij = torch.exp(S_ij - m_new)
            
            # TODO 5: 计算当前块的局部指数和 l_block，并更新全局指数和 l_new
            l_block = torch.sum(P_ij, dim=-1, keepdim=True)
            l_new = l_i * torch.exp(m_i - m_new) + l_block
            
            # TODO 6: 更新输出 O_i（使用 Online Softmax 的修正公式）
            out_i = out_i * (l_i * torch.exp(m_i - m_new) / l_new) + (P_ij @ v_block) / l_new
            
            # 更新全局状态
            m_i = m_new
            l_i = l_new
        
        # 写回全局变量
        out[i:i+block_size] = out_i
        m[i:i+block_size] = m_i
        l[i:i+block_size] = l_i
            
    return out
```

### 解析

**1. TODO 1: 初始化全局状态**
- **实现方式**：`out = torch.zeros((seq_len, dim))`，`m = torch.full((seq_len, 1), -float('inf'))`，`l = torch.zeros((seq_len, 1))`
- **关键点**：m 初始化为负无穷，确保第一个块的最大值能正确更新；l 初始化为 0，用于累加指数和
- **技术细节**：使用 `keepdim=True` 保持二维列向量形状，便于后续广播运算

**2. TODO 2: 计算当前块的未归一化分数 S_ij**
- **实现方式**：`S_ij = q_block @ k_block.transpose(-2, -1)`
- **关键点**：这是标准的 Attention Score 计算，但只针对当前的 Q 块和 K 块
- **技术细节**：q_block 已经在外层循环中乘以了 scale，避免重复缩放

**3. TODO 3: 计算局部最大值并更新全局最大值**
- **实现方式**：`m_block = torch.max(S_ij, dim=-1, keepdim=True)[0]`，`m_new = torch.maximum(m_i, m_block)`
- **关键点**：Online Softmax 的核心——动态更新最大值，用于数值稳定性
- **技术细节**：使用 `torch.maximum` 而非 `torch.max`，因为需要逐元素比较两个张量

**4. TODO 4: 计算归一化的注意力权重 P_ij**
- **实现方式**：`P_ij = torch.exp(S_ij - m_new)`
- **关键点**：减去 m_new 防止指数溢出，这是 Softmax 的标准数值稳定技巧

**5. TODO 5: 计算局部指数和并更新全局指数和**
- **实现方式**：`l_block = torch.sum(P_ij, dim=-1, keepdim=True)`，`l_new = l_i * torch.exp(m_i - m_new) + l_block`
- **关键点**：Online Softmax 的修正公式——当最大值变化时，需要用指数因子修正旧的指数和
- **技术细节**：`l_i * torch.exp(m_i - m_new)` 是修正项，将旧的指数和调整到新的基准 m_new

**6. TODO 6: 更新输出 O_i**
- **实现方式**：`out_i = out_i * (l_i * torch.exp(m_i - m_new) / l_new) + (P_ij @ v_block) / l_new`
- **关键点**：同时修正旧输出和累加新输出，确保最终结果等价于标准 Attention
- **技术细节**：第一项是修正后的旧输出，第二项是当前块的贡献

**工程优化要点**
- **空间复杂度**：从 O(N²) 降至 O(N)，避免存储完整的 Attention Score 矩阵
- **数值稳定性**：通过动态更新最大值 m，确保指数运算不会溢出
- **分块策略**：block_size 是关键超参数，需要根据硬件的 SRAM 大小调优
- **在线更新**：无需等待所有块计算完成，每个块处理后立即更新全局状态
- **工业实现**：真实的 FlashAttention 使用 CUDA/Triton 实现，利用共享内存和寄存器优化访存