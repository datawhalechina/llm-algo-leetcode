# 20. FlashAttention Sim | FlashAttention 模拟
**难度：** Hard | **环境：** GPU required | **标签：** `推理优化`, `FlashAttention`, `Attention` | **目标人群：** 推理优化与系统开发

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/20_FlashAttention_Sim.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


先把反向传播后的显存压力和前向计算路径看清，再进入 FlashAttention 的前向模拟会更顺。

**关键词：** `FlashAttention`, `online softmax`, `tiling`


## 前置阅读

**导语：** 先理解 GPU 内存层级、混合精度和显存分析，再看 FlashAttention 的前向模拟会更容易。
- [P1: 03. GPU Architecture and Memory | GPU 物理架构与内存层级](../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.md)
- [P1: 12. TensorCore and Mixed Precision | Tensor Core 与混合精度](../01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.md)
- [13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
- [P1: 14. FlashAttention Memory Model | FlashAttention 显存模型](../01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.md)

## 相关阅读

**导语：** FlashAttention 后，可以继续看注意力显存优化、KV Cache 和算子融合。
- [P1: 04. Attention Variants and Memory Optimization | 注意力机制变体与显存优化](../01_Hardware_Math_and_Systems/04_Attention_Memory_Optimization.md)
- [P1: 11. KV Cache and Memory Growth | KV Cache 与显存增长](../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md)
- [P1: 19. Operator Fusion Introduction | 算子融合导论](../01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.md)

### Step 1: 核心理论与 Online Softmax

> **标准 Softmax 的痛点：**
> 1. 求每一行的最大值 $m = \max(x)$ (防溢出)。
> 2. 求每一行的指数和 $l = \sum e^{x - m}$。
> 3. 求最终结果 $y_i = \frac{e^{x_i - m}}{l}$。
> 这意味着在算出所有 $x$ 之前，你无法算出 $m$ 和 $l$，因此必须把所有的 $x$ 先存下来。在 Attention 中，$x$ 就是那个大规模的 $S = QK^T$ 矩阵！这也是 FlashAttention 必须引入分块计算的根本原因。

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

### Step 4: 工业界的演进 —— FlashAttention V1 vs V2 vs V3

了解了基础的 Online Softmax 和分块机制后，我们再看业界是如何一步步把 GPU 硬件性能榨出来的。这一段是理解 FlashAttention 演进脉络的核心，也是高阶面试里经常会被追问的部分。

> **FlashAttention-1 (2022)：打破显存墙**
> - **核心创新**：通过 Tiling（分块）和 Recomputation（重计算），把空间复杂度从 $O(N^2)$ 降到 $O(N)$。
> - **局限**：Thread Block 内部的 Non-Matmul 计算偏多，且在短 Batch / 长序列场景下 Occupancy 不高。

> **FlashAttention-2 (2023)：算法级优化与多维并行**
> - **核心创新 1：减少 Non-Matmul FLOPs**。调整内部循环和归一化逻辑，减少每步不必要的标量运算，把更多算力留给 Tensor Core。
> - **核心创新 2：Sequence Parallelism（序列级并行）**。把序列长度维度也纳入切块调度，让长文本推理时 GPU 更容易保持满载。

> **FlashAttention-3 (2024)：绑定 Hopper (H100) 的极限压榨**
> - **核心创新 1：WGMMA 异步计算**。利用 Warp Group 级指令，让 Tensor Core 在后台异步执行。
> - **核心创新 2：TMA（Tensor Memory Accelerator）**。使用硬件级搬运器把数据从全局显存搬到共享内存，释放搬运线程。
> - **核心创新 3：2-Stage to Ping-Pong Pipeline**。通过更高效的软件流水线掩盖访存延迟，实现计算与访存的重叠。

> **FlashAttention-4：CuTeDSL 与 Blackwell 方向**
> - **核心方向**：继续沿着工程化优化推进，把更底层的 kernel 构建、内存调度和流水线协同做得更细。
> - **直观理解**：相比 V1/V2/V3 更强调“代码生成 + kernel 组织”的一体化优化，而不是只停留在数学公式层面的改写。
> - **教学定位**：这一版可以理解为 FlashAttention 工程演进的最新补充，读者只要知道它是继续面向新 GPU 架构演化即可。

### 思考题

在 V1 的算法中，我们在内层循环每次更新块时，都会执行 `v_block = v_block * scale1 + v_i * scale2`。这个标量乘法是跑在 CUDA Core 上的，速度很慢。
如果我们要朝着 FlashAttention-2 的方向优化上面的纯 PyTorch 模拟代码，应该怎么在数学上修改这段 `Online Softmax`，使得 `v_block` 的缩放只在整个循环结束时发生一次？
### Step 5: 动手实战

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
    
    # TODO 1: 初始化输出 O，全局最大值 m，全局指数和 l
    # 提示: 先构造与 seq_len / dim 对齐的输出张量，再初始化 m 和 l
    # out = ???
    # m = ???
    # l = ???
    
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
            # S_ij = ???
            
            # TODO 3: 计算当前块的局部最大值 m_block，并求出新的全局最大值 m_new
            # m_block = ???
            # m_new = ???
            
            # TODO 4: 计算 P_ij = exp(S_ij - m_new)
            # P_ij = ???
            
            # TODO 5: 计算当前块的局部指数和 l_block，并更新全局指数和 l_new
            # l_block = ???
            # l_new = ???
            
            # TODO 6: 更新输出 O_i（使用 Online Softmax 的修正公式）
            # out_i = ???
            
            # 更新全局状态
            # m_i = ???
            # l_i = ???
            pass
        
        # 写回全局变量
        # out[i:i+block_size] = ???
        # m[i:i+block_size] = ???
        # l[i:i+block_size] = ???
            
    return out

```


```python
# 测试你的实现
def test_flash_attention_sim():
    try:
        import math

        def run_case(seq_len, dim, block_size, seed):
            torch.manual_seed(seed)
            q = torch.randn(seq_len, dim)
            k = torch.randn(seq_len, dim)
            v = torch.randn(seq_len, dim)

            scale = 1.0 / math.sqrt(dim)
            scores = (q @ k.transpose(-2, -1)) * scale
            attn = torch.nn.functional.softmax(scores, dim=-1)
            out_ref = attn @ v

            out_sim = flash_attention_forward_sim(q, k, v, block_size=block_size)
            diff = torch.max(torch.abs(out_ref - out_sim))
            print(f"[seq={seq_len}, dim={dim}, block={block_size}] 最大误差: {diff.item():.6e}")
            assert diff < 1e-5, f"计算结果与标准 Attention 不一致！(seq={seq_len}, dim={dim}, block={block_size})"

        run_case(seq_len=8, dim=4, block_size=2, seed=42)
        run_case(seq_len=5, dim=3, block_size=3, seed=7)
        run_case(seq_len=3, dim=2, block_size=1, seed=123)

        print("✅ Online Softmax 与分块计算逻辑正确！")
        print("\n FlashAttention 分块计算逻辑验证通过。")

    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError, RuntimeError) as e:
        if isinstance(e, AttributeError):
            print("代码未完成，无法找到必要的属性")
        elif isinstance(e, NameError):
            print("代码可能未完成，导致了变量未定义")
        elif isinstance(e, TypeError):
            print("代码可能未完成，导致了操作错误")
        elif isinstance(e, ValueError):
            print("代码可能未完成，导致了张量维度错误")
        elif isinstance(e, RuntimeError):
            print("代码可能未完成，导致了运行时错误")
        else:
            print("代码可能未完成，导致了断言失败")
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise


test_flash_attention_sim()

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

**进阶思考**
- 如果把 `v_block` 的缩放统一推迟到循环结束，再一次性完成，会如何影响实现复杂度和数值稳定性？
