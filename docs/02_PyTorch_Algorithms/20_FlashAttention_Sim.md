# 20. FlashAttention Sim | FlashAttention 模拟
**难度：** Hard | **环境：** CPU-first | **标签：** `推理优化`, `Attention`, `FlashAttention` | **目标人群：** 推理优化学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/20_FlashAttention_Sim.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

上下文变长以后，Attention 的问题不只是矩阵乘法变大，更麻烦的是中间的 attention score 矩阵会按序列长度平方增长。标准实现往往要把整块 `QK^T` 写到显存再读回来做 softmax 和加权求和，计算还没结束，显存和带宽就已经被中间结果拖住。

FlashAttention 的思路是不要把完整 score 矩阵作为中间结果落到显存里，而是把 Q/K/V 分块，在小块上边算边更新 softmax 统计量和输出。本节用纯 PyTorch 模拟这条前向路径：通过 online softmax 和 tiling 观察数据流如何改变。

**证据边界：** 本节是 CPU-first 的前向机制模拟，测试只比较数值结果与标准 Attention 的一致性；不测量真实 GPU 峰值显存、HBM 带宽、kernel 性能或吞吐。真实 FlashAttention 的收益需要在固定 GPU workload 下单独 benchmark。

**关键词：** `FlashAttention`, `online softmax`, `tiling`

---

## 前置阅读

- [P1: 03. GPU Architecture and Memory | GPU 物理架构与内存层级](../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.md)
- [P1: 14. FlashAttention Memory Model | FlashAttention 显存模型](../01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.md)
- [P1: 24. SRAM Optimization Techniques | SRAM 优化技术](../01_Hardware_Math_and_Systems/24_SRAM_Optimization_Techniques.md)

## 相关阅读

- [22. vLLM PagedAttention | vLLM 分页注意力](./22_vLLM_PagedAttention.md)
- [34. Prefix Caching and Chunked Prefill | 前缀缓存与分块预填充](./34_Prefix_Caching_and_Chunked_Prefill.md)
- [66. Inference Performance Comparison | 推理性能对比项目](./66_Inference_Performance_Comparison.md)

---

### Step 1: 核心理论与 Online Softmax

> **标准 Softmax 的痛点：**
> 1. 求每一行的最大值 $m = \max(x)$ (防溢出)。
> 2. 求每一行的指数和 $l = \sum e^{x - m}$。
> 3. 求最终结果 $y_i = \frac{e^{x_i - m}}{l}$。
> 在显式 materialized Attention 实现中，通常要先得到完整的 $x$，再计算行级的 $m$ 和 $l$，因此会暂存完整 score 矩阵。在 Attention 中，$x$ 对应 $S = QK^T$；分块实现则可以边处理边维护 softmax 状态。

> **Online Softmax 的机制：**
> 我们可以在只看到**部分数据**时，持续更新一个局部的最大值 $m_{new}$ 和局部的指数和 $l_{new}$。
> 当新来一个分块 (Block) 时，如果新块的最大值更大，我们可以用一个数学技巧，把之前算好的部分“修正”过来，而不需要重新算前面的块！
> 
> **更新公式：**
> - 新的局部最大值：$m_{new} = \max(m_{old}, m_{block})$
> - 修正旧的指数和：$l_{new} = l_{old} \cdot e^{m_{old} - m_{new}} + l_{block} \cdot e^{m_{block} - m_{new}}$
> - 修正旧的输出结果（乘积累加）：$O_{new} = O_{old} \cdot \frac{l_{old} \cdot e^{m_{old} - m_{new}}}{l_{new}} + \frac{e^{S_{block} - m_{new}} \cdot V_{block}}{l_{new}}$

### Step 2: Flash Attention 分块机制原理
在显式 materialized Attention 实现中，Attention Score 矩阵 $S = QK^T$ 的空间和访存压力随序列长度平方增长，长上下文时可能成为 OOM 或性能瓶颈。FlashAttention 在序列维度上对 Q、K、V 分块（Tiling），通过外层遍历 Q 块、内层遍历 K/V 块来避免保存完整的 score 矩阵。本节只讨论前向中间工作集的变化，不给出完整训练或推理显存的统一复杂度结论。

![FlashAttention 分块图](/02_PyTorch_Algorithms/20_flashattention_tiling.svg)

### Step 3: 代码实现框架
代码使用两层 Python 分块循环：外层遍历 $Q_{block}$，内层遍历 $K_{block}$ 和 $V_{block}$；块内矩阵乘法仍由 PyTorch 完成。每处理一个 K/V 块，就更新当前 Q 块的最大值 $m$、指数和 $l$ 以及输出。

### Step 4: 从算法模拟到真实 FlashAttention Kernel

本节代码只覆盖 Online Softmax 和分块数据流。真实 Kernel 还需要处理 tile layout、shared memory、register、Tensor Core、异步加载和线程块调度。下面的版本演进作为扩展背景，不纳入本节的 CPU 测试结论。

> **FlashAttention-1 (2022)：建立分块前向的基本范式**
> - **关注点**：Tiling、online softmax 与必要的重计算，减少显式 Attention 矩阵的中间存储。
> - **边界**：这里描述的是算法与数据流线索；实际复杂度和收益还取决于 kernel、硬件和 workload。

> **FlashAttention-2 (2023)：继续优化并行划分与非矩阵乘开销**
> - **关注点**：调整 work partition 和循环组织，让更多时间用于矩阵乘并提高并行利用率。
> - **边界**：本节不复现其 CUDA kernel，也不据此推断某个 GPU 上的吞吐提升。

> **FlashAttention-3 (2024)：面向 Hopper 的硬件协同方向**
> - **关注点**：WGMMA、TMA 与软件流水线等机制如何配合异步计算和数据搬运。
> - **边界**：这些机制需要对应硬件与 kernel 实现；CPU 模拟无法验证它们。

> **FlashAttention-4：CuTeDSL 与新一代 GPU kernel 方向（扩展背景）**
> - **关注点**：把 kernel 构建、内存调度、流水线组织和代码生成放在更紧密的工程链路中考虑。
> - **边界**：本节只把它作为后续阅读入口，不把具体版本特性或性能数字当作本节结论。

### 思考题

在 V1 的算法中，我们在内层循环每次更新块时，都会执行 `v_block = v_block * scale1 + v_i * scale2`。这个标量乘法是跑在 CUDA Core 上的，速度很慢。
如果我们要朝着 FlashAttention-2 的方向优化上面的纯 PyTorch 模拟代码，应该怎么在数学上修改这段 `Online Softmax`，使得 `v_block` 的缩放只在整个循环结束时发生一次？
### Step 5: 动手实战

**要求**：请补全下方 `flash_attention_forward_sim` 函数，实现二维 `[seq_len, dim]` 输入上的分块 QKV 乘法和 Online Softmax。这里不实现 batch/head、mask、dropout、反向传播或真实 CUDA/Triton kernel。


```python
import torch
import math
```


```python
def flash_attention_forward_sim(q, k, v, block_size=2):
    """计算二维输入上的 FlashAttention 前向模拟。

    Args:
        q, k, v: [seq_len, dim] 张量，device 和 dtype 应保持一致。
        block_size: Q/K/V 的分块大小，必须为正数。

    Returns:
        [seq_len, dim] 的 attention 输出。

    Note:
        只模拟 online softmax 和分块数据流，不实现真实 CUDA/Triton kernel、
        batch/head、mask、dropout 或 backward。
    """
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError('q、k、v 必须是 [seq_len, dim] 二维张量')
    if q.shape != k.shape or k.shape != v.shape:
        raise ValueError('q、k、v 的形状必须一致')
    if q.device != k.device or k.device != v.device:
        raise ValueError('q、k、v 必须位于同一 device')
    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise TypeError('q、k、v 必须使用相同 dtype')
    if block_size <= 0:
        raise ValueError('block_size 必须为正数')

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
            
            # TODO 2: 计算当前 Q/K block 的缩放 score S_ij
            # S_ij = (Q_i / sqrt(d)) @ K_j.T
            
            # TODO 3: 计算当前块的局部最大值 m_block，并求出新的全局最大值 m_new
            # m_block = ???
            # m_new = ???
            
            # TODO 4: 计算尚未归一化的指数权重 exp_scores
            # exp_scores = exp(S_ij - m_new)
            
            # TODO 5: 计算当前块的局部指数和 l_block，并更新全局指数和 l_new
            # l_block = ???
            # l_new = ???
            
            # TODO 6: 更新输出 O_i（修正旧状态并累加当前 V block）
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

        try:
            flash_attention_forward_sim(torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2), block_size=0)
        except ValueError:
            pass
        else:
            raise AssertionError('block_size <= 0 应该被拒绝')

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
    """计算二维输入上的 FlashAttention 前向模拟。

    Args:
        q, k, v: [seq_len, dim] 张量，device 和 dtype 应保持一致。
        block_size: Q/K/V 的分块大小，必须为正数。

    Returns:
        [seq_len, dim] 的 attention 输出。

    Note:
        只模拟 online softmax 和分块数据流，不实现真实 CUDA/Triton kernel、
        batch/head、mask、dropout 或 backward。
    """
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError('q、k、v 必须是 [seq_len, dim] 二维张量')
    if q.shape != k.shape or k.shape != v.shape:
        raise ValueError('q、k、v 的形状必须一致')
    if q.device != k.device or k.device != v.device:
        raise ValueError('q、k、v 必须位于同一 device')
    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise TypeError('q、k、v 必须使用相同 dtype')
    if block_size <= 0:
        raise ValueError('block_size 必须为正数')

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
            
            # TODO 2: 计算当前 Q/K block 的缩放 score S_ij
            S_ij = q_block @ k_block.transpose(-2, -1)
            
            # TODO 3: 计算当前块的局部最大值 m_block，并求出新的全局最大值 m_new
            m_block = torch.max(S_ij, dim=-1, keepdim=True)[0]
            m_new = torch.maximum(m_i, m_block)
            
            # TODO 4: 计算尚未归一化的指数权重 exp_scores
            exp_scores = torch.exp(S_ij - m_new)
            
            # TODO 5: 计算当前块的局部指数和 l_block，并更新全局指数和 l_new
            l_block = torch.sum(exp_scores, dim=-1, keepdim=True)
            l_new = l_i * torch.exp(m_i - m_new) + l_block
            
            # TODO 6: 更新输出 O_i（使用 Online Softmax 的修正公式）
            out_i = out_i * (l_i * torch.exp(m_i - m_new) / l_new) + (exp_scores @ v_block) / l_new
            
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

**2. TODO 2: 计算当前块的缩放 score S_ij**
- **实现方式**：`S_ij = q_block @ k_block.transpose(-2, -1)`
- **关键点**：这是标准的 Attention Score 计算，但只针对当前的 Q 块和 K 块
- **技术细节**：q_block 已经在外层循环中乘以了 scale，避免重复缩放

**3. TODO 3: 计算局部最大值并更新全局最大值**
- **实现方式**：`m_block = torch.max(S_ij, dim=-1, keepdim=True)[0]`，`m_new = torch.maximum(m_i, m_block)`
- **关键点**：Online Softmax 的核心——动态更新最大值，用于数值稳定性
- **技术细节**：使用 `torch.maximum` 而非 `torch.max`，因为需要逐元素比较两个张量

**4. TODO 4: 计算尚未归一化的指数权重**
- **实现方式**：`exp_scores = torch.exp(S_ij - m_new)`
- **关键点**：这里还不是最终概率，后续需要除以 `l_new`；减去 `m_new` 用于数值稳定。

**5. TODO 5: 计算局部指数和并更新全局指数和**
- **实现方式**：`l_block = torch.sum(exp_scores, dim=-1, keepdim=True)`，`l_new = l_i * torch.exp(m_i - m_new) + l_block`
- **关键点**：Online Softmax 的修正公式——当最大值变化时，需要用指数因子修正旧的指数和
- **技术细节**：`l_i * torch.exp(m_i - m_new)` 是修正项，将旧的指数和调整到新的基准 m_new

**6. TODO 6: 更新输出 O_i**
- **实现方式**：`out_i = out_i * (l_i * torch.exp(m_i - m_new) / l_new) + (exp_scores @ v_block) / l_new`
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
