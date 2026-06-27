# 08. Triton Flash Attention | 真正的 Flash Attention 前向算子

**难度：** Hard | **标签：** `Triton`, `FlashAttention`, `Memory Bound` | **目标人群：** 核心 Infra 与算子开发

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/03_Triton_Kernels/08_Triton_Flash_Attention.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


在 `02_PyTorch_Algorithms/13_FlashAttention_Sim` 和 `03_Triton_Kernels/06_Triton_Fused_Softmax` 中，我们已经覆盖了 Flash Attention 的两大数学核心：**分块计算 (Tiling)** 和 **在线安全 Softmax 归约 (Online Safe Softmax)**。
本节我们将把这两者结合起来，利用 Triton 在 SRAM 中的极速读写，编写一个可运行在 GPU 上的 Flash Attention 前向计算内核。这是大模型推理与训练提速的重要算子之一。

## 前置

**导语：** 这一节会把分块、在线 Softmax 和注意力核的访存路径接到一起。

- [Part 1: 1B 单卡硬件与访存优化](../01_Hardware_Math_and_Systems/1B.md)
- [Part 1: 1D 异构调度与算子编程](../01_Hardware_Math_and_Systems/1D.md)
- [Part 1: 18 Triton Block 模型](../01_Hardware_Math_and_Systems/18_Triton_Block_Model.md)
- [Part 1: 20 NCCL 与 AllReduce 基础](../01_Hardware_Math_and_Systems/20_NCCL_and_AllReduce_Basics.md)

## 相关阅读
**导语：** 如果你想先把 FlashAttention 的 PyTorch 推导过一遍，可以继续看这页；不影响继续读本节，但会更容易对齐在线 Softmax。
- [Part 2: 13 FlashAttention Sim](../02_PyTorch_Algorithms/13_FlashAttention_Sim.md)

### Step 1: Flash Attention 内核的执行逻辑

> **任务分配 (Grid)：**
> 本节先讲 per-head 的 2D 简化版。若上层输入是 `(batch, seq_len, n_heads, head_dim)`，可以先把 batch 和 head 维展平，再由 wrapper 调用这个 kernel。
> 在这个简化版里，Grid 先写成 `(num_blocks_q,)`，也就是每个 Triton Program 负责一个 Q block 的最终输出。

> **SRAM 内的循环计算过程 (对于当前分配到的 Q Block)：**
> 1. 初始化累加器 `acc` (用于存最终的 O) 和 `m_i`, `l_i` (用于 Online Softmax)。
> 2. 将这块 Q 从 HBM 加载到 SRAM。当前示例先假设边界可以整除，便于聚焦主流程；如果要支持任意长度，需要在 load/store 处补 mask。
> 3. **内层循环遍历 K 和 V 的所有 Block：**
>    - 加载当前块的 K 和 V 到 SRAM。
>    - 计算注意力分数 $S = Q \times K^T$ (SRAM 内的矩阵乘)。
>    - 执行 Online Softmax 更新逻辑，计算出修正系数。
>    - 修正过去的累加器 `acc`，并累加新的 $P \times V$。
> 4. 循环结束后，将最终归一化的结果 `acc / l_i` 写回 HBM。

### Step 2: Online Softmax 与 Tiling 深度剖析
为了能够在 SRAM 中处理长序列点积，Flash Attention 发明了 Tiling 与 Online Softmax。在内层循环迭代块 $j$ 时，局部最大值 $m_{new} = \max(m_{old}, m_j)$ 会发生变化。我们需要乘上修正系数 $e^{m_{old} - m_{new}}$ 来弥补之前循环计算结果造成的偏差，使其在数学上严格等价于完整的全局 Softmax。

### Step 3: 内核基本框架
本节先实现 2D per-head 版本：输入已经整理成 `(seqlen, head_dim)`。batch / multi-head 的 flatten 与 reshape 交给外层 wrapper；如果要支持任意长度，还需要在 load / store 处补 mask。然后建立 1D Grid 架构，每个程序固定加载一个 $Q$ 块和累加器。遍历 $K$ 和 $V$ 的序列长度：加载 $K_j$ 计算局部点积，提取局部最大值，修正历史累加的 $L$ 和 $Acc$ 变量，累加当前的 $P_j \times V_j$，然后继续读取下一个块。最后归一化并写入 HBM。

### Step 4: 动手实战

**要求**：请补全下方 `flash_attn_fwd_kernel`，补全最核心的 SRAM 内部矩阵乘法与 Softmax 状态更新逻辑。当前主线仍是 2D 简化版，不在这一页里展开完整 4D batch / multi-head wrapper。


```python
try:
    import triton
except ModuleNotFoundError:
    try:
        import google.colab  # type: ignore
    except Exception:
        raise
    import subprocess, sys
    print('Installing Triton for Part 3...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'triton'])
    import triton

import torch
import triton
import triton.language as tl
```


```python

@triton.jit
def flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, sm_scale,
    Out_ptr,
    seqlen_q, seqlen_k, head_dim,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    # 1. 确定当前 Program 的 ID 和处理的 Q 块范围
    start_m = tl.program_id(0)
    
    # 初始化指向 Q 块的指针偏移
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # 本节先实现 2D per-head 简化版；4D 输入请先在 wrapper 中 flatten。
    # 我们假设 head_dim 恰好等于 BLOCK_DMODEL，且当前 block 范围可以整除。
    q_ptrs = Q_ptr + (offs_m[:, None] * BLOCK_DMODEL + offs_d[None, :])
    q = tl.load(q_ptrs)
    
    # 2. 初始化累加器和 Online Softmax 的状态
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float('inf')
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    # 3. 内层循环：遍历 K 和 V 的所有块
    num_n_blocks = tl.cdiv(seqlen_k, BLOCK_N)
    
    for start_n in range(0, num_n_blocks):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        
        # 加载 K 块和 V 块
        k_ptrs = K_ptr + (offs_n[:, None] * BLOCK_DMODEL + offs_d[None, :])
        v_ptrs = V_ptr + (offs_n[:, None] * BLOCK_DMODEL + offs_d[None, :])
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        
        # ==========================================
        # TODO 1: 计算注意力分数 S = Q @ K^T * scale
        # ==========================================
        # qk 的形状是 (BLOCK_M, BLOCK_N)
        qk = tl.dot(q, tl.trans(k))
        qk *= sm_scale
        
        # ==========================================
        # TODO 2: 计算 Online Softmax 的新状态
        # ==========================================
        m_block = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_block)
        
        # ==========================================
        # TODO 3: 计算指数并更新 l_i 和 p
        # ==========================================
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_new = l_i * alpha + tl.sum(p, axis=1)
        
        # ==========================================
        # TODO 4: 修正过去的输出结果 acc，并累加新的 p @ v
        # ==========================================
        acc = acc * alpha[:, None]
        # p 先保留为 FP32，再按 V 的 dtype 做乘法，平衡精度和带宽。
        acc += tl.dot(p.to(v.dtype), v)
        
        # 更新状态准备进入下一块的循环
        m_i = m_new
        l_i = l_new
        
    # 4. 循环结束，除以全局的 l_i 完成最终归一化
    acc = acc / l_i[:, None]
    
    # 5. 写回显存
    out_ptrs = Out_ptr + (offs_m[:, None] * BLOCK_DMODEL + offs_d[None, :])
    tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty))

def triton_flash_attention(q, k, v, sm_scale):
    # 限制条件简化：仅支持 2D per-head 张量；4D 输入请先在 wrapper 中 flatten。
    seqlen_q, head_dim = q.shape
    seqlen_k, _ = k.shape
    
    out = torch.empty_like(q)
    
    # 配置分块大小
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_DMODEL = triton.next_power_of_2(head_dim)
    
    grid = (triton.cdiv(seqlen_q, BLOCK_M), )
    
    flash_attn_fwd_kernel[grid](
        q, k, v, sm_scale, out,
        seqlen_q, seqlen_k, head_dim,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
    )
    return out
raise NotImplementedError("请先完成 TODO 1-4")

```


```python
# 测试你的实现
import math

def test_triton_flash_attention():
    if not torch.cuda.is_available():
        print("⏭️ 无 GPU，完成结构检查；运行级验证需要 GPU。")
        assert "flash_attn_fwd_kernel" in globals(), "缺少 flash_attn_fwd_kernel"
        assert "triton_flash_attention" in globals(), "缺少 triton_flash_attention"
        print("✅ Triton Flash Attention 结构检查通过")
        return True
        
    try:
        torch.manual_seed(42)
        cases = [
            (256, 256, 64),
            (128, 256, 64),
            (64, 64, 32),
        ]
        
        for seqlen_q, seqlen_k, head_dim in cases:
            q = torch.randn(seqlen_q, head_dim, device='cuda', dtype=torch.float16)
            k = torch.randn(seqlen_k, head_dim, device='cuda', dtype=torch.float16)
            v = torch.randn(seqlen_k, head_dim, device='cuda', dtype=torch.float16)
            sm_scale = 1.0 / math.sqrt(head_dim)
            
            attn = (q @ k.transpose(-2, -1)) * sm_scale
            attn = torch.nn.functional.softmax(attn, dim=-1)
            out_ref = attn @ v
            out_tri = triton_flash_attention(q, k, v, sm_scale)
            diff = torch.max(torch.abs(out_ref - out_tri))
            print(f"[{seqlen_q}x{seqlen_k}x{head_dim}] 最大误差: {diff.item():.6e}")
            assert diff < 1e-3, f"Triton Flash Attention 结果不正确！(seqlen_q={seqlen_q}, seqlen_k={seqlen_k}, head_dim={head_dim})"
        
        print("✅ Triton Flash Attention 前向计算内核实现成功！")
        print(" 实现了 SRAM 中块与块之间的局部最大值归约更新，进一步理解了 Online Softmax 的核心机制。")
        
        print()
        print("--- 性能基准测试 (Benchmark) ---")
        seqlen_q = 2048
        seqlen_k = 2048
        head_dim = 64
        q_l = torch.randn(seqlen_q, head_dim, device='cuda', dtype=torch.float16)
        k_l = torch.randn(seqlen_k, head_dim, device='cuda', dtype=torch.float16)
        v_l = torch.randn(seqlen_k, head_dim, device='cuda', dtype=torch.float16)
        sm_scale = 1.0 / math.sqrt(head_dim)
        
        def torch_attn(q, k, v):
            attn = (q @ k.transpose(-2, -1)) * sm_scale
            attn = torch.nn.functional.softmax(attn, dim=-1)
            return attn @ v
            
        quantiles = [0.5, 0.2, 0.8]
        try:
            ms_pt, _, _ = triton.testing.do_bench(lambda: torch_attn(q_l, k_l, v_l), quantiles=quantiles)
            ms_tr, _, _ = triton.testing.do_bench(lambda: triton_flash_attention(q_l, k_l, v_l, sm_scale), quantiles=quantiles)
            print(f"PyTorch Time (O(N^2) memory): {ms_pt:.4f} ms")
            print(f"Triton Time (O(N) memory):    {ms_tr:.4f} ms")
            print(f"Speedup:                      {ms_pt / ms_tr:.2f}x")
        except Exception as e:
            print(f"Benchmark 跳过或失败：{e}")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise

test_triton_flash_attention()


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

@triton.jit
def flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, sm_scale,
    Out_ptr,
    seqlen_q, seqlen_k, head_dim,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    # 1. 确定当前 Program 的 ID 和处理的 Q 块范围
    start_m = tl.program_id(0)
    
    # 初始化指向 Q 块的指针偏移
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # 本节先实现 2D per-head 简化版；4D 输入请先在 wrapper 中 flatten。
    # 我们假设 head_dim 恰好等于 BLOCK_DMODEL，且当前 block 范围可以整除。
    q_ptrs = Q_ptr + (offs_m[:, None] * BLOCK_DMODEL + offs_d[None, :])
    q = tl.load(q_ptrs)
    
    # 2. 初始化累加器和 Online Softmax 的状态
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float('inf')
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    # 3. 内层循环：遍历 K 和 V 的所有块
    num_n_blocks = tl.cdiv(seqlen_k, BLOCK_N)
    
    for start_n in range(0, num_n_blocks):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        
        # 加载 K 块和 V 块
        k_ptrs = K_ptr + (offs_n[:, None] * BLOCK_DMODEL + offs_d[None, :])
        v_ptrs = V_ptr + (offs_n[:, None] * BLOCK_DMODEL + offs_d[None, :])
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        
        # ==========================================
        # TODO 1: 计算注意力分数 S = Q @ K^T * scale
        # ==========================================
        # qk 的形状是 (BLOCK_M, BLOCK_N)
        qk = tl.dot(q, tl.trans(k))
        qk *= sm_scale
        
        # ==========================================
        # TODO 2: 计算 Online Softmax 的新状态
        # ==========================================
        m_block = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_block)
        
        # ==========================================
        # TODO 3: 计算指数并更新 l_i 和 p
        # ==========================================
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_new = l_i * alpha + tl.sum(p, axis=1)
        
        # ==========================================
        # TODO 4: 修正过去的输出结果 acc，并累加新的 p @ v
        # ==========================================
        acc = acc * alpha[:, None]
        # p 先保留为 FP32，再按 V 的 dtype 做乘法，平衡精度和带宽。
        acc += tl.dot(p.to(v.dtype), v)
        
        # 更新状态准备进入下一块的循环
        m_i = m_new
        l_i = l_new
        
    # 4. 循环结束，除以全局的 l_i 完成最终归一化
    acc = acc / l_i[:, None]
    
    # 5. 写回显存
    out_ptrs = Out_ptr + (offs_m[:, None] * BLOCK_DMODEL + offs_d[None, :])
    tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty))

def triton_flash_attention(q, k, v, sm_scale):
    # 限制条件简化：仅支持 2D per-head 张量；4D 输入请先在 wrapper 中 flatten。
    seqlen_q, head_dim = q.shape
    seqlen_k, _ = k.shape
    
    out = torch.empty_like(q)
    
    # 配置分块大小
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_DMODEL = triton.next_power_of_2(head_dim)
    
    grid = (triton.cdiv(seqlen_q, BLOCK_M), )
    
    flash_attn_fwd_kernel[grid](
        q, k, v, sm_scale, out,
        seqlen_q, seqlen_k, head_dim,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
    )
    return out
```

### 解析

**1. TODO 1: 计算注意力分数 S = Q @ K^T * scale**
- **实现方式**：
  ```python
  qk = tl.dot(q, tl.trans(k))
  qk *= sm_scale
  ```
- **关键点**：使用 `tl.dot()` 在 SRAM 内计算矩阵乘法，`tl.trans(k)` 对 K 进行转置，输出 `qk` 的形状是 `(BLOCK_M, BLOCK_N)`
- **技术细节**：`sm_scale = 1/√d` 是注意力机制的标准缩放因子，防止点积值过大导致 softmax 梯度消失。在 SRAM 内完成矩阵乘法避免了中间结果写回 HBM 的开销。
- **接口范围**：本节先覆盖 `(seqlen, head_dim)` 的 per-head 简化版；如果上层输入是 `(batch, seq_len, n_heads, head_dim)`，可以先 flatten 再调用 kernel。

**2. TODO 2: 计算 Online Softmax 的新状态**
- **实现方式**：
  ```python
  m_block = tl.max(qk, axis=1)
  m_new = tl.maximum(m_i, m_block)
  ```
- **关键点**：计算当前块的行最大值，并与历史最大值比较更新
- **技术细节**：Online Softmax 的核心是维护全局最大值 `m_i`。每处理一个新的 K/V 块，都需要更新最大值。`tl.maximum()` 逐元素比较，确保 `m_new` 是迄今为止所有块的全局最大值。

**3. TODO 3: 计算指数并更新 l_i 和 p**
- **实现方式**：
  ```python
  p = tl.exp(qk - m_new[:, None])
  alpha = tl.exp(m_i - m_new)
  l_new = l_i * alpha + tl.sum(p, axis=1)
  ```
- **关键点**：计算修正系数 `alpha` 和更新归一化分母 `l_i`
- **技术细节**：
  - `p = tl.exp(qk - m_new[:, None])` 计算当前块的 softmax 分子（减去最大值保证数值稳定）
  - `alpha = tl.exp(m_i - m_new)` 是修正系数，用于缩放之前累加的结果。当 `m_new > m_i` 时，之前的累加值被高估了，需要乘以 `alpha < 1` 进行修正
  - `l_new = l_i * alpha + tl.sum(p, axis=1)` 更新归一化分母，左边是修正后的历史分母，右边是当前块的贡献

**4. TODO 4: 修正过去的输出结果 acc，并累加新的 p @ v**
- **实现方式**：
  ```python
  acc = acc * alpha[:, None]
  acc += tl.dot(p.to(v.dtype), v)
  ```
- **关键点**：先修正历史累加值，再累加当前块的贡献
- **技术细节**：
  - `acc * alpha[:, None]` 将之前累加的输出乘以修正系数，补偿最大值变化带来的影响
  - `tl.dot(p.to(v.dtype), v)` 计算当前块的加权输出并累加。`p.to(v.dtype)` 是精度与带宽的折中：先在 FP32 中完成 softmax，再按 V 的 dtype 参与乘法（通常 v 是 FP16，p 是 FP32）
  - 循环结束后 `acc / l_i[:, None]` 完成最终归一化

**工程优化要点**
- **O(N) 显存复杂度**：标准 Attention 需要存储 `(seq_len, seq_len)` 的注意力矩阵，显存复杂度 O(N²)。Flash Attention 通过分块计算和 Online Softmax，只需 O(N) 显存存储输入输出和累加器。
- **SRAM 优化**：所有计算（矩阵乘法、softmax、加权求和）都在 SRAM 内完成，最小化 HBM 访问。每个 Q 块只需从 HBM 读取一次，所有 K/V 块遍历完成后写回一次。
- **Tiling 策略**：将 Q 分成 `BLOCK_M` 大小的块，K/V 分成 `BLOCK_N` 大小的块。典型配置 `BLOCK_M = BLOCK_N = 128`，在 A100 上可以充分利用 SRAM（192KB）。
- **Online Softmax 数学原理**：通过维护全局最大值 `m_i` 和归一化分母 `l_i`，实现增量式 softmax 计算。修正系数 `alpha = exp(m_old - m_new)` 确保数学上等价于一次性计算全局 softmax。
- **数值稳定性**：始终减去最大值后再计算指数，避免 `exp(large_number)` 导致的溢出。
- **工业级应用**：Flash Attention 是 GPT-3/4、LLaMA 等大模型训练和推理的标配，相比标准实现可获得 2-4x 加速，并支持更长的上下文长度（受限于显存的问题得到根本解决）。