# 07. Triton Fused RoPE | Triton 进阶：融合旋转位置编码 (Fused RoPE)

**难度：** Hard | **标签：** `Triton`, `RoPE`, `Llama` | **目标人群：** 核心 Infra 与算子开发

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/03_Triton_Kernels/07_Triton_Fused_RoPE.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


在 `02_PyTorch_Algorithms` 章节中，我们学习过 LLaMA 的 RoPE 是通过将连续特征配对，进行复数旋转注入位置信息的。
标准的 PyTorch 实现涉及张量切片 (`x[..., 0::2]`)、拼接 (`cat`) 以及逐元素乘法 (`* cos`, `* sin`)，在推理时会带来额外的显存带宽开销。
主流推理引擎（如 vLLM 和 TensorRT-LLM）通常会使用 **Triton / CUDA 融合算子** 来直接就地 (In-place) 计算 RoPE。本节我们将实现这一常见工业应用的算子。

## 前置

**导语：** 这一节先看 Part 1 的执行模型和 Triton Block 相关 Group，把 RoPE 的块级计算和访存路径先接上。
- [Part 1: 1B 单卡硬件与访存优化](../01_Hardware_Math_and_Systems/1B.md)
- [Part 1: 1D 异构调度与算子编程](../01_Hardware_Math_and_Systems/1D.md)
- [Part 1: 18 Triton Block 模型](../01_Hardware_Math_and_Systems/18_Triton_Block_Model.md)

## 相关阅读

**导语：** 如果想对照 RoPE 在 PyTorch 层的旋转实现，可以继续看这页；不影响继续读本节，但会更容易对应公式。
- [Part 2: 03 RoPE Tutorial](../02_PyTorch_Algorithms/03_RoPE_Tutorial.md)

### Step 1: RoPE 的数学原理与内存布局

> **RoPE 的复数旋转本质：**
> 将相邻的两个特征对 $(x_{2i}, x_{2i+1})$ 视为复数 $z = x_{2i} + i x_{2i+1}$。
> 乘上旋转因子 $e^{i\theta} = \cos\theta + i\sin\theta$，展开后就得到 RoPE 的旋转公式。
> 因此，偶数位和奇数位必须成对处理，不能彼此独立看待。

> **Triton 的并行策略：**
> 1. 我们分配 **一个 Program 处理一个 Token 的一个 Head**（长度为 $d$）。
> 2. 由于 $d$ 通常较小（例如 LLaMA 中常见为 128），可将整段 head 向量一次性 Load 进 SRAM。
> 3. 从内存提取偶数索引和奇数索引的元素。
> 4. 加载对应的 $\cos$ 和 $\sin$ 频率值。
> 5. 在 SRAM 中执行旋转乘加运算。
> 6. 将结果按照交错顺序 Store 回显存（就地修改）。

### Step 2: In-place Triton 设计思路
传统 PyTorch 在执行 RoPE 时会产生切片（Slicing）和拼接（Concat），带来额外的显存带宽开销。通过 Triton 融合，我们可以把偶数位和奇数位的旋转直接在 SRAM 中完成，然后原路写回原地址，避免中间张量落回 HBM。

### Step 3: 动手实战前的接口边界
当前主线只覆盖 `(seq_len, n_heads, head_dim)` 的连续输入。内核会为每个 Token/Head 申请一个 Program，并使用 `cos` / `sin` 的预计算块做逐元素旋转。如果上层输入带有 `batch` 维度，通常先展平为 `(batch * seq_len, n_heads, head_dim)` 再调用这个 kernel。

###  Step 4: 动手实战

**要求**：请补全下方 `fused_rope_kernel`，实现底层的旋转逻辑。


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
def fused_rope_kernel(
    t_ptr, cos_ptr, sin_ptr,
    seq_len, n_heads, head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # 1. 获取全局 Token 索引 (平铺 batch*seq_len*n_heads)
    pid = tl.program_id(0)
    
    # 2. 定位当前 Token、Head 的特征起始指针
    # 假设张量 t 是连续的，形状是 (total_tokens, n_heads, head_dim)
    t_offset = pid * head_dim
    
    # 获取当前 token 位置 (用于取 cos 和 sin)
    token_idx = pid // n_heads
    
    # 3. 计算偶数和奇数的特征偏移量
    half_dim = head_dim // 2
    evens = tl.arange(0, BLOCK_SIZE // 2) * 2
    odds = evens + 1
    
    mask = evens < head_dim
    
    # 4. 加载特征 x
    x_evens = tl.load(t_ptr + t_offset + evens, mask=mask)
    x_odds = tl.load(t_ptr + t_offset + odds, mask=mask)
    
    # 5. 加载 cos 和 sin
    freq_offset = token_idx * half_dim + tl.arange(0, BLOCK_SIZE // 2)
    freq_mask = tl.arange(0, BLOCK_SIZE // 2) < half_dim
    
    cos_vals = tl.load(cos_ptr + freq_offset, mask=freq_mask)
    sin_vals = tl.load(sin_ptr + freq_offset, mask=freq_mask)
    
    # ==========================================
    # TODO 1: 执行旋转公式
    # ==========================================
    # out_evens = ???
    # out_odds = ???
    
    # ==========================================
    # TODO 2: 将计算结果写回 t_ptr (In-place 修改)
    # ==========================================
    # tl.store(...)
    # tl.store(...)


def triton_apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    x: 形状 (seq_len, n_heads, head_dim)
    cos/sin: 形状 (seq_len, head_dim // 2)
    """
    seq_len, n_heads, head_dim = x.shape
    
    # 这里通常假设连续内存
    x = x.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    
    BLOCK_SIZE = triton.next_power_of_2(head_dim)
    n_elements = seq_len * n_heads
    grid = (n_elements, )
    
    fused_rope_kernel[grid](
        x, cos, sin,
        seq_len, n_heads, head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return x
raise NotImplementedError("请先完成 TODO 代码！")
```


```python
# 测试你的实现
def test_fused_rope():
    if not torch.cuda.is_available():
        print("⏭️忽略测试：无 GPU。")
        return
        
    try:
        torch.manual_seed(42)
        seq_len, n_heads, head_dim = 16, 8, 128
        
        x = torch.randn(seq_len, n_heads, head_dim, device='cuda')
        x_ref = x.clone() # 复制供对照，Triton 是原地修改
        
        cos = torch.randn(seq_len, head_dim // 2, device='cuda')
        sin = torch.randn(seq_len, head_dim // 2, device='cuda')
        
        # 1. PyTorch 原生计算
        x_evens = x_ref[..., 0::2]
        x_odds = x_ref[..., 1::2]
        cos_b = cos.unsqueeze(1)
        sin_b = sin.unsqueeze(1)
        
        out_evens = x_evens * cos_b - x_odds * sin_b
        out_odds = x_evens * sin_b + x_odds * cos_b
        
        y_ref = torch.empty_like(x_ref)
        y_ref[..., 0::2] = out_evens
        y_ref[..., 1::2] = out_odds
        
        # 2. Triton In-place 计算
        triton_apply_rope(x, cos, sin)
        
        # 3. 验证结果
        diff = torch.max(torch.abs(y_ref - x))
        print(f"最大误差: {diff.item():.6e}")
        assert diff < 1e-5, "Triton RoPE 计算结果不正确！"
        
        print("✅ 原地 (In-place) Triton Fused RoPE 算子实现成功！")
        
    
        print("\n--- 性能基准测试 (Benchmark) ---")
        # 增大测试规模 (模拟真实的 Prefill 阶段)
        seq_len, n_heads, head_dim = 2048, 32, 128
        x_large = torch.randn(seq_len, n_heads, head_dim, device='cuda', dtype=torch.float16)
        cos_large = torch.randn(seq_len, head_dim // 2, device='cuda', dtype=torch.float16)
        sin_large = torch.randn(seq_len, head_dim // 2, device='cuda', dtype=torch.float16)
        
        def torch_rope(x, cos, sin):
            x_evens = x[..., 0::2]
            x_odds = x[..., 1::2]
            cos_b = cos.unsqueeze(1)
            sin_b = sin.unsqueeze(1)
            out_evens = x_evens * cos_b - x_odds * sin_b
            out_odds = x_evens * sin_b + x_odds * cos_b
            y = torch.empty_like(x)
            y[..., 0::2] = out_evens
            y[..., 1::2] = out_odds
            return y
            
        quantiles = [0.5, 0.2, 0.8]
        ms_pt, _, _ = triton.testing.do_bench(lambda: torch_rope(x_large.clone(), cos_large, sin_large), quantiles=quantiles)
        ms_tr, _, _ = triton.testing.do_bench(lambda: triton_apply_rope(x_large, cos_large, sin_large), quantiles=quantiles)
        
        print(f"PyTorch Time (Allocates new tensors): {ms_pt:.4f} ms")
        print(f"Triton Time (In-place operation):     {ms_tr:.4f} ms")
        print(f"Speedup:                              {ms_pt / ms_tr:.2f}x")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

test_fused_rope()

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
def fused_rope_kernel(
    t_ptr, cos_ptr, sin_ptr,
    seq_len, n_heads, head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # 1. 获取全局 Token 索引
    pid = tl.program_id(0)
    
    # 2. 定位当前 Token、Head 的特征起始指针
    t_offset = pid * head_dim
    
    # 获取当前 token 位置
    token_idx = pid // n_heads
    
    # 3. 计算偶数和奇数的特征偏移量
    half_dim = head_dim // 2
    evens = tl.arange(0, BLOCK_SIZE // 2) * 2
    odds = evens + 1
    
    mask = evens < head_dim
    
    # 4. 加载特征 x
    x_evens = tl.load(t_ptr + t_offset + evens, mask=mask)
    x_odds = tl.load(t_ptr + t_offset + odds, mask=mask)
    
    # 5. 加载 cos 和 sin
    freq_offset = token_idx * half_dim + tl.arange(0, BLOCK_SIZE // 2)
    freq_mask = tl.arange(0, BLOCK_SIZE // 2) < half_dim
    
    cos_vals = tl.load(cos_ptr + freq_offset, mask=freq_mask)
    sin_vals = tl.load(sin_ptr + freq_offset, mask=freq_mask)
    
    # ==========================================
    # TODO 1: 执行旋转公式
    # ==========================================
    out_evens = x_evens * cos_vals - x_odds * sin_vals
    out_odds = x_evens * sin_vals + x_odds * cos_vals
    
    # ==========================================
    # TODO 2: 将计算结果写回 t_ptr (In-place 修改)
    # ==========================================
    tl.store(t_ptr + t_offset + evens, out_evens, mask=mask)
    tl.store(t_ptr + t_offset + odds, out_odds, mask=mask)

def triton_apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    x: 形状 (seq_len, n_heads, head_dim)
    cos/sin: 形状 (seq_len, head_dim // 2)
    """
    seq_len, n_heads, head_dim = x.shape
    
    # 这里通常假设连续内存
    x = x.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    
    BLOCK_SIZE = triton.next_power_of_2(head_dim)
    n_elements = seq_len * n_heads
    grid = (n_elements, )
    
    fused_rope_kernel[grid](
        x, cos, sin,
        seq_len, n_heads, head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return x
```

### 解析

**1. TODO 1: 执行旋转公式**
- **实现方式**：使用复数旋转公式计算旋转后的偶数和奇数位置特征
  ```python
  out_evens = x_evens * cos_vals - x_odds * sin_vals
  out_odds = x_evens * sin_vals + x_odds * cos_vals
  ```
- **关键点**：这是复数乘法的实部和虚部计算，将相邻的两个特征 $(x_{2i}, x_{2i+1})$ 视为复数进行旋转
- **技术细节**：旋转角度 $\theta$ 由位置编码决定，通过 $\cos(\theta)$ 和 $\sin(\theta)$ 实现旋转变换。数学上等价于复数乘法 $(x_{2i} + ix_{2i+1}) \cdot (\cos\theta + i\sin\theta)$

**2. TODO 2: In-place 写回显存**
- **实现方式**：使用 `tl.store()` 将计算结果直接写回原地址
  ```python
  tl.store(t_ptr + t_offset + evens, out_evens, mask=mask)
  tl.store(t_ptr + t_offset + odds, out_odds, mask=mask)
  ```
- **关键点**：In-place 修改节省显存，避免分配新的输出张量
- **技术细节**：通过交错寻址 (evens/odds)，将旋转后的结果按原始顺序写回。使用 mask 保护确保只写入有效位置。

**工程优化要点**
- **In-place 修改优化**：直接在原张量上修改，相比分配新张量节省 50% 显存，这在大模型推理中至关重要。
- **交错寻址 (Interleaved Addressing)**：通过 `evens = tl.arange(0, BLOCK_SIZE // 2) * 2` 和 `odds = evens + 1` 实现高效的配对访问，避免复杂的索引计算。
- **融合算子减少开销**：将 RoPE 的所有操作（加载、旋转、写回）融合在一个 kernel 中，减少 kernel launch 开销和 HBM 访问次数。
- **1D Grid 并行策略**：每个 Program 处理一个 Token 的一个 Head，简化了索引计算。对于 `(seq_len, n_heads, head_dim)` 的输入，启动 `seq_len * n_heads` 个 Program，充分利用 GPU 并行度。
- **内存连续性保证**：通过 `x.contiguous()` 确保输入张量在内存中连续存储，避免跨步访问带来的性能损失。
- **工业级应用**：vLLM、TensorRT-LLM 等主流推理引擎都使用类似的融合 RoPE kernel，相比 PyTorch 原生实现可获得 2-3x 加速。