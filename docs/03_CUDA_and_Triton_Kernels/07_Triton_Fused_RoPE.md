# 07 Triton Fused RoPE

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/03_CUDA_and_Triton_Kernels/07_Triton_Fused_RoPE.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 07. Triton 进阶：融合旋转位置编码 (Fused RoPE)

**难度：** Hard | **标签：** `Triton`, `RoPE`, `Llama` | **目标人群：** 核心 Infra 与算子开发

在 `02_PyTorch_Algorithms` 章节中，我们学习过 LLaMA 的 RoPE 是通过将连续特征配对，进行复数旋转注入位置信息的。
标准的 PyTorch 实现涉及张量切片 (`x[..., 0::2]`)、拼接 (`cat`) 以及逐元素乘法 (`* cos`, `* sin`)，在推理时高度消耗显存带宽。
主流推理引擎（如 vLLM 和 TensorRT-LLM）底层普遍使用 **Triton / CUDA 融合算子** 来直接就地 (In-place) 计算 RoPE。本节我们将实现这一工业界高频应用的算子。


> **相关阅读**:
> 本节使用 Triton 实现了底层的极致显存与计算优化。
> 如果你对该算子的数学公式推导和纯 PyTorch 高层结构还不熟悉，建议先复习 PyTorch 篇：
>  [`../02_PyTorch_Algorithms/03_RoPE_Tutorial.ipynb`](../02_PyTorch_Algorithms/03_RoPE_Tutorial.ipynb)


### Step 1: RoPE 的物理内存布局与并行策略

> **特征交错与配对：**
> 假设 Head 维度大小为 $d$。对于一个 Token 的 Head 向量 $[x_0, x_1, x_2, x_3]$。
> 我们需要把 $(x_0, x_1)$ 配对旋转，$(x_2, x_3)$ 配对旋转。
> 旋转公式：
> $x'_{2i} = x_{2i} \cos(	heta) - x_{2i+1} \sin(	heta)$
> $x'_{2i+1} = x_{2i+1} \cos(	heta) + x_{2i} \sin(	heta)$

> **Triton 算子设计思路：**
> 1. 我们分配**一个 Program 处理一个 Token 的一个 Head**（长度为 $d$）。
> 2. 由于 $d$ 通常较小（例如 LLaMA 中为 128），可将 128 个元素一次性 Load 进 SRAM。
> 3. 从内存提取偶数索引和奇数索引的元素。
> 4. 加载对应的 $\cos$ 和 $\sin$ 频率值。
> 5. 在 SRAM 中执行旋转乘加运算。
> 6. 将结果按照交错顺序 Store 回显存（就地修改）。


### Step 2: RoPE 物理内存布局与并行策略
传统 PyTorch 在执行 RoPE 时会产生切片（Slicing），带来严重的 Memory Bound 碎片开销。通过 Triton 融合，我们可以编写一个 In-place 算子：将奇数列和偶数列的旋转直接在 SRAM 计算后原路写回所在的显存地址，彻底消除内存抖动。


### Step 3: In-place 内核代码框架
内核需要获取当前特征向量的前半部分指针和后半部分指针。利用预先计算好的 `cos` 和 `sin` 块缓存，执行 $X_{new} = X \cdot \cos - X_{shift} \cdot \sin$ 逻辑，并将结果强行覆盖写入原有的 $X$ 指针地址。


###  Step 4: 动手实战

**要求**：请补全下方 `fused_rope_kernel`，实现底层的旋转逻辑。


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
    
    # 必须保证连续内存
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

```python
# 测试你的实现
def test_fused_rope():
    if not torch.cuda.is_available():
        print("⏭️ 忽略测试：无 GPU。")
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
        
    
        print("\n--- ⚡ 性能基准测试 (Benchmark) ---")
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

::: details 💡 点击查看官方解析与参考代码

### 📝 Fused RoPE 参考实现解析

1. **多维 Grid 寻址**: 不同于之前我们用 1D Grid 自己算偏移，Triton 支持最高 3D 的 Grid。对于 `[Batch, Seq, Head]` 刚好对应 3 个维度。使用 `pid_seq * stride_seq + pid_head * stride_head` 这样的形式可以精准跳转到连续内存中的对应位置。
2. **交错寻址 (Interleaved)**: 在 LLaMA 中，相邻的两个元素组成一对。我们生成了偶数索引和奇数索引 `evens` 与 `odds`，分别并行读取。
3. **In-place 修改**: 最后直接 `tl.store` 覆写原地址的显存数据，可以节省一半显存占用。这非常关键。
4. **复数乘法**: 结合公式将 `[x0, x1]` 旋转角度 $	heta$ 的实现非常简洁。

```python
# ==========================================
# 💡 参考答案
# ==========================================

import torch
import triton
import triton.language as tl

@triton.jit
def rope_kernel(
    t_ptr, cos_ptr, sin_ptr,
    seq_len, head_dim,
    stride_batch, stride_seq, stride_head, stride_dim,
    stride_cos_seq, stride_cos_dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid_seq = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_batch = tl.program_id(2)

    t_offset = pid_batch * stride_batch + pid_seq * stride_seq + pid_head * stride_head

    half_dim = head_dim // 2
    evens = tl.arange(0, BLOCK_SIZE // 2) * 2
    odds = evens + 1

    mask = evens < head_dim
    freq_mask = tl.arange(0, BLOCK_SIZE // 2) < half_dim

    x_evens = tl.load(t_ptr + t_offset + evens, mask=mask)
    x_odds = tl.load(t_ptr + t_offset + odds, mask=mask)

    freq_offset = pid_seq * stride_cos_seq + tl.arange(0, BLOCK_SIZE // 2) * stride_cos_dim

    cos_vals = tl.load(cos_ptr + freq_offset, mask=freq_mask)
    sin_vals = tl.load(sin_ptr + freq_offset, mask=freq_mask)

    # 1. 执行旋转公式 (复数乘法)
    out_evens = x_evens * cos_vals - x_odds * sin_vals
    out_odds = x_evens * sin_vals + x_odds * cos_vals

    # 2. In-place 写回显存
    tl.store(t_ptr + t_offset + evens, out_evens, mask=mask)
    tl.store(t_ptr + t_offset + odds, out_odds, mask=mask)

def triton_apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    batch_size, seq_len, num_heads, head_dim = x.shape
    
    BLOCK_SIZE = triton.next_power_of_2(head_dim)
    
    # Grid: 每个序列位置、每个注意力头、每个批次 分配一个 Block
    grid = (seq_len, num_heads, batch_size)
    
    rope_kernel[grid](
        x, cos, sin,
        seq_len, head_dim,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0), cos.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return x

```

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
