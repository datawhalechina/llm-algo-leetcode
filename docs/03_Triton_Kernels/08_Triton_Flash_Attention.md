# 08. Triton Flash Attention | 批量与多头前向内核

**难度：** Hard | **标签：** Triton、FlashAttention、Memory Bound | **目标人群：** 核心 Infra 与算子开发

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/03_Triton_Kernels/08_Triton_Flash_Attention.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本节从单头注意力出发，补全带边界掩码与因果掩码的 Triton 前向内核，再用批量、多头输入核对结果。内核只覆盖前向计算，不实现反向传播或训练接口。

### Step 1: 从 Q/K/V 到 Triton Grid

输入采用连续的 batch、heads、sequence、head_dim 四维布局。Grid 的第 0 维选择 Q 的序列块，第 1 维选择一个 batch-head 对；不同头独立计算，不能把 batch 和 head 展平成同一条注意力序列。

每个 Program 载入一个 Q 块，并逐块读取 K/V。序列末尾与 head_dim 不满分块时用掩码保护；因果模式只允许位置 j 小于或等于 i 的键参与归一化。

### Step 2: Online Softmax 与 Tiling 深度剖析
为了能够在 SRAM 中处理长序列点积，Flash Attention 发明了 Tiling 与 Online Softmax。在内层循环迭代块 $j$ 时，局部最大值 $m_{new} = \max(m_{old}, m_j)$ 会发生变化。我们需要乘上修正系数 $e^{m_{old} - m_{new}}$ 来弥补之前循环计算结果造成的偏差，使其在数学上严格等价于完整的全局 Softmax。

### Step 3: 内核状态与支持范围

每个 Q 行维护在线 Softmax 的最大值、指数和与输出累加器。处理新 K/V 块时，先重标定已有状态，再合并当前块贡献；循环结束后用指数和归一化输出。

本实现要求 Q/K/V 连续、位于同一 CUDA 设备，数据类型为 FP16 或 BF16，head_dim 不大于 128。支持非整块序列长度、批量和多头；因果模式要求 Q/K 序列长度相同。它是前向教学内核，不支持 dropout、反向传播或 GQA。

### Step 4: 实现并验证前向内核

补全 Triton kernel 与 Python wrapper：处理 batch-head 索引、边界掩码、因果约束及在线 Softmax 状态更新。测试会覆盖非整块长度、非 2 次幂 head_dim、批量/多头以及因果与非因果输入，并与 PyTorch SDPA 对照。


```python
import torch
import triton
import triton.language as tl

```


```python
@triton.jit
def flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    seqlen_q, seqlen_k, head_dim, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr, IS_CAUSAL: tl.constexpr,
):
    # TODO 1：将 program_id(1) 映射到 batch-head，并用掩码载入 Q。
    # TODO 2：分块载入 K/V，计算分数并应用序列尾部与因果掩码。
    # TODO 3：在线更新最大值、指数和与输出累加器。
    # TODO 4：归一化并按 Q 序列与 head_dim 边界掩码写回。
    return


def triton_flash_attention(q, k, v, *, is_causal=False):
    '实现连续四维 Q/K/V 的前向注意力；不支持 dropout 或反向传播。'
    # TODO 5：验证形状、设备、数据类型、连续性与因果长度约束。
    # TODO 6：创建输出，配置二维 Grid，并启动 flash_attn_fwd_kernel。
    raise NotImplementedError("请完成 TODO 1-6")

```


```python
def test_triton_flash_attention():
    '对照 PyTorch SDPA，检查边界、布局、精度与因果语义。'
    if not torch.cuda.is_available():
        assert callable(triton_flash_attention)
        print("⏭️ 当前无 CUDA；仅完成接口检查，数值验证需在 GPU 环境运行。")
        return

    torch.manual_seed(42)
    cases = [
        ((2, 3, 70, 70, 48), True),
        ((1, 2, 73, 73, 64), True),
        ((1, 2, 67, 81, 32), False),
        ((1, 1, 1, 1, 32), True),
    ]
    dtypes = [torch.float16]
    if torch.cuda.is_bf16_supported():
        dtypes.append(torch.bfloat16)

    for dtype in dtypes:
        tolerance = 5e-3 if dtype == torch.float16 else 2e-2
        for (batch, heads, seqlen_q, seqlen_k, head_dim), is_causal in cases:
            q = torch.randn(batch, heads, seqlen_q, head_dim, device="cuda", dtype=dtype)
            k = torch.randn(batch, heads, seqlen_k, head_dim, device="cuda", dtype=dtype)
            v = torch.randn_like(k)
            actual = triton_flash_attention(q, k, v, is_causal=is_causal)
            expected = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=is_causal,
            )
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
            assert actual.shape == q.shape and actual.dtype == q.dtype
    print("✅ Triton 前向注意力与 PyTorch SDPA 数值核对通过。")


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
@triton.jit
def flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    seqlen_q, seqlen_k, head_dim, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr, IS_CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q_ptr + (pid_bh * seqlen_q + offs_m[:, None]) * head_dim + offs_d[None, :]
    q_mask = (offs_m[:, None] < seqlen_q) & (offs_d[None, :] < head_dim)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    num_n_blocks = tl.cdiv(seqlen_k, BLOCK_N)

    for block_n in range(0, num_n_blocks):
        current_n = block_n * BLOCK_N + offs_n
        kv_mask = (current_n[:, None] < seqlen_k) & (offs_d[None, :] < head_dim)
        k_ptrs = K_ptr + (pid_bh * seqlen_k + current_n[:, None]) * head_dim + offs_d[None, :]
        v_ptrs = V_ptr + (pid_bh * seqlen_k + current_n[:, None]) * head_dim + offs_d[None, :]
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0)

        scores = tl.dot(q, tl.trans(k)) * sm_scale
        valid_keys = current_n[None, :] < seqlen_k
        if IS_CAUSAL:
            valid_keys = valid_keys & (current_n[None, :] <= offs_m[:, None])
        scores = tl.where(valid_keys, scores, float("-inf"))

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        p = tl.exp(scores - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_new = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
        l_i = l_new

    out = acc / l_i[:, None]
    out_ptrs = Out_ptr + (pid_bh * seqlen_q + offs_m[:, None]) * head_dim + offs_d[None, :]
    out_mask = (offs_m[:, None] < seqlen_q) & (offs_d[None, :] < head_dim)
    tl.store(out_ptrs, out, mask=out_mask)


def triton_flash_attention(q, k, v, *, is_causal=False):
    'Run forward-only attention on contiguous [batch, heads, sequence, dim] tensors.'
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Q/K/V must have shape [batch, heads, sequence, head_dim]")
    if q.shape[:2] != k.shape[:2] or k.shape[:2] != v.shape[:2]:
        raise ValueError("Q/K/V batch and head dimensions must match")
    if q.shape[-1] != k.shape[-1] or k.shape[-1] != v.shape[-1]:
        raise ValueError("Q/K/V head_dim must match")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("K/V sequence lengths must match")
    if not q.is_cuda or q.device != k.device or q.device != v.device:
        raise ValueError("Q/K/V must be on the same CUDA device")
    if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("Q/K/V must share FP16 or BF16 dtype")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("Q/K/V must be contiguous")
    if q.requires_grad or k.requires_grad or v.requires_grad:
        raise ValueError("This educational kernel is forward-only and does not support autograd")

    batch, heads, seqlen_q, head_dim = q.shape
    seqlen_k = k.shape[-2]
    if not 0 < head_dim <= 128:
        raise ValueError("head_dim must be in [1, 128]")
    if is_causal and seqlen_q != seqlen_k:
        raise ValueError("Causal self-attention requires equal Q/K sequence lengths")

    block_m, block_n = 32, 64
    block_dmodel = triton.next_power_of_2(head_dim)
    output = torch.empty_like(q)
    grid = (triton.cdiv(seqlen_q, block_m), batch * heads)
    flash_attn_fwd_kernel[grid](
        q, k, v, output, seqlen_q, seqlen_k, head_dim,
        1.0 / math.sqrt(head_dim), block_m, block_n, block_dmodel, is_causal,
        num_warps=4,
    )
    return output

```

### 解析

1. **索引与 Q 加载：** 一个 program 处理一个 Q 序列块和一个 batch-head 对。Q/K/V 保持 `[batch, heads, sequence, head_dim]`，序列尾部与 head_dim 尾部都由掩码处理。
2. **分块分数与掩码：** 每轮载入一块 K/V，计算缩放点积；先屏蔽 K 序列越界位置，再按需要屏蔽未来 token。
3. **在线 Softmax：** `row_max`、`row_sum` 和 `acc` 分别保存当前最大值、归一化分母与加权输出。最大值变化时，用指数修正因子缩放历史分母和累加器，再合并当前 K/V 块。
4. **归一化与写回：** 遍历完 K/V 后用 `acc / row_sum` 得到输出，并按 Q 序列和 head_dim 边界写回。
5. **Python 接口：** wrapper 检查四维形状、同设备同 dtype、连续布局与支持范围；Grid 的第二维覆盖 `batch * heads`，因果模式限定为等长自注意力。
6. **验证：** 测试以 PyTorch SDPA 为数值参照，覆盖批量、多头、非整块长度、非 2 次幂 head_dim，以及因果和非因果路径。该内核只实现前向，不含反向传播、dropout 或 GQA。
