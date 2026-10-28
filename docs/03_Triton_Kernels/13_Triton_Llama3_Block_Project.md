# 13. Triton Llama3 Block Project | 接入 Triton Attention 前向内核

**难度：** Hard | **标签：** Triton、Attention、Block 集成 | **目标人群：** 核心 Infra 与算子开发

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/03_Triton_Kernels/13_Triton_Llama3_Block_Project.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本节把前向注意力内核接入一个 decoder block，并与 PyTorch SDPA 基线做同权重对照。只有 Attention 使用 Triton；归一化、RoPE、线性层和 MLP 保留 PyTorch 实现，因此结果不能称为“全 Triton Block”加速。

### Step 1: 固定 Block 的输入输出关系

输入为 batch-first 隐状态，Q/K/V 投影后整理为 batch、head、sequence、head_dim。位置编码在注意力前应用，Attention 输出恢复原布局后进入输出投影与残差分支。

### Step 2: 明确可替换的执行边界

本实验只切换注意力实现：PyTorch SDPA 与 Triton 前向 kernel。两条路径共用同一组权重、输入、RoPE、因果约束和 MLP；其余算子均为 PyTorch，因此配对结果只归因到 Attention 路径。

### Step 3: 集成约束与验证顺序

Q/K/V 必须整理为连续的四维张量；Triton 路径使用因果前向实现。先比较同一 Block 的输出误差，再测重复前向耗时与峰值显存。短序列边界与非整块长度由测试覆盖；性能数字只代表当前设备、dtype 与输入形状。

### Step 4: 实现可切换的 Attention Block

补全 Block 前向中的归一化、Q/K/V 投影、RoPE、注意力调用、输出投影、残差与 MLP。注意只把 Attention 调用切到 Triton；PyTorch SDPA 保留为配对基线。


```python
import statistics

import torch
import torch.nn as nn
from tools.triton_attention_runtime import triton_flash_attention

```


```python
def rmsnorm_reference(x, weight, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight


def rope_reference(q, k, cos, sin):
    cos = cos.to(device=q.device, dtype=q.dtype)[None, None, :, :]
    sin = sin.to(device=q.device, dtype=q.dtype)[None, None, :, :]

    def rotate(x):
        even, odd = x[..., ::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    return rotate(q), rotate(k)


def swiglu_reference(x, gate_weight, up_weight, down_weight):
    gate = x @ gate_weight.T
    up = x @ up_weight.T
    return (torch.nn.functional.silu(gate) * up) @ down_weight.T


class LlamaBlockWithAttention(nn.Module):
    def __init__(self, dim, hidden_dim, n_heads, attention_backend):
        super().__init__()
        if dim % n_heads:
            raise ValueError("dim must be divisible by n_heads")
        if attention_backend not in {"torch_sdpa", "triton"}:
            raise ValueError("attention_backend must be torch_sdpa or triton")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.attention_backend = attention_backend
        self.attn_q = nn.Linear(dim, dim, bias=False)
        self.attn_k = nn.Linear(dim, dim, bias=False)
        self.attn_v = nn.Linear(dim, dim, bias=False)
        self.attn_o = nn.Linear(dim, dim, bias=False)
        self.mlp_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.mlp_up = nn.Linear(dim, hidden_dim, bias=False)
        self.mlp_down = nn.Linear(hidden_dim, dim, bias=False)
        self.norm1_weight = nn.Parameter(torch.ones(dim))
        self.norm2_weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, cos, sin):
        # TODO 1：完成前置 RMSNorm 与 Q/K/V 投影。
        # TODO 2：整理张量布局并对 Q/K 应用 RoPE。
        # TODO 3：按 attention_backend 选择 SDPA 或 Triton 因果前向。
        # TODO 4：恢复布局，完成输出投影、残差、RMSNorm 与 SwiGLU MLP。
        raise NotImplementedError("请完成 TODO 1-4")

```


```python
def test_attention_block_matches_reference():
    """Check Triton Attention integration against the same-weight SDPA block."""
    if not torch.cuda.is_available():
        print("⏭️ 无 CUDA；Block 数值核对需在 llm_algo_cnb_gpu 环境运行。")
        return

    torch.manual_seed(42)
    dim, hidden_dim, heads = 256, 512, 4
    reference = LlamaBlockWithAttention(dim, hidden_dim, heads, "torch_sdpa")
    candidate = LlamaBlockWithAttention(dim, hidden_dim, heads, "triton")
    candidate.load_state_dict(reference.state_dict())
    reference = reference.cuda().half().eval()
    candidate = candidate.cuda().half().eval()

    batch, seq = 2, 73
    head_dim = dim // heads
    x = torch.randn(batch, seq, dim, device="cuda", dtype=torch.float16)
    phase = torch.arange(seq, device="cuda", dtype=torch.float32)[:, None] / 1000
    freq = torch.arange(head_dim // 2, device="cuda", dtype=torch.float32)[None, :] + 1
    cos, sin = torch.cos(phase * freq), torch.sin(phase * freq)

    with torch.inference_mode():
        expected = reference(x, cos, sin)
        actual = candidate(x, cos, sin)
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
    print("✅ 同权重 Block 的 SDPA / Triton Attention 输出核对通过。")


test_attention_block_matches_reference()

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
def rmsnorm_reference(x, weight, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight


def rope_reference(q, k, cos, sin):
    cos = cos.to(device=q.device, dtype=q.dtype)[None, None, :, :]
    sin = sin.to(device=q.device, dtype=q.dtype)[None, None, :, :]

    def rotate(x):
        even, odd = x[..., ::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    return rotate(q), rotate(k)


def swiglu_reference(x, gate_weight, up_weight, down_weight):
    gate = x @ gate_weight.T
    up = x @ up_weight.T
    return (torch.nn.functional.silu(gate) * up) @ down_weight.T


class LlamaBlockWithAttention(nn.Module):
    """Llama-style block; only the causal attention path is selectable."""

    def __init__(self, dim, hidden_dim, n_heads, attention_backend):
        super().__init__()
        if dim % n_heads:
            raise ValueError("dim must be divisible by n_heads")
        if attention_backend not in {"torch_sdpa", "triton"}:
            raise ValueError("attention_backend must be torch_sdpa or triton")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.attention_backend = attention_backend
        self.attn_q = nn.Linear(dim, dim, bias=False)
        self.attn_k = nn.Linear(dim, dim, bias=False)
        self.attn_v = nn.Linear(dim, dim, bias=False)
        self.attn_o = nn.Linear(dim, dim, bias=False)
        self.mlp_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.mlp_up = nn.Linear(dim, hidden_dim, bias=False)
        self.mlp_down = nn.Linear(hidden_dim, dim, bias=False)
        self.norm1_weight = nn.Parameter(torch.ones(dim))
        self.norm2_weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, cos, sin):
        batch_size, seq_len, _ = x.shape
        h = rmsnorm_reference(x, self.norm1_weight)
        q = self.attn_q(h).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.attn_k(h).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.attn_v(h).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
        q, k = rope_reference(q, k, cos, sin)

        if self.attention_backend == "triton":
            attn_output = triton_flash_attention(q, k, v, is_causal=True)
        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        h = x + self.attn_o(attn_output)
        normalized = rmsnorm_reference(h, self.norm2_weight)
        mlp_output = swiglu_reference(normalized, self.mlp_gate.weight, self.mlp_up.weight, self.mlp_down.weight)
        return h + mlp_output


def benchmark_block_attention_path(repeats=5, warmup=3):
    """Pair SDPA and Triton on identical model weights and inputs."""
    if not torch.cuda.is_available():
        raise RuntimeError("GPU benchmark requires the llm_algo_cnb_gpu environment")
    torch.manual_seed(42)
    dim, hidden_dim, heads = 256, 512, 4
    batch, seq = 1, 128
    head_dim = dim // heads
    x = torch.randn(batch, seq, dim, device="cuda", dtype=torch.float16)
    phase = torch.arange(seq, device="cuda", dtype=torch.float32)[:, None] / 1000
    freq = torch.arange(head_dim // 2, device="cuda", dtype=torch.float32)[None, :] + 1
    cos, sin = torch.cos(phase * freq), torch.sin(phase * freq)
    seed_model = LlamaBlockWithAttention(dim, hidden_dim, heads, "torch_sdpa")
    state = {key: value.detach().cpu().clone() for key, value in seed_model.state_dict().items()}
    del seed_model

    rows = {}
    for backend in ("torch_sdpa", "triton"):
        model = LlamaBlockWithAttention(dim, hidden_dim, heads, backend).cuda().half().eval()
        model.load_state_dict(state)
        with torch.inference_mode():
            for _ in range(warmup):
                model(x, cos, sin)
            torch.cuda.synchronize()
            samples_ms = []
            torch.cuda.reset_peak_memory_stats()
            for _ in range(repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(x, cos, sin)
                end.record()
                end.synchronize()
                samples_ms.append(start.elapsed_time(end))
            rows[backend] = {
                "latency_ms_median": statistics.median(samples_ms),
                "latency_ms_samples": samples_ms,
                "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
            }
        del model
        torch.cuda.empty_cache()
    return rows


test_attention_block_matches_reference()
print(benchmark_block_attention_path())

```

### 解析

1. **投影与布局：** 输入保持 `[batch, sequence, dim]`；Q/K/V 投影后转为 `[batch, heads, sequence, head_dim]`，并保证连续存储。
2. **位置编码与注意力：** RoPE 只作用于 Q/K。基线使用 PyTorch SDPA，候选使用共享的 Triton 前向内核；两者都启用因果掩码。
3. **Block 其余部分：** 输出投影和两条残差路径、RMSNorm、SwiGLU MLP 都保留 PyTorch，因此本节只评估 Attention 替换，不把结果解释成整块 Triton 加速。
4. **核对方式：** 测试使用相同权重和非整块序列长度，比较 Block 输出；微基准重复测量两种 Attention 路径并记录样本时延与峰值分配显存。实际性能结论需要在目标 GPU 环境运行后登记。
