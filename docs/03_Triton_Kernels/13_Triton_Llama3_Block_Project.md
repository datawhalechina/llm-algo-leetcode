# 13. Triton Llama3 Block Project | Triton 工程集成实战（LLaMA-3 Block）

**难度：** Hard | **标签：** `Triton`, `End-to-End Project`, `LLaMA-3`, `Integration` | **目标人群：** 核心 Infra 与算子开发

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/03_Triton_Kernels/13_Triton_Llama3_Block_Project.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


这是本教程 **Triton 算子开发**章节的工程集成题 (Capstone Project)。
在工业界，写出几个零散的算子只是 Demo。你需要把这些算子接回标准的 `torch.autograd.Function` 或 `nn.Module` 接口，去**平替** PyTorch 原生的耗时层，最终拼装出一个可独立运行的 `Llama3TritonBlock`。

在本节中，我们将：
1. 回顾并接入前几节手写的 Triton 算子。
2. 处理好 PyTorch `nn.Module` 的接口边界。
3. 运行端到端的 Benchmark，验证工程集成后的收益。

**说明：** 为了让 Notebook 可以独立运行，下面先提供 reference adapter；如果你已经在当前 runtime 中加载了前序章节的真实 Triton kernel，可以直接替换。

## 前置

**导语：** 这一节会把前面学过的 Triton 单算子真正组合成一个工程化的完整 Block。

- [Part 1: 1B 单卡硬件与访存优化](../01_Hardware_Math_and_Systems/1B.md)
- [Part 1: 1D 异构调度与算子编程](../01_Hardware_Math_and_Systems/1D.md)
- [Part 1: 18 Triton Block 模型](../01_Hardware_Math_and_Systems/18_Triton_Block_Model.md)
- [Part 1: 19 算子融合导论](../01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.md)
- [Part 1: 20 NCCL 与 AllReduce 基础](../01_Hardware_Math_and_Systems/20_NCCL_and_AllReduce_Basics.md)

## 相关阅读
**导语：** 如果你想先把 LLaMA3 Block 的 PyTorch 组装过一遍，可以继续看这页；不影响继续读本节，但会更容易理解工程化集成。
- [Part 2: 05 LLaMA3 Block Tutorial](../02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.md)

### Step 1: 算子替换与模块集成规范

> **PyTorch 原生实现为什么慢？**
> 我们在 `02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial` 中写的 Block：
> `x = x + Attention(RMSNorm(x))`
> `x = x + MLP(RMSNorm(x))`
> 这个过程产生了大量的中间张量 (Intermediate Tensors)，由于频繁的内存读写 (Memory Bound)，严重拖慢了速度。

> **如何进行工程级替换 (Integration)？**
> 1. 继承 `nn.Module` 编写自定义的 Layer。
> 2. 在 Layer 的 `forward` 方法中，直接调用包含 `kernel[grid](...)` 的 Triton 封装函数。
> 3. （如果需要训练）继承 `torch.autograd.Function` 实现 `forward` 和 `backward`，并在 `nn.Module` 中调用 `YourFunction.apply`。本节为了聚焦前向推理性能，只集成推理部分的替换。

### Step 2: 模型封装与接口隔离
这是一个架构拼装工程。虽然我们在前面手写出了所有加速算子，但要组装回基于 `nn.Module` 的 PyTorch 模型中，必须处理好接口（Interface）封装问题，并确保前向传播在 AutoGrad (反向图) 中的逻辑隔离或兼容。

### Step 3: 集成代码框架
定义一个 `Llama3TritonBlock(nn.Module)` 类。在 `__init__` 中保留 `nn.Linear` 管理权重，但在 `forward` 阶段，把原生算子替换为前序章节已经实现的 Triton 封装，并尽量保持接口与布局一致。

### Step 4: 动手实战

**要求**：请补全下方 `TritonLlama3Block`，把前序章节的 Triton 算子按工程接口接入模型前向路径。


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
import torch.nn as nn
import triton
import math
```


```python

# ==========================================
# 这里先给出纯 PyTorch reference adapter，方便 Notebook 独立运行。
# 如果前序章节的真实 Triton kernel 已经在当前 runtime 中加载，可以直接替换这些 adapter。
# ==========================================
def triton_rmsnorm(x, weight, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight

def triton_rope(q, k, cos, sin):
    cos = cos.to(device=q.device, dtype=q.dtype)[None, None, :, :]
    sin = sin.to(device=q.device, dtype=q.dtype)[None, None, :, :]

    def rotate(x):
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        rotated = torch.stack(
            (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos),
            dim=-1,
        )
        return rotated.flatten(-2)

    return rotate(q), rotate(k)

def triton_flash_attn(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)

def triton_swiglu(x, gate_weight, up_weight, down_weight):
    gate = x @ gate_weight.T
    up = x @ up_weight.T
    act = torch.nn.functional.silu(gate) * up
    return act @ down_weight.T

# ==========================================
# 组装完整的 Triton 加速 Block
# ==========================================
class TritonLlama3Block(nn.Module):
    def __init__(self, dim, hidden_dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        # 权重定义
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
        # TODO 1: 接入第一层归一化（先确认 shape / stride / dtype 与前序 kernel 对齐）
        # h = ???
        
        # QKV 投影并变维 (batch, seq, n_heads, head_dim)
        # batch_size, seq_len, _ = ???
        # q = ???
        # k = ???
        # v = ???
        
        # TODO 2: 接入位置编码处理（尽量避免在 PyTorch 层做多余 transpose）
        # q, k = ???
        
        # TODO 3: 接入注意力主干（注意输入布局与 causal / non-causal 约定）
        # attn_output = ???
        
        # 恢复形状并输出投影
        # attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        # h = x + self.attn_o(attn_output)
        
        # TODO 4: 接入 MLP 分支（关注残差连接、权重布局和融合边界）
        # out = ???

        raise NotImplementedError("请先完成 TODO 1-4")

import time

def run_end_to_end_benchmark():
    if not torch.cuda.is_available():
        print("⏭️ 无 GPU，跳过测试")
        return
    
    # 模拟 LLaMA-3 的一个标准层配置
    dim = 4096
    hidden_dim = 14336
    n_heads = 32
    batch, seq = 2, 2048
    
    triton_block = TritonLlama3Block(dim, hidden_dim, n_heads).cuda().half()
    x = torch.randn(batch, seq, dim, device='cuda', dtype=torch.float16)
    
    # 模拟 cos 和 sin
    head_dim = dim // n_heads
    cos = torch.randn(seq, head_dim // 2, device='cuda', dtype=torch.float16)
    sin = torch.randn(seq, head_dim // 2, device='cuda', dtype=torch.float16)
    
    print(" 开始运行端到端 Benchmark (Warmup 10 次，记录 50 次)...")
    # Warmup
    for _ in range(10):
        _ = triton_block(x, cos, sin)
    torch.cuda.synchronize()
    
    # 测试 Triton 整合版的耗时
    start = time.time()
    for _ in range(50):
        _ = triton_block(x, cos, sin)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / 50.0 * 1000 # ms
    
    print(f"✅ 全 Triton 加速的 LLaMA-3 Block 单层前向延迟: {triton_time:.2f} ms")
    print(" 通过算子融合和 SRAM 内计算，Triton 实现显著降低了 Memory Bound 操作的开销。")
raise NotImplementedError("请先完成 TODO 1-4")

```


```python
# 标准测试函数
def test_llama3_block():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32
    if device == 'cpu':
        print("⏭️ 无 GPU，跳过运行级验证。")
        return True
    
    def reference_forward(block, x, cos, sin):
        h = triton_rmsnorm(x, block.norm1_weight)
        batch_size, seq_len, _ = h.shape
        q = block.attn_q(h).view(batch_size, seq_len, block.n_heads, block.head_dim).transpose(1, 2)
        k = block.attn_k(h).view(batch_size, seq_len, block.n_heads, block.head_dim).transpose(1, 2)
        v = block.attn_v(h).view(batch_size, seq_len, block.n_heads, block.head_dim).transpose(1, 2)
        q, k = triton_rope(q, k, cos, sin)
        attn_output = triton_flash_attn(q, k, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        h = x + block.attn_o(attn_output)
        normed_h = triton_rmsnorm(h, block.norm2_weight)
        mlp_out = triton_swiglu(normed_h, block.mlp_gate.weight, block.mlp_up.weight, block.mlp_down.weight)
        return h + mlp_out
    
    torch.manual_seed(42)
    cases = [
        dict(dim=512, hidden_dim=2048, n_heads=8, batch=2, seq=128),
        dict(dim=384, hidden_dim=1536, n_heads=6, batch=1, seq=33),
        dict(dim=256, hidden_dim=1024, n_heads=4, batch=1, seq=1),
    ]
    
    for case in cases:
        dim = case["dim"]
        hidden_dim = case["hidden_dim"]
        n_heads = case["n_heads"]
        batch = case["batch"]
        seq = case["seq"]
        
        triton_block = TritonLlama3Block(dim, hidden_dim, n_heads).to(device=device, dtype=dtype).eval()
        x = torch.randn(batch, seq, dim, device=device, dtype=dtype)
        head_dim = dim // n_heads
        cos = torch.randn(seq, head_dim // 2, device=device, dtype=dtype)
        sin = torch.randn(seq, head_dim // 2, device=device, dtype=dtype)
        
        output = triton_block(x, cos, sin)
        ref = reference_forward(triton_block, x, cos, sin)
        
        assert output.shape == x.shape, "输出形状错误"
        assert output.dtype == x.dtype, "输出 dtype 错误"
        assert not torch.isnan(output).any(), "输出包含 NaN"
        assert not torch.isinf(output).any(), "输出包含 Inf"
        assert torch.allclose(output, ref, atol=2e-3, rtol=2e-3), "输出与参考实现不一致"
        
    
    print("✅ Triton LLaMA-3 Block 测试通过")

test_llama3_block()

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
import torch.nn as nn
import triton
import math

# ==========================================
# 这里用纯 PyTorch 参考实现模拟前序章节的 Triton 封装，
# 这样 Notebook 可以独立运行，同时保留与工程实现一致的接口。
# ==========================================
def triton_rmsnorm(x, weight, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight

def triton_rope(q, k, cos, sin):
    cos = cos.to(device=q.device, dtype=q.dtype)[None, None, :, :]
    sin = sin.to(device=q.device, dtype=q.dtype)[None, None, :, :]

    def rotate(x):
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        rotated = torch.stack(
            (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos),
            dim=-1,
        )
        return rotated.flatten(-2)

    return rotate(q), rotate(k)

def triton_flash_attn(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)

def triton_swiglu(x, gate_weight, up_weight, down_weight):
    gate = x @ gate_weight.T
    up = x @ up_weight.T
    act = torch.nn.functional.silu(gate) * up
    return act @ down_weight.T

# ==========================================
# 组装完整的 Triton 加速 Block
# ==========================================
class TritonLlama3Block(nn.Module):
    def __init__(self, dim, hidden_dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        # 权重定义
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
        # TODO 1: 接入 Triton RMSNorm 替换原生 Norm
        h = triton_rmsnorm(x, self.norm1_weight)
        
        # QKV 投影并变维 (batch, seq, n_heads, head_dim)
        batch_size, seq_len, _ = h.shape
        q = self.attn_q(h).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.attn_k(h).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.attn_v(h).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # TODO 2: 接入 Triton 融合 RoPE 处理 q 和 k
        q, k = triton_rope(q, k, cos, sin)
        
        # TODO 3: 接入 Triton Flash Attention
        attn_output = triton_flash_attn(q, k, v)
        
        # 恢复形状并输出投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        h = x + self.attn_o(attn_output)
        
        # TODO 4: 接入 Triton MLP 部分
        normed_h = triton_rmsnorm(h, self.norm2_weight)
        mlp_out = triton_swiglu(normed_h, self.mlp_gate.weight, self.mlp_up.weight, self.mlp_down.weight)
        out = h + mlp_out
        
        return out

#  # 端到端性能测试
# import time

# def run_end_to_end_benchmark():
#     if not torch.cuda.is_available():
#         print("⏭️ 无 GPU，跳过测试")
#         return
    
#     # 模拟 LLaMA-3 的一个标准层配置
#     dim = 4096
#     hidden_dim = 14336
#     n_heads = 32
#     batch, seq = 2, 2048
    
#     triton_block = TritonLlama3Block(dim, hidden_dim, n_heads).cuda().half()
#     x = torch.randn(batch, seq, dim, device='cuda', dtype=torch.float16)
    
#     # 模拟 cos 和 sin
#     head_dim = dim // n_heads
#     cos = torch.randn(seq, head_dim // 2, device='cuda', dtype=torch.float16)
#     sin = torch.randn(seq, head_dim // 2, device='cuda', dtype=torch.float16)
    
#     print(" 开始运行端到端 Benchmark (Warmup 10 次，记录 50 次)...")
#     # Warmup
#     for _ in range(10):
#         _ = triton_block(x, cos, sin)
#     torch.cuda.synchronize()
    
#     # 测试 Triton 整合版的耗时
#     start = time.time()
#     for _ in range(50):
#         _ = triton_block(x, cos, sin)
#     torch.cuda.synchronize()
#     triton_time = (time.time() - start) / 50.0 * 1000 # ms
    
#     print(f"✅ 全 Triton 加速的 LLaMA-3 Block 单层前向延迟: {triton_time:.2f} ms")
#     print(" 通过算子融合和 SRAM 内计算，Triton 实现显著降低了 Memory Bound 操作的开销。")





```

### 解析

**1. TODO 1: 接入第一层归一化到前向路径**
- **实现方式**：
  ```python
  h = triton_rmsnorm(x, self.norm1_weight)
  ```
- **关键点**：这是 Attention 前的第一次归一化，使用 Triton 融合算子替代 PyTorch 原生实现
- **技术细节**：
  - `triton_rmsnorm` 在 SRAM 中完成归一化计算，避免中间张量的 HBM 读写
  - 输入 `x` 形状：`(batch, seq, dim)`
  - 输出 `h` 形状：`(batch, seq, dim)`
  - `self.norm1_weight` 是可学习的缩放参数，形状为 `(dim,)`

**2. TODO 2: 接入位置编码处理到 q / k 路径**
- **实现方式**：
  ```python
  q, k = triton_rope(q, k, cos, sin)
  ```
- **关键点**：对 Query 和 Key 应用旋转位置编码，使用 Triton 融合算子实现 in-place 操作
- **技术细节**：
  - `q` 和 `k` 形状：`(batch, n_heads, seq, head_dim)`
  - `cos` 和 `sin` 是预计算的旋转矩阵，形状为 `(seq, head_dim // 2)`
  - Triton RoPE 算子在 SRAM 中完成旋转操作，避免额外的内存分配
  - 返回的 `q` 和 `k` 已经应用了位置编码

**3. TODO 3: 接入注意力主干到前向路径**
- **实现方式**：
  ```python
  attn_output = triton_flash_attn(q, k, v)
  ```
- **关键点**：使用 Flash Attention 算法计算注意力，避免存储完整的注意力矩阵
- **技术细节**：
  - 输入形状：`q`, `k`, `v` 均为 `(batch, n_heads, seq, head_dim)`
  - 输出形状：`(batch, n_heads, seq, head_dim)`
  - Flash Attention 使用分块计算和 Online Softmax，显存占用从 O(seq²) 降低到 O(seq)
  - 在 SRAM 中完成注意力计算，最小化 HBM 访问次数

**4. TODO 4: 接入 MLP 到残差分支**
- **实现方式**：
  ```python
  normed_h = triton_rmsnorm(h, self.norm2_weight)
  mlp_out = triton_swiglu(normed_h, self.mlp_gate.weight, self.mlp_up.weight, self.mlp_down.weight)
  out = h + mlp_out
  ```
- **关键点**：使用 Triton 融合算子实现 MLP 层，包括归一化、SwiGLU 激活和残差连接
- **技术细节**：
  - `triton_rmsnorm(h, self.norm2_weight)`：对 Attention 输出进行归一化
  - `triton_swiglu`：融合了 Gate 投影、Up 投影、SwiGLU 激活和 Down 投影
  - SwiGLU 公式：`SwiGLU(x) = (Swish(x @ W_gate) ⊙ (x @ W_up)) @ W_down`
  - 融合算子避免了中间激活张量的存储，显著降低显存占用
  - 残差连接：`out = h + mlp_out`，保持梯度流动

**真实 Triton 集成模板**
- 这里的 `triton_rmsnorm / triton_rope / triton_flash_attn / triton_swiglu` 默认是 reference adapter，便于 Notebook 独立运行。
- 如果前序章节的真实 Triton kernel 已经在当前 runtime 中加载，可以直接把这些 adapter 替换成真实调用。
- 集成 RMSNorm 时，重点检查 `shape / stride / eps` 是否与前序 kernel 的约定一致。
- 集成 RoPE 和 FlashAttention 时，重点检查 `transpose`、`causal`、`layout` 是否与 kernel 入口对齐。
- 集成 MLP 时，重点检查融合边界：是只融合激活，还是把投影和激活一起封装。

```python
# 示例：如果前序 kernel 已经在当前 runtime 中可用
# h = triton_rmsnorm_real(x, self.norm1_weight)
# q, k = triton_rope_real(q, k, cos, sin)
# attn_output = triton_flash_attn_real(q, k, v)
# mlp_out = triton_swiglu_real(normed_h, self.mlp_gate.weight, self.mlp_up.weight, self.mlp_down.weight)
```

**性能测试建议**
- `run_end_to_end_benchmark()` 建议保留并按需取消注释，而不是长期注释掉。
- 先跑功能正确性，再跑 benchmark。
- 对比时至少看三项：前向时延、HBM 访问开销、Triton 与原生 PyTorch 的加速比。

**工程优化要点**

- **算子融合**：把 RMSNorm / RoPE / FlashAttention / MLP 接回统一 forward，减少中间张量。
- **接口兼容**：reference adapter 只用于 notebook 独立运行，真实环境可直接替换为前序 kernel。
- **真实集成**：重点检查 shape、stride、layout 和 causal 约定是否一致。
- **性能关注**：主要看前向时延和显存峰值，不再展开太多应用场景说明。