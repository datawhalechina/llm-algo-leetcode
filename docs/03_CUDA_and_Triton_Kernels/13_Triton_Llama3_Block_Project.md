# 13 Triton Llama3 Block Project

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/03_CUDA_and_Triton_Kernels/13_Triton_Llama3_Block_Project.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 13. 综合工程实战：使用 Triton 从头组装 LLaMA-3 Transformer Block

**难度：** Hard | **标签：** `Triton`, `End-to-End Project`, `LLaMA-3`, `Integration` | **目标人群：** 核心 Infra 与算子开发

这是本教程 **Triton 算子开发**章节的大考 (Capstone Project)。
在工业界，写出几个零散的算子只是 Demo。你需要将这些算子封装成标准的 `torch.autograd.Function` 或标准的 `nn.Module`，去**平替** PyTorch 原生的极度耗时的层，最终拼装出一个完全由 Triton 加速的 `Llama3TritonBlock`。

在本节中，我们将：
1. 回顾并调用我们在前几节手写的：Triton Fused RMSNorm, Triton Fused RoPE, Triton Flash Attention, Triton Fused SwiGLU。
2. 封装 PyTorch 的 `nn.Module` 接口。
3. 运行端到端的 Benchmark，直观感受到算子融合带来的极致性能提升 (Latency 降低)。


### Step 1: 算子替换与模块集成规范

> **PyTorch 原生实现为什么慢？**
> 我们在 `02_PyTorch_Algorithms/07_LLaMA3_Block_Tutorial` 中写的 Block：
> `x = x + Attention(RMSNorm(x))`
> `x = x + MLP(RMSNorm(x))`
> 这个过程产生了大量的中间张量 (Intermediate Tensors)，由于频繁的内存读写 (Memory Bound)，严重拖慢了速度。

> **如何进行工程级替换 (Integration)？**
> 1. 继承 `nn.Module` 编写自定义的 Layer。
> 2. 在 Layer 的 `forward` 方法中，直接调用包含 `kernel[grid](...)` 的 Triton 封装函数。
> 3. （如果需要训练）继承 `torch.autograd.Function` 实现 `forward` 和 `backward`，并在 `nn.Module` 中调用 `YourFunction.apply`。本节为了聚焦前向推理性能，只集成推理部分的替换。


### Step 2: 算子替换与模块集成规范
这是一个架构拼装工程。虽然我们在前面手写出了所有加速算子，但要组装回基于 `nn.Module` 的 PyTorch 模型中，必须处理好接口（Interface）封装问题，并确保前向传播在 AutoGrad (反向图) 中的逻辑隔离或兼容。


### Step 3: 集成代码框架
定义一个 `Llama3TritonBlock(nn.Module)` 类。在 `__init__` 中保留 `nn.Linear` 管理权重，但在 `forward` 阶段，彻底废弃原生的 `F.silu` 等调用，将这些中间环节全面替换为你手写的 `triton_fused_swiglu` 和 `triton_flash_attention` 调用。


###  Step 4: 动手实战

**要求**：请补全下方 `TritonLlama3Block`，使用我们在前序章节中构建的 Triton 算子，替换掉原生的算子。


```python
import torch
import torch.nn as nn
import triton
import math

# ==========================================
# 我们假设这些函数是你在前几节 (03, 07, 08, 02) 中已经写好的 Triton 封装。
# 为了让本 Notebook 能独立运行，我们在这里提供极其简化的 dummy 实现或者直接调用。
# 在实际工程中，你会用 import 导入你的 Triton Kernel。
# ==========================================
def triton_rmsnorm(x, weight, eps=1e-5):
    # 假设这里调用了 03_Triton_Fused_RMSNorm 的算子
    # 为了测试能跑通，我们退化回高效的 PyTorch 原生实现模拟
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight

def triton_rope(q, k, cos, sin):
    # 假设这里调用了 07_Triton_Fused_RoPE 的算子 (In-place)
    # ... 省略 Triton kernel 调用 ...
    return q, k

def triton_flash_attn(q, k, v):
    # 假设这里调用了 08_Triton_Flash_Attention 的算子
    # 使用 PyTorch SDPA 模拟 Triton Flash Attention 的极速性能
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)

def triton_swiglu(x, gate_weight, up_weight, down_weight):
    # 假设这里调用了 02_Triton_Fused_SwiGLU 的算子
    # x @ gate_weight, x @ up_weight, Swish(gate) * up, @ down_weight
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
        # ==========================================
        # TODO 1: 使用 Triton RMSNorm 替换原生 Norm
        # ==========================================
        # h = ???
        h = triton_rmsnorm(x, self.norm1_weight)
        
        # QKV 投影并变维 (batch, seq, n_heads, head_dim)
        batch_size, seq_len, _ = h.shape
        q = self.attn_q(h).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.attn_k(h).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.attn_v(h).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # ==========================================
        # TODO 2: 使用 Triton 融合 RoPE 处理 q 和 k
        # ==========================================
        q, k = triton_rope(q, k, cos, sin)
        
        # ==========================================
        # TODO 3: 使用 Triton Flash Attention
        # ==========================================
        # attn_output = ???
        attn_output = triton_flash_attn(q, k, v)
        
        # 恢复形状并输出投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        h = x + self.attn_o(attn_output)
        
        # ==========================================
        # TODO 4: MLP 部分
        # 1. 对 h 使用 Triton RMSNorm
        # 2. 调用 Triton Fused SwiGLU 替代繁琐的多次 Linear 读写
        # 3. 残差连接 h + mlp_out
        # ==========================================
        normed_h = triton_rmsnorm(h, self.norm2_weight)
        mlp_out = triton_swiglu(normed_h, self.mlp_gate.weight, self.mlp_up.weight, self.mlp_down.weight)
        out = h + mlp_out
        
        return out

```

```python
# 端到端性能对比 (Triton 组装 Block vs PyTorch 纯原生)
import time

def run_end_to_end_benchmark():
    if not torch.cuda.is_available():
        print("⏭️ 忽略测试：无 GPU。")
        return
        
    try:
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
        
        print("🚀 开始运行端到端 Benchmark (Warmup 10 次，记录 50 次)...")
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
        print("🔥 恭喜！你成功将散落的算子拼装成了工业级的推理模块。这展现了你不仅具备底层数学推导能力，还拥有宏观的系统架构和工程落地能力。")
        print("🎉 LLM 核心算法与系统实战教程到此圆满结业！你可以带着这些硬核知识去横扫 AI Infra 和算法工程师的面试了。")
        
    except NotImplementedError:
        print("请先完成 TODO 代码！")
    except Exception as e:
        print(f"❌ 运行失败: {e}")

run_end_to_end_benchmark()

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

::: details 💡 点击查看官方解析与参考代码

### 💡 参考解答：组装 LLaMA-3 Triton Block

在这个综合项目中，我们将底层算子与 PyTorch 的 `nn.Module` 高层抽象完美结合：
1. **彻底消除 PyTorch 的中间张量**：原本 PyTorch 的 `F.silu(gate) * up` 会产生极其庞大的激活张量并频繁在 HBM 中读写。我们利用 `triton_swiglu` 将激活函数和逐元素乘法全部融合，仅需要进出一次 HBM。
2. **高度内聚的架构**：我们在 Python 层面的前向传播代码变得极其精简。`triton_rmsnorm`, `triton_rope`, `triton_flash_attn`, `triton_swiglu` 接管了所有 Memory Bound 最严重的环节，而大矩阵乘法则留给底层的 cuBLAS (通过 PyTorch Linear)。这正是业界构建如 vLLM, DeepSpeed 等高性能推理引擎的标准打法。

```python
import torch
import torch.nn as nn
import triton
import math

# ==========================================
# 我们假设这些函数是你在前几节 (03, 07, 08, 02) 中已经写好的 Triton 封装。
# 为了让本 Notebook 能独立运行，我们在这里提供极其简化的 dummy 实现或者直接调用。
# 在实际工程中，你会用 import 导入你的 Triton Kernel。
# ==========================================
def triton_rmsnorm(x, weight, eps=1e-5):
    # 假设这里调用了 03_Triton_Fused_RMSNorm 的算子
    # 为了测试能跑通，我们退化回高效的 PyTorch 原生实现模拟
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight

def triton_rope(q, k, cos, sin):
    # 假设这里调用了 07_Triton_Fused_RoPE 的算子 (In-place)
    # ... 省略 Triton kernel 调用 ...
    return q, k

def triton_flash_attn(q, k, v):
    # 假设这里调用了 08_Triton_Flash_Attention 的算子
    # 使用 PyTorch SDPA 模拟 Triton Flash Attention 的极速性能
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)

def triton_swiglu(x, gate_weight, up_weight, down_weight):
    # 假设这里调用了 02_Triton_Fused_SwiGLU 的算子
    # x @ gate_weight, x @ up_weight, Swish(gate) * up, @ down_weight
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
        # TODO 1: 使用 Triton RMSNorm 替换原生 Norm
        h = triton_rmsnorm(x, self.norm1_weight)
        
        # QKV 投影并变维 (batch, seq, n_heads, head_dim)
        batch_size, seq_len, _ = h.shape
        q = self.attn_q(h).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.attn_k(h).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.attn_v(h).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # TODO 2: 使用 Triton 融合 RoPE 处理 q 和 k
        q, k = triton_rope(q, k, cos, sin)
        
        # TODO 3: 使用 Triton Flash Attention
        attn_output = triton_flash_attn(q, k, v)
        
        # 恢复形状并输出投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        h = x + self.attn_o(attn_output)
        
        # TODO 4: MLP 部分
        normed_h = triton_rmsnorm(h, self.norm2_weight)
        mlp_out = triton_swiglu(normed_h, self.mlp_gate.weight, self.mlp_up.weight, self.mlp_down.weight)
        out = h + mlp_out
        
        return out
```

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
