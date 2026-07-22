# 40. FP8 and KV Cache Quantization | FP8 与 KV Cache 量化
**难度：** Hard | **环境：** GPU required | **标签：** `量化`, `FP8`, `KV Cache` | **目标人群：** 推理部署与系统工程

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/40_FP8_and_KV_Cache_Quantization.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


先把权重量化看清，再看 FP8 推理和 KV Cache 量化如何一起压缩推理成本、带宽成本和长上下文缓存成本。

**关键词：** `FP8`, `KV cache quantization`, `deployment`

## 前置阅读

**导语：** 先看权重量化、PagedAttention 和显存模型，再看 FP8 与 KV Cache 量化会更容易。

- [39. GPTQ and AWQ Weight Quantization | GPTQ 与 AWQ 权重量化](../02_PyTorch_Algorithms/39_GPTQ_and_AWQ_Weight_Quantization.md)
- [25. Quantization W8A16 | W8A16 量化](../02_PyTorch_Algorithms/25_Quantization_W8A16.md)
- [22. vLLM PagedAttention | vLLM PagedAttention](../02_PyTorch_Algorithms/22_vLLM_PagedAttention.md)
- [P1: 06. VRAM Calculation and ZeRO | 显存计算与 ZeRO 优化](../01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.md)

## 相关阅读

**导语：** FP8 与 KV Cache 量化之后，可以继续看 KV cache 调度和通信 profiling。

- [41. KV Cache Scheduling | KV Cache 调度](../02_PyTorch_Algorithms/41_KV_Cache_Scheduling.md)
- [42. Communication Profiling with NCCL | NCCL 通信性能剖析](../02_PyTorch_Algorithms/42_Communication_Profiling_with_NCCL.md)
- [P1: 14. FlashAttention Memory Model | FlashAttention 显存模型](../01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.md)

### Step 1: 原理与痛点

FP8 适合把推理过程里的部分张量压到更低位宽，从而减轻存储和带宽压力；KV Cache 量化则是把长上下文推理里的缓存成本继续压下来。两者都不是单纯追求更小，而是在“哪些张量值得压、压到什么程度、恢复后会不会明显伤输出”这三个问题上找平衡。

### Step 2: 代码实现框架

下面的代码会把 FP8 近似转换和 KV Cache 量化拆成独立动作，再做一个最小推理链路对比。你需要关注的是“哪一部分适合量化、量化后如何恢复、恢复后对吞吐和缓存开销有什么影响”。

### Step 3: 核心机制

这一页的重点是理解：推理里并不是所有张量都要以同样方式量化。权重、激活、KV cache 各自有不同的容忍度和收益边界，所以要按“谁最占空间、谁最影响带宽、谁最容易保精度”拆开看。

### Step 4: 动手实战

**要求**：请补全下方 `FP8KVCacheSim`，实现一个极简版的 FP8 / KV Cache 量化模拟器。先把 FP8 近似和 KV Cache 量化的基本链路跑通，再考虑更复杂的精度与吞吐权衡。

```python
import torch
import torch.nn as nn


class FP8KVCacheSim(nn.Module):
    """A tiny FP8 + KV cache quantization simulator.

    The goal is to keep the teaching loop explicit:
    quantify what can be compressed, store the scale, restore it,
    and inspect how much error we introduced.
    """

    def __init__(self, fp8_qmax: int = 127, kv_group_size: int = 64, eps: float = 1e-8):
        super().__init__()
        self.fp8_qmax = fp8_qmax
        self.kv_group_size = kv_group_size
        self.eps = eps

        self.register_buffer("fp8_q", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("fp8_scale", torch.tensor(1.0), persistent=False)
        self.register_buffer("kv_q", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("kv_scale", torch.empty(0), persistent=False)
        self.fp8_shape = None
        self.kv_shape = None

    # TODO 1: 对称量化一个张量并返回量化值与 scale
    def _sym_quantize(self, x: torch.Tensor, qmax: int):
        raise NotImplementedError

    # TODO 2: 把对称量化结果恢复回浮点
    def _sym_dequantize(self, q: torch.Tensor, scale: torch.Tensor):
        raise NotImplementedError

    # TODO 3: 量化 FP8 近似张量
    def quantize_fp8(self, x: torch.Tensor):
        raise NotImplementedError

    # TODO 4: 恢复 FP8 近似张量
    def dequantize_fp8(self):
        raise NotImplementedError

    # TODO 5: 量化 KV cache
    def quantize_kv_cache(self, kv_cache: torch.Tensor):
        raise NotImplementedError

    # TODO 6: 恢复 KV cache
    def dequantize_kv_cache(self):
        raise NotImplementedError

    # TODO 7: 记录一次量化实验
    def fit(self, hidden_states: torch.Tensor, kv_cache: torch.Tensor | None = None):
        raise NotImplementedError

    # TODO 8: 前向返回恢复后的张量
    def forward(self, hidden_states: torch.Tensor, kv_cache: torch.Tensor | None = None):
        raise NotImplementedError

    # TODO 9: 计算重构误差
    def mse(self, original: torch.Tensor, restored: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
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
# TODO：下面是题目区的参考实现。

import torch
import torch.nn as nn


class FP8KVCacheSim(nn.Module):
    """A tiny FP8 + KV cache quantization simulator.

    The goal is to keep the teaching loop explicit:
    quantify what can be compressed, store the scale, restore it,
    and inspect how much error we introduced.
    """

    def __init__(self, fp8_qmax: int = 127, kv_group_size: int = 64, eps: float = 1e-8):
        super().__init__()
        self.fp8_qmax = fp8_qmax
        self.kv_group_size = kv_group_size
        self.eps = eps

        self.register_buffer("fp8_q", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("fp8_scale", torch.tensor(1.0), persistent=False)
        self.register_buffer("kv_q", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("kv_scale", torch.empty(0), persistent=False)
        self.fp8_shape = None
        self.kv_shape = None

    def _sym_quantize(self, x: torch.Tensor, qmax: int):
        x = x.detach().float()
        absmax = torch.max(torch.abs(x))
        scale = qmax / absmax.clamp_min(self.eps)
        q = torch.clamp(torch.round(x * scale), -qmax, qmax).to(torch.int8)
        return q, scale

    def _sym_dequantize(self, q: torch.Tensor, scale: torch.Tensor):
        return q.to(scale.dtype) / scale.clamp_min(self.eps)

    def quantize_fp8(self, x: torch.Tensor):
        """Approximate FP8 with a symmetric low-precision quantizer."""
        q, scale = self._sym_quantize(x, self.fp8_qmax)
        self.fp8_q = q
        self.fp8_scale = scale
        self.fp8_shape = tuple(x.shape)
        return q, scale

    def dequantize_fp8(self):
        if self.fp8_shape is None:
            raise RuntimeError("Call quantize_fp8() before dequantize_fp8().")
        return self._sym_dequantize(self.fp8_q, self.fp8_scale)

    def quantize_kv_cache(self, kv_cache: torch.Tensor):
        """Quantize KV cache with a small group-wise scale."""
        kv = kv_cache.detach().float()
        if kv.ndim < 2:
            raise ValueError("KV cache should have at least 2 dimensions.")

        last_dim = kv.size(-1)
        n_groups = (last_dim + self.kv_group_size - 1) // self.kv_group_size
        qkv = torch.zeros_like(kv, dtype=torch.int8)
        scales = torch.zeros(kv.shape[:-1] + (n_groups,), dtype=kv.dtype, device=kv.device)

        flat = kv.reshape(-1, last_dim)
        flat_q = qkv.reshape(-1, last_dim)
        flat_scale = scales.reshape(-1, n_groups)

        for row in range(flat.size(0)):
            for g in range(n_groups):
                start = g * self.kv_group_size
                end = min(start + self.kv_group_size, last_dim)
                chunk = flat[row, start:end]
                if chunk.numel() == 0:
                    continue
                q, scale = self._sym_quantize(chunk, self.fp8_qmax)
                flat_q[row, start:end] = q
                flat_scale[row, g] = scale

        self.kv_q = qkv
        self.kv_scale = scales
        self.kv_shape = tuple(kv.shape)
        return qkv, scales

    def dequantize_kv_cache(self):
        if self.kv_shape is None:
            raise RuntimeError("Call quantize_kv_cache() before dequantize_kv_cache().")

        kv = self.kv_q.to(self.kv_scale.dtype)
        last_dim = kv.size(-1)
        n_groups = self.kv_scale.size(-1)
        flat = kv.reshape(-1, last_dim)
        flat_out = torch.zeros_like(flat, dtype=self.kv_scale.dtype)
        flat_scale = self.kv_scale.reshape(-1, n_groups)

        for row in range(flat.size(0)):
            for g in range(n_groups):
                start = g * self.kv_group_size
                end = min(start + self.kv_group_size, last_dim)
                scale = flat_scale[row, g]
                flat_out[row, start:end] = self._sym_dequantize(flat[row, start:end], scale)

        return flat_out.reshape(self.kv_shape)

    def fit(self, hidden_states: torch.Tensor, kv_cache: torch.Tensor | None = None):
        """Quantize the tensors we want to inspect and cache the results."""
        self.quantize_fp8(hidden_states)
        if kv_cache is not None:
            self.quantize_kv_cache(kv_cache)
        return self

    def forward(self, hidden_states: torch.Tensor, kv_cache: torch.Tensor | None = None):
        """Quantize-dequantize the given tensors and return the restored result."""
        fp8_q, fp8_scale = self._sym_quantize(hidden_states, self.fp8_qmax)
        fp8_restored = self._sym_dequantize(fp8_q, fp8_scale)

        if kv_cache is None:
            return fp8_restored

        kv_q, kv_scale = self.quantize_kv_cache(kv_cache)
        kv_restored = self.dequantize_kv_cache()
        return fp8_restored, kv_restored

    def mse(self, original: torch.Tensor, restored: torch.Tensor) -> torch.Tensor:
        return torch.mean((original.float() - restored.float()) ** 2)

```

### 解析
- FP8 关注推理张量的带宽与存储成本，KV cache 量化关注长上下文缓存成本。
- 这一页的核心是：哪些张量值得压、压到什么程度、恢复后是否还能接受。
- 先把“量化 / 恢复 / 误差”闭环跑通，再考虑更复杂的推理优化。
### 测试

```python
import torch

torch.manual_seed(0)
sim = FP8KVCacheSim(fp8_qmax=127, kv_group_size=4)
hidden = torch.randn(2, 8)
kv = torch.randn(2, 3, 8)
sim.fit(hidden, kv)
hidden_restore = sim.dequantize_fp8()
kv_restore = sim.dequantize_kv_cache()
out_hidden, out_kv = sim.forward(hidden, kv)
assert hidden_restore.shape == hidden.shape
assert kv_restore.shape == kv.shape
assert out_hidden.shape == hidden.shape
assert out_kv.shape == kv.shape
print('✅ FP8KVCacheSim 测试通过')
```
