# 39. GPTQ and AWQ Weight Quantization | GPTQ 与 AWQ 权重量化
**难度：** Hard | **环境：** GPU required | **标签：** `量化`, `GPTQ`, `AWQ` | **目标人群：** 模型压缩与部署工程

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/39_GPTQ_and_AWQ_Weight_Quantization.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


先把 W8A16 和 QLoRA 的量化直觉看清，再看 GPTQ 和 AWQ 会更容易理解权重量化里“校准数据怎么选、误差怎么分配、敏感通道怎么保护”这三件事。

**关键词：** `GPTQ`, `AWQ`, `weight quantization`

## 前置阅读

**导语：** 先把 W8A16、QLoRA 和量化理论理顺，再看 GPTQ / AWQ 会更容易。

- [25. Quantization W8A16 | W8A16 量化](../02_PyTorch_Algorithms/25_Quantization_W8A16.md)
- [26. QLoRA and 4bit Quantization | QLoRA 与 4-bit 量化](../02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization.md)
- [P1: 21. Quantization Theory and INT4/INT8 | 量化理论与 INT4/INT8](../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.md)
- [P1: 01. Data Types and Precision | 大模型的数据格式与混合精度](../01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.md)

## 相关阅读

**导语：** 权重量化之后，可以继续看 FP8、KV Cache Quantization 和 cache scheduling。

- [40. FP8 and KV Cache Quantization | FP8 与 KV Cache 量化](../02_PyTorch_Algorithms/40_FP8_and_KV_Cache_Quantization.md)
- [41. KV Cache Scheduling | KV Cache 调度](../02_PyTorch_Algorithms/41_KV_Cache_Scheduling.md)
- [P1: 03. GPU Architecture and Memory | GPU 物理架构与内存层级](../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.md)

### Step 1: 原理与痛点

W8A16 解决的是“先把权重压成 8 bit，再保留激活为高精度”的实用方案，而 GPTQ / AWQ 则进一步把权重量化的校准数据、分组策略和误差最小化过程细化出来。它们要解决的核心问题是：如何在更低比特下尽量保住模型输出质量，并把误差尽量压在不敏感的权重上。

### Step 2: 代码实现框架

下面的代码会把权重量化拆成几个小动作：收集校准样本、统计 scale、做分组量化、反量化恢复，以及对比量化前后的误差。你需要关注的不是某个特定库的细节，而是“校准 -> 分组 -> 量化 -> 恢复 -> 评估”的流程。

### Step 3: 核心机制

GPTQ 和 AWQ 的差别可以先从直觉上理解：GPTQ 更偏向后训练量化里的重构误差最小化，AWQ 更强调激活感知和对少数敏感通道的保护。真正决定收益的，不只是 bit 数，而是校准方式、分组方式和误差分配是否合理。

### Step 4: 动手实战

**要求**：请补全下方 `WeightQuantizerSim`，实现一个极简版的 GPTQ / AWQ 权重量化模拟器。先把校准、分组、量化和恢复这条链路跑通，再考虑更复杂的工程优化。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightQuantizerSim(nn.Module):
    """A tiny GPTQ / AWQ style weight quantizer simulator.

    This simulator keeps the teaching focus on the core loop:
    calibration -> grouping -> quantization -> dequantization -> error check.
    For AWQ, a small ratio of sensitive channels is kept in full precision
    to reflect the "activation-aware protection" intuition.
    """

    def __init__(self, bits: int = 4, group_size: int = 32, method: str = "gptq", protect_ratio: float = 0.05, eps: float = 1e-8):
        super().__init__()
        if bits < 2:
            raise ValueError("bits must be >= 2")
        self.bits = bits
        self.group_size = group_size
        self.method = method.lower()
        self.protect_ratio = protect_ratio
        self.eps = eps
        self.qmax = 2 ** (bits - 1) - 1

        self.register_buffer("qweight", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("scales", torch.empty(0), persistent=False)
        self.register_buffer("protected_weight", torch.empty(0), persistent=False)
        self.register_buffer("protected_mask", torch.empty(0, dtype=torch.bool), persistent=False)
        self.register_buffer("importance", torch.empty(0), persistent=False)
        self.weight_shape = None

    # TODO 1: 根据激活统计通道重要性
    def _collect_importance(self, activations: torch.Tensor, in_features: int) -> torch.Tensor:
        raise NotImplementedError

    # TODO 2: 完成校准、分组、量化和保护通道记录
    def fit(self, weight: torch.Tensor, activations: torch.Tensor | None = None) -> "WeightQuantizerSim":
        raise NotImplementedError

    # TODO 3: 把量化权重恢复成浮点权重
    def dequantize(self) -> torch.Tensor:
        raise NotImplementedError

    # TODO 4: 用恢复后的权重做线性前向
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # TODO 5: 计算重构误差
    def mse(self, weight: torch.Tensor) -> torch.Tensor:
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
import torch.nn.functional as F


class WeightQuantizerSim(nn.Module):
    """A tiny GPTQ / AWQ style weight quantizer simulator.

    This simulator keeps the teaching focus on the core loop:
    calibration -> grouping -> quantization -> dequantization -> error check.
    For AWQ, a small ratio of sensitive channels is kept in full precision
    to reflect the "activation-aware protection" intuition.
    """

    def __init__(self, bits: int = 4, group_size: int = 32, method: str = "gptq", protect_ratio: float = 0.05, eps: float = 1e-8):
        super().__init__()
        if bits < 2:
            raise ValueError("bits must be >= 2")
        self.bits = bits
        self.group_size = group_size
        self.method = method.lower()
        self.protect_ratio = protect_ratio
        self.eps = eps
        self.qmax = 2 ** (bits - 1) - 1

        self.register_buffer("qweight", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("scales", torch.empty(0), persistent=False)
        self.register_buffer("protected_weight", torch.empty(0), persistent=False)
        self.register_buffer("protected_mask", torch.empty(0, dtype=torch.bool), persistent=False)
        self.register_buffer("importance", torch.empty(0), persistent=False)
        self.weight_shape = None

    def _collect_importance(self, activations: torch.Tensor, in_features: int) -> torch.Tensor:
        act = activations.detach().float()
        if act.ndim == 1:
            importance = act.abs()
        else:
            reduce_dims = tuple(range(act.ndim - 1))
            importance = act.pow(2).mean(dim=reduce_dims).sqrt()
        if importance.numel() != in_features:
            raise ValueError(
                f"Calibration importance dim mismatch: expected {in_features}, got {importance.numel()}"
            )
        return importance

    def fit(self, weight: torch.Tensor, activations: torch.Tensor | None = None) -> "WeightQuantizerSim":
        """Quantize a 2D weight matrix and cache the quantized state."""
        w = weight.detach().float()
        if w.ndim != 2:
            raise ValueError("WeightQuantizerSim only supports 2D linear weights.")

        out_features, in_features = w.shape
        self.weight_shape = (out_features, in_features)

        if activations is None:
            importance = torch.ones(in_features, device=w.device, dtype=w.dtype)
        else:
            importance = self._collect_importance(activations, in_features)
        self.importance = importance

        n_groups = (in_features + self.group_size - 1) // self.group_size
        qweight = torch.zeros_like(w, dtype=torch.int8)
        scales = torch.zeros((out_features, n_groups), dtype=w.dtype, device=w.device)
        protected_weight = torch.zeros_like(w)
        protected_mask = torch.zeros_like(w, dtype=torch.bool)

        for row in range(out_features):
            for g in range(n_groups):
                start = g * self.group_size
                end = min(start + self.group_size, in_features)
                wg = w[row, start:end]
                ig = importance[start:end]

                if wg.numel() == 0:
                    continue

                if self.method == "awq":
                    # Protect the most sensitive channels in each group.
                    k = max(1, int(round(wg.numel() * self.protect_ratio)))
                    k = min(k, wg.numel())
                    topk = torch.topk(ig, k=k, largest=True).indices
                    mask = torch.zeros_like(ig, dtype=torch.bool)
                    mask[topk] = True

                    protected_mask[row, start:end] = mask
                    protected_weight[row, start:end][mask] = wg[mask]

                    base = wg[~mask]
                    base_imp = ig[~mask]
                    if base.numel() == 0:
                        base = wg
                        base_imp = ig

                    # Use a more conservative scale for the remaining channels.
                    weight_factor = base_imp / (base_imp.max() + self.eps)
                    scale = ((base.abs() * (0.5 + 0.5 * weight_factor)).max() / self.qmax).clamp_min(self.eps)

                    q_group = torch.zeros_like(wg, dtype=torch.int8)
                    q_group[~mask] = torch.clamp(torch.round(base / scale), -self.qmax, self.qmax).to(torch.int8)
                else:
                    # GPTQ-style intuition: use activation importance to bias the calibration scale.
                    weight_factor = ig / (ig.max() + self.eps)
                    scale = ((wg.abs() * weight_factor).max() / self.qmax).clamp_min(self.eps)
                    q_group = torch.clamp(torch.round(wg / scale), -self.qmax, self.qmax).to(torch.int8)

                qweight[row, start:end] = q_group
                scales[row, g] = scale

        self.qweight = qweight
        self.scales = scales
        self.protected_weight = protected_weight
        self.protected_mask = protected_mask
        return self

    def dequantize(self) -> torch.Tensor:
        if self.weight_shape is None:
            raise RuntimeError("Call fit() before dequantize().")

        out_features, in_features = self.weight_shape
        n_groups = self.scales.size(1)
        weight = torch.zeros((out_features, in_features), dtype=self.scales.dtype, device=self.scales.device)

        for row in range(out_features):
            for g in range(n_groups):
                start = g * self.group_size
                end = min(start + self.group_size, in_features)
                scale = self.scales[row, g]
                q_group = self.qweight[row, start:end].to(self.scales.dtype)
                dequant = q_group * scale
                if self.protected_mask.numel() > 0:
                    protected = self.protected_mask[row, start:end]
                    if protected.any():
                        dequant = dequant.clone()
                        dequant[protected] = self.protected_weight[row, start:end][protected]
                weight[row, start:end] = dequant

        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight_shape is None:
            raise RuntimeError("Call fit() before forward().")
        weight = self.dequantize().to(x.dtype)
        return F.linear(x, weight)

    def mse(self, weight: torch.Tensor) -> torch.Tensor:
        """A small utility to inspect reconstruction error."""
        recon = self.dequantize().to(weight.dtype)
        return torch.mean((weight - recon) ** 2)

```

### 解析
- GPTQ / AWQ 的差异重点在校准方式和敏感通道保护。
- 量化不是单纯“压得更低”，而是要平衡误差、恢复和推理稳定性。
- 先把“校准 -> 分组 -> 恢复 -> 误差”跑通，再讨论更复杂的实现。
### 测试

```python
import torch

torch.manual_seed(0)
weight = torch.randn(4, 8)
acts = torch.randn(16, 8)
sim = WeightQuantizerSim(bits=4, group_size=4, method='awq').fit(weight, acts)
restored = sim.dequantize()
y = sim.forward(torch.randn(2, 8))
assert restored.shape == weight.shape
assert y.shape == (2, 4)
assert float(sim.mse(weight)) >= 0.0
print('✅ WeightQuantizerSim 测试通过')
```
