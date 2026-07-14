# 18. Memory Profiling and Optimization | 显存分析与优化

**难度：** Medium | **环境：** GPU optional | **标签：** `PyTorch`, `显存`, `训练优化` | **目标人群：** Part 2-4 前置补课者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/18_Memory_Profiling_and_Optimization.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本页聚焦显存分析和优化的最小判断链：先拆参数、梯度、优化器状态、激活和临时缓冲区，再决定是缩 batch、做 accumulation，还是上 checkpoint，不把显存优化写成经验清单。

**关键词：** `memory`, `checkpoint`, `accumulation`

## 前置阅读
**导语：** 先看 0E 组页，把性能分析和显存账本的边界对齐，再进入这一页会更顺。
- [17. PyTorch Profiling Basics | PyTorch 性能分析基础](./17_PyTorch_Profiling_Basics.md)
- [0E 组页](./0E.md)
- [03. GPU Architecture and Memory | GPU 物理架构、内存层级与核心硬件单元](../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.md)
- [06. VRAM Calculation and ZeRO | 显存计算与 ZeRO 优化](../01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.md)

## 相关阅读
**导语：** 本页先把显存账本和优化手段的最小判断讲清楚；如果想继续看调试技巧，再顺着看下面这一页。
- [19. Debugging and Anomaly Localization | 调试与异常定位](./19_Debugging_and_Anomaly_Localization.md)

## Q1：显存账本先分哪几层？

先把常驻显存（参数、梯度、优化器状态）和动态显存（激活、临时 buffer）拆开；只看 `allocated` 不够，还要知道峰值是在哪一层堆起来的。


```python
import torch
import torch.nn as nn


def pretty_mb(nbytes):
    return f"{nbytes / 1024**2:.2f} MB"


class TinyLedgerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
        )

    def forward(self, x):
        return self.net(x)


model = TinyLedgerNet()
param_count = sum(p.numel() for p in model.parameters())
param_bytes = param_count * 4
grad_bytes = param_count * 4
adam_state_bytes = param_count * 8

batch_size, seq_len, hidden_size = 16, 64, 512
activation_bytes = batch_size * seq_len * hidden_size * 4

print(f"parameter count: {param_count:,}")
print(f"parameter bytes (fp32): {pretty_mb(param_bytes)}")
print(f"gradient bytes (fp32):   {pretty_mb(grad_bytes)}")
print(f"adam state bytes:        {pretty_mb(adam_state_bytes)}")
print(f"one activation block:    {pretty_mb(activation_bytes)}")

```

## Q2：什么时候该缩 batch，什么时候该做梯度累积？

两者都在降显存压力，但只有梯度累积能保住有效 batch；代价是更多次前向/反向。


```python
configs = [
    {"name": "baseline", "micro_batch": 16, "accum_steps": 1, "world_size": 1},
    {"name": "accumulated", "micro_batch": 4, "accum_steps": 4, "world_size": 1},
]

seq_len, hidden_size, dtype_bytes = 64, 512, 2

def mb(nbytes):
    return nbytes / 1024**2

for cfg in configs:
    effective_batch = cfg["micro_batch"] * cfg["accum_steps"] * cfg["world_size"]
    activation_bytes = cfg["micro_batch"] * seq_len * hidden_size * dtype_bytes
    print(
        f"{cfg['name']}: micro={cfg['micro_batch']}, accum={cfg['accum_steps']}, "
        f"effective={effective_batch}, activation_peak={mb(activation_bytes):.2f} MB"
    )

print("same effective batch, but activation peak follows micro batch")

```

## Q3：混合精度和梯度检查点分别省什么？

混合精度主要省 dtype bytes；检查点主要省要保存的激活。它们常常是叠加关系，不是互相替代。


```python
import math

layers = 12
batch_size, seq_len, hidden_size = 8, 32, 512

def activation_mb(dtype_bytes, checkpoint_factor=1):
    saved_layers = math.ceil(layers / checkpoint_factor)
    return saved_layers * batch_size * seq_len * hidden_size * dtype_bytes / 1024**2

rows = [
    ("fp32 / no checkpoint", activation_mb(4, 1)),
    ("bf16 / no checkpoint", activation_mb(2, 1)),
    ("fp32 / checkpoint every 2 blocks", activation_mb(4, 2)),
    ("bf16 / checkpoint every 2 blocks", activation_mb(2, 2)),
]

for name, memory in rows:
    print(f"{name:<32} {memory:>6.2f} MB")

print("checkpointing mainly reduces saved activations; bf16 mainly cuts dtype bytes")

```

## Q4：显存泄漏怎么排查？

先看是不是把 Tensor、loss 或中间激活长期挂在容器里；再看是不是该用 `detach()`、`no_grad()` 或显式清理。


```python
import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(0)
model = nn.Linear(4, 2)
x = torch.randn(8, 4)
y = torch.randint(0, 2, (8,))

loss_bucket = []
for _ in range(2):
    out = model(x)
    loss = F.cross_entropy(out, y)
    loss_bucket.append(loss)

scalar_bucket = []
for _ in range(2):
    out = model(x)
    loss = F.cross_entropy(out, y)
    scalar_bucket.append(loss.item())

with torch.no_grad():
    pred = model(x)

detached = model(x).detach()

print(f"keep tensor: type={type(loss_bucket[0]).__name__}, grad_fn={loss_bucket[0].grad_fn is not None}")
print(f"keep scalar: type={type(scalar_bucket[0]).__name__}, value={scalar_bucket[0]:.4f}")
print(f"no_grad output requires_grad={pred.requires_grad}")
print(f"detach output requires_grad={detached.requires_grad}, grad_fn={detached.grad_fn}")
print("store tensors only when you really need the graph")

```

## Q5：常驻显存和动态显存，谁更像当前瓶颈？

先把常驻显存和动态显存分开看，再判断当前瓶颈是参数规模、优化器状态，还是训练过程里的激活峰值；这样才能知道是改模型大小，还是改训练策略。


```python
def dominant_memory_term(param_mb, grad_mb, optim_mb, activation_mb):
    items = {
        'parameters': param_mb,
        'gradients': grad_mb,
        'optimizer_state': optim_mb,
        'activations': activation_mb,
    }
    return max(items, key=items.get), items


winner, items = dominant_memory_term(param_mb=1200, grad_mb=1200, optim_mb=4800, activation_mb=1800)
print('items:', items)
print('dominant:', winner)
# 输出示例: optimizer_state 往往最先吃满显存

```

## Q6：batch、gradient accumulation、checkpoint 的决策顺序怎么定？

先看 batch 是否还能缩，再看 accumulation 是否足够保住有效 batch，最后才把 checkpoint 当作进一步降峰值的兜底方案；顺序定清楚，才不会把三者混成一团。


```python
def estimate_step_budget(param_mb, grad_mb, optim_mb, activation_mb, can_fit):
    total_mb = param_mb + grad_mb + optim_mb + activation_mb
    if not can_fit:
        return 'reduce_batch_or_checkpoint', total_mb
    if activation_mb >= optim_mb:
        return 'checkpoint', total_mb
    if grad_mb + optim_mb > param_mb * 2:
        return 'gradient_accumulation', total_mb
    return 'keep_batch_and_profile', total_mb


strategy, total = estimate_step_budget(param_mb=1200, grad_mb=1200, optim_mb=4800, activation_mb=1800, can_fit=False)
print('strategy:', strategy)
print('estimated_total_mb:', total)
# 输出示例: strategy -> reduce_batch_or_checkpoint; estimated_total_mb -> 9000

```
