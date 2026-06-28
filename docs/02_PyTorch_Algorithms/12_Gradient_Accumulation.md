# 12. Gradient Accumulation | 梯度累积

**难度：** Medium | **标签：** `训练技巧`, `显存优化`, `PyTorch` | **目标人群：** 模型微调与工程部署

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/12_Gradient_Accumulation.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


在做大模型微调时，显存通常先被 batch size 吃满。梯度累积的核心思路是：把一个大 batch 拆成多个 micro-batch，分多次 backward，最后只 step 一次，从而在不增加峰值显存的前提下，模拟更大的有效 batch。

## 前置

**导语：** 先看 SFT、LoRA 和学习率调度，再进入梯度累积会更容易理解它的用途。
- [Part 2: 09 SFT Training Loop](./09_SFT_Training_Loop.md)
- [Part 2: 11 Warmup-Stable-Decay Scheduler](./11_LR_Schedulers_WSD_Cosine.md)

## 相关阅读

**导语：** 梯度累积之后，最自然的收口就是端到端微调实验。
- [Part 2: 13 End-to-End Fine-Tuning Experiment](./13_End_to_End_Fine_Tuning_Experiment.md)
- [Part 2: 14 RLHF PPO Memory](./14_RLHF_PPO_Memory.md)


### Step 1: 为什么需要梯度累积

> **大 batch 的好处**：梯度更稳定，更新方向更平滑。
>
> **但显存不够怎么办？**
> - 直接增大 batch 往往会先爆显存。
> - 梯度累积通过“多次反传、一次更新”保留大 batch 的优化效果。
> - 只要每个 micro-batch 的 loss 按 `accum_steps` 做缩放，最终效果就和一次性喂入大 batch 非常接近。

### Step 2: 数学等价性

设一个完整 batch 被切成 `K` 个 micro-batch。若每个 micro-batch 的损失记为 `L_i`，则梯度累积相当于计算：

$$
\nabla L = \frac{1}{K} \sum_{i=1}^{K} \nabla L_i
$$

工程上最关键的细节只有两个：
1. 每次 `backward()` 前把 loss 除以 `accum_steps`。
2. 只在最后一个 micro-batch 后执行 `optimizer.step()` 和 `optimizer.zero_grad()`。

### Step 3: 代码实现框架

下面我们实现两个更新步骤：
- `train_step_full_batch`：一次性使用完整 batch 更新。
- `train_step_with_accumulation`：把 batch 拆成多个 micro-batch，累积梯度后再更新。

这两种方式在等价条件下，参数更新应该几乎一致。


```python
import copy
import torch
import torch.nn as nn

```


```python
class TinyRegressor(nn.Module):
    def __init__(self, in_dim=4, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_step_full_batch(model, optimizer, x, y):
    model.train()
    criterion = nn.MSELoss(reduction='mean')
    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    return loss.detach().item()


def train_step_with_accumulation(model, optimizer, x, y, accum_steps=4):
    """
    使用梯度累积执行一次参数更新。
    """
    if x.size(0) % accum_steps != 0:
        raise ValueError("batch size 必须能被 accum_steps 整除")

    model.train()
    criterion = nn.MSELoss(reduction='mean')
    optimizer.zero_grad()

    micro_size = x.size(0) // accum_steps
    total_loss = 0.0
    for idx in range(accum_steps):
        xb = x[idx * micro_size:(idx + 1) * micro_size]
        yb = y[idx * micro_size:(idx + 1) * micro_size]
        pred = model(xb)
        loss = criterion(pred, yb) / accum_steps
        loss.backward()
        total_loss += loss.detach().item()

    optimizer.step()
    optimizer.zero_grad()
    return total_loss

```


```python
# 运行此单元格以测试你的实现
def test_gradient_accumulation():
    try:
        torch.manual_seed(42)
        x = torch.randn(8, 4)
        y = torch.randn(8, 2)

        base_model = TinyRegressor()
        model_full = copy.deepcopy(base_model)
        model_accum = copy.deepcopy(base_model)

        opt_full = torch.optim.SGD(model_full.parameters(), lr=0.1)
        opt_accum = torch.optim.SGD(model_accum.parameters(), lr=0.1)

        loss_full = train_step_full_batch(model_full, opt_full, x, y)
        loss_accum = train_step_with_accumulation(model_accum, opt_accum, x, y, accum_steps=4)

        print(f"Full batch loss: {loss_full:.6f}")
        print(f"Accumulated loss: {loss_accum:.6f}")

        for p_full, p_accum in zip(model_full.parameters(), model_accum.parameters()):
            assert torch.allclose(p_full, p_accum, atol=1e-6), "梯度累积与 full batch 更新不一致！"

        print("✅ 测试通过！梯度累积与完整 batch 的参数更新一致。")
    except NotImplementedError:
        print("请先完成 TODO 部分。")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise e

test_gradient_accumulation()

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
import copy
import torch
import torch.nn as nn

class TinyRegressor(nn.Module):
    def __init__(self, in_dim=4, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_step_full_batch(model, optimizer, x, y):
    model.train()
    criterion = nn.MSELoss(reduction='mean')
    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    return loss.detach().item()


def train_step_with_accumulation(model, optimizer, x, y, accum_steps=4):
    if x.size(0) % accum_steps != 0:
        raise ValueError("batch size 必须能被 accum_steps 整除")

    model.train()
    criterion = nn.MSELoss(reduction='mean')
    optimizer.zero_grad()

    micro_size = x.size(0) // accum_steps
    total_loss = 0.0
    for idx in range(accum_steps):
        xb = x[idx * micro_size:(idx + 1) * micro_size]
        yb = y[idx * micro_size:(idx + 1) * micro_size]
        pred = model(xb)
        loss = criterion(pred, yb) / accum_steps
        loss.backward()
        total_loss += loss.detach().item()

    optimizer.step()
    optimizer.zero_grad()
    return total_loss

```

### 解析

**1. 为什么要把 loss 除以 `accum_steps`**

如果不缩放，累积出来的梯度会比完整 batch 的梯度大 `accum_steps` 倍，等价于悄悄提高了学习率。正确做法是让每个 micro-batch 的 loss 先按累积次数缩放，再执行反向传播。

**2. 为什么只在最后一步 `optimizer.step()`**

梯度累积的本质就是“先攒梯度，再统一更新”。如果每个 micro-batch 都 step，一次大 batch 会被拆成多次小更新，训练行为就变了。

**3. 什么时候最适合使用梯度累积**

- 显存不足以支持目标 batch size。
- 训练希望保持较大的有效 batch。
- 多卡训练下，希望在不改模型结构的前提下提高吞吐和稳定性。
