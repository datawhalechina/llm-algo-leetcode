# 12. Gradient Accumulation | 梯度累积

**难度：** Medium | **环境：** CPU-first | **标签：** `训练微调`, `梯度累积`, `显存优化` | **目标人群：** 训练机制学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/12_Gradient_Accumulation.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

增大 batch 可能降低单次梯度估计的噪声，但实际收益取决于数据、学习率和优化器设置。大模型微调中，batch size 往往先受到显存限制；micro-batch 过小时，还需要关注吞吐和训练口径。

梯度累积将一个逻辑 batch 拆成多个 micro-batch，分多次 backward，最后执行一次 optimizer step。本节与 `11` 一起服务 `13`：`11` 讨论学习率节奏，`12` 讨论有效 batch 的实现和显存代价。代码使用 toy regression 对比 full batch 与 accumulation；结果只说明当前条件下的近似等价，不代表所有模型上的严格等价、完整训练吞吐或真实 GPU 显存收益。

**关键词：** `gradient accumulation`, `micro-batch`, `effective batch`

---
## 前置阅读

**导语：** 先把模型封装、优化器和训练循环补齐，再看多个 micro-batch 如何合成一次有效更新。
- [P0: 09. PyTorch nn.Module Basics | nn.Module 基础](../00_Prerequisites/09_PyTorch_nn_Module_Basics.md)
- [P0: 11. PyTorch Optimizers and Loss | 优化器与损失](../00_Prerequisites/11_PyTorch_Optimizers_and_Loss.md)
- [P0: 13. Simple Neural Network Training | 简单神经网络训练](../00_Prerequisites/13_Simple_Neural_Network_Training.md)

## 相关阅读

**导语：** 理解梯度累积后，下一步最自然的是把它放进端到端微调闭环、LoRA 项目和训练性能分析里，看它怎样在显存与吞吐之间做权衡。
- [13. End-to-End Fine-Tuning Experiment | 端到端微调实验](../02_PyTorch_Algorithms/13_End_to_End_Fine_Tuning_Experiment.md)
- [60. LoRA Fine-Tuning Project | LoRA 微调项目](../02_PyTorch_Algorithms/60_LoRA_Fine_Tuning_Project.md)
- [73. Training Performance Analysis | 训练性能分析](../02_PyTorch_Algorithms/73_Training_Performance_Analysis.md)
- [76. Activation Checkpoint Offload Benchmark | Checkpoint 与 Offload 对比项目](../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.md)

---
### Step 1: 为什么需要梯度累积
梯度累积用于在显存受限时实现较大的有效 batch：完整 batch 不能一次放入显存时，可以拆成多个 micro-batch。

> **大 batch 的可能收益**：梯度估计噪声可能更低，但实际效果取决于训练设置。
>
> **但显存不够怎么办？**
> - 直接增大 batch 往往会先爆显存。
> - 梯度累积通过“多次反传、一次更新”近似复现大 batch 的平均梯度，但是否等价取决于训练口径。
> - 在 reduction、数据顺序、随机状态和更新节奏对齐时，梯度累积可以近似复现一次性大 batch 的平均梯度。

在 SFT / LoRA 微调里，实际关注的是有效 batch：

```python
effective_batch_size = micro_batch_size * accum_steps
```

梯度累积降低的是单次 forward/backward 的 activation 峰值，不会减少模型参数、LoRA 参数或优化器状态本身的显存占用。因此它经常和 LoRA、checkpointing、量化一起使用。

### Step 2: 数学等价性
重点不只是 loss 缩放公式，而是先缩放再反传，最后按累积步数统一更新。

在 micro-batch 大小相同、每个 micro-batch 使用相同 `mean` reduction 的简化设置下，设一个完整 batch 被切成 `K` 个 micro-batch，若每个 micro-batch 的损失记为 `L_i`，则梯度累积相当于计算：

$$
\nabla L = \frac{1}{K} \sum_{i=1}^{K} \nabla L_i
$$

实现时需要检查三个细节：
1. 每次 `backward()` 前把 loss 除以 `accum_steps`。
2. 只在最后一个 micro-batch 后执行 `optimizer.step()`。
3. `input_ids / attention_mask / labels` 这类 batch 字典必须一起切，不能只切输入不切标签。

如果忘了除以 `accum_steps`，等价于把学习率悄悄放大了 `accum_steps` 倍；如果 batch size 不能被 `accum_steps` 整除，要么 `drop_last`，要么做动态累积，本节先用显式报错保持逻辑简单。
### Step 3: 代码实现框架与任务拆解
代码框架把完整 batch 和累积 batch 的更新路径并排对齐。

下面我们实现两个更新步骤：
- `train_step_full_batch`：一次性使用完整 batch 更新。
- `train_step_with_accumulation`：把 batch 拆成多个 micro-batch，累积梯度后再更新。

同时保留一个 `slice_micro_batch` 辅助函数，用来演示 SFT 场景里的 batch 字典如何整体切分。它不改变本题的回归 toy model，只负责把本节和后面的端到端微调实验接起来。

核心判断是：在 reduction、数据顺序、随机状态和更新节奏对齐时，累积 batch 的平均梯度和参数更新应与完整 batch 接近；这是一种有条件的近似等价，不是对所有模型和训练设置的保证。
#### 图解：micro-batch 如何合成 effective batch

梯度累积的关键是“多次 backward，一次 step”。

```text
full batch:       [ sample 0 1 2 3 4 5 6 7 ] ──► loss ──► backward ──► step

accumulation:     [0 1] ─► loss / K ─► backward ┐
                  [2 3] ─► loss / K ─► backward ├─ accumulated grad ─► step
                  [4 5] ─► loss / K ─► backward ┤
                  [6 7] ─► loss / K ─► backward ┘
```

SFT batch 字典切分时，三件套必须同步：

```python
mb = {
    "input_ids": input_ids[start:end],
    "attention_mask": attention_mask[start:end],
    "labels": labels[start:end],
}
```

`effective_batch_size = micro_batch_size * accum_steps`。它降低的是单次 activation 峰值，不会减少参数和优化器状态占用。

![梯度累积图](/02_PyTorch_Algorithms/12_gradient_accumulation.svg)


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


def slice_micro_batch(batch: dict[str, torch.Tensor], idx: int, accum_steps: int):
    """按 micro-batch 同步切分 SFT batch 字典。"""
    if accum_steps <= 0:
        raise ValueError("accum_steps 必须为正数")
    if idx < 0 or idx >= accum_steps:
        raise IndexError("micro-batch idx 超出范围")
    batch_size = next(iter(batch.values())).size(0)
    if any(value.size(0) != batch_size for value in batch.values()):
        raise ValueError("batch 字典中的 tensor 第一维必须一致")
    if batch_size % accum_steps != 0:
        raise ValueError("batch size 必须能被 accum_steps 整除")
    micro_size = batch_size // accum_steps
    start = idx * micro_size
    end = (idx + 1) * micro_size
    return {key: value[start:end] for key, value in batch.items()}


def train_step_full_batch(model, optimizer, x, y):
    """使用完整 batch 完成一次参数更新。

    Args:
        model: 待训练模型。
        optimizer: 与 model 参数绑定的优化器。
        x, y: 第一维为 batch 的输入和目标张量。

    Returns:
        未缩放的当前 batch loss。
    """
    model.train()
    criterion = nn.MSELoss(reduction='mean')
    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    return loss.detach().item()


def train_step_with_accumulation(model, optimizer, x, y, accum_steps=4):
    """使用多个 micro-batch 完成一次参数更新。

    Args:
        model: 待训练模型。
        optimizer: 与 model 参数绑定的优化器。
        x, y: 第一维为 batch 的输入和目标张量。
        accum_steps: micro-batch 数量，要求 batch size 可整除。

    Returns:
        未缩放的累计 loss，用于日志记录。

    Note:
        每个 micro-batch 的 loss 除以 accum_steps 后再 backward；
        整个逻辑 batch 只执行一次 optimizer.step()。
    """
    if accum_steps <= 0:
        raise ValueError("accum_steps 必须为正数")
    if x.size(0) % accum_steps != 0:
        raise ValueError("batch size 必须能被 accum_steps 整除")

    model.train()
    criterion = nn.MSELoss(reduction='mean')
    optimizer.zero_grad()

    micro_size = x.size(0) // accum_steps
    total_loss = 0.0
    for idx in range(accum_steps):
        # ==========================================
        # 先切出当前 micro-batch，逐个处理而不是一次性喂完整 batch。
        # TODO 1: 切分当前 micro-batch
        # 提示：按 idx 和 micro_size 同步切分 x / y，保持样本对应。
        # ==========================================
        # xb = ???
        # yb = ???

        pred = model(xb)

        # ==========================================
        # TODO 2: 处理当前 micro-batch 的 loss
        # 提示: 先计算 micro_loss，再除以 accum_steps 后调用 backward()，
        #       保证累积后的梯度仍然对应完整 batch 的平均梯度。
        #       当前使用 MSELoss(reduction='mean')，不要把返回日志 loss 一起缩放。
        # ==========================================
        # loss = ???
        loss.backward()
        # total_loss = ???

    # ==========================================
    # TODO 3: 完成一次参数更新并返回结果
    # 提示: 所有 micro-batch 都 backward 后，只调用一次 optimizer.step()，
    #       返回未缩放 loss 的累计值。
    # ==========================================
    # 优化器操作
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
        assert abs(loss_full - loss_accum) < 1e-6, "梯度累积的 loss 口径不一致"


        sft_batch = {
            "input_ids": torch.arange(24).view(8, 3),
            "attention_mask": torch.ones(8, 3, dtype=torch.long),
            "labels": torch.arange(24).view(8, 3),
        }
        mb = slice_micro_batch(sft_batch, idx=1, accum_steps=4)
        assert mb["input_ids"].shape == (2, 3), "SFT micro-batch 切分 shape 错误"
        assert torch.equal(mb["input_ids"], sft_batch["input_ids"][2:4]), "SFT micro-batch 切分范围错误"

        for p_full, p_accum in zip(model_full.parameters(), model_accum.parameters()):
            assert torch.allclose(p_full, p_accum, atol=1e-6), "梯度累积与 full batch 更新不一致！"

        print("✅ CPU 机制验证通过：当前 toy 设置下，梯度累积与完整 batch 的 loss 和参数更新口径一致。")
    except NotImplementedError:
        print("请先完成 TODO 部分。")
        raise
    except (AttributeError, NameError, TypeError, ValueError) as e:
        print("代码可能未完成，导致变量未定义" if isinstance(e, NameError) else "代码可能未完成，导致了类型错误")
        raise NotImplementedError("请先完成 TODO 部分。") from e
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
        raise NotImplementedError("请先完成 TODO 部分。") from e
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise

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


def slice_micro_batch(batch: dict[str, torch.Tensor], idx: int, accum_steps: int):
    """按 micro-batch 同步切分 SFT batch 字典。"""
    if accum_steps <= 0:
        raise ValueError("accum_steps 必须为正数")
    if idx < 0 or idx >= accum_steps:
        raise IndexError("micro-batch idx 超出范围")
    batch_size = next(iter(batch.values())).size(0)
    if any(value.size(0) != batch_size for value in batch.values()):
        raise ValueError("batch 字典中的 tensor 第一维必须一致")
    if batch_size % accum_steps != 0:
        raise ValueError("batch size 必须能被 accum_steps 整除")
    micro_size = batch_size // accum_steps
    start = idx * micro_size
    end = (idx + 1) * micro_size
    return {key: value[start:end] for key, value in batch.items()}


def train_step_full_batch(model, optimizer, x, y):
    """使用完整 batch 完成一次参数更新。

    Args:
        model: 待训练模型。
        optimizer: 与 model 参数绑定的优化器。
        x, y: 第一维为 batch 的输入和目标张量。

    Returns:
        未缩放的当前 batch loss。
    """
    model.train()
    criterion = nn.MSELoss(reduction='mean')
    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    return loss.detach().item()


def train_step_with_accumulation(model, optimizer, x, y, accum_steps=4):
    """使用多个 micro-batch 完成一次参数更新。

    Args:
        model: 待训练模型。
        optimizer: 与 model 参数绑定的优化器。
        x, y: 第一维为 batch 的输入和目标张量。
        accum_steps: micro-batch 数量，要求 batch size 可整除。

    Returns:
        未缩放的累计 loss，用于日志记录。

    Note:
        每个 micro-batch 的 loss 除以 accum_steps 后再 backward；
        整个逻辑 batch 只执行一次 optimizer.step()。
    """
    if accum_steps <= 0:
        raise ValueError("accum_steps 必须为正数")
    if x.size(0) % accum_steps != 0:
        raise ValueError("batch size 必须能被 accum_steps 整除")

    model.train()
    criterion = nn.MSELoss(reduction='mean')
    optimizer.zero_grad()

    micro_size = x.size(0) // accum_steps
    total_loss = 0.0
    for idx in range(accum_steps):
        # 先切出当前 micro-batch，逐个处理而不是一次性喂完整 batch。
        # TODO 1: 切分当前 micro-batch
        xb = x[idx * micro_size:(idx + 1) * micro_size]
        yb = y[idx * micro_size:(idx + 1) * micro_size]

        pred = model(xb)

        # 先缩放 loss，确保累积后的总梯度尺度和完整 batch 一致。
        # TODO 2: 缩放 loss 并反传
        # 提示：当前使用 MSELoss(reduction='mean')，先除以 accum_steps 再 backward。
        loss = criterion(pred, yb) / accum_steps
        loss.backward()
        total_loss += loss.detach().item()

    # 所有 micro-batch 反传完后再统一更新参数。
    # TODO 3: 统一更新参数并返回累计 loss
    # 提示：只调用一次 optimizer.step()，返回未缩放口径的日志 loss。
    optimizer.step()
    optimizer.zero_grad()
    return total_loss
```

### 答案与直觉

- **这一题要解决什么**：把大 batch 的更新效果用 micro-batch 累积模拟出来。
- **为什么这样做**：显存不够时靠多次 backward、一次 step，在条件对齐时近似保持完整 batch 的更新口径。
- **带走的直觉**：梯度累积的关键不是拆 batch，而是保持梯度尺度不变并延后参数更新。

**1. TODO 1 (切分当前 micro-batch)**

- **切分逻辑：** 梯度累积不是一次喂完整 batch，而是先把 `x / y` 按 `accum_steps` 拆成多个 micro-batch。
- **训练目标：** 每一轮循环都只处理当前片段，这样才能模拟大 batch 的效果，同时把峰值显存压低。
- **实现重点：** 先确定当前 micro-batch 的切片范围，再把输入和标签切出来。

**2. TODO 2 (缩放 loss 并反传)**

- **梯度对齐：** 每个 micro-batch 的 loss 必须先除以 `accum_steps`，再执行 `backward()`。
- **等价性：** 在相同 reduction、数据顺序和随机状态等条件下，这样累积出来的平均梯度才与完整 batch 接近，不会悄悄把更新幅度放大 `accum_steps` 倍。
- **实现重点：** 这一层的核心是“先缩放，再反传，再累加”。

**3. TODO 3 (统一更新参数并返回累计 loss)**

- **先攒后更：** 所有 micro-batch 都完成 backward 之后，再统一执行一次 `optimizer.step()` 和 `optimizer.zero_grad()`。
- **闭环意义：** 这样一次参数更新才与完整 batch 的更新口径接近；遇到 dropout、BatchNorm 或不同 scheduler 节奏时，需要重新验证。
- **结果记录：** 最后返回累计 `history` 或 `total_loss`，方便观察训练过程中 loss 是否下降。

**4. 进阶思考：为什么要做重复样本验证？**

- **一致性检查：** 通过 full batch 对照可以验证当前 toy 设置下的近似等价，不能直接推广到所有模型和训练配置。
- **工程价值：** 只要这套链路对齐，后续再切换更复杂的数据和更大的 batch 也更稳。
- **实践意义：** 这条链路把 `SFT Loss`、`梯度累积`、`参数更新` 连接成一个可运行的小闭环。

**5. SFT batch 字典怎么切**

- **同步切分**：`input_ids`、`attention_mask`、`labels` 必须按同一个 `[start:end]` 范围切分。
- **有效 batch**：`effective_batch_size = micro_batch_size * accum_steps`，scheduler 和日志通常按 `optimizer.step()` 后的一次有效更新计数。
- **显存边界**：梯度累积减少的是每个 micro-batch 的 activation 峰值，不会减少参数、梯度和优化器状态的长期占用。
