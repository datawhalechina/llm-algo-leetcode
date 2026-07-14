# 12. Gradient Accumulation | 梯度累积

**难度：** Medium | **环境：** CPU-first | **标签：** `训练技巧`, `显存优化`, `PyTorch` | **目标人群：** 模型微调与工程部署

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/12_Gradient_Accumulation.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


在做大模型微调时，显存通常先被 batch size 吃满。梯度累积的核心思路是：把一个大 batch 拆成多个 micro-batch，分多次 backward，最后只 step 一次，从而在不增加峰值显存的前提下，模拟更大的有效 batch。可以先把它记成：把大 batch 拆成小块，分多次反传，最后一次更新。

**关键词：** `gradient accumulation`, `micro-batch`, `effective batch`
## 前置阅读

**导语：** 先把梯度累积需要的 PyTorch 训练接口和张量操作补齐，再来看小 batch 聚合更新。
- [09. PyTorch nn.Module Basics | nn.Module 基础](../00_Prerequisites/09_PyTorch_nn_Module_Basics.md)
- [11. PyTorch Optimizers and Loss | 优化器与损失](../00_Prerequisites/11_PyTorch_Optimizers_and_Loss.md)
- [13. Simple Neural Network Training | 简单神经网络训练](../00_Prerequisites/13_Simple_Neural_Network_Training.md)

## 相关阅读

**导语：** 梯度累积和训练性能、显存占用关系紧密，可结合硬件和 profiling 一起看。
- [06. VRAM Calculation and ZeRO | 显存估算与 ZeRO](../01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.md)
- [13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
- [20. NCCL and AllReduce Basics | NCCL 与 AllReduce 基础](../01_Hardware_Math_and_Systems/20_NCCL_and_AllReduce_Basics.md)
### Step 1: 为什么需要梯度累积
这一节先把“大 batch 更稳，但显存不够”这个矛盾说清楚。

> **大 batch 的好处**：梯度更稳定，更新方向更平滑。
>
> **但显存不够怎么办？**
> - 直接增大 batch 往往会先爆显存。
> - 梯度累积通过“多次反传、一次更新”保留大 batch 的优化效果。
> - 只要每个 micro-batch 的 loss 按 `accum_steps` 做缩放，最终效果就和一次性喂入大 batch 非常接近。

### Step 2: 数学等价性
这里要抓住的重点不是公式本身，而是缩放和反传的顺序。

设一个完整 batch 被切成 `K` 个 micro-batch。若每个 micro-batch 的损失记为 `L_i`，则梯度累积相当于计算：

$$
\nabla L = \frac{1}{K} \sum_{i=1}^{K} \nabla L_i
$$

工程上最关键的细节只有两个：
1. 每次 `backward()` 前把 loss 除以 `accum_steps`。
2. 只在最后一个 micro-batch 后执行 `optimizer.step()` 和 `optimizer.zero_grad()`。
这也是为什么 `train_step_with_accumulation` 的实现要先缩放再反传。

### Step 3: 代码实现框架
这一段把完整 batch 和累积 batch 的更新路径并排对齐。

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
        # ==========================================
        # 先切出当前 micro-batch，逐个处理而不是一次性喂完整 batch。
        # TODO 1: 切分当前 micro-batch
        # 提示: 从 x / y 中按 idx 和 micro_size 取出对应片段
        # ==========================================
        # xb = ???
        # yb = ???

        pred = model(xb)

        # ==========================================
        # TODO 2: 处理当前 micro-batch 的 loss
        # 提示: 先算出当前 micro-batch 的 loss，再完成后续的训练动作
        # ==========================================
        # loss = ???
        loss.backward()
        # total_loss = ???

    # ==========================================
    # TODO 3: 完成一次参数更新并返回结果
    # 提示: 这一部分对应整轮 micro-batch 的收尾
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

        for p_full, p_accum in zip(model_full.parameters(), model_accum.parameters()):
            assert torch.allclose(p_full, p_accum, atol=1e-6), "梯度累积与 full batch 更新不一致！"

        print("✅ 测试通过！梯度累积与完整 batch 的参数更新一致。")
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
        # 先切出当前 micro-batch，逐个处理而不是一次性喂完整 batch。
        # TODO 1: 切分当前 micro-batch
        xb = x[idx * micro_size:(idx + 1) * micro_size]
        yb = y[idx * micro_size:(idx + 1) * micro_size]

        pred = model(xb)

        # 先缩放 loss，确保累积后的总梯度尺度和完整 batch 一致。
        # TODO 2: 缩放 loss 并反传
        loss = criterion(pred, yb) / accum_steps
        loss.backward()
        total_loss += loss.detach().item()

    # 所有 micro-batch 反传完后再统一更新参数。
    # TODO 3: 统一更新参数并返回累计 loss
    optimizer.step()
    optimizer.zero_grad()
    return total_loss
```

### 答案与直觉

- **这一题要解决什么**：把大 batch 的更新效果用 micro-batch 累积模拟出来。
- **为什么这样做**：显存不够时靠多次 backward、一次 step 保持等价更新。
- **带走的直觉**：梯度累积的关键不是拆 batch，而是保持梯度尺度不变并延后参数更新。

**1. TODO 1 (切分当前 micro-batch)**

- **切分逻辑：** 梯度累积不是一次喂完整 batch，而是先把 `x / y` 按 `accum_steps` 拆成多个 micro-batch。
- **训练目标：** 每一轮循环都只处理当前片段，这样才能模拟大 batch 的效果，同时把峰值显存压低。
- **实现重点：** 这里要先确定当前 micro-batch 的切片范围，再把输入和标签切出来。

**2. TODO 2 (缩放 loss 并反传)**

- **梯度对齐：** 每个 micro-batch 的 loss 必须先除以 `accum_steps`，再执行 `backward()`。
- **等价性：** 这样累积出来的总梯度才和完整 batch 的梯度一致，不会悄悄把更新幅度放大 `accum_steps` 倍。
- **实现重点：** 这一层的核心是“先缩放，再反传，再累加”。

**3. TODO 3 (统一更新参数并返回累计 loss)**

- **先攒后更：** 所有 micro-batch 都完成 backward 之后，再统一执行一次 `optimizer.step()` 和 `optimizer.zero_grad()`。
- **闭环意义：** 这样一次参数更新就等价于完整 batch 的更新，梯度累积的逻辑才真正闭环。
- **结果记录：** 最后返回累计 `history` 或 `total_loss`，方便观察训练过程中 loss 是否下降。

**4. 进阶思考：为什么要做重复样本验证？**

- **一致性检查：** 通过重复样本验证，可以确认梯度累积是否真的等价于完整 batch。
- **工程价值：** 只要这套链路对齐，后续再切换更复杂的数据和更大的 batch 也更稳。
- **实践意义：** 这一步把 `SFT Loss`、`梯度累积`、`参数更新` 连接成一个可运行的小闭环。