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

训练规模、更新节奏和显存预算往往需要同时考虑：扩大训练规模可能改善统计稳定性，却也会提高单次计算的资源压力。本节帮助你建立这三者之间的判断口径，理解 micro-batch 如何累积成 effective batch，以及什么时候才执行一次 `optimizer.step()`。

学习过程中，你会逐步区分一次更新看到了多少数据、单次计算承担了多少数据，以及 optimizer update 和 scheduler 应该以什么节奏发生；随后可以把这些计算方法用于微调和训练性能实验，比较显存、吞吐与更新结果。

**关键词：** `gradient accumulation`, `micro-batch`, `effective batch`

---
## 前置阅读

**导语：** 先把模型封装、优化器和训练循环补齐，再看多个 micro-batch 如何合成一次有效更新。
- [P0: 09. PyTorch nn.Module Basics | nn.Module 基础](../00_Prerequisites/09_PyTorch_nn_Module_Basics.md)
- [P0: 11. PyTorch Optimizers and Loss | 优化器与损失](../00_Prerequisites/11_PyTorch_Optimizers_and_Loss.md)
- [P0: 13. Simple Neural Network Training | 简单神经网络训练](../00_Prerequisites/13_Simple_Neural_Network_Training.md)

---
### Step 1: 一次逻辑更新由什么组成
完整 batch 不能一次放入显存时，可以把一次逻辑参数更新拆成多个较小的 micro-batch：每个 micro-batch 完成前向和反向，梯度暂时汇总，达到设定次数后再完成一次更新。先明确一次更新看到多少样本，以及一次计算与一次更新分别承担什么任务。
下面的表格定义本节的输入与输出，主图展示从逻辑 batch 到参数更新的推进关系；显存账本和更新一致性将在后续继续展开。

| 观察对象 | 它回答的问题 | 梯度累积带来的变化 |
| --- | --- | --- |
| 单次计算批次 | 一次前向 / 反向处理多少样本？ | 定义一次 micro-batch 的计算单位 |
| 累积次数 | 多少次小批次合成一次更新？ | 定义一次逻辑更新包含多少次计算 |
| 有效 batch | 一次参数更新看到了多少样本？ | 通常为单次计算批次 × 累积次数 |

![梯度累积总览](../public/02_PyTorch_Algorithms/12_gradient_accumulation_overview.svg)

### Step 2: 单次计算的显存账本如何变化
梯度累积把一次更新拆成多个 micro-batch 后，最直接的变化发生在单次计算的 activation 峰值；参数、梯度和 optimizer state 仍需持续驻留。观察显存时，同时记录单次计算的峰值和整个训练过程中的长期状态。

| 账本对象 | 梯度累积改变什么 | 梯度累积不改变什么 | 需要观察的结果 |
| --- | --- | --- | --- |
| 单次 activation | 单次输入规模和 activation 峰值 | 参数、梯度和 optimizer state 的长期规模 | peak memory、activation 峰值；必要时记录单步时间 |
| 梯度缓存 | 在多个 micro-batch 之间持续累积 | 缓存大小通常与模型参数规模相关，不会随累积次数按比例缩小 | 累积期间的显存占用 |
| 参数与 optimizer state | — | 一次更新期间持续驻留 | 长期显存占用 |

### Step 3: 如何让梯度累积与完整 batch 对齐
显存峰值下降后，还要确认参数更新仍然对应同一个逻辑 batch。设一个完整 batch 被切成 `K` 个 micro-batch；如果每个 micro-batch 使用 `mean` reduction，就需要先按 `K` 缩放 loss，再累积梯度。

$$\nabla L = \frac{1}{K} \sum_{i=1}^{K} \nabla L_i$$

下表把影响更新一致性的条件集中起来。输入、目标和掩码等 batch 字段也要同步切分；如果忘记缩放 loss，累计梯度会放大约 `K` 倍。对于包含随机层、动态 loss 或不同 scheduler 节奏的训练配置，还需要单独复查更新结果。

| 对齐条件 | 学习者需要做什么 | 不满足时的影响 |
| --- | --- | --- |
| 样本与顺序 | 两条路径使用同一批样本和顺序 | 梯度来源不同 |
| loss reduction | 两条路径使用相同 reduction | 梯度尺度不同 |
| loss 缩放 | 每个 micro-batch 的 loss 除以 `K` | 梯度约放大 `K` 倍 |
| 更新时机 | 累积完成后只执行一次 `optimizer.step()` | 参数更新次数不同 |

![梯度累积的更新一致性条件](../public/02_PyTorch_Algorithms/12_gradient_accumulation_update_contract.svg)
### Step 4: 实现一次逻辑更新并验证
本步把前面的机制落到三个对象：切分 batch、建立 full batch 对照、完成梯度累积更新。题目区要求 batch 可被 `accum_steps` 整除，测试区检查 loss、输出、参数更新、异常输入和 `optimizer.step()` 次数。`slice_micro_batch` 用于验证多字段 batch 的同步切分；当前回归训练路径直接切分 `x/y`，两者遵循相同的索引原则。日志保留未缩放的 mean loss，反向传播使用缩放后的 loss。

| 实现对象 | 作用 | 输入 | 输出 | TODO / 验证重点 |
| --- | --- | --- | --- | --- |
| `slice_micro_batch` | 同步切分多字段 batch | batch 字典、索引、累积步数 | 对齐后的 micro-batch | 第一维一致、切片范围正确 |
| `train_step_full_batch` | 提供完整 batch 对照 | 模型、优化器、`x/y` | 一次更新和未缩放 loss | baseline loss 与参数更新 |
| `train_step_with_accumulation` | 切分、缩放、累积并更新 | 模型、优化器、`x/y`、`accum_steps` | 一次更新和日志 loss | loss 缩放、一次 step、输出和参数一致 |



```python
import copy
import torch
import torch.nn as nn

```


```python
class TinyRegressor(nn.Module):
    """用于比较 full batch 与梯度累积更新结果的最小回归模型。"""
    def __init__(self, in_dim=4, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim),
        )

    def forward(self, x):
        """接收形状为 `[batch, in_dim]` 的输入并返回预测值。"""
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
    """使用完整 batch 完成一次参数更新，作为梯度累积的对照基线。

    `x` 与 `y` 的第一维必须相同；返回未缩放的 mean loss。
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
    """使用多个 micro-batch 完成一次逻辑更新。

    `x` 与 `y` 的第一维必须相同，且 batch size 必须能被
    `accum_steps` 整除；返回按原始 mean loss 口径记录的日志值。
    """
    if x.size(0) != y.size(0):
        raise ValueError("x 和 y 的 batch 维度必须一致")
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
        # TODO 1：切分当前 micro-batch
        # 提示：按 idx 和 micro_size 使用同一个 [start:end] 范围切分 x / y，保持样本对应；不要在这里重新打乱样本。
        # ==========================================
        # xb = ???  # 当前 micro-batch 的输入
        # yb = ???  # 当前 micro-batch 的目标

        pred = model(xb)

        # ==========================================
        # TODO 2：计算 micro_loss，生成 scaled_loss 并完成反向传播
        # 提示：先得到未缩放的 micro_loss，再构造 scaled_loss 用于 backward；
        #       日志累计仍使用未缩放的 micro_loss，保证与 full batch 的 mean loss 同口径。
        #       当前使用 MSELoss(reduction='mean')。
        # ==========================================
        # micro_loss = ???  # 当前 micro-batch 的原始 mean loss
        # scaled_loss = ???  # 用于 backward 的缩放后 loss
        # scaled_loss.backward()
        # total_loss += micro_loss.detach().item() / accum_steps

    # ==========================================
    # TODO 3：完成一次参数更新并返回结果
    # 提示：所有 micro-batch 都完成 backward 后，只调用一次 optimizer.step()；
    #       然后清空梯度，返回各 micro_loss 的平均值，不返回 scaled_loss。
    # ==========================================
    # optimizer_step = ???  # 只执行一次逻辑更新：调用 optimizer.step()
    # clear_grad = ???  # 为下一次逻辑 batch 清空梯度：调用 optimizer.zero_grad()
    # result = ???  # 返回未缩放 loss 的平均值：使用 total_loss
    return result

```


```python
# 运行此单元格以测试你的实现
def _legacy_test_gradient_accumulation():
    try:
        torch.manual_seed(42)
        x = torch.randn(8, 4)
        y = torch.randn(8, 2)

        base_model = TinyRegressor()
        model_full = copy.deepcopy(base_model)
        model_accum = copy.deepcopy(base_model)

        opt_full = torch.optim.SGD(model_full.parameters(), lr=0.1)
        opt_accum = torch.optim.SGD(model_accum.parameters(), lr=0.1)

        full_step_calls = [0]
        accum_step_calls = [0]
        original_full_step = opt_full.step
        original_accum_step = opt_accum.step

        def counted_full_step(*args, **kwargs):
            full_step_calls[0] += 1
            return original_full_step(*args, **kwargs)

        def counted_accum_step(*args, **kwargs):
            accum_step_calls[0] += 1
            return original_accum_step(*args, **kwargs)

        opt_full.step = counted_full_step
        opt_accum.step = counted_accum_step
        loss_full = train_step_full_batch(model_full, opt_full, x, y)
        loss_accum = train_step_with_accumulation(model_accum, opt_accum, x, y, accum_steps=4)

        print(f"Full batch loss: {loss_full:.6f}")
        print(f"Accumulated loss: {loss_accum:.6f}")
        assert abs(loss_full - loss_accum) < 1e-6, "梯度累积的 loss 口径不一致"
        assert full_step_calls[0] == 1, "full batch 应只执行一次 optimizer.step()"
        assert accum_step_calls[0] == 1, "一次逻辑 batch 应只执行一次 optimizer.step()"


        sft_batch = {
            "input_ids": torch.arange(24).view(8, 3),
            "attention_mask": torch.ones(8, 3, dtype=torch.long),
            "labels": torch.arange(24).view(8, 3),
        }
        mb = slice_micro_batch(sft_batch, idx=1, accum_steps=4)
        assert mb["input_ids"].shape == (2, 3), "SFT micro-batch 切分 shape 错误"
        assert torch.equal(mb["input_ids"], sft_batch["input_ids"][2:4]), "SFT micro-batch 切分范围错误"
        assert torch.equal(mb["attention_mask"], sft_batch["attention_mask"][2:4]), "attention_mask 未同步切分"
        assert torch.equal(mb["labels"], sft_batch["labels"][2:4]), "labels 未同步切分"

        with torch.no_grad():
            output_full = model_full(x)
            output_accum = model_accum(x)
        assert torch.allclose(output_full, output_accum, atol=1e-6), "两条路径更新后的输出不一致"

        for p_full, p_accum in zip(model_full.parameters(), model_accum.parameters()):
            assert torch.allclose(p_full, p_accum, atol=1e-6), "梯度累积与 full batch 更新不一致！"
        bad_model = copy.deepcopy(base_model)
        bad_optimizer = torch.optim.SGD(bad_model.parameters(), lr=0.1)
        for bad_steps in (0, 3):
            try:
                train_step_with_accumulation(bad_model, bad_optimizer, x, y, accum_steps=bad_steps)
            except ValueError:
                pass
            else:
                raise AssertionError(f"accum_steps={bad_steps} 应被拒绝")
        try:
            train_step_with_accumulation(
                copy.deepcopy(base_model),
                torch.optim.SGD(base_model.parameters(), lr=0.1),
                x,
                y[:-1],
                accum_steps=4,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("x 和 y 的 batch 维度不一致时应被拒绝")

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

# 19 节风格的机制测试入口：每个辅助函数只负责一类证据。
def _build_accumulation_case():
    torch.manual_seed(42)
    return torch.randn(8, 4), torch.randn(8, 2), TinyRegressor()

def _assert_micro_batch_contract():
    batch = {
        "input_ids": torch.arange(24).view(8, 3),
        "attention_mask": torch.ones(8, 3, dtype=torch.long),
        "labels": torch.arange(24).view(8, 3),
    }
    micro = slice_micro_batch(batch, idx=1, accum_steps=4)
    for name, value in batch.items():
        assert torch.equal(micro[name], value[2:4]), f"{name} 未同步切分"

def _assert_step_count_and_loss(x, y, base_model):
    full_model = copy.deepcopy(base_model)
    accum_model = copy.deepcopy(base_model)
    full_opt = torch.optim.SGD(full_model.parameters(), lr=0.1)
    accum_opt = torch.optim.SGD(accum_model.parameters(), lr=0.1)
    full_calls, accum_calls = [0], [0]
    full_step, accum_step = full_opt.step, accum_opt.step
    def counted_full_step(*args, **kwargs):
        full_calls[0] += 1
        return full_step(*args, **kwargs)
    def counted_accum_step(*args, **kwargs):
        accum_calls[0] += 1
        return accum_step(*args, **kwargs)
    full_opt.step = counted_full_step
    accum_opt.step = counted_accum_step
    loss_full = train_step_full_batch(full_model, full_opt, x, y)
    loss_accum = train_step_with_accumulation(accum_model, accum_opt, x, y, accum_steps=4)
    assert abs(loss_full - loss_accum) < 1e-6, "loss 口径不一致"
    assert full_calls[0] == 1 and accum_calls[0] == 1, "一次逻辑 batch 必须只更新一次"
    return full_model, accum_model

def _assert_update_equivalence(full_model, accum_model, x):
    with torch.no_grad():
        assert torch.allclose(full_model(x), accum_model(x), atol=1e-6), "更新后的输出不一致"
    for full_param, accum_param in zip(full_model.parameters(), accum_model.parameters()):
        assert torch.allclose(full_param, accum_param, atol=1e-6), "参数更新不一致"

def _assert_invalid_inputs(x, y, base_model):
    for bad_steps in (0, 3):
        try:
            train_step_with_accumulation(copy.deepcopy(base_model), torch.optim.SGD(base_model.parameters(), lr=0.1), x, y, bad_steps)
        except ValueError:
            continue
        raise AssertionError(f"accum_steps={bad_steps} 应被拒绝")
    try:
        train_step_with_accumulation(copy.deepcopy(base_model), torch.optim.SGD(base_model.parameters(), lr=0.1), x, y[:-1], 4)
    except ValueError:
        return
    raise AssertionError("x 和 y 的 batch 维度不一致时应被拒绝")

def test_gradient_accumulation():
    try:
        x, y, base_model = _build_accumulation_case()
        _assert_micro_batch_contract()
        full_model, accum_model = _assert_step_count_and_loss(x, y, base_model)
        _assert_update_equivalence(full_model, accum_model, x)
        _assert_invalid_inputs(x, y, base_model)
        print("✅ CPU 机制验证通过：梯度累积与完整 batch 的更新口径一致。")
    except (AttributeError, NameError, TypeError, ValueError) as e:
        print("代码可能未完成，请先完成 TODO。")
        raise NotImplementedError("请先完成 TODO 部分。") from e
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
        raise NotImplementedError("请先完成 TODO 部分。") from e

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
    if x.size(0) != y.size(0):
        raise ValueError("x 和 y 的 batch 维度必须一致")
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
        # TODO 1：切分当前 micro-batch
        xb = x[idx * micro_size:(idx + 1) * micro_size]  # TODO 1 对应挖空：xb，当前 micro-batch 输入
        yb = y[idx * micro_size:(idx + 1) * micro_size]  # TODO 1 对应挖空：yb，当前 micro-batch 目标

        pred = model(xb)

        # 先缩放反向 loss，确保累积后的总梯度尺度和完整 batch 一致；日志仍使用未缩放值。
        # TODO 2：计算 micro_loss，生成 scaled_loss 并完成反向传播
        # 先得到 micro_loss，再构造 scaled_loss；日志使用 micro_loss，反向使用 scaled_loss。
        micro_loss = criterion(pred, yb)  # TODO 2 对应挖空：micro_loss，原始 mean loss
        scaled_loss = micro_loss / accum_steps  # TODO 2 对应挖空：scaled_loss，反向传播 loss
        scaled_loss.backward()
        total_loss += micro_loss.detach().item() / accum_steps

    # 所有 micro-batch 反传完后再统一更新参数。
    # TODO 3：统一更新参数并返回累计 loss
    # 只调用一次 optimizer.step()，随后清空梯度；返回各 micro_loss 的平均值。
    optimizer_step = optimizer.step()  # TODO 3 对应挖空：optimizer_step，调用一次逻辑更新
    clear_grad = optimizer.zero_grad()  # TODO 3 对应挖空：clear_grad，清空累计梯度
    result = total_loss  # TODO 3 对应挖空：result，保存平均日志 loss
    return result
```

### 解析：实现说明与验证口径

本题的参考实现把一次逻辑更新拆成多个 micro-batch，并分别验证切分、梯度尺度、输出与参数更新、step 次数、日志口径和非法输入。



**1. TODO 1 (切分当前 micro-batch)**

- **切分逻辑：** 梯度累积不是一次喂完整 batch，而是先把 `x / y` 按 `accum_steps` 拆成多个 micro-batch。
- **显存效果：** 每一轮循环只处理当前片段，降低单次 activation 峰值；有效 batch 的样本总量仍由全部 micro-batch 合计。
- **实现重点：** 先确定当前 micro-batch 的切片范围，再把输入和标签切出来。

**2. TODO 2 (缩放 loss 并反传)**

- **梯度对齐：** 每个 micro-batch 的 loss 必须先除以 `accum_steps`，再执行 `backward()`。
- **等价性：** 在相同 reduction、数据顺序和随机状态等条件下，这样累积出来的平均梯度才与完整 batch 接近，不会悄悄把更新幅度放大 `accum_steps` 倍。
- **实现重点：** 梯度路径先缩放再反传；返回值按原始 mean loss 口径汇总，便于和 full batch 日志比较。

**3. TODO 3 (统一更新参数并返回累计 loss)**

- **先攒后更：** 所有 micro-batch 都完成 backward 之后，再统一执行一次 `optimizer.step()` 和 `optimizer.zero_grad()`。
- **更新结果：** 只有在所有 micro-batch 完成 `backward()` 后执行一次 `optimizer.step()`，参数更新才与 full batch 的一次更新相对应；遇到随机层或不同 scheduler 节奏时，需要重新验证。
- **结果记录：** 最后返回按原始 mean loss 口径汇总的 `total_loss`，方便与 full batch 日志比较。

**4. 对照验证与 SFT 扩展**

- **一致性检查：** 使用同一批 `x / y` 和相同初始模型，对照两条路径的 loss、输出和参数，验证当前 toy 设置下的近似等价；结果不能直接推广到所有模型和训练配置。
- **工程价值：** 对照结果可确认当前 `MSELoss` 示例中的 loss、梯度累积和参数更新口径已经连接成一个可运行闭环；迁移到 SFT 时，再替换为 token-level loss 并重新验证。

- **SFT batch 扩展：** `input_ids`、`attention_mask`、`labels` 必须按同一个 `[start:end]` 范围同步切分。
- **有效 batch：** `effective_batch_size = micro_batch_size * accum_steps`，scheduler 和日志通常按一次 `optimizer.step()` 记录。
- **显存边界：** 梯度累积减少的是每个 micro-batch 的 activation 峰值，不会减少参数、梯度和优化器状态的长期占用。

## 相关阅读

完成本节后，可以按“机制实现 → 微调应用 → 性能测量”的顺序继续学习，比较显存、吞吐与更新口径。

### 机制实现
- [Hugging Face Accelerate：梯度累积指南](https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation)

### 微调应用
- [13. End-to-End Fine-Tuning Experiment | 端到端微调实验](../02_PyTorch_Algorithms/13_End_to_End_Fine_Tuning_Experiment.md)
- [60. LoRA Fine-Tuning Project | LoRA 微调项目](../02_PyTorch_Algorithms/60_LoRA_Fine_Tuning_Project.md)

### 性能测量
- [73. Training Performance Analysis | 训练性能分析](../02_PyTorch_Algorithms/73_Training_Performance_Analysis.md)
