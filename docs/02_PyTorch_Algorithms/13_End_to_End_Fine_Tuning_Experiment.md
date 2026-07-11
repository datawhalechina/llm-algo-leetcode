# 13. End to End Fine Tuning Experiment | 端到端微调实验

**难度：** Medium | **环境：** CPU-first | **标签：** `训练闭环`, `SFT`, `PyTorch` | **目标人群：** 模型微调与工程部署

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/13_End_to_End_Fine_Tuning_Experiment.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


这一页把前面的训练要素串起来：数据构造、SFT Loss、梯度累积和参数更新。目标不是造一个大而全的框架，而是搭一个最小但完整的微调闭环，让你能把 2.3 组学到的内容真正跑通。

**关键词：** `end-to-end`, `fine-tuning`, `loop`
## 前置阅读

**导语：** 先把 SFT、LoRA、训练闭环和显存优化相关的具体小节看过，再做端到端微调实验最顺。
- [09. PyTorch nn.Module Basics | PyTorch nn.Module 基础](./09_PyTorch_nn_Module_Basics.md)
- [11. PyTorch Optimizers and Loss | PyTorch 优化器与损失](./11_PyTorch_Optimizers_and_Loss.md)
- [13. Simple Neural Network Training | 简单神经网络训练](./13_Simple_Neural_Network_Training.md)

## 相关阅读

**导语：** 完成训练收束后，可以继续进入显存与并行、性能分析相关的小节。
- [06. VRAM Calculation and ZeRO | 显存计算与 ZeRO](./06_VRAM_Calculation_and_ZeRO.md)
- [13. Profiling and Bottleneck Analysis | Profiling 与瓶颈分析](./13_Profiling_and_Bottleneck_Analysis.md)
- [20. NCCL and AllReduce Basics | NCCL 与 AllReduce 基础](./20_NCCL_and_AllReduce_Basics.md)

### Step 1: 端到端训练闭环长什么样

一个完整的微调实验通常包含四层：
1. **数据层**：把 prompt / response 整理成 `input_ids` 和 `labels`。
2. **模型层**：输入 token，输出每个位置的 logits。
3. **优化层**：计算 SFT loss，执行 backward、step 和 zero_grad。
4. **调度层**：可选地叠加学习率调度器和梯度累积。

这节我们用一个极小的语言模型，把这些步骤串成一个可训练的闭环。后面的 `TODO 1-3` 会分别把这四层拆开实现，再重新合回一个训练闭环。

### Step 2: 为什么要把它做成实验

如果只会单点函数，很容易在面试或真实项目里出现“会公式，不会落地”的问题。端到端实验的价值在于：
- 你能确认数据、模型、loss、优化器之间的接口是对的。
- 你能观察训练 loss 是否真的下降。
- 你能快速定位是数据问题、loss 问题，还是优化器问题。

### Step 3: 代码实现框架

下面会实现三个函数：
- `build_sft_batch`：构造一批 SFT 样本。
- `TinyCausalLM`：一个很小的自回归模型。
- `run_finetuning_experiment`：把数据、loss、梯度累积和参数更新串起来。
这三块正好对应后面的 `TODO 1 / TODO 2 / TODO 3`。


```python
import copy
import torch
import torch.nn as nn

```


```python
def build_sft_batch(prompt_ids, response_ids, pad_id=0, max_len=10):
    # ==========================================
    # TODO 1: 构造 SFT 样本
    # 提示: prompt 部分的 labels 需要 mask 成 -100，response 部分保留原标签
    # ==========================================
    # input_ids = ???
    # labels = ???

    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]
    else:
        pad_len = max_len - len(input_ids)
        input_ids = input_ids + [pad_id] * pad_len
        labels = labels + [-100] * pad_len

    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=64, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        hidden, _ = self.rnn(x)
        logits = self.lm_head(hidden)
        return logits


def compute_sft_loss(logits, labels):
    # ==========================================
    # TODO 2: 对齐 next-token 预测并计算 SFT loss
    # 提示: 先 shift logits / labels，再用 CrossEntropyLoss(ignore_index=-100)
    # ==========================================
    # shift_logits = ???
    # shift_labels = ???
    # loss = ???
    return loss


def run_finetuning_experiment(model, optimizer, input_ids, labels, accum_steps=2, num_updates=40):
    """
    在同一批样本上反复训练，观察端到端训练闭环是否跑通。
    """
    if input_ids.size(0) % accum_steps != 0:
        raise ValueError("batch size 必须能被 accum_steps 整除")

    # ==========================================
    # TODO 3: 端到端训练闭环
    # 提示: 切 micro-batch -> 缩放 loss 并 backward -> 最后 step / zero_grad / 返回 history
    # ==========================================
    history = []
    micro_size = input_ids.size(0) // accum_steps

    for _ in range(num_updates):
        model.train()
        optimizer.zero_grad()

        total_loss = 0.0
        for idx in range(accum_steps):
            # mb_input = ???
            # mb_labels = ???
            # logits = ???
            # loss = ???
            # total_loss = ???
            pass

        optimizer.step()
        history.append(total_loss)

    return history
```


```python
# 运行此单元格以测试你的实现
def test_end_to_end_finetuning():
    try:
        torch.manual_seed(7)

        prompt = [1, 2, 3]
        response = [4, 5, 6, 7]
        single_input, single_labels = build_sft_batch(prompt, response, pad_id=0, max_len=8)

        # 构造一个 batch，重复同一条样本，便于快速过拟合并验证训练闭环
        input_ids = single_input.unsqueeze(0).repeat(4, 1)
        labels = single_labels.unsqueeze(0).repeat(4, 1)

        model = TinyCausalLM(vocab_size=64, hidden_size=32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)

        with torch.no_grad():
            init_loss = compute_sft_loss(model(input_ids), labels).item()

        history = run_finetuning_experiment(model, optimizer, input_ids, labels, accum_steps=2, num_updates=30)

        final_loss = compute_sft_loss(model(input_ids), labels).item()
        print(f"Initial loss: {init_loss:.4f}")
        print(f"Final loss  : {final_loss:.4f}")

        assert len(history) == 30, "训练步数不对"
        assert final_loss < init_loss, "训练没有让 loss 下降，闭环可能有问题"

        print("✅ 测试通过！端到端微调闭环运行正常，loss 已下降。")
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

test_end_to_end_finetuning()
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


def build_sft_batch(prompt_ids, response_ids, pad_id=0, max_len=10):
    # TODO 1: 构造 SFT 样本
    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids

    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]
    else:
        pad_len = max_len - len(input_ids)
        input_ids = input_ids + [pad_id] * pad_len
        labels = labels + [-100] * pad_len

    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=64, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        hidden, _ = self.rnn(x)
        logits = self.lm_head(hidden)
        return logits


def compute_sft_loss(logits, labels):
    # TODO 2: 对齐 next-token 预测并计算 SFT loss
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def run_finetuning_experiment(model, optimizer, input_ids, labels, accum_steps=2, num_updates=40):
    # TODO 3: 端到端训练闭环
    if input_ids.size(0) % accum_steps != 0:
        raise ValueError("batch size 必须能被 accum_steps 整除")

    history = []
    micro_size = input_ids.size(0) // accum_steps

    for _ in range(num_updates):
        model.train()
        optimizer.zero_grad()

        total_loss = 0.0
        for idx in range(accum_steps):
            mb_input = input_ids[idx * micro_size:(idx + 1) * micro_size]
            mb_labels = labels[idx * micro_size:(idx + 1) * micro_size]
            logits = model(mb_input)
            loss = compute_sft_loss(logits, mb_labels) / accum_steps
            loss.backward()
            total_loss += loss.detach().item()

        optimizer.step()
        history.append(total_loss)

    return history
```

### 解析

**1. TODO 1 (构造 SFT 样本)**

- **拼接输入：** `input_ids` 由 `prompt + response` 拼接得到，保持样本的完整上下文。
- **监督标签：** `labels` 里，`prompt` 对应的位置要 mask 成 `-100`，只让模型学习回答部分。
- **长度处理：** 超过 `max_len` 时要裁剪，不足时要补 `pad_id` 和 `-100`。
- **训练目标：** SFT 关注的是“模型对回答部分的预测能力”，而不是复述提示词本身。

**2. TODO 2 (对齐 next-token 并计算 SFT loss)**

- **一位错位：** `shift_logits = logits[..., :-1, :]`，`shift_labels = labels[..., 1:]`。
- **损失函数：** 使用 `CrossEntropyLoss(ignore_index=-100)` 计算 loss，让 prompt 和 padding 位置自然忽略。
- **监督范围：** 训练信号只来自 response 的有效 token，next-token 对齐要和 causal LM 的训练目标一致。
- **形状检查：** 这一层本质上是在确认 logits 和 labels 的时间步是否对齐。

**3. TODO 3 (训练闭环)**

- **micro-batch：** 先把 batch 切成多个 `micro-batch`，再逐个累积梯度。
- **loss 缩放：** 每个 `micro-batch` 的 loss 要除以 `accum_steps`，保证和完整 batch 的梯度一致。
- **参数更新：** 所有 `micro-batch` 处理完之后，再统一执行 `optimizer.step()` 和 `optimizer.zero_grad()`。
- **训练记录：** 最后返回 `history`，方便观察训练过程中 loss 是否下降。

**进阶思考：为什么要做重复样本验证？**

- **一致性检查：** 通过重复样本验证，可以确认梯度累积是否真的等价于完整 batch。
- **闭环意义：** 这一步把 `SFT Loss`、`梯度累积`、`参数更新` 连接成一个可运行的小闭环。
- **工程价值：** 只要这套链路对齐，后续再切换更复杂的数据和更大的 batch 也更稳。