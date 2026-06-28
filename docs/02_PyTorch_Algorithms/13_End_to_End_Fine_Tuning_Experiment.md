# 13. End to End Fine Tuning Experiment | 端到端微调实验

**难度：** Medium | **标签：** `训练闭环`, `SFT`, `PyTorch` | **目标人群：** 模型微调与工程部署

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/13_End_to_End_Fine_Tuning_Experiment.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


这一页把前面的训练要素串起来：数据构造、SFT Loss、梯度累积和参数更新。目标不是造一个大而全的框架，而是搭一个最小但完整的微调闭环，让你能把 2.3 组学到的内容真正跑通。

## 前置

**导语：** 先把 SFT、LoRA、WSD 和梯度累积都看过，再做端到端微调实验最顺。
- [Part 2: 09 SFT Training Loop](./09_SFT_Training_Loop.md)
- [Part 2: 10 LoRA Tutorial](./10_LoRA_Tutorial.md)
- [Part 2: 12 Gradient Accumulation](./12_Gradient_Accumulation.md)

## 相关阅读

**导语：** 完成训练收束后，可以继续进入对齐章节。
- [Part 2: 14 RLHF PPO Memory](./14_RLHF_PPO_Memory.md)
- [Part 2: 15 DPO Loss Tutorial](./15_DPO_Loss_Tutorial.md)


### Step 1: 端到端训练闭环长什么样

一个完整的微调实验通常包含四层：
1. **数据层**：把 prompt / response 整理成 `input_ids` 和 `labels`。
2. **模型层**：输入 token，输出每个位置的 logits。
3. **优化层**：计算 SFT loss，执行 backward、step 和 zero_grad。
4. **调度层**：可选地叠加学习率调度器和梯度累积。

这节我们用一个极小的语言模型，把这些步骤串成一个可训练的闭环。

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


```python
import copy
import torch
import torch.nn as nn

```


```python
def build_sft_batch(prompt_ids, response_ids, pad_id=0, max_len=10):
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
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def run_finetuning_experiment(model, optimizer, input_ids, labels, accum_steps=2, num_updates=40):
    """
    在同一批样本上反复训练，观察端到端训练闭环是否跑通。
    """
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
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise e

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
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def run_finetuning_experiment(model, optimizer, input_ids, labels, accum_steps=2, num_updates=40):
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

**1. 数据层先把 prompt / response 拼好**

端到端实验不是只管模型本身，第一步永远是把输入和标签整理正确。SFT 中最常见的错误不是模型写错，而是 `labels` 没有正确 mask prompt。

**2. 模型层只需要提供稳定的前向接口**

这里的 `TinyCausalLM` 很小，但它已经体现了真实微调的关键结构：embedding、序列建模和 token 级输出头。只要这个接口稳定，后面的 loss 和优化器就可以直接接上。

**3. 训练层把多个概念串成闭环**

这节把 `SFT Loss`、`梯度累积` 和 `参数更新` 串到一起，验证训练确实能够跑通并让 loss 下降。实际项目里再继续叠加学习率调度器、混合精度、分布式训练即可。

**4. 为什么要做重复样本验证**

重复样本能快速检验训练闭环是否有问题。如果模型连一个可过拟合的小 batch 都学不会，通常说明：
- loss 对齐有问题；
- labels mask 有问题；
- 反向传播或优化器调用顺序有问题。
