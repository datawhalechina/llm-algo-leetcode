# 13. End to End Fine Tuning Experiment | 端到端微调实验

**难度：** Medium | **环境：** CPU-first / GPU optional | **标签：** `训练微调`, `SFT`, `训练闭环` | **目标人群：** 训练机制学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/13_End_to_End_Fine_Tuning_Experiment.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

前面的小节已经分别讲过模型封装、优化器、损失函数和梯度累积，但真实微调不是把这些概念单独跑通就结束。只要数据构造、label 对齐、loss 计算或参数更新里有一个环节接错，训练就会表现成 loss 不降、shape 对不上，或者看似运行但模型没有真正学习。

本节把这些训练要素收成一个最小端到端 SFT 实验：先构造 train / val 样本，再计算自回归 loss，最后走完 backward、梯度累积、optimizer step 和周期性评估。它是本部分第一个训练闭环小项目：前面分别实现训练组件，这里验证组件能否共同产出可解释的训练结果；后续第 64 节先检查数据准入，第 62 节验证指令微调任务，第 60 节再比较全参数更新与 LoRA 适配是否值得交付。

主线使用 CPU 即可完成；有 GPU 时可选运行 Step 5 的真实 SFT smoke，用于确认真实模型和真实数据能够走通训练闭环。GPU 实验的配置、自动下载方式和证据边界见 Step 5；环境安装与预检见[使用指南](../docs/guide.md)。

**关键词：** `end-to-end`, `fine-tuning`, `train/val`, `report`

---
## 前置阅读

**导语：** 先掌握 SFT 数据与训练循环、优化器与 loss、梯度累积这三个直接前置，再做端到端微调实验。
- [09. SFT Training Loop | 监督微调训练循环](../02_PyTorch_Algorithms/09_SFT_Training_Loop.md)
- [P0: 11. PyTorch Optimizers and Loss | PyTorch 优化器与损失](../00_Prerequisites/11_PyTorch_Optimizers_and_Loss.md)
- [12. Gradient Accumulation | 梯度累积](../02_PyTorch_Algorithms/12_Gradient_Accumulation.md)

## 相关阅读

**导语：** 完成最小 SFT 闭环后，下一步最自然的是把它推进到 LoRA 项目、指令微调项目和训练性能分析里。
- [60. LoRA Fine-Tuning Project | LoRA 微调项目](../02_PyTorch_Algorithms/60_LoRA_Fine_Tuning_Project.md)
- [62. Instruction Fine-Tuning Project | 指令微调项目](../02_PyTorch_Algorithms/62_Instruction_Fine_Tuning_Project.md)
- [73. Training Performance Analysis | 训练性能分析](../02_PyTorch_Algorithms/73_Training_Performance_Analysis.md)
- [74. Profiling-Driven End-to-End Optimization | Profiling 驱动的端到端优化](../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.md)

---
### Step 1: 端到端训练闭环长什么样
端到端微调实验的核心，是把数据、模型、loss、优化器和评估接成一个可运行闭环。

一个完整的微调实验通常包含五层：
1. **数据层**：将 prompt/response 构造为 tokenized batch（input_ids + attention_mask + labels），并进行 padding 对齐，作为模型的直接输入。
2. **模型层**：输入 token 经过 embedding -> Transformer / RNN -> LM head，输出每个位置的 logits。
3. **优化层**：计算 SFT loss，执行 backward、step 和 zero_grad。
4. **训练控制层**：控制梯度累积、参数更新频率和 loss 记录。
5. **评估层**：在训练中定期记录 train / val loss，并用小样本 overfit 检查确认闭环真的接通。

前面的训练组件分别解决了局部问题；本节把它们放进一个极小语言模型的完整训练循环，观察数据如何经过前向、loss、反向传播和参数更新，并在 train / val 中形成可检查的报告。模型已经给出，具体实现任务见 Step 4。

![端到端微调的五个组成部分](../docs/public/02_PyTorch_Algorithms/13_training_components.svg)

<div align="center"><strong>端到端微调的组成部分：</strong> 图片按功能拆开数据、模型、优化、训练控制和评估五个部分，帮助你建立后续代码的阅读索引。</div>

### Step 2: 运行前先想清楚要观察什么

这次实验的任务不是调出一个有代表性的模型，而是确认训练链路真的在工作；因此先不调学习率，运行后按下表依次检查四个现象。重复样本只用于验证闭环，val loss 也不代表真实任务的泛化能力；正式项目还需要独立验证集、任务指标和样例回归。

| 先看什么 | 正常现象 | 它说明了什么 |
|:---|:---|:---|
| 输入和输出 | batch、logits、labels 的 shape 能对上 | 数据可以进入 loss |
| 参数和状态 | `optimizer.step()` 后参数变化，optimizer 产生 state | 梯度和优化器已经接上 |
| 训练曲线 | 重复样本上的 train loss 下降 | 模型确实在利用梯度学习 |
| 评估结果 | train / val 能用同一口径计算 | 训练和评估流程没有断开 |

如果某一步没有出现预期现象，就回到表格中的对应环节排查。

### Step 3: 先读懂四个接口各自负责什么

先不要修改模型。`TinyCausalLM` 只是一个可快速运行的验证模型；本节真正要接通的是它周围的四个接口：

| 接口 | 负责的事情 | 你要关注的连接 |
|:---|:---|:---|
| `build_sft_batch` / `collate_sft_batch` | 把 prompt、response 变成 batch | 哪些 token 参与监督，哪些位置被 mask |
| `compute_sft_loss` | 对齐 next-token 预测并计算 loss | logits、labels 和 padding 是否错开一位 |
| `evaluate_loss` | 在不更新参数的情况下计算 loss | `eval()` 和 `no_grad()` 的作用范围 |
| `run_finetuning_experiment` | 累积梯度、更新参数并记录报告 | 一个 update 如何串起 forward 到 train / val |

下面的图把这些接口之间的数据流展开；阅读时重点看每个接口的输入、输出，以及它在训练闭环中的位置。

![端到端训练闭环](../docs/public/02_PyTorch_Algorithms/13_training_loop.svg)

<div align="center"><strong>端到端训练闭环：</strong> 图中箭头表示 batch、loss 和参数在训练闭环中的数据或控制流，不表示运行时间、显存占用或性能比例。</div>

### Step 4：完成 TODO，并用结果检查闭环

现在回到题目区，按“数据构造 → loss → 评估 → 训练报告”的顺序完成 4 个 TODO。不要改动 `TinyCausalLM`；每完成一个接口，先看它的返回值是否符合下一个接口的输入。最后运行测试，确认 loss 能计算、重复样本上的 train loss 会下降、参数确实更新，并生成最小报告。


```python
import torch
import torch.nn as nn
```


```python

def build_sft_batch(prompt_ids, response_ids, pad_id=0, eos_id=2, max_len=10):
    """拼接一条 SFT 样本，并返回定长的输入、掩码和监督标签。

    prompt 只提供上下文，response + EOS 才是监督目标；padding 不应产生 loss。
    """
    # ==========================================
    # TODO 1: 构造单条 SFT 样本
    # 提示：先拼接 prompt 与 response_with_eos；prompt 的 labels 填 -100，
    #       response/EOS 保留原 token。截断后必须仍有一个有效监督 token。
    #       最后右侧 padding 到 max_len，并返回三个等长的 long tensor。
    # ==========================================
    response_with_eos = response_ids + [eos_id]
    # input_ids = ??? prompt_ids + response_with_eos
    # labels = ??? prompt 部分填 -100，response_with_eos 保留原 token

    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]
    if not any(label != -100 for label in labels):
        raise ValueError("截断后没有有效监督 token")

    # attention_mask = ??? 真实 token 为 1，padding 为 0
    # pad_len = ??? max_len - len(input_ids)
    # input_ids = ??? 在右侧补 pad_id
    # attention_mask = ??? 在右侧补 0
    # labels = ??? 在右侧补 -100

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def collate_sft_batch(samples, pad_id=0, eos_id=2, max_len=10):
    """将多条定长 SFT 样本堆叠成 batch。"""
    items = [build_sft_batch(prompt, response, pad_id=pad_id, eos_id=eos_id, max_len=max_len) for prompt, response in samples]
    return {key: torch.stack([item[key] for item in items], dim=0) for key in items[0]}


class TinyCausalLM(nn.Module):
    """用于验证训练接口的最小语言模型；本题不要求修改模型结构。"""
    def __init__(self, vocab_size=64, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        hidden, _ = self.rnn(x)
        logits = self.lm_head(hidden)
        return logits


def compute_sft_loss(logits, labels, attention_mask=None):
    """按 causal LM 的 next-token 对齐规则计算 SFT loss。"""
    # ==========================================
    # TODO 2: 对齐 next-token 预测并计算 SFT loss
    # 提示：logits 取前 t-1 个位置，labels 取后 t-1 个位置；
    #       attention_mask 只需作用在目标 label 位置，并继续使用 -100 忽略。
    #       使用 CrossEntropyLoss(ignore_index=-100)，返回一个标量 loss。
    # ==========================================
    # shift_logits = ??? 保留前 t-1 个位置
    # shift_labels = ??? 从第 2 个 token 开始对齐
    # if attention_mask is not None:
    #     shift_attention_mask = ??? 取目标 token 对应的 mask
    #     shift_labels = ??? 将 padding 对应位置改为 -100
    # if 没有任何 shift_labels != -100:  # 应主动拒绝空监督 batch
    #     raise ValueError(???)
    # loss = ??? CrossEntropyLoss(ignore_index=-100)(...)
    return loss


def evaluate_loss(model, batch):
    """在不记录梯度的条件下，用统一口径计算一个 batch 的 loss。"""
    # ==========================================
    # TODO 3: 在 eval 模式下计算 batch loss
    # 提示：切换到 eval 模式并关闭梯度记录；调用同一个 loss 函数，
    #       保持训练和验证的 label / padding 口径一致，返回 Python float。
    # ==========================================
    # model.eval()
    # with torch.no_grad():
    #     logits = ???
    #     loss = ???
    # return ???
    pass


def run_finetuning_experiment(model, optimizer, train_batch, val_batch=None, accum_steps=2, num_updates=40, eval_every=10):
    """
    在小批样本上训练，验证梯度累积、参数更新和 train / val 报告。

    每次 optimizer.step() 前处理 accum_steps 个等大的 micro-batch。
    """
    if train_batch["input_ids"].size(0) % accum_steps != 0:
        raise ValueError("batch size 必须能被 accum_steps 整除")

    # ==========================================
    # TODO 4: 端到端训练闭环与报告
    # 提示：先记录初始 train/val loss；每次 update 切分等大的 micro-batch，
    #       累积梯度后只调用一次 optimizer.step()，并在指定节点追加 history。
    # ==========================================
    report = {
        "initial_train_loss": evaluate_loss(model, train_batch),
        "initial_val_loss": evaluate_loss(model, val_batch) if val_batch is not None else None,
        "final_train_loss": None,
        "final_val_loss": None,
        "history": [],
    }
    # micro_size = ??? train_batch 大小 / accum_steps
    for step in range(1, num_updates + 1):
        model.train()
        optimizer.zero_grad()
        for idx in range(accum_steps):
            # mb = ??? 取出第 idx 个 micro-batch
            # logits = ??? 调用模型得到输出
            # loss = ??? 当前 micro-batch 的 loss / accum_steps
            # loss.backward()
            pass
        # optimizer.step()
        # if ??? 第 1 步、评估间隔或最后一步:
        #     report["history"].append(...)
    # report["final_train_loss"] = ???
    # report["final_val_loss"] = ???
    # return report
    pass

```


```python
# 运行此单元格以测试你的实现
def test_end_to_end_finetuning():
    try:
        torch.manual_seed(7)

        train_samples = [
            ([1, 2, 3], [4, 5, 6, 7]),
            ([1, 2, 3], [4, 5, 6, 7]),
            ([1, 2, 3], [4, 5, 6, 7]),
            ([1, 2, 3], [4, 5, 6, 7]),
        ]
        val_samples = [
            ([1, 2, 3], [4, 5, 6, 7]),
            ([1, 2, 3], [4, 5, 6, 7]),
        ]

        train_batch = collate_sft_batch(train_samples, pad_id=0, eos_id=2, max_len=9)
        val_batch = collate_sft_batch(val_samples, pad_id=0, eos_id=2, max_len=9)

        assert train_batch["input_ids"].shape == (4, 9), "train batch shape 错误"
        assert train_batch["attention_mask"].sum().item() == 32, "attention_mask 统计错误"
        assert torch.any(train_batch["labels"] == -100), "prompt/padding 应该被 mask"
        assert torch.all(train_batch["labels"][:, :3] == -100), "prompt token 不应参与监督"
        assert torch.all(train_batch["attention_mask"][:, 8] == 0), "padding 的 attention_mask 应为 0"
        assert torch.all(train_batch["labels"][:, 8] == -100), "padding 不应参与 loss"

        truncated = build_sft_batch([1, 2, 3, 4], [5, 6], max_len=5)
        assert truncated["input_ids"].tolist() == [1, 2, 3, 4, 5], "截断顺序错误"
        assert truncated["labels"].tolist() == [-100, -100, -100, -100, 5], "截断后监督标签错误"
        try:
            build_sft_batch([1, 2, 3, 4], [5], max_len=4)
        except ValueError as exc:
            assert "有效监督" in str(exc), "无监督样本的错误提示不清晰"
        else:
            raise AssertionError("截断后没有监督 token 时应报错")

        torch.manual_seed(9)
        probe_logits = torch.randn(1, 4, 8)
        probe_labels = torch.tensor([[-100, 2, 3, -100]])
        probe_mask = torch.tensor([[1, 1, 1, 0]])
        actual_loss = compute_sft_loss(probe_logits, probe_labels, probe_mask)
        expected_labels = probe_labels[:, 1:].masked_fill(probe_mask[:, 1:] == 0, -100)
        expected_loss = nn.CrossEntropyLoss(ignore_index=-100)(
            probe_logits[:, :-1, :].reshape(-1, 8), expected_labels.reshape(-1)
        )
        assert torch.allclose(actual_loss, expected_loss), "next-token shift 或 padding mask 错误"

        model = TinyCausalLM(vocab_size=64, hidden_size=32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
        before_update = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

        report = run_finetuning_experiment(
            model,
            optimizer,
            train_batch,
            val_batch=val_batch,
            accum_steps=2,
            num_updates=30,
            eval_every=10,
        )

        changed = any(not torch.equal(before_update[name], parameter.detach()) for name, parameter in model.named_parameters())
        assert changed, "optimizer.step() 后至少应有一组参数发生变化"
        assert len(optimizer.state) > 0, "完成 optimizer.step() 后应生成 optimizer state"

        print(f"Initial train loss: {report['initial_train_loss']:.4f}")
        print(f"Final train loss  : {report['final_train_loss']:.4f}")
        print(f"Initial val loss  : {report['initial_val_loss']:.4f}")
        print(f"Final val loss    : {report['final_val_loss']:.4f}")
        print(f"History           : {report['history']}")

        assert len(report["history"]) >= 3, "训练过程应该至少记录 3 次评估"
        assert report["final_train_loss"] < report["initial_train_loss"], "训练没有让 train loss 下降"
        assert report["final_val_loss"] < report["initial_val_loss"], "训练没有让 val loss 下降"
        assert report["final_train_loss"] < 0.2, "重复样本过拟合不充分，闭环可能有问题"

        print(f"Optimizer updated parameters: {changed}; state entries: {len(optimizer.state)}")
        print("✅ 测试通过！端到端微调闭环、参数更新、评估和报告均运行正常。")
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
#### CPU 实验报告

`report` 是 `run_finetuning_experiment(...)` 返回的 Python 字典，包含初始 / 最终 train、val loss 和 `history`。运行题目区与测试区后，复制 `report` 和测试输出，填写下表。

运行题目区 → 运行测试区 → 复制结果 → 填写报告 → 写出结论

| 检查项 | 来源 | 记录内容 | CPU 实验说明 |
|:---|:---|:---|:---|
| 训练曲线 | `report` | `initial_train_loss` → `final_train_loss` | 重复样本上的 train loss 应总体下降；只说明训练链路接通 |
| 验证口径 | `report` | `initial_val_loss` → `final_val_loss` | val loss 能计算，且复用同一套 loss 规则 |
| 评估记录 | `report["history"]` | 第 1 步、间隔步和最后一步的 train / val loss | 检查评估记录逻辑，不代表泛化能力 |
| 参数更新 | 测试输出 | `Optimizer updated parameters` | 为 `True`，证明 `optimizer.step()` 生效 |
| 优化器状态 | 测试输出 | `state entries` | 大于 `0`，证明优化器状态已建立 |
| 测试状态 | 测试区提示 | 端到端闭环测试结果 | 应为通过 |

可直接使用下面的简短格式记录：

```text
实验环境：CPU；模型：TinyCausalLM；数据：重复样本
训练结果：train loss ______ → ______；val loss ______ → ______
闭环检查：参数更新 ______；optimizer state entries ______；测试 ______
结论：CPU 训练闭环 ______，依据是 ____________________。
证据边界：本报告不说明真实 GPU 显存、吞吐或任务泛化能力。
```
(注意：本报告只验证 CPU 上的训练闭环；真实模型、GPU 显存和训练耗时见 Step 5，不能与本表数值直接比较。)
## 参考代码与解析

### 代码


```python
import torch
import torch.nn as nn


def build_sft_batch(prompt_ids, response_ids, pad_id=0, eos_id=2, max_len=10):
    # TODO 1: 构造单条 SFT 样本
    """拼接一条 SFT 样本，并返回定长的输入、掩码和监督标签。"""
    # 提示：先拼接 prompt 与 response_with_eos；prompt 的 labels 填 -100，response/EOS 保留原 token。
    #       截断后必须仍有一个有效监督 token；最后右侧 padding 到 max_len。
    response_with_eos = response_ids + [eos_id]
    input_ids = prompt_ids + response_with_eos
    labels = [-100] * len(prompt_ids) + response_with_eos

    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]
    if not any(label != -100 for label in labels):
        raise ValueError("截断后没有有效监督 token")

    attention_mask = [1] * len(input_ids)
    pad_len = max_len - len(input_ids)
    if pad_len > 0:
        input_ids = input_ids + [pad_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len
        labels = labels + [-100] * pad_len

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def collate_sft_batch(samples, pad_id=0, eos_id=2, max_len=10):
    """将多条定长 SFT 样本堆叠成 batch。"""
    items = [build_sft_batch(prompt, response, pad_id=pad_id, eos_id=eos_id, max_len=max_len) for prompt, response in samples]
    return {key: torch.stack([item[key] for item in items], dim=0) for key in items[0]}


class TinyCausalLM(nn.Module):
    """用于验证训练接口的最小语言模型；本题不要求修改模型结构。"""
    def __init__(self, vocab_size=64, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        hidden, _ = self.rnn(x)
        logits = self.lm_head(hidden)
        return logits


def compute_sft_loss(logits, labels, attention_mask=None):
    # TODO 2: 对齐 next-token 预测并计算 SFT loss
    """按 causal LM 的 next-token 对齐规则计算 SFT loss。"""
    # 提示：logits 取前 t-1 个位置，labels 取后 t-1 个位置；padding 使用 -100 忽略。
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    if attention_mask is not None:
        shift_attention_mask = attention_mask[..., 1:].contiguous()
        shift_labels = shift_labels.masked_fill(shift_attention_mask == 0, -100)
    if not torch.any(shift_labels != -100):
        raise ValueError("当前 batch 没有有效监督 token")

    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss


def evaluate_loss(model, batch):
    # TODO 3: 在 eval 模式下计算 batch loss
    """在不记录梯度的条件下，用统一口径计算一个 batch 的 loss。"""
    # 提示：使用 eval() 和 no_grad()；复用 compute_sft_loss，并返回 Python float。
    model.eval()
    with torch.no_grad():
        logits = model(batch["input_ids"], batch.get("attention_mask"))
        loss = compute_sft_loss(logits, batch["labels"], batch.get("attention_mask"))
    return loss.item()


def run_finetuning_experiment(model, optimizer, train_batch, val_batch=None, accum_steps=2, num_updates=40, eval_every=10):
    # TODO 4: 端到端训练闭环与报告
    """在小批样本上训练，验证梯度累积、参数更新和 train / val 报告。"""
    # 提示：每次 update 切分等大的 micro-batch；累积梯度后只调用一次 step()，再按间隔记录 history。
    if train_batch["input_ids"].size(0) % accum_steps != 0:
        raise ValueError("batch size 必须能被 accum_steps 整除")

    report = {
        "initial_train_loss": evaluate_loss(model, train_batch),
        "initial_val_loss": evaluate_loss(model, val_batch) if val_batch is not None else None,
        "final_train_loss": None,
        "final_val_loss": None,
        "history": [],
    }
    micro_size = train_batch["input_ids"].size(0) // accum_steps

    for step in range(1, num_updates + 1):
        model.train()
        optimizer.zero_grad()

        for idx in range(accum_steps):
            start = idx * micro_size
            end = (idx + 1) * micro_size
            mb = {key: value[start:end] for key, value in train_batch.items()}
            logits = model(mb["input_ids"], mb.get("attention_mask"))
            loss = compute_sft_loss(logits, mb["labels"], mb.get("attention_mask")) / accum_steps
            loss.backward()

        optimizer.step()

        if step == 1 or step % eval_every == 0 or step == num_updates:
            record = {"step": step, "train_loss": evaluate_loss(model, train_batch)}
            if val_batch is not None:
                record["val_loss"] = evaluate_loss(model, val_batch)
            report["history"].append(record)

    report["final_train_loss"] = evaluate_loss(model, train_batch)
    report["final_val_loss"] = evaluate_loss(model, val_batch) if val_batch is not None else None
    return report

```

### 解析

本节把 SFT batch、next-token loss、评估和梯度累积接成一个最小训练闭环。解析按题目区的 TODO 顺序展开。

**1. TODO 1：构造 SFT batch**

- **实现方式**：`input_ids` 依次拼接 `prompt`、`response` 和 `EOS`；`labels` 对 prompt 位置填 `-100`，对 response 和 EOS 保留 token。
- **padding 规则**：真实 token 的 `attention_mask` 为 `1`，padding 为 `0`；padding 对应的 label 也填 `-100`。
- **长度边界**：先按 `max_len` 截断，再右侧 padding；截断后若没有有效监督 token，应主动报错。

**2. TODO 2：计算 SFT loss**

- **next-token 对齐**：`shift_logits = logits[..., :-1, :]`，`shift_labels = labels[..., 1:]`，让当前位置的输出预测下一个 token。
- **mask 处理**：只在目标 label 位置应用 `attention_mask`，将 padding 改为 `-100`；`CrossEntropyLoss(ignore_index=-100)` 会忽略 prompt 和 padding。
- **防御检查**：如果 shift 后没有任何有效 label，直接报错，避免返回没有训练意义的 loss。

**3. TODO 3：评估 loss**

- **评估模式**：调用 `model.eval()`，并在 `torch.no_grad()` 中完成前向，避免建立反向图。
- **保持同口径**：训练和验证都调用 `compute_sft_loss`，因此使用相同的 shift、label mask 和 padding 规则。
- **返回值**：报告只需要 Python float；真实项目还要增加任务指标和样例回归。

**4. TODO 4：训练闭环与报告**

- **梯度累积**：将 batch 切成 `accum_steps` 个 micro-batch，每次反向前把 loss 除以 `accum_steps`。
- **更新时机**：所有 micro-batch 完成反向传播后，只调用一次 `optimizer.step()`，然后清理下一次 update 的梯度。
- **报告记录**：第 1 步、固定间隔和最后一步记录 train / val loss，形成可检查的 `history`。
- **优化器检查**：测试区同时确认参数发生变化、`optimizer.state` 已生成，分别验证更新路径和优化器状态的建立。

**测试边界**

测试中的重复样本只用于检查数据构造、loss 对齐、梯度累积和参数更新是否接通；loss 下降不代表真实任务泛化能力。真实微调还需要独立验证集、任务指标、样例回归和错误案例分析。

### Step 5：在真实 GPU 上跑通最小 SFT

完成 Step 4 并通过 CPU 测试后，再运行本步。它用真实模型和真实指令数据走通一次最小 SFT 链路，检查模型加载、监督标签、反向传播、参数更新和结果保存；它不是性能 benchmark，也不能据此判断训练质量。没有 GPU 时保持 `RUN_GPU_SMOKE = False`。

| 环境 | 内容 |
|:---|:---|
| GPU + 监督微调环境 | CUDA PyTorch、`transformers`、`datasets`、`accelerate`；环境预检和安装方式见[使用指南](../docs/guide.md)与 60 节实验入口 |
| 默认组合 | `qwen25_small` + `alpaca`，与 60 节一致，适合 T4、12GB 笔记本 GPU 和 Colab smoke |
| 可选组合 | `qwen25_medium`、`deepseek_r1_small`；用于观察模型规模或模型类型变化，不与默认结果直接横比 |
| 数据选择 | `alpaca` 或 `alpaca_cleaned`，都按 `instruction / input / output` 字段读取 |
| 下载与产物 | 自动下载或复用缓存，不手填路径；实际 profile、ID、dtype 和指标写入 `benchmarks/results/13_real_gpu_sft.json` |
| 证据范围 | 真实模型 / 真实数据的 CUDA smoke，不是性能 benchmark 或质量结论 |


```python
# GPU smoke 配置：只修改本单元，然后运行下一个实验单元。
RUN_GPU_SMOKE = False  # True 才会下载/加载真实模型并占用 GPU。
GPU_MODEL_PROFILE = 'qwen25_small'  # qwen25_small / qwen25_medium / deepseek_r1_small。
GPU_DATASET_PROFILE = 'alpaca'  # alpaca / alpaca_cleaned。

# 默认配置适合先验证链路；增大样本、序列长度或模型后，结果不再是同口径 smoke。
GPU_MAX_SAMPLES = 1  # 下载后实际取用的样本数。
GPU_MAX_LENGTH = 256  # tokenizer 截断长度，影响输入张量和显存。
GPU_UPDATES = 2  # optimizer 更新次数，只用于检查训练闭环。
GPU_BATCH_SIZE = 1  # 当前 smoke 实现一次处理一个 batch。

GPU_MODEL_PROFILES = {
    'qwen25_small': {'model_id': 'Qwen/Qwen2.5-0.5B-Instruct', 'purpose': 'default_sft_smoke'},
    'qwen25_medium': {'model_id': 'Qwen/Qwen2.5-1.5B-Instruct', 'purpose': 'model_scale_extension'},
    'deepseek_r1_small': {'model_id': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', 'purpose': 'reasoning_model_extension'},
}
GPU_DATASET_PROFILES = {
    'alpaca': {'dataset_id': 'tatsu-lab/alpaca', 'purpose': 'default_instruction_sft'},
    'alpaca_cleaned': {'dataset_id': 'yahma/alpaca-cleaned', 'purpose': 'cleaned_instruction_sft'},
}

```


```python
import json
import torch
import time
import sys
from pathlib import Path

# 配置来自上一个单元；这里不重复定义，避免两个单元的参数发生漂移。
if GPU_MODEL_PROFILE not in GPU_MODEL_PROFILES:
    raise ValueError(f'未知 GPU_MODEL_PROFILE: {GPU_MODEL_PROFILE}')
if GPU_DATASET_PROFILE not in GPU_DATASET_PROFILES:
    raise ValueError(f'未知 GPU_DATASET_PROFILE: {GPU_DATASET_PROFILE}')
GPU_MODEL_ID = GPU_MODEL_PROFILES[GPU_MODEL_PROFILE]['model_id']
GPU_DATASET_ID = GPU_DATASET_PROFILES[GPU_DATASET_PROFILE]['dataset_id']
def _find_project_root():
    """从当前目录向上查找仓库根目录，避免学习者手填绝对路径。"""
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / 'tools' / 'model_runtime.py').exists():
            return candidate
    raise RuntimeError('未找到项目根目录，请从仓库中的 Notebook 启动。')

if not RUN_GPU_SMOKE:
    print('跳过可选 GPU 验证：保持 CPU-first 主线。')
elif not torch.cuda.is_available():
    raise RuntimeError('RUN_GPU_SMOKE=True 但 CUDA 不可用，请先检查 GPU 环境。')
else:
    project_root = _find_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from tools.model_runtime import resolve_model
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 自动复用项目缓存；缓存不存在时由 model_runtime 下载模型。
    model_path = resolve_model(GPU_MODEL_ID, source='auto', cache_dir=project_root / 'model_cache')
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # datasets 同样会优先使用本地缓存；只取少量样本，控制 smoke 成本。
    records = list(load_dataset(GPU_DATASET_ID, split='train').select(range(GPU_MAX_SAMPLES)))
    texts = []
    for record in records:
        prompt = record.get('instruction', '')
        if record.get('input'):
            prompt += '\n' + record['input']
        response = record.get('output', '')
        texts.append(f'### Instruction:\n{prompt}\n### Response:\n{response}{tokenizer.eos_token}')
    batch = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=GPU_MAX_LENGTH)
    # padding 位置不参与 loss，保持与 CPU 主线的监督标签约定一致。
    batch['labels'] = batch['input_ids'].masked_fill(batch['attention_mask'] == 0, -100)
    device = torch.device('cuda')
    try:
        native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:  # 兼容没有 including_emulation 参数的旧版 PyTorch。
        native_bf16 = torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if native_bf16 else torch.float16
    # smoke 使用全参数模型，只为验证真实 SFT 链路；不代表 LoRA/QLoRA 显存方案。
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    model.config.use_cache = False
    model.to(device).train()
    batch = {key: value.to(device) for key, value in batch.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    losses = []
    # 每次更新都完整执行 zero_grad → forward → backward → optimizer.step。
    for _ in range(GPU_UPDATES):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=dtype):
            loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    result = {
        'model_profile': GPU_MODEL_PROFILE,
        'model': GPU_MODEL_ID,
        'dataset_profile': GPU_DATASET_PROFILE,
        'dataset': GPU_DATASET_ID,
        'samples': len(records),
        'batch_size': GPU_BATCH_SIZE,
        'seq_len': GPU_MAX_LENGTH,
        'dtype': str(dtype),
        'device': torch.cuda.get_device_name(0),
        'updates': GPU_UPDATES,
        'wall_time_ms_per_update': round(elapsed * 1000 / GPU_UPDATES, 3),
        'peak_memory_mb': round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        'peak_reserved_mb': round(torch.cuda.max_memory_reserved() / 1024**2, 2),
        'losses': losses,
        'evidence_level': 'real_model_real_data_cuda_smoke',
    }
    output_path = project_root / 'benchmarks' / 'results' / '13_real_gpu_sft.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    del optimizer, model, batch
    torch.cuda.empty_cache()

```
