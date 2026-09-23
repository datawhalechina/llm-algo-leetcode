# 09. SFT Training Loop | 监督微调训练循环

**难度：** Medium | **环境：** CPU-first | **标签：** `训练微调`, `SFT`, `训练循环` | **目标人群：** 训练机制学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/09_SFT_Training_Loop.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

把模型结构写出来以后，下一步就是让它按监督数据学习回答。Fine-tuning 是继续训练预训练模型以适应下游任务的总称，SFT（监督微调）是其中使用 prompt-response 标注数据的一种训练方式。SFT 最容易出错的地方并不在 optimizer，而在数据和 loss 的对齐：模型输入通常是 `[prompt + response]`，真正应该学习的是 response，而不是让模型去复述 prompt。

本节聚焦 SFT 训练循环中的数据与 loss 对齐：用 prompt masking 将上下文位置设为 `ignore_index`，用 `attention_mask` 区分真实 token 和 padding，再通过 shift logits / labels 对齐下一个 token 预测。完成后，你应该能看懂 `input_ids`、`attention_mask`、`labels` 和 cross entropy 之间的关系，并能把它们接到 `backward → optimizer.step() → scheduler.step()` 的训练更新链上。参数更新的细节、effective batch 和 LoRA / RLHF 训练将在后续内容中沿用这套对齐规则。

**关键词：** `SFT`, `masking`, `attention_mask`, `shift logits`

---
## 前置阅读

**导语：** 先把模型封装、训练循环和优化器基础看清，再读 SFT 的数据构造与 loss 对齐会更顺。

- [P0: 09. PyTorch nn.Module Basics | PyTorch nn.Module 基础](../00_Prerequisites/09_PyTorch_nn_Module_Basics.md)
- [P0: 11. PyTorch Optimizers and Loss | PyTorch 优化器与损失](../00_Prerequisites/11_PyTorch_Optimizers_and_Loss.md)
- [P0: 13. Simple Neural Network Training | 简单神经网络训练循环](../00_Prerequisites/13_Simple_Neural_Network_Training.md)

---
### Step 1：从一条问答样本理解 SFT

SFT 使用“输入—回答”样本训练模型。例如：

- prompt：`请计算 2 + 2。`
- response：`答案是 4。`

`EOS`（end of sequence）是表示序列结束的特殊 token。它让模型知道回答何时结束。

图片说明：训练时将 prompt 与 response 拼接成序列并逐位置预测下一个 token；推理时只输入 prompt，再逐步生成 response。

![SFT 训练序列与预测目标](../public/02_PyTorch_Algorithms/09_sft_example_flow.svg)

本节先从问答样本理解模型如何逐步生成回答，再逐步实现数据张量与 loss 对齐。

### Step 2：构造输入数据三件套

一条 SFT 样本进入模型前，要整理成 `input_ids`、`attention_mask` 和 `labels`。本 Step 先看序列长度与 padding 对齐，labels 的监督范围放到 Step 3。

`padding` 是为了让同一 batch 的序列长度一致而补的占位位置；`mask` 用来标记哪些位置有效、哪些位置需要忽略。

下面的表只检查序列结构；labels 的具体监督范围留到 Step 3。读表时确认真实 token 的 `attention_mask` 为 1、padding 位为 0；`input_ids` 可以包含 padding，但 padding 不应成为训练目标。

| 位置 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 语义 | prompt | prompt | prompt | response | response | response | response | EOS | padding |
| `input_ids` | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 2 | 0 |
| `attention_mask` | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| `labels` | 先不填 | 先不填 | 先不填 | 先不填 | 先不填 | 先不填 | 先不填 | 先不填 | 先不填 |

具体的 prompt masking 与监督 token 由 Step 3 统一确定。
### Step 3：对齐 next-token loss

先记住一个位置关系：模型读到位置 `t` 的内容后，要预测位置 `t+1` 的 token。模型对词表中每个 token 给出的分数叫 `logits`；计算 loss 前，要把最后一个 logit 去掉，并把 labels 从第二个位置开始取，也就是 `logits[..., :-1, :]` 对齐 `labels[..., 1:]`。

交叉熵可以看成“预测错得有多严重”：先把 logits 转成概率，再查看正确目标 token 的概率；概率越低，loss 越大。Step 2 的序列中，prompt 和 padding 的 label 设为 `-100`，response 与可选 EOS 保留目标 token；`CrossEntropyLoss(ignore_index=-100)` 会跳过 `-100` 位置并对其余目标求平均。若传入 `attention_mask`，还要同步屏蔽 shift 后的 padding。

**实现约束：** response 至少要留下一个监督 token；如果加入 EOS，它放在 response 后面；本节构造函数采用右侧 padding。截断后没有监督 token 时，应该调大 `max_len`、过滤样本或报错。

下表把 label 的监督范围与 shift 后的预测位置放在一起，帮助检查 logits、目标 token 和 loss 是否对应。读表时先确认 `labels=-100` 决定监督范围，再确认 shift 决定 logits 与目标 token 的时间位置。

| 预测位置 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| logit 预测的 token | 20 | 30 | 40 | 50 | 60 | 70 | 2 | 0 |
| shift 后的 label | -100 | -100 | 40 | 50 | 60 | 70 | 2 | -100 |
| 是否计入 loss | 否 | 否 | 是 | 是 | 是 | 是 | 是 | 否 |

![SFT 数据与损失对齐](../public/02_PyTorch_Algorithms/09_sft_alignment.svg)


### Step 4：实现并验证 SFT 数据与 Loss

**要求**：请补全下方两个函数的 `TODO`。TODO 1–3 对应 Step 2 的数据阶段：监督 labels、截断后的有效 token 数和 padding 长度；TODO 4–5 对应 Step 3 的 loss 阶段：shift 对齐、有效监督检查和交叉熵。题目区已经给出形状保持不变的切片与 mask 骨架，请把注意力放在监督边界和 loss 对齐上。

这里仍然使用 token id 直接演示；真实工程中的 chat template 和 tokenizer 会在进入本函数前完成。完成代码后运行测试，检查样本结构和 loss 对齐。


```python
import torch
import torch.nn as nn
```


```python
# 题目设计：把 prompt-response 样本转换为可验证的监督训练契约。
# TODO 依次覆盖监督范围、截断保护、padding 屏蔽和 next-token 对齐。
def build_sft_data(
    prompt_ids: list[int],
    response_ids: list[int],
    pad_id: int = 0,
    eos_id: int | None = None,
    max_len: int = 16,
    min_response_tokens: int = 1,
):
    """构造一条右侧 padding 的 SFT 样本。

    参数：prompt_ids / response_ids 为 token id 列表；pad_id 和 eos_id
    分别指定 padding 与可选 EOS；max_len 是输出固定长度；
    min_response_tokens 是截断后必须保留的监督 token 数。

    返回三个长度为 max_len 的 torch.long 张量：input_ids、
    attention_mask 和 labels；labels 中的 -100 不参与交叉熵。
    截断后有效监督 token 不足时抛出 ValueError。
    """
    response_with_eos = response_ids + ([] if eos_id is None else [eos_id])

    # 1. 拼接成完整序列。
    input_ids = prompt_ids + response_with_eos

    # ==========================================
    # Prompt 部分先统一标成 ignore_index，确保只对 Response/EOS 计算损失。
    # TODO 1: 构造 labels
    # 规则：
    # - 长度与 input_ids 相同
    # - prompt 部分的 label 设置为 -100
    # - response/EOS 部分的 label 保持原样
    # ==========================================
    # labels = ???

    # ==========================================
    # TODO 2: 截断（Truncation）与有效监督检查
    # 规则：
    # - input_ids 和 labels 使用同一个 [:max_len] 范围截断
    # - 从截断后的 labels 统计有效监督 token
    # - 截断后至少保留 min_response_tokens 个可监督 token
    # ==========================================
    input_ids = input_ids[:max_len]
    labels = labels[:max_len]
    # valid_supervised = sum(label != -100 for label in labels)
    # if valid_supervised < min_response_tokens:
    #     raise ValueError(...)

    # ==========================================
    # TODO 3: attention mask 与填充 (Padding)
    # 规则：
    # - 真实 token 的 attention_mask 为 1
    # - 计算 pad_len；若大于 0，再分别追加 pad_id、0 和 -100
    # ==========================================
    attention_mask = [1] * len(input_ids)
    # pad_len = max_len - len(input_ids)
    # if pad_len > 0: 追加 padding 到三个序列

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def compute_sft_loss(logits: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor | None = None):
    """计算自回归 SFT 的 token-level cross entropy。

    logits 的 shape 为 [batch_size, seq_len, vocab_size]；labels 和
    attention_mask 的 shape 为 [batch_size, seq_len]。返回标量 loss；
    shift 后没有有效监督 token 时抛出 ValueError。
    """
    # ==========================================
    # TODO 4: 实现 Shift 错位对齐
    # 将 logits 的最后一个 token 切掉
    # 将 labels 的第一个 token 切掉
    # 如果传入 attention_mask，也同步切掉第一个位置
    # ==========================================
    # shift_logits = ???
    # shift_labels = ???
    # if attention_mask is not None:
    #     shift_attention_mask = ???
    #     shift_labels = ???

    # ==========================================
    # TODO 5: 检查有效监督 token，并计算交叉熵
    # 提示：loss_fct 使用 ignore_index=-100；计算前将
    # shift_logits 整理为 [有效位置数, vocab_size]，
    # shift_labels 整理为 [有效位置数]。
    # ==========================================
    # if ???:
    #     raise ValueError(...)
    # loss_fct = ???
    # loss = ???

    return loss

```


```python
# 测试设计：用固定样本验证数据契约、监督范围、shift 对齐和 padding 屏蔽。
# baseline 是中性 logits；candidate 只改变 prompt、response 或 padding 位置，
# 用 loss 是否变化来验证 labels 与 attention_mask 的监督边界。
def _build_sft_test_case():
    """构造可重复的 CPU-first SFT 测试样本。"""
    prompt = [10, 20, 30]
    response = [40, 50, 60, 70]
    pad_id, eos_id, max_len = 0, 2, 9
    tensors = build_sft_data(prompt, response, pad_id, eos_id, max_len)
    return tensors, prompt, pad_id, eos_id, max_len

def _assert_sft_data_contract(tensors):
    """验证 input_ids、attention_mask 和 labels 的结构契约。"""
    input_ids, attention_mask, labels = tensors
    assert input_ids.tolist() == [10, 20, 30, 40, 50, 60, 70, 2, 0], "Input IDs 构造错误！"
    assert attention_mask.tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 0], "attention_mask 构造错误！"
    assert labels.tolist() == [-100, -100, -100, 40, 50, 60, 70, 2, -100], "Labels 构造或 Padding 错误！"

def _assert_truncation_guard(prompt, pad_id, eos_id):
    """验证截断后不能丢失全部有效 response 监督。"""
    try:
        build_sft_data(prompt, [40], pad_id=pad_id, eos_id=eos_id, max_len=3)
    except ValueError:
        return
    raise AssertionError("截断后没有 response token 时应该报错")

def _make_aligned_logits(labels, max_len):
    """构造一个只在有效 response 位置预测正确的 logits。"""
    torch.manual_seed(42)
    logits = torch.randn(1, max_len, 100)
    for position, target in [(2, 40), (3, 50), (4, 60), (5, 70), (6, 2)]:
        logits[0, position, target] = 50.0
    return logits

def _assert_loss_alignment(tensors, max_len):
    """验证 shift 后的 logits、labels 与 attention_mask 对齐。"""
    input_ids, attention_mask, labels = tensors
    logits = _make_aligned_logits(labels, max_len)
    loss = compute_sft_loss(logits, labels.unsqueeze(0), attention_mask.unsqueeze(0))
    assert loss.item() < 0.01, f"Loss 异常偏大，可能包含了 Prompt 或 Padding 的计算！Loss = {loss.item()}"
    return logits, labels.unsqueeze(0), attention_mask.unsqueeze(0)

def _assert_supervision_scope(logits, labels_batch, attention_batch):
    """验证 prompt 不监督、response 受监督、padding 被屏蔽。"""
    neutral_logits = torch.zeros_like(logits)
    neutral_loss = compute_sft_loss(neutral_logits, labels_batch, attention_batch)

    prompt_candidate = neutral_logits.clone()
    prompt_candidate[0, 0, 99] = 50.0
    prompt_loss = compute_sft_loss(prompt_candidate, labels_batch, attention_batch)
    assert torch.allclose(prompt_loss, neutral_loss), "Prompt 位置不应参与 loss"

    response_candidate = neutral_logits.clone()
    response_candidate[0, 2, 40] = 10.0
    response_loss = compute_sft_loss(response_candidate, labels_batch, attention_batch)
    assert response_loss < neutral_loss, "response 目标位置没有参与 loss"

    labels_with_pad_target = labels_batch.clone()
    labels_with_pad_target[0, -1] = 99
    protected_loss = compute_sft_loss(neutral_logits, labels_with_pad_target, attention_batch)
    assert torch.allclose(protected_loss, neutral_loss), "padding 目标没有被 attention_mask 屏蔽"

def test_sft_pipeline():
    """运行 CPU-first 的 SFT 机制测试。"""
    try:
        tensors, prompt, pad_id, eos_id, max_len = _build_sft_test_case()
        _assert_sft_data_contract(tensors)
        _assert_truncation_guard(prompt, pad_id, eos_id)
        logits, labels_batch, attention_batch = _assert_loss_alignment(tensors, max_len)
        _assert_supervision_scope(logits, labels_batch, attention_batch)

        input_ids, attention_mask, labels = tensors
        print(f"Input IDs      : {input_ids.tolist()}")
        print(f"Attention Mask : {attention_mask.tolist()}")
        print(f"Labels         : {labels.tolist()}")
        print("✅ All Tests Passed! SFT 数据与 loss 对齐逻辑实现正确。")

    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
        raise
    except (AttributeError, NameError, TypeError, ValueError) as e:
        print("代码可能未完成，导致变量未定义" if isinstance(e, NameError) else "代码可能未完成，导致了类型错误")
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e
    except Exception as e:
        print(f"❌ 发生异常: {e}")
        raise

test_sft_pipeline()

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
def build_sft_data(
    prompt_ids: list[int],
    response_ids: list[int],
    pad_id: int = 0,
    eos_id: int | None = None,
    max_len: int = 16,
    min_response_tokens: int = 1,
):
    """构造一条右侧 padding 的 SFT 样本。

    参数：prompt_ids / response_ids 为 token id 列表；pad_id 和 eos_id
    指定 padding 与可选 EOS；max_len 是输出固定长度；
    min_response_tokens 是截断后必须保留的监督 token 数。

    返回三个长度为 max_len 的 torch.long 张量；labels 中的 -100 不参与交叉熵。
    截断后有效监督 token 不足 min_response_tokens 时抛出 ValueError。
    """
    response_with_eos = response_ids + ([] if eos_id is None else [eos_id])
    input_ids = prompt_ids + response_with_eos

    # TODO 1: 构造 labels
    labels = [-100] * len(prompt_ids) + response_with_eos

    # TODO 2: 截断与有效监督检查
    input_ids = input_ids[:max_len]
    labels = labels[:max_len]
    valid_supervised = sum(label != -100 for label in labels)
    if valid_supervised < min_response_tokens:
        raise ValueError("截断后没有足够的 response token 参与监督，请调大 max_len 或过滤该样本。")

    # TODO 3: attention mask 与填充
    attention_mask = [1] * len(input_ids)
    pad_len = max_len - len(input_ids)
    if pad_len > 0:
        input_ids = input_ids + [pad_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len
        labels = labels + [-100] * pad_len

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def compute_sft_loss(logits: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor | None = None):
    """按 next-token 对齐规则计算忽略 padding 的交叉熵。

    logits、labels 和 attention_mask 的 shape 分别为 [B, T, V]、
    [B, T] 和可选的 [B, T]；返回标量 loss。
    """
    # 预测位置向左对齐一位，对应 next-token prediction。
    # TODO 4: 实现 Shift 错位对齐
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    if attention_mask is not None:
        shift_attention_mask = attention_mask[..., 1:].contiguous()
        shift_labels = shift_labels.masked_fill(shift_attention_mask == 0, -100)

    # TODO 5: 检查有效监督 token 并计算交叉熵
    if not torch.any(shift_labels != -100):
        raise ValueError("当前 batch 没有任何有效监督 token，请检查 labels、padding 或截断策略。")

    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    loss = loss_fct(shift_logits, shift_labels)

    return loss

```

### 解析

以下内容按题目区 TODO 1–5 展开，围绕 SFT 的数据构造与 loss 对齐说明每一步的原因。

**1. TODO 1: 构造 labels**

- **实现方式**：`labels = [-100] * len(prompt_ids) + response_with_eos`
- **核心思想**：Prompt 部分全部设为 -100（忽略），Response 和可选 EOS 保持原 token id。
- **Loss Masking 原理**：PyTorch 的 `CrossEntropyLoss` 中，`ignore_index=-100` 的位置不会产生梯度，也不会计入损失。
- **为什么要 mask Prompt**：SFT 的目标是让模型学会“回答”，而不是“背诵提问”。如果 Prompt 也参与损失计算，模型会浪费容量去记忆人类的提问方式。

**2. TODO 2: 截断与有效监督检查**

- **截断逻辑**：`input_ids = input_ids[:max_len]`，`labels = labels[:max_len]`。
- **监督检查**：截断后至少要保留 `min_response_tokens` 个非 `-100` 的 label，否则样本没有足够的训练信号。
- **工程细节**：真实微调里如果 prompt 太长、response 被截没，应该调大 `max_len`、缩短 prompt，或直接过滤样本。

**3. TODO 3: attention mask 与填充**

- **attention mask**：真实 token 设为 `1`，padding 设为 `0`。
- **填充逻辑**：
  - `input_ids` 填充 `pad_id`（通常是 tokenizer 的 pad token）
  - `attention_mask` 填充 `0`
  - `labels` 填充 `-100`（确保 Padding 位置不产生梯度）
- **区别**：`attention_mask` 管模型看不看 padding，`labels=-100` 管 loss 学不学这个位置。

**4. TODO 4: Shift 错位对齐**

- **实现方式**：
  ```python
  shift_logits = logits[..., :-1, :].contiguous()
  shift_labels = labels[..., 1:].contiguous()
  ```
- **自回归原理**：模型用前 $t$ 个 token 预测第 $t+1$ 个 token。
- **对齐逻辑**：
  - `logits[0]` 预测的是 `labels[1]`
  - `logits[1]` 预测的是 `labels[2]`
  - 因此需要切掉 `logits` 的最后一个位置，切掉 `labels` 的第一个位置
- **attention mask 二次保护**：如果传入 `attention_mask`，shift 后的 padding label 会再次被设为 `-100`，避免数据构造错误漏进 loss。

**5. TODO 5: 展平并计算交叉熵**

- **实现方式**：
  ```python
  loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
  shift_logits = shift_logits.view(-1, shift_logits.size(-1))
  shift_labels = shift_labels.view(-1)
  loss = loss_fct(shift_logits, shift_labels)
  ```
- **形状要求**：`CrossEntropyLoss` 期望 logits 形状为 `[N, C]`，labels 形状为 `[N]`。
- **有效监督检查**：如果整个 batch 的 labels 都是 `-100`，loss 没有意义，应该显式报错。
- **数据构造**：真实工程中通常在 tokenizer / DataLoader / collator 里批量生成这三件套，而不是逐条手写。

测试区还额外检查了三件事：改变 prompt 位置的分数不会改变 loss，改变 response 目标位置会改变 loss，以及给 padding 错写目标 token 后仍会被 `attention_mask` 屏蔽。它们验证的是 mask 规则真正参与了 loss 计算，而不只是检查张量形状。

## 相关阅读

完成这个最小训练循环后，可以继续看 LoRA 如何改变参数更新路径，再进入端到端微调和性能证据。

- [10. LoRA Tutorial | LoRA 教程](../02_PyTorch_Algorithms/10_LoRA_Tutorial.md)
- [13. End-to-End Fine-Tuning Experiment | 端到端微调实验](../02_PyTorch_Algorithms/13_End_to_End_Fine_Tuning_Experiment.md)
- [P0: 17. PyTorch Profiling Basics | PyTorch 性能剖析基础](../00_Prerequisites/17_PyTorch_Profiling_Basics.md)
- [P0: 18. Memory Profiling and Optimization | 显存剖析与优化](../00_Prerequisites/18_Memory_Profiling_and_Optimization.md)
- [Hugging Face TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [PyTorch CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)