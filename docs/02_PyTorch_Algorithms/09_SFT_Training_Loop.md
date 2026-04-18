# 09. SFT Training Loop | 监督微调训练框架: 数据构造与 Loss Masking (SFT Training Loop)

**难度：** Medium | **标签：** `训练框架`, `SFT`, `PyTorch` | **目标人群：** 模型微调与工程部署

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/09_SFT_Training_Loop.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


在面试大模型算法工程师时，面试官极大概率会问：“在做 SFT（监督微调）时，你是怎么构造 `input_ids` 和 `labels` 的？”、“为什么要 `shift logits`？”
本节我们将实现 SFT 训练中最容易写错的代码：**Prompt Masking**（忽略提问部分的 Loss）和 **交叉熵对齐**。

### Step 1: 核心思想与痛点

> **预训练 (Pre-training) vs 微调 (SFT)**
> * **预训练**：模型预测下一个 Token。给定一本书，每一个字都要算 Loss。
> * **SFT**：给定 `[Prompt] + [Response]`。我们**只关心**模型能不能输出正确的 `Response`。如果把 `Prompt` 也纳入 Loss 计算，模型就会去“背诵”人类的提问方式，而不是去“回答”问题。
> 
> **如何解决？（Loss Masking）**
> 在 PyTorch 的 `CrossEntropyLoss` 中，有一个神仙参数叫 `ignore_index`，默认值是 `-100`。我们只要把 `labels` 中属于 `Prompt` 和 `Padding` 的部分全部替换成 `-100`，这部分就不会产生任何梯度！

### Step 2: Causal Masking 与 Shift Logits

在自回归语言模型中，预测第 $t+1$ 个词完全依赖于前 $t$ 个词。因此，在计算 CrossEntropyLoss 时，模型的预测输出序列（Logits）需要向左偏移（Shift）一位，与真实的标签序列（Labels）对齐。此外，对于 SFT 提示词部分，通常需要设置 `ignore_index = -100` 以避免它们产生梯度传播。

### Step 3: 动手实战

**要求**：请补全下方 `build_sft_data`（构造单条 SFT 数据）和 `compute_sft_loss`（计算损失）的 `TODO` 逻辑。


```python
import torch
import torch.nn as nn
```


```python
def build_sft_data(prompt_ids: list[int], response_ids: list[int], pad_id: int = 0, max_len: int = 16):
    """
    构造单条 SFT 训练数据
    """
    # 1. 拼接成完整的序列
    input_ids = prompt_ids + response_ids
    
    # ==========================================
    # TODO 1: 构造 labels
    # 规则：
    # - 长度与 input_ids 相同
    # - prompt 部分的 label 设置为 -100
    # - response 部分的 label 保持原样 (等于 input_ids 对应位置的值)
    # ==========================================
    # labels = ???
    
    # ==========================================
    # TODO 2: 截断 (Truncation) 与 填充 (Padding)
    # 规则：
    # - 如果超出 max_len，从末尾截断
    # - 如果不足 max_len，在末尾填充 (input_ids 填 pad_id，labels 填 -100)
    # ==========================================
    # 如果超长:
    # input_ids = ???
    # labels = ???
    
    # 如果不足:
    # pad_len = ???
    # input_ids = ???
    # labels = ???
    
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

def compute_sft_loss(logits: torch.Tensor, labels: torch.Tensor):
    """
    计算自回归 SFT Loss
    Args:
        logits: [batch_size, seq_len, vocab_size]
        labels: [batch_size, seq_len]
    """
    # ==========================================
    # TODO 3: 实现 Shift 错位对齐
    # 将 logits 的最后一个 token 切掉
    # 将 labels 的第一个 token 切掉
    # ==========================================
    # shift_logits = ???
    # shift_labels = ???
    
    # ==========================================
    # TODO 4: 将 tensor 展平并计算交叉熵
    # 提示: CrossEntropyLoss 期望的 logits 形状是 [N, C]，labels 形状是 [N]
    # 使用 nn.CrossEntropyLoss(ignore_index=-100)
    # ==========================================
    # loss_fct = ???
    # loss = ???
    
    # return loss
    pass

```


```python
# 运行此单元格以测试你的实现
def test_sft_pipeline():
    try:
        # --- 测试数据构造 ---
        prompt = [10, 20, 30]
        response = [40, 50, 60, 70]
        pad_id = 0
        max_len = 8
        
        input_ids, labels = build_sft_data(prompt, response, pad_id, max_len)
        
        print(f"Input IDs: {input_ids.tolist()}")
        print(f"Labels   : {labels.tolist()}")
        
        # 验证 Prompt 被 mask，Response 保留，Padding 被 mask
        assert labels.tolist() == [-100, -100, -100, 40, 50, 60, 70, -100], "Labels 构造或 Padding 错误！"
        
        # --- 测试 Loss 计算 ---
        batch_size = 1
        vocab_size = 100
        logits = torch.randn(batch_size, max_len, vocab_size)
        
        # 手动让它预测准确
        logits[0, 2, 40] = 50.0
        logits[0, 3, 50] = 50.0
        logits[0, 4, 60] = 50.0
        logits[0, 5, 70] = 50.0
        
        labels_batch = labels.unsqueeze(0)
        loss = compute_sft_loss(logits, labels_batch)
        
        assert loss.item() < 0.01, f"Loss 异常偏大，可能包含了 Prompt 或 Padding 的计算！Loss = {loss.item()}"
        
        print("\n✅ All Tests Passed! SFT 核心逻辑实现正确。")
        
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
    except TypeError as e:
        print("代码未完成导致返回 None 错误。")
    except Exception as e:
        print(f"❌ 发生异常: {e}")

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
def build_sft_data(prompt_ids: list[int], response_ids: list[int], pad_id: int = 0, max_len: int = 16):
    # 1. 拼接成完整的序列
    input_ids = prompt_ids + response_ids
    
    # TODO 1: 构造 labels
    labels = [-100] * len(prompt_ids) + response_ids
    
    # TODO 2: 截断与填充
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]
    else:
        pad_len = max_len - len(input_ids)
        input_ids = input_ids + [pad_id] * pad_len
        labels = labels + [-100] * pad_len
    
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

def compute_sft_loss(logits: torch.Tensor, labels: torch.Tensor):
    # TODO 3: 实现 Shift 错位对齐
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # TODO 4: 展平并计算交叉熵
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    loss = loss_fct(shift_logits, shift_labels)
    
    return loss

```

### 解析

**1. TODO 1: 构造 labels**

- **实现方式**：`labels = [-100] * len(prompt_ids) + response_ids`
- **核心思想**：Prompt 部分全部设为 -100（忽略），Response 部分保持原样。
- **Loss Masking 原理**：PyTorch 的 `CrossEntropyLoss` 中，`ignore_index=-100` 的位置不会产生梯度，也不会计入损失。
- **为什么要 mask Prompt**：SFT 的目标是让模型学会"回答"，而不是"背诵提问"。如果 Prompt 也参与损失计算，模型会浪费容量去记忆人类的提问方式。

**2. TODO 2: 截断与填充**

- **截断逻辑**：`input_ids = input_ids[:max_len]`，`labels = labels[:max_len]`
- **填充逻辑**：
  - `input_ids` 填充 `pad_id`（通常是 0）
  - `labels` 填充 `-100`（确保 Padding 位置不产生梯度）
- **工程细节**：填充必须在 labels 中也设为 -100，否则模型会学习预测 Padding token，浪费计算资源。

**3. TODO 3: Shift 错位对齐**

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
- **contiguous() 的必要性**：切片后的 tensor 可能不连续，`contiguous()` 确保内存连续，避免后续操作报错。

**4. TODO 4: 展平并计算交叉熵**

- **实现方式**：
  ```python
  loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
  shift_logits = shift_logits.view(-1, shift_logits.size(-1))
  shift_labels = shift_labels.view(-1)
  loss = loss_fct(shift_logits, shift_labels)
  ```
- **形状要求**：`CrossEntropyLoss` 期望 logits 形状为 `[N, C]`，labels 形状为 `[N]`。
- **展平操作**：将 `[batch_size, seq_len, vocab_size]` 展平为 `[batch_size * seq_len, vocab_size]`。
- **ignore_index 生效**：所有值为 -100 的位置会被自动忽略，不参与损失计算和梯度回传。

**工程要点**

- **内存效率**：使用 `ignore_index` 比手动 mask 更高效，因为 PyTorch 底层会跳过这些位置的计算。
- **梯度稳定性**：Shift 对齐确保每个位置的预测目标明确，避免了"预测自己"的混乱。
- **数据构造**：在实际工程中，通常在 DataLoader 中批量构造 labels，而不是逐条处理。
