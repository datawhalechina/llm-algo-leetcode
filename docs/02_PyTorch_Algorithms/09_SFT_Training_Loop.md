# 09 SFT Training Loop

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/09_SFT_Training_Loop.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 09. 监督微调训练框架: 数据构造与 Loss Masking (SFT Training Loop)

**难度：** Medium | **标签：** `训练框架`, `SFT`, `PyTorch` | **目标人群：** 模型微调与工程部署

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


### Step 3: 代码实现框架

切片操作是关键：`logits = logits[..., :-1, :].contiguous()`，`labels = labels[..., 1:].contiguous()`。然后将展平后的 Logits 和 Labels 送入 `nn.CrossEntropyLoss(ignore_index=-100)`。

### Step 4: 核心机制：Shift Logits

大模型是**自回归**（Auto-regressive）的。意思是：用前 $t$ 个 Token，预测第 $t+1$ 个 Token。

假设我们的完整序列 `input_ids` 是：`[A, B, C, D]`
* 模型经过前向传播得到的 `logits` 也是 4 个位置的输出：`[Out_A, Out_B, Out_C, Out_D]`
* 其中 `Out_A` 预测的是 `B`。`Out_B` 预测的是 `C`。

**错位对齐（Shift）：**
为了计算 Loss，我们需要把 `logits` 的最后一个位置切掉（因为它预测的是序列外的东西），把 `labels` 的第一个位置切掉（因为它没有前置输入）。
*   `shift_logits = logits[..., :-1, :]`
*   `shift_labels = labels[..., 1:]`

### Step 5: 动手实战

**要求**：请补全下方 `build_sft_data`（构造单条 SFT 数据）和 `compute_sft_loss`（计算损失）的 `TODO` 逻辑。

```python
import torch
import torch.nn as nn

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
        # 伪造 logits，假设模型非常准确，每次都把 next token 的 logit 设得很高
        logits = torch.randn(batch_size, max_len, vocab_size)
        
        # 手动让它预测准确（注意是 shift 的预测）
        # input_ids: [10, 20, 30, 40, 50, 60, 70, 0]
        # label 在 shift 后: [20, 30, 40, 50, 60, 70, 0] (但 20, 30 是 -100)
        # 所以起作用的 label 是: 40, 50, 60, 70
        # 它们对应的预测位置是 index 2, 3, 4, 5 的 logits (即看到 30 预测 40)
        
        logits[0, 2, 40] = 50.0  # 看到 30(Prompt结尾)，预测 40
        logits[0, 3, 50] = 50.0  # 看到 40，预测 50
        logits[0, 4, 60] = 50.0  # 看到 50，预测 60
        logits[0, 5, 70] = 50.0  # 看到 60，预测 70
        
        # 把 labels 包装成 batch 维度
        labels_batch = labels.unsqueeze(0)
        loss = compute_sft_loss(logits, labels_batch)
        
        # 因为我们把对应的正确类的 logit 设到了 50，Loss 应该非常接近于 0
        assert loss.item() < 0.01, f"Loss 异常偏大，可能包含了 Prompt 或 Padding 的计算！Loss = {loss.item()}"
        
        print("\n✅ All Tests Passed! 恭喜你，成功实现 SFT 核心逻辑！")
        
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

::: details 💡 点击查看官方解析与参考代码

监督微调（SFT）的训练循环与普通模型的训练非常类似，重点在于只对特定的输出Token计算损失。通常需要将标签中对应输入部分的掩盖为特殊值（如-100），以此确保模型只在生成正确回复时获得梯度。

```python
def create_sft_labels(input_ids, prompt_length, ignore_index=-100):
    labels = input_ids.clone()
    labels[:, :prompt_length] = ignore_index
    return labels

def compute_sft_loss(logits, labels, ignore_index=-100):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    
    loss = loss_fct(shift_logits, shift_labels)
    return loss
```

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
