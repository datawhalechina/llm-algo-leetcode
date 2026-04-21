# 10. LoRA Tutorial | 参数高效微调: 深入剖析 LoRA (PEFT)

**难度：** Medium | **标签：** `微调`, `PEFT`, `PyTorch` | **目标人群：** 模型微调与工程部署

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/10_LoRA_Tutorial.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本节我们将解析大语言模型领域最具影响力的微调算法：**LoRA (Low-Rank Adaptation)**。我们将实现一个 `LoRALinear` 层，替换标准的 `nn.Linear`，体验矩阵秩分解是如何极大地节省显存开销的。

### Step 1: 核心思想与痛点

> **为什么需要 LoRA？**
> 全参微调 (Full Fine-tuning) 一个 7B 模型需要大规模的显存来保存优化器状态（Adam 需要保存参数的动量和方差，占用额外 8 倍参数量的显存）。绝大多数中小企业和个人开发者无法承担。
> **LoRA 的本质：**
> 冻结原始的预训练模型权重，并在每个 Dense 层旁边注入可训练的“旁路”降秩矩阵（A 和 B）。微调时只更新这非常少量的参数。最终推理时，可以将旁路权重无损“合并（Merge）”回主权重中。

### Step 2: LoRA 代码框架
在 PyTorch 实现中，除了保留原始冻结的线性层权重外，我们需要并排初始化两个很小的可训练矩阵 A 和 B。A 通常用 Kaiming 均匀分布或高斯分布初始化，而 B 严格初始化为零，以保证训练开始时 $W = W_0 + B A \approx W_0$。

###  Step 3: 核心公式与张量维度

**前向传播公式：**
给定预训练权重 $W_0 \in \mathbb{R}^{d \times k}$，输入 $x$，LoRA 修改后的输出为：
$$ h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x $$

*   $A \in \mathbb{R}^{r \times k}$：降维矩阵，通常使用随机高斯分布初始化（Kaiming Uniform）。
*   $B \in \mathbb{R}^{d \times r}$：升维矩阵，**必须初始化为全 0**，以保证初始状态下 $\Delta W = 0$，也就是微调前的输出和预训练模型完全一致。
*   $r$ (rank)：矩阵的秩，通常设置极小，如 8 或 16。
*   $\alpha$：缩放因子（Scaling Factor），用来控制 $\Delta W$ 的影响程度。

**推理时合并权重 (Merge Weights)：**
$$ W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A $$
这样在部署时，计算图里没有 A 和 B，完全没有额外的推理耗时（No Inference Latency）。

###  Step 4: 动手实战

**要求**：请补全下方 `LoRALinear` 的初始化、前向传播和合并权重的 `TODO` 逻辑。


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
```


```python
class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int = 8, lora_alpha: int = 16):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        
        # ==========================================
        # TODO 1: 初始化主权重和 LoRA 矩阵
        # ==========================================
        # self.linear = ???
        # self.linear.weight.requires_grad = ???
        # self.lora_A = ???
        # self.lora_B = ???
        self.linear = nn.Linear(in_features, out_features, bias=False)   # 占位初始化      
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))  # 占位初始化                                                                                                                 
        self.lora_B = nn.Parameter(torch.zeros(out_features, r)) # 占位初始化    

        self.reset_parameters()

    def reset_parameters(self):
        # ==========================================
        # TODO 2: 初始化权重
        # ==========================================
        # nn.init.kaiming_uniform_(???)
        # nn.init.kaiming_uniform_(???)
        # nn.init.zeros_(???)
        
        # 占位初始化
        nn.init.ones_(self.linear.weight)  # 占位初始化
        nn.init.ones_(self.lora_A) # 占位初始化
        nn.init.ones_(self.lora_B)  # 占位初始化

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ==========================================
        # TODO 3: 实现前向传播
        # 1. 计算主权重的输出
        # 2. 计算 LoRA 分支的输出（先降维再升维，最后乘以缩放因子）
        # 3. 将两者相加
        # 提示: 注意矩阵转置和乘法顺序
        # ==========================================
        # result = ???
        # lora_out = ???

        return torch.zeros(x.shape[0], x.shape[1], self.linear.out_features, device=x.device) # 占位初始化
        

    def merge_weights(self):
        # ==========================================
        # TODO 4: 合并权重（零延迟推理）
        # 提示: 将 LoRA 的低秩更新合并到主权重中
        # ==========================================
        # self.linear.weight.data += ???
        

```


```python
# 运行此单元格以测试你的实现
def test_lora():
    try:
        in_dim, out_dim = 128, 256
        batch_size, seq_len = 32, 10
        layer = LoRALinear(in_dim, out_dim, r=8, lora_alpha=16)
        
        x = torch.randn(batch_size, seq_len, in_dim)
        
        # 1. 验证初始化导致 B 全零，所以初始输出等于冻结权重的输出
        with torch.no_grad():
            out_lora = layer(x)
            out_base = layer.linear(x)
            assert torch.allclose(out_lora, out_base), "初始化错误: lora_B 未被初始化为 0"
        
        # 2. 模拟训练一步，改变 B 的值
        layer.lora_B.data.normal_(0, 0.02)
        
        out_trained = layer(x)
        assert not torch.allclose(out_trained, out_base), "前向传播错误: 旁路未能注入梯度值"
        
        # 3. 验证合并权重的正确性
        layer.merge_weights()
        out_merged = layer.linear(x)
        assert torch.allclose(out_trained, out_merged, atol=1e-5), "权重合并错误: 合并后的输出与分离时的输出不一致！"
        
        print("\n✅ All Tests Passed! LoRA 核心算子实现正确。")
        
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        raise e

test_lora()

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
class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int = 8, lora_alpha: int = 16):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        
        # TODO 1: 初始化主权重和 LoRA 矩阵
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.weight.requires_grad = False
        
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        
        self.reset_parameters()

    def reset_parameters(self):
        # TODO 2: 初始化权重
        nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 3: 实现前向传播
        result = self.linear(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        result += lora_out
        return result

    def merge_weights(self):
        # TODO 4: 合并权重（零延迟推理）
        self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling

```

### 解析

**1. TODO 1 & 2: 初始化主权重和 LoRA 矩阵**

- **主权重冻结**：`self.linear.weight.requires_grad = False` 是 LoRA 的核心，确保预训练权重不参与梯度计算，只更新 A 和 B。
- **LoRA 矩阵形状**：
  - `lora_A`: `[r, in_features]` - 降维矩阵
  - `lora_B`: `[out_features, r]` - 升维矩阵
- **初始化规则**：
  - `lora_A`: 使用 Kaiming 初始化，提供随机性
  - `lora_B`: **必须初始化为全 0**，确保训练开始时 $\Delta W = BA = 0$，即微调模型的初始输出与预训练模型完全一致
- **参数量对比**：原始权重 `[out_features, in_features]`，LoRA 参数 `r * (in_features + out_features)`。当 `r << min(in_features, out_features)` 时，参数量大幅减少。

**2. TODO 3: 前向传播与缩放**

- **实现方式**：
  ```python
  result = self.linear(x)
  lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
  result += lora_out
  ```
- **数学公式**：$h = W_0 x + \frac{\alpha}{r} B A x$
- **缩放因子**：`scaling = lora_alpha / r`，通常 `lora_alpha = 16`，`r = 8`，则 `scaling = 2`。
- **缩放的意义**：在改变秩 $r$ 时，不需要重新调整学习率。较小的 $r$ 会自动获得较大的缩放，保持更新幅度的稳定性。
- **计算顺序**：先 `x @ A^T` 降维到 `[..., r]`，再 `@ B^T` 升维到 `[..., out_features]`，最后乘以 `scaling`。

**3. TODO 4: 合并权重（零延迟推理）**

- **实现方式**：`self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling`
- **核心原理**：由于 $h = Wx + BAx = (W + BA)x$，可以直接将 $BA$ 加到 $W$ 中。
- **零延迟推理**：合并后，模型结构与标准 Linear 层完全相同，没有额外的矩阵乘法，推理速度与原始模型一致。
- **部署优势**：合并后可以直接丢弃 A 和 B 矩阵，节省显存和计算。这是 LoRA 相比 Adapter 等方法的重要优势。
- **可逆性**：如果需要，可以通过 `W - BA` 恢复原始权重，实现"即插即拔"的效果。

**工程要点**

- **显存节省**：7B 模型全参微调需要约 112GB 显存（参数 + 梯度 + 优化器状态），LoRA (r=8) 只需约 14GB。
- **多任务切换**：可以为不同任务训练不同的 A/B 矩阵，推理时动态加载，实现"一个基座模型 + 多个 LoRA 适配器"。
- **秩的选择**：`r=8` 通常足够，`r=16` 可能带来边际提升，`r=32` 以上收益递减。
