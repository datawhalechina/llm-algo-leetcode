# 09 LoRA Tutorial

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/09_LoRA_Tutorial.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 09. 参数高效微调 (PEFT): 深入剖析 LoRA

**难度：** Medium | **标签：** `微调`, `PEFT`, `PyTorch` | **目标人群：** 模型微调与工程部署

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
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int = 8, lora_alpha: int = 16):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        
        # ==========================================
        # TODO 1: 定义原有的 linear 层，并将其参数冻结 (requires_grad = False)
        # 提示: 使用 nn.Linear，不需要 bias
        # ==========================================
        # self.linear = ???
        # ???
        
        # ==========================================
        # TODO 2: 定义 lora_A 和 lora_B 为 Parameter
        # 注意: lora_A 的形状是 [r, in_features]，lora_B 是 [out_features, r]
        # ==========================================
        # self.lora_A = nn.Parameter(???)
        # self.lora_B = nn.Parameter(???)
        
        self.reset_parameters()

    def reset_parameters(self):
        # 原版 Linear 的初始化
        nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
        
        # ==========================================
        # TODO 3: 初始化 lora_A (使用 Kaiming 均匀分布) 和 lora_B (初始化为全 0)
        # ==========================================
        # ???
        # ???
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ==========================================
        # TODO 4: 实现带有旁路注入的前向传播
        # 公式: output = W_0(x) + scaling * (B @ (A @ x^T))^T
        # 提示: 可以利用 F.linear(x, weight) 或者直接用线性代数乘法
        # ==========================================
        # result = ???
        # result += ???
        
        # return result
        pass

    def merge_weights(self):
        # ==========================================
        # TODO 5: 将 B * A 合并回主权重 (注意乘以 scaling factor)
        # 提示: self.linear.weight.data += ???
        # ==========================================
        # ???
        pass

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
        out_merged = layer.linear(x)  # 合并后直接调用底层的 linear
        assert torch.allclose(out_trained, out_merged, atol=1e-5), "权重合并错误: 合并后的输出与分离时的输出不一致！"
        
        print("\n✅ All Tests Passed! 恭喜你，工业级 LoRA 核心算子实现成功！")
        
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")

test_lora()

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

::: details 💡 点击查看官方解析与参考代码

低秩自适应（LoRA）是一种参数高效的微调技术，核心在于通过低秩矩阵分解来更新权重。代码中通过创建并行的降维和升维线性层，并在前向传播时按缩放因子合并结果，大幅降低了可训练参数量。

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, lora_alpha=16, lora_dropout=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r

        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.weight.requires_grad = False
        
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        
        self.dropout = nn.Dropout(p=lora_dropout)
        
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        result = self.linear(x)
        if self.r > 0:
            lora_out = self.lora_B(self.lora_A(self.dropout(x)))
            result += lora_out * self.scaling
        return result
```

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
