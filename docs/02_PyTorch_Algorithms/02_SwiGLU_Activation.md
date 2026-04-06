# 02 SwiGLU Activation

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/02_SwiGLU_Activation.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 02. 激活函数与门控机制 (SwiGLU Activation)

**难度：** Easy | **标签：** `模型架构`, `激活函数` | **目标人群：** 模型微调与工程部署

在组装 LLaMA-3 的那一节中，我们使用了 `SwiGLU` 作为 MLP 的激活函数。为什么所有主流大模型（LLaMA, Qwen, Mistral, PaLM）都在抛弃 ReLU/GELU 而转向 SwiGLU？
本节我们将深入推导 SwiGLU 的设计原理，特别是**如何调整隐藏层的维度**，以保证参数量与标准 Transformer 严格对齐。这是面试中非常经典的**架构推导题**。

> **相关阅读**:
> 本节使用纯 PyTorch 实现了算法逻辑与数学推导。
> 如果你想学习工业界如何打破该算子的 Memory Bound (访存瓶颈)，请前往 Triton 篇：
>  [`../03_Triton_Kernels/02_Triton_Fused_SwiGLU.ipynb`](../03_Triton_Kernels/02_Triton_Fused_SwiGLU.ipynb)
>


### Step 1: 核心思想与痛点

> **什么是 GLU (Gated Linear Unit)？**
> 传统 MLP 是 $W_{down}(\sigma(W_{up}x))$。
> 门控机制 (GLU) 引入了“两条路”：一条路做激活（作为门控开关），另一条路保持线性，然后两者逐元素相乘（Hadamard Product）。
> 公式：$\text{GLU}(x, W_1, W_2) = (xW_1 \otimes \sigma(xW_2))W_{down}$。
> 这种机制类似于 LSTM 中的遗忘门，极大地增强了模型捕捉复杂模式的能力。

> **什么是 SwiGLU？**
> 就是把 GLU 中的激活函数 $\sigma$ 换成了 **Swish**（即 $x \cdot \text{Sigmoid}(\beta x)$，在 PyTorch 中 $\beta=1$ 时等于 `SiLU`）。


### Step 2: 代码实现框架
经典的 SwiGLU 实现通常需要三个权重矩阵：`W_gate`, `W_up`, 和 `W_down`。输入 `x` 分别经过前两个线性层，其中一个经过 SiLU 激活，然后两者逐元素相乘，最后再经过 `W_down` 投影回原始维度。


###  Step 3: 核心数学机制：参数量对齐

**经典的面试拷问：**
> “在 GPT-2 中，隐藏层维度通常是输入维度 $d$ 的 4 倍（即 $4d$）。但在使用 SwiGLU 的 LLaMA 中，为什么隐藏层维度变成了 $\frac{8}{3}d$ 并向上取整？”

**推导过程：**
1. **标准 MLP 参数量**：
   输入为 $d$，隐藏层为 $h$。
   有两个投影矩阵（升维 $d \to h$，降维 $h \to d$）。
   总参数量 = $2 \cdot (d \times h)$。
   当 $h = 4d$ 时，总参数量 = $2 \cdot 4d^2 = \mathbf{8d^2}$。

2. **SwiGLU MLP 参数量**：
   输入为 $d$，隐藏层为 $h_{swiglu}$。
   因为有**门控机制**，升维阶段需要**两个**投影矩阵（$W_{gate}$ 和 $W_{up}$，均是 $d \to h_{swiglu}$）。
   降维阶段需要**一个**矩阵（$W_{down}$，是 $h_{swiglu} \to d$）。
   总参数量 = $3 \cdot (d \times h_{swiglu})$。

3. **对齐参数量**：
   为了使得 SwiGLU 的计算开销（参数量）与原始模型完全相同：
   $3 \cdot d \cdot h_{swiglu} = 8d^2$
   解得：$h_{swiglu} = \mathbf{\frac{8}{3}d}$
   
*这正是 LLaMA 源码中对中间层维度进行 `int(8 * hidden_size / 3)` 计算的根本原因！*


###  Step 4: 动手实战

**要求**：请补全下方 `calculate_intermediate_size` 和 `SwiGLU` 模块的代码。


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate_intermediate_size(hidden_size: int, multiple_of: int = 256):
    """
    计算 LLaMA 风格的 SwiGLU 隐藏层维度
    
    规则：
    1. 取 hidden_size 的 8/3
    2. 为了硬件对齐（如 Tensor Core），通常要求是 multiple_of 的倍数。
       因此将结果除以 multiple_of，向上取整后再乘以 multiple_of。
    """
    # ==========================================
    # TODO 1: 计算理论隐藏层大小 (8/3 * hidden_size)
    # 提示: 使用 int() 或 //
    # ==========================================
    # intermediate_size = ???
    
    # ==========================================
    # TODO 2: 向 multiple_of 对齐 (向上取整)
    # 提示: ((size + multiple_of - 1) // multiple_of) * multiple_of
    # ==========================================
    # aligned_size = ???
    
    # return aligned_size
    pass

class SwiGLU_MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        # ==========================================
        # TODO 3: 定义 SwiGLU 所需的三个投影矩阵
        # (全部无 bias)
        # ==========================================
        # self.gate_proj = ???
        # self.up_proj = ???
        # self.down_proj = ???
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ==========================================
        # TODO 4: 组装 SwiGLU 前向传播
        # 公式: down_proj( SiLU(gate_proj(x)) * up_proj(x) )
        # ==========================================
        # return ???
        pass

```

```python
# 运行此单元格以测试你的实现
def test_swiglu():
    try:
        # 1. 测试维度推导函数
        hidden_size = 4096 # LLaMA-7B 的 hidden_size
        
        # 理论值 = 4096 * 8 / 3 = 10922.66 -> 10922
        # 对齐到 256 倍数: 10922 / 256 = 42.66 -> 取 43
        # 43 * 256 = 11008
        
        aligned_size = calculate_intermediate_size(hidden_size, multiple_of=256)
        assert aligned_size == 11008, f"维度计算错误，期望 11008，实际得到 {aligned_size}"
        print(f"✅ 隐藏层维度推导正确！4096 -> {aligned_size}")
        
        # 2. 测试参数量对齐
        # 标准 MLP: 2 * (4096 * 16384) = 134,217,728
        # LLaMA SwiGLU: 3 * (4096 * 11008) = 135,266,304 (因为向上取整，略大一点点)
        mlp = SwiGLU_MLP(hidden_size, aligned_size)
        total_params = sum(p.numel() for p in mlp.parameters())
        assert total_params == 135266304, f"参数量异常！{total_params}"
        print(f"✅ SwiGLU 实例参数量验证正确：{total_params} 个参数")
        
        # 3. 测试前向传播连通性
        x = torch.randn(2, 10, hidden_size)
        out = mlp(x)
        assert out.shape == x.shape, "输出形状不等于输入形状！"
        print("\n✅ All Tests Passed! 你已经吃透了当前大模型最主流激活函数的底层数学逻辑！")
        
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
    except TypeError as e:
        print("代码可能未完成，导致了操作错误。")
    except Exception as e:
        print(f"❌ 发生异常: {e}")

test_swiglu()

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

::: details 💡 点击查看官方解析与参考代码

SwiGLU 替代了传统 MLP，用门控机制（Gate 和 Up）进行升维，然后点乘再 Down 降维。通过推导公式 3 * d * h = 8 * d^2，隐藏层维度为 8/3 倍。为了提升硬件的访存效率，我们还需要将该维度向上取整对齐到 multiple_of 的倍数。代码中实现了计算这个对齐大小，以及前向计算 down_proj(SiLU(gate_proj(x)) * up_proj(x))。

```python
def calculate_intermediate_size(hidden_size: int, multiple_of: int = 256):
    intermediate_size = int(hidden_size * 8 / 3)
    aligned_size = ((intermediate_size + multiple_of - 1) // multiple_of) * multiple_of
    return aligned_size

class SwiGLU_MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
