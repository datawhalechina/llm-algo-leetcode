# 02. SwiGLU Activation | SwiGLU 激活

**难度：** Easy | **环境：** CPU-first | **标签：** `模型架构`, `激活函数`, `PyTorch` | **目标人群：** 模型微调与工程部署

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/02_SwiGLU_Activation.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


在组装 LLaMA-3 的那一节中，我们使用了 `SwiGLU` 作为 MLP 的激活函数。为什么所有主流大模型（LLaMA, Qwen, Mistral, PaLM）都在抛弃 ReLU/GELU 而转向 SwiGLU？
本节我们将深入推导 SwiGLU 的设计原理，特别是**如何调整隐藏层的维度**，以保证参数量与标准 Transformer 严格对齐。这是面试中非常经典的**架构推导题**。
如果你对 Transformer 的 MLP 还不熟，可以先记住一个最小概念：`hidden size` 是 token 向量的长度，`gate` 和 `up` 是 SwiGLU 里并行的两条投影分支。

**关键词：** `SwiGLU`, `GLU`, `gating`
## 前置阅读

**导语：** 如果还没把张量运算、激活函数和归一化顺理清楚，先看下面几页会更容易进入门控激活。

- [P0: 05. PyTorch Tensor Fundamentals | PyTorch 张量基础操作](../00_Prerequisites/05_PyTorch_Tensor_Fundamentals.md)
- [P0: 14. Activation Functions | 激活函数](../00_Prerequisites/14_Activation_Functions.md)
- [P0: 15. Normalization Techniques | 归一化技术](../00_Prerequisites/15_Normalization_Techniques.md)

## 相关阅读

**导语：** 本节先用纯 PyTorch 讲清 SwiGLU 的门控原理与维度变化；如果想看同一结构在更高吞吐实现里怎么落地，再看混合精度和算子融合相关页面。

- [P1: 12. TensorCore and Mixed Precision | Tensor Core 与混合精度](../01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.md)
- [P1: 19. Operator Fusion Introduction | 算子融合导论](../01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.md)

### Step 1: 核心思想与痛点

从 Activation 到 Sigmoid 再到 Swish，门控机制的每一步演化都在回答同一个问题：怎么让信息流动得更好。

> **什么是 GLU (Gated Linear Unit)？**
> 
> 传统MLP是 $W_{down}({Activation}(W_{up}x))$。其中 $W_{up}∈R^{d×h}$ 是升维投影矩阵（将输入从 $d$ 维映射到隐藏层 $h$ 维），$W_{down}∈R^{h×d}$ 是降维投影矩阵（将隐藏层映射回 $d$ 维）。
> 
> 门控机制GLU引入了“两条路”：一条路做激活（作为门控开关），另一条路保持线性，然后两者逐元素相乘（Hadamard Product）。
> 公式：$\text{GLU}(x, W_1, W_2, W_3) = (xW_1 \odot Sigmoid(xW_2))W_3$。
> 其中 $W_1,W_2∈R^{d×h}$ 是两个升维投影矩阵（$d$ 为输入维度，$h$为隐藏层维度），$W_3∈R^{h×d}$ 是输出投影矩阵。$xW_1$​ 走线性路，$xW_2$​ 经过激活函数走门控路，两者逐元素相乘后再由 $W_3$​ 投影回 d 维。
> 这种机制类似于 LSTM 中的遗忘门，极大地增强了模型捕捉复杂模式的能力。

> **什么是 SwiGLU？**
> 把 GLU 中的激活函数换成了 **Swish**，即 $ Swish(x) = x \cdot \text{Sigmoid}(\beta x)$，在 PyTorch 中 $\beta=1$ 时等于 `SiLU`）。$ SwiGLU(x) = (xW_1 \odot Swish(xW_2))W_3$。

标准 MLP 的 Activation 对所有神经元一视同仁；GLU 用 Sigmoid 作为"门"，让网络学会按需保留或阻断信息；SwiGLU 用 Swish 替换 Sigmoid，在负值区保留了一定梯度，既保留了门控的选择能力，又避免了梯度在门控单元接近 0 时完全截断——信息流动更顺畅，训练也更稳定。
### Step 2: 核心数学机制：参数量对齐

公式看懂了，但 LLaMA 为什么要把隐藏层设为$\frac{8}{3}d$？这一节不让你背数字，而是带你走一遍参数量推导——门控结构多了个矩阵，维度自然要跟着调，才能和标准 Transformer 对齐。

**典型的面试问题：**
> “在 GPT-2 中，隐藏层维度通常是输入维度 $d$ 的 4 倍（即 $4d$）。但在使用 SwiGLU 的 LLaMA 中，为什么隐藏层维度变成了 $\frac{8}{3}d$ 并向上取整？”

**推导过程：**
1. **标准 MLP 参数量**：
   输入为 $d$，隐藏层为 $h$。
   有两个投影矩阵（升维 $d \to h$，降维 $h \to d$）。
   总参数量 = $2 \cdot (d \times h)$。
   当 $h = 4d$ 时，总参数量 = $2 \cdot 4d^2 = \mathbf{8d^2}$。

2. **SwiGLU MLP 参数量**：
   输入为 $d$，隐藏层为 $h$。
   因为有**门控机制**，升维阶段需要**两个**投影矩阵（$W_{gate}$ 和 $W_{up}$，均是 $d \to h$）。
   降维阶段需要**一个**矩阵（$W_{down}$，是 $h \to d$）。
   总参数量 = $d\times h+d \times h+h \times d = 3 \cdot (d \times h)$。

3. **对齐参数量**：
   为了使得 SwiGLU 的计算开销（参数量）与原始模型完全相同：
   $3 \cdot d \cdot h = 8d^2$
   解得：$h = \mathbf{\frac{8}{3}d}$
   
这正是 LLaMA 源码中对中间层维度进行 `int(8 * hidden_size / 3)`计算，并进一步对齐到特定倍数（如 256）的根本原因。
### Step 3: 工业级实现框架与性能陷阱 (Memory Bound)

公式和维度都对齐了，接下来进入真实的训练框架。SwiGLU 的实现远比写三行 Linear 复杂——融合矩阵、并行计算、内存带宽，每一步都藏着工程取舍。

在理解了 SwiGLU 的基本公式（`down_proj(SiLU(gate_proj(x)) * up_proj(x))`）和 $8/3$ 维度由来后，如何把它写进真实的训练框架中？

**性能陷阱 1：张量并行 (TP) 与内存对齐**

在真实的 LLaMA 源码中，除了按 $8/3$ 计算出隐藏层维度，还需要将其向上取整对齐到一个 `multiple_of`（通常是 256）的倍数。
这不仅是为了让单卡 Tensor Core（通常要求 8-byte 或 32-byte 对齐）跑得更快，更是因为大模型训练会使用**张量并行 (Tensor Parallelism)**。如果隐藏层维度不能被 GPU 数量整除（例如 $TP=8$ 时，对齐到 256 的倍数后分给 8 张卡，每张卡至少能分到 32 维），张量并行切分权重矩阵时会导致分片不完整或运行时崩溃。

**性能陷阱 2：访存瓶颈 (Memory Bound) 与矩阵融合**

在最朴素的代码实现中，开发者会分别定义并执行 `gate_proj(x)` 和 `up_proj(x)`。
由于这两个线性层**共享完全相同的输入张量 $x$**，分开计算会导致巨大的输入 $x$ 被 GPU 从全局显存 (HBM) 中读取两次。

> **工业界解法 (Matrix Fusion)**：
> 在 vLLM、Megatron 等主流框架中，标准的做法是将 $W_{gate}$ 和 $W_{up}$ 这两个形状为 $[d,h]$ 的权重矩阵，在初始化时沿列方向（dim=-1）拼接成一个**巨大的融合矩阵 `gate_up_proj`**，其形状为 $[d, 2 \times h]$。
> 
> 在前向传播时，输入 $x$ 只需要被读取一次，进行一次矩阵乘法。得到的结果再通过 `torch.chunk(2, dim=-1)` 切分为两半，分别作为 gate 和 up 块。这极大地缓解了内存带宽瓶颈。
###  Step 4: 动手实战

这里开始把参数量推导和门控逻辑落实到最小可运行代码里，重点看每个张量是怎么流动的。

**要求**：请补全下方 `calculate_intermediate_size` 和 `SwiGLU` 模块的代码。


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```


```python
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
    # 提示: 注意使用整数除法
    # ==========================================
    # intermediate_size = ???
    
    # ==========================================
    # TODO 2: 向 multiple_of 对齐 (向上取整)
    # 提示: 思考如何利用整除的特性实现向上取整
    # ==========================================
    # aligned_size = ???
    
    return aligned_size

class SwiGLU_MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        # ==========================================
        # TODO 3: 定义工业级 SwiGLU 的投影矩阵
        # ==========================================
        # self.gate_up_proj = ???
        # self.down_proj = ???

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ==========================================
        # TODO 4: 组装工业级 SwiGLU 前向传播
        # ==========================================
        # output = ???
        return output

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
        
        # 检查是否使用了融合矩阵
        assert hasattr(mlp, 'gate_up_proj'), "请实现融合的 gate_up_proj 矩阵！"
        
        total_params = sum(p.numel() for p in mlp.parameters())
        assert total_params == 135266304, f"参数量异常！{total_params}"
        print(f"✅ SwiGLU 实例参数量验证正确：{total_params} 个参数 (包含融合矩阵)")
        
        # 3. 测试前向传播连通性
        x = torch.randn(2, 10, hidden_size)
        out = mlp(x)
        assert out.shape == x.shape, "输出形状不等于输入形状！"
        print("\n✅ All Tests Passed! 你已经掌握了当前大模型最主流激活函数的底层数学逻辑与访存优化！")
        
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
        raise
    except (AttributeError, NameError, TypeError) as e:
        if isinstance(e, AttributeError):
            print("代码未完成，无法找到必要的属性")
        elif isinstance(e, NameError):
            print("代码可能未完成，导致了变量未定义")
        else:
            print("代码可能未完成，导致了操作错误。")
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
        raise
    except Exception as e:
        print(f"❌ 发生异常: {e}")
        raise

test_swiglu()

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
def calculate_intermediate_size(hidden_size: int, multiple_of: int = 256):
    # 先按 8/3 算出理论中间维度，再按 multiple_of 向上对齐。
    # TODO 1 & 2
    intermediate_size = int(hidden_size * 8 / 3)
    aligned_size = ((intermediate_size + multiple_of - 1) // multiple_of) * multiple_of
    return aligned_size

class SwiGLU_MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        # gate_up_proj 把 gate 和 up 两条分支合并成一次线性投影。
        # TODO 3
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先切分 gate / up，再让门控分支通过 SiLU。
        # TODO 4
        gate_up = self.gate_up_proj(x)
        gate, up = torch.chunk(gate_up, 2, dim=-1)
        return self.down_proj(F.silu(gate) * up)
```

### 答案与直觉

**1. TODO 1 & 2 (隐藏层维度计算)**

- **这一题要解决什么：** 先算出 SwiGLU 的理论中间维度，再把它对齐到硬件友好的倍数。
- **为什么是 8/3：** 门控结构多了一条投影分支，所以参数量公式从标准 MLP 的 $2d \times h$ 变成SwiGLU 的 $3d \times h$，对齐后得到 $\frac{8}{3}d$。其中，$d$ 为输入维度，$h$ 为 SwiGLU 隐藏层维度。
- **为什么还要对齐：** 在张量并行和 Tensor Core 场景里，维度必须能被后续切分和硬件访问友好处理。

**2. TODO 3 (定义矩阵)**

- **这一题要解决什么：** 把两条并行投影合并成一个矩阵，减少重复读取输入张量。
- **为什么要融合：** 减少 gate 和 up 分支对同一输入`x`重复读取，降低显存带宽压力。

**3. TODO 4 (前向传播)**

- **这一题要解决什么：** 把融合后的输出切回两条分支，再按 SwiGLU 公式完成前向计算。
- **为什么先切分再激活：** `gate` 负责门控，`up` 负责保留线性信息；先把它们拆开，各自的计算角色更明确。
- **带走的直觉：** SwiGLU 不是单个激活函数，而是“门控 + 线性分支 + 融合投影”的一整套结构。