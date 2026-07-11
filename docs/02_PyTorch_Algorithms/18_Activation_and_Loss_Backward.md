# 18. Activation and Loss Backward | 激活与损失反向

**难度：** Medium | **环境：** CPU-first | **标签：** `Backward`, `Activation`, `Loss` | **目标人群：** 反向传播与数值推导入门者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/18_Activation_and_Loss_Backward.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


这页把激活函数和损失函数的反向推导串成一条线：先看梯度如何穿过中间非线性，再看损失如何把训练信号送回前面层。

**关键词：** `activation`, `loss`, `gradients`

## 前置阅读

**导语：** 先把 Autograd 和训练闭环的基础补齐，再看激活函数和损失函数的反向推导会更顺。

- [07. PyTorch Autograd and Backward | PyTorch 自动求导与反向传播](../00_Prerequisites/07_PyTorch_Autograd_and_Backward.md)
- [13. Simple Neural Network Training | 简单神经网络训练循环](../00_Prerequisites/13_Simple_Neural_Network_Training.md)
- [20. Profiling and Memory Ledger | 性能分析与显存账本](../00_Prerequisites/20_Profiling_and_Memory_Ledger.md)


## 相关阅读

**导语：** 完成反向推导后，建议继续看激活检查点和 FlashAttention 模拟。

- [13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
- [18. Activation Checkpointing and Activation Offload | 激活检查点与激活卸载](../01_Hardware_Math_and_Systems/18_Activation_Checkpointing_and_Activation_Offload.md)
- [20. FlashAttention Sim | FlashAttention 模拟](../02_PyTorch_Algorithms/20_FlashAttention_Sim.md)

### Step 1: 激活函数的反向传播

激活函数的反向传播通常不是大矩阵运算，而是一个**逐元素门控**过程。以 ReLU 为例，前向是 `y = max(0, x)`，反向时只要把上游梯度乘上一个布尔掩码即可：

- 当 `x > 0`，梯度保留；
- 当 `x <= 0`，梯度直接置零。

这也是为什么激活函数会影响梯度流动：它不只是改变了数值，还决定了梯度能否顺利穿过这一层。

### Step 2: 损失函数如何把信号送回前面

损失函数负责把任务目标转成训练信号。对于分类任务里最常见的交叉熵，它的反向通常会简化成 `softmax(prob) - one_hot(target)` 的形式，所以你经常会看到：

- **前向**：先把 logits 转成概率，再计算负对数似然；
- **反向**：梯度直接回到 logits，不需要再手工展开一整条复杂公式。

理解这一点很重要，因为很多训练问题并不出在模型主体，而是出在 loss 的定义、归一化方式或 label 处理上。

### Step 3: 最小代码实现

下面用两个最小函数把上面的直觉跑出来：一个演示 ReLU 的逐元素反向，一个演示交叉熵的梯度如何回到 logits。

这一页的实现顺序就是先做 ReLU backward，再核对交叉熵梯度，最后和 PyTorch 自动求导对比。

```python
import torch
import torch.nn.functional as F

```


```python
def relu_backward(grad_out, x):
    # ==========================================
    # TODO 1: 构造 ReLU 的反向门控掩码
    # ==========================================
    # mask = ???
    return grad_out * mask


def softmax_ce_loss_and_grad(logits, labels):
    # ==========================================
    # TODO 2: 计算 softmax、one_hot、loss 和梯度
    # ==========================================
    # probs = ???
    # one_hot = ???
    # loss = ???
    # grad = ???
    return loss, grad

```


```python
def test_activation_and_loss_backward():
    try:
        x = torch.tensor([-2.0, -0.5, 0.0, 1.0, 3.0], requires_grad=True)
        upstream = torch.ones_like(x)
        relu_out = F.relu(x)
        relu_out.backward(upstream)

        manual_relu = relu_backward(upstream, x.detach())
        assert torch.allclose(x.grad, manual_relu), "ReLU backward 不一致"

        logits = torch.tensor([[1.0, 0.5, -0.2], [0.2, -0.3, 1.2]], requires_grad=True)
        labels = torch.tensor([0, 2])
        loss, manual_grad = softmax_ce_loss_and_grad(logits, labels)

        ce = F.cross_entropy(logits, labels)
        ce.backward()

        assert torch.allclose(logits.grad, manual_grad, atol=1e-6), "CrossEntropy backward 不一致"

        print(f"ReLU grad: {x.grad.tolist()}")
        print(f"CE loss  : {loss.item():.4f}")
        print("✅ 测试通过！激活与损失的反向直觉和 PyTorch 自动求导一致。")
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError) as e:
        if isinstance(e, AttributeError):
            print("代码未完成，无法找到必要的属性")
        elif isinstance(e, NameError):
            print("代码可能未完成，导致了变量未定义")
        elif isinstance(e, TypeError):
            print("代码可能未完成，导致了类型错误")
        elif isinstance(e, ValueError):
            print("代码可能未完成，导致了张量维度错误")
        else:
            print(f"代码可能未完成，导致了断言失败: {e}")
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise

test_activation_and_loss_backward()

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
import torch.nn.functional as F


def relu_backward(grad_out, x):
    # TODO 1: 构造 ReLU 的反向门控掩码
    mask = (x > 0).to(grad_out.dtype)
    return grad_out * mask


def softmax_ce_loss_and_grad(logits, labels):
    # TODO 2: 计算 softmax、one_hot、loss 和梯度
    probs = torch.softmax(logits, dim=-1)
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
    loss = -(one_hot * torch.log(probs + 1e-12)).sum(dim=1).mean()
    grad = (probs - one_hot) / logits.size(0)
    return loss, grad

```

### 解析

**1. TODO 1: 构造 ReLU 的反向门控掩码**

- **实现方式**：`mask = (x > 0).to(grad_out.dtype)`
- **数学含义**：ReLU 的导数在正半轴为 1，在非正半轴为 0。
- **工程意义**：这一步展示了激活函数如何通过逐元素门控影响梯度流动。

**2. TODO 2: 计算 softmax、one_hot、loss 和梯度**

- **实现方式**：先算 `probs = softmax(logits)`，再构造 `one_hot`，然后得到 `loss` 和 `grad`。
- **数学含义**：交叉熵的梯度会化成 `softmax(prob) - one_hot(target)` 这一类形式。
- **工程意义**：理解这条链路，能更快定位训练里和 label / 归一化相关的问题。

**进阶思考**

- 为什么 ReLU 的反向可以只靠一个布尔掩码？
- 为什么交叉熵的梯度可以直接写成 `prob - one_hot`？
- 如果 label 处理错了，训练曲线会发生什么？
