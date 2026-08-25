# 18. Activation and Loss Backward | 激活与损失反向

**难度：** Medium | **环境：** CPU-first | **标签：** `显存优化`, `激活值`, `反向传播` | **目标人群：** 显存优化学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/18_Activation_and_Loss_Backward.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

在 Part00 07 节理解计算图和 `backward()` 之后，还需要单独检查两个局部梯度环节：梯度如何穿过激活函数，以及损失函数如何产生 logits 梯度。它们的口径不一致时，可能出现梯度为 0、loss 不匹配或参数更新异常。Attention backward 的专门推导见相邻的 17 节，本节不依赖那一节才能运行。

本节分别实现 ReLU 和交叉熵的局部反向。ReLU 根据输入符号保留或清零上游梯度；交叉熵把分类误差转换为 logits 梯度。完成后，用 PyTorch 自动求导结果检查手写实现。这里的显存关联只到机制层：激活输出可能被反向使用，但本节不估算完整训练峰值，也不比较 checkpoint 或 offload 的收益。

本节的代码任务是手写 ReLU 和 CrossEntropy 的局部 backward，并与 PyTorch reference 对照；重点是逐元素梯度门控、`mean` reduction 和数值稳定性。它只解释局部反向机制，不负责完整模型训练质量、完整训练峰值显存或 checkpoint / offload 收益。

**关键词：** `activation`, `loss`, `gradients`

---

## 前置阅读

**导语：** 先理解 Autograd、训练循环和显存账本，再看激活与损失的反向路径会更容易把公式和工程现象对应起来。

- [P0: 07. PyTorch Autograd and Backward | PyTorch 自动求导与反向传播](../00_Prerequisites/07_PyTorch_Autograd_and_Backward.md)
- [P0: 13. Simple Neural Network Training | 简单神经网络训练循环](../00_Prerequisites/13_Simple_Neural_Network_Training.md)
- [P0: 20. Profiling and Memory Ledger | 性能分析与显存账本](../00_Prerequisites/20_Profiling_and_Memory_Ledger.md)


## 相关阅读

**导语：** 理解梯度穿过激活和损失之后，可以继续看训练时如何用检查点省显存，以及 Attention 反向相关的性能优化。

- [P1: 13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
- [19. Activation Checkpointing and Activation Offload | 激活检查点与激活卸载](../02_PyTorch_Algorithms/19_Activation_Checkpointing_and_Activation_Offload.md)
- [20. FlashAttention Sim | FlashAttention 模拟](../02_PyTorch_Algorithms/20_FlashAttention_Sim.md)

---
### Step 1: 激活函数的反向传播

激活函数的反向传播通常不是大矩阵运算，而是一个**逐元素门控**过程。以 ReLU 为例，前向是 `y = max(0, x)`，反向时只要把上游梯度乘上一个布尔掩码即可：

- 当 `x > 0`，梯度保留；
- 当 `x <= 0`，梯度直接置零。

这也是为什么激活函数会影响梯度流动：它不只是改变了数值，还决定了梯度能否顺利穿过这一层。

### Step 2: 损失函数如何把信号送回前面

损失函数负责把任务目标转成训练信号。对于分类任务里最常见的交叉熵，在 `reduction="mean"` 且没有 `ignore_index` 时，它对 logits 的梯度可以写成 `(softmax(logits) - one_hot(target)) / batch_size`，所以你经常会看到：

- **前向**：先把 logits 转成概率，再计算负对数似然；
- **反向**：梯度直接回到 logits，不需要再手工展开一整条复杂公式。

理解这一点很重要，因为很多训练问题并不出在模型主体，而是出在 loss 的定义、归一化方式或 label 处理上。

### Step 3: 最小代码实现与口径校验

下面用两个最小函数把上面的直觉跑出来：一个演示 ReLU 的逐元素反向，一个演示交叉熵的梯度如何回到 logits。

这一页的实现顺序就是先做 ReLU backward，再核对交叉熵梯度，最后和 PyTorch 自动求导对比。

#### 训练里的口径

- `reduction="mean"` 会影响梯度缩放，做手写实现时要和参考实现保持一致。
- `ignore_index`、padding 和 label 的处理会直接决定哪些 token 会参与反向传播。
- 这一节的代码重点不是做复杂网络，而是把 ReLU 门控和交叉熵的梯度口径写对。
- 测试会覆盖 `x == 0`、单样本 batch 和较大 logits；它们只检查公式与数值稳定性，不代表完整训练任务的质量。

### 提示

- ReLU backward 本质上就是逐元素门控：正半轴保留梯度，非正半轴置零。
- 交叉熵梯度通常可化成 `softmax(logits) - one_hot` 的形式，但要注意 batch 维上的平均口径；实现 loss 时优先使用 `log_softmax` 保持数值稳定。
- 如果后面要接 `09 / 13 / 30`，这节的目标是把“loss 怎么把信号送回前面”说清，而不是扩成完整训练循环。


```python
import torch
import torch.nn.functional as F

```


```python
def relu_backward(grad_out, x):
    """手写 ReLU 的反向传播。

    Args:
        grad_out: 上游梯度，与 x 形状相同。
        x: ReLU 前的输入张量。x == 0 时本题按梯度 0 处理。

    Returns:
        传回输入 x 的梯度。
    """
    if grad_out.shape != x.shape:
        raise ValueError('grad_out 和 x 必须具有相同形状')
    if grad_out.device != x.device:
        raise ValueError('grad_out 和 x 必须位于同一 device')
    # TODO 1: 构造 ReLU 的反向门控掩码
    # ==========================================
    # mask = ???
    return grad_out * mask


def softmax_ce_loss_and_grad(logits, labels):
    """计算 reduction='mean' 的交叉熵及其 logits 梯度。

    Args:
        logits: 形状为 [batch, classes] 的未归一化分数。
        labels: 形状为 [batch] 的整数类别标签。

    Returns:
        (loss, grad)，分别是标量损失和形状与 logits 相同的梯度。

    Note:
        本题使用 mean reduction，暂不处理 ignore_index 和 padding mask。
    """
    if logits.ndim != 2 or labels.ndim != 1 or logits.size(0) != labels.size(0):
        raise ValueError('logits 应为 [batch, classes]，labels 应为 [batch]')
    if logits.device != labels.device:
        raise ValueError('logits 和 labels 必须位于同一 device')
    if not logits.is_floating_point():
        raise TypeError('logits 必须是浮点张量')
    if labels.dtype not in (torch.int32, torch.int64):
        raise TypeError('labels 必须是整数类别索引')
    if labels.numel() and (labels.min() < 0 or labels.max() >= logits.size(1)):
        raise ValueError('labels 超出类别范围')
    # ==========================================
    # TODO 2: 依次计算 log_probs、one_hot、mean loss 和 logits gradient
    # 提示：先用 log_softmax 保证数值稳定，再按 batch 维构造 one_hot 和平均损失。
    # ==========================================
    # log_probs = ???
    # probs = log_probs.exp()
    # one_hot = ???
    # loss = ???
    # grad = ???
    return loss, grad

```

### 测试

运行下面的测试单元，确认手写 ReLU / CrossEntropy backward 和 PyTorch 自动求导一致。

```python
def test_activation_and_loss_backward():
    try:
        x = torch.tensor([-2.0, -0.5, 0.0, 1.0, 3.0], requires_grad=True)
        upstream = torch.tensor([0.5, -1.0, 2.0, 0.25, -0.75])
        relu_out = F.relu(x)
        relu_out.backward(upstream)

        manual_relu = relu_backward(upstream, x.detach())
        assert torch.allclose(x.grad, manual_relu), "ReLU backward 不一致"

        logits = torch.tensor([[1.0, 0.5, -0.2], [0.2, -0.3, 1.2]], requires_grad=True)
        labels = torch.tensor([0, 2])
        loss, manual_grad = softmax_ce_loss_and_grad(logits, labels)

        ce = F.cross_entropy(logits, labels)
        ce.backward()

        assert torch.allclose(loss, ce.detach(), atol=1e-6), "CrossEntropy loss 不一致"
        assert torch.allclose(logits.grad, manual_grad, atol=1e-6), "CrossEntropy backward 不一致"

        # 边界检查：batch=1 且 logits 数值较大时，手写实现仍应稳定。
        edge_logits = torch.tensor([[1000.0, 0.0, -1000.0]], requires_grad=True)
        edge_labels = torch.tensor([0])
        edge_loss, edge_grad = softmax_ce_loss_and_grad(edge_logits, edge_labels)
        edge_ref = F.cross_entropy(edge_logits, edge_labels)
        edge_ref.backward()
        assert torch.isfinite(edge_loss) and torch.isfinite(edge_grad).all(), "边界输入产生了非有限值"
        assert torch.allclose(edge_loss, edge_ref.detach(), atol=1e-6)
        assert torch.allclose(edge_logits.grad, edge_grad, atol=1e-6)

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
    """手写 ReLU 的反向传播。

    Args:
        grad_out: 上游梯度，与 x 形状相同。
        x: ReLU 前的输入张量。x == 0 时本题按梯度 0 处理。

    Returns:
        传回输入 x 的梯度。
    """
    if grad_out.shape != x.shape:
        raise ValueError('grad_out 和 x 必须具有相同形状')
    if grad_out.device != x.device:
        raise ValueError('grad_out 和 x 必须位于同一 device')
    # TODO 1: 构造 ReLU 的反向门控掩码
    mask = (x > 0).to(grad_out.dtype)
    return grad_out * mask


def softmax_ce_loss_and_grad(logits, labels):
    """计算 reduction='mean' 的交叉熵及其 logits 梯度。

    Args:
        logits: 形状为 [batch, classes] 的未归一化分数。
        labels: 形状为 [batch] 的整数类别标签。

    Returns:
        (loss, grad)，分别是标量损失和形状与 logits 相同的梯度。

    Note:
        本题使用 mean reduction，暂不处理 ignore_index 和 padding mask。
    """
    if logits.ndim != 2 or labels.ndim != 1 or logits.size(0) != labels.size(0):
        raise ValueError('logits 应为 [batch, classes]，labels 应为 [batch]')
    if labels.dtype not in (torch.int32, torch.int64):
        raise TypeError('labels 必须是整数类别索引')
    if labels.numel() and (labels.min() < 0 or labels.max() >= logits.size(1)):
        raise ValueError('labels 超出类别范围')
    if logits.device != labels.device:
        raise ValueError('logits 和 labels 必须位于同一 device')
    if not logits.is_floating_point():
        raise TypeError('logits 必须是浮点张量')
    # TODO 2: 依次计算 log_probs、one_hot、mean loss 和 logits gradient
    # 提示：先用 log_softmax 保证数值稳定，再按 batch 维构造 one_hot 和平均损失。
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
    loss = -(one_hot * log_probs).sum(dim=1).mean()
    grad = (probs - one_hot) / logits.size(0)
    return loss, grad

```

### 解析

**1. TODO 1: 构造 ReLU 的反向门控掩码**

- **实现方式**：`mask = (x > 0).to(grad_out.dtype)`
- **数学含义**：ReLU 的导数在正半轴为 1，在非正半轴为 0。
- **工程意义**：这一步展示了激活函数如何通过逐元素门控影响梯度流动。

**2. TODO 2: 计算 softmax、one_hot、loss 和梯度**

- **实现方式**：先算 `log_probs = log_softmax(logits)`，再构造 `one_hot`，然后得到稳定的 `loss` 和 `grad`。
- **数学含义**：交叉熵的 logits 梯度会化成 `(softmax(logits) - one_hot(target)) / batch_size`。
- **工程意义**：理解这条链路，能更快定位训练里和 label / 归一化相关的问题。

**进阶思考**

- 为什么 ReLU 的反向可以只靠一个布尔掩码？
- 为什么交叉熵的梯度可以直接写成 `prob - one_hot`？
- 如果 label 处理错了，训练曲线会发生什么？
