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

训练中的梯度信号既要由目标产生，也要沿模型结构传递。本节作为 2.5 的通用反向传播基础，帮助你建立观察反向训练信号的共同框架，理解局部环节如何影响整体更新，并为后续的显存与训练性能分析准备统一的语言。Attention 中更昂贵的矩阵级反向链，放在相邻的 17 节展开。

学习时先观察训练信号从目标到模型内部的路径，再比较不同局部环节对信号方向、范围和缩放的影响，最后用小规模结果检查自己的判断。这样先建立通用反向语言，再进入 17 节的 Attention 高成本案例。

**关键词：** `activation`, `loss`, `gradients`

---

## 前置阅读

**导语：** 先理解 Autograd、训练循环和显存账本，再看激活与损失的反向路径会更容易把公式和工程现象对应起来。

- [P0: 07. PyTorch Autograd and Backward | PyTorch 自动求导与反向传播](../00_Prerequisites/07_PyTorch_Autograd_and_Backward.md)
- [P0: 13. Simple Neural Network Training | 简单神经网络训练循环](../00_Prerequisites/13_Simple_Neural_Network_Training.md)

---
### Step 1: 激活与损失如何形成反向训练信号

训练反向可以先看成两段相互衔接的局部过程：目标比较产生训练信号，模型内部的计算关系继续传递这个信号。前者决定信号从哪里开始，后者决定信号怎样回到更早的表示。

一条训练样本经过模型得到输出，目标比较会产生返回模型内部的梯度；这个梯度继续沿计算关系传播，并在局部变换处受到前向状态的影响。理解这条路径时，要把“信号的来源”和“信号的传播”分开观察。

阅读主图时，可以先沿前向路径看模型如何得到训练目标，再沿反向路径看信号如何返回。后面的两个机制步骤分别观察局部传播和目标比较，最后在 Step 4 中把它们落实为可验证的实现。

![反向传播：激活门控与损失梯度如何接起来](../public/02_PyTorch_Algorithms/18_activation_loss_backward.svg)

### Step 2: 激活函数如何门控梯度

激活函数的反向传播通常根据前向输入决定上游梯度如何通过。以 ReLU 为例，输入位于正区间时保留梯度，输入位于负区间时将梯度置为零，边界点的处理需要采用一致的约定。

这个表格描述的是梯度门控：它改变已有梯度如何通过，但任务误差仍由损失函数产生。先按输入所在区域判断门控结果，再进入下一步的交叉熵梯度。

| 前向输入 `x` | ReLU 的局部导数 | 传回的输入梯度 |
| --- | --- | --- |
| `x > 0` | `1` | 保留上游梯度 |
| `x < 0` | `0` | 梯度变为 `0` |
| `x == 0` | 本节约定为 `0` | 梯度变为 `0` |

### Step 3: 交叉熵如何把误差送回 logits

损失函数负责把任务目标转成训练信号。以分类任务中的交叉熵为例，输出分数先与目标类别比较，再得到返回输出端的梯度。先区分训练信号如何产生，再观察平均、求和以及忽略位置如何改变梯度的参与范围和缩放。

| 环节 | 计算或配置 | 对反向的影响 |
| --- | --- | --- |
| 前向 | `logits → softmax → 目标标签 → loss` | 得到每个样本的分类误差 |
| 反向 | `softmax(logits) - one_hot(target)` | 梯度直接回到 logits |
| 平均或求和 | 对样本级误差进行聚合 | 平均会按参与样本数缩放梯度，求和保留各项梯度的累积量 |
| 忽略位置或 padding | 指定位置不参与目标比较 | 对应位置不贡献 loss，也不向前传递梯度 |

![交叉熵 backward：loss 如何把信号送回 logits](../public/02_PyTorch_Algorithms/18_loss_gradient_path.svg)

### Step 4: 实现并验证两个局部 backward

现在进入代码实现：补全一个 ReLU backward 和一个 CrossEntropy backward，并分别与 PyTorch 自动求导结果对照。代码任务聚焦两个局部 backward；实现口径和测试重点集中在下面的表格中。

| 实现对象 | 输入 | 输出 | 实现口径与验证重点 |
| --- | --- | --- | --- |
| ReLU backward | `x`、上游梯度 | 输入梯度 | 逐元素门控；`x == 0` 按本节约定置零；检查形状和梯度数值 |
| CrossEntropy backward | logits、target、reduction | loss、logits 梯度 | 支持 `mean/sum`；使用 `log_softmax`；检查 label 对齐、较大 logits 和有限值 |


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


def softmax_ce_loss_and_grad(logits, labels, reduction="mean"):
    """计算 mean / sum reduction 的交叉熵及其 logits 梯度。

    Args:
        logits: 形状为 [batch, classes] 的未归一化分数。
        labels: 形状为 [batch] 的整数类别标签。
        reduction: 只能是 "mean" 或 "sum"，决定如何聚合逐样本 loss。

    Returns:
        (loss, grad)，分别是标量损失和形状与 logits 相同的梯度。

    Note:
        本题支持 mean / sum reduction，暂不处理 ignore_index 和 padding mask。
    """
    if logits.ndim != 2 or labels.ndim != 1 or logits.size(0) != labels.size(0):
        raise ValueError('logits 应为 [batch, classes]，labels 应为 [batch]')
    if logits.device != labels.device:
        raise ValueError('logits 和 labels 必须位于同一 device')
    if reduction not in ("mean", "sum"):
        raise ValueError('reduction 只能是 mean 或 sum')
    if not logits.is_floating_point():
        raise TypeError('logits 必须是浮点张量')
    if labels.dtype not in (torch.int32, torch.int64):
        raise TypeError('labels 必须是整数类别索引')
    if labels.numel() and (labels.min() < 0 or labels.max() >= logits.size(1)):
        raise ValueError('labels 超出类别范围')
    # ==========================================
    # TODO 2: 构造 log_probs、probs 和 one_hot
    # 提示：先用 log_softmax 保证数值稳定，再按 labels 的类别索引构造 one_hot。
    # ==========================================
    # log_probs = ???
    # probs = log_probs.exp()
    # one_hot = ???
    # TODO 3: 计算逐样本 loss、按 reduction 聚合，并得到 logits gradient
    # 提示：先得到 per_sample_loss；mean 时除以 batch size，sum 时保留总和。
    # per_sample_loss = ???
    # loss = ???
    # grad = ???
    return loss, grad

```

### 测试

运行下面的测试单元，确认手写 ReLU / CrossEntropy backward 和 PyTorch 自动求导一致。

```python
def _legacy_test_activation_and_loss_backward():
    """验证两个局部 backward 的数值、边界行为和输入契约。"""
    x = torch.tensor([-2.0, -0.5, 0.0, 1.0, 3.0], requires_grad=True)
    upstream = torch.tensor([0.5, -1.0, 2.0, 0.25, -0.75])
    F.relu(x).backward(upstream)
    manual_relu = relu_backward(upstream, x.detach())
    assert torch.allclose(x.grad, manual_relu), "ReLU backward 不一致"
    assert torch.isfinite(manual_relu).all(), "ReLU 梯度包含 NaN 或 Inf"

    logits = torch.tensor([[1.0, 0.5, -0.2], [0.2, -0.3, 1.2]], requires_grad=True)
    labels = torch.tensor([0, 2])
    loss, manual_grad = softmax_ce_loss_and_grad(logits, labels)
    ce = F.cross_entropy(logits, labels)
    ce.backward()
    assert torch.allclose(loss, ce.detach(), atol=1e-6), "CrossEntropy loss 不一致"
    assert torch.allclose(logits.grad, manual_grad, atol=1e-6), "CrossEntropy backward 不一致"
    assert torch.isfinite(loss) and torch.isfinite(manual_grad).all(), "CrossEntropy 结果包含 NaN 或 Inf"

    # reduction 观察：mean 的 loss 和梯度应是 sum 结果除以 batch size。
    sum_logits = logits.detach().clone().requires_grad_()
    sum_loss = F.cross_entropy(sum_logits, labels, reduction='sum')
    sum_loss.backward()
    assert torch.allclose(sum_loss, loss.detach() * labels.numel(), atol=1e-6)
    assert torch.allclose(sum_logits.grad, manual_grad * labels.numel(), atol=1e-6)
    manual_sum_loss, manual_sum_grad = softmax_ce_loss_and_grad(logits.detach(), labels, reduction='sum')
    assert torch.allclose(manual_sum_loss, sum_loss.detach(), atol=1e-6)
    assert torch.allclose(manual_sum_grad, sum_logits.grad, atol=1e-6)

    # batch=1 且 logits 数值较大时，log_softmax 仍应保持数值稳定。
    edge_logits = torch.tensor([[1000.0, 0.0, -1000.0]], requires_grad=True)
    edge_labels = torch.tensor([0])
    edge_loss, edge_grad = softmax_ce_loss_and_grad(edge_logits, edge_labels)
    edge_ref = F.cross_entropy(edge_logits, edge_labels)
    edge_ref.backward()
    assert torch.isfinite(edge_loss) and torch.isfinite(edge_grad).all(), "大数值输入产生了非有限值"
    assert torch.allclose(edge_loss, edge_ref.detach(), atol=1e-6)
    assert torch.allclose(edge_logits.grad, edge_grad, atol=1e-6)

    # 输入契约：类别标签必须落在 logits 的类别范围内。
    try:
        softmax_ce_loss_and_grad(torch.zeros(1, 3), torch.tensor([3]))
    except ValueError as error:
        assert "类别范围" in str(error)
    else:
        raise AssertionError("越界标签应该触发 ValueError")

    try:
        softmax_ce_loss_and_grad(torch.zeros(1, 3), torch.tensor([0]), reduction='median')
    except ValueError as error:
        assert "reduction" in str(error)
    else:
        raise AssertionError("非法 reduction 应该触发 ValueError")

    # 输入契约：ReLU 的上游梯度和输入形状必须一致。
    try:
        relu_backward(torch.ones(2), torch.ones(3))
    except ValueError as error:
        assert "相同形状" in str(error)
    else:
        raise AssertionError("ReLU 输入形状不一致应该触发 ValueError")

    print(f"ReLU grad: {x.grad.tolist()}")
    print(f"CE loss  : {loss.item():.4f}")
    print("✅ 测试通过！激活与损失的反向直觉和 PyTorch 自动求导一致。")

# 机制测试入口：分别验证 ReLU 门控、交叉熵梯度、reduction、稳定性和输入契约。
def _build_activation_backward_case():
    x = torch.tensor([-2.0, -0.5, 0.0, 1.0, 3.0], requires_grad=True)
    upstream = torch.tensor([0.5, -1.0, 2.0, 0.25, -0.75])
    logits = torch.tensor([[1.0, 0.5, -0.2], [0.2, -0.3, 1.2]], requires_grad=True)
    labels = torch.tensor([0, 2])
    return x, upstream, logits, labels

def _assert_relu_backward(x, upstream):
    F.relu(x).backward(upstream)
    manual = relu_backward(upstream, x.detach())
    assert torch.allclose(x.grad, manual), "ReLU backward 不一致"
    assert torch.isfinite(manual).all(), "ReLU 梯度包含非有限值"

def _assert_cross_entropy_backward(logits, labels):
    loss, manual_grad = softmax_ce_loss_and_grad(logits, labels)
    reference = F.cross_entropy(logits, labels)
    reference.backward()
    assert torch.allclose(loss, reference.detach(), atol=1e-6), "CrossEntropy loss 不一致"
    assert torch.allclose(logits.grad, manual_grad, atol=1e-6), "CrossEntropy backward 不一致"
    return loss, manual_grad

def _assert_reduction_and_stability(logits, labels, mean_loss, mean_grad):
    sum_logits = logits.detach().clone().requires_grad_()
    sum_loss = F.cross_entropy(sum_logits, labels, reduction='sum')
    sum_loss.backward()
    manual_sum_loss, manual_sum_grad = softmax_ce_loss_and_grad(sum_logits.detach(), labels, reduction='sum')
    assert torch.allclose(manual_sum_loss, sum_loss.detach(), atol=1e-6)
    assert torch.allclose(manual_sum_grad, sum_logits.grad, atol=1e-6)
    assert torch.allclose(sum_loss, mean_loss.detach() * labels.numel(), atol=1e-6)
    edge_logits = torch.tensor([[1000.0, 0.0, -1000.0]], requires_grad=True)
    edge_loss, edge_grad = softmax_ce_loss_and_grad(edge_logits, torch.tensor([0]))
    assert torch.isfinite(edge_loss) and torch.isfinite(edge_grad).all(), "大数值输入产生了非有限值"

def _assert_input_contract():
    try:
        softmax_ce_loss_and_grad(torch.zeros(1, 3), torch.tensor([3]))
    except ValueError as error:
        assert "类别范围" in str(error)
    else:
        raise AssertionError("越界标签应该触发 ValueError")
    try:
        relu_backward(torch.ones(2), torch.ones(3))
    except ValueError as error:
        assert "相同形状" in str(error)
    else:
        raise AssertionError("ReLU 输入形状不一致应该触发 ValueError")

def test_activation_and_loss_backward():
    x, upstream, logits, labels = _build_activation_backward_case()
    _assert_relu_backward(x, upstream)
    mean_loss, mean_grad = _assert_cross_entropy_backward(logits, labels)
    _assert_reduction_and_stability(logits, labels, mean_loss, mean_grad)
    _assert_input_contract()
    print("✅ 激活与损失的反向机制测试通过。")

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


def softmax_ce_loss_and_grad(logits, labels, reduction="mean"):
    """计算 mean / sum reduction 的交叉熵及其 logits 梯度。

    Args:
        logits: 形状为 [batch, classes] 的未归一化分数。
        labels: 形状为 [batch] 的整数类别标签。
        reduction: 只能是 "mean" 或 "sum"，决定如何聚合逐样本 loss。

    Returns:
        (loss, grad)，分别是标量损失和形状与 logits 相同的梯度。

    Note:
        本题支持 mean / sum reduction，暂不处理 ignore_index 和 padding mask。
    """
    if logits.ndim != 2 or labels.ndim != 1 or logits.size(0) != labels.size(0):
        raise ValueError('logits 应为 [batch, classes]，labels 应为 [batch]')
    if reduction not in ("mean", "sum"):
        raise ValueError('reduction 只能是 mean 或 sum')
    if labels.dtype not in (torch.int32, torch.int64):
        raise TypeError('labels 必须是整数类别索引')
    if labels.numel() and (labels.min() < 0 or labels.max() >= logits.size(1)):
        raise ValueError('labels 超出类别范围')
    if logits.device != labels.device:
        raise ValueError('logits 和 labels 必须位于同一 device')
    if not logits.is_floating_point():
        raise TypeError('logits 必须是浮点张量')
    # TODO 2: 构造 log_probs、probs 和 one_hot
    # 提示：先用 log_softmax 保证数值稳定，再按 labels 的类别索引构造 one_hot。
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
    # TODO 3: 计算逐样本 loss、按 reduction 聚合，并得到 logits gradient
    # 提示：先得到 per_sample_loss；mean 时除以 batch size，sum 时保留总和。
    per_sample_loss = -(one_hot * log_probs).sum(dim=1)
    loss = per_sample_loss.mean() if reduction == "mean" else per_sample_loss.sum()
    grad = probs - one_hot
    if reduction == "mean":
        grad = grad / logits.size(0)
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

## 相关阅读

本节的梯度公式可以继续对照 PyTorch 的损失函数实现，再进入 checkpoint、FlashAttention 和性能分析。

- [PyTorch CrossEntropyLoss 官方文档](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [PyTorch 激活函数文档](https://pytorch.org/docs/stable/nn.html#non-linear-activations)
- [17. Attention 反向传播与自定义 Autograd](../02_PyTorch_Algorithms/17_Autograd_Basics.md)
- [19. 激活检查点](../02_PyTorch_Algorithms/19_Activation_Checkpointing.md)
- [20. FlashAttention 模拟](../02_PyTorch_Algorithms/20_FlashAttention_Sim.md)
- [P0: 20. 性能分析与显存账本](../00_Prerequisites/20_Profiling_and_Memory_Ledger.md)
- [P1: 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
