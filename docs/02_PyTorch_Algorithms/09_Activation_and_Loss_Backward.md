# 09. Activation and Loss Backward | 激活与损失反向传播

**难度：** Medium | **环境：** [CPU 主线](#runtime-environment) | **标签：** `训练机制`, `激活函数`, `损失函数`, `反向传播` | **目标人群：** 训练机制学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/09_Activation_and_Loss_Backward.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

训练中的梯度信号既要由目标产生，也要沿模型结构传递。损失函数决定误差从哪里进入计算图，激活函数决定已有梯度如何通过局部变换；两者共同影响参数最终接收到的更新信号。

本节先建立损失梯度与激活门控的整体路径，再分别分析 ReLU 和交叉熵的局部反向。完成后，你将能检查梯度方向、有效监督范围和归约尺度，并继续分析 Attention 等矩阵级反向链。

**关键词：** `activation`, `loss`, `gradients`

---
<a id="runtime-environment"></a>

## 运行环境

- **基础练习：** 默认使用 [`BASE-CPU`](../guide.md#base-cpu)，运行 [Step 4 的题目与测试](#mechanism-practice)。
- **框架核对：** 使用 [`TRAIN-CPU`](../topic_discussion/backpropagation_training_mechanism/quickstart.md#train-cpu-quickstart)，完成 Step 4 后运行 [框架一致性核对](#framework-check)。

---
## 前置阅读

**导语：** 先理解 Autograd、训练循环和显存账本，再看激活与损失的反向路径会更容易把公式和工程现象对应起来。

- [P0: 07. PyTorch Autograd and Backward | PyTorch 自动求导与反向传播](../00_Prerequisites/07_PyTorch_Autograd_and_Backward.md)
- [P0: 12. PyTorch Data Interface and Training Lifecycle | 数据接口与训练生命周期](../00_Prerequisites/12_PyTorch_Minimal_Training_Interface.md)

---
## Step 1: 激活与损失如何形成反向训练信号

一条训练样本先沿前向路径得到模型输出和损失，再由损失产生输出端梯度；梯度沿计算依赖返回较早表示，并在激活函数处受前向状态门控。理解这条链时要区分两个责任：损失决定训练信号从哪里产生，局部算子决定已有信号怎样继续传播。主图用前向与反向两条方向相反的路径标出这两个责任。

![反向传播：激活门控与损失梯度如何接起来](../public/02_PyTorch_Algorithms/09_activation_loss_backward.svg)

## Step 2: ReLU 如何根据前向状态门控梯度

激活函数的局部反向接收上游梯度，再根据前向状态形成局部门控。对 $y=\operatorname{ReLU}(x)$，门控可写成 $g(x)=\mathbf{1}[x>0]$，因此 $\frac{\partial L}{\partial x}=\frac{\partial L}{\partial y}\odot g(x)$。正区间保留上游梯度的符号和大小，负区间阻断信号，零点则采用一致约定。

门控只改变已有梯度能否通过，不负责产生任务误差。表中的三区间共同决定一次局部向量—雅可比乘积的结果。

| 前向输入 `x` | ReLU 的局部导数 | 传回的输入梯度 |
| --- | --- | --- |
| `x > 0` | `1` | 保留上游梯度 |
| `x < 0` | `0` | 梯度变为 `0` |
| `x == 0` | 本节约定为 `0` | 梯度变为 `0` |

## Step 3: 交叉熵如何把误差送回 logits

损失函数负责把任务目标转成训练信号。设第 $i$ 个位置的 logits 为 $z_i$，Softmax 概率为 $p_i=\operatorname{softmax}(z_i)$，one-hot 目标为 $y_i$，有效监督位置集合为 $\mathcal V$。使用有效位置平均时，交叉熵及其 logits 梯度为：

$$
L=-\frac{1}{|\mathcal V|}\sum_{i\in\mathcal V}\sum_c y_{ic}\log p_{ic},
\qquad
\frac{\partial L}{\partial z_{ic}}=
\begin{cases}
\dfrac{p_{ic}-y_{ic}}{|\mathcal V|}, & i\in\mathcal V,\\
0, & i\notin\mathcal V.
\end{cases}
$$

其中 $c$ 表示类别，$|\mathcal V|$ 是有效监督位置数。公式同时说明了目标差、忽略位置和 `mean` 分母；若使用 `sum`，只需去掉 $|\mathcal V|$ 的缩放。

| 环节 | 计算或配置 | 对反向的影响 |
| --- | --- | --- |
| 前向目标比较 | `logits → 类别概率 p → 目标分布 y → loss` | 得到每个有效位置的分类误差 |
| 反向目标差 | $p-y$ | 梯度直接回到 logits |
| 平均或求和 | 对样本级误差进行聚合 | `mean` 按有效位置数缩放，`sum` 保留累积量 |
| 无监督或填充位置 | 从 $\mathcal V$ 中排除 | loss 与梯度均为零，也不计入平均分母 |

![交叉熵 backward：loss 如何把信号送回 logits](../public/02_PyTorch_Algorithms/09_loss_gradient_path.svg)
<a id="mechanism-practice"></a>

## Step 4：实现 ReLU 门控与交叉熵 logits 反向

题目区提供上游梯度、激活输入、logits、标签、忽略标记和归约方式。学习者依次完成 ReLU 门控、有效监督位置上的稳定交叉熵，以及与有效位置数量一致的 loss 和梯度归约；测试覆盖零点约定、忽略位置、归约缩放、较大 logits 和非法输入。

| TODO / 函数 | 学习者完成的机制 | 必须满足的约束 | 测试证据 |
| --- | --- | --- | --- |
| TODO 1：构造 ReLU 门控<br>`relu_backward` | 用前向输入的符号决定上游梯度是否通过 | `x == 0` 按本节约定置零，保持 shape、dtype 与 device | `test_relu_backward` 对照自动求导并覆盖零点 |
| TODO 2：建立有效监督分布<br>`softmax_ce_loss_and_grad` | 构造稳定概率、有效位置掩码与目标分布 | 忽略位置不参与目标比较，较大 logits 仍保持有限 | `test_cross_entropy_gradient` 对照原生交叉熵 |
| TODO 3：归约 loss 与梯度<br>`softmax_ce_loss_and_grad` | 按有效位置数量完成 `mean/sum` 归约 | 忽略位置梯度为零，`mean` 分母只能是有效位置数 | `test_reduction_and_stability` 覆盖归约、全忽略和极值输入 |


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
    # ==========================================
    # TODO 1: 构造 ReLU 的反向门控掩码
    # ==========================================
    # mask = ???  # 与 x 同形状；正区间为 1，其余位置为 0
    raise NotImplementedError("TODO 1: 请构造 ReLU 反向门控")
    return grad_out * mask


def softmax_ce_loss_and_grad(logits, labels, reduction="mean", ignore_index=-100):
    """计算支持忽略位置的交叉熵及其 logits 梯度。

    Args:
        logits: 形状为 [batch, classes] 的未归一化分数。
        labels: 形状为 [batch] 的整数类别标签。
        reduction: 只能是 "mean" 或 "sum"，决定如何聚合有效位置的 loss。
        ignore_index: 不参与 loss、梯度和 mean 分母的标签值。

    Returns:
        (loss, grad)，分别是标量损失和形状与 logits 相同的梯度。

    """
    if logits.ndim != 2 or labels.ndim != 1 or logits.size(0) != labels.size(0):
        raise ValueError('logits 应为 [batch, classes]，labels 应为 [batch]')
    if logits.size(0) == 0 or logits.size(1) == 0:
        raise ValueError('batch 和 classes 必须为正数')
    if logits.device != labels.device:
        raise ValueError('logits 和 labels 必须位于同一 device')
    if reduction not in ("mean", "sum"):
        raise ValueError('reduction 只能是 mean 或 sum')
    if not logits.is_floating_point():
        raise TypeError('logits 必须是浮点张量')
    if labels.dtype not in (torch.int32, torch.int64):
        raise TypeError('labels 必须是整数类别索引')
    invalid_labels = (labels != ignore_index) & ((labels < 0) | (labels >= logits.size(1)))
    if invalid_labels.any():
        raise ValueError('labels 超出类别范围')
    # ==========================================
    # TODO 2: 构造有效位置掩码、稳定概率和目标分布
    # ==========================================
    # valid_mask = ???  # [batch]，True 表示该位置参与监督
    # safe_labels = ???  # [batch]，忽略位置替换为可用于索引的占位类别
    # log_probs = ???  # [batch, classes]，数值稳定且元素有限的对数概率
    # one_hot = ???  # [batch, classes]，有效位置为目标分布，忽略位置整行必须为 0
    raise NotImplementedError("TODO 2: 请建立有效监督分布")
    probs = log_probs.exp()
    # ==========================================
    # TODO 3: 按有效位置数量归约 loss 和 logits 梯度
    # ==========================================
    # per_sample_loss = ???  # [batch]，忽略位置为 0
    # loss = ???  # 标量；mean 只除以有效位置数
    # grad = ???  # 与 logits 同形状；忽略位置整行必须为 0
    raise NotImplementedError("TODO 3: 请完成 loss 与梯度归约")
    return loss, grad

```

## 测试

运行下方 CPU 机制测试，分别检查 ReLU 的梯度门控、交叉熵的 logits 梯度、`mean/sum` 缩放、数值稳定性和输入契约。

```python
# 每个测试函数对应 Step 4 的一项机制或输入边界。

def test_relu_backward():
    """验证 TODO 1：ReLU 门控与零点约定匹配自动求导。"""
    x = torch.tensor([-2.0, -0.5, 0.0, 1.0, 3.0], requires_grad=True)
    upstream = torch.tensor([0.5, -1.0, 2.0, 0.25, -0.75])
    F.relu(x).backward(upstream)
    manual = relu_backward(upstream, x.detach())
    assert torch.allclose(x.grad, manual), "ReLU backward 不一致"
    assert manual[2].item() == 0.0, "x == 0 时梯度应按本题约定置零"

def test_cross_entropy_gradient():
    """验证 TODO 2：有效监督分布和忽略位置匹配原生交叉熵。"""
    logits = torch.tensor([[1.0, 0.5, -0.2], [0.3, 0.1, -0.4], [0.2, -0.3, 1.2]], requires_grad=True)
    labels = torch.tensor([0, -100, 2])
    loss, manual_grad = softmax_ce_loss_and_grad(logits.detach(), labels)
    reference = F.cross_entropy(logits, labels, ignore_index=-100)
    reference.backward()
    assert torch.allclose(loss, reference.detach(), atol=1e-6), "CrossEntropy loss 不一致"
    assert torch.allclose(logits.grad, manual_grad, atol=1e-6), "CrossEntropy backward 不一致"
    assert torch.count_nonzero(manual_grad[1]).item() == 0, "忽略位置不应产生梯度"

def test_reduction_and_stability():
    """验证 TODO 3：mean 分母、sum 归约和极值输入保持正确。"""
    logits = torch.tensor([[1.0, 0.5, -0.2], [0.3, 0.1, -0.4], [0.2, -0.3, 1.2]])
    labels = torch.tensor([0, -100, 2])
    mean_loss, mean_grad = softmax_ce_loss_and_grad(logits, labels, reduction='mean')
    sum_loss, sum_grad = softmax_ce_loss_and_grad(logits, labels, reduction='sum')
    valid_count = (labels != -100).sum()
    assert torch.allclose(sum_loss, mean_loss * valid_count, atol=1e-6)
    assert torch.allclose(sum_grad, mean_grad * valid_count, atol=1e-6)
    edge_loss, edge_grad = softmax_ce_loss_and_grad(torch.tensor([[1000.0, 0.0, -1000.0]]), torch.tensor([0]))
    assert torch.isfinite(edge_loss) and torch.isfinite(edge_grad).all(), "极值输入产生了非有限值"

def test_activation_loss_input_contract():
    """验证越界标签、全忽略 batch 和形状错误会被明确拒绝。"""
    invalid_cases = [
        lambda: softmax_ce_loss_and_grad(torch.zeros(1, 3), torch.tensor([3])),
        lambda: softmax_ce_loss_and_grad(torch.zeros(2, 3), torch.tensor([-100, -100])),
        lambda: relu_backward(torch.ones(2), torch.ones(3)),
    ]
    for case in invalid_cases:
        try:
            case()
        except ValueError:
            continue
        raise AssertionError("非法输入应该触发 ValueError")

test_relu_backward()
test_cross_entropy_gradient()
test_reduction_and_stability()
test_activation_loss_input_contract()
print("✅ 激活与损失的局部反向机制测试通过。")

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


def softmax_ce_loss_and_grad(logits, labels, reduction="mean", ignore_index=-100):
    """计算支持忽略位置的交叉熵及其 logits 梯度。

    Args:
        logits: 形状为 [batch, classes] 的未归一化分数。
        labels: 形状为 [batch] 的整数类别标签。
        reduction: 只能是 "mean" 或 "sum"，决定如何聚合有效位置的 loss。
        ignore_index: 不参与 loss、梯度和 mean 分母的标签值。

    Returns:
        (loss, grad)，分别是标量损失和形状与 logits 相同的梯度。

    """
    if logits.ndim != 2 or labels.ndim != 1 or logits.size(0) != labels.size(0):
        raise ValueError('logits 应为 [batch, classes]，labels 应为 [batch]')
    if logits.size(0) == 0 or logits.size(1) == 0:
        raise ValueError('batch 和 classes 必须为正数')
    if reduction not in ("mean", "sum"):
        raise ValueError('reduction 只能是 mean 或 sum')
    if labels.dtype not in (torch.int32, torch.int64):
        raise TypeError('labels 必须是整数类别索引')
    invalid_labels = (labels != ignore_index) & ((labels < 0) | (labels >= logits.size(1)))
    if invalid_labels.any():
        raise ValueError('labels 超出类别范围')
    if logits.device != labels.device:
        raise ValueError('logits 和 labels 必须位于同一 device')
    if not logits.is_floating_point():
        raise TypeError('logits 必须是浮点张量')
    # TODO 2: 构造有效位置掩码、稳定概率和目标分布
    valid_mask = labels != ignore_index
    if not valid_mask.any():
        raise ValueError('至少需要一个有效监督位置')
    safe_labels = labels.masked_fill(~valid_mask, 0)
    log_probs = F.log_softmax(logits, dim=-1)
    one_hot = F.one_hot(safe_labels, num_classes=logits.size(1)).to(logits.dtype) * valid_mask.unsqueeze(1)
    probs = log_probs.exp()
    # TODO 3: 按有效位置数量归约 loss 和 logits 梯度
    per_sample_loss = -(one_hot * log_probs).sum(dim=1)
    valid_count = valid_mask.sum()
    loss_sum = per_sample_loss.sum()
    loss = loss_sum / valid_count if reduction == "mean" else loss_sum
    grad = (probs - one_hot) * valid_mask.unsqueeze(1)
    if reduction == "mean":
        grad = grad / valid_count
    return loss, grad

```

### 解析

**TODO 1：构造 ReLU 门控**

- 门控只由前向输入所在区间决定，`x == 0` 使用本题约定的零梯度。
- `test_relu_backward` 用相同上游梯度与自动求导对照，并单独检查零点。

**TODO 2：建立有效监督分布**

- 忽略标签不能用于类别索引，也不能在目标分布中留下概率质量；先建立有效位置，再构造稳定概率和目标分布。
- `test_cross_entropy_gradient` 将含忽略位置的 loss 和 logits 梯度与原生交叉熵对照。

**TODO 3：归约 loss 与梯度**

- `sum` 保留有效位置贡献之和，`mean` 只按有效位置数缩放；忽略位置的整行梯度必须为零。
- `test_reduction_and_stability` 检查两种归约的比例、极值输入和全忽略失败路径。

<a id="framework-check"></a>

## 框架一致性核对（可选）

下面在同一组输入上比较本节手写结果与 PyTorch 原生算子；该核对不下载模型，也不更新参数。

| 核对对象 | 本节实现 | 成熟框架入口 | 通过条件 |
| --- | --- | --- | --- |
| ReLU 输入梯度 | `relu_backward` | PyTorch Autograd | 包含零点在内的输入梯度逐元素一致 |
| 交叉熵 loss | `softmax_ce_loss_and_grad` 返回的 loss | `torch.nn.functional.cross_entropy` | 忽略位置与有效位置平均口径下数值一致 |
| 交叉熵 logits 梯度 | `softmax_ce_loss_and_grad` 返回的梯度 | PyTorch Autograd | 有效位置梯度一致，忽略位置均为零 |

```python
from importlib.metadata import version


def check_activation_loss_framework_conformance():
    """核对手写 ReLU 与交叉熵反向是否符合 PyTorch 原生算子语义。"""
    torch.manual_seed(42)
    x = torch.tensor([[-1.5, 0.0, 2.0]], dtype=torch.float64, requires_grad=True)
    grad_out = torch.tensor([[0.3, -0.7, 1.2]], dtype=torch.float64)
    torch.relu(x).backward(grad_out)
    local_relu_grad = relu_backward(grad_out, x.detach())

    logits = torch.tensor(
        [[2.0, -1.0, 0.5], [0.2, 1.3, -0.4], [4.0, -3.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    labels = torch.tensor([0, -100, 2])
    framework_loss = F.cross_entropy(logits, labels, ignore_index=-100, reduction='mean')
    framework_loss.backward()
    local_loss, local_grad = softmax_ce_loss_and_grad(
        logits.detach(), labels, reduction='mean', ignore_index=-100
    )
    checks = {
        'relu_gradient_match': bool(torch.allclose(local_relu_grad, x.grad, atol=1e-12, rtol=1e-12)),
        'cross_entropy_loss_match': bool(torch.allclose(local_loss, framework_loss.detach(), atol=1e-12, rtol=1e-12)),
        'cross_entropy_gradient_match': bool(torch.allclose(local_grad, logits.grad, atol=1e-12, rtol=1e-12)),
        'ignored_position_zero': bool(torch.count_nonzero(local_grad[1]).item() == 0),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f'框架一致性核对失败：{failed}')
    return {
        'status': 'passed',
        'check_kind': 'framework_conformance',
        'versions': {'torch': version('torch')},
        'checks': checks,
    }


framework_check = check_activation_loss_framework_conformance()
framework_check
```

## 相关阅读

完成局部梯度后，可以继续分析矩阵级 Attention backward，或观察 checkpoint 如何改变前向状态的保存方式。

- [PyTorch CrossEntropyLoss 官方文档](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [PyTorch 激活函数文档](https://pytorch.org/docs/stable/nn.html#non-linear-activations)
- [10. Attention 反向传播与自定义 Autograd](../02_PyTorch_Algorithms/10_Attention_Backward_and_Custom_Autograd.md)
- [19. 激活检查点](../02_PyTorch_Algorithms/19_Activation_Checkpointing.md)
