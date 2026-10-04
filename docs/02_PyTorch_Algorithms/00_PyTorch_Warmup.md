# 00. PyTorch Warmup | PyTorch 热身

**难度：** Easy | **环境：** CPU-first | **标签：** `基础实现`, `PyTorch`, `入门热身` | **目标人群：** 基础实现学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/00_PyTorch_Warmup.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

Part 02 的 Decoder 组件都从同一份输入状态开始：token id 先查表得到 hidden state，随后再进入 Norm、MLP、位置编码和 Attention。本节不把这些组件逐一展开，而是先建立它们共同依赖的三类契约：张量形状不能改变语义，token id 必须能安全查表，loss 的梯度必须能回到参数。

三个短练习分别验证序列重排、Embedding lookup 和最小反向传播。读完后，能把离散输入、连续表示和训练梯度看成同一条 Decoder 输入链的不同阶段。

**关键词：** `reshape`, `Embedding`, `backpropagation`

---
## 前置阅读

**导语：** 下面会直接使用张量重排、Embedding 和 Autograd；如果这些操作还不熟悉，可先用对应页面补齐一个最小例子。

- [P0: 05. PyTorch Tensor Fundamentals | PyTorch 张量基础操作](../00_Prerequisites/05_PyTorch_Tensor_Fundamentals.md)
- [P0: 07. PyTorch Autograd and Backward | PyTorch 自动求导与反向传播](../00_Prerequisites/07_PyTorch_Autograd_and_Backward.md)
- [P0: 09. PyTorch nn.Module Basics | PyTorch nn.Module 基础](../00_Prerequisites/09_PyTorch_nn_Module_Basics.md)

---
### Step 1：把空间特征整理成序列

Transformer 接收的是序列。图像或其他二维特征进入模型前，通常需要先把空间位置排成 token 序列，同时保留每个位置上的通道表示。`permute + reshape` 显式展示维度顺序；`einops.rearrange` 用名称表达同一变换。练习会要求两种写法得到相同结果。

| 输入 | 目标 | 需要保持的语义 |
| --- | --- | --- |
| `[B, C, H, W]` 图像特征 | `[B, H×W, C]` 序列特征 | 每个空间位置仍对应原来的 `C` 个通道值 |
| 多头表示 `[B, H, S, D]` | `[B, S, H×D]` | token 顺序不变，只合并 head 与 head_dim |

![张量形状从基础操作连接到 Attention](../public/02_PyTorch_Algorithms/00_tensor_shape_flow.svg)
### Step 2：从 tokenizer 输出到连续表示

Tokenizer 将文本切成词表可识别的 token，并输出整数 `input_ids`。模型首先要确认这些 id 的形状、dtype 和取值范围，再由 `nn.Embedding` 查一张可训练的词表矩阵，得到每个 token 对应的连续向量；多个向量按原顺序组成模型输入序列。查表和 `weight[input_ids]` 是同一件事；后续 RoPE 会在这些序列表示上加入位置信息。

| 对象 | 形状 | 含义 |
| --- | --- | --- |
| `input_ids` | `[B, S]`、`torch.long` | tokenizer 输出的词表编号，必须落在 `[0,V)` |
| embedding weight | `[V, D]` | `V` 个 token 的可训练向量表 |
| embedding output | `[B, S, D]` | 与输入位置逐一对应的连续表示 |

![Embedding 从 token id 连接到上下文表示](../public/02_PyTorch_Algorithms/00_embedding_flow.svg)
### Step 3：让梯度沿计算图回到参数

训练不仅需要前向得到输出，还需要把 loss 对输出的变化量传回输入和参数。以 `Linear → ReLU` 为例：前向产生线性结果与激活输出；反向则先由 ReLU 的正负状态筛掉梯度，再分别计算输入、权重和偏置的梯度。这个最小例子也解释了后续训练中为什么要保存激活状态，以及 checkpoint 会通过重算来交换显存。

| 阶段 | 关键状态 | 它在后续计算中的作用 |
| --- | --- | --- |
| Linear 前向 | `x`、`weight` | 决定权重梯度与输入梯度的矩阵方向 |
| ReLU 前向 | `z > 0` 的 mask | 决定哪些位置允许梯度继续传递 |
| 反向传播 | `grad_output` | 依次回传到 `x`、`weight`、`bias` |

![前向传播与反向传播的训练闭环](../public/02_PyTorch_Algorithms/00_forward_backward_flow.svg)
### Step 4：代码设计与基础契约验证

本题把 token id、张量重排、Embedding 查表以及 Linear → ReLU 的前向/反向拆成五个机制责任。固定骨架提供输入接口和中间状态名称；学习者补全能够决定结果的检查、变换与梯度计算。

| TODO | 代码契约与机制责任 | 关键约束 | 测试证据 |
| --- | --- | --- | --- |
| TODO 1 | 校验 tokenizer 输出的 token id 张量 | `[B, S]`、`torch.long`、id 位于词表范围 | 正常输入通过；维度、dtype、越界 id 被拒绝 |
| TODO 2 | 将二维特征重排为序列表示 | 元素数不变，目标形状与输入一致 | 原生重排与 `einops` 数值一致 |
| TODO 3 | 验证 Embedding 是按 id 查表 | id 选择对应权重行 | 官方接口与直接索引等价 |
| TODO 4 | 完成 Linear → ReLU 前向并保存反向所需状态 | 输出形状与激活掩码可追踪 | 与 PyTorch 前向结果一致 |
| TODO 5 | 从保存状态回传输入、权重和 bias 梯度 | 三类梯度形状与参数对应 | 与 Autograd 梯度一致 |

```python
# 导入所有必需的库
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
```


```python
def validate_token_ids(input_ids: torch.Tensor, vocab_size: int) -> None:
    """检查 tokenizer 输出能否直接作为 Embedding 的输入。"""
    # TODO 1：验证 token id 输入契约。
    # is_rank_two = ???
    # is_integer_ids = ???
    # ids_in_vocab = ???
    # 若任一条件不满足，抛出 ValueError。
    raise NotImplementedError('TODO 1：请验证 token id 输入')


def tensor_warmup(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """将 `[B, C, H, W]` 图像特征整理为 `[B, H×W, C]` 序列。"""
    if x.ndim != 4:
        raise ValueError('x 必须是 [batch, channels, height, width]')
    # TODO 2：分别使用原生张量操作和 einops 完成同一重排。
    # x_native = ???
    # x_einops = ???
    raise NotImplementedError('TODO 2：请完成空间特征到序列的重排')
    return x_native, x_einops


def embedding_warmup(input_ids: torch.Tensor, vocab_size: int, hidden_dim: int):
    """对照 `nn.Embedding` 与直接权重索引的查表结果。"""
    validate_token_ids(input_ids, vocab_size)
    if hidden_dim <= 0:
        raise ValueError('hidden_dim 必须为正数')
    # TODO 3：构造词表矩阵，并用两种方式按 token id 查表。
    # emb_layer = ???
    # out_official = ???
    # out_manual = ???
    raise NotImplementedError('TODO 3：请完成 Embedding 查表对照')
    return out_official, out_manual


class LinearReLUFunction(torch.autograd.Function):
    """实现 `relu(x @ W^T + b)`，并显式保存 ReLU 反向所需状态。"""
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
        # TODO 4：完成 Linear → ReLU 前向，并保存反向所需张量。
        # z = ???
        # y = ???
        # mask = ???
        # ctx.save_for_backward(???)
        raise NotImplementedError('TODO 4：请实现 Linear → ReLU 前向')
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight, mask = ctx.saved_tensors
        # TODO 5：先通过 ReLU mask 过滤上游梯度，再回传 x / weight / bias。
        # grad_z = ???
        # grad_x = ???
        # grad_weight = ???
        # grad_bias = ???
        raise NotImplementedError('TODO 5：请实现 Linear → ReLU 反向')
        return grad_x, grad_weight, grad_bias
```


```python
# 测试设计：分别验证 token id、张量重排、Embedding 查表和 Linear → ReLU 的前向/反向。
# 每个测试对应一个机制责任，方便定位未完成的 TODO。

def test_token_id_contract():
    validate_token_ids(torch.tensor([[0, 2], [1, 3]], dtype=torch.long), 4)
    for input_ids in (torch.tensor([0, 1]), torch.tensor([[0.0, 1.0]]), torch.tensor([[0, 4]], dtype=torch.long)):
        try: validate_token_ids(input_ids, 4)
        except ValueError: continue
        raise AssertionError('不合法的 tokenizer 输出应被拒绝')

def test_spatial_to_sequence_rearrangement():
    x=torch.arange(2*3*4*5,dtype=torch.float32).reshape(2,3,4,5)
    native, semantic=tensor_warmup(x)
    assert native.shape == semantic.shape == (2,20,3)
    assert torch.equal(native,semantic) and torch.equal(native[0,0],x[0,:,0,0])
    try: tensor_warmup(torch.randn(2,3,4))
    except ValueError: pass
    else: raise AssertionError('非四维特征应被拒绝')

def test_embedding_lookup_equivalence():
    official,manual=embedding_warmup(torch.tensor([[0,3],[2,1]],dtype=torch.long),4,6)
    assert official.shape == manual.shape == (2,2,6)
    assert torch.allclose(official,manual)

def test_linear_relu_forward():
    x=torch.tensor([[1.,-2.],[.5,3.]],requires_grad=True)
    weight=torch.tensor([[1.,1.],[-1.,.5]],requires_grad=True); bias=torch.tensor([0.,-.5],requires_grad=True)
    actual=LinearReLUFunction.apply(x,weight,bias); expected=F.relu(F.linear(x,weight,bias))
    assert torch.allclose(actual,expected)
    assert torch.all(actual[F.linear(x,weight,bias)<0] == 0)

def test_linear_relu_backward():
    x=torch.randn(2,4,requires_grad=True); weight=torch.randn(3,4,requires_grad=True); bias=torch.randn(3,requires_grad=True)
    F.relu(F.linear(x,weight,bias)).sum().backward(); expected=(x.grad.clone(),weight.grad.clone(),bias.grad.clone())
    x.grad=weight.grad=bias.grad=None
    LinearReLUFunction.apply(x,weight,bias).sum().backward()
    assert torch.allclose(x.grad,expected[0]) and torch.allclose(weight.grad,expected[1]) and torch.allclose(bias.grad,expected[2])

def run_warmup_tests():
    test_token_id_contract(); test_spatial_to_sequence_rearrangement(); test_embedding_lookup_equivalence(); test_linear_relu_forward(); test_linear_relu_backward()
    print('✅ Warmup：输入、重排、查表、前向与反向契约测试通过')

try: run_warmup_tests()
except NotImplementedError:
    print('请先完成 TODO 1–5，再运行测试。'); raise
```

## 参考代码与解析

### 代码

```python
def validate_token_ids(input_ids: torch.Tensor, vocab_size: int) -> None:
    """检查 tokenizer 输出能否直接作为 Embedding 的输入。"""
    # TODO 1：验证 token id 输入契约。
    is_rank_two = input_ids.ndim == 2
    is_integer_ids = input_ids.dtype == torch.long
    ids_in_vocab = vocab_size > 0 and (input_ids.numel() == 0 or (input_ids.min() >= 0 and input_ids.max() < vocab_size))
    if not (is_rank_two and is_integer_ids and ids_in_vocab):
        raise ValueError('input_ids 必须是词表范围内的二维 torch.long 张量')


def tensor_warmup(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """将 `[B, C, H, W]` 图像特征整理为 `[B, H×W, C]` 序列。"""
    if x.ndim != 4:
        raise ValueError('x 必须是 [batch, channels, height, width]')
    # TODO 2：分别使用原生张量操作和 einops 完成同一重排。
    x_native = x.permute(0, 2, 3, 1).reshape(x.shape[0], x.shape[2] * x.shape[3], x.shape[1])
    x_einops = einops.rearrange(x, 'b c h w -> b (h w) c')
    return x_native, x_einops


def embedding_warmup(input_ids: torch.Tensor, vocab_size: int, hidden_dim: int):
    """对照 `nn.Embedding` 与直接权重索引的查表结果。"""
    validate_token_ids(input_ids, vocab_size)
    if hidden_dim <= 0:
        raise ValueError('hidden_dim 必须为正数')
    # TODO 3：构造词表矩阵，并用两种方式按 token id 查表。
    emb_layer = nn.Embedding(vocab_size, hidden_dim)
    emb_layer.weight.data.normal_(0, 0.1)
    out_official = emb_layer(input_ids)
    out_manual = emb_layer.weight[input_ids]
    return out_official, out_manual


class LinearReLUFunction(torch.autograd.Function):
    """实现 `relu(x @ W^T + b)`，并显式保存 ReLU 反向所需状态。"""
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
        # TODO 4：完成 Linear → ReLU 前向，并保存反向所需张量。
        z = F.linear(x, weight, bias)
        y = F.relu(z)
        mask = z > 0
        ctx.save_for_backward(x, weight, mask)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight, mask = ctx.saved_tensors
        # TODO 5：先通过 ReLU mask 过滤上游梯度，再回传 x / weight / bias。
        grad_z = grad_output * mask
        grad_x = grad_z @ weight
        grad_weight = grad_z.transpose(-1, -2) @ x
        grad_bias = grad_z.sum(dim=0)
        return grad_x, grad_weight, grad_bias
```

### 解析

本题把模型输入到梯度回传的最小链路拆为五个可检查步骤。答案与题目区使用同一套函数和类，只补全输入契约、形状变换、查表和前后向机制。

**TODO 1：校验 token id**：Embedding 接收二维 `torch.long` id，且每个 id 都要落在 `[0, vocab_size)`。

**TODO 2：空间特征变成序列**：先将通道移到最后，再合并 `H` 与 `W`；原生操作与 `einops` 的数值必须一致。

**TODO 3：Embedding 是查表**：`nn.Embedding(input_ids)` 与 `weight[input_ids]` 都按 id 取出词表矩阵的对应行。

**TODO 4：保存前向状态**：Linear 得到 `z`，ReLU 输出 `y`；`z > 0` 的 mask 决定哪些上游梯度可以通过。

**TODO 5：按链式法则回传**：先用 mask 得到 `grad_z`，再分别计算输入、权重和 bias 的梯度。
## 相关阅读

本节的基础算子可以沿两个方向继续：先看 PyTorch 的官方张量与自动求导接口，再把这些操作放回 GPU 精度和内存层级。

- [PyTorch 张量文档](https://pytorch.org/docs/stable/tensors.html)
- [PyTorch 自动求导文档](https://pytorch.org/docs/stable/autograd.html)
- [einops 开源仓库](https://github.com/arogozhnikov/einops)
- [PyTorch 原论文：An Imperative Style, High-Performance Deep Learning Library](https://arxiv.org/abs/1912.01703)
- [P1: 大模型的数据格式与混合精度](../01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.md)
- [P1: GPU 物理架构与内存层级](../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.md)
