# 00. PyTorch Warmup | PyTorch 热身

**难度：** Easy | **环境：** CPU-first | **标签：** `PyTorch`, `基础入门`, `反向传播` | **目标人群：** 通用基础 (算法/Infra)

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/00_PyTorch_Warmup.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


在深入大模型的浩瀚海洋（如 Attention、LoRA、MoE）之前，我们必须确保自己的“底层积木”是非常扎实的。
本节作为**热身关卡**，将用三个非常经典的实战填空，带你快速找回 PyTorch 的核心肌肉记忆：张量维度变换 (Tensor Reshaping)、嵌入层查表 (Embedding Lookup) 以及链式法则的反向传播 (Backpropagation)。
如果你对 Transformer 还不熟，可以先把几个词记成最小概念：`token id` 是离散编号，`embedding` 是把编号查表成向量的层，`hidden dim` 是这个向量的长度，`sequence` 则是把一串 token 组成的输入。
本页不重复展开前置概念，而是把这些知识直接落到可运行、可验证的练习里，重点看清每一步代码为什么这么写。

**关键词：** `reshape`, `Embedding`, `backpropagation`
## 前置阅读

**导语：** 这一节先把后续章节要用到的基础张量、Autograd 和训练接口先补齐。

- [P0: 05. PyTorch Tensor Fundamentals | PyTorch 张量基础操作](../00_Prerequisites/05_PyTorch_Tensor_Fundamentals.md)
- [P0: 07. PyTorch Autograd and Backward | PyTorch 自动求导与反向传播](../00_Prerequisites/07_PyTorch_Autograd_and_Backward.md)
- [P0: 09. PyTorch nn.Module Basics | PyTorch nn.Module 基础](../00_Prerequisites/09_PyTorch_nn_Module_Basics.md)

## 相关阅读

**导语：** 本节先把 PyTorch 的热身算子讲清楚；如果想继续看张量数据类型和 GPU 架构，再顺着读下面这些页。

- [P1: 01. Data Types and Precision | 大模型的数据格式与混合精度](../01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.md)
- [P1: 03. GPU Architecture and Memory | GPU 物理架构与内存层级](../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.md)


```python
# 导入所有必需的库
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
```

### Part 1: 张量维度变换与 `einops`

无论是注意力里的多头合并，还是各种特征整理，都会反复用到张量形状重排同一类操作。

> **为什么我们需要 `einops`？**
> 在大模型开发中，张量形状不匹配（`RuntimeError: size mismatch`）是最高频的调试痛点之一。熟练掌握原生的 `view`, `reshape`, `transpose`, `permute` 是算法工程师的基础功底。
> 
> 然而，在实际的工业级代码中（尤其是 Transformer 的多头注意力机制等高维张量操作），原生方法往往缺乏可读性且极易出错。
> 举个典型的例子：将形状为 `[batch, heads, seq_len, head_dim]` 的多头张量合并为 `[batch, seq_len, hidden_dim]`：
> - **原生实现**：`x.permute(0, 2, 1, 3).reshape(batch, seq_len, -1)` —— 开发者必须在脑海中硬记数字索引 `(0,2,1,3)` 的物理含义，代码维护成本极高。
> - **`einops` 实现**：`rearrange(x, 'b h s d -> b s (h d)')` —— 维度变换的语义直接写在字符串中，代码即文档（Self-documenting）。
>
> 这正是为什么现代深度学习框架和开源模型广泛拥抱 **`einops`** 库，它能让复杂的张量操作变得语义清晰、安全可防错。

```python
def tensor_warmup(x: torch.Tensor):
    """
    假设 x 是一批图像的特征 (例如在多模态大模型中)，形状为 [batch_size, channels, height, width]
    我们需要将其展平为序列 (Sequence)，以输入给 Transformer。
    目标形状: [batch_size, height * width, channels]
    """
    
    # 这一段先把图像特征重排成 Transformer 更容易消费的序列格式。
    # 重点不是记住某个 API，而是理解“先换维度顺序，再合并空间维”的处理思路。
    # ==========================================
    # TODO 1.1: 使用原生的 PyTorch 方法 (permute + reshape/flatten) 完成变换
    # 提示: 先调整维度顺序，再合并空间维度
    # ==========================================
    # x_native = ???
    
    # 这里再用 `einops` 做同一件事，方便对照“显式维度操作”和“语义化写法”。
    # 两种写法应该得到完全一致的结果，便于确认你理解的是形状变换本身。
    # ==========================================
    # TODO 1.2: 使用 einops.rearrange 优雅地完成完全相同的操作
    # 提示: 使用括号表示要合并的维度
    # ==========================================
    # x_einops = ???
    return x_native, x_einops

```

### Part 2: 嵌入层 (Embedding Layer) 的本质


大模型的第一步，是把离散的文本转化为连续的数学表示：
原始文本经字节对编码(Byte Pair Encoding，BPE)分词后得到`token id`（即词典里的编号），再通过嵌入层（`Embedding`）查表，将该编号映射为固定长度的连续向量（其维度称为`hidden dim`或者`hidden size`）；随后注入旋转位置编码（主流方案如`RoPE`及其变体）以保留顺序信息。若一句话包含多个token，这些向量会按顺序拼接成完整的输入序列（`sequence`），供模型后续处理，从而让模型真正开始“阅读”和理解文本。

>**那 Embedding 层本质上是什么？就是一张巨大的查找表（Lookup Table）。** 给定一个 ID 列表，它直接把对应的行向量抽出来拼在一起。
>它在数学上等价于：把离散的 ID 转换成 One-hot 向量，然后去乘以一个全连接层（Linear）。


```python
def embedding_warmup(input_ids: torch.Tensor, vocab_size: int, hidden_dim: int):
    """
    演示 Embedding 查表的过程，并用纯 Tensor 索引模拟它。
    
    Args:
        input_ids: 形状 [batch_size, seq_len]，包含整数类型的 Token IDs
    """
    # 这一段先把“词表查表”这个抽象过程落成官方实现。
    # 先看 nn.Embedding 如何工作，再手写索引复现同样的输出。
    # ==========================================
    # TODO 2.1: 实例化一个官方的 nn.Embedding，并用其进行前向传播
    # ==========================================
    # emb_layer = ???
    # emb_layer.weight.data.normal_(0, 0.1)  # 随便初始化一下
    # out_official = ???
    
    # 这里不再调用 nn.Embedding，而是直接用权重矩阵做索引。
    # 这样可以把“Embedding 本质上就是查表”这件事看得更直观。
    # ==========================================
    # TODO 2.2: 使用纯 PyTorch 张量索引 (Advanced Indexing)，不使用 nn.Embedding，
    # 达到和上面官方 API 完全一模一样的输出。
    # 提示: Embedding 的本质是查表，思考如何用索引从权重矩阵中提取向量
    # ==========================================
    # out_manual = ???
    return out_official, out_manual
```

### Part 3: 前向传播与反向传播 (Forward & Backward)


> **为什么要理解前向和反向传播？**
> 大模型的训练机制完全建立在**反向传播算法 (Backpropagation)** 与 **链式法则 (Chain Rule)** 之上。
> 
> **前向传播 (Forward Pass)：** 数据从输入层流向输出层，经过一系列的线性变换和非线性激活函数。在这个过程中，我们需要保存中间结果（如激活值、mask 等），供反向传播使用。
> 
> **反向传播 (Backward Pass)：** 梯度从输出层反向流向输入层，利用链式法则逐层计算每个参数的梯度。这是深度学习训练的核心机制。
> 
> 在日常使用中，我们只需要写前向传播，然后调用 `loss.backward()`，PyTorch 的 Autograd 会自动帮我们算梯度。但为了真正理解底层原理，我们需要手动实现一个包含 Linear 和 ReLU 的自定义算子的完整前向和反向逻辑。
> 
> **本节目标：** 实现一个 `LinearReLU` 算子，公式为 `y = relu(x @ W^T + b)`，并手动推导其梯度。这将帮助你深入理解：
> - 前向传播如何计算输出并保存中间结果
> - 反向传播如何利用链式法则计算梯度
> - 为什么需要在前向传播时保存某些张量（如 mask）
> 
> **闭环提示：** 先算 `Linear -> ReLU`，再保存 `mask`；反向时先过 `mask`，再把梯度回传到输入和参数。


```python
class LinearReLUFunction(torch.autograd.Function):
    """
    实现一个包含 Linear + ReLU 的算子，并推导其反向传播的梯度。
    公式: y = relu(x @ W^T + b)
    """
    
    @staticmethod
    def forward(ctx, x, weight, bias):
        # 这里先把线性变换和激活拆开，便于明确哪些中间量要留给反向传播。
        # mask 的保存是关键：它决定了 ReLU 之后哪些位置还能继续传梯度。
        # ==========================================
        # TODO 3.1: 实现前向传播
        # 1. 使用 F.linear 计算线性变换
        # 2. 使用 F.relu 计算激活
        # 3. 计算并保存 mask，用于反向传播时判断哪些位置需要传递梯度
        # 4. 保存必要的张量供反向传播使用
        # ==========================================
        # z = ???
        # y = ???
        # mask = ???
        # ctx.save_for_backward(???)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """
        接收从上一层传回来的梯度 (grad_output)，形状同 y。
        返回对当前层三个输入 (x, weight, bias) 的梯度。
        """
        x, weight, mask = ctx.saved_tensors
        
        # 先把上游梯度过 ReLU 的门，再进入线性层的矩阵求导。
        # 这一小段是整个自定义算子最核心的反向链路。
        # ==========================================
        # TODO 3.2: 反传过 ReLU
        # 提示: ReLU 的导数在正值处为 1，负值处为 0
        # ==========================================
        # grad_z = ???
        
        # 接着把梯度回传到输入和参数，分别得到 x / weight / bias 的梯度。
        # 这里的形状对齐和转置关系，是矩阵求导最常见的检查点。
        # ==========================================
        # TODO 3.3: 反传过 Linear
        # 提示: 利用矩阵求导的链式法则，分别计算对 x, weight, bias 的梯度
        # 注意矩阵维度的匹配和转置操作
        # ==========================================
        # grad_x = ???
        # grad_weight = ???
        # grad_bias = ???
        
        return grad_x, grad_weight, grad_bias

```


```python
# 运行此单元格以测试你的实现
def test_warmup():
    try:
        print("=" * 60)
        print("开始测试 PyTorch Warmup 练习")
        print("=" * 60)
        
        # ==========================================
        # Test 1: 张量维度变换
        # ==========================================
        print("\n【Test 1】张量维度变换测试")
        x_img = torch.randn(2, 3, 224, 224)
        n, e = tensor_warmup(x_img)
        
        # 测试形状
        assert n.shape == (2, 224*224, 3), f"原生方法输出形状错误: 期望 (2, 50176, 3), 实际 {n.shape}"
        assert e.shape == (2, 224*224, 3), f"einops 输出形状错误: 期望 (2, 50176, 3), 实际 {e.shape}"
        
        # 测试两种方法结果一致
        assert torch.allclose(n, e), "原生方法与 einops 结果不一致！"
        
        # 测试数值正确性：验证第一个样本的第一个 patch
        expected_first_patch = x_img[0, :, 0, 0]  # [channels]
        actual_first_patch = n[0, 0, :]  # [channels]
        assert torch.allclose(expected_first_patch, actual_first_patch), "维度变换后数值不正确！"
        
        print("  ✅ 形状测试通过")
        print("  ✅ 原生方法与 einops 一致性测试通过")
        print("  ✅ 数值正确性测试通过")
        
        # ==========================================
        # Test 2: Embedding 层模拟
        # ==========================================
        print("\n【Test 2】Embedding 层查表测试")
        ids = torch.randint(0, 1000, (4, 16))
        off, man = embedding_warmup(ids, vocab_size=1000, hidden_dim=64)
        
        # 测试形状
        assert off.shape == (4, 16, 64), f"官方 Embedding 输出形状错误: 期望 (4, 16, 64), 实际 {off.shape}"
        assert man.shape == (4, 16, 64), f"手动索引输出形状错误: 期望 (4, 16, 64), 实际 {man.shape}"
        
        # 测试两种方法结果一致
        assert torch.allclose(off, man), "手动 Embedding 查表与官方实现不一致！"
        
        print("  ✅ 形状测试通过")
        print("  ✅ 官方实现与手动索引一致性测试通过")
        
        # ==========================================
        # Test 3.1: 前向传播测试
        # ==========================================
        print("\n【Test 3.1】前向传播测试")
        x = torch.randn(2, 4, requires_grad=True)
        w = torch.randn(3, 4, requires_grad=True)
        b = torch.randn(3, requires_grad=True)
        
        # 使用自定义算子
        y_custom = LinearReLUFunction.apply(x, w, b)
        
        # 使用官方算子作为标准答案
        y_std = F.relu(F.linear(x, w, b))
        
        # 测试形状
        assert y_custom.shape == (2, 3), f"前向传播输出形状错误: 期望 (2, 3), 实际 {y_custom.shape}"
        
        # 测试数值一致性
        assert torch.allclose(y_custom, y_std, rtol=1e-5, atol=1e-6), "前向传播结果与官方实现不一致！"
        
        # 测试 ReLU 是否正确：负值应该被置零
        z_before_relu = F.linear(x, w, b)
        negative_mask = z_before_relu < 0
        assert torch.all(y_custom[negative_mask] == 0), "ReLU 未正确将负值置零！"
        
        print("  ✅ 形状测试通过")
        print("  ✅ 与官方实现一致性测试通过")
        print("  ✅ ReLU 负值置零测试通过")
        
        # ==========================================
        # Test 3.2 & 3.3: 反向传播测试
        # ==========================================
        print("\n【Test 3.2 & 3.3】反向传播测试")
        
        # 重新创建张量（因为上面已经计算过梯度）
        x = torch.randn(2, 4, requires_grad=True)
        w = torch.randn(3, 4, requires_grad=True)
        b = torch.randn(3, requires_grad=True)
        
        # 使用官方算子计算梯度
        y_std = F.relu(F.linear(x, w, b))
        y_std.sum().backward()
        std_gx, std_gw, std_gb = x.grad.clone(), w.grad.clone(), b.grad.clone()
        
        # 清零梯度
        x.grad.zero_()
        w.grad.zero_()
        b.grad.zero_()
        
        # 使用自定义算子计算梯度
        y_custom = LinearReLUFunction.apply(x, w, b)
        y_custom.sum().backward()
        
        # 测试梯度一致性
        assert torch.allclose(x.grad, std_gx, rtol=1e-5, atol=1e-6), "对 x 的梯度计算不正确！"
        assert torch.allclose(w.grad, std_gw, rtol=1e-5, atol=1e-6), "对 weight 的梯度计算不正确！"
        assert torch.allclose(b.grad, std_gb, rtol=1e-5, atol=1e-6), "对 bias 的梯度计算不正确！"
        
        # 测试梯度形状
        assert x.grad.shape == x.shape, f"x 的梯度形状错误: 期望 {x.shape}, 实际 {x.grad.shape}"
        assert w.grad.shape == w.shape, f"weight 的梯度形状错误: 期望 {w.shape}, 实际 {w.grad.shape}"
        assert b.grad.shape == b.shape, f"bias 的梯度形状错误: 期望 {b.shape}, 实际 {b.grad.shape}"
        
        print("  ✅ 对 x 的梯度测试通过")
        print("  ✅ 对 weight 的梯度测试通过")
        print("  ✅ 对 bias 的梯度测试通过")
        print("  ✅ 梯度形状测试通过")
        
        # ==========================================
        # 全部通过
        # ==========================================
        print("\n" + "=" * 60)
        print(" PyTorch 核心操作测试通过。")
        print("   所有测试用例均已通过，可以正式开启大模型的浩瀚旅程了！")
        print("=" * 60)
        
    except NotImplementedError:
        print("\n❌ 测试失败: 请先完成 TODO 部分的代码！")
        raise
    except (AttributeError, NameError, TypeError) as e:
        if isinstance(e, AttributeError):
            print("\n❌ 测试失败: 代码未完成，无法找到必要的属性")
        elif isinstance(e, NameError):
            print("\n❌ 测试失败: 代码可能未完成，导致了变量未定义")
        else:
            print("\n❌ 测试失败: 代码可能未完成，导致类型错误")
            print(f"   错误信息: {e}")
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        raise
    except Exception as e:
        print(f"\n❌ 发生未知异常: {type(e).__name__}: {e}")
        raise

test_warmup()
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
def tensor_warmup(x: torch.Tensor):
    # TODO 1.1 & 1.2
    # 先把通道维挪到最后，再把空间维合并成序列维。
    x_native = x.permute(0, 2, 3, 1).reshape(x.shape[0], x.shape[2] * x.shape[3], x.shape[1])
    # `einops` 只是同一件事的语义化写法，方便对照理解。
    x_einops = einops.rearrange(x, "b c h w -> b (h w) c")
    return x_native, x_einops

def embedding_warmup(input_ids: torch.Tensor, vocab_size: int, hidden_dim: int):
    # TODO 2.1 & 2.2
    emb_layer = nn.Embedding(vocab_size, hidden_dim)
    emb_layer.weight.data.normal_(0, 0.1)
    # 官方实现直接按 token id 查表，返回对应行向量。
    out_official = emb_layer(input_ids)
    # 手动实现也是同样的高级索引查表。
    out_manual = emb_layer.weight[input_ids]
    return out_official, out_manual

class LinearReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        # TODO 3.1
        # 先算线性项，再过 ReLU；mask 要保存给反向传播使用。
        z = F.linear(x, weight, bias)
        y = F.relu(z)
        mask = (z > 0).float()
        ctx.save_for_backward(x, weight, mask)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, mask = ctx.saved_tensors
        
        # TODO 3.2
        # ReLU 的梯度就是一个 0/1 掩码，直接过滤上游梯度。
        grad_z = grad_output * mask
        
        # TODO 3.3
        # 再把梯度分别回传到输入、权重和偏置。
        grad_x = grad_z @ weight
        grad_weight = grad_z.T @ x
        grad_bias = grad_z.sum(dim=0)
        
        return grad_x, grad_weight, grad_bias
```

### 答案与直觉

**1. TODO 1.1 & 1.2 (张量维度变换)**

- **这一题要解决什么：** 把 `[batch, channels, height, width]` 的图像特征整理成 `[batch, height * width, channels]` 的序列格式，方便后续送入 Transformer。
- **为什么先 `permute` 再 `reshape`：** 先把通道维挪到最后，再合并空间维，这样形状语义最清楚，也最不容易写错。
- **`einops` 的意义：** `rearrange(x, 'b c h w -> b (h w) c')` 只是更语义化的表达，核心做的仍然是同一件形状重排。
- **带走的直觉：** 这类题的重点不是记 API，而是形成“先换顺序，再合并维度”的稳定思维。

**2. TODO 2.1 & 2.2 (Embedding 层模拟)**

- **这一题要解决什么：** 把离散的 token id 映射成连续向量，让文本输入真正变成神经网络可处理的张量。
- **为什么能手动复现：** `nn.Embedding` 本质上就是维护一个词表矩阵，然后按 `input_ids` 去取对应行，所以 `emb_layer.weight[input_ids]` 可以得到相同结果。
- **为什么比 One-hot 更合适：** 查表直接取值，避免了大规模稀疏向量乘法，词表越大越能体现这种实现方式的优势。
- **带走的直觉：** 遇到 Embedding 时，先把它理解成“查表层”，再去看实现细节。

**3. TODO 3.1 (前向传播)**

- **这一题要解决什么：** 手动拼出一个最小的 `Linear + ReLU` 前向链路，并明确哪些中间量要保留给反向传播。
- **关键步骤：** 先算 `z = x @ weight.T + bias`，再过 `ReLU` 得到输出 `y`，这是最标准的前向组合。
- **为什么要保存 `mask`：** `mask = (z > 0).float()` 记录了哪些位置能继续传梯度，反向时它会直接参与链式法则。
- **带走的直觉：** 前向不是只管算出结果，还要为反向预留必要的中间状态。

**4. TODO 3.2 (ReLU 反向传播)**

- **这一题要解决什么：** 把上游梯度先穿过 ReLU 的“门”，只让正区间的位置继续传回去。
- **为什么是逐元素相乘：** ReLU 的导数本身就是一个 0/1 掩码，所以 `grad_z = grad_output * mask` 正好对应链式法则。
- **带走的直觉：** 激活函数不是纯前向的装饰，它会直接改变反向传播的梯度流。

**5. TODO 3.3 (Linear 反向传播)**

- **这一题要解决什么：** 把梯度从输出层回传到输入和参数，完整补上一个线性层的反向链路。
- **三个梯度分别对应什么：** `grad_x` 回到输入，`grad_weight` 回到参数矩阵，`grad_bias` 回到广播到 batch 维的偏置项。
- **为什么要看转置和求和：** 矩阵求导里最容易出错的就是维度对齐，`grad_weight = grad_z.T @ x` 和 `grad_bias = grad_z.sum(dim=0)` 正是在处理这两件事。
- **带走的直觉：** 会手推这一层，后面再看更复杂的自定义算子、融合算子和 CUDA 实现时，就更容易理解它们为什么要保存哪些状态。