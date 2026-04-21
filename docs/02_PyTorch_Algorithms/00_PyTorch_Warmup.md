# 00. PyTorch Warmup | PyTorch 核心基础热身: 张量、前反向传播与 Embedding (Warmup)

**难度：** Easy | **标签：** `PyTorch`, `Foundation` | **目标人群：** 通用基础 (算法/Infra)

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/00_PyTorch_Warmup.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


在深入大模型的浩瀚海洋（如 Attention、LoRA、MoE）之前，我们必须确保自己的“底层积木”是非常扎实的。
本节作为**热身关卡**，将用三个非常经典的实战填空，带你快速找回 PyTorch 的核心肌肉记忆：张量维度变换 (Tensor Reshaping)、嵌入层查表 (Embedding Lookup) 以及链式法则的反向传播 (Backpropagation)。


```python
# 导入所有必需的库
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
```

### Part 1: 张量维度变换与 `einops`

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
    
    # ==========================================
    # TODO 1.1: 使用原生的 PyTorch 方法 (permute + reshape/flatten) 完成变换
    # 提示: 先调整维度顺序，再合并空间维度
    # ==========================================
    # x_native = ???
    
    # ==========================================
    # TODO 1.2: 使用 einops.rearrange 优雅地完成完全相同的操作
    # 提示: 使用括号表示要合并的维度
    # ==========================================
    # x_einops = ???
    
    # return x_native, x_einops
    pass
```

### Part 2: 嵌入层 (Embedding Layer) 的本质

>文本是离散的（Token IDs，如 `[10, 42, 99]`）。神经网络只能处理连续的稠密向量（Dense Vectors）。
>**Embedding 层的本质：** 就是一个大规模的查表（Lookup Table）。给定一个 ID 列表，它直接把对应的行向量抽出来拼在一起。
>它在数学上等价于：把离散的 ID 转换成 One-hot 向量，然后去乘以一个全连接层（Linear）。


```python
def embedding_warmup(input_ids: torch.Tensor, vocab_size: int, hidden_dim: int):
    """
    演示 Embedding 查表的过程，并用纯 Tensor 索引模拟它。
    
    Args:
        input_ids: 形状 [batch_size, seq_len]，包含整数类型的 Token IDs
    """
    # ==========================================
    # TODO 2.1: 实例化一个官方的 nn.Embedding，并用其进行前向传播
    # ==========================================
    # emb_layer = ???
    # emb_layer.weight.data.normal_(0, 0.1)  # 随便初始化一下
    # out_official = ???
    
    # ==========================================
    # TODO 2.2: 使用纯 PyTorch 张量索引 (Advanced Indexing)，不使用 nn.Embedding，
    # 达到和上面官方 API 完全一模一样的输出。
    # 提示: Embedding 的本质是查表，思考如何用索引从权重矩阵中提取向量
    # ==========================================
    # out_manual = ???
    
    # return out_official, out_manual
    pass
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

```python
class LinearReLUFunction(torch.autograd.Function):
    """
    实现一个包含 Linear + ReLU 的算子，并推导其反向传播的梯度。
    公式: y = relu(x @ W^T + b)
    """
    
    @staticmethod
    def forward(ctx, x, weight, bias):
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
        # return ???
        pass

    @staticmethod
    def backward(ctx, grad_output):
        """
        接收从上一层传回来的梯度 (grad_output)，形状同 y。
        返回对当前层三个输入 (x, weight, bias) 的梯度。
        """
        x, weight, mask = ctx.saved_tensors
        
        # ==========================================
        # TODO 3.2: 反传过 ReLU
        # 提示: ReLU 的导数在正值处为 1，负值处为 0
        # ==========================================
        # grad_z = ???
        
        # ==========================================
        # TODO 3.3: 反传过 Linear
        # 提示: 利用矩阵求导的链式法则，分别计算对 x, weight, bias 的梯度
        # 注意矩阵维度的匹配和转置操作
        # ==========================================
        # grad_x = ???
        # grad_weight = ???
        # grad_bias = ???
        
        # return grad_x, grad_weight, grad_bias
        pass
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
    except TypeError as e:
        print(f"\n❌ 测试失败: 代码可能未完成，导致类型错误")
        print(f"   错误信息: {e}")
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
    except Exception as e:
        print(f"\n❌ 发生未知异常: {type(e).__name__}: {e}")

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
    x_native = x.permute(0, 2, 3, 1).reshape(x.shape[0], x.shape[2] * x.shape[3], x.shape[1])
    x_einops = einops.rearrange(x, "b c h w -> b (h w) c")
    return x_native, x_einops

def embedding_warmup(input_ids: torch.Tensor, vocab_size: int, hidden_dim: int):
    # TODO 2.1 & 2.2
    emb_layer = nn.Embedding(vocab_size, hidden_dim)
    emb_layer.weight.data.normal_(0, 0.1)
    out_official = emb_layer(input_ids)
    out_manual = emb_layer.weight[input_ids]
    return out_official, out_manual

class LinearReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        # TODO 3.1
        z = F.linear(x, weight, bias)
        y = F.relu(z)
        mask = (z > 0).float()
        ctx.save_for_backward(x, weight, mask)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, mask = ctx.saved_tensors
        
        # TODO 3.2
        grad_z = grad_output * mask
        
        # TODO 3.3
        grad_x = grad_z @ weight
        grad_weight = grad_z.T @ x
        grad_bias = grad_z.sum(dim=0)
        
        return grad_x, grad_weight, grad_bias
```

### 解析

**1. TODO 1.1 & 1.2 (张量维度变换)**

- **原生实现：** 使用 `permute(0, 2, 3, 1)` 将维度重排为 `[batch, height, width, channels]`，再用 `reshape` 合并空间维度。必须使用 `x.shape[...]` 动态获取维度，避免硬编码。
- **`einops` 实现：** `rearrange(x, 'b c h w -> b (h w) c')` 一行完成，括号 `(h w)` 表示合并维度，语义清晰。
- **工程推荐：** 在多头注意力等复杂张量操作中，`einops` 的可读性远超原生方法，代码即文档。

**2. TODO 2.1 & 2.2 (Embedding 层模拟)**

- **官方实现：** `nn.Embedding(vocab_size, hidden_dim)` 内部维护权重矩阵 `[vocab_size, hidden_dim]`，调用时用 `input_ids` 索引提取行向量。
- **手动实现：** 直接使用高级索引 `emb_layer.weight[input_ids]` 达到相同效果，揭示了 Embedding 的本质是查表而非矩阵乘法。
- **进阶思考：** 为什么查表比 One-hot 乘法快？One-hot 产生大量零元素（稀疏矩阵），而直接索引只需一次内存访问，在大词表场景（50k+ tokens）性能优势明显。

**3. TODO 3.1 (前向传播)**

- **Linear 层计算：** `F.linear(x, weight, bias)` 完成 `z = x @ weight.T + bias`，注意 `F.linear` 内部自动转置 `weight`。
- **ReLU 激活：** `F.relu(z)` 将负值置零，数学定义为 `relu(z) = max(0, z)`。
- **保存中间结果：** 计算 `mask = (z > 0).float()` 并通过 `ctx.save_for_backward(x, weight, mask)` 保存，供反向传播使用。`mask` 记录哪些位置大于0，`x` 和 `weight` 用于计算梯度。

**4. TODO 3.2 (ReLU 反向传播)**

- **梯度计算：** `grad_z = grad_output * mask`，其中 `mask` 是前向保存的 `(z > 0).float()`，充当 ReLU 导数的角色。
- **数学原理：** ReLU 导数为 `d_relu(z)/dz = 1 if z > 0 else 0`，根据链式法则 `grad_z = grad_output * mask`。

**5. TODO 3.3 (Linear 反向传播)**

- **对输入 `x` 的梯度：** `grad_x = grad_z @ weight`，根据矩阵求导链式法则计算。
- **对权重 `weight` 的梯度：** `grad_weight = grad_z.T @ x`，需转置 `grad_z` 以匹配 `weight` 形状 `[out_features, in_features]`。
- **对偏置 `bias` 的梯度：** `grad_bias = grad_z.sum(dim=0)`，因为 `bias` 在前向传播中被广播到每个样本，反向时需沿 batch 维度求和累加。
- **进阶思考：** 理解手动推导是编写自定义 CUDA 算子（如 Flash Attention、Fused Operators）的必备基础，这些高性能算子需要手动实现前向和反向传播以实现算子融合和内存优化。