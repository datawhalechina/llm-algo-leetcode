# 00 PyTorch Warmup

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/00_PyTorch_Warmup.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 00. PyTorch 核心基础热身: 张量、前反向传播与 Embedding (Warmup)

**难度：** Easy | **标签：** `PyTorch`, `Foundation` | **目标人群：** 通用基础 (算法/Infra)

在深入大模型的浩瀚海洋（如 Attention、LoRA、MoE）之前，我们必须确保自己的“底层积木”是非常扎实的。
本节作为**热身关卡**，将用三个非常经典的实战填空，带你快速找回 PyTorch 的核心肌肉记忆：张量维度变换 (Tensor Reshaping)、嵌入层查表 (Embedding Lookup) 以及链式法则的反向传播 (Backpropagation)。


### Part 1: 张量维度变换与 `einops`

>在大模型代码中，最常报的错就是 `RuntimeError: size mismatch`。
>熟练掌握 `view`, `reshape`, `transpose`, `permute` 是算法工程师的基本功。
>更进一步，工业界广泛使用 **`einops`** 库来让维度变换语义化、可读化。


```python
import torch
import einops

def tensor_warmup(x: torch.Tensor):
    """
    假设 x 是一批图像的特征 (例如在多模态大模型中)，形状为 [batch_size, channels, height, width]
    我们需要将其展平为序列 (Sequence)，以输入给 Transformer。
    目标形状: [batch_size, height * width, channels]
    """
    
    # ==========================================
    # TODO 1.1: 使用原生的 PyTorch 方法 (permute + reshape/flatten) 完成变换
    # 提示: 先把 channels 换到最后，然后再把 h 和 w 合并
    # ==========================================
    # x_native = ???
    
    # ==========================================
    # TODO 1.2: 使用 einops.rearrange 优雅地完成完全相同的操作
    # 提示: 语法为 'b c h w -> b (h w) c'
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
import torch.nn as nn

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
    # 提示: emb_layer.weight 里面存了所有的向量，形状为 [vocab_size, hidden_dim]
    # 我们直接用 input_ids 作为索引去“切片”这个 weight 矩阵。
    # ==========================================
    # out_manual = ???
    
    # return out_official, out_manual
    pass

```

### Part 3: 前向传播与反向传播 (Forward & Backward)

>大模型的训练机制完全建立在**反向传播算法 (Backpropagation)** 与 **链式法则 (Chain Rule)** 之上。
>在日常使用中，我们只写前向传播，然后调用 `loss.backward()`，PyTorch 的 Autograd 会自动帮我们算梯度。
>为了真正理解底层，我们来实现一个包含 Linear 和 ReLU 的非常简易的自定义算子的前向和反向逻辑。


```python
class LinearReLUFunction(torch.autograd.Function):
    """
    实现一个包含 Linear + ReLU 的算子，并推导其反向传播的梯度。
    公式: y = relu(x @ W^T + b)
    """
    
    @staticmethod
    def forward(ctx, x, weight, bias):
        # 1. 前向计算
        z = F.linear(x, weight, bias)
        y = F.relu(z)
        
        # 2. 保存上下文 ctx 给反向传播使用
        # 在算梯度时，我们需要 x, weight, 以及区分 ReLU 哪里大于0 的标记 (mask)
        mask = (z > 0).float()
        ctx.save_for_backward(x, weight, mask)
        
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """
        接收从上一层传回来的梯度 (grad_output)，形状同 y。
        返回对当前层三个输入 (x, weight, bias) 的梯度。
        """
        x, weight, mask = ctx.saved_tensors
        
        # ==========================================
        # TODO 3.1: 反传过 ReLU
        # 公式: grad_z = grad_output * (ReLU的导数)
        # ==========================================
        # grad_z = ???
        
        # ==========================================
        # TODO 3.2: 反传过 Linear
        # 1. 对 x 的梯度: grad_z @ weight
        # 2. 对 weight 的梯度: grad_z^T @ x
        # 3. 对 bias 的梯度: grad_z 沿着 batch 维度(dim=0)求和
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
        # --- Test 1 ---
        x_img = torch.randn(2, 3, 224, 224)
        n, e = tensor_warmup(x_img)
        assert n.shape == (2, 224*224, 3)
        assert torch.allclose(n, e), "PyTorch Native 与 Einops 结果不一致！"
        print("✅ Part 1: Tensor 维度变换热身通过！")
        
        # --- Test 2 ---
        ids = torch.randint(0, 1000, (4, 16))
        off, man = embedding_warmup(ids, vocab_size=1000, hidden_dim=64)
        assert off.shape == (4, 16, 64)
        assert torch.allclose(off, man), "手动 Embedding 查表与官方实现不一致！"
        print("✅ Part 2: Embedding 查表模拟通过！")
        
        # --- Test 3 ---
        x = torch.randn(2, 4, requires_grad=True)
        w = torch.randn(3, 4, requires_grad=True)
        b = torch.randn(3, requires_grad=True)
        
        # 使用官方算子跑一边
        y_std = F.relu(F.linear(x, w, b))
        y_std.sum().backward()
        std_gx, std_gw, std_gb = x.grad.clone(), w.grad.clone(), b.grad.clone()
        
        # 清零梯度
        x.grad.zero_()
        w.grad.zero_()
        b.grad.zero_()
        
        # 使用你实现的算子跑一遍
        y_custom = LinearReLUFunction.apply(x, w, b)
        y_custom.sum().backward()
        
        assert torch.allclose(x.grad, std_gx) and torch.allclose(w.grad, std_gw) and torch.allclose(b.grad, std_gb), "反向传播推导梯度与 Autograd 结果不一致！"
        print("✅ Part 3: 前反向传播推导算子通过！")
        print("\n🔥 恭喜你，PyTorch 核心积木热身完毕，可以正式开启大模型的浩瀚旅程了！")
        
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
    except TypeError as e:
        print("代码可能未完成，导致变量解包或操作错误。")
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
    except Exception as e:
        print(f"❌ 发生未知异常: {e}")

test_warmup()

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

::: details 💡 点击查看官方解析与参考代码

本节通过三个实战任务带你快速复习 PyTorch 核心操作。在张量维度变换中，原生的 permute+reshape 和 einops.rearrange 都能完成通道转换与展平；在 Embedding 层模拟中，深入体会了 nn.Embedding 本质就是权重的查表；在反向传播推导中，我们利用链式法则计算了经过 ReLU 和 Linear 层的梯度，这是理解大模型底层 Autograd 机制的关键。

```python
def tensor_warmup(x: torch.Tensor):
    x_native = x.permute(0, 2, 3, 1).reshape(x.shape[0], x.shape[2] * x.shape[3], x.shape[1])
    x_einops = einops.rearrange(x, "b c h w -> b (h w) c")
    return x_native, x_einops

def embedding_warmup(input_ids: torch.Tensor, vocab_size: int, hidden_dim: int):
    emb_layer = nn.Embedding(vocab_size, hidden_dim)
    emb_layer.weight.data.normal_(0, 0.1)
    out_official = emb_layer(input_ids)
    out_manual = emb_layer.weight[input_ids]
    return out_official, out_manual

class LinearReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        z = F.linear(x, weight, bias)
        y = F.relu(z)
        mask = (z > 0).float()
        ctx.save_for_backward(x, weight, mask)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, mask = ctx.saved_tensors
        grad_z = grad_output * mask
        grad_x = grad_z @ weight
        grad_weight = grad_z.T @ x
        grad_bias = grad_z.sum(dim=0)
        return grad_x, grad_weight, grad_bias
```

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
