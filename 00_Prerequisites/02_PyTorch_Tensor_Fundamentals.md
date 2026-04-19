# 02. PyTorch Tensor Fundamentals | PyTorch 张量基础操作

**难度：** Easy | **标签：** `PyTorch`, `Tensor`, `基础操作` | **目标人群：** 所有学习者

## 🎯 学习目标

- 掌握 Tensor 的创建、操作、设备转移
- 理解 view、reshape、permute 的区别
- 学会 Tensor 的索引和切片
- 理解内存连续性的概念

---

## 📚 前置知识

- Python 基础语法
- NumPy 数组操作（推荐先完成 01 题）

---

## 💡 核心概念

### 什么是 Tensor？

Tensor（张量）是 PyTorch 中的核心数据结构，类似于 NumPy 的 ndarray，但具有以下特点：
- 可以在 GPU 上运行
- 支持自动求导
- 针对深度学习优化

### Tensor vs NumPy Array

| 特性 | NumPy Array | PyTorch Tensor |
|------|-------------|----------------|
| 设备 | 仅 CPU | CPU 或 GPU |
| 自动求导 | ❌ | ✅ |
| 深度学习优化 | ❌ | ✅ |
| 互操作性 | - | 可与 NumPy 互转 |

---

## 📖 Part 1: Tensor 创建

### 1.1 从数据创建

```python
import torch

# 从 Python 列表创建
x = torch.tensor([1, 2, 3, 4, 5])
print(x)  # tensor([1, 2, 3, 4, 5])

# 从 NumPy 数组创建
import numpy as np
arr = np.array([1, 2, 3])
x = torch.from_numpy(arr)
print(x)  # tensor([1, 2, 3])

# 多维 Tensor
x = torch.tensor([[1, 2], [3, 4]])
print(x.shape)  # torch.Size([2, 2])
```

### 1.2 特殊 Tensor 创建

```python
# 全零 Tensor
zeros = torch.zeros(3, 4)  # 3x4 的全零矩阵

# 全一 Tensor
ones = torch.ones(2, 3)  # 2x3 的全一矩阵

# 随机 Tensor
rand = torch.rand(2, 3)  # 均匀分布 [0, 1)
randn = torch.randn(2, 3)  # 标准正态分布

# 单位矩阵
eye = torch.eye(3)  # 3x3 单位矩阵

# 等差数列
arange = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace = torch.linspace(0, 1, 5)  # [0.0, 0.25, 0.5, 0.75, 1.0]
```

### 1.3 根据已有 Tensor 创建

```python
x = torch.randn(2, 3)

# 创建相同形状的 Tensor
zeros_like = torch.zeros_like(x)
ones_like = torch.ones_like(x)
rand_like = torch.rand_like(x)
```

---

## 📖 Part 2: Tensor 属性

```python
x = torch.randn(2, 3, 4)

# 形状
print(x.shape)  # torch.Size([2, 3, 4])
print(x.size())  # torch.Size([2, 3, 4])

# 维度
print(x.ndim)  # 3

# 元素总数
print(x.numel())  # 24 (2*3*4)

# 数据类型
print(x.dtype)  # torch.float32

# 设备
print(x.device)  # cpu

# 是否需要梯度
print(x.requires_grad)  # False
```

---

## 📖 Part 3: 形状变换

### 3.1 view() - 快速但要求内存连续

```python
x = torch.randn(2, 3, 4)

# 展平为一维
x_flat = x.view(-1)  # torch.Size([24])

# 变换为其他形状
x_reshaped = x.view(6, 4)  # torch.Size([6, 4])
x_reshaped = x.view(2, -1)  # torch.Size([2, 12])，-1 自动推断

# ⚠️ view() 要求内存连续，否则会报错
x_t = x.transpose(0, 1)  # 转置后内存不连续
# x_t.view(-1)  # 会报错！
```

### 3.2 reshape() - 更灵活，自动处理内存不连续

```python
x = torch.randn(2, 3, 4)

# reshape 与 view 类似，但更智能
x_reshaped = x.reshape(6, 4)  # torch.Size([6, 4])

# 即使内存不连续也能工作
x_t = x.transpose(0, 1)
x_reshaped = x_t.reshape(-1)  # ✅ 可以工作
```

### 3.3 permute() - 改变维度顺序

```python
x = torch.randn(2, 3, 4)  # (batch, seq, hidden)

# 交换维度顺序
x_permuted = x.permute(1, 0, 2)  # (seq, batch, hidden)
print(x_permuted.shape)  # torch.Size([3, 2, 4])

# 常用于 Transformer 中的维度变换
# (batch, seq, num_heads, head_dim) -> (batch, num_heads, seq, head_dim)
x = torch.randn(2, 10, 8, 64)
x_permuted = x.permute(0, 2, 1, 3)
print(x_permuted.shape)  # torch.Size([2, 8, 10, 64])
```

### 3.4 transpose() - 交换两个维度

```python
x = torch.randn(2, 3, 4)

# 交换维度 0 和 1
x_t = x.transpose(0, 1)
print(x_t.shape)  # torch.Size([3, 2, 4])

# 等价于 permute(1, 0, 2)
x_t2 = x.permute(1, 0, 2)
assert torch.equal(x_t, x_t2)
```

### 3.5 squeeze() 和 unsqueeze() - 增减维度

```python
x = torch.randn(2, 1, 3, 1, 4)

# 移除所有大小为 1 的维度
x_squeezed = x.squeeze()
print(x_squeezed.shape)  # torch.Size([2, 3, 4])

# 移除指定维度（必须大小为 1）
x_squeezed = x.squeeze(1)
print(x_squeezed.shape)  # torch.Size([2, 3, 1, 4])

# 在指定位置增加维度
x = torch.randn(2, 3)
x_unsqueezed = x.unsqueeze(0)  # 在第 0 维增加
print(x_unsqueezed.shape)  # torch.Size([1, 2, 3])

x_unsqueezed = x.unsqueeze(1)  # 在第 1 维增加
print(x_unsqueezed.shape)  # torch.Size([2, 1, 3])
```

---

## 📖 Part 4: 索引和切片

```python
x = torch.randn(3, 4, 5)

# 基础索引
print(x[0])  # 第一个元素（4x5 的 Tensor）
print(x[0, 1])  # 第一个元素的第二行（长度为 5 的 Tensor）
print(x[0, 1, 2])  # 单个标量

# 切片
print(x[:, 0, :])  # 所有 batch 的第一行
print(x[0, :2, :])  # 第一个 batch 的前两行
print(x[..., -1])  # 所有元素的最后一列（... 表示所有维度）

# 高级索引
indices = torch.tensor([0, 2])
print(x[indices])  # 选择第 0 和第 2 个元素

# 布尔索引
mask = x > 0
print(x[mask])  # 选择所有大于 0 的元素（返回一维 Tensor）

# masked_fill - 根据 mask 填充值
x_masked = x.masked_fill(mask, 0)  # 将所有大于 0 的元素设为 0
```

---

## 📖 Part 5: 设备转移（CPU ↔ GPU）

```python
# 创建 CPU Tensor
x = torch.randn(2, 3)
print(x.device)  # cpu

# 转移到 GPU（如果可用）
if torch.cuda.is_available():
    x_gpu = x.to('cuda')  # 或 x.cuda()
    print(x_gpu.device)  # cuda:0
    
    # 转回 CPU
    x_cpu = x_gpu.to('cpu')  # 或 x_gpu.cpu()
    print(x_cpu.device)  # cpu
    
    # 指定 GPU 设备
    x_gpu1 = x.to('cuda:1')  # 转到第二块 GPU

# 通用写法（自动检测）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)
```

---

## 📖 Part 6: 数据类型转换

```python
x = torch.randn(2, 3)
print(x.dtype)  # torch.float32

# 转换数据类型
x_int = x.to(torch.int32)  # 或 x.int()
x_long = x.to(torch.int64)  # 或 x.long()
x_float = x.to(torch.float32)  # 或 x.float()
x_double = x.to(torch.float64)  # 或 x.double()
x_half = x.to(torch.float16)  # 或 x.half()

# 常用数据类型
# torch.float32 (float) - 默认，4 字节
# torch.float16 (half) - 半精度，2 字节
# torch.int64 (long) - 长整型，8 字节
# torch.int32 (int) - 整型，4 字节
# torch.bool - 布尔型
```

---

## 📖 Part 7: 内存连续性

```python
x = torch.randn(2, 3, 4)

# 检查是否内存连续
print(x.is_contiguous())  # True

# 转置后内存不连续
x_t = x.transpose(0, 1)
print(x_t.is_contiguous())  # False

# 使内存连续
x_t_contiguous = x_t.contiguous()
print(x_t_contiguous.is_contiguous())  # True

# 为什么需要连续内存？
# - view() 要求内存连续
# - 某些操作在连续内存上更快
# - 与 C/C++ 代码交互时需要连续内存
```

---

## 📖 Part 8: Tensor 与 NumPy 互转

```python
import numpy as np

# Tensor -> NumPy
x = torch.randn(2, 3)
x_np = x.numpy()  # 共享内存，修改一个会影响另一个
print(type(x_np))  # <class 'numpy.ndarray'>

# NumPy -> Tensor
arr = np.array([1, 2, 3])
x = torch.from_numpy(arr)  # 共享内存
print(type(x))  # <class 'torch.Tensor'>

# ⚠️ 注意：共享内存意味着修改会相互影响
x = torch.randn(2, 3)
x_np = x.numpy()
x_np[0, 0] = 999
print(x[0, 0])  # 999.0（也被修改了）

# 如果不想共享内存，使用 clone()
x_np = x.clone().numpy()  # 不共享内存
```

---

## 🎯 实战练习

### 练习 1: 创建 Causal Mask

在 Transformer 的自回归生成中，需要创建一个上三角 mask，防止当前位置看到未来的信息。

```python
def create_causal_mask(seq_len):
    """
    创建 Causal Mask
    
    Args:
        seq_len: 序列长度
    
    Returns:
        mask: 形状为 (seq_len, seq_len) 的布尔 Tensor
              mask[i, j] = True 表示位置 i 可以看到位置 j
    
    示例:
        seq_len = 4
        返回:
        [[True, False, False, False],
         [True,  True, False, False],
         [True,  True,  True, False],
         [True,  True,  True,  True]]
    """
    # TODO: 实现 Causal Mask
    # 提示: 使用 torch.triu() 创建上三角矩阵
    pass

# 测试
mask = create_causal_mask(4)
print(mask)
```

### 练习 2: 多头注意力的维度变换

在多头注意力中，需要将 `(batch, seq, hidden)` 变换为 `(batch, num_heads, seq, head_dim)`。

```python
def split_heads(x, num_heads):
    """
    将隐藏维度拆分为多个头
    
    Args:
        x: 形状为 (batch, seq, hidden) 的 Tensor
        num_heads: 头的数量
    
    Returns:
        形状为 (batch, num_heads, seq, head_dim) 的 Tensor
        其中 head_dim = hidden // num_heads
    
    示例:
        x.shape = (2, 10, 512), num_heads = 8
        返回 shape = (2, 8, 10, 64)
    """
    batch, seq, hidden = x.shape
    head_dim = hidden // num_heads
    
    # TODO: 实现维度变换
    # 提示: 先 reshape 再 permute
    pass

# 测试
x = torch.randn(2, 10, 512)
x_split = split_heads(x, num_heads=8)
print(x_split.shape)  # 应该是 torch.Size([2, 8, 10, 64])
```

### 练习 3: 批量矩阵乘法

实现批量矩阵乘法，用于计算 Attention 的 QK^T。

```python
def batch_matmul(a, b):
    """
    批量矩阵乘法
    
    Args:
        a: 形状为 (batch, n, m) 的 Tensor
        b: 形状为 (batch, m, p) 的 Tensor
    
    Returns:
        形状为 (batch, n, p) 的 Tensor
    
    示例:
        a.shape = (2, 3, 4), b.shape = (2, 4, 5)
        返回 shape = (2, 3, 5)
    """
    # TODO: 实现批量矩阵乘法
    # 提示: 使用 torch.bmm() 或 @
    pass

# 测试
a = torch.randn(2, 3, 4)
b = torch.randn(2, 4, 5)
c = batch_matmul(a, b)
print(c.shape)  # 应该是 torch.Size([2, 3, 5])
```

---

## 📚 参考答案

<details>
<summary>点击查看练习 1 答案</summary>

```python
def create_causal_mask(seq_len):
    # 创建上三角矩阵（对角线及以上为 1）
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    # 转换为布尔型，并取反（True 表示可以看到）
    mask = mask == 0
    return mask
```
</details>

<details>
<summary>点击查看练习 2 答案</summary>

```python
def split_heads(x, num_heads):
    batch, seq, hidden = x.shape
    head_dim = hidden // num_heads
    
    # 先 reshape: (batch, seq, hidden) -> (batch, seq, num_heads, head_dim)
    x = x.reshape(batch, seq, num_heads, head_dim)
    
    # 再 permute: (batch, seq, num_heads, head_dim) -> (batch, num_heads, seq, head_dim)
    x = x.permute(0, 2, 1, 3)
    
    return x
```
</details>

<details>
<summary>点击查看练习 3 答案</summary>

```python
def batch_matmul(a, b):
    # 方法 1: 使用 torch.bmm()
    return torch.bmm(a, b)
    
    # 方法 2: 使用 @ 运算符（推荐）
    return a @ b
```
</details>

---

## 🔗 相关资源

- [PyTorch Tensor 官方文档](https://pytorch.org/docs/stable/tensors.html)
- [PyTorch Tensor 操作速查表](https://pytorch.org/tutorials/beginner/ptcheat.html)

---

## 🎓 总结

本节学习了 PyTorch Tensor 的核心操作：
- ✅ Tensor 的创建和属性
- ✅ 形状变换（view、reshape、permute）
- ✅ 索引和切片
- ✅ 设备转移（CPU ↔ GPU）
- ✅ 数据类型转换
- ✅ 内存连续性
- ✅ 与 NumPy 的互操作

**下一步：** 学习 [03. PyTorch Autograd and Backward](./03_PyTorch_Autograd_and_Backward.md)，掌握自动求导的原理。
