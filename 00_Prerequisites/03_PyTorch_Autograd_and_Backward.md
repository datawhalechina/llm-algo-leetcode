# 03. PyTorch Autograd and Backward | PyTorch 自动求导与反向传播

**难度：** Medium | **标签：** `PyTorch`, `Autograd`, `反向传播` | **目标人群：** 所有学习者

## 🎯 学习目标

- 理解自动求导的原理
- 掌握梯度计算和反向传播
- 学会自定义 autograd.Function
- 理解梯度累积和梯度清零

---

## 📚 前置知识

- PyTorch Tensor 基础操作（02 题）
- 微积分基础（导数、链式法则）

---

## 💡 核心概念

### 什么是自动求导？

自动求导（Automatic Differentiation, Autograd）是深度学习框架的核心功能，它能够自动计算函数的梯度，无需手动推导和编写梯度计算代码。

### 计算图（Computational Graph）

PyTorch 使用动态计算图（Dynamic Computational Graph）来追踪操作：
- **前向传播**：构建计算图，记录每个操作
- **反向传播**：沿着计算图反向计算梯度

```
前向传播:  x → f(x) → y
反向传播:  ∂L/∂x ← ∂L/∂y
```

---

## 📖 Part 1: 基础自动求导

### 1.1 简单示例

```python
import torch

# 创建需要梯度的 Tensor
x = torch.tensor([2.0], requires_grad=True)

# 前向传播
y = x ** 2  # y = x^2

# 反向传播
y.backward()

# 查看梯度
print(x.grad)  # tensor([4.])  因为 dy/dx = 2x = 2*2 = 4
```

### 1.2 多步计算

```python
x = torch.tensor([2.0], requires_grad=True)

# 多步计算
y = x ** 2      # y = x^2
z = y * 3       # z = 3y = 3x^2
out = z.mean()  # out = z

# 反向传播
out.backward()

# 梯度: dout/dx = d(3x^2)/dx = 6x = 12
print(x.grad)  # tensor([12.])
```

### 1.3 多变量求导

```python
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

# 计算 z = x^2 + y^2
z = x ** 2 + y ** 2

# 反向传播
z.backward()

# 查看梯度
print(x.grad)  # tensor([4.])  因为 dz/dx = 2x = 4
print(y.grad)  # tensor([6.])  因为 dz/dy = 2y = 6
```

---

## 📖 Part 2: requires_grad 和 grad_fn

### 2.1 requires_grad

```python
# 默认不需要梯度
x = torch.randn(2, 3)
print(x.requires_grad)  # False

# 创建时指定需要梯度
x = torch.randn(2, 3, requires_grad=True)
print(x.requires_grad)  # True

# 后续设置需要梯度
x = torch.randn(2, 3)
x.requires_grad_(True)  # 原地修改
print(x.requires_grad)  # True
```

### 2.2 grad_fn - 记录操作历史

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2

print(x.grad_fn)  # None（叶子节点）
print(y.grad_fn)  # <PowBackward0>（记录了 ** 操作）

z = y * 3
print(z.grad_fn)  # <MulBackward0>（记录了 * 操作）
```

### 2.3 叶子节点（Leaf Tensor）

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2

print(x.is_leaf)  # True（用户创建的 Tensor）
print(y.is_leaf)  # False（由操作产生的 Tensor）

# 只有叶子节点会保留梯度
y.backward()
print(x.grad)  # tensor([4.])
print(y.grad)  # None（非叶子节点的梯度会被释放）
```

---

## 📖 Part 3: 梯度累积和清零

### 3.1 梯度累积

```python
x = torch.tensor([2.0], requires_grad=True)

# 第一次反向传播
y1 = x ** 2
y1.backward()
print(x.grad)  # tensor([4.])

# 第二次反向传播（梯度会累加！）
y2 = x ** 3
y2.backward()
print(x.grad)  # tensor([16.])  = 4 + 12
```

### 3.2 梯度清零

```python
x = torch.tensor([2.0], requires_grad=True)

# 第一次
y1 = x ** 2
y1.backward()
print(x.grad)  # tensor([4.])

# 清零梯度
x.grad.zero_()

# 第二次
y2 = x ** 3
y2.backward()
print(x.grad)  # tensor([12.])（不再累加）
```

### 3.3 训练循环中的梯度管理

```python
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    # 前向传播
    output = model(input)
    loss = criterion(output, target)
    
    # 清零梯度（重要！）
    optimizer.zero_grad()
    
    # 反向传播
    loss.backward()
    
    # 更新参数
    optimizer.step()
```

---

## 📖 Part 4: 控制梯度计算

### 4.1 torch.no_grad() - 禁用梯度计算

```python
x = torch.tensor([2.0], requires_grad=True)

# 正常计算（会追踪梯度）
y = x ** 2
print(y.requires_grad)  # True

# 禁用梯度计算
with torch.no_grad():
    y = x ** 2
    print(y.requires_grad)  # False

# 用于推理阶段，节省内存
model.eval()
with torch.no_grad():
    output = model(input)
```

### 4.2 detach() - 分离计算图

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2

# 分离 y，不再追踪梯度
y_detached = y.detach()
print(y_detached.requires_grad)  # False

# y_detached 的操作不会影响 x 的梯度
z = y_detached * 3
z.backward()  # 报错！因为 z 不需要梯度
```

### 4.3 @torch.no_grad() 装饰器

```python
@torch.no_grad()
def inference(model, input):
    """推理函数，不计算梯度"""
    return model(input)

# 等价于
def inference(model, input):
    with torch.no_grad():
        return model(input)
```

---

## 📖 Part 5: 高级梯度操作

### 5.1 retain_graph - 保留计算图

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2

# 第一次反向传播
y.backward(retain_graph=True)
print(x.grad)  # tensor([4.])

# 第二次反向传播（需要 retain_graph=True）
x.grad.zero_()
y.backward()
print(x.grad)  # tensor([4.])
```

### 5.2 grad_outputs - 指定输出梯度

```python
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x ** 2  # y = [4, 9]

# 指定 dy = [1, 2]
y.backward(torch.tensor([1.0, 2.0]))

# dx = dy * dy/dx = [1, 2] * [4, 6] = [4, 12]
print(x.grad)  # tensor([4., 12.])
```

### 5.3 create_graph - 计算高阶导数

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 3

# 一阶导数
grad_y = torch.autograd.grad(y, x, create_graph=True)[0]
print(grad_y)  # tensor([12.])  因为 dy/dx = 3x^2 = 12

# 二阶导数
grad2_y = torch.autograd.grad(grad_y, x)[0]
print(grad2_y)  # tensor([12.])  因为 d^2y/dx^2 = 6x = 12
```

---

## 📖 Part 6: 自定义 autograd.Function

### 6.1 基本结构

```python
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """
        前向传播
        
        Args:
            ctx: 上下文对象，用于保存信息供反向传播使用
            input: 输入 Tensor
        
        Returns:
            output: 输出 Tensor
        """
        # 保存输入，供反向传播使用
        ctx.save_for_backward(input)
        
        # 计算输出
        output = input.clamp(min=0)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播
        
        Args:
            ctx: 上下文对象
            grad_output: 输出的梯度 (dL/dy)
        
        Returns:
            grad_input: 输入的梯度 (dL/dx)
        """
        # 获取保存的输入
        input, = ctx.saved_tensors
        
        # 计算输入的梯度
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0  # ReLU 的导数
        
        return grad_input

# 使用自定义函数
relu = MyReLU.apply

x = torch.tensor([-1.0, 2.0, -3.0, 4.0], requires_grad=True)
y = relu(x)
print(y)  # tensor([0., 2., 0., 4.])

y.sum().backward()
print(x.grad)  # tensor([0., 1., 0., 1.])
```

### 6.2 实战示例：自定义 Sigmoid

```python
class MySigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = 1 / (1 + torch.exp(-input))
        ctx.save_for_backward(output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        # sigmoid 的导数: σ'(x) = σ(x) * (1 - σ(x))
        grad_input = grad_output * output * (1 - output)
        return grad_input

# 测试
sigmoid = MySigmoid.apply
x = torch.tensor([0.0], requires_grad=True)
y = sigmoid(x)
y.backward()
print(x.grad)  # tensor([0.25])  因为 σ'(0) = 0.5 * 0.5 = 0.25
```

---

## 📖 Part 7: 梯度检查（Gradient Checking）

### 7.1 数值梯度 vs 解析梯度

```python
def numerical_gradient(f, x, eps=1e-5):
    """
    使用数值方法计算梯度
    
    Args:
        f: 函数
        x: 输入
        eps: 扰动大小
    
    Returns:
        数值梯度
    """
    grad = torch.zeros_like(x)
    
    for i in range(x.numel()):
        # f(x + eps)
        x_plus = x.clone()
        x_plus.view(-1)[i] += eps
        f_plus = f(x_plus)
        
        # f(x - eps)
        x_minus = x.clone()
        x_minus.view(-1)[i] -= eps
        f_minus = f(x_minus)
        
        # 数值梯度
        grad.view(-1)[i] = (f_plus - f_minus) / (2 * eps)
    
    return grad

# 测试
def f(x):
    return (x ** 2).sum()

x = torch.tensor([2.0, 3.0], requires_grad=True)

# 解析梯度
y = f(x)
y.backward()
analytical_grad = x.grad.clone()

# 数值梯度
x.grad.zero_()
numerical_grad = numerical_gradient(f, x)

# 对比
print("Analytical:", analytical_grad)  # tensor([4., 6.])
print("Numerical:", numerical_grad)    # tensor([4., 6.])
print("Difference:", (analytical_grad - numerical_grad).abs().max())  # 很小
```

### 7.2 使用 torch.autograd.gradcheck

```python
from torch.autograd import gradcheck

# 定义函数
class MyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input ** 2
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * 2 * input

# 测试
input = torch.randn(3, 4, dtype=torch.double, requires_grad=True)
test = gradcheck(MyFunction.apply, input, eps=1e-6, atol=1e-4)
print("Gradient check:", "Passed" if test else "Failed")
```

---

## 🎯 实战练习

### 练习 1: 实现 Softmax 的自定义 autograd

```python
class MySoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """
        Softmax 前向传播
        
        Args:
            input: 形状为 (batch, num_classes) 的 Tensor
        
        Returns:
            output: Softmax 输出
        """
        # TODO: 实现 Softmax
        # 提示: softmax(x) = exp(x) / sum(exp(x))
        # 注意数值稳定性: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
        pass
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Softmax 反向传播
        
        Args:
            grad_output: 输出的梯度
        
        Returns:
            grad_input: 输入的梯度
        """
        # TODO: 实现 Softmax 的梯度
        # 提示: ∂softmax_i/∂x_j = softmax_i * (δ_ij - softmax_j)
        pass

# 测试
softmax = MySoftmax.apply
x = torch.randn(2, 3, requires_grad=True)
y = softmax(x)
print(y)
print(y.sum(dim=1))  # 应该全为 1
```

### 练习 2: 梯度累积模拟大 Batch

```python
def train_with_gradient_accumulation(model, data_loader, optimizer, accumulation_steps=4):
    """
    使用梯度累积模拟大 batch size
    
    Args:
        model: 模型
        data_loader: 数据加载器
        optimizer: 优化器
        accumulation_steps: 累积步数
    """
    model.train()
    optimizer.zero_grad()
    
    for i, (inputs, targets) in enumerate(data_loader):
        # TODO: 实现梯度累积
        # 提示:
        # 1. 前向传播
        # 2. 计算损失（除以 accumulation_steps）
        # 3. 反向传播
        # 4. 每 accumulation_steps 步更新一次参数
        pass

# 测试
# model = ...
# data_loader = ...
# optimizer = ...
# train_with_gradient_accumulation(model, data_loader, optimizer, accumulation_steps=4)
```

### 练习 3: 实现梯度裁剪

```python
def clip_gradients(model, max_norm=1.0):
    """
    梯度裁剪，防止梯度爆炸
    
    Args:
        model: 模型
        max_norm: 最大梯度范数
    
    Returns:
        total_norm: 裁剪前的总梯度范数
    """
    # TODO: 实现梯度裁剪
    # 提示:
    # 1. 计算所有参数梯度的总范数
    # 2. 如果总范数 > max_norm，按比例缩放所有梯度
    pass

# 测试
# model = ...
# loss.backward()
# total_norm = clip_gradients(model, max_norm=1.0)
# optimizer.step()
```

---

## 📚 参考答案

<details>
<summary>点击查看练习 1 答案</summary>

```python
class MySoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # 数值稳定的 Softmax
        input_max = input.max(dim=1, keepdim=True)[0]
        exp_input = torch.exp(input - input_max)
        output = exp_input / exp_input.sum(dim=1, keepdim=True)
        
        ctx.save_for_backward(output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        
        # Softmax 的 Jacobian 矩阵
        # ∂softmax_i/∂x_j = softmax_i * (δ_ij - softmax_j)
        grad_input = output * grad_output
        sum_grad = (output * grad_output).sum(dim=1, keepdim=True)
        grad_input = grad_input - output * sum_grad
        
        return grad_input
```
</details>

<details>
<summary>点击查看练习 2 答案</summary>

```python
def train_with_gradient_accumulation(model, data_loader, optimizer, accumulation_steps=4):
    model.train()
    optimizer.zero_grad()
    
    for i, (inputs, targets) in enumerate(data_loader):
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 损失除以累积步数
        loss = loss / accumulation_steps
        
        # 反向传播
        loss.backward()
        
        # 每 accumulation_steps 步更新一次
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
```
</details>

<details>
<summary>点击查看练习 3 答案</summary>

```python
def clip_gradients(model, max_norm=1.0):
    # 计算总梯度范数
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    # 裁剪梯度
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
    
    return total_norm

# 或者使用 PyTorch 内置函数
# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
</details>

---

## 🔗 相关资源

- [PyTorch Autograd 官方文档](https://pytorch.org/docs/stable/autograd.html)
- [PyTorch Autograd 教程](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- [自动求导原理](https://pytorch.org/blog/overview-of-pytorch-autograd-engine/)

---

## 🎓 总结

本节学习了 PyTorch 自动求导的核心概念：
- ✅ 自动求导的基本原理和使用
- ✅ 计算图和梯度传播
- ✅ 梯度累积和清零
- ✅ 控制梯度计算（no_grad、detach）
- ✅ 自定义 autograd.Function
- ✅ 梯度检查和调试

**下一步：** 学习 [04. PyTorch nn.Module Basics](./04_PyTorch_nn_Module_Basics.md)，掌握模块定义和参数管理。
