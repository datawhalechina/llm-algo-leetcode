# 11. Memory Profiling and Optimization | 显存分析与优化

**难度：** Medium | **标签：** `PyTorch`, `显存优化`, `性能优化` | **目标人群：** 所有学习者

## 🎯 学习目标

- 掌握显存分析工具的使用
- 学会优化显存使用
- 理解梯度累积的原理
- 掌握混合精度训练

---

## 📚 前置知识

- PyTorch Tensor 基础（02 题）
- PyTorch Autograd（03 题）
- 神经网络训练循环（06 题）
- PyTorch Profiling（10 题）

---

## 💡 核心概念

### 什么是显存（VRAM）？

显存（Video RAM）是 GPU 上的内存，用于存储：
- **模型参数**：权重和偏置
- **梯度**：反向传播计算的梯度
- **优化器状态**：Adam 的动量和方差
- **激活值**：前向传播的中间结果
- **临时缓冲区**：计算过程中的临时数据

### 显存不足的常见症状

```python
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

---

## 📖 Part 1: 显存监控

### 1.1 基础显存查询

```python
import torch

if torch.cuda.is_available():
    # 当前已分配的显存（字节）
    allocated = torch.cuda.memory_allocated()
    print(f"Allocated: {allocated / 1024**2:.2f} MB")
    
    # 当前已预留的显存（字节）
    reserved = torch.cuda.memory_reserved()
    print(f"Reserved: {reserved / 1024**2:.2f} MB")
    
    # 峰值显存
    max_allocated = torch.cuda.max_memory_allocated()
    print(f"Max Allocated: {max_allocated / 1024**2:.2f} MB")
    
    # 显存摘要
    print(torch.cuda.memory_summary())
```

### 1.2 重置显存统计

```python
# 重置峰值统计
torch.cuda.reset_peak_memory_stats()

# 清空显存缓存（释放未使用的显存）
torch.cuda.empty_cache()
```

### 1.3 实时监控显存

```python
def print_memory_usage(prefix=""):
    """打印当前显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"{prefix} Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

# 使用示例
print_memory_usage("Before model creation:")
model = nn.Sequential(nn.Linear(1000, 1000) for _ in range(10))
model = model.cuda()
print_memory_usage("After model creation:")

x = torch.randn(128, 1000, device='cuda')
print_memory_usage("After input creation:")

y = model(x)
print_memory_usage("After forward pass:")

y.sum().backward()
print_memory_usage("After backward pass:")
```

---

## 📖 Part 2: 显存分析

### 2.1 使用 Profiler 分析显存

```python
from torch.profiler import profile, ProfilerActivity

model = nn.Sequential(
    nn.Linear(1000, 2000),
    nn.ReLU(),
    nn.Linear(2000, 1000)
).cuda()

input = torch.randn(128, 1000, device='cuda')

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True
) as prof:
    output = model(input)
    output.sum().backward()

# 按显存使用排序
print(prof.key_averages().table(
    sort_by="self_cuda_memory_usage",
    row_limit=10
))
```

### 2.2 分析显存峰值

```python
def measure_peak_memory(func, *args, **kwargs):
    """测量函数执行的峰值显存"""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # 记录初始显存
    initial = torch.cuda.memory_allocated()
    
    # 执行函数
    result = func(*args, **kwargs)
    
    # 记录峰值显存
    peak = torch.cuda.max_memory_allocated()
    
    print(f"Initial: {initial / 1024**2:.2f} MB")
    print(f"Peak: {peak / 1024**2:.2f} MB")
    print(f"Increase: {(peak - initial) / 1024**2:.2f} MB")
    
    return result

# 使用示例
def train_step(model, input, target):
    output = model(input)
    loss = F.cross_entropy(output, target)
    loss.backward()
    return loss

model = MyModel().cuda()
input = torch.randn(32, 3, 224, 224, device='cuda')
target = torch.randint(0, 10, (32,), device='cuda')

measure_peak_memory(train_step, model, input, target)
```

---

## 📖 Part 3: 显存优化技巧

### 3.1 减小 Batch Size

```python
# 显存不足时最简单的方法
# 从 batch_size=128 降到 64 或 32

# 原始
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

# 优化后
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### 3.2 梯度累积（Gradient Accumulation）

模拟大 batch size，同时节省显存：

```python
def train_with_gradient_accumulation(
    model, train_loader, optimizer, criterion, 
    accumulation_steps=4, device='cuda'
):
    """
    使用梯度累积训练
    
    Args:
        accumulation_steps: 累积多少步后更新一次参数
    """
    model.train()
    optimizer.zero_grad()
    
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 损失除以累积步数
        loss = loss / accumulation_steps
        
        # 反向传播（梯度累积）
        loss.backward()
        
        # 每 accumulation_steps 步更新一次参数
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

# 效果：batch_size=32, accumulation_steps=4
# 等价于 batch_size=128，但显存只需要 batch_size=32 的量
```

### 3.3 混合精度训练（Mixed Precision）

使用 FP16 代替 FP32，显存减半：

```python
from torch.cuda.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scaler = GradScaler()

for inputs, targets in train_loader:
    inputs, targets = inputs.cuda(), targets.cuda()
    
    optimizer.zero_grad()
    
    # 使用混合精度
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    # 缩放损失并反向传播
    scaler.scale(loss).backward()
    
    # 更新参数
    scaler.step(optimizer)
    scaler.update()

# 显存节省：约 40-50%
# 速度提升：约 2-3x（在支持 Tensor Core 的 GPU 上）
```

### 3.4 梯度检查点（Gradient Checkpointing）

用计算换显存：

```python
from torch.utils.checkpoint import checkpoint

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(1000, 1000) for _ in range(10)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            # 使用 checkpoint 包裹每一层
            x = checkpoint(layer, x, use_reentrant=False)
        return x

# 显存节省：50-80%（取决于模型深度）
# 时间增加：20-30%（需要重新计算前向传播）
```

### 3.5 及时释放不需要的 Tensor

```python
# 不好的做法：保留所有中间结果
results = []
for i in range(100):
    x = torch.randn(1000, 1000, device='cuda')
    y = x @ x.T
    results.append(y)  # 显存持续增长

# 好的做法：只保留需要的结果
results = []
for i in range(100):
    x = torch.randn(1000, 1000, device='cuda')
    y = x @ x.T
    results.append(y.sum().item())  # 只保留标量
    del x, y  # 显式释放（可选，Python 会自动回收）

# 更好的做法：使用 torch.no_grad()
with torch.no_grad():
    for i in range(100):
        x = torch.randn(1000, 1000, device='cuda')
        y = x @ x.T
        results.append(y.sum().item())
```

### 3.6 使用 inplace 操作

```python
# 不好的做法：创建新 Tensor
x = torch.randn(1000, 1000, device='cuda')
x = x + 1  # 创建新 Tensor

# 好的做法：原地操作
x = torch.randn(1000, 1000, device='cuda')
x.add_(1)  # 原地操作，不创建新 Tensor

# 常见的 inplace 操作
x.relu_()      # 原地 ReLU
x.clamp_(0, 1) # 原地裁剪
x.mul_(2)      # 原地乘法
```

---

## 📖 Part 4: 显存泄漏排查

### 4.1 常见的显存泄漏原因

```python
# 原因 1：保留了计算图
losses = []
for i in range(100):
    output = model(input)
    loss = criterion(output, target)
    losses.append(loss)  # ❌ 保留了计算图
    loss.backward()

# 解决方法：只保留标量值
losses = []
for i in range(100):
    output = model(input)
    loss = criterion(output, target)
    losses.append(loss.item())  # ✅ 只保留标量
    loss.backward()
```

```python
# 原因 2：没有清零梯度
for i in range(100):
    output = model(input)
    loss = criterion(output, target)
    loss.backward()  # ❌ 梯度累积
    optimizer.step()

# 解决方法：每次清零梯度
for i in range(100):
    optimizer.zero_grad()  # ✅ 清零梯度
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

```python
# 原因 3：循环引用
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cache = []  # ❌ 可能导致循环引用
    
    def forward(self, x):
        result = self.linear(x)
        self.cache.append(result)  # ❌ 保留了 Tensor
        return result

# 解决方法：使用 detach() 或不保留
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cache = []
    
    def forward(self, x):
        result = self.linear(x)
        self.cache.append(result.detach())  # ✅ 分离计算图
        return result
```

### 4.2 使用 gc 模块排查

```python
import gc

# 强制垃圾回收
gc.collect()
torch.cuda.empty_cache()

# 查找未释放的 Tensor
for obj in gc.get_objects():
    if torch.is_tensor(obj):
        print(type(obj), obj.size())
```

---

## 📖 Part 5: 显存优化对比

### 5.1 对比不同优化策略

```python
import time

def benchmark_memory_and_time(func, *args, num_runs=10):
    """测量显存和时间"""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # 预热
    for _ in range(3):
        func(*args)
    
    # 测试
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(num_runs):
        func(*args)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2
    avg_time = elapsed / num_runs * 1000
    
    return peak_memory, avg_time

# 测试不同策略
strategies = {
    "Baseline": lambda: train_baseline(model, input, target),
    "Gradient Accumulation": lambda: train_with_grad_accum(model, input, target),
    "Mixed Precision": lambda: train_with_amp(model, input, target),
    "Gradient Checkpointing": lambda: train_with_checkpoint(model, input, target),
}

print(f"{'Strategy':<25} {'Memory (MB)':<15} {'Time (ms)':<15}")
print("-" * 55)

for name, func in strategies.items():
    memory, time_ms = benchmark_memory_and_time(func)
    print(f"{name:<25} {memory:<15.2f} {time_ms:<15.2f}")
```

---

## 🎯 实战练习

### 练习 1: 分析模型的显存占用

```python
import torch
import torch.nn as nn

class LargeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(2048, 2048) for _ in range(20)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x

# TODO: 分析这个模型的显存占用
# 1. 计算模型参数占用的显存
# 2. 测量前向传播的峰值显存
# 3. 测量反向传播的峰值显存
# 4. 分析激活值占用的显存
```

### 练习 2: 实现显存优化

```python
# TODO: 对上面的模型进行优化
# 1. 使用梯度累积减少 batch size
# 2. 使用混合精度训练
# 3. 使用梯度检查点
# 4. 对比优化前后的显存和速度
```

### 练习 3: 排查显存泄漏

```python
# TODO: 找出并修复以下代码的显存泄漏问题
def train_with_leak(model, train_loader, optimizer, criterion):
    model.train()
    losses = []
    
    for epoch in range(10):
        for inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss)  # 问题在这里
            
            loss.backward()
            optimizer.step()
    
    return losses
```

---

## 📚 参考答案

<details>
<summary>点击查看练习 1 答案</summary>

```python
model = LargeModel().cuda()

# 1. 计算模型参数显存
param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"Model parameters: {param_memory / 1024**2:.2f} MB")

# 2. 前向传播峰值显存
torch.cuda.reset_peak_memory_stats()
input = torch.randn(32, 2048, device='cuda')
output = model(input)
forward_peak = torch.cuda.max_memory_allocated() / 1024**2
print(f"Forward peak: {forward_peak:.2f} MB")

# 3. 反向传播峰值显存
torch.cuda.reset_peak_memory_stats()
input = torch.randn(32, 2048, device='cuda', requires_grad=True)
output = model(input)
output.sum().backward()
backward_peak = torch.cuda.max_memory_allocated() / 1024**2
print(f"Backward peak: {backward_peak:.2f} MB")

# 4. 激活值显存（近似）
activation_memory = backward_peak - param_memory
print(f"Activation memory (approx): {activation_memory:.2f} MB")
```
</details>

<details>
<summary>点击查看练习 2 答案</summary>

```python
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint

# 优化 1: 梯度累积
def train_with_grad_accum(model, data_loader, optimizer, criterion, accum_steps=4):
    optimizer.zero_grad()
    for i, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, targets) / accum_steps
        loss.backward()
        
        if (i + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

# 优化 2: 混合精度
def train_with_amp(model, data_loader, optimizer, criterion):
    scaler = GradScaler()
    for inputs, targets in data_loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

# 优化 3: 梯度检查点
class OptimizedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(2048, 2048) for _ in range(20)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = checkpoint(lambda x: torch.relu(layer(x)), x, use_reentrant=False)
        return x
```
</details>

<details>
<summary>点击查看练习 3 答案</summary>

```python
def train_without_leak(model, train_loader, optimizer, criterion):
    model.train()
    losses = []
    
    for epoch in range(10):
        for inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            
            optimizer.zero_grad()  # 添加：清零梯度
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())  # 修复：只保留标量
            
            loss.backward()
            optimizer.step()
    
    return losses
```
</details>

---

## 🔗 相关资源

- [PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
- [Gradient Checkpointing](https://pytorch.org/docs/stable/checkpoint.html)

---

## 🎓 总结

本节学习了显存分析和优化的核心技能：
- ✅ 显存监控和分析工具
- ✅ 梯度累积技术
- ✅ 混合精度训练
- ✅ 梯度检查点
- ✅ 显存泄漏排查

**关键要点：**
- 显存优化是训练大模型的关键
- 梯度累积可以模拟大 batch size
- 混合精度训练可以节省 40-50% 显存
- 梯度检查点用计算换显存
- 及时释放不需要的 Tensor

**优化策略对比：**
| 策略 | 显存节省 | 速度影响 | 适用场景 |
|------|---------|---------|---------|
| 减小 Batch Size | 线性 | 可能变慢 | 简单快速 |
| 梯度累积 | 线性 | 几乎无影响 | 模拟大 Batch |
| 混合精度 | 40-50% | 加速 2-3x | 现代 GPU |
| 梯度检查点 | 50-80% | 慢 20-30% | 深层模型 |

**下一步：** 完成 Chapter 0 的学习后，可以进入 [Chapter 1: 硬件、数学与系统](../01_Hardware_Math_and_Systems/intro.md)。
