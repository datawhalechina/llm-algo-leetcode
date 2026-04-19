# 10. PyTorch Profiling Basics | PyTorch 性能分析基础

**难度：** Medium | **标签：** `PyTorch`, `Profiling`, `性能优化` | **目标人群：** 所有学习者

## 🎯 学习目标

- 掌握 torch.profiler 的基本使用
- 学会分析 CPU/GPU 时间分布
- 理解性能瓶颈的定位方法
- 学会使用 TensorBoard 可视化性能数据

---

## 📚 前置知识

- PyTorch Tensor 基础（02 题）
- PyTorch nn.Module（04 题）
- 神经网络训练循环（06 题）

---

## 💡 核心概念

### 什么是 Profiling？

Profiling（性能分析）是测量程序执行时间和资源使用的过程，帮助我们：
- 找出性能瓶颈
- 优化代码执行效率
- 理解硬件利用率

### 为什么需要 Profiling？

在深度学习中，性能问题通常来自：
- **访存瓶颈**：频繁的 CPU-GPU 数据传输
- **算子效率**：某些操作特别慢
- **并行度不足**：GPU 利用率低
- **不必要的同步**：CPU 等待 GPU

---

## 📖 Part 1: torch.profiler 基础

### 1.1 简单示例

```python
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

# 定义一个简单的模型
model = nn.Sequential(
    nn.Linear(100, 200),
    nn.ReLU(),
    nn.Linear(200, 10)
)

# 准备输入
input = torch.randn(32, 100)

# 使用 profiler
with profile(activities=[ProfilerActivity.CPU]) as prof:
    output = model(input)

# 打印性能报告
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```

### 1.2 同时分析 CPU 和 GPU

```python
# 如果有 GPU
if torch.cuda.is_available():
    model = model.cuda()
    input = input.cuda()
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
    ) as prof:
        output = model(input)
    
    # 按 CUDA 时间排序
    print(prof.key_averages().table(
        sort_by="cuda_time_total", 
        row_limit=10
    ))
```

---

## 📖 Part 2: 性能报告解读

### 2.1 关键指标

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,  # 记录张量形状
    profile_memory=True,  # 记录内存使用
    with_stack=True  # 记录调用栈
) as prof:
    output = model(input)

# 查看详细报告
print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=10
))
```

**输出示例：**
```
---------------------------------  ------------  ------------  ------------  ------------
                             Name    Self CPU %      Self CPU   CPU total %     CPU total
---------------------------------  ------------  ------------  ------------  ------------
                     aten::linear        10.00%       1.000ms        50.00%       5.000ms
                       aten::relu         5.00%       0.500ms        10.00%       1.000ms
                       aten::addmm        30.00%       3.000ms        30.00%       3.000ms
---------------------------------  ------------  ------------  ------------  ------------
```

**指标说明：**
- **Self CPU time**：操作本身的 CPU 时间（不含子操作）
- **CPU total**：操作及其子操作的总 CPU 时间
- **CUDA time**：GPU 执行时间
- **CPU Mem** / **CUDA Mem**：内存/显存使用

---

## 📖 Part 3: 导出和可视化

### 3.1 导出 Chrome Trace

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
) as prof:
    output = model(input)

# 导出为 Chrome Trace 格式
prof.export_chrome_trace("trace.json")

# 在 Chrome 浏览器中打开 chrome://tracing
# 然后加载 trace.json 文件
```

### 3.2 使用 TensorBoard 可视化

```python
from torch.profiler import profile, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    on_trace_ready=tensorboard_trace_handler('./log/profiler')
) as prof:
    for _ in range(10):
        output = model(input)
        prof.step()  # 记录每一步

# 启动 TensorBoard: tensorboard --logdir=./log/profiler
```

---

## 📖 Part 4: 分析训练循环

### 4.1 完整训练循环的 Profiling

```python
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(100, 200),
    nn.ReLU(),
    nn.Linear(200, 10)
)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# 准备数据
inputs = torch.randn(32, 100)
targets = torch.randint(0, 10, (32,))

if torch.cuda.is_available():
    model = model.cuda()
    inputs = inputs.cuda()
    targets = targets.cuda()

# Profiling 训练循环
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    
    # 更新参数
    optimizer.step()

# 查看报告
print(prof.key_averages().table(
    sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
    row_limit=20
))
```

### 4.2 多步训练的 Profiling

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=1,    # 预热 1 步
        warmup=1,  # 预热 1 步
        active=3,  # 记录 3 步
        repeat=2   # 重复 2 次
    ),
    on_trace_ready=tensorboard_trace_handler('./log/profiler')
) as prof:
    for step in range(10):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        prof.step()  # 通知 profiler 进入下一步
```

---

## 📖 Part 5: 常见性能问题定位

### 5.1 CPU-GPU 数据传输瓶颈

```python
# 不好的做法：频繁的 CPU-GPU 传输
for i in range(100):
    x = torch.randn(100, 100)
    x = x.cuda()  # 每次循环都传输
    y = x @ x.T
    result = y.cpu()  # 每次循环都传回

# 好的做法：减少传输次数
x = torch.randn(100, 100).cuda()  # 只传输一次
for i in range(100):
    y = x @ x.T
result = y.cpu()  # 只传回一次
```

### 5.2 同步操作导致的性能下降

```python
# 不好的做法：频繁同步
for i in range(100):
    x = torch.randn(100, 100, device='cuda')
    y = x @ x.T
    print(y.sum().item())  # .item() 会导致 CPU-GPU 同步

# 好的做法：批量处理
results = []
for i in range(100):
    x = torch.randn(100, 100, device='cuda')
    y = x @ x.T
    results.append(y.sum())

# 最后一次性同步
final_results = [r.item() for r in results]
```

### 5.3 使用 torch.utils.bottleneck 快速定位

```python
# 在命令行运行
# python -m torch.utils.bottleneck your_script.py

# 会自动分析：
# - CPU 时间分布
# - CUDA 时间分布
# - 内存使用
# - 瓶颈操作
```

---

## 📖 Part 6: 实用技巧

### 6.1 只分析部分代码

```python
# 使用上下文管理器
def train_step(model, data, target):
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    return loss

# 只分析训练步骤
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    loss = train_step(model, data, target)

print(prof.key_averages().table())
```

### 6.2 对比不同实现的性能

```python
import time

def benchmark(func, *args, num_runs=100):
    """简单的性能测试函数"""
    # 预热
    for _ in range(10):
        func(*args)
    
    # 同步 GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # 测试
    start = time.time()
    for _ in range(num_runs):
        func(*args)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end = time.time()
    avg_time = (end - start) / num_runs * 1000  # ms
    
    return avg_time

# 对比两种实现
def method1(x):
    return x @ x.T

def method2(x):
    return torch.matmul(x, x.T)

x = torch.randn(1000, 1000, device='cuda')

time1 = benchmark(method1, x)
time2 = benchmark(method2, x)

print(f"Method 1: {time1:.2f} ms")
print(f"Method 2: {time2:.2f} ms")
```

### 6.3 使用 torch.cuda.Event 精确测量 GPU 时间

```python
if torch.cuda.is_available():
    # 创建 CUDA 事件
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # 记录开始
    start_event.record()
    
    # 执行操作
    output = model(input)
    
    # 记录结束
    end_event.record()
    
    # 等待 GPU 完成
    torch.cuda.synchronize()
    
    # 计算时间（毫秒）
    elapsed_time = start_event.elapsed_time(end_event)
    print(f"GPU time: {elapsed_time:.2f} ms")
```

---

## 🎯 实战练习

### 练习 1: 分析模型的性能瓶颈

```python
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# TODO: 使用 profiler 分析这个模型
# 1. 找出最耗时的操作
# 2. 计算 CPU 和 GPU 时间的比例
# 3. 分析内存使用情况
```

### 练习 2: 对比不同 batch size 的性能

```python
# TODO: 对比 batch_size = [1, 8, 32, 128] 的性能
# 1. 测量每个 batch size 的平均时间
# 2. 计算吞吐量（samples/second）
# 3. 分析 GPU 利用率
```

### 练习 3: 优化数据加载

```python
from torch.utils.data import DataLoader, TensorDataset

# TODO: 对比不同 num_workers 的性能
# 1. num_workers = 0, 2, 4, 8
# 2. 测量数据加载时间
# 3. 找出最优的 num_workers 数量
```

---

## 📚 参考答案

<details>
<summary>点击查看练习 1 答案</summary>

```python
model = MyModel()
input = torch.randn(32, 3, 32, 32)

if torch.cuda.is_available():
    model = model.cuda()
    input = input.cuda()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    output = model(input)

# 按 CUDA 时间排序
print(prof.key_averages().table(
    sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
    row_limit=10
))

# 分析结果
print("\n=== 性能分析 ===")
total_cpu_time = sum([item.cpu_time_total for item in prof.key_averages()])
total_cuda_time = sum([item.cuda_time_total for item in prof.key_averages()])

print(f"Total CPU time: {total_cpu_time / 1000:.2f} ms")
print(f"Total CUDA time: {total_cuda_time / 1000:.2f} ms")
print(f"CPU/GPU ratio: {total_cpu_time / (total_cuda_time + 1e-6):.2f}")
```
</details>

<details>
<summary>点击查看练习 2 答案</summary>

```python
import time

model = MyModel()
if torch.cuda.is_available():
    model = model.cuda()

batch_sizes = [1, 8, 32, 128]
results = {}

for bs in batch_sizes:
    input = torch.randn(bs, 3, 32, 32)
    if torch.cuda.is_available():
        input = input.cuda()
    
    # 预热
    for _ in range(10):
        _ = model(input)
    
    # 测试
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.time()
    num_runs = 100
    for _ in range(num_runs):
        _ = model(input)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    avg_time = elapsed / num_runs * 1000  # ms
    throughput = bs * num_runs / elapsed  # samples/sec
    
    results[bs] = {
        'time': avg_time,
        'throughput': throughput
    }
    
    print(f"Batch size {bs:3d}: {avg_time:6.2f} ms, {throughput:8.1f} samples/sec")
```
</details>

---

## 🔗 相关资源

- [PyTorch Profiler 官方文档](https://pytorch.org/docs/stable/profiler.html)
- [PyTorch Profiler 教程](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [TensorBoard Profiler 插件](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html)

---

## 🎓 总结

本节学习了 PyTorch 性能分析的核心技能：
- ✅ torch.profiler 的基本使用
- ✅ 性能报告的解读
- ✅ Chrome Trace 和 TensorBoard 可视化
- ✅ 常见性能问题的定位
- ✅ 实用的性能测试技巧

**关键要点：**
- Profiling 是性能优化的第一步
- 关注 CPU/GPU 时间比例
- 避免频繁的 CPU-GPU 同步
- 使用可视化工具辅助分析

**下一步：** 学习 [11. Memory Profiling and Optimization](./11_Memory_Profiling_and_Optimization.md)，掌握显存优化技巧。
