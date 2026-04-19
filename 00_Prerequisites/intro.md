# Chapter 0: 前置知识与环境准备 - 完整导学

## 🎯 本章概览

本章包含 14 道题，覆盖从 Python 基础到 PyTorch 深度学习的完整入门知识。通过本章学习，你将建立扎实的编程基础，为后续章节的学习做好充分准备。

### 为什么需要 Chapter 0？

在进入大模型的核心内容之前，掌握基础工具和概念至关重要：

- **Python 基础**：列表推导、字典操作、装饰器等 LLM 代码中的常见语法
- **NumPy 操作**：数组广播、einsum 符号，为理解 Attention 机制做准备
- **PyTorch 基础**：Tensor 操作、自动求导、模块定义，深度学习的核心工具
- **性能分析**：Profiling 工具、显存优化、调试技巧，工程实践的必备技能

**本章的价值：**
- 平滑的学习曲线：避免直接进入 Chapter 1/2 时的陡峭难度
- 实用的工程技能：Profiling、调试、显存优化等生产环境必备能力
- 完整的知识体系：从基础语法到深度学习概念的系统学习

---

## 📚 学习组划分

本章按主题分为 4 个学习组：

| 学习组 | 题目范围 | 主题 | 难度 |
|:---|:---|:---|:---|
| **0A: Python 基础** | 00-01 | Python 语法与 NumPy | Easy |
| **0B: PyTorch 基础** | 02-05 | Tensor、Autograd、模块定义 | Easy-Medium |
| **0C: 深度学习基础** | 06-09 | 训练循环、激活函数、归一化 | Medium |
| **0D: 工具与调试** | 10-13 | Profiling、显存优化、调试技巧 | Medium |

---

## 📚 推荐学习路径

### 路径 1：快速入门
**适合：** 有一定 Python 基础，想快速上手 PyTorch

**学习顺序：**
1. **0B: PyTorch 基础**（02-05 题）→ 掌握 PyTorch 核心操作
2. **0C: 深度学习基础**（06 题）→ 理解训练循环
3. **0D: 工具与调试**（10 题）→ 学习 Profiling 基础

**核心收获：** 能够使用 PyTorch 实现简单的神经网络并进行性能分析

---

### 路径 2：系统学习
**适合：** 零基础或基础薄弱，需要系统学习

**学习顺序：**
1. **0A: Python 基础** → 巩固 Python 语法
2. **0B: PyTorch 基础** → 掌握深度学习框架
3. **0C: 深度学习基础** → 理解深度学习概念
4. **0D: 工具与调试** → 掌握工程实践技能

**核心收获：** 建立完整的深度学习知识体系，为后续章节打下坚实基础

---

### 路径 3：专项突破

**专注 PyTorch 实战：**
- 0B（02-05）→ 0C（06）→ Chapter 2（00-04）

**专注性能优化：**
- 0B（02-05）→ 0D（10-12）→ Chapter 3（05）

**专注调试技能：**
- 0B（03-05）→ 0D（10-13）

---

## 📗 0A: Python 基础（00-01）

### 🎯 学习目标

- ✅ 掌握 LLM 代码中常见的 Python 语法
- ✅ 理解 NumPy 数组操作和广播机制
- ✅ 学会使用 einsum 符号（为 Attention 做准备）

### 📚 题目列表

| 题号 | 题目 | 难度 | 核心知识点 |
|:---|:---|:---|:---|
| 00 | [Python Essentials for LLM](./00_Python_Essentials_for_LLM.ipynb) | Easy | 列表推导、字典、函数、装饰器、类 |
| 01 | [NumPy and Einsum](./01_NumPy_and_Einsum.ipynb) | Easy | 数组操作、广播、einsum 符号 |

### 📖 详细题目指南

#### 00: Python Essentials for LLM

**学习重点：**
- **列表推导**：生成序列、过滤数据
- **字典操作**：管理模型配置、参数字典
- **装饰器**：实现缓存、计时等功能
- **类与继承**：定义模型基类

**常见错误：**
- ❌ 列表推导中的条件语句位置错误
- ❌ 字典的 get() 和 [] 访问的区别
- ❌ 装饰器的参数传递

**进阶方向：**
- 理解 Python 的闭包和作用域
- 学习 functools 模块的高级用法

---

#### 01: NumPy and Einsum

**学习重点：**
- **数组创建与索引**：理解多维数组的操作
- **广播机制**：自动扩展维度进行运算
- **einsum 符号**：简洁地表达复杂的张量运算

**核心公式：**
- 矩阵乘法：`np.einsum('ij,jk->ik', A, B)`
- Attention 的 QK^T：`np.einsum('bqd,bkd->bqk', Q, K)`
- Batch 矩阵乘法：`np.einsum('bij,bjk->bik', A, B)`

**常见错误：**
- ❌ einsum 下标顺序错误
- ❌ 广播维度不匹配
- ❌ 忘记指定输出维度

**进阶方向：**
- 理解 einsum 的性能优化
- 学习 einops 库的使用

---

## 📗 0B: PyTorch 基础（02-05）

### 🎯 学习目标

- ✅ 掌握 Tensor 的创建、操作、设备转移
- ✅ 理解自动求导的原理和使用
- ✅ 能够定义自定义的 nn.Module
- ✅ 掌握损失函数和优化器的使用

### 📚 题目列表

| 题号 | 题目 | 难度 | 核心知识点 |
|:---|:---|:---|:---|
| 02 | [PyTorch Tensor Fundamentals](./02_PyTorch_Tensor_Fundamentals.ipynb) | Easy | Tensor 创建、操作、设备转移、数据类型 |
| 03 | [PyTorch Autograd and Backward](./03_PyTorch_Autograd_and_Backward.ipynb) | Medium | 自动求导、梯度计算、反向传播 |
| 04 | [PyTorch nn.Module Basics](./04_PyTorch_nn_Module_Basics.ipynb) | Medium | 模块定义、前向传播、参数管理 |
| 05 | [PyTorch Optimizers and Loss](./05_PyTorch_Optimizers_and_Loss.ipynb) | Medium | 损失函数、优化器、学习率 |

### 核心概念解析

#### Tensor 操作（02）

**关键操作：**
- **创建**：`torch.randn()`, `torch.zeros()`, `torch.ones()`
- **形状变换**：`view()`, `reshape()`, `permute()`, `transpose()`
- **设备转移**：`tensor.to('cuda')`, `tensor.cpu()`
- **索引切片**：`tensor[0, :, 1:3]`, `tensor.masked_fill()`

**view vs reshape vs permute：**
- `view()`：要求内存连续，速度快
- `reshape()`：自动处理内存不连续的情况
- `permute()`：改变维度顺序

---

#### 自动求导（03）

**核心概念：**
- **计算图**：PyTorch 自动构建的有向无环图
- **梯度累积**：多次 backward() 会累加梯度
- **梯度清零**：`optimizer.zero_grad()` 或 `tensor.grad.zero_()`

**自定义 autograd.Function：**
```python
class MyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_input
```

---

#### nn.Module（04）

**核心要点：**
- **继承 nn.Module**：所有自定义模块都要继承
- **__init__() 中定义子模块**：使用 `self.layer = nn.Linear(...)`
- **forward() 中定义前向传播**：不要手动调用 forward()
- **参数管理**：`parameters()`, `named_parameters()`, `state_dict()`

---

## 📗 0C: 深度学习基础（06-09）

### 🎯 学习目标

- ✅ 掌握完整的训练循环
- ✅ 理解常见激活函数的原理
- ✅ 掌握归一化技术
- ✅ 理解 Attention 机制的基础

### 📚 题目列表

| 题号 | 题目 | 难度 | 核心知识点 |
|:---|:---|:---|:---|
| 06 | [Simple Neural Network Training](./06_Simple_Neural_Network_Training.ipynb) | Medium | 训练循环、验证、保存模型 |
| 07 | [Activation Functions](./07_Activation_Functions.ipynb) | Easy | ReLU、GELU、SiLU 的实现与对比 |
| 08 | [Normalization Techniques](./08_Normalization_Techniques.ipynb) | Medium | BatchNorm、LayerNorm 的原理与实现 |
| 09 | [Attention Mechanism Intro](./09_Attention_Mechanism_Intro.ipynb) | Medium | Scaled Dot-Product Attention 基础 |

### 核心概念解析

#### 训练循环（06）

**标准训练流程：**
```python
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    # 验证阶段
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            output = model(batch)
            val_loss = criterion(output, target)
```

**关键技巧：**
- **model.train() vs model.eval()**：影响 Dropout、BatchNorm 的行为
- **torch.no_grad()**：验证时不计算梯度，节省显存
- **早停 (Early Stopping)**：防止过拟合

---

#### 激活函数（07）

**常见激活函数对比：**

| 激活函数 | 公式 | 优点 | 缺点 |
|---------|------|------|------|
| ReLU | `max(0, x)` | 简单、快速 | 死神经元问题 |
| GELU | `x * Φ(x)` | 平滑、性能好 | 计算稍慢 |
| SiLU (Swish) | `x * sigmoid(x)` | 平滑、自门控 | 计算稍慢 |

**为什么 LLM 使用 GELU/SiLU？**
- 平滑的梯度，训练更稳定
- 自门控机制，表达能力更强

---

#### 归一化技术（08）

**BatchNorm vs LayerNorm：**

| 特性 | BatchNorm | LayerNorm |
|------|-----------|-----------|
| 归一化维度 | Batch 维度 | Feature 维度 |
| 适用场景 | CNN、大 batch | RNN、Transformer、小 batch |
| 依赖 batch size | 是 | 否 |

**为什么 Transformer 使用 LayerNorm？**
- 不依赖 batch size，适合序列长度不固定的场景
- 每个样本独立归一化，适合自回归生成

---

#### Attention 机制（09）

**Scaled Dot-Product Attention：**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**关键要点：**
- **缩放因子 √d_k**：防止 softmax 饱和
- **Causal Mask**：自回归生成时屏蔽未来信息
- **多头注意力**：并行计算多个 Attention（Chapter 2 详细讲解）

---

## 📗 0D: 工具与调试（10-13）

### 🎯 学习目标

- ✅ 掌握 PyTorch Profiler 的使用
- ✅ 学会显存分析和优化
- ✅ 掌握常见的调试技巧
- ✅ 了解 Jupyter 和 Git 的基础使用

### 📚 题目列表

| 题号 | 题目 | 难度 | 核心知识点 |
|:---|:---|:---|:---|
| 10 | [PyTorch Profiling Basics](./10_PyTorch_Profiling_Basics.ipynb) | Medium | torch.profiler、性能分析、瓶颈定位 |
| 11 | [Memory Profiling and Optimization](./11_Memory_Profiling_and_Optimization.ipynb) | Medium | 内存分析、显存优化、梯度累积 |
| 12 | [Debugging Techniques](./12_Debugging_Techniques.ipynb) | Medium | 梯度检查、NaN 调试、断点调试 |
| 13 | [Jupyter and Git Basics](./13_Jupyter_and_Git_Basics.ipynb) | Easy | Notebook 使用、版本控制基础 |

### 核心概念解析

#### PyTorch Profiler（10）

**核心功能：**
- **性能分析**：测量每个操作的 CPU/GPU 时间
- **内存分析**：追踪显存使用情况
- **可视化**：导出 Chrome Trace 或 TensorBoard

**关键指标：**
- `Self CPU time`：操作本身的 CPU 时间
- `CUDA time`：GPU 执行时间
- `CPU Mem` / `CUDA Mem`：内存/显存使用

**使用示例：**
```python
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    output = model(input)
    loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

---

#### 显存优化（11）

**常用技巧：**
1. **梯度累积**：模拟大 batch size
2. **混合精度训练**：使用 FP16 节省显存
3. **梯度检查点**：用计算换显存
4. **及时释放**：`del tensor`, `torch.cuda.empty_cache()`

**显存分析命令：**
```python
torch.cuda.memory_allocated()      # 已分配的显存
torch.cuda.max_memory_allocated()  # 峰值显存
torch.cuda.memory_summary()        # 显存摘要
```

---

#### 调试技巧（12）

**常见问题排查：**
- **Loss 不下降**：检查学习率、梯度、数据标准化
- **Loss 变成 NaN**：检查学习率过大、数值溢出
- **显存溢出**：减小 batch size、使用梯度累积
- **训练速度慢**：使用 profiler 定位瓶颈

**梯度检查：**
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any():
            print(f"NaN in {name}")
        if torch.isinf(param.grad).any():
            print(f"Inf in {name}")
```

---

## 💡 学习建议

### 学习方法

1. **动手实践**：每个题目都要自己实现，不要只看答案
2. **对比验证**：用 PyTorch 官方实现验证你的代码
3. **循序渐进**：按照学习组顺序学习，不要跳跃
4. **记录笔记**：记录关键概念和常见错误

### 关键数字速查表

**PyTorch 数据类型：**
- `torch.float32` (FP32)：4 Bytes
- `torch.float16` (FP16)：2 Bytes
- `torch.bfloat16` (BF16)：2 Bytes
- `torch.int8`：1 Byte

**常用 Tensor 操作：**
- `view()` vs `reshape()`：view 要求内存连续
- `permute()` vs `transpose()`：permute 可以任意维度，transpose 只能两个维度
- `contiguous()`：使 Tensor 内存连续

### 常见问题

**Q: 没有 GPU 能学吗？**
- A: 可以！Chapter 0 的所有内容都可以在 CPU 上运行

**Q: 需要多长时间完成 Chapter 0？**
- A: 快速通关 1-3 天，系统学习 1 周左右

**Q: Chapter 0 和 Chapter 2 的 00 题有什么区别？**
- A: Chapter 0 更基础，Chapter 2 的 00 题是 PyTorch Warmup，假设你已经掌握了 Chapter 0 的内容

**Q: 可以跳过 Chapter 0 直接学习后续章节吗？**
- A: 如果你已经熟悉 PyTorch，可以跳过 0A-0C，但建议学习 0D（Profiling 和调试）

---

## 📝 学习检查清单

完成本章学习后，你应该能够：

**0A: Python 基础**
- [ ] 熟练使用列表推导和字典操作
- [ ] 理解 NumPy 的广播机制
- [ ] 能够使用 einsum 表达复杂的张量运算

**0B: PyTorch 基础**
- [ ] 熟练创建和操作 Tensor
- [ ] 理解自动求导的原理
- [ ] 能够定义自定义的 nn.Module
- [ ] 掌握损失函数和优化器的使用

**0C: 深度学习基础**
- [ ] 能够编写完整的训练循环
- [ ] 理解常见激活函数的特点
- [ ] 掌握 BatchNorm 和 LayerNorm 的区别
- [ ] 理解 Attention 机制的基础原理

**0D: 工具与调试**
- [ ] 能够使用 torch.profiler 分析性能
- [ ] 掌握显存优化的常用技巧
- [ ] 能够调试 NaN 和梯度问题
- [ ] 了解 Jupyter 和 Git 的基础使用

---

## 🔗 与其他章节的联系

**Chapter 0 → Chapter 1：**
- 00-01 题（Python/NumPy）→ 01 题（数学推导和公式理解）
- 02-05 题（PyTorch 基础）→ 01 题（理论可以用代码验证）
- 10 题（Profiling）→ 03 题（性能分析理论）

**Chapter 0 → Chapter 2：**
- 05 题（训练循环）→ 09-10 题（SFT、LoRA 训练）
- 07-08 题（激活函数、归一化）→ 01-02 题（RMSNorm、SwiGLU）
- 09 题（Attention 基础）→ 04 题（MHA/GQA）
- 10 题（Profiling）→ 所有题目的性能对比

**Chapter 0 → Chapter 3：**
- 03 题（Autograd）→ 01-05 题（自定义 Triton 算子）
- 10 题（Profiling）→ 05 题（Triton Profiling）
- 11 题（Memory Profiling）→ 08 题（Flash Attention 显存优化）
- 12 题（Debugging）→ 12 题（Triton 调试技巧）

---

## 🎓 结语

Chapter 0 是整个学习路径的起点，虽然内容基础，但这些知识是理解后续章节的关键。

**学习建议：**
- **不要急于求成**：基础打牢，后续学习会更轻松
- **动手实践**：每个题目都要自己实现
- **善用工具**：Profiling 和调试技能在实际工作中非常重要
- **循序渐进**：按照推荐路径学习，不要跳跃

**记住：**
- 基础是一切的根基
- Profiling 是性能优化的第一步
- 调试技能是工程师的核心竞争力

祝学习愉快！🚀
