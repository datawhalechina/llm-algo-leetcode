# 01. Data Types and Precision Practice | Data Types and Precision - 计算练习

**难度：** Easy | **标签：** `显存计算`, `混合精度`, `量化` | **目标人群：** 所有学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/01_Data_Types_and_Precision_Practice.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本练习配套理论文档：[01_Data_Types_and_Precision.md](./01_Data_Types_and_Precision.md)

---

## 🎯 学习目标

- 掌握不同数据格式的显存占用计算
- 理解混合精度训练的显存分布
- 学会计算量化后的显存节省
- 能够为实际场景选择合适的数据格式

---
## Part 1: 基础显存计算

### 练习 1.1: 实现显存计算函数

实现一个函数，计算给定参数量和数据格式的模型显存占用。

```python
def calculate_model_memory(num_params_b, dtype):
    """
    计算模型参数的显存占用
    
    Args:
        num_params_b: 参数量（单位：B，即十亿）
        dtype: 数据类型，可选 'fp32', 'fp16', 'bf16', 'int8', 'int4'
    
    Returns:
        memory_gb: 显存占用（单位：GB）
    
    示例:
        >>> calculate_model_memory(7, 'fp16')
        14.0
        >>> calculate_model_memory(7, 'int8')
        7.0
    """
    # 每种数据类型占用的字节数
    bytes_per_param = {
        'fp32': 4,
        'fp16': 2,
        'bf16': 2,
        'int8': 1,
        'int4': 0.5
    }
    
    # TODO: 实现显存计算
    # 提示: 显存(GB) = 参数量(B) × 每参数字节数
    # memory_gb = ???
    
    memory_gb = 0  # 占位初始化
    return memory_gb
```


```python
# 测试函数
def test_calculate_model_memory():
    try:
        # 测试用例 1: LLaMA-7B FP16
        result = calculate_model_memory(7, 'fp16')
        assert result == 14, f"错误：LLaMA-7B FP16 应该是 14 GB，实际 {result} GB"
        
        # 测试用例 2: LLaMA-7B INT8
        result = calculate_model_memory(7, 'int8')
        assert result == 7, f"错误：LLaMA-7B INT8 应该是 7 GB，实际 {result} GB"
        
        # 测试用例 3: LLaMA-13B FP16
        result = calculate_model_memory(13, 'fp16')
        assert result == 26, f"错误：LLaMA-13B FP16 应该是 26 GB，实际 {result} GB"
        
        # 测试用例 4: LLaMA-70B INT4
        result = calculate_model_memory(70, 'int4')
        assert result == 35, f"错误：LLaMA-70B INT4 应该是 35 GB，实际 {result} GB"
        
        print("✅ 所有测试通过！")
        
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
    except Exception as e:
        print(f"❌ 运行错误: {e}")

test_calculate_model_memory()
```

### 练习 1.2: 对比不同数据格式

使用上面的函数，对比 LLaMA-7B 在不同数据格式下的显存占用。

```python
# TODO: 计算 LLaMA-7B 在不同格式下的显存占用
model_name = "LLaMA-7B"
num_params = 7
dtypes = ['fp32', 'fp16', 'bf16', 'int8', 'int4']

print(f"{model_name} 显存占用对比：")
print("-" * 40)
for dtype in dtypes:
    memory = calculate_model_memory(num_params, dtype)
    print(f"{dtype.upper():<8} {memory:>6.1f} GB")
```

---

## Part 2: 混合精度训练显存计算

### 练习 2.1: 实现训练显存计算函数

在混合精度训练中，显存占用包括：
- 模型参数（FP16/BF16）：2Φ
- 梯度（FP16/BF16）：2Φ
- 优化器状态（FP32）：
  - FP32 主权重：4Φ
  - 一阶动量（Adam）：4Φ
  - 二阶动量（Adam）：4Φ
  - 总计：12Φ

**总显存 = 2Φ + 2Φ + 12Φ = 16Φ**

```python
def calculate_training_memory(num_params_b, model_dtype='fp16', optimizer='adam'):
    """
    计算混合精度训练的总显存占用
    
    Args:
        num_params_b: 参数量（单位：B）
        model_dtype: 模型数据类型（'fp16' 或 'bf16'）
        optimizer: 优化器类型（'adam' 或 'sgd'）
    
    Returns:
        total_memory_gb: 总显存占用（单位：GB）
    
    示例:
        >>> calculate_training_memory(7, 'fp16', 'adam')
        112.0
    """
    # TODO: 实现训练显存计算
    # 提示:
    # 1. 模型参数（FP16/BF16）：2Φ
    # 2. 梯度（FP16/BF16）：2Φ
    # 3. 优化器状态（FP32）：
    #    - Adam: 12Φ（主权重4Φ + 动量4Φ + 方差4Φ）
    #    - SGD: 4Φ（只有主权重）
    
    total_memory_gb = 0  # 占位初始化
    return total_memory_gb
```


```python
# 测试函数
def test_calculate_training_memory():
    try:
        # 测试用例 1: LLaMA-7B + Adam
        result = calculate_training_memory(7, 'fp16', 'adam')
        assert result == 112, f"错误：应该是 112 GB，实际 {result} GB"
        
        # 测试用例 2: LLaMA-7B + SGD
        result = calculate_training_memory(7, 'fp16', 'sgd')
        assert result == 32, f"错误：应该是 32 GB，实际 {result} GB"
        
        # 测试用例 3: LLaMA-13B + Adam
        result = calculate_training_memory(13, 'bf16', 'adam')
        assert result == 208, f"错误：应该是 208 GB，实际 {result} GB"
        
        print("✅ 所有测试通过！")
        
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
    except Exception as e:
        print(f"❌ 运行错误: {e}")

test_calculate_training_memory()
```

### 练习 2.2: 分析显存分布

分析混合精度训练中，各部分显存的占比。

```python
# TODO: 分析 LLaMA-7B 训练时的显存分布
num_params = 7

model_params = num_params * 2  # FP16 模型参数
gradients = num_params * 2     # FP16 梯度
optimizer_states = num_params * 12  # FP32 优化器状态
total = model_params + gradients + optimizer_states

print(f"LLaMA-7B 混合精度训练显存分布：")
print("-" * 50)
print(f"模型参数 (FP16):      {model_params:>6.1f} GB ({model_params/total*100:>5.1f}%)")
print(f"梯度 (FP16):          {gradients:>6.1f} GB ({gradients/total*100:>5.1f}%)")
print(f"优化器状态 (FP32):    {optimizer_states:>6.1f} GB ({optimizer_states/total*100:>5.1f}%)")
print("-" * 50)
print(f"总计:                 {total:>6.1f} GB")
print("\n💡 结论：优化器状态占据了大部分显存！")
```

---

## Part 3: 量化显存节省计算

### 练习 3.1: 实现量化节省计算函数

```python
def calculate_quantization_savings(num_params_b, from_dtype, to_dtype):
    """
    计算量化后的显存节省
    
    Args:
        num_params_b: 参数量（单位：B）
        from_dtype: 原始数据类型
        to_dtype: 量化后的数据类型
    
    Returns:
        savings_gb: 节省的显存（单位：GB）
        savings_percent: 节省的百分比
    
    示例:
        >>> calculate_quantization_savings(7, 'fp16', 'int8')
        (7.0, 50.0)
    """
    # TODO: 实现量化节省计算
    # 提示:
    # 1. 计算原始显存
    # 2. 计算量化后显存
    # 3. 计算节省量和百分比
    
    savings_gb = 0  # 占位初始化
    savings_percent = 0  # 占位初始化
    return savings_gb, savings_percent
```


```python
# 测试函数
def test_calculate_quantization_savings():
    try:
        # 测试用例 1: FP16 -> INT8
        savings_gb, savings_percent = calculate_quantization_savings(7, 'fp16', 'int8')
        assert savings_gb == 7, f"错误：应该节省 7 GB，实际 {savings_gb} GB"
        assert savings_percent == 50, f"错误：应该节省 50%，实际 {savings_percent}%"
        
        # 测试用例 2: FP16 -> INT4
        savings_gb, savings_percent = calculate_quantization_savings(7, 'fp16', 'int4')
        assert savings_gb == 10.5, f"错误：应该节省 10.5 GB，实际 {savings_gb} GB"
        assert savings_percent == 75, f"错误：应该节省 75%，实际 {savings_percent}%"
        
        print("✅ 所有测试通过！")
        
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
    except Exception as e:
        print(f"❌ 运行错误: {e}")

test_calculate_quantization_savings()
```

### 练习 3.2: 量化方案对比

对比 LLaMA-70B 在不同量化方案下的显存占用。

```python
# TODO: 对比 LLaMA-70B 的不同量化方案
num_params = 70
quantization_schemes = [
    ('FP16 (原始)', 'fp16', 'fp16'),
    ('INT8 量化', 'fp16', 'int8'),
    ('INT4 量化', 'fp16', 'int4'),
]

print(f"LLaMA-70B 量化方案对比：")
print("-" * 60)
print(f"{'方案':<15} {'显存占用':<12} {'节省显存':<12} {'节省比例'}")
print("-" * 60)

baseline_memory = calculate_model_memory(num_params, 'fp16')

for name, from_dtype, to_dtype in quantization_schemes:
    memory = calculate_model_memory(num_params, to_dtype)
    savings_gb, savings_percent = calculate_quantization_savings(num_params, from_dtype, to_dtype)
    print(f"{name:<15} {memory:>6.1f} GB    {savings_gb:>6.1f} GB    {savings_percent:>5.1f}%")
```

---

## Part 4: 实际场景应用

### 练习 4.1: GPU 显存容量规划

给定 GPU 显存容量，计算能加载多大的模型。

```python
def max_model_size(gpu_memory_gb, dtype, overhead_ratio=0.2):
    """
    计算给定 GPU 显存能加载的最大模型参数量
    
    Args:
        gpu_memory_gb: GPU 显存容量（单位：GB）
        dtype: 数据类型
        overhead_ratio: 预留给 KV Cache 和激活值的显存比例（默认 20%）
    
    Returns:
        max_params_b: 最大参数量（单位：B）
    
    示例:
        >>> max_model_size(80, 'fp16', 0.2)
        32.0
    """
    # TODO: 实现最大模型参数量计算
    # 提示:
    # 1. 可用显存 = GPU 显存 × (1 - overhead_ratio)
    # 2. 最大参数量 = 可用显存 / 每参数字节数
    
    max_params_b = 0  # 占位初始化
    return max_params_b
```


```python
# 测试不同 GPU 能加载的最大模型
gpus = [
    ('RTX 3090', 24),
    ('RTX 4090', 24),
    ('A100 40GB', 40),
    ('A100 80GB', 80),
    ('H100 80GB', 80),
]

print("不同 GPU 能加载的最大模型参数量（FP16，预留 20% 显存）：")
print("-" * 60)
print(f"{'GPU':<15} {'显存':<10} {'最大模型 (FP16)':<20} {'最大模型 (INT8)'}")
print("-" * 60)

for gpu_name, memory in gpus:
    max_fp16 = max_model_size(memory, 'fp16', 0.2)
    max_int8 = max_model_size(memory, 'int8', 0.2)
    print(f"{gpu_name:<15} {memory:>4} GB     {max_fp16:>6.1f}B              {max_int8:>6.1f}B")
```

### 练习 4.2: 综合场景分析

**场景：** 你有 8 张 A100 80GB，想训练一个大模型。

**问题：**
1. 使用 DDP（数据并行），每张卡需要完整的模型 + 梯度 + 优化器状态，最大能训练多大的模型？
2. 如果使用 ZeRO-3（将所有状态切分到 8 张卡），最大能训练多大的模型？

**提示：**
- DDP：每卡显存 = 16Φ（混合精度训练）
- ZeRO-3：每卡显存 = 16Φ / 8
- 预留 20% 显存给激活值

```python
# TODO: 计算 DDP 和 ZeRO-3 的最大模型参数量
gpu_memory = 80  # A100 80GB
num_gpus = 8
overhead_ratio = 0.2

# 可用显存
available_memory = gpu_memory * (1 - overhead_ratio)

# DDP: 每卡需要 16Φ
max_params_ddp = available_memory / 16

# ZeRO-3: 每卡需要 16Φ / 8
max_params_zero3 = available_memory / (16 / num_gpus)

print(f"8 张 A100 80GB 训练配置对比：")
print("-" * 50)
print(f"DDP (数据并行):       最大 {max_params_ddp:>5.1f}B 参数")
print(f"ZeRO-3 (状态切分):    最大 {max_params_zero3:>5.1f}B 参数")
print(f"\n💡 ZeRO-3 相比 DDP 提升了 {max_params_zero3/max_params_ddp:.1f}x！")
```

---

## 🎓 总结

通过本练习，你应该掌握了：
- ✅ 不同数据格式的显存占用计算
- ✅ 混合精度训练的显存分布
- ✅ 量化后的显存节省计算
- ✅ 实际场景中的 GPU 显存规划

**关键要点：**
- FP16/BF16 相比 FP32 节省 50% 显存
- 混合精度训练中，优化器状态占据 75% 的显存
- INT8 量化可以节省 50% 显存，INT4 可以节省 75%
- ZeRO-3 可以将训练显存线性扩展到多卡

**下一步：** 学习 [02. LLM Params and FLOPs](./02_LLM_Params_and_FLOPs.md)，掌握参数量和 FLOPs 的计算。