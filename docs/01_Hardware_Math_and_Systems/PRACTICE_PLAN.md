# PRACTICE. PLAN | Chapter 1 练习题规划方案

## 🎯 总体方案：混合方案 C

结合理论计算练习（.md）和代码实践（.ipynb），为 Chapter 1 添加动手练习环节。

---

## 📊 实施策略

### 阶段 1：所有题目添加计算练习（.md 文件）

为 10 个题目的 .md 文件末尾添加"🎯 计算练习"部分，每个题目 3-5 个练习。

| 题号 | 题目 | 练习数量 | 练习类型 |
|:---|:---|:---|:---|
| 01 | Data Types and Precision | 5 | 显存计算、混合精度、量化 |
| 02 | LLM Params and FLOPs | 5 | 参数量、FLOPs、训练时间 |
| 03 | GPU Architecture and Memory | 4 | 带宽计算、内存层次 |
| 04 | Attention Memory Optimization | 4 | Attention 显存、FlashAttention |
| 05 | Communication Topologies | 3 | 通信带宽、延迟计算 |
| 06 | VRAM Calculation and ZeRO | 5 | ZeRO 显存、梯度累积 |
| 07 | CPU GPU Heterogeneous Scheduling | 3 | 数据传输时间 |
| 08 | Programming Models CUDA Triton | 3 | 性能对比 |
| 09 | AI Compilers and Graph Optimization | 3 | 算子融合收益 |
| 10 | Domestic AI Chips Overview | 2 | 性能对比 |

**总计：** 37 个计算练习

---

### 阶段 2：重点题目创建代码练习（.ipynb 文件）

为 3 个最重要的题目创建配套的 Jupyter Notebook 练习文件。

| 题号 | 题目 | Notebook 文件名 | 练习内容 |
|:---|:---|:---|:---|
| 01 | Data Types and Precision | `01_Data_Types_and_Precision_Practice.ipynb` | 显存计算函数、混合精度、量化 |
| 02 | LLM Params and FLOPs | `02_LLM_Params_and_FLOPs_Practice.ipynb` | 参数量计算、FLOPs 计算、训练时间估算 |
| 06 | VRAM Calculation and ZeRO | `06_VRAM_Calculation_and_ZeRO_Practice.ipynb` | ZeRO 显存计算、最优配置选择 |

---

## 📖 练习题设计模板

### .md 文件中的计算练习模板

```markdown
## 🎯 计算练习

### 练习 1: [练习标题]

**问题：** [具体问题描述]

**已知条件：**
- 条件 1
- 条件 2
- 条件 3

**提示：** [解题提示或公式]

**答案：**
<details>
<summary>点击查看答案</summary>

**解题步骤：**
1. 步骤 1
2. 步骤 2
3. 步骤 3

**最终答案：** XXX

**解析：** [为什么这样计算，有什么实际意义]
</details>

---

### 练习 2: [练习标题]
...
```

---

### .ipynb 文件的练习模板

```python
# Part 1: 函数实现

def calculate_xxx(param1, param2):
    """
    计算 XXX
    
    Args:
        param1: 参数 1 说明
        param2: 参数 2 说明
    
    Returns:
        result: 结果说明
    
    示例:
        >>> calculate_xxx(7, 'fp16')
        14.0
    """
    # TODO: 实现计算逻辑
    result = 0  # 占位初始化
    return result

# Part 2: 测试用例

def test_calculate_xxx():
    # 测试用例 1
    result = calculate_xxx(7, 'fp16')
    assert result == 14, f"错误：期望 14，实际 {result}"
    
    # 测试用例 2
    result = calculate_xxx(7, 'int8')
    assert result == 7, f"错误：期望 7，实际 {result}"
    
    print("✅ 所有测试通过！")

test_calculate_xxx()

# Part 3: 实战应用

# 应用示例：计算不同模型的显存占用
models = {
    'LLaMA-7B': 7,
    'LLaMA-13B': 13,
    'LLaMA-70B': 70,
}

for name, params in models.items():
    memory = calculate_xxx(params, 'fp16')
    print(f"{name}: {memory:.1f} GB")
```

---

## 📝 详细练习题设计

### 01_Data_Types_and_Precision

#### .md 文件中的计算练习（5 个）

**练习 1: 基础显存计算**
- 问题：LLaMA-7B 使用 FP16 加载需要多少显存？
- 答案：14 GB

**练习 2: 混合精度训练显存**
- 问题：使用混合精度训练 7B 模型，总显存占用？
- 答案：112 GB（16Φ）

**练习 3: 量化显存节省**
- 问题：FP16 量化到 INT8 节省多少显存？
- 答案：50%

**练习 4: 不同精度对比**
- 问题：对比 FP32、FP16、INT8、INT4 的显存占用
- 答案：28GB、14GB、7GB、3.5GB

**练习 5: 实际场景计算**
- 问题：A100 80GB 能加载多大的 FP16 模型？
- 答案：约 40B（考虑 KV Cache 和激活值）

#### .ipynb 文件的练习（3 个函数）

**函数 1: `calculate_model_memory(num_params, dtype)`**
- 计算模型参数显存占用

**函数 2: `calculate_training_memory(num_params, dtype, optimizer='adam')`**
- 计算训练时的总显存占用（参数 + 梯度 + 优化器）

**函数 3: `calculate_quantization_savings(num_params, from_dtype, to_dtype)`**
- 计算量化后的显存节省

---

### 02_LLM_Params_and_FLOPs

#### .md 文件中的计算练习（5 个）

**练习 1: Transformer 参数量计算**
- 问题：计算 LLaMA-7B 的参数量分布
- 答案：Embedding + Attention + FFN

**练习 2: 训练 FLOPs 计算**
- 问题：训练 7B 模型 1T tokens 需要多少 FLOPs？
- 答案：6 × 7B × 1T = 42e21 FLOPs

**练习 3: 推理 FLOPs 计算**
- 问题：推理 1000 tokens 需要多少 FLOPs？
- 答案：2 × 7B × 1000 = 14e12 FLOPs

**练习 4: 训练时间估算**
- 问题：A100 训练 7B 模型需要多少时间？
- 答案：基于 312 TFLOPS 计算

**练习 5: Chinchilla 定律应用**
- 问题：根据 Chinchilla 定律，7B 模型应该用多少数据训练？
- 答案：约 140B tokens

#### .ipynb 文件的练习（3 个函数）

**函数 1: `calculate_transformer_params(vocab_size, hidden_dim, num_layers, num_heads)`**
- 计算 Transformer 参数量

**函数 2: `calculate_training_flops(num_params, num_tokens)`**
- 计算训练 FLOPs

**函数 3: `estimate_training_time(num_params, num_tokens, gpu_flops, num_gpus)`**
- 估算训练时间

---

### 06_VRAM_Calculation_and_ZeRO

#### .md 文件中的计算练习（5 个）

**练习 1: DDP 显存计算**
- 问题：8 卡 DDP 训练 7B 模型，每卡显存占用？
- 答案：112 GB（16Φ）

**练习 2: ZeRO-1 显存节省**
- 问题：ZeRO-1 相比 DDP 节省多少显存？
- 答案：节省 10.5Φ（87.5% 的优化器状态）

**练习 3: ZeRO-2 显存节省**
- 问题：ZeRO-2 每卡显存占用？
- 答案：2Φ + 14Φ/N

**练习 4: ZeRO-3 显存节省**
- 问题：ZeRO-3 每卡显存占用？
- 答案：16Φ/N

**练习 5: 梯度累积**
- 问题：使用梯度累积模拟 batch_size=128，实际 batch_size 应该设为多少？
- 答案：取决于 accumulation_steps

#### .ipynb 文件的练习（3 个函数）

**函数 1: `calculate_zero_memory(num_params, num_gpus, zero_stage)`**
- 计算 ZeRO 不同阶段的显存占用

**函数 2: `calculate_gradient_accumulation(target_batch_size, gpu_memory, model_params)`**
- 计算需要的梯度累积步数

**函数 3: `find_optimal_config(model_params, num_gpus, gpu_memory)`**
- 找出最优的训练配置（ZeRO stage + batch size + accumulation）

---

## 🎯 练习题设计原则

1. **实用性**：所有练习都基于真实场景（LLaMA、GPT-3 等）
2. **递进性**：从简单到复杂，从单一计算到综合应用
3. **可验证**：提供明确答案或测试用例
4. **面试导向**：覆盖高频面试题
5. **工程价值**：帮助做出实际的工程决策

---

## 📅 实施时间表

### 阶段 1：计算练习（预计 2-3 小时）
- [ ] 01 题：5 个练习
- [ ] 02 题：5 个练习
- [ ] 03 题：4 个练习
- [ ] 04 题：4 个练习
- [ ] 05 题：3 个练习
- [ ] 06 题：5 个练习
- [ ] 07 题：3 个练习
- [ ] 08 题：3 个练习
- [ ] 09 题：3 个练习
- [ ] 10 题：2 个练习

### 阶段 2：代码练习（预计 3-4 小时）
- [ ] 01_Data_Types_and_Precision_Practice.ipynb
- [ ] 02_LLM_Params_and_FLOPs_Practice.ipynb
- [ ] 06_VRAM_Calculation_and_ZeRO_Practice.ipynb

### 阶段 3：测试和完善（预计 1 小时）
- [ ] 验证所有答案的正确性
- [ ] 测试 .ipynb 文件的测试用例
- [ ] 更新 intro.md 说明练习题

---

## 🔗 与其他章节的联系

**Chapter 1 练习 → Chapter 2 实践**
- 01 题（显存计算）→ Chapter 2 的 21 题（Gradient Checkpointing）
- 02 题（参数量）→ Chapter 2 的 05-08 题（模型架构）
- 06 题（ZeRO）→ Chapter 2 的 23 题（ZeRO 模拟）

**Chapter 1 练习 → Chapter 3 优化**
- 03 题（GPU 架构）→ Chapter 3 的 Triton 优化
- 04 题（Attention 优化）→ Chapter 3 的 FlashAttention

---

## 📊 预期效果

完成练习题后，学习者能够：
- ✅ 快速计算模型的显存占用
- ✅ 估算训练时间和成本
- ✅ 选择合适的 ZeRO 策略
- ✅ 做出正确的工程决策
- ✅ 回答面试中的计算题

---

## 🎓 总结

**方案 C 的优势：**
1. ✅ 理论与实践结合
2. ✅ 循序渐进（.md 计算 → .ipynb 代码）
3. ✅ 覆盖面广（10 个题目 37 个练习）
4. ✅ 重点突出（3 个核心题目有代码实践）
5. ✅ 可验证性强（测试用例）

**实施建议：**
- 先完成阶段 1（计算练习），快速提升实用性
- 再完成阶段 2（代码练习），提供深度实践
- 最后完善测试和文档
