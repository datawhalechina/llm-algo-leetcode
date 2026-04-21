# 24. Tensor Parallelism Sim | 突破单卡显存上限：张量并行 (Tensor Parallelism, TP) 的矩阵切片模拟

**难度：** Hard | **标签：** `Distributed Training`, `Tensor Parallelism`, `Megatron-LM` | **目标人群：** 核心 Infra 与算子开发

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/24_Tensor_Parallelism_Sim.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


ZeRO 切分了状态，但**模型参数本身（Weights）**在每张卡上依然是完整的。如果模型高达 70B（140GB 显存），甚至连一张 80G 的 A100 都放不下完整的模型，这时 ZeRO-1/2 也无能为力了。
Megatron-LM 提出了 **Tensor Parallelism (张量并行，TP)**。它将一个大规模的矩阵乘法 $Y = XA$，把权重矩阵 $A$ 切成几小块，分别放在不同的 GPU 上算，最后再把结果拼起来。
本节我们将纯粹用张量切片操作，模拟 TP 是如何把一个大规模的 Linear 层拆分到 2 张逻辑卡上的。

### Step 1: TP的两种切法

假设输入 $X$ 形状为 `(batch, in_dim)`，权重 $A$ 形状为 `(in_dim, out_dim)`，经过线性层变为 $Y = XA$，形状 `(batch, out_dim)`。

> **Column Parallel (列切分)：切分 $A$ 的列 (输出维度)**
> 1. $A$ 被竖着切成左右两块 $A_1, A_2$ 分别放到 GPU 0 和 1。
> 2. GPU 0 计算 $Y_1 = X A_1$，GPU 1 计算 $Y_2 = X A_2$。
> 3. **通信：** 各自算完后，通过 `All-Gather`，把左右结果拼起来，得到完整的 $Y = [Y_1, Y_2]$。
> *适用场景：MLP 的第一个全连接层（扩大隐藏维度时）。*

> **Row Parallel (行切分)：切分 $A$ 的行 (输入维度)**
> 1. $A$ 被横着切成上下两块 $A_1, A_2$ 分别放到 GPU 0 和 1。
> 2. 输入 $X$ 也要沿着特征维度切成左右两半 $X_1, X_2$ 给不同的卡。
> 3. GPU 0 计算 $Y_1 = X_1 A_1$，GPU 1 计算 $Y_2 = X_2 A_2$。
> 4. **通信：** 完整的结果其实是两者的加和：$Y = Y_1 + Y_2$。所以需要做一次 `All-Reduce (Sum)`。
> *适用场景：MLP 的第二个全连接层（缩回原始维度时）。*

**精妙之处**：如果把 Column Parallel 放前面，Row Parallel 放后面，中间甚至可以省掉一次通信！

### Step 2: Column 与 Row Parallelism 推导
在一个两层的前馈网络 $Y = X \cdot W_1 \cdot W_2$ 中：
- 我们将 $W_1$ 按列切分（Column Parallel），得到两块。计算后各个 GPU 得到不完整的部分输出矩阵。
- 紧接着，将 $W_2$ 按行切分（Row Parallel），利用刚才的部分输出分别与之相乘。
- 最后，所有 GPU 执行一次 `All-Reduce` 聚合结果。这样在两层神经网络中，只产生了一次通信开销！

### Step 3: 代码实现框架
你需要实现张量切片操作（类似 `torch.chunk`），分别针对线性层的权重矩阵在维度 0 或维度 1 进行切割。然后在模拟多进程执行时，分别利用切好的局部权重完成前向传播，最终利用 `torch.sum` 模拟一次 All-Reduce 收集聚合数据。

###  Step 4: 动手实战

**要求**：请补全下方代码，手动将一个大规模的矩阵乘法拆分成两张“逻辑卡”上的 Column Parallel 操作，并验证结果拼接后与单卡全量计算一致。


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
```


```python
def tensor_parallel_column_sim(X: torch.Tensor, A: torch.Tensor, num_gpus: int = 2):
    """
    模拟 Column Parallel Linear: Y = X @ A
    将权重 A 沿列 (输出特征维度) 切分，分布到不同的 GPU 上计算，最后拼接。
    
    参数:
    X: 形状 (batch, in_features)
    A: 形状 (in_features, out_features)
    """
    in_features, out_features = A.shape
    assert out_features % num_gpus == 0, "输出维度必须能被 GPU 数量整除"
    
    chunk_size = out_features // num_gpus
    
    # 1. 模拟将权重加载到不同 GPU 的显存中
    # gpu_weights 是一个列表，代表各 GPU 本地保存的权重分片
    gpu_weights = []
    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        # ==========================================
        # TODO 1: 沿列方向 (dim=1) 对 A 进行切片
        # ==========================================
        # weight_chunk = ???
        # gpu_weights.append(weight_chunk)
        weight_chunk = torch.zeros(in_features, chunk_size)  # 占位初始化
        gpu_weights.append(weight_chunk)
        
    # 2. 模拟各 GPU 并行前向计算
    # 在真实环境中，X 会被广播到所有 GPU (因为是列切分，输入不需要切)
    gpu_outputs = []
    for i in range(num_gpus):
        # ==========================================
        # TODO 2: 每张卡使用自己本地的权重分片，对输入 X 进行矩阵乘法计算
        # ==========================================
        # local_out = ???
        # gpu_outputs.append(local_out)
        local_out = X @ gpu_weights[i]  # 占位初始化
        gpu_outputs.append(local_out)
        
    # 3. 模拟 All-Gather 通信操作
    # ==========================================
    # TODO 3: 将各 GPU 计算的结果沿特征维度 (dim=1) 拼接起来
    # ==========================================
    # Y_gathered = ???
    Y_gathered = torch.cat(gpu_outputs, dim=1)  # 占位初始化
    return Y_gathered

```


```python
# 测试你的实现
def test_tensor_parallel():
    try:
        torch.manual_seed(42)
        batch_size = 4
        in_dim = 16
        out_dim = 32
        
        # 原始数据
        X = torch.randn(batch_size, in_dim)
        A = torch.randn(in_dim, out_dim)
        
        # 1. 单卡全量计算作为 Ground Truth
        Y_ref = X @ A
        
        # 2. 模拟 2 张卡的 Column Parallel
        Y_tp = tensor_parallel_column_sim(X, A, num_gpus=2)
        
        # 3. 验证结果完全一致
        diff = torch.max(torch.abs(Y_ref - Y_tp))
        print(f"最大误差: {diff.item():.6e}")
        assert diff < 1e-5, "TP 模拟结果与单卡全量计算不一致！"
        
        print("✅ Column Parallel (列切分) 矩阵计算与拼接逻辑正确！")
        print("掌握了 Megatron-LM 的核心张量切分思路，单卡装不下的大规模参数量再也不是问题。")
        
    except NotImplementedError:
        print("请先完成 TODO 代码！")
    except TypeError as e:
        print("代码可能未完成，导致了操作错误。")
        raise e
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise e

test_tensor_parallel()

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
def tensor_parallel_column_sim(X, A, num_gpus):
    # 1. 权重切分 (Scatter)
    a_chunks = torch.chunk(A, num_gpus, dim=1)
    
    # 2. 独立计算 (Local MatMul)
    y_chunks = []
    for i in range(num_gpus):
        a_local = a_chunks[i]
        y_local = X @ a_local
        y_chunks.append(y_local)
        
    # 3. 结果合并 (All-Gather)
    Y_tp = torch.cat(y_chunks, dim=-1)
    return Y_tp
```
