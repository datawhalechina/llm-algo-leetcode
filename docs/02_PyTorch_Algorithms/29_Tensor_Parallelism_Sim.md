# 29. Tensor Parallelism Sim | Tensor 并行模拟

**难度：** Hard | **环境：** CPU-first | **标签：** `分布式训练`, `Tensor Parallelism`, `通信` | **目标人群：** 分布式训练工程师

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/29_Tensor_Parallelism_Sim.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


先把张量切分和通信模式理清，再看 Column / Row Parallelism 的组合关系会更容易理解张量并行。

**关键词：** `Tensor Parallelism`, `Column Parallel`, `Row Parallel`

## 前置阅读

**导语：** 先看 ZeRO 和 Pipeline，再看 Tensor Parallelism 会更容易把三种并行策略区分开。

- [05. Communication Topologies | 通信拓扑与分布式基石](../01_Hardware_Math_and_Systems/05_Communication_Topologies.md)
- [20. NCCL and AllReduce Basics | NCCL 与 AllReduce 基础](../01_Hardware_Math_and_Systems/20_NCCL_and_AllReduce_Basics.md)
- [26. Parallel Strategy Decision Framework | 并行策略决策框架](../01_Hardware_Math_and_Systems/26_Parallel_Strategy_Decision_Framework.md)


## 相关阅读

**导语：** 并行策略看完后，就可以进入项目实战页做综合收口。

- [17. CUDA Stream and Asynchrony | CUDA Stream 与异步执行](../01_Hardware_Math_and_Systems/17_CUDA_Stream_and_Asynchrony.md)
- [27. Communication Scheduling Optimization | 通信调度优化](../01_Hardware_Math_and_Systems/27_Communication_Scheduling_Optimization.md)
- [08. Programming Models and CUDA/Triton | 编程模型演进](../01_Hardware_Math_and_Systems/08_Programming_Models_CUDA_Triton.md)

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

### Step 4: 动手实战
**要求**：请补全下方代码，分别实现 Column Parallel 和 Row Parallel 两种张量并行切分方式，并验证它们与单卡全量计算一致。


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
    # a_chunks 是一个列表，代表各 GPU 本地保存的权重分片
    a_chunks = []
    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        # ==========================================
        # TODO 1: 沿列方向 (dim=1) 对 A 进行切片
        # ==========================================
        # a_chunk = ???
        a_chunks.append(a_chunk)
        
    # 2. 模拟各 GPU 并行前向计算
    # 在真实环境中，X 会被广播到所有 GPU (因为是列切分，输入不需要切)
    y_chunks = []
    for i in range(num_gpus):
        # ==========================================
        # TODO 2: 每张卡使用自己本地的权重分片，对输入 X 进行矩阵乘法计算
        # ==========================================
        # y_local = ???
        y_chunks.append(y_local)
        
    # 3. 模拟 All-Gather 通信操作
    # ==========================================
    # TODO 3: 将各 GPU 计算的结果沿特征维度 (dim=1) 拼接起来
    # ==========================================
    # Y_tp = ???
    return Y_tp


def tensor_parallel_row_sim(X: torch.Tensor, A: torch.Tensor, num_gpus: int = 2):
    """
    模拟 Row Parallel Linear: Y = X @ A
    将权重 A 沿行 (输入特征维度) 切分，输入 X 也同步切分，最后将各卡输出求和。
    
    参数:
    X: 形状 (batch, in_features)
    A: 形状 (in_features, out_features)
    """
    in_features, out_features = A.shape
    assert in_features % num_gpus == 0, "输入维度必须能被 GPU 数量整除"
    
    chunk_size = in_features // num_gpus
    
    # 1. 模拟将输入和权重切分给不同 GPU 的显存中
    x_chunks = []
    a_chunks = []
    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        # ==========================================
        # TODO 4: 沿行方向 (dim=0) 对 A 进行切片，并同步切分 X
        # ==========================================
        # a_chunk = ???
        # x_chunk = ???
        a_chunks.append(a_chunk)
        x_chunks.append(x_chunk)
        
    # 2. 模拟各 GPU 并行前向计算
    y_outputs = []
    for i in range(num_gpus):
        # ==========================================
        # TODO 5: 每张卡使用自己本地的输入/权重分片，进行矩阵乘法计算
        # ==========================================
        # y_local = ???
        y_outputs.append(y_local)
        
    # 3. 模拟 All-Reduce (Sum)
    # ==========================================
    # TODO 6: 将各 GPU 的部分结果按元素相加，恢复完整输出
    # ==========================================
    # Y_tp = ???
    return Y_tp

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
        Y_col = tensor_parallel_column_sim(X, A, num_gpus=2)
        diff_col = torch.max(torch.abs(Y_ref - Y_col))
        print(f"Column Parallel 最大误差: {diff_col.item():.6e}")
        assert Y_col.shape == Y_ref.shape, "Column Parallel 输出形状错误！"
        assert diff_col < 1e-5, "Column Parallel 模拟结果与单卡全量计算不一致！"
        
        # 3. 模拟 2 张卡的 Row Parallel
        Y_row = tensor_parallel_row_sim(X, A, num_gpus=2)
        diff_row = torch.max(torch.abs(Y_ref - Y_row))
        print(f"Row Parallel 最大误差: {diff_row.item():.6e}")
        assert Y_row.shape == Y_ref.shape, "Row Parallel 输出形状错误！"
        assert diff_row < 1e-5, "Row Parallel 模拟结果与单卡全量计算不一致！"
        
        # 4. 维度约束检查
        try:
            tensor_parallel_column_sim(X, A[:, :30], num_gpus=2)
            raise AssertionError("Column Parallel 应该要求输出维度可整除")
        except AssertionError:
            pass
        
        try:
            tensor_parallel_row_sim(X[:, :15], A[:15], num_gpus=2)
            raise AssertionError("Row Parallel 应该要求输入维度可整除")
        except AssertionError:
            pass
        
        print("✅ Column Parallel (列切分) 矩阵计算与拼接逻辑正确！")
        print("✅ Row Parallel (行切分) 矩阵计算与求和逻辑正确！")
        print("掌握了 Megatron-LM 的核心张量切分思路，单卡装不下的大规模参数量再也不是问题。")
        
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError, RuntimeError) as e:
        if isinstance(e, AttributeError):
            print("代码未完成，无法找到必要的属性")
        elif isinstance(e, NameError):
            print("代码可能未完成，导致了变量未定义")
        elif isinstance(e, TypeError):
            print("代码可能未完成，导致了类型错误")
        elif isinstance(e, ValueError):
            print("代码可能未完成，导致了张量维度错误")
        elif isinstance(e, AssertionError):
            print("代码可能未完成，导致了断言失败")
        else:
            print("代码可能未完成，导致了运行时错误")
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise

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
    # TODO 1: 权重切分 (Scatter)
    in_features, out_features = A.shape
    chunk_size = out_features // num_gpus
    a_chunks = []
    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        a_chunk = A[:, start_idx:end_idx]
        a_chunks.append(a_chunk)
    
    # TODO 2: 独立计算 (Local MatMul)
    y_chunks = []
    for i in range(num_gpus):
        a_local = a_chunks[i]
        y_local = X @ a_local
        y_chunks.append(y_local)
        
    # TODO 3: 结果合并 (All-Gather)
    Y_tp = torch.cat(y_chunks, dim=-1)
    return Y_tp


def tensor_parallel_row_sim(X, A, num_gpus):
    # TODO 4: 输入和权重切分 (Scatter)
    in_features, out_features = A.shape
    chunk_size = in_features // num_gpus
    x_chunks = []
    a_chunks = []
    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        x_chunk = X[:, start_idx:end_idx]
        a_chunk = A[start_idx:end_idx, :]
        x_chunks.append(x_chunk)
        a_chunks.append(a_chunk)
    
    # TODO 5: 独立计算 (Local MatMul)
    y_chunks = []
    for i in range(num_gpus):
        x_local = x_chunks[i]
        a_local = a_chunks[i]
        y_local = x_local @ a_local
        y_chunks.append(y_local)
        
    # TODO 6: 结果求和 (All-Reduce)
    Y_tp = torch.stack(y_chunks, dim=0).sum(dim=0)
    return Y_tp

```

### 解析

**1. TODO 1: Column Parallel 的权重切分 (Scatter)**
- **实现方式**：`a_chunks = torch.chunk(A, num_gpus, dim=1)`
- **关键点**：将权重沿输出特征维度切分，每张卡只保存一部分列分片
- **技术细节**：Column Parallel 的核心是切分权重，而不是切分输入

**2. TODO 2: Column Parallel 的独立计算 (Local MatMul)**
- **实现方式**：`y_local = X @ a_local`
- **关键点**：输入 `X` 会被广播到所有卡，每张卡独立计算自己的输出分片
- **技术细节**：各卡计算得到的是输出特征的一部分，不是完整输出

**3. TODO 3: Column Parallel 的结果合并 (All-Gather)**
- **实现方式**：`Y_tp = torch.cat(y_chunks, dim=-1)`
- **关键点**：将各卡输出沿特征维拼接，恢复完整输出
- **技术细节**：这一步对应张量并行中的 All-Gather / 拼接操作

**4. TODO 4: Row Parallel 的输入和权重切分 (Scatter)**
- **实现方式**：`x_chunks = torch.chunk(X, num_gpus, dim=1)`，`a_chunks = torch.chunk(A, num_gpus, dim=0)`
- **关键点**：Row Parallel 同时切输入和权重，分别对应输入特征和权重行
- **技术细节**：这一步是 Row Parallel 与 Column Parallel 的核心差异之一

**5. TODO 5: Row Parallel 的独立计算 (Local MatMul)**
- **实现方式**：`y_local = x_local @ a_local`
- **关键点**：每张卡只计算自己分片对应的部分输出
- **技术细节**：各卡结果是“部分和”，还不能直接作为最终输出

**6. TODO 6: Row Parallel 的结果求和 (All-Reduce)**
- **实现方式**：`Y_tp = torch.stack(y_chunks, dim=0).sum(dim=0)`
- **关键点**：将各卡结果按元素相加，恢复完整输出
- **技术细节**：这一步对应张量并行中的 All-Reduce (Sum)

**工程要点**
- **通信特点**：Column Parallel 需要广播输入、合并输出；Row Parallel 需要切分输入、最后求和
- **适用场景**：Column Parallel 更适合扩维层，Row Parallel 更适合缩维层
- **组合方式**：在两层 MLP 中常见 Column -> Row 的组合，可以减少中间通信
