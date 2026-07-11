# 27. ZeRO Optimizer Sim | ZeRO 优化器模拟

**难度：** Hard | **环境：** CPU-first | **标签：** `分布式训练`, `ZeRO`, `显存优化` | **目标人群：** 分布式训练工程师

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/27_ZeRO_Optimizer_Sim.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


先把优化器状态切分和通信代价看清，再看 ZeRO 的分层策略会更容易理解显存优化的本质。

**关键词：** `ZeRO`, `Reduce-Scatter`, `All-Gather`

## 前置阅读

**导语：** 先看量化与推理优化，再进入 ZeRO 会更容易理解显存切分的意义。

- [11. PyTorch Optimizers and Loss | PyTorch 优化器与损失函数](../00_Prerequisites/11_PyTorch_Optimizers_and_Loss.md)
- [13. Simple Neural Network Training | 简单神经网络训练循环](../00_Prerequisites/13_Simple_Neural_Network_Training.md)
- [20. Profiling and Memory Ledger | 性能剖析与显存账本](../00_Prerequisites/20_Profiling_and_Memory_Ledger.md)


## 相关阅读

**导语：** ZeRO 后，建议继续看 Pipeline 和 Tensor Parallelism。

- [05. Communication Topologies | 通信拓扑与分布式基石](../01_Hardware_Math_and_Systems/05_Communication_Topologies.md)
- [17. CUDA Stream and Asynchrony | CUDA Stream 与异步执行](../01_Hardware_Math_and_Systems/17_CUDA_Stream_and_Asynchrony.md)
- [26. Parallel Strategy Decision Framework | 并行策略决策框架](../01_Hardware_Math_and_Systems/26_Parallel_Strategy_Decision_Framework.md)

### Step 1: ZeRO-1 核心思想

> **传统的 Data Parallel (DP，数据并行)：**
> 每张卡都有一份完整的模型权重、完整的梯度、完整的优化器状态。
> 各个卡算完自己这批数据的梯度后，进行 `All-Reduce` 求平均。然后每张卡用自己的优化器更新自己完整的权重。
> **痛点：严重浪费！每张卡都在重复保存一样的优化器状态和重复做一样的参数更新。**

> **ZeRO-1 的机制：**
> 1. 每张卡依然有完整的模型权重和完整的梯度（前向和反向传播与 DP 完全一样）。
> 2. **切分：** 优化器状态被切分成 $N$ 份（假设有 $N$ 张卡），每张卡只负责维护 $\frac{1}{N}$ 的优化器状态，并只负责更新这 $\frac{1}{N}$ 的模型权重。
> 3. **通信：** 反向传播结束后，不需要对所有梯度做 `All-Reduce`，而是做 `Reduce-Scatter`，让每张卡只拿到属于自己那 $\frac{1}{N}$ 权重的平均梯度。
> 4. 每张卡更新自己负责的 $\frac{1}{N}$ 权重后，再通过 `All-Gather` 将更新后的片段广播给所有卡，拼合出完整的新权重。
### Step 2: 代码实现框架
我们需要模拟一个分布式环境（可以简单用 Python 列表或字典代替不同的 GPU 节点）。在优化器初始化时，收集所有的模型参数，并将它们的 FP32 梯度和 Optimizer State（例如 Adam 的 Momentum 和 Variance）切分成 N 块，分别存放在对应的 GPU 上。在更新时，每张卡只负责计算自己那一小块的参数更新。

###  Step 3: 动手实战

**要求**：请补全下方 `ZeRO1_Optimizer_Sim`。我们将模拟在单机上用逻辑 Tensor 划分来代替真实的跨卡通信 (All-Gather / Reduce-Scatter)。


```python
import torch
import torch.nn as nn
```


```python
class SimpleModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 为了演示切分，我们用一个包含偶数个参数的线性层
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.fc2 = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

class ZeRO1_Optimizer_Sim:
    """
    模拟 2 张 GPU 上的 ZeRO-1 优化器行为。
    为了简化，我们假设模型的所有参数都被展平 (Flatten) 为一个一维张量，然后均分为 2 份。
    """
    def __init__(self, model_params, lr=0.1, num_gpus=2):
        self.lr = lr
        self.num_gpus = num_gpus
        
        # 将所有参数的引用收集起来
        self.params = list(model_params)
        
        # ==========================================
        # TODO 1: 将参数切分给不同的 GPU 负责
        # 提示: GPU 0 负责前半部分参数，GPU 1 负责后半部分参数
        # half_idx = ???
        # self.gpu_partitions = ???
        # ==========================================
        pass

        # ==========================================
        # TODO 2: 为每个 GPU 初始化局部状态
        # 提示: 只为各自负责的参数保存动量，初始值全为 0
        # self.optimizer_states = ???
        # ==========================================
        pass

        
    def step(self, gradients_from_all_gpus: dict):
        """
        gradients_from_all_gpus 模拟了 Reduce-Scatter 的结果。
        结构：{gpu_id: [它负责的参数的平均梯度]}
        """
        # ==========================================
        # TODO 3: 模拟每张卡只更新自己负责的那部分权重
        # 对于每个 gpu_id:
        #   params = ???
        #   grads = ???
        #   states = ???
        # 对于每个参数：
        #   momentum = ???
        #   states[id(p)] = ???
        #   p.data = ???
        # ==========================================
        for gpu_id in range(self.num_gpus):
            pass
            
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

```


```python
# 测试你的实现
def test_zero1_sim():
    try:
        torch.manual_seed(42)
        model = SimpleModel(dim=4)
        optimizer = ZeRO1_Optimizer_Sim(model.parameters(), lr=0.1, num_gpus=2)
        
        # 保存初始权重用于对比
        initial_w1 = model.fc1.weight.data.clone()
        initial_w2 = model.fc2.weight.data.clone()
        
        # 模拟反向传播产生的平均梯度 (Reduce-Scatter 的结果)
        # 假设 fc1 是 GPU 0 负责，fc2 是 GPU 1 负责
        simulated_reduce_scatter_grads = {
            0: [torch.ones_like(model.fc1.weight)],  # GPU 0 收到 fc1 的梯度
            1: [torch.full_like(model.fc2.weight, 2.0)] # GPU 1 收到 fc2 的梯度
        }
        
        # 验证优化器状态切分 (ZeRO-1 的核心显存节约)
        assert len(optimizer.optimizer_states[0]) == 1, "GPU 0 应该只维护 fc1 的状态"
        assert len(optimizer.optimizer_states[1]) == 1, "GPU 1 应该只维护 fc2 的状态"
        
        # 执行更新
        optimizer.step(simulated_reduce_scatter_grads)
        
        # 验证更新结果是否正确应用到了原模型上 (隐式的 All-Gather)
        diff_w1 = initial_w1 - model.fc1.weight.data
        diff_w2 = initial_w2 - model.fc2.weight.data
        
        # 预期：momentum 从 0 变成 1，w1 减去 lr * 1 = 0.1
        # 预期：momentum 从 0 变成 2，w2 减去 lr * 2 = 0.2
        assert torch.allclose(diff_w1, torch.full_like(diff_w1, 0.1)), "GPU 0 负责的权重更新错误！"
        assert torch.allclose(diff_w2, torch.full_like(diff_w2, 0.2)), "GPU 1 负责的权重更新错误！"
        
        print("✅ ZeRO-1 优化器状态切分与更新逻辑测试通过！")
        
    except NotImplementedError:
        print("请先完成 TODO 代码！")
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
        raise NotImplementedError("请先完成 TODO 代码！") from e
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise

test_zero1_sim()
```

### Step 4: ZeRO 家族的终极进化 (ZeRO-1 vs ZeRO-2 vs ZeRO-3)

在上面的代码中，上述代码模拟了 ZeRO-1。但微软 DeepSpeed 团队在此基础上进行了进一步的演进。为了训练千亿甚至万亿参数的模型，他们提出了完整的 ZeRO 三部曲。

这是分布式训练领域的核心知识点：

> **模型训练的显存大头（以 FP16 混合精度 + Adam 为例）：**
> 假设模型参数量为 $\Phi$。
> 1. **模型参数 (FP16)**：$2\Phi$ bytes
> 2. **梯度 (FP16)**：$2\Phi$ bytes
> 3. **优化器状态 (FP32)**：包含 FP32的参数备份 ($4\Phi$) + FP32的一阶动量 ($4\Phi$) + FP32的二阶动量 ($4\Phi$) = $12\Phi$ bytes
> **总计：至少 $16\Phi$ 的显存（还不算 Activation）。**

#### ZeRO-1：切分优化器状态 ($P_{os}$)
- **原理**：正如我们刚刚手写的，只把 $12\Phi$ 的优化器状态切分到 $N$ 张卡上。
- **显存占用**：$2\Phi + 2\Phi + \frac{12\Phi}{N}$。通信量与数据并行 (DP) 完全相同。

#### ZeRO-2：切分梯度 ($P_{os+g}$)
- **原理**：在 ZeRO-1 的基础上，进一步切分梯度。既然第 $i$ 张卡只负责更新第 $i$ 块参数，那么在反向传播算完梯度后，它**只需要保留第 $i$ 块的梯度**，把其他的梯度直接扔掉释放显存！
- **通信变化**：传统的 DP 在反向传播后做 `All-Reduce`（求和并广播）。ZeRO-2 改为 `Reduce-Scatter`（求和但只把各自负责的那块梯度发给对应的卡）。
- **显存占用**：$2\Phi + \frac{2\Phi}{N} + \frac{12\Phi}{N}$。

#### ZeRO-3：切分参数 ($P_{os+g+p}$)
- **原理**：最为完全的切分方案。每张卡不再保留完整的模型参数：平时只存自己负责的那一小块参数。当前向/反向传播走到某一层时，通过 `All-Gather` 从其他卡把那一层的参数**临时广播过来**，计算完成后即刻释放：
- **通信变化**：牺牲了高达 50% 的额外通信量（因为要在前反向频繁 All-Gather），换取了显著提升的模型容量上限。
- **显存占用**：$\frac{16\Phi}{N}$。通过增加计算节点，可以支持更大规模的模型训练。
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
class SimpleModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.fc2 = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

class ZeRO1_Optimizer_Sim:
    """
    模拟 2 张 GPU 上的 ZeRO-1 优化器行为。
    """
    def __init__(self, model_params, lr=0.1, num_gpus=2):
        self.lr = lr
        self.num_gpus = num_gpus
        
        # 将所有参数的引用收集起来
        self.params = list(model_params)
        
        # TODO 1: 将参数切分给不同的 GPU 负责
        half_idx = len(self.params) // 2
        self.gpu_partitions = {
            0: self.params[:half_idx],
            1: self.params[half_idx:]
        }
        
        # TODO 2: 为每个 GPU 初始化局部状态
        self.optimizer_states = {
            0: {id(p): torch.zeros_like(p.data) for p in self.gpu_partitions[0]},
            1: {id(p): torch.zeros_like(p.data) for p in self.gpu_partitions[1]}
        }
        
    def step(self, gradients_from_all_gpus: dict):
        """
        gradients_from_all_gpus 模拟了 Reduce-Scatter 的结果。
        """
        # TODO 3: 模拟每张卡只更新自己负责的那部分权重
        for gpu_id in range(self.num_gpus):
            params = self.gpu_partitions[gpu_id]
            grads = gradients_from_all_gpus[gpu_id]
            states = self.optimizer_states[gpu_id]
            
            for p, g in zip(params, grads):
                # 更新动量
                momentum = states[id(p)]
                momentum = momentum + g  # 简化版：直接累加梯度作为动量
                states[id(p)] = momentum
                
                # 使用动量更新参数
                p.data = p.data - self.lr * momentum
                
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

```

### 解析

**1. TODO 1: 参数切分**
- **实现方式**：`half_idx = len(self.params) // 2`，`self.gpu_partitions = {0: self.params[:half_idx], 1: self.params[half_idx:]}`
- **关键点**：将模型参数均分给不同的 GPU，每个 GPU 只负责一部分参数的优化
- **技术细节**：使用字典存储每个 GPU 负责的参数列表

**2. TODO 2: 初始化局部优化器状态**
- **实现方式**：`self.optimizer_states = {0: {id(p): torch.zeros_like(p.data) for p in self.gpu_partitions[0]}, 1: {id(p): torch.zeros_like(p.data) for p in self.gpu_partitions[1]}}`
- **关键点**：每个 GPU 只维护自己负责的参数的优化器状态（如动量、方差）
- **技术细节**：使用参数的 `id()` 作为键，避免参数对象本身作为字典键的问题

**3. TODO 3: 分布式参数更新**
- **实现方式**：遍历每个 GPU，只更新该 GPU 负责的参数
- **关键点**：每个 GPU 只计算和更新自己负责的那部分参数，避免重复计算
- **技术细节**：`momentum = momentum + g`，`p.data = p.data - self.lr * momentum`

**ZeRO 系列对比**

| 方案 | 切分内容 | 显存占用 | 通信开销 | 适用场景 |
|------|---------|---------|---------|---------|
| **Data Parallel** | 无切分 | $16\Phi$ | All-Reduce | 小模型 |
| **ZeRO-1** | 优化器状态 | $2\Phi + 2\Phi + \frac{12\Phi}{N}$ | All-Reduce | 中等模型 |
| **ZeRO-2** | 优化器状态 + 梯度 | $2\Phi + \frac{14\Phi}{N}$ | Reduce-Scatter | 大模型 |
| **ZeRO-3** | 优化器状态 + 梯度 + 参数 | $\frac{16\Phi}{N}$ | All-Gather (高) | 超大模型 |

**工程优化要点**
- **显存节省**：ZeRO-1 将优化器状态显存从 $12\Phi$ 降至 $\frac{12\Phi}{N}$，N=8 时节省 87.5%
- **通信模式**：Reduce-Scatter（求和并分发）+ All-Gather（收集并广播）
- **权衡**：ZeRO-3 显存最优但通信开销最大，需要高速互联（如 NVLink、InfiniBand）
- **混合策略**：可以在节点内用 ZeRO-3，节点间用 ZeRO-2，平衡显存和通信
- **工业实践**：DeepSpeed 支持 ZeRO-1/2/3 自动切换，训练 GPT-3（175B）使用 ZeRO-3
