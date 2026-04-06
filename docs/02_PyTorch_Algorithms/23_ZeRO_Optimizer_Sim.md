# 23 ZeRO Optimizer Sim

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/23_ZeRO_Optimizer_Sim.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 23. 显存优化进阶：模拟 ZeRO-1 切分与 ZeRO 原理 (ZeRO Optimizer)

**难度：** Hard | **标签：** `Distributed Training`, `ZeRO`, `Memory Bound` | **目标人群：** 核心 Infra 与算子开发

在 01_Hardware 章节中，我们计算过：对于 AdamW 优化器，它需要保存梯度 (Grad)、一阶动量 (Momentum) 和二阶动量 (Variance)。这意味着在 FP16 混合精度训练下，**优化器状态和梯度占据了绝大部分显存 (至少是模型权重的 6 倍以上)**。
Microsoft DeepSpeed 提出的 ZeRO (Zero Redundancy Optimizer) 是打破单卡显存壁垒的关键。其中，**ZeRO-1** 仅仅将**优化器状态 (Optimizer States)** 均匀切分到各个 GPU 上，就极大地缓解了显存压力。
本节我们将用纯 PyTorch 模拟 2 张卡的 ZeRO-1 工作原理。


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

### Step 1: 代码实现框架
我们需要模拟一个分布式环境（可以简单用 Python 列表或字典代替不同的 GPU 节点）。在优化器初始化时，收集所有的模型参数，并将它们的 FP32 梯度和 Optimizer State（例如 Adam 的 Momentum 和 Variance）切分成 N 块，分别存放在对应的 GPU 上。在更新时，每张卡只负责计算自己那一小块的参数更新。


###  Step 2: 动手实战

**要求**：请补全下方 `ZeRO1_Optimizer_Sim`。我们将模拟在单机上用逻辑 Tensor 划分来代替真实的跨卡通信 (All-Gather / Reduce-Scatter)。


```python
import torch
import torch.nn as nn

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
        
        # 1. 模拟将参数切分给不同的 GPU 负责
        # GPU 0 负责前半部分参数，GPU 1 负责后半部分参数
        # (实际实现中是把展平的 1D 张量切片，这里我们用更简单的方式：按层分配)
        half_idx = len(self.params) // 2
        self.gpu_partitions = {
            0: self.params[:half_idx],  # GPU 0 负责的参数
            1: self.params[half_idx:]   # GPU 1 负责的参数
        }
        
        # 2. 为每个 GPU 只初始化它所负责参数的优化器状态 (这里用 SGD with Momentum 举例)
        self.optimizer_states = {
            0: {id(p): torch.zeros_like(p.data) for p in self.gpu_partitions[0]},
            1: {id(p): torch.zeros_like(p.data) for p in self.gpu_partitions[1]}
        }
        
    def step(self, gradients_from_all_gpus: dict):
        """
        gradients_from_all_gpus 模拟了 Reduce-Scatter 的结果。
        结构：{gpu_id: [它负责的参数的平均梯度]}
        """
        # ==========================================
        # TODO 1: 模拟每张卡只更新自己负责的那部分权重
        # 遍历每个 gpu_id，拿到它负责的 params 和 gradients
        # 对于每个参数：
        #   1. 更新该参数对应的动量 (momentum = 0.9 * momentum + grad)
        #   2. 使用动量更新该参数的值 (param = param - lr * momentum)
        # ==========================================
        for gpu_id in range(self.num_gpus):
            # params = self.gpu_partitions[gpu_id]
            # grads = gradients_from_all_gpus[gpu_id]
            # states = self.optimizer_states[gpu_id]
            
            # for p, g in zip(params, grads):
            #    momentum = states[id(p)]
            #    momentum = 0.9 * momentum + g
            #    states[id(p)] = momentum
            #    p.data = p.data - self.lr * momentum
            pass
            
        # ==========================================
        # TODO 2: 模拟 All-Gather (同步)
        # 实际中，每张卡更新完自己的那 1/N 权重后，需要把新权重广播给所有卡。
        # 在我们的单机模拟中，因为修改的是 p.data 原位引用，相当于自动完成了 All-Gather。
        # 你不需要写代码，只需理解这一步在真实多卡环境中的必要性。
        # ==========================================
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
        print("🔥 你成功模拟了 DeepSpeed 的核心技术！这让你理解了为什么切分优化器状态能让你用同样的卡训更大的模型。")
        
    except NotImplementedError:
        print("请先完成 TODO 代码！")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

test_zero1_sim()

```

### Step 4: ZeRO 家族的终极进化 (ZeRO-1 vs ZeRO-2 vs ZeRO-3)

在上面的代码中，我们成功模拟了 ZeRO-1。但微软 DeepSpeed 团队在此基础上进行了进一步的演进。为了训练千亿甚至万亿参数的模型，他们提出了完整的 ZeRO 三部曲。

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
- **原理**：最为彻底的切分方案。每张卡不再保留完整的模型参数：平时只存自己负责的那一小块参数。当前向/反向传播走到某一层时，通过 `All-Gather` 从其他卡把那一层的参数**临时广播过来**，计算完成后即刻释放：
- **通信变化**：牺牲了高达 50% 的额外通信量（因为要在前反向频繁 All-Gather），换取了显著提升的模型容量上限。
- **显存占用**：$\frac{16\Phi}{N}$。通过增加计算节点，可以支持更大规模的模型训练。

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

::: details 💡 点击查看官方解析与参考代码

ZeRO-1 将优化器状态在多个 GPU 之间切分，是分布式训练的主流策略之一。在实现中，每个设备只保存一部分优化器状态参数更新，然后在通信阶段（All-Gather）将更新后的权重同步回所有设备。

```python
class ZeRO1Optimizer:
    def __init__(self, optimizer, world_size, rank):
        self.optimizer = optimizer
        self.world_size = world_size
        self.rank = rank
        
    def step(self):
        self.optimizer.step()
        
    def zero_grad(self):
        self.optimizer.zero_grad()
        
    def consolidate_state_dict(self):
        return self.optimizer.state_dict()
```

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
