# 27. ZeRO Optimizer Sim | ZeRO 优化器模拟

**难度：** Hard | **环境：** CPU-first | **标签：** `并行通信`, `ZeRO`, `参数切分` | **目标人群：** 并行通信学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/27_ZeRO_Optimizer_Sim.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

数据并行能把 batch 分到多张卡上计算，但它有一个明显浪费：每张卡都保存完整参数、完整梯度和完整优化器状态。模型一大，真正压垮显存的往往不是前向本身，而是这些训练状态在每张卡上的重复存储。

ZeRO 将重复训练状态分给不同 GPU 维护，并按阶段逐步切分优化器状态、梯度和参数。它可以与 Pipeline、Tensor 或 Expert Parallelism 组合使用。本节先用同一组参数编号比较 Data Parallel、ZeRO-1、ZeRO-2 与 ZeRO-3 的状态归属，再沿一次 ZeRO-1 更新追踪局部状态如何参与参数更新。完成后，你应能解释每个阶段省下了什么，以及额外需要什么通信。

**关键词：** `ZeRO`, `optimizer state`, `gradient synchronization`

---

## 前置阅读

**导语：** 进入本节前，先能区分参数、梯度和优化器状态，并能从显存账本中看出它们为什么会在每张 GPU 上重复保存。

- [P0: 11. PyTorch Optimizers and Loss | PyTorch 优化器与损失函数](../00_Prerequisites/11_PyTorch_Optimizers_and_Loss.md)
- [P0: 18. Memory Profiling and Optimization | 显存分析与优化](../00_Prerequisites/18_Memory_Profiling_and_Optimization.md)
- [P1: 20. NCCL and AllReduce Basics | NCCL 与 AllReduce 基础](../01_Hardware_Math_and_Systems/20_NCCL_and_AllReduce_Basics.md)


---
### Step 1：从重复训练状态理解 ZeRO 分阶段

Data Parallel 中，每张卡都保存完整参数、梯度和优化器状态；梯度同步后，每张卡仍重复维护同一份训练状态。ZeRO 把这些重复副本逐步改为分片：ZeRO-1 先切优化器状态，ZeRO-2 再切梯度，ZeRO-3 最后连参数也切分。

同一个参数在不同阶段的“谁保存、谁更新、何时临时聚合”会改变。下面的主图先给出状态从复制到分片的全景；Step 2 再沿一次更新解释这些状态如何流动。

![ZeRO：把训练状态从每卡复制改成分片](../public/02_PyTorch_Algorithms/27_zero_sharding.svg)

### Step 2：沿一次更新追踪状态与通信

一次训练更新包含前向、反向、梯度同步和参数更新。三种 ZeRO 阶段的差异不在数学上的梯度公式，而在每一步中哪些张量保留副本、哪些张量由 owner rank 保存。

- **ZeRO-1**：每张卡仍可见完整参数和梯度；优化器状态只在 owner rank 保存并更新。
- **ZeRO-2**：梯度也只保留在 owner rank，通常通过 Reduce-Scatter 将同步后的梯度直接送到对应 owner。
- **ZeRO-3**：参数同样按 owner 切分；某一层参与计算前，才临时收集这一层需要的参数。

因此，切分越深入，单卡状态越少，但参数收集与释放越频繁；这也是 ZeRO-3 对链路和调度更敏感的原因。

### Step 3：用显存账本选择 ZeRO 阶段

下面用同一份 FP16 + Adam 账本比较四种状态布局。选择不只看“节省多少显存”：当参数本身放不下时才需要 ZeRO-3；若只是不想复制 Adam 状态，ZeRO-1 往往拥有更低的通信复杂度。

以模型参数量 $\Phi$ 为单位，参数占 $2\Phi$ bytes、梯度占 $2\Phi$ bytes、Adam 优化器状态占 $12\Phi$ bytes。此处把 ZeRO-1/2/3 的状态账本和通信代价放在同一张表中，避免与解析区重复。

| 策略 | 分片对象 | 单卡状态（近似） | 主要通信 | 适用信号 |
|---|---|---:|---|---|
| Data Parallel | 无分片 | $16\Phi$ | All-Reduce 梯度 | 模型和训练状态都能放下 |
| ZeRO-1 | 优化器状态 | $2\Phi + 2\Phi + 12\Phi/N$ | 局部梯度分发与参数同步 | 优化器状态主导显存 |
| ZeRO-2 | 优化器状态 + 梯度 | $2\Phi + 14\Phi/N$ | 梯度 Reduce-Scatter | 梯度也成为压力 |
| ZeRO-3 | 优化器状态 + 梯度 + 参数 | $16\Phi/N$ | 按层 All-Gather，通信更频繁 | 参数本身已经放不下 |

当 $N=8$ 时，ZeRO-1 只把优化器状态部分降为 $1/8$；ZeRO-3 的单卡状态最低，但更依赖高速互联和稳定的按层参数聚合。Step 4 会把这张账本落到可检查的参数 owner 布局。

![ZeRO 分阶段：省下的状态对应新增的通信](../public/02_PyTorch_Algorithms/27_zero_stages.svg)

### Step 4：CPU 实现——比较 ZeRO 分片布局与局部更新

本题先用参数编号建立 DP、ZeRO-1、ZeRO-2、ZeRO-3 的 owner 布局，再用两个逻辑 rank 观察 ZeRO-1 如何只保留局部优化器状态。骨架已提供参数、分片和梯度输入；你将补全状态归属、owner 初始化和局部更新。

测试区分别检查分片是否完整覆盖、优化器状态是否只属于 owner，以及同步后的局部梯度是否只更新对应参数。

| 任务与目的 | 已知条件 | 完成标准 | 测试验证 |
| --- | --- | --- | --- |
| TODO 1：建立阶段 owner。看清 ZeRO 每一阶段多切分了哪类训练状态。 | `num_params`、`num_ranks`、`stage`、`replicated / sharded` 布局 | 参数、梯度、优化器状态按 DP / ZeRO-1 / 2 / 3 规则归属；分片完整且不重叠 | 各阶段三类 owner 的长度与覆盖范围 |
| TODO 2：分配参数 owner。让每个逻辑 rank 只负责一段参数。 | `model_params`、`num_gpus`、`params_per_rank` | 每个参数恰好属于一个 `gpu_partitions[rank]` | 参数分片覆盖与重复归属检查 |
| TODO 3：初始化局部优化器状态。体现 ZeRO-1 节省的是优化器状态副本。 | `gpu_partitions`、参数对象 `id`、零初始化状态 | 每个 rank 只为本地参数建立状态 | 每张逻辑卡的状态字典大小 |
| TODO 4：执行局部更新。把同步后的局部梯度写回对应 owner。 | `gradients_from_all_gpus`、`states`、`lr` | 只更新 owner 参数及其状态；梯度数量必须匹配局部参数 | 更新幅度、状态值与梯度数量边界 |


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

def build_zero_stage_layout(num_params, num_ranks, stage):
    """
    返回不同 ZeRO 阶段中参数、梯度和优化器状态的逻辑 owner。

    stage 可取 'dp'、'zero1'、'zero2' 或 'zero3'。每个 owner 列表保存
    参数编号；它描述状态归属，不模拟真实 GPU 张量或 collective。
    """
    if num_params <= 0 or num_ranks <= 0 or num_params % num_ranks != 0:
        raise ValueError('参数数量与 rank 数必须为正，且参数数量能被 rank 数整除')
    if stage not in {'dp', 'zero1', 'zero2', 'zero3'}:
        raise ValueError('stage 必须是 dp、zero1、zero2 或 zero3')

    param_ids = list(range(num_params))
    replicated = {rank: param_ids[:] for rank in range(num_ranks)}
    shard_size = num_params // num_ranks
    sharded = {
        rank: param_ids[rank * shard_size:(rank + 1) * shard_size]
        for rank in range(num_ranks)
    }

    # ==========================================
    # TODO 1：按 ZeRO 阶段返回三类训练状态的 owner。
    # 变量提示（每个变量各占一行）：
    # parameter_owner = replicated 或 sharded
    # gradient_owner = replicated 或 sharded
    # optimizer_owner = replicated 或 sharded
    # return {'parameters': ..., 'gradients': ..., 'optimizer_states': ...}
    # ==========================================
    pass

class ZeRO1_Optimizer_Sim:
    """
    模拟多个逻辑 rank 上的 ZeRO-1 优化器状态分片。

    每个参数对象只归属于一个 rank；本类接收的局部梯度已经完成同步。
    """
    def __init__(self, model_params, lr=0.1, num_gpus=2):
        self.lr = lr
        self.num_gpus = num_gpus
        
        # 保留参数对象引用；这个 CPU 模拟不复制真实 GPU 张量。
        self.params = list(model_params)
        
        if self.num_gpus <= 0 or len(self.params) % self.num_gpus != 0:
            raise ValueError('num_gpus 必须为正，且参数对象数量必须能被它整除')
        params_per_rank = len(self.params) // self.num_gpus

        # ==========================================
        # TODO 2：按 rank 建立 ZeRO-1 参数 owner
        # 变量提示（每个变量各占一行）：
        # rank = ...
        # start = rank * params_per_rank
        # end = start + params_per_rank
        # self.gpu_partitions[rank] = self.params[start:end]
        # ==========================================
        self.gpu_partitions = {}
        for rank in range(self.num_gpus):
            pass

        # ==========================================
        # TODO 3：只为 owner 参数创建优化器状态
        # 变量提示（每个变量各占一行）：
        # rank = ...
        # local_params = self.gpu_partitions[rank]
        # self.optimizer_states[rank] = {id(p): torch.zeros_like(p.data) for p in local_params}
        # ==========================================
        self.optimizer_states = {}
        for rank in range(self.num_gpus):
            pass

        
    def step(self, gradients_from_all_gpus: dict):
        """
        local_gradients_by_rank 是同步后的局部梯度。
        结构：{rank: [该 rank 负责参数的梯度]}。
        """
        # ==========================================
        # TODO 4：完成 ZeRO-1 的局部状态与参数更新
        # 变量提示（每个变量各占一行）：
        # state = states[id(param)]
        # updated_state = state + grad
        # states[id(param)] = updated_state
        # param.data = param.data - self.lr * updated_state
        # ==========================================
        for rank in range(self.num_gpus):
            params = self.gpu_partitions[rank]
            grads = gradients_from_all_gpus[rank]
            states = self.optimizer_states[rank]
            if len(params) != len(grads):
                raise ValueError('局部梯度数量必须与该 rank 的参数数量一致')
            for param, grad in zip(params, grads):
                pass
            
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

```


```python
# 测试你的实现
def test_zero_stage_state_ownership():
    """不同 ZeRO 阶段只切分规定的训练状态。"""
    dp = build_zero_stage_layout(num_params=8, num_ranks=2, stage='dp')
    zero1 = build_zero_stage_layout(num_params=8, num_ranks=2, stage='zero1')
    zero2 = build_zero_stage_layout(num_params=8, num_ranks=2, stage='zero2')
    zero3 = build_zero_stage_layout(num_params=8, num_ranks=2, stage='zero3')

    assert all(len(items) == 8 for items in dp['optimizer_states'].values())
    assert all(len(items) == 8 for items in zero1['parameters'].values())
    assert all(len(items) == 8 for items in zero1['gradients'].values())
    assert all(len(items) == 4 for items in zero1['optimizer_states'].values())
    assert all(len(items) == 4 for items in zero2['gradients'].values())
    assert all(len(items) == 4 for items in zero3['parameters'].values())

def test_zero_stage_shards_cover_each_state_once():
    """被分片的状态必须完整覆盖，且任一编号不能重复归属。"""
    for stage, field in [('zero1', 'optimizer_states'), ('zero2', 'gradients'), ('zero3', 'parameters')]:
        layout = build_zero_stage_layout(num_params=8, num_ranks=2, stage=stage)
        owned = [item for items in layout[field].values() for item in items]
        assert sorted(owned) == list(range(8))
        assert len(owned) == len(set(owned))

def test_zero1_sim():
    """验证 ZeRO-1 的状态分片、局部更新和参数可见性。"""
    try:
        torch.manual_seed(42)
        model = SimpleModel(dim=4)
        optimizer = ZeRO1_Optimizer_Sim(model.parameters(), lr=0.1, num_gpus=2)
        
        # 保存初始权重用于对比
        initial_w1 = model.fc1.weight.data.clone()
        initial_w2 = model.fc2.weight.data.clone()
        
        # 模拟同步后分发给各 rank 的局部梯度。
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
        
        # 共享 Python 引用让测试可读取所有更新后的参数；这不模拟真实 All-Gather。
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

def test_zero1_partition_state():
    """每个 rank 只持有自己负责参数的优化器状态。"""
    model = SimpleModel(dim=4)
    optimizer = ZeRO1_Optimizer_Sim(model.parameters(), num_gpus=2)
    assert all(len(states) == 1 for states in optimizer.optimizer_states.values())

def test_zero1_local_update_value():
    """局部梯度只改变所属 rank 的参数，并写回局部状态。"""
    model = SimpleModel(dim=4)
    optimizer = ZeRO1_Optimizer_Sim(model.parameters(), lr=0.1, num_gpus=2)
    before = [p.detach().clone() for p in model.parameters()]
    grads = {0: [torch.ones_like(model.fc1.weight)], 1: [torch.ones_like(model.fc2.weight)]}
    optimizer.step(grads)
    deltas = [old - new for old, new in zip(before, model.parameters())]
    assert all(torch.allclose(delta, torch.full_like(delta, 0.1)) for delta in deltas)
    assert all(torch.allclose(state, torch.ones_like(state)) for states in optimizer.optimizer_states.values() for state in states.values())

def test_zero1_parameter_visibility():
    """参数分片必须完整覆盖，且任一参数不能同时属于两个 rank。"""
    model = SimpleModel(dim=4)
    optimizer = ZeRO1_Optimizer_Sim(model.parameters(), num_gpus=2)
    owned = [id(param) for params in optimizer.gpu_partitions.values() for param in params]
    assert len(owned) == len(set(owned)) == len(list(model.parameters()))

test_zero_stage_state_ownership()
test_zero_stage_shards_cover_each_state_once()
test_zero1_partition_state()
test_zero1_local_update_value()
test_zero1_parameter_visibility()
test_zero1_sim()
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
class SimpleModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.fc2 = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

def build_zero_stage_layout(num_params, num_ranks, stage):
    """返回 DP、ZeRO-1、ZeRO-2、ZeRO-3 的逻辑训练状态 owner。"""
    if num_params <= 0 or num_ranks <= 0 or num_params % num_ranks != 0:
        raise ValueError('参数数量与 rank 数必须为正，且参数数量能被 rank 数整除')
    if stage not in {'dp', 'zero1', 'zero2', 'zero3'}:
        raise ValueError('stage 必须是 dp、zero1、zero2 或 zero3')

    param_ids = list(range(num_params))
    replicated = {rank: param_ids[:] for rank in range(num_ranks)}
    shard_size = num_params // num_ranks
    sharded = {
        rank: param_ids[rank * shard_size:(rank + 1) * shard_size]
        for rank in range(num_ranks)
    }

    # TODO 1：ZeRO 每增加一个阶段，就多切分一类训练状态。
    parameter_owner = sharded if stage == 'zero3' else replicated
    gradient_owner = sharded if stage in {'zero2', 'zero3'} else replicated
    optimizer_owner = sharded if stage in {'zero1', 'zero2', 'zero3'} else replicated
    return {
        'parameters': parameter_owner,
        'gradients': gradient_owner,
        'optimizer_states': optimizer_owner,
    }

class ZeRO1_Optimizer_Sim:
    """
    模拟多个逻辑 rank 上的 ZeRO-1 优化器状态分片。
    """
    def __init__(self, model_params, lr=0.1, num_gpus=2):
        self.lr = lr
        self.num_gpus = num_gpus
        
        # 保留参数对象引用；这个 CPU 模拟不复制真实 GPU 张量。
        self.params = list(model_params)
        
        if self.num_gpus <= 0 or len(self.params) % self.num_gpus != 0:
            raise ValueError('num_gpus 必须为正，且参数对象数量必须能被它整除')
        params_per_rank = len(self.params) // self.num_gpus

        # TODO 2：按 rank 建立 ZeRO-1 参数 owner。
        self.gpu_partitions = {}
        for rank in range(self.num_gpus):
            start = rank * params_per_rank
            end = start + params_per_rank
            self.gpu_partitions[rank] = self.params[start:end]
        
        # TODO 3：只为 owner 参数初始化优化器状态。
        self.optimizer_states = {}
        for rank, local_params in self.gpu_partitions.items():
            self.optimizer_states[rank] = {id(p): torch.zeros_like(p.data) for p in local_params}
        
    def step(self, gradients_from_all_gpus: dict):
        """
        gradients_from_all_gpus 是同步后的局部梯度。
        """
        # TODO 4：每个 rank 只更新自己负责的参数。
        for rank in range(self.num_gpus):
            params = self.gpu_partitions[rank]
            grads = gradients_from_all_gpus[rank]
            states = self.optimizer_states[rank]
            
            for param, grad in zip(params, grads):
                state = states[id(param)]
                updated_state = state + grad  # 教学用累积状态，不等同于 Adam。
                states[id(param)] = updated_state
                param.data = param.data - self.lr * updated_state
                
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

```

### 解析

**TODO 1：比较分阶段状态 owner**
- Data Parallel 不切分三类状态；ZeRO-1 只切优化器状态；ZeRO-2 再切梯度；ZeRO-3 再切参数。
- `replicated` 与 `sharded` 是逻辑所有权账本：它们不执行 collective，但能验证每个阶段实际减少了哪一类副本。

**TODO 2、TODO 3：建立 ZeRO-1 的参数 owner 与局部状态**
- `params_per_rank` 把参数对象均匀分给逻辑 rank；每个对象必须恰好出现一次。
- 每个 rank 只为自己负责的参数创建状态张量；参数对象的 id 只用于把状态映射回参数对象。

**TODO 4：局部状态更新**
- `gradients_from_all_gpus` 被视为同步后交给每个 rank 的局部梯度。
- 每个 rank 更新自己的参数和教学用累积状态；共享 Python 引用只让测试读取完整参数。

### Step 5（可选）：GPU ZeRO 分片复测

先用 CPU 实验理解训练状态如何分片，再在同一 workload 下观察不同 ZeRO 阶段的实际显存与时间代价。

| 学习问题 | CPU 已确认 | GPU 要确认与后续使用 |
| --- | --- | --- |
| ZeRO 各阶段如何减少训练状态副本 | 参数、梯度、优化器状态的 owner、分片覆盖和局部更新 | 记录 DDP 与 ZeRO 各阶段的峰值显存和最慢 rank step time，用于固定 workload 下的策略判断 |

#### 5.1 环境与固定 workload

至少使用两张 GPU；以下配置使 DDP 与各 ZeRO 阶段使用同一训练口径。

| 类别 | 固定并记录的字段 | 用途 |
| --- | --- | --- |
| 运行环境 | GPU 数、DeepSpeed、PyTorch / CUDA、NCCL | 确认多卡分片路径可运行 |
| workload | 合成模型、dtype、global / micro batch、序列长度、warmup、repeats | 保持各阶段输入一致 |
| 对照 | DDP baseline、ZeRO-1 / 2 / 3 | 只比较状态分片策略 |
| 输出 | `RESULT_DIR`、每 stage 的 JSON、默认关闭开关 | 保留复测与 failure |

```python
# 5.1：默认关闭；先固定所有 ZeRO 阶段共用的模型、数据与训练口径。
RUN_GPU_EXPERIMENT = False
WORLD_SIZE = 2
MODEL_KIND = 'synthetic_transformer_mlp'
MODEL_WIDTH = 1024
MODEL_LAYERS = 4
DTYPE = 'bf16'
GLOBAL_BATCH_SIZE = 4
MICRO_BATCH_SIZE = 1
SEQ_LEN = 512
WARMUP = 2
REPEATS = 5
RESULT_DIR = 'benchmarks/results/27_zero'
STRATEGIES = ['ddp', 'zero1', 'zero2', 'zero3']
print({'run': RUN_GPU_EXPERIMENT, 'strategies': STRATEGIES, 'world_size': WORLD_SIZE, 'model': MODEL_KIND, 'dtype': DTYPE, 'result_dir': RESULT_DIR})

```

#### 5.2 执行并保存 JSON

使用 DeepSpeed 在同一个合成 Transformer-shaped workload 上分别运行 DDP、ZeRO-1、ZeRO-2 与 ZeRO-3，并为每种策略保存一份 JSON。结果将把状态分片带来的每 rank 显存、最慢 rank 时间和运行状态放在同一份证据中。

```python
# 5.2：依次运行真实 DeepSpeed stage；默认关闭。
if RUN_GPU_EXPERIMENT:
    import subprocess
    from pathlib import Path
    Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)
    for strategy in STRATEGIES:
        output = str(Path(RESULT_DIR) / f'{strategy}.json')
        command = [
            'torchrun', '--standalone', '--nproc_per_node', str(WORLD_SIZE),
            'tools/run_zero_stage_benchmark.py', '--strategy', strategy, '--output', output,
            '--width', str(MODEL_WIDTH), '--layers', str(MODEL_LAYERS),
            '--micro-batch', str(MICRO_BATCH_SIZE), '--seq-len', str(SEQ_LEN),
            '--warmup', str(WARMUP), '--repeats', str(REPEATS), '--dtype', DTYPE,
        ]
        subprocess.run(command, check=False)  # stage failure 会写入各自 JSON。
else:
    print('GPU benchmark 默认关闭。')

```

#### 5.3 读取结果、解释指标与形成决策

读取四个 stage 的真实 JSON，先确认 workload、hardware、metrics、evidence_level、failure 和 decision 齐全；再结合每 rank 峰值显存、最慢 rank 的 step time、吞吐代理和所用 GPU 数，选择下一轮 ZeRO 策略。

```python
# 5.3：汇总各 stage 的真实结果；没有 JSON 时不输出性能结论。
import json
from pathlib import Path
results = {}
required = {'strategy', 'workload', 'hardware', 'metrics', 'evidence_level', 'failure', 'decision'}
for strategy in STRATEGIES:
    path = Path(RESULT_DIR) / f'{strategy}.json'
    if not path.exists():
        continue
    result = json.loads(path.read_text())
    missing = required - set(result)
    if missing:
        raise ValueError(f'{strategy} 结果 JSON 缺少字段：{sorted(missing)}')
    results[strategy] = result
print(results if results else f'尚无真实 ZeRO 结果：{RESULT_DIR}')

```

| 读取项 | 判断与下一步 |
| --- | --- |
| 运行状态 | `failure` 非空时记录失败原因，不比较性能指标 |
| 可比口径 | `workload / hardware` 必须与对照一致，否则重新运行 |
| 核心指标 | 比较每 rank 峰值显存、最慢 rank `step_time` 与吞吐代理 |
| 结论 | 用 `evidence_level / decision` 标明证据强度和下一轮 ZeRO 策略 |
## 相关阅读

ZeRO 的核心是训练状态分片；读完本节后，可以再用论文、DeepSpeed 实现和并行策略页面核对通信与显存取舍。

- [ZeRO 原论文：Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [DeepSpeed ZeRO 官方文档](https://www.deepspeed.ai/tutorials/zero/)
- [28. Pipeline 并行微批次](../02_PyTorch_Algorithms/28_Pipeline_Parallelism_MicroBatch.md)
- [29. Tensor 并行模拟](../02_PyTorch_Algorithms/29_Tensor_Parallelism_Sim.md)
- [P1: 并行策略决策框架](../01_Hardware_Math_and_Systems/26_Parallel_Strategy_Decision_Framework.md)
