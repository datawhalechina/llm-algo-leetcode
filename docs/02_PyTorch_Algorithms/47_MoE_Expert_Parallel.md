# 47. MoE Expert Parallel | MoE 专家并行

**难度：** Hard | **环境：** CPU-first | **标签：** `并行通信`, `MoE`, `Expert Parallel`, `All-to-All` | **目标人群：** 并行通信学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/47_MoE_Expert_Parallel.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

MoE 不是简单地“把参数变多”，它真正难的地方在于：token 被路由到不同专家后，专家往往分布在不同设备上，训练和推理都会引入额外通信、负载不均和 capacity 溢出。也就是说，MoE 一旦走到多设备阶段，它面对的就不再只是模型结构问题，而是一个典型的并行与通信问题。

06 解释 router 如何选专家，07 解释负载均衡损失如何约束选择；本节继续追踪路由结果进入多设备系统后的变化：token 如何被 dispatch、哪些 token 跨 rank、热点怎样造成 capacity 溢出，以及结果为何要再 combine 回原请求。

这一节比较不同专家放置与路由分布：同样数量的 token，均匀、本地命中和跨卡偏斜会产生不同的 overflow、通信字节和长尾风险。

**关键词：** `expert parallel`, `dispatch`, `all-to-all`

---

## 前置阅读

**导语：** 进入本节前，先能说明 Router 如何选择专家、负载均衡如何约束选择，再观察专家跨设备后如何产生 dispatch 和 all-to-all 通信。

- [06. MoE Router | MoE 路由器](./06_MoE_Router.md)
- [07. MoE Load Balancing Loss | MoE 负载均衡损失](./07_MoE_Load_Balancing_Loss.md)
- [46. Communication Profiling with NCCL | NCCL 通信 Profiling](./46_Communication_Profiling_with_NCCL.md)
---
### Step 1：从 MoE 路由到 Expert Parallel

MoE 的稀疏激活让每个 token 只计算少量专家。Expert Parallel 再把专家分给不同 rank：router 选出的专家若不在 token 的来源 rank，就必须先把 token 表示送过去；专家完成计算后，结果还要按原 token 顺序 combine。

这条链路同时引入三类变量：**路由分布**决定专家负载，**capacity**决定是否溢出，**专家放置**决定多少 token 需要跨卡 all-to-all。它们共同决定稀疏计算是否真正带来系统收益。

![MoE 专家并行总览](../public/02_PyTorch_Algorithms/47_expert_parallel_overview.svg)

### Step 2：路由、capacity 与 token dispatch

每个 token 的 `routes` 给出 top-k 专家候选。系统先统计 `expert_loads`，再依据 `capacity_factor` 为每个专家分配可接收路由数；超过 capacity 的部分成为 `overflow`。被接受的路由根据专家所属 rank 分成同 rank 与跨 rank 两类，后者才需要实际 dispatch。

因此，全部 route 的表示规模与真正跨 rank 的网络传输必须分开记录：前者是逻辑工作量，后者才取决于专家放置。图中的 dispatch 与 combine 分别是同一轮路由的去程和回程；若某个 expert 接收的 token 明显更多，会同时放大该 rank 的计算等待与通信长尾。

![MoE Dispatch 与 Combine 的数据流](../public/02_PyTorch_Algorithms/47_dispatch_combine_flow.svg)

### Step 3：比较路由分布与专家放置

设 `A=sum(len(route_i))` 为总路由次数。平均负载是 `A / num_experts`，单专家 capacity 是 `ceil((A / num_experts) × capacity_factor)`，超过 capacity 的路由构成 overflow。每条 token 表示的字节近似为 `hidden_size × bytes_per_elem`。

| 变化 | 首先改变什么 | 需要一起观察的量 |
|---|---|---|
| top-k 增大 | 总路由数与专家计算增加 | overflow、dispatch bytes、有效 token 吞吐 |
| 路由偏斜 | 热门专家与其 rank 成为长尾 | balance ratio、overflow、最长 rank 负载 |
| capacity factor 增大 | 更少 token 被拒绝 | 预留空间、padding 与计算浪费 |
| 专家放置更本地化 | 跨 rank dispatch 减少 | 本地命中率、跨卡 bytes、负载是否反而偏斜 |

### Step 4：CPU 机制实现——比较路由、容量与专家放置

题目区先从路由得到 expert load、capacity 与 overflow，再将专家映射到逻辑 rank，区分本地和跨 rank token。最后依据负载、overflow 和跨卡比例给出 `accept / tune / reject`。

| TODO | 函数 / 机制 | 输入与关键变量 | 输出与不变量 | 测试关注点 |
| --- | --- | --- | --- | --- |
| 1 | `summarize_expert_parallel`：路由账本 | `routes`、`num_experts`、`capacity_factor` | 返回负载、capacity、overflow 与逻辑 route 字节；专家编号必须合法 | 均匀、偏斜与空路由 |
| 2 | `summarize_dispatch_locality`：专家放置 | token 来源 rank、expert owner rank、`hidden_size` | 返回本地 / 跨 rank routes、去程与往返字节；两类 route 之和等于总 routes | 本地命中率与跨卡 bytes |
| 3 | `recommend_moe_parallel_plan`：并行决策 | overflow、`balance_ratio`、可选 locality | 返回 `accept / tune / reject`；负载与通信风险都参与判断 | 溢出、偏斜和跨卡比例对照 |

### 题目提示

- `routes` 里的每个元素表示一个 token 被路由到哪些专家；先把它展开统计成每个 expert 的负载。
- `TODO 1` 先统计路由与负载，再计算 capacity、overflow 和总通信账本。
- `capacity = ceil((total_routes / num_experts) * capacity_factor)`，`overflow` 只统计超出 capacity 的那部分路由数。
- `TODO 2` 根据 token 来源 rank 与专家 owner rank 统计本地/跨卡 dispatch；不要把所有路由都当成跨卡通信。
- `TODO 3` 用 overflow、负载偏斜和跨卡比例判断当前计划是否值得继续。

```python
from collections import Counter
from dataclasses import dataclass
import math
import torch

```


```python
@dataclass
class MoEParallelSummary:
    total_tokens: int
    total_routes: int
    expert_loads: list
    capacity: int
    overflow: int
    logical_route_bytes: int
    balance_ratio: float

@dataclass
class DispatchLocalitySummary:
    total_routes: int
    local_routes: int
    cross_rank_routes: int
    cross_rank_dispatch_bytes: int
    cross_rank_roundtrip_bytes: int
    local_hit_ratio: float


def summarize_expert_parallel(routes, num_experts, hidden_size, bytes_per_elem=2, capacity_factor=1.0):
    """
    TODO 1:
    汇总 expert parallel 的最小账本，返回 MoEParallelSummary。
    需要包含 expert_loads、capacity、overflow、logical_route_bytes、balance_ratio。
    """
    # 提示：先统计 total_tokens / total_routes，再得到 expert_loads。
    # 提示：接着补出 capacity、overflow、logical_route_bytes、balance_ratio。
    # total_tokens = ???
    # total_routes = ???
    # expert_loads = ???
    # capacity = ???
    # overflow = ???
    # logical_route_bytes = ???
    # balance_ratio = ???
    raise NotImplementedError


def summarize_dispatch_locality(routes, token_source_ranks, expert_owner_ranks, hidden_size, bytes_per_elem=2):
    """
    TODO 2：统计专家放置后的本地与跨 rank dispatch。

    ``token_source_ranks[i]`` 表示第 i 个 token 的来源 rank；
    ``expert_owner_ranks[e]`` 表示 expert e 所在的 rank。
    """
    if len(routes) != len(token_source_ranks):
        raise ValueError('每个 token 都需要一个来源 rank')
    local_routes = 0
    cross_rank_routes = 0
    # ==========================================
    # TODO 2：按来源 rank 与专家 owner rank 区分路由。
    # 变量提示（每个变量各占一行）：
    # source_rank = token_source_ranks[token_idx]
    # owner_rank = expert_owner_ranks[expert_id]
    # local_routes += 1 或 cross_rank_routes += 1
    # cross_rank_dispatch_bytes = cross_rank_routes * hidden_size * bytes_per_elem
    # cross_rank_roundtrip_bytes = cross_rank_dispatch_bytes * 2
    # ==========================================
    for token_idx, token_routes in enumerate(routes):
        pass

    total_routes = local_routes + cross_rank_routes
    return DispatchLocalitySummary(
        total_routes=total_routes,
        local_routes=local_routes,
        cross_rank_routes=cross_rank_routes,
        cross_rank_dispatch_bytes=cross_rank_routes * hidden_size * bytes_per_elem,
        cross_rank_roundtrip_bytes=cross_rank_routes * hidden_size * bytes_per_elem * 2,
        local_hit_ratio=local_routes / total_routes if total_routes else 0.0,
    )


def recommend_moe_parallel_plan(summary: MoEParallelSummary, locality=None) -> str:
    """
    TODO 3：
    根据 summary 的 overflow、balance_ratio 与可选 locality，返回 'accept' / 'tune' / 'reject'。
    """
    # 提示：先处理空 summary；跨卡比例过高时，即使无 overflow 也应至少 tune。
    # if ???:
    #     return "accept"
    # if ???:
    #     return "tune"
    raise NotImplementedError


def compare_dense_vs_moe_cost(token_count, top_k, hidden_size, num_experts, bytes_per_elem=2):
    """
    辅助账本：返回最小 dense vs MoE 成本对比字典。
    """
    # 提示：先算 dense_compute_units 和 moe_routes，再补通信量与 sparsity_ratio。
    # dense_compute_units = ???
    # moe_routes = ???
    # moe_logical_route_bytes = ???
    # sparsity_ratio = ???
    raise NotImplementedError


def combine_expert_outputs(token_indices, expert_outputs, routing_weights, num_tokens):
    """按原 token 顺序，将各专家的加权输出 combine 回去。

    这是 dispatch 的反向对应步骤：同一 token 的 top-k 专家输出先乘路由权重，
    再累加到同一个 token 位置。这里提供骨架，便于把注意力放在三项并行账本 TODO 上。
    """
    if token_indices.ndim != 1 or routing_weights.ndim != 1:
        raise ValueError('token_indices 与 routing_weights 必须是一维')
    if expert_outputs.ndim != 2 or len(token_indices) != len(expert_outputs) or len(token_indices) != len(routing_weights):
        raise ValueError('专家输出、token 索引与路由权重长度必须一致')
    combined = torch.zeros(num_tokens, expert_outputs.size(-1), dtype=expert_outputs.dtype, device=expert_outputs.device)
    combined.index_add_(0, token_indices, expert_outputs * routing_weights.unsqueeze(-1))
    return combined

```

### 测试

运行下面的测试，检查你的并行负载统计、容量判断和代价估算是否正确。

```python
def test_moe_load_capacity_contract():
    summary = summarize_expert_parallel([[0, 1], [1, 2], [0, 2]], 3, 4)
    assert summary.total_tokens == 3
    assert summary.expert_loads == [2, 2, 2]
    assert summary.capacity == 2
    assert summary.overflow == 0
    assert summary.balance_ratio == 1.0

def test_moe_decision_contract():
    tuned = summarize_expert_parallel([[0], [0], [0], [1]], 2, 8)
    rejected = summarize_expert_parallel([[0]] * 10 + [[1]], 2, 8)
    assert recommend_moe_parallel_plan(tuned) == "tune"
    assert recommend_moe_parallel_plan(rejected) == "reject"

def test_moe_dispatch_locality_contract():
    """专家放置应区分本地命中与必须跨 rank 的 token dispatch。"""
    routes = [[0], [1], [0, 1]]
    locality = summarize_dispatch_locality(
        routes=routes,
        token_source_ranks=[0, 1, 0],
        expert_owner_ranks={0: 0, 1: 1},
        hidden_size=8,
    )
    assert locality.total_routes == 4
    assert locality.local_routes == 3
    assert locality.cross_rank_routes == 1
    assert locality.cross_rank_dispatch_bytes == 16
    assert locality.cross_rank_roundtrip_bytes == 32
    assert locality.local_hit_ratio == 0.75

def test_moe_combine_contract():
    """top-k 专家输出应按 token id 和路由权重回写。"""
    combined = combine_expert_outputs(
        token_indices=torch.tensor([0, 0, 1]),
        expert_outputs=torch.tensor([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0]]),
        routing_weights=torch.tensor([0.25, 0.75, 1.0]),
        num_tokens=2,
    )
    assert torch.allclose(combined, torch.tensor([[2.5, 0.0], [0.0, 2.0]]))

def test_dense_moe_cost_contract():
    cost = compare_dense_vs_moe_cost(token_count=4, top_k=2, hidden_size=8, num_experts=4)
    assert cost["dense_compute_units"] == 32
    assert cost["moe_routes"] == 8
    assert cost["moe_logical_route_bytes"] == 128
    assert cost["sparsity_ratio"] == 0.5

def test_moe_expert_parallel_end_to_end():
    try:
        balanced = summarize_expert_parallel(
            routes=[[0, 1], [1, 2], [0, 2]],
            num_experts=3,
            hidden_size=4,
            bytes_per_elem=2,
            capacity_factor=1.0,
        )
        assert balanced.total_tokens == 3
        assert balanced.total_routes == 6
        assert balanced.expert_loads == [2, 2, 2]
        assert balanced.capacity == 2
        assert balanced.overflow == 0
        assert balanced.logical_route_bytes == 48
        assert balanced.balance_ratio == 1.0
        assert recommend_moe_parallel_plan(balanced) == "accept"

        tuned = summarize_expert_parallel(
            routes=[[0], [0], [0], [1]],
            num_experts=2,
            hidden_size=8,
            bytes_per_elem=2,
            capacity_factor=1.0,
        )
        assert tuned.total_routes == 4
        assert tuned.expert_loads == [3, 1]
        assert tuned.capacity == 2
        assert tuned.overflow == 1
        assert recommend_moe_parallel_plan(tuned) == "tune"

        rejected = summarize_expert_parallel(
            routes=[[0]] * 10 + [[1]],
            num_experts=2,
            hidden_size=8,
            bytes_per_elem=2,
            capacity_factor=1.0,
        )
        assert rejected.total_routes == 11
        assert rejected.expert_loads == [10, 1]
        assert rejected.capacity == 6
        assert rejected.overflow == 4
        assert rejected.balance_ratio > 1.8
        assert recommend_moe_parallel_plan(rejected) == "reject"

        empty = summarize_expert_parallel(
            routes=[],
            num_experts=2,
            hidden_size=8,
            bytes_per_elem=2,
            capacity_factor=1.0,
        )
        assert empty.total_tokens == 0
        assert empty.total_routes == 0
        assert empty.expert_loads == [0, 0]
        assert empty.capacity == 0
        assert empty.overflow == 0
        assert recommend_moe_parallel_plan(empty) == "reject"

        cost = compare_dense_vs_moe_cost(token_count=4, top_k=2, hidden_size=8, num_experts=4)
        assert cost["dense_compute_units"] == 32
        assert cost["moe_routes"] == 8
        assert cost["moe_logical_route_bytes"] == 128
        assert cost["sparsity_ratio"] == 0.5
        print("moe expert parallel passed")
    except NotImplementedError:
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError) as e:
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e


test_moe_load_capacity_contract()
test_moe_decision_contract()
test_moe_dispatch_locality_contract()
test_dense_moe_cost_contract()
test_moe_expert_parallel_end_to_end()
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
from collections import Counter
from dataclasses import dataclass
import math
import torch

@dataclass
class MoEParallelSummary:
    total_tokens: int
    total_routes: int
    expert_loads: list
    capacity: int
    overflow: int
    logical_route_bytes: int
    balance_ratio: float

@dataclass
class DispatchLocalitySummary:
    total_routes: int
    local_routes: int
    cross_rank_routes: int
    cross_rank_dispatch_bytes: int
    cross_rank_roundtrip_bytes: int
    local_hit_ratio: float


def summarize_expert_parallel(routes, num_experts, hidden_size, bytes_per_elem=2, capacity_factor=1.0):
    """
    TODO 1:
    汇总 expert parallel 的最小账本，返回 MoEParallelSummary。
    需要包含 expert_loads、capacity、overflow、logical_route_bytes、balance_ratio。
    """
    # 提示 1: 先校验 num_experts，再遍历 routes 统计每个 expert 的路由数。
    # 提示 2: capacity = ceil((total_routes / num_experts) * capacity_factor)。
    # 提示 3: logical_route_bytes = total_routes * hidden_size * bytes_per_elem；真实网络字节由专家放置决定。
    # 提示 4: balance_ratio = max(expert_loads) / avg_load；如果 avg_load 为 0，则返回 0.0。
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")

    counts = Counter()
    total_tokens = len(routes)
    total_routes = 0
    for token_routes in routes:
        for expert_id in token_routes:
            if expert_id < 0 or expert_id >= num_experts:
                raise ValueError("expert id out of range")
            counts[expert_id] += 1
            total_routes += 1

    expert_loads = [counts.get(i, 0) for i in range(num_experts)]
    avg_load = total_routes / num_experts if num_experts else 0.0
    capacity = math.ceil(avg_load * capacity_factor) if total_routes else 0
    overflow = sum(max(load - capacity, 0) for load in expert_loads)
    logical_route_bytes = total_routes * hidden_size * bytes_per_elem
    balance_ratio = (max(expert_loads) / avg_load) if avg_load > 0 else 0.0

    return MoEParallelSummary(
        total_tokens=total_tokens,
        total_routes=total_routes,
        expert_loads=expert_loads,
        capacity=capacity,
        overflow=overflow,
        logical_route_bytes=logical_route_bytes,
        balance_ratio=round(balance_ratio, 3),
    )


def summarize_dispatch_locality(routes, token_source_ranks, expert_owner_ranks, hidden_size, bytes_per_elem=2):
    """统计专家放置后哪些 token 路由留在本地、哪些需要跨 rank dispatch。"""
    if len(routes) != len(token_source_ranks):
        raise ValueError('每个 token 都需要一个来源 rank')
    local_routes = 0
    cross_rank_routes = 0
    # TODO 2：专家 owner 与 token 来源相同则本地命中，否则需要 all-to-all dispatch。
    for token_idx, token_routes in enumerate(routes):
        source_rank = token_source_ranks[token_idx]
        for expert_id in token_routes:
            if expert_id not in expert_owner_ranks:
                raise ValueError('expert owner 缺少路由到的专家')
            owner_rank = expert_owner_ranks[expert_id]
            if source_rank == owner_rank:
                local_routes += 1
            else:
                cross_rank_routes += 1
    total_routes = local_routes + cross_rank_routes
    return DispatchLocalitySummary(
        total_routes=total_routes,
        local_routes=local_routes,
        cross_rank_routes=cross_rank_routes,
        cross_rank_dispatch_bytes=cross_rank_routes * hidden_size * bytes_per_elem,
        cross_rank_roundtrip_bytes=cross_rank_routes * hidden_size * bytes_per_elem * 2,
        local_hit_ratio=local_routes / total_routes if total_routes else 0.0,
    )


def recommend_moe_parallel_plan(summary: MoEParallelSummary, locality=None) -> str:
    """
    TODO 3：根据负载、overflow 与可选跨 rank 比例返回部署建议。
    """
    # 提示 1: 空 summary 直接 reject。
    # 提示 2: 无 overflow、负载均匀且跨卡比例可控时 accept。
    # 提示 3: 少量 overflow、轻度失衡或跨卡比例偏高时 tune。
    if summary.total_routes == 0:
        return "reject"
    cross_rank_ratio = (locality.cross_rank_routes / locality.total_routes) if locality and locality.total_routes else 0.0
    if summary.overflow == 0 and summary.balance_ratio <= 1.2 and cross_rank_ratio <= 0.5:
        return "accept"
    if summary.overflow <= 1 and summary.balance_ratio <= 1.8 and cross_rank_ratio <= 0.9:
        return "tune"
    return "reject"


def compare_dense_vs_moe_cost(token_count, top_k, hidden_size, num_experts, bytes_per_elem=2):
    """
    辅助账本：
    返回一个最小 dense vs MoE 成本对比字典。
    """
    # 提示 1: dense_compute_units = token_count * hidden_size。
    # 提示 2: moe_routes = token_count * top_k。
    # 提示 3: sparsity_ratio 可以写成 top_k / num_experts；num_experts 为 0 时返回 0.0。
    dense_compute_units = token_count * hidden_size
    moe_routes = token_count * top_k
    moe_logical_route_bytes = moe_routes * hidden_size * bytes_per_elem
    return {
        "dense_compute_units": dense_compute_units,
        "moe_routes": moe_routes,
        "moe_logical_route_bytes": moe_logical_route_bytes,
        "sparsity_ratio": round(top_k / num_experts, 3) if num_experts else 0.0,
    }


def combine_expert_outputs(token_indices, expert_outputs, routing_weights, num_tokens):
    """按原 token 顺序，将各专家的加权输出 combine 回去。

    同一 token 的 top-k 专家输出先乘路由权重，再累加到对应 token 位置。
    """
    if token_indices.ndim != 1 or routing_weights.ndim != 1:
        raise ValueError('token_indices 与 routing_weights 必须是一维')
    if expert_outputs.ndim != 2 or len(token_indices) != len(expert_outputs) or len(token_indices) != len(routing_weights):
        raise ValueError('专家输出、token 索引与路由权重长度必须一致')
    combined = torch.zeros(num_tokens, expert_outputs.size(-1), dtype=expert_outputs.dtype, device=expert_outputs.device)
    combined.index_add_(0, token_indices, expert_outputs * routing_weights.unsqueeze(-1))
    return combined

```

### Step 5（可选）：GPU Expert Parallel 机制基准

下面固定 token 数、hidden size 和专家计算，在真实 NCCL 环境中执行 dispatch → 本地专家计算 → combine，并比较均匀与偏斜路由。

| 学习问题 | CPU 已确认 | GPU 要确认与后续使用 |
| --- | --- | --- |
| 路由负载如何改变 EP 的端到端步骤 | expert load、capacity、overflow、本地命中与跨卡字节账本 | 比较均匀/偏斜路由下的 dispatch、专家计算、combine 时间、负载偏斜和峰值显存，再由 80 扩展到真实模型 |
#### 5.1 环境与固定 workload

至少使用两张 GPU；以下配置固定专家放置和跨 rank dispatch 的 payload。

| 类别 | 固定并记录的字段 | 用途 |
| --- | --- | --- |
| 运行环境 | GPU 数、PyTorch / CUDA、NCCL、world size | 确认 all-to-all 可启动 |
| workload | token 数、hidden size、top-k、专家放置、dtype、warmup、repeats | 固定 dispatch 规模 |
| 对照 | balanced / skewed 路由 | 观察负载不均如何进入最慢 rank 时间 |
| 输出 | `RESULT_DIR`、默认关闭开关 | 分别保存两组 benchmark JSON 与 failure |


```python
# 5.1：默认关闭；真实 all-to-all 需要 torchrun、多进程与 NCCL。
RUN_GPU_EXPERIMENT = False
WORLD_SIZE = 2
NUM_EXPERTS = 4
TOP_K = 2
TOKENS_PER_RANK = 256
HIDDEN_SIZE = 1024
DTYPE = 'bf16'
RUNNER_DTYPE = 'bfloat16' if DTYPE == 'bf16' else 'float32'
ROUTING_SCENARIOS = ['balanced', 'skewed']
WARMUP = 5
REPEATS = 20
RESULT_DIR = 'benchmarks/results/47_expert_parallel'
print({'run': RUN_GPU_EXPERIMENT, 'routing': ROUTING_SCENARIOS, 'world_size': WORLD_SIZE, 'experts': NUM_EXPERTS, 'top_k': TOP_K, 'tokens_per_rank': TOKENS_PER_RANK, 'result_dir': RESULT_DIR})

```

#### 5.2 执行并保存 JSON

开启后对两种路由分别运行真实 dispatch、专家矩阵计算与 combine。除通信 payload 外，还记录最慢 rank 时间、接收 token 分布、负载偏斜和显存。


```python
# 5.2：执行真实 EP dispatch—compute—combine benchmark；默认关闭。
if RUN_GPU_EXPERIMENT:
    import subprocess
    from pathlib import Path
    Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)
    for routing in ROUTING_SCENARIOS:
        result_path = str(Path(RESULT_DIR) / f'{routing}.json')
        command = [
            'torchrun', '--standalone', '--nproc_per_node', str(WORLD_SIZE),
            'tools/run_parallel_mechanism_benchmark.py', '--mode', 'ep_dispatch',
            '--tokens-per-rank', str(TOKENS_PER_RANK), '--hidden-dim', str(HIDDEN_SIZE),
            '--routing', routing, '--dtype', RUNNER_DTYPE,
            '--warmup', str(WARMUP), '--repeats', str(REPEATS), '--output', result_path,
        ]
        subprocess.run(command, check=True)
else:
    print('GPU Expert Parallel benchmark 默认关闭。')

```

#### 5.3 读取结果、解释指标与形成决策

先确认两组运行条件一致，再比较接收 token 分布、负载偏斜、最慢 rank 时间、dispatch bytes 和显存。偏斜路由若显著拉高最慢 rank 时间，应先调整容量或路由，再进入 80 的真实模型 benchmark。


```python
# 5.3：读取 balanced / skewed 两组真实 EP 机制结果。
import json
from pathlib import Path
results = {}
required = {'workload', 'hardware', 'metrics', 'evidence_level', 'failure', 'decision'}
for routing in ROUTING_SCENARIOS:
    path = Path(RESULT_DIR) / f'{routing}.json'
    if not path.exists():
        continue
    result = json.loads(path.read_text())
    missing = required - set(result)
    if missing:
        raise ValueError(f'{routing} 结果 JSON 缺少字段：{sorted(missing)}')
    results[routing] = result
print(results if results else f'尚无 Expert Parallel benchmark 结果：{RESULT_DIR}')

```

| 读取项 | 判断与下一步 |
| --- | --- |
| 运行状态 | `failure` 非空时记录 all-to-all 或多进程初始化问题，不比较延迟 |
| 可比口径 | `world_size / route scenario / payload / environment` 必须与对照一致 |
| 核心指标 | 比较接收 token 分布、负载偏斜、最慢 rank 时间、dispatch bytes 与峰值显存 |
| 结论 | 用 `evidence_level / decision` 决定是否到 80 复测吞吐与长尾 |
### 解析

**1. TODO 1：汇总 expert parallel 的最小账本**
- 先校验 `num_experts`，再遍历 `routes` 统计每个 expert 的路由数。
- 然后按 `capacity_factor` 计算 `capacity`，再汇总 `overflow`、`logical_route_bytes` 和 `balance_ratio`。逻辑路由字节用于比较路由规模，不代表实际网络传输。
- 这一部分的核心不是模拟真实分布式 kernel，而是建立最小并行通信账本，帮助你看清“负载是否偏、容量是否溢出、通信是否变贵”。

**2. TODO 2：区分本地与跨 rank dispatch**
- 同一份 `routes` 在不同专家放置下会有不同的网络代价：来源 rank 与 expert owner 相同的路由留在本地，其余路由才需要跨卡传输。
- `cross_rank_dispatch_bytes` 只统计去程 dispatch；`cross_rank_roundtrip_bytes` 再加上 combine 回程。真实 MoE 还会受到 padding 和 collective 调度影响。

**3. TODO 3：形成专家并行决策**
- `accept` 需要同时满足无明显 overflow、负载均匀和跨卡比例可控；不能只因 top-k 小就认为 EP 一定划算。
- `tune` 表示仍可通过 capacity、router、专家放置或并行组调整改善；严重溢出或热点则应 `reject`。


**4. Combine helper：回写同一 token 的专家结果**
- `combine_expert_outputs` 把每条专家输出乘对应路由权重，再按 `token_indices` 累加回原 token；这与 dispatch 一起构成 EP 的前后两个数据移动方向。

辅助的 dense/MoE 账本只用于解释稀疏激活和路由数的关系。Step 5 运行真实 dispatch—compute—combine 机制基准；真实模型的有效 token 吞吐与长尾由 80 的 MoE benchmark 记录。
## 相关阅读

完成 expert load、capacity、overflow 和 all-to-all 账本后，可以继续阅读 MoE 论文、分布式实现与真实 benchmark。

- [Switch Transformers 原论文](https://arxiv.org/abs/2101.03961)
- [Megatron-LM 官方仓库](https://github.com/NVIDIA/Megatron-LM)
- [79. Distributed Parallel Benchmark | 分布式并行 Benchmark](./79_Distributed_Parallel_Benchmark.md)
- [80. MoE Expert Parallel Benchmark | MoE 专家并行 Benchmark](./80_MoE_Expert_Parallel_Benchmark.md)
- [81. Distributed Inference Project | 分布式推理项目](./81_Distributed_Inference_Project.md)
