# 42. Activation Offload | 激活卸载

**难度：** Hard | **环境：** CPU-first

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/42_Activation_Offload.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

**标签：** `显存优化`, `激活值`, `Offload` | **目标人群：** 显存优化学习者

---

## 本节导读

当 activation 占用超过 GPU 预算时，除了 checkpointing，还可以把部分激活搬到 CPU 或 host memory，并在反向传播需要时取回。本节不实现真实的分布式搬运，而是用预算模型计算保留量、搬运量和理论传输时间。

这是一节**机制原理节**：它和 `19` 是兄弟关系。`19` 主讲 checkpointing 的重算路线；`42` 主讲 offload 的搬运路线。两者都在回答“怎么把训练显存压下来”，但实现机制不同，代价模型也不同。

在显存路线里，offload 是否值得采用取决于 GPU 预算、可搬运对象、host / PCIe / NVLink 带宽和额外传输时间。若传输代价过高，应同时比较 checkpointing、batch 调整和 mixed precision 等方案。

本节的代码任务是完成 offload 预算汇总、可行性判断和教学用策略比较：按简化的复用距离选择搬运对象，计算保留量和理论传输时间。它不执行真实 CPU↔GPU copy，也不验证异步重叠、往返带宽、真实 step time 或生产决策；这些结论需要回到 `76` 的固定 workload benchmark。

**关键词：** `offload`, `transfer`, `bandwidth`

---

## 前置阅读

**导语：** 先看 checkpointing 和反向传播，再看 offload 会更容易把“重算”与“搬运”分开。

- [19. Activation Checkpointing | 激活检查点](./19_Activation_Checkpointing_and_Activation_Offload.md)
- [18. Activation and Loss Backward | 激活与损失反向](./18_Activation_and_Loss_Backward.md)
- [P0: 07. CPU/GPU Heterogeneous Scheduling | CPU/GPU 异构调度](../01_Hardware_Math_and_Systems/07_CPU_GPU_Heterogeneous_Scheduling.md)

## 相关阅读

**导语：** 学完 offload 后，下一步重点是把搬运策略放回性能分析和对比验证里，确认显存收益是否真的值得它带来的时延代价。

- [73. Training Performance Analysis | 训练性能分析](./73_Training_Performance_Analysis.md)
- [75. Memory Budget Compression Project | 显存预算压缩项目](./75_Memory_Budget_Compression_Project.md)
- [76. Activation Checkpoint Offload Benchmark | Checkpoint 与 Offload 对比项目](./76_Activation_Checkpoint_Offload_Benchmark.md)

---
### Step 1: 核心思想与痛点

> **Activation Offload 的基本思路：**
> 当某些中间激活在短时间内不会被频繁访问，但又不能像 checkpointing 那样完全丢掉时，可以把它们临时搬到 CPU 或 host memory。这样 GPU 显存就能腾出空间，继续放更大的 batch、更长的序列或更多层的中间状态。
>
> **它和 checkpointing 的区别：**
> - checkpointing 是**重算**：前向时少存，反向时再算一遍。
> - offload 是**搬运**：前向时先搬走，反向时再搬回来。
>
> **两条路线的对照：**
> | 路线 | 前向阶段 | 反向阶段 | 主要代价 | 本节对应 |
> |---|---|---|---|---|
> | Checkpointing | 减少区段内部激活保存 | 重新执行部分前向 | 额外计算与 step time | [19](./19_Activation_Checkpointing_and_Activation_Offload.md) |
> | Offload | 将短期不需要的激活搬离 GPU | 需要时搬回 | CPU/host 带宽、同步与传输时间 | 本节 |
>
> 两者也可以组合，但组合后的收益和代价不能由这张表直接推出，仍需在相同 workload 下测量。
>
> **核心权衡：**
> offload 省的是 GPU 显存，但代价是 PCIe / NVLink / 内存带宽上的搬运时间。网络越慢、搬得越多，收益就越容易被传输开销吞掉。

### Step 2: 代价模型与边界

设一组激活块的总大小为 `A`，GPU 可用预算为 `B`，带宽为 `bw`。

- 如果 `A <= B`，说明理论上不需要 offload。
- 如果 `A > B`，可以优先把“较冷”的激活块搬出 GPU。
- 本节先用单程理论时间近似：`transfer_ms = offloaded_bytes / bandwidth`；真实反向通常还要考虑搬回 GPU 的往返流量。
- offload 不是越多越好；如果搬运时间过长，整体 step time 会被拖慢。若不可搬运对象仍使 `kept_bytes` 超过预算，方案就是不可行。

这一步的重点不是精确模拟硬件，而是把“GPU 显存节省”和“数据搬运成本”放到同一张账上。

### Step 3: 代码实现框架

本题是**显存预算模拟**，不是实际的 CPU↔GPU 搬运。每个激活块保留四个教学字段：名称、大小、`reuse_delay_layers`（距离下一次使用预计还有多少层）和 `offloadable`。这里用复用距离近似冷热程度，不等同于真实运行时的调度指标；真实系统还需要结合反向顺序、生命周期、带宽和同步。延迟越大，在本题模型中越适合优先搬出 GPU。

1. 统计总激活大小。
2. 按 `reuse_delay_layers` 从高到低优先 offload。
3. 计算 offload 后的 GPU 剩余占用和搬运时间。
4. 先检查预算是否可行，再根据本题预设的教学阈值给出 `accept / tune / reject`；这个结果只用于练习，不代表真实工程决策。

#### 字段说明与手算示例

| 字段 | 含义 | 单位 | 本题如何使用 |
|---|---|---|---|
| `bytes_` | 一个激活块占用的字节数 | byte | 汇总总激活量和搬运量 |
| `reuse_delay_layers` | 距离该激活下一次使用预计还有多少层 | layer | 数值越大越优先 offload |
| `gpu_budget_bytes` | 允许激活继续留在 GPU 的预算 | byte | 当保留量超过预算时开始搬运 |
| `bandwidth_gbps` | 假设的有效搬运带宽（代码按 GiB/s 解释） | GiB/s | 估算单程理论传输时间 |

例如总激活为 736 MiB、GPU 预算为 384 MiB，需要搬出 352 MiB；若假设带宽为 8 GiB/s，**单程**理论时间约为 `352 / 8 × 1000 = 44 ms`。真实训练通常还要考虑搬回 GPU 的另一程，以及 pinned memory、异步拷贝和重叠；这里的数值只是纸面估算，不等于真实 step time。
### Step 4: 动手实战（预算模拟）

完成下面三个函数。先说明边界：本题只验证“预算不足时搬哪些块、理论上搬多少数据、估算多少传输时间”，不创建 CUDA tensor，也不执行真实 CPU↔GPU copy。完成后仍必须在 `76` 的真实 GPU benchmark 中验证。

1. `summarize_activation_offload`：汇总 offload 计划、显存节省、搬运成本和预算可行性。
2. `recommend_offload_policy`：根据预设的节省比例和理论搬运时间阈值给出教学用的 `accept / tune / reject`。
3. `compare_offload_vs_checkpointing`：**扩展题**，用一个教学用粗略分数比较两条路线；它不包含质量、吞吐、往返搬运和重叠效果，不能作为工程结论。

### 提示

- `reuse_delay_layers` 是教学模型中的“距离下一次使用的层数”，不是硬件指标；数值越大，表示这块激活越冷。
- offload 先搬运冷块，但真实系统还要考虑反向访问顺序、pinned memory、异步拷贝、往返搬运、重叠和同步，这些不在本题模拟范围内。
- `bandwidth_gbps` 是理论有效带宽假设，代码按 GiB/s 解释；它不是 `nvidia-smi` 能直接给出的实测传输带宽。这里保留旧字段名以兼容示例，实际含义更接近 `bandwidth_gib_per_s`。
- 比较 offload 和 checkpointing 时，可以先看“单位时间省了多少显存”，但只有在预算可行、质量和吞吐都满足要求时才有工程意义。


```python
from dataclasses import dataclass

```


```python
@dataclass
class ActivationChunkSpec:
    """用于预算模拟的一个激活块描述。"""
    name: str  # 激活块名称，便于解释最终搬出/保留了谁
    bytes_: int  # 激活块大小，单位 byte
    reuse_delay_layers: int  # 距离下一次使用的预计层数，越大越冷
    offloadable: bool = True  # 是否允许本策略搬出

@dataclass
class ActivationOffloadSummary:
    """记录 offload 后的显存、传输和预算可行性指标。"""
    total_bytes: int
    gpu_budget_bytes: int
    kept_bytes: int
    offloaded_bytes: int
    transfer_ms: float
    pressure_ratio: float
    saved_ratio: float
    offloaded_names: list
    kept_names: list
    feasible: bool
    overflow_bytes: int


def summarize_activation_offload(chunks, gpu_budget_bytes, bandwidth_gbps=12.0):
    """汇总一次 activation offload 预算计划，不执行真实 tensor 搬运。

    Args:
        chunks: ActivationChunkSpec 或等价 dict 的序列。
        gpu_budget_bytes: 允许保留在 GPU 的 activation 预算，单位 byte。
        bandwidth_gbps: 理论有效带宽，代码按 GiB/s 解释。

    Returns:
        ActivationOffloadSummary；feasible 表示不可搬运对象也没有超预算。
    """
    if gpu_budget_bytes <= 0:
        raise ValueError("gpu_budget_bytes must be positive")
    if bandwidth_gbps <= 0:
        raise ValueError("bandwidth_gbps must be positive")

    normalized = []
    for c in chunks:
        if isinstance(c, ActivationChunkSpec):
            normalized.append(c)
        elif isinstance(c, dict):
            normalized.append(ActivationChunkSpec(**c))
        else:
            raise TypeError("chunks must contain ActivationChunkSpec or dict")

    names = [c.name for c in normalized]
    if len(names) != len(set(names)):
        raise ValueError("chunk names must be unique")
    if any(c.bytes_ < 0 for c in normalized):
        raise ValueError("chunk bytes must be non-negative")
    if any(c.reuse_delay_layers < 0 for c in normalized):
        raise ValueError("reuse_delay_layers must be non-negative")

    total_bytes = sum(c.bytes_ for c in normalized)
    kept_bytes = total_bytes
    offloaded_names = []

    candidates = sorted(
        [c for c in normalized if c.offloadable],
        key=lambda c: (-c.reuse_delay_layers, -c.bytes_, c.name),
    )

    for chunk in candidates:
        if kept_bytes <= gpu_budget_bytes:
            break
        kept_bytes -= chunk.bytes_
        offloaded_names.append(chunk.name)

    kept_names = [c.name for c in normalized if c.name not in offloaded_names]

    # ==========================================
    # TODO 1: 汇总 offload 结果和显存/带宽指标
    # 提示：先算 offloaded_bytes = total_bytes - kept_bytes，
    # 再根据带宽估算 transfer_ms，并补出 pressure_ratio / saved_ratio；
    # overflow_bytes > 0 时 feasible 必须为 False。
    # ==========================================
    overflow_bytes = max(kept_bytes - gpu_budget_bytes, 0)
    feasible = overflow_bytes == 0
    # offloaded_bytes = ???
    # transfer_ms = ???
    # pressure_ratio = ???
    # saved_ratio = ???

    return ActivationOffloadSummary(
        total_bytes=total_bytes,
        gpu_budget_bytes=gpu_budget_bytes,
        kept_bytes=kept_bytes,
        offloaded_bytes=offloaded_bytes,
        transfer_ms=round(transfer_ms, 2),
        pressure_ratio=round(pressure_ratio, 3),
        saved_ratio=round(saved_ratio, 3),
        offloaded_names=offloaded_names,
        kept_names=kept_names,
        feasible=feasible,
        overflow_bytes=overflow_bytes,
    )


def recommend_offload_policy(summary: ActivationOffloadSummary, min_saved_ratio=0.25, max_transfer_ms=60.0):
    """根据教学阈值返回策略建议。

    Args:
        summary: summarize_activation_offload 返回的预算摘要。
        min_saved_ratio: 教学用最低显存节省比例。
        max_transfer_ms: 教学用单程理论传输时间上限。

    Returns:
        'accept'、'tune' 或 'reject'；不代表真实工程决策。
    """
    if summary.offloaded_bytes <= 0:
        return "reject"

    if not summary.feasible:
        return "reject"
    # ==========================================
    # TODO 2: 补全策略判断逻辑
    # 提示：先拒绝不可行方案，再检查 saved_ratio 和 transfer_ms；
    # 收益达到阈值且传输可接受时 accept，边界放宽时 tune，其余 reject。
    # ==========================================
    # if ???:
    #     return "accept"
    # if ???:
    #     return "tune"

    return "reject"


def compare_offload_vs_checkpointing(offload_summary: ActivationOffloadSummary, checkpoint_saved_bytes, checkpoint_extra_ms):
    """用理论节省字节数除以理论额外代价做教学比较。

    Args:
        offload_summary: offload 预算摘要。
        checkpoint_saved_bytes: checkpoint 理论节省字节数。
        checkpoint_extra_ms: checkpoint 理论额外时间，单位 ms。

    Returns:
        包含两种 score 和 preferred 的字典；不替代真实 benchmark。
    """
    # ==========================================
    # TODO 3: 完成 offload 与 checkpointing 的性价比比较
    # 提示：只比较“理论节省字节数 / 理论额外时间”；
    # offload 使用 offloaded_bytes / transfer_ms，checkpoint 使用
    # checkpoint_saved_bytes / checkpoint_extra_ms，返回 score 更大的 preferred。
    # ==========================================
    # offload_score = ???
    # checkpoint_score = ???
    # preferred = ???
    return {
        "offload_score": round(offload_score, 3),
        "checkpoint_score": round(checkpoint_score, 3),
        "preferred": preferred,
    }

```

### 测试

运行下面的测试，检查你的 offload 计划、策略判断和路线比较是否正确。

```python
def test_activation_offload():
    try:
        chunks = [
            ActivationChunkSpec("embed", 256 * 1024 * 1024, 0),
            ActivationChunkSpec("mid_a", 192 * 1024 * 1024, 2),
            ActivationChunkSpec("mid_b", 160 * 1024 * 1024, 3),
            ActivationChunkSpec("tail", 128 * 1024 * 1024, 1),
        ]
        summary = summarize_activation_offload(chunks, gpu_budget_bytes=384 * 1024 * 1024, bandwidth_gbps=8.0)
        assert summary.total_bytes == 736 * 1024 * 1024
        assert summary.offloaded_bytes == 352 * 1024 * 1024
        assert summary.kept_bytes == 384 * 1024 * 1024
        assert summary.offloaded_names == ["mid_b", "mid_a"]
        assert summary.kept_names == ["embed", "tail"]
        assert 42.0 < summary.transfer_ms < 44.0
        assert 0.47 < summary.saved_ratio < 0.49
        assert recommend_offload_policy(summary) == "accept"

        tuned = summarize_activation_offload(chunks, gpu_budget_bytes=384 * 1024 * 1024, bandwidth_gbps=4.0)
        assert recommend_offload_policy(tuned) == "tune"

        rejected = summarize_activation_offload(chunks, gpu_budget_bytes=384 * 1024 * 1024, bandwidth_gbps=1.0)
        assert recommend_offload_policy(rejected) == "reject"

        empty = summarize_activation_offload(chunks, gpu_budget_bytes=1024 * 1024 * 1024, bandwidth_gbps=8.0)
        assert empty.offloaded_bytes == 0
        assert empty.feasible is True
        assert empty.overflow_bytes == 0
        assert recommend_offload_policy(empty) == "reject"

        blocked = summarize_activation_offload(
            [ActivationChunkSpec("fixed", 512 * 1024 * 1024, 0, offloadable=False)],
            gpu_budget_bytes=256 * 1024 * 1024,
            bandwidth_gbps=8.0,
        )
        assert blocked.feasible is False
        assert blocked.overflow_bytes == 256 * 1024 * 1024
        assert recommend_offload_policy(blocked) == "reject"

        invalid_specs = [
            [ActivationChunkSpec("bad", -1, 0)],
            [ActivationChunkSpec("bad", 1, -1)],
            [ActivationChunkSpec("dup", 1, 0), ActivationChunkSpec("dup", 1, 1)],
        ]
        for invalid in invalid_specs:
            try:
                summarize_activation_offload(invalid, gpu_budget_bytes=1, bandwidth_gbps=1.0)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid activation metadata should be rejected")

        better_offload = compare_offload_vs_checkpointing(summary, checkpoint_saved_bytes=300 * 1024 * 1024, checkpoint_extra_ms=80.0)
        assert better_offload["preferred"] == "offload"

        better_ckpt = compare_offload_vs_checkpointing(summary, checkpoint_saved_bytes=500 * 1024 * 1024, checkpoint_extra_ms=20.0)
        assert better_ckpt["preferred"] == "checkpointing"
        print("✅ 预算模拟验证通过：offload 计划、可行性和教学策略判断符合预期。")
        print("证据边界：这是理论预算模拟，不是实际 CPU↔GPU 搬运或真实 step time 测量。")
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError) as e:
        if isinstance(e, AttributeError):
            print("代码未完成，无法找到必要的属性")
        elif isinstance(e, NameError):
            print("代码可能未完成，导致了变量未定义")
        elif isinstance(e, TypeError):
            print("代码可能未完成，导致了类型错误")
        elif isinstance(e, ValueError):
            print("代码可能未完成，导致了参数或数值错误")
        else:
            print(f"代码可能未完成，导致了断言失败: {e}")
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise


test_activation_offload()
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
@dataclass
class ActivationChunkSpec:
    name: str  # 激活块名称，便于解释最终搬出/保留了谁
    bytes_: int  # 激活块大小，单位 byte
    reuse_delay_layers: int  # 距离下一次使用的预计层数，越大越冷
    offloadable: bool = True  # 是否允许本策略搬出

@dataclass
class ActivationOffloadSummary:
    total_bytes: int
    gpu_budget_bytes: int
    kept_bytes: int
    offloaded_bytes: int
    transfer_ms: float
    pressure_ratio: float
    saved_ratio: float
    offloaded_names: list
    kept_names: list
    feasible: bool
    overflow_bytes: int


def summarize_activation_offload(chunks, gpu_budget_bytes, bandwidth_gbps=12.0):
    """汇总一次 activation offload 预算计划，不执行真实 tensor 搬运。

    Args:
        chunks: ActivationChunkSpec 或等价 dict 的序列。
        gpu_budget_bytes: GPU activation 预算，单位 byte。
        bandwidth_gbps: 理论有效带宽，代码按 GiB/s 解释。

    Returns:
        ActivationOffloadSummary；feasible 表示最终保留量未超过预算。
    """
    if gpu_budget_bytes <= 0:
        raise ValueError("gpu_budget_bytes must be positive")
    if bandwidth_gbps <= 0:
        raise ValueError("bandwidth_gbps must be positive")

    normalized = []
    for c in chunks:
        if isinstance(c, ActivationChunkSpec):
            normalized.append(c)
        elif isinstance(c, dict):
            normalized.append(ActivationChunkSpec(**c))
        else:
            raise TypeError("chunks must contain ActivationChunkSpec or dict")

    names = [c.name for c in normalized]
    if len(names) != len(set(names)):
        raise ValueError("chunk names must be unique")
    if any(c.bytes_ < 0 for c in normalized):
        raise ValueError("chunk bytes must be non-negative")
    if any(c.reuse_delay_layers < 0 for c in normalized):
        raise ValueError("reuse_delay_layers must be non-negative")

    total_bytes = sum(c.bytes_ for c in normalized)
    kept_bytes = total_bytes
    offloaded_names = []

    candidates = sorted(
        [c for c in normalized if c.offloadable],
        key=lambda c: (-c.reuse_delay_layers, -c.bytes_, c.name),
    )

    for chunk in candidates:
        if kept_bytes <= gpu_budget_bytes:
            break
        kept_bytes -= chunk.bytes_
        offloaded_names.append(chunk.name)

    kept_names = [c.name for c in normalized if c.name not in offloaded_names]

    # ==========================================
    # TODO 1: 汇总 offload 结果和显存/带宽指标
    # 提示：先算 offloaded_bytes = total_bytes - kept_bytes，
    # 再根据带宽估算 transfer_ms，并补出 pressure_ratio / saved_ratio；
    # overflow_bytes > 0 时 feasible 必须为 False。
    # ==========================================
    overflow_bytes = max(kept_bytes - gpu_budget_bytes, 0)
    feasible = overflow_bytes == 0
    offloaded_bytes = total_bytes - kept_bytes
    transfer_ms = offloaded_bytes / (bandwidth_gbps * (1024 ** 3)) * 1000 if offloaded_bytes else 0.0
    pressure_ratio = total_bytes / gpu_budget_bytes
    saved_ratio = offloaded_bytes / total_bytes if total_bytes else 0.0

    return ActivationOffloadSummary(
        total_bytes=total_bytes,
        gpu_budget_bytes=gpu_budget_bytes,
        kept_bytes=kept_bytes,
        offloaded_bytes=offloaded_bytes,
        transfer_ms=round(transfer_ms, 2),
        pressure_ratio=round(pressure_ratio, 3),
        saved_ratio=round(saved_ratio, 3),
        offloaded_names=offloaded_names,
        kept_names=kept_names,
        feasible=feasible,
        overflow_bytes=overflow_bytes,
    )


def recommend_offload_policy(summary: ActivationOffloadSummary, min_saved_ratio=0.25, max_transfer_ms=60.0):
    """根据教学阈值返回策略建议。

    Args:
        summary: summarize_activation_offload 返回的预算摘要。
        min_saved_ratio: 教学用最低显存节省比例。
        max_transfer_ms: 教学用单程理论传输时间上限。

    Returns:
        'accept'、'tune' 或 'reject'；不代表真实工程决策。
    """
    if summary.offloaded_bytes <= 0:
        return "reject"

    # ==========================================
    # TODO 2: 补全策略判断逻辑
    # 提示：先拒绝不可行方案，再检查 saved_ratio 和 transfer_ms；
    # 收益达到阈值且传输可接受时 accept，边界放宽时 tune，其余 reject。
    # ==========================================
    if not summary.feasible:
        return "reject"
    if summary.kept_bytes <= summary.gpu_budget_bytes and summary.saved_ratio >= min_saved_ratio and summary.transfer_ms <= max_transfer_ms:
        return "accept"
    if summary.saved_ratio >= min_saved_ratio / 2 and summary.transfer_ms <= max_transfer_ms * 2:
        return "tune"

    return "reject"


def compare_offload_vs_checkpointing(offload_summary: ActivationOffloadSummary, checkpoint_saved_bytes, checkpoint_extra_ms):
    """用理论节省字节数除以理论额外代价做教学比较。

    Args:
        offload_summary: offload 预算摘要。
        checkpoint_saved_bytes: checkpoint 理论节省字节数。
        checkpoint_extra_ms: checkpoint 理论额外时间，单位 ms。

    Returns:
        包含两种 score 和 preferred 的字典；不替代真实 benchmark。
    """
    # ==========================================
    # TODO 3: 完成 offload 与 checkpointing 的性价比比较
    # 提示：只比较“理论节省字节数 / 理论额外时间”；
    # offload 使用 offloaded_bytes / transfer_ms，checkpoint 使用
    # checkpoint_saved_bytes / checkpoint_extra_ms，返回 score 更大的 preferred。
    # ==========================================
    offload_score = offload_summary.offloaded_bytes / max(offload_summary.transfer_ms, 1e-6)
    checkpoint_score = checkpoint_saved_bytes / max(checkpoint_extra_ms, 1e-6)
    preferred = "offload" if offload_score >= checkpoint_score else "checkpointing"
    return {
        "offload_score": round(offload_score, 3),
        "checkpoint_score": round(checkpoint_score, 3),
        "preferred": preferred,
    }
```

### 解析

**1. TODO 1：汇总 offload 结果和显存/带宽指标**
- **实现方式**：先用 `offloaded_bytes = total_bytes - kept_bytes` 得到本题预算模型中的搬运量，再按假设带宽估算 `transfer_ms`，最后补出 `pressure_ratio` 和 `saved_ratio`。它不是实际 CUDA copy 的测量值。
- **关键点**：`kept_bytes` 反映 offload 后还留在 GPU 上的激活量，`offloaded_bytes` 反映这次计划实际搬走了多少。
- **工程意义**：这一组指标把“省了多少显存”和“付出了多少搬运代价”放到同一张账上，是后面做策略判断的前提。

**2. TODO 2：补全策略判断逻辑**
- **实现方式**：先排除“根本没有发生 offload”的情况，再按“收益足够且搬运可接受”判断 `accept`，最后把“有收益但还不够稳”的情况归到 `tune`。
- **关键点**：这里不是只看显存节省，也不是只看搬运时间，而是两者一起看。
- **工程意义**：offload 不是默认值得做的优化；只有当显存确实被压进预算，且传输成本没有吞掉收益时，才值得接受。

**3. TODO 3：完成 offload 与 checkpointing 的性价比比较**
- **实现方式**：两条路线都按“节省字节数 / 额外时间”计算一个最小 score，再比较谁更大。
- **关键点**：`offload_score` 近似表示“单位搬运时间换回多少显存”，`checkpoint_score` 近似表示“单位重算时间换回多少显存”。
- **工程意义**：这不是精确性能模型，而是一个最小决策框架，帮助你判断当前瓶颈更像带宽问题还是重算问题。