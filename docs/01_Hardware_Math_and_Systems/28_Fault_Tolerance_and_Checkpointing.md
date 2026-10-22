# 28. Fault Tolerance and Checkpointing | 容错与检查点

**难度：** Medium-Hard | **环境：** CPU-first | **标签：** `系统工程`, `容错恢复`, `Checkpoint` | **目标人群：** 系统性能入门者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/28_Fault_Tolerance_and_Checkpointing.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

长时间训练可能因进程、设备或节点故障中断。检查点（checkpoint）必须保存足够状态，使训练恢复后继续沿一致的更新轨迹运行。

本节建立“可恢复状态 → 保存频率 → 写入成本 → 恢复校验”的关系。完成后，你应能列出恢复所需状态，并比较故障损失、存储开销和恢复时间之间的取舍。

**关键词：** 检查点、恢复、容错

![本节概念关系](../public/01_Hardware_Math_and_Systems/28_checkpoint_recovery_map.svg)
## 前置阅读

**导语：** 先把显存压力、并行决策和通信调度对应到训练状态，再检查哪些状态必须保存、恢复后如何验证，以及保存策略如何影响训练吞吐和恢复时间。

- [06. VRAM Calculation and ZeRO | 显存计算与 ZeRO 优化](./06_VRAM_Calculation_and_ZeRO.md)
- [26. Parallel Strategy Decision Framework | 并行策略决策框架](./26_Parallel_Strategy_Decision_Framework.md)
- [27. Communication Scheduling Optimization | 通信调度优化](./27_Communication_Scheduling_Optimization.md)
## Q1：Checkpoint 保存的到底是什么？


检查点保存的不是单个模型文件，而是恢复训练所需的一组状态及其清单。

通常至少包括：
- 模型权重
- 优化器状态
- 学习率调度器状态
- 随机数种子
- 训练进度

每项状态都应在 manifest 中登记角色、大小和标识。只有完整清单提交后，检查点才可被恢复端读取；只保存权重只能用于加载模型，不能保证恢复同一训练轨迹。

### Q1小验证
构造检查点 manifest，验证缺少必需状态时不能标记为可恢复。


```python
REQUIRED_ROLES = {'model', 'optimizer', 'scheduler', 'rng', 'progress'}

def inspect_checkpoint_manifest(entries, committed):
    """检查训练恢复所需角色、文件大小和提交状态。"""
    roles = {entry['role'] for entry in entries}
    missing = sorted(REQUIRED_ROLES - roles)
    invalid_files = sorted(entry['path'] for entry in entries if entry['size_mib'] <= 0)
    total_mib = sum(entry['size_mib'] for entry in entries)
    return {
        'total_mib': round(total_mib, 2),
        'missing_roles': missing,
        'invalid_files': invalid_files,
        'committed': committed,
        'recoverable': committed and not missing and not invalid_files,
    }

complete_entries = [
    {'role': 'model', 'path': 'model.bin', 'size_mib': 2400},
    {'role': 'optimizer', 'path': 'optimizer.bin', 'size_mib': 4800},
    {'role': 'scheduler', 'path': 'scheduler.json', 'size_mib': 0.01},
    {'role': 'rng', 'path': 'rng.bin', 'size_mib': 0.01},
    {'role': 'progress', 'path': 'trainer.json', 'size_mib': 0.01},
]
complete = inspect_checkpoint_manifest(complete_entries, committed=True)
weights_only = inspect_checkpoint_manifest(complete_entries[:1], committed=True)
assert complete['recoverable']
assert not weights_only['recoverable'] and 'optimizer' in weights_only['missing_roles']
print('完整检查点：', complete)
print('仅有权重：', weights_only)

```

## Q2：失败概率和恢复时间如何影响有效训练时间？


长训练需要同时估计故障次数、平均回滚量和重新加载时间。若故障在两个检查点之间近似均匀发生，平均回滚计算量约为保存间隔的一半。

- 任务越长，出故障概率越高
- 重新从头跑的损失可能非常大
- checkpoint 的目标，是把故障后的恢复成本压低

该估算不是故障预测器，而是用于比较检查点间隔的统一成本口径。

### Q2小验证
估算给定保存间隔下的预期回滚与恢复开销。


```python
def expected_failure_overhead(run_hours, fault_rate_per_hour, interval_minutes, reload_minutes):
    """按均匀故障假设估算回滚计算与重新加载时间。"""
    if min(run_hours, fault_rate_per_hour, interval_minutes, reload_minutes) < 0 or interval_minutes == 0:
        raise ValueError('时间和故障率不得为负，检查点间隔必须大于 0')
    expected_faults = run_hours * fault_rate_per_hour
    rollback_minutes = expected_faults * interval_minutes / 2
    reload_total_minutes = expected_faults * reload_minutes
    return {
        'expected_faults': round(expected_faults, 3),
        'expected_rollback_minutes': round(rollback_minutes, 2),
        'expected_reload_minutes': round(reload_total_minutes, 2),
        'expected_failure_overhead_minutes': round(rollback_minutes + reload_total_minutes, 2),
    }

hourly = expected_failure_overhead(72, 0.02, 60, 8)
half_hourly = expected_failure_overhead(72, 0.02, 30, 8)
assert half_hourly['expected_rollback_minutes'] < hourly['expected_rollback_minutes']
print('每 60 分钟保存：', hourly)
print('每 30 分钟保存：', half_hourly)

```

## Q3：保存和恢复为什么也有成本？


检查点写入会暂停或竞争训练资源。缩短间隔会减少故障后的平均回滚量，却增加写入次数；延长间隔则相反。

如果保存太频繁，会拖慢训练；如果保存太少，一旦故障，回滚代价又会变大。分布式场景里，保存和恢复还会涉及并行 IO、一致性和调度窗口。

分布式场景还要记录写入是否异步、各 rank 分片是否一致，以及 manifest 何时原子提交。这里先比较同步写入时的理论开销。

### Q3小验证
扫描多个保存间隔，观察写入成本和故障回滚成本如何变化。


```python
import math

def checkpoint_interval_cost(run_minutes, interval_minutes, save_minutes, fault_rate_per_hour, reload_minutes):
    """比较同步保存、预期回滚和重新加载三类开销。"""
    if interval_minutes <= 0 or min(run_minutes, save_minutes, fault_rate_per_hour, reload_minutes) < 0:
        raise ValueError('检查点间隔必须大于 0，其余输入不得为负')
    save_count = math.floor(run_minutes / interval_minutes)
    save_overhead = save_count * save_minutes
    expected_faults = run_minutes / 60 * fault_rate_per_hour
    failure_overhead = expected_faults * (interval_minutes / 2 + reload_minutes)
    return {
        'interval_minutes': interval_minutes,
        'save_count': save_count,
        'save_overhead_minutes': round(save_overhead, 2),
        'failure_overhead_minutes': round(failure_overhead, 2),
        'total_overhead_minutes': round(save_overhead + failure_overhead, 2),
    }

intervals = [15, 30, 60, 120]
reports = [checkpoint_interval_cost(1440, interval, 2, 0.02, 8) for interval in intervals]
best = min(reports, key=lambda report: report['total_overhead_minutes'])
assert best in reports
for report in reports:
    print(report)
print('当前假设下开销最小的间隔：', best['interval_minutes'], '分钟')

```

## Q4：如何判断一个 checkpoint 方案真的可恢复？


分布式检查点只有在以下条件同时满足时才可恢复：必需状态齐全，所有 rank 分片属于同一个 checkpoint ID，分片数量与 manifest 一致，且 manifest 已完成提交。

当恢复时的 world size 发生变化，还需要框架明确支持 reshard；不能仅凭文件存在就假定可以加载。

### Q4小验证
验证分布式分片、检查点标识和提交状态是否一致。


```python
def validate_sharded_checkpoint(manifest, shards):
    """检查分片数量、检查点标识、rank 覆盖和原子提交状态。"""
    issues = []
    expected_ranks = set(range(manifest['world_size']))
    actual_ranks = {shard['rank'] for shard in shards}
    if actual_ranks != expected_ranks:
        issues.append(f'rank coverage mismatch: expected {sorted(expected_ranks)}, got {sorted(actual_ranks)}')
    if any(shard['checkpoint_id'] != manifest['checkpoint_id'] for shard in shards):
        issues.append('checkpoint IDs are inconsistent')
    if any(shard['size_mib'] <= 0 for shard in shards):
        issues.append('one or more shards are empty')
    if not manifest['committed']:
        issues.append('manifest is not committed')
    return {'recoverable': not issues, 'issues': issues}

manifest = {'checkpoint_id': 'step-1200', 'world_size': 4, 'committed': True}
complete_shards = [
    {'rank': rank, 'checkpoint_id': 'step-1200', 'size_mib': 900} for rank in range(4)
]
missing_shard = complete_shards[:-1]
assert validate_sharded_checkpoint(manifest, complete_shards)['recoverable']
assert not validate_sharded_checkpoint(manifest, missing_shard)['recoverable']
print('完整分片：', validate_sharded_checkpoint(manifest, complete_shards))
print('缺少分片：', validate_sharded_checkpoint(manifest, missing_shard))

```

## 相关阅读

**导语：** 继续把状态保存、恢复校验和训练产物交付放进同一条恢复链。

- [04 SFT 实验、Artifact 与交付](../topic_discussion/post_training_optimization/04_end_to_end_experiment.md)
- [62. Instruction Fine Tuning Project | 指令微调项目](../02_PyTorch_Algorithms/62_Instruction_Fine_Tuning_Project.md)
- [PyTorch Distributed Checkpoint | PyTorch 分布式检查点文档](https://pytorch.org/docs/stable/distributed.checkpoint.html)
---
