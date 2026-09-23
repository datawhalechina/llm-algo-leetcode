# 76. Activation Checkpoint Offload Benchmark | Activation / Checkpoint / Offload 对比项目

**难度：** Hard | **环境：** CPU 可完成正确性验证；GPU 用于策略 benchmark | **标签：** `显存优化`, `Checkpoint/Offload`, `基准对比` | **目标人群：** 准备比较训练显存策略的学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

前面的内容介绍了激活值如何参与反向传播，以及保存、重算和搬运之间的代价。本节围绕同一训练任务比较 checkpoint、offload 和组合方案，观察它们如何改变显存占用、训练速度和运行稳定性。完成后，你应能根据显存压力和速度代价判断哪种策略值得继续验证。

**关键词：** `activation`, `checkpoint`, `offload`, `memory`, `benchmark`

---
## 前置阅读

**导语：** 先理解 checkpoint 和 offload 如何改变激活值的保存与计算，再参考 73 节的测量口径，比较同一训练任务下不同策略的显存收益和速度代价。
- [19. Activation Checkpointing | 激活检查点](./19_Activation_Checkpointing.md)
- [42. Activation Offload | 激活卸载](./42_Activation_Offload.md)
- [73. Training Performance Analysis | 训练性能分析](./73_Training_Performance_Analysis.md)


---
### Step 1：确定显存策略与实验输入输出
训练前向会产生反向传播需要的激活值。本节固定同一训练任务，只比较激活值的保存、重算和搬运方式，观察显存峰值、训练速度与训练结果如何变化。

| 实验要素 | 本节固定或比较的内容 | 最终输出 |
|:---|:---|:---|
| 比较对象 | baseline、checkpoint、offload、hybrid | 四种策略的可比较集合 |
| 共同任务 | 固定模型、输入、训练目标和 workload | 排除任务差异造成的结果变化 |
| 变化变量 | 只改变激活值的保存、重算或搬运方式 | 每种策略的显存、速度和训练结果 |
| 实验输出 | 汇总峰值显存、step time、吞吐、loss 和运行状态 | 策略对照结果与后续预算判断依据 |



![76 训练激活值的保存方式与代价](../public/02_PyTorch_Algorithms/76_strategy_lifecycle.svg)
### Step 2：统一实验条件与实验协议
统一实验条件，只改变显存策略；不同 workload 或 dtype 另存报告。随机输入只保证策略间可比，不能替代真实数据集质量评估。

| 项目 | 固定内容 | 目的 |
|:---|:---|:---|
| 模型与输入 | 同一模型、dtype、backend、随机输入、标签和初始化 seed | 保证策略间训练任务一致 |
| 训练配置 | optimizer、学习率、训练步数、Gradient Accumulation、effective batch | 排除训练过程差异 |
| workload | batch、seq_len、warmup、iters、seed、dtype | 保证压力和统计口径一致 |
| 对照关系 | baseline 作为显存、速度、吞吐和 loss 的参照；候选来自 Step1 | 保证差值方向一致 |


### Step 3：指标与项目判定
显存最低的方案不一定最值得采用。先统一指标，再比较容量、速度和训练状态；需要预算取舍时，再使用 75 节的决策表。

| 指标 | 含义 | 用于判断 |
|:---|:---|:---|
| peak allocated | 张量实际达到的显存峰值 | 观察显存收益 |
| peak reserved | allocator 保留的显存峰值 | 观察缓存和碎片影响 |
| step time / samples/s | 重算、搬运和同步的时间代价 | 观察吞吐损失 |
| val loss / OOM | 训练质量和可运行性 | 过滤不可接受方案 |
| 判定规则 | 未 OOM、质量达标，并满足显存收益和吞吐保留率 | 输出 `accept / tune / reject` |

![76 显存策略指标与预算决策](../public/02_PyTorch_Algorithms/76_strategy_decision_flow.svg)


### Step 4：实现 CPU 决策逻辑

请补全下方 3 个函数，把实验报告字段转换为可复核的显存策略决策。完成后先运行 CPU 测试；这里不启动模型、不读取 GPU 状态，真实指标由 Step 5 单独采集。

| TODO | 函数职责 | 实现重点 | 主要输出 |
|:---|:---|:---|:---|
| TODO 1 | `validate_strategy_budget` | 检查预算和质量阈值 | 校验结果 |
| TODO 2 | `summarize_memory_strategy_candidates` | 汇总候选状态和收益 | 候选状态、可行候选、收益指标 |
| TODO 3 | `decide_memory_strategy_project` | 根据阈值形成项目决策 | `decision`、`reason`、`next_action` |



```python
from typing import Dict, List

```


```python
# 3 个核心 TODO：预算检查、候选汇总、项目结论
# 目标：把 baseline / checkpoint / offload / hybrid 的方案比较收束成一份项目报告。
# CPU 题目区只验证预算与决策逻辑；真实策略、saved tensors、重算和搬运代价属于 GPU 实验。

def validate_strategy_budget(budget: Dict[str, float], quality_floor: Dict[str, float]) -> Dict[str, object]:
    """校验显存、吞吐和质量阈值，返回可供后续筛选使用的诊断结果。"""
    # ==========================================
    # TODO 1：完成预算和质量阈值校验。
    # 提示：先列出两组必需字段，再分别收集缺失字段和非法数值。
    # required_budget_keys = ???
    # required_quality_keys = ???
    # missing_keys = ???
    # invalid_keys = ???
    # 返回 is_valid、missing_keys 和 invalid_keys；不负责判断某个策略是否最优。
    # ==========================================
    raise NotImplementedError("请先完成 TODO 代码！")

def summarize_memory_strategy_candidates(candidates: List[Dict[str, object]], budget: Dict[str, float], quality_floor: Dict[str, float]) -> Dict[str, object]:
    """按统一预算筛选候选策略，并保留 OOM、无效指标和拒绝原因。"""
    # ==========================================
    # TODO 2：完成候选汇总，区分无效候选、不可行候选和可行候选。
    # 候选字段：name、status、peak_memory_mb、samples_per_s、eval_loss / val_loss。
    # 提示：先处理 OOM、重复和无效指标，再检查显存、吞吐和质量条件。
    # evaluations = ???
    # feasible = ???
    # seen_names = ???
    #       memory_ok、speed_ok、quality_ok 是每个候选的派生判断，不单独挖空；
    # evaluations 保存每个候选的可追溯状态，feasible 只保存通过全部门槛的候选。
    # ==========================================
    raise NotImplementedError("请先完成 TODO 代码！")

def decide_memory_strategy_project(summary: Dict[str, object]) -> Dict[str, object]:
    """根据可行候选和显存收益输出 accept、tune 或 reject。

    accept 仅表示当前 workload 和预算下值得继续保留，不表示普遍有效。
    """
    # ==========================================
    # TODO 3：完成项目决策。
    # 提示：先检查 baseline，再检查可行候选，最后判断显存收益和吞吐保留率。
    # baseline_available = ???
    # feasible_count = ???
    # best_candidate = ???
    #       meaningful_memory_gain、acceptable_throughput 根据 summary 派生；不单独挖空。
    # 返回 decision、reason 和 next_action；不要硬编码具体策略名称。
    # ==========================================
    raise NotImplementedError("请先完成 TODO 代码！")

```


```python
# 测试你的实现
def test_memory_strategy_project():
    try:
        budget = {'memory_cap_mb': 12000.0, 'min_samples_per_s': 6.0, 'min_memory_saving_mb': 512.0, 'min_throughput_ratio': 0.70}
        quality_floor = {'max_val_loss': 1.15}
        check = validate_strategy_budget(budget, quality_floor)
        assert check['is_valid'] is True, '预算检查应通过'
        assert check['missing_keys'] == [], '完整预算不应缺字段'
        missing = validate_strategy_budget({'memory_cap_mb': 12000.0}, quality_floor)
        assert missing['is_valid'] is False, '缺少吞吐门槛时应拒绝预算配置'
        assert 'min_samples_per_s' in missing['missing_keys'], '应指出缺少的预算字段'
        invalid = validate_strategy_budget(
            {'memory_cap_mb': -1.0, 'min_samples_per_s': 0.0, 'min_memory_saving_mb': 512.0, 'min_throughput_ratio': 1.2},
            {'max_val_loss': 1.15},
        )
        assert invalid['is_valid'] is False, '非法预算值应被拒绝'
        assert {'memory_cap_mb', 'min_throughput_ratio'} <= set(invalid['invalid_keys'])

        candidates = [
            {'name': 'baseline', 'peak_memory_mb': 18000.0, 'samples_per_s': 8.0, 'val_loss': 1.06},
            {'name': 'checkpoint', 'peak_memory_mb': 11800.0, 'samples_per_s': 6.6, 'val_loss': 1.08},
            {'name': 'offload', 'peak_memory_mb': 10500.0, 'samples_per_s': 5.2, 'val_loss': 1.09},
            {'name': 'hybrid', 'peak_memory_mb': 9800.0, 'samples_per_s': 6.1, 'val_loss': 1.11},
        ]
        summary = summarize_memory_strategy_candidates(candidates, budget, quality_floor)
        assert summary['feasible_count'] == 2, '应有两个方案满足预算与质量'
        assert summary['best_candidate'] == 'hybrid', 'hybrid 应成为最省显存的可行方案'
        evaluation_by_name = {item['name']: item for item in summary['evaluations']}
        assert evaluation_by_name['baseline']['status'] == 'rejected'
        assert 'memory_over_budget' in evaluation_by_name['baseline']['reasons']
        assert evaluation_by_name['checkpoint']['status'] == 'feasible'
        assert evaluation_by_name['offload']['status'] == 'rejected'
        assert 'throughput_below_floor' in evaluation_by_name['offload']['reasons']
        assert evaluation_by_name['hybrid']['status'] == 'feasible'

        decision = decide_memory_strategy_project(summary)
        assert decision['decision'] == 'accept', '可行且最优的方案应被接受'
        custom_summary = summarize_memory_strategy_candidates(
            [
                {'name': 'baseline', 'peak_memory_mb': 18000.0, 'samples_per_s': 8.0, 'val_loss': 1.06},
                {'name': 'activation_checkpoint', 'peak_memory_mb': 11800.0, 'samples_per_s': 6.6, 'val_loss': 1.08},
            ],
            budget,
            quality_floor,
        )
        assert decide_memory_strategy_project(custom_summary)['decision'] == 'accept', '自定义策略也应可被接受'
        no_baseline = summarize_memory_strategy_candidates(
            [{'name': 'checkpoint', 'peak_memory_mb': 10000.0, 'samples_per_s': 7.0, 'val_loss': 1.05}],
            budget,
            quality_floor,
        )
        assert decide_memory_strategy_project(no_baseline)['reason'] == 'baseline_missing_or_invalid'

        hard_summary = summarize_memory_strategy_candidates(
            [
                {'name': 'checkpoint', 'peak_memory_mb': 13000.0, 'samples_per_s': 6.4, 'val_loss': 1.10},
                {'name': 'offload', 'peak_memory_mb': 11000.0, 'samples_per_s': 4.8, 'val_loss': 1.12},
            ],
            budget,
            quality_floor,
        )
        hard_decision = decide_memory_strategy_project(hard_summary)
        assert hard_decision['decision'] == 'reject', '没有满足预算与质量时应 reject'

        edge_summary = summarize_memory_strategy_candidates(
            [
                {'name': 'oom_candidate', 'status': 'oom'},
                {'name': 'incomplete', 'peak_memory_mb': 10000.0},
            ],
            budget,
            quality_floor,
        )
        assert edge_summary['oom_count'] == 1, 'OOM 候选应单独计数'
        assert edge_summary['invalid_count'] == 1, '缺少指标的候选应标记为 invalid'
        edge_evaluations = {item['name']: item for item in edge_summary['evaluations']}
        assert edge_evaluations['oom_candidate']['status'] == 'oom'
        assert edge_evaluations['incomplete']['status'] == 'invalid'
        assert decide_memory_strategy_project(edge_summary)['decision'] == 'reject', '没有可行策略时应 reject'

        invalid_baseline = summarize_memory_strategy_candidates(
            [
                {'name': 'baseline', 'peak_memory_mb': float('nan'), 'samples_per_s': 8.0, 'eval_loss': 1.06},
                {'name': 'checkpoint', 'peak_memory_mb': 11800.0, 'samples_per_s': 6.6, 'eval_loss': 1.08},
            ],
            budget,
            quality_floor,
        )
        assert invalid_baseline['baseline_available'] is False, 'NaN baseline 不应进入收益计算'

        small_gain = summarize_memory_strategy_candidates(
            [
                {'name': 'baseline', 'peak_memory_mb': 12000.0, 'samples_per_s': 8.0, 'eval_loss': 1.06},
                {'name': 'checkpoint', 'peak_memory_mb': 11700.0, 'samples_per_s': 7.0, 'eval_loss': 1.07},
            ],
            budget,
            quality_floor,
        )
        assert decide_memory_strategy_project(small_gain)['decision'] == 'tune', '显存节省不足时应 tune'
        print('所有测试通过！')
    except NotImplementedError:
        print('请先完成 TODO 代码！')
        raise
    except AssertionError as e:
        print(f'测试失败: {e}')
        raise NotImplementedError('请先完成 TODO 代码！') from e
    except Exception as e:
        print(f'发生非预期错误: {e}')
        raise


test_memory_strategy_project()

```

### Step 4 测试区补充：CPU 正确性检查

没有 GPU 时，不能验证真实 peak memory、CUDA kernel 或 CPU-GPU 搬运速度，但仍可以用一个小型网络检查策略语义：baseline、checkpoint、offload 和 hybrid 应该得到一致的 loss 与参数梯度。测试还会观察 baseline / checkpoint 的 saved tensors；`save_on_cpu` 会接管 offload 的保存钩子，因此 offload / hybrid 不伪造张量数量。这个测试验证的是 autograd 正确性，不是显存收益；真实显存结论由 Step 5 的 GPU benchmark 给出。

运行这个 cell 前不需要下载模型或准备 CUDA。重点检查三件事：四种策略的 loss 是否一致、参数梯度是否一致、offload 是否确实改变了保存位置。

```python
# CPU 正确性 cell：使用小型网络验证四种策略的 autograd 语义，不启动 GPU benchmark。
# 预期输出是 loss / gradient 一致性和 saved-tensor 观察结果；不要把这些数值填入 GPU 结果表。
import contextlib
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

VALID_STRATEGIES = {'baseline', 'checkpoint', 'offload', 'hybrid'}

def run_cpu_strategy(strategy, observe_saved_tensors=False):
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f'未知策略: {strategy}')
    torch.manual_seed(42)
    first = nn.Linear(8, 16)
    second = nn.Linear(16, 4)
    x = torch.randn(2, 8)
    target = torch.randn(2, 4)
    offload_context = (
        torch.autograd.graph.save_on_cpu(pin_memory=False)
        if strategy in {'offload', 'hybrid'} else contextlib.nullcontext()
    )
    saved, unpacked = [], []
    def pack(tensor):
        saved.append({'shape': tuple(tensor.shape), 'numel': tensor.numel(), 'device': str(tensor.device)})
        return tensor
    def unpack(tensor):
        unpacked.append({'shape': tuple(tensor.shape), 'numel': tensor.numel(), 'device': str(tensor.device)})
        return tensor
    observe_context = (
        torch.autograd.graph.saved_tensors_hooks(pack, unpack)
        if observe_saved_tensors and strategy not in {'offload', 'hybrid'} else contextlib.nullcontext()
    )
    with observe_context:
        with offload_context:
            if strategy in {'checkpoint', 'hybrid'}:
                hidden = checkpoint(first, x, use_reentrant=False)
            else:
                hidden = first(x)
            loss = ((second(hidden) - target) ** 2).mean()
            loss.backward()
    gradients = [first.weight.grad.detach().clone(), second.weight.grad.detach().clone()]
    observation = {
        'saved_count': len(saved) if observe_saved_tensors and strategy not in {'offload', 'hybrid'} else None,
        'unpacked_count': len(unpacked) if observe_saved_tensors and strategy not in {'offload', 'hybrid'} else None,
        'saved_numel': sum(item['numel'] for item in saved) if observe_saved_tensors and strategy not in {'offload', 'hybrid'} else None,
        'note': 'save_on_cpu owns the autograd hooks; inspect offload placement with GPU profiling.' if strategy in {'offload', 'hybrid'} else 'saved_tensors_hooks observation',
    }
    return (second(hidden).detach(), loss.detach(), gradients, observation)

baseline_output, baseline_loss, baseline_grads, baseline_observation = run_cpu_strategy('baseline', observe_saved_tensors=True)
assert torch.isfinite(baseline_output).all() and torch.isfinite(baseline_loss).all()
assert baseline_observation['saved_count'] > 0
assert baseline_observation['saved_count'] == baseline_observation['unpacked_count']
print(f"baseline saved tensors: {baseline_observation['saved_count']}, saved elements: {baseline_observation['saved_numel']}")
for strategy in ('checkpoint', 'offload', 'hybrid'):
    output, loss, grads, observation = run_cpu_strategy(strategy, observe_saved_tensors=strategy == 'checkpoint')
    assert torch.isfinite(output).all() and torch.isfinite(loss).all()
    assert torch.allclose(output, baseline_output, atol=1e-6, rtol=1e-5)
    assert torch.allclose(loss, baseline_loss, atol=1e-6, rtol=1e-5)
    assert all(torch.allclose(g, b, atol=1e-6, rtol=1e-5) for g, b in zip(grads, baseline_grads))
    if strategy == 'checkpoint':
        assert observation['saved_count'] > 0
        assert observation['saved_count'] == observation['unpacked_count']
    else:
        assert observation['saved_count'] is None
    print(f'{strategy}: loss/gradient check passed')
try:
    run_cpu_strategy('unknown')
except ValueError:
    pass
else:
    raise AssertionError('未知策略应明确报错')
print('CPU correctness test passed; this does not measure real GPU memory saving.')

```

🛑 **STOP HERE** 🛑

## 参考代码与解析

### 代码


```python
import math
from typing import Dict, List
def validate_strategy_budget(budget: Dict[str, float], quality_floor: Dict[str, float]) -> Dict[str, object]:
    # ==========================================
    # TODO 1 对应变量 required_budget_keys / required_quality_keys：收集两组契约字段。
    required_budget_keys = ['memory_cap_mb', 'min_samples_per_s', 'min_memory_saving_mb', 'min_throughput_ratio']  # 预算必需字段
    required_quality_keys = ['max_val_loss']
    # TODO 1 对应变量 missing_keys / invalid_keys：分别记录缺失字段和非法数值。
    missing_keys = [key for key in required_budget_keys if key not in budget]
    missing_keys += [key for key in required_quality_keys if key not in quality_floor]
    numeric_values = {key: budget.get(key) for key in required_budget_keys}
    numeric_values.update({key: quality_floor.get(key) for key in required_quality_keys})
    invalid_keys = [
        key for key, value in numeric_values.items()
        if key not in missing_keys and (not isinstance(value, (int, float)) or not math.isfinite(value))
    ]
    memory_cap = budget.get('memory_cap_mb')
    min_throughput = budget.get('min_samples_per_s')
    min_saving = budget.get('min_memory_saving_mb')
    throughput_ratio = budget.get('min_throughput_ratio')
    max_loss = quality_floor.get('max_val_loss')
    if isinstance(memory_cap, (int, float)) and memory_cap <= 0:
        invalid_keys.append('memory_cap_mb')
    if isinstance(min_throughput, (int, float)) and min_throughput < 0:
        invalid_keys.append('min_samples_per_s')
    if isinstance(min_saving, (int, float)) and min_saving < 0:
        invalid_keys.append('min_memory_saving_mb')
    if isinstance(throughput_ratio, (int, float)) and not 0 <= throughput_ratio <= 1:
        invalid_keys.append('min_throughput_ratio')
    if isinstance(max_loss, (int, float)) and max_loss < 0:
        invalid_keys.append('max_val_loss')
    invalid_keys = list(dict.fromkeys(invalid_keys))
    return {
        'is_valid': len(missing_keys) == 0 and len(invalid_keys) == 0,
        'missing_keys': missing_keys,
        'invalid_keys': invalid_keys,
    }


def summarize_memory_strategy_candidates(candidates: List[Dict[str, object]], budget: Dict[str, float], quality_floor: Dict[str, float]) -> Dict[str, object]:
    # ==========================================
    # TODO 2 对应变量 evaluations / feasible / seen_names：记录候选状态、可行候选和去重名称。
    # 提示：memory_ok / speed_ok / quality_ok 是每个候选的派生门槛，不作为独立挖空。
    # ==========================================
    feasible: List[Dict[str, float]] = []  # 通过全部门槛的候选
    quality_failed = 0
    invalid_count = 0
    oom_count = 0
    evaluations = []
    seen_names = set()

    for candidate in candidates:
        # 依次处理名称、OOM、指标完整性，再判断预算、吞吐和质量条件。
        name = candidate.get('name')
        if not isinstance(name, str) or not name or name in seen_names:
            invalid_count += 1
            evaluations.append({'name': name, 'status': 'invalid', 'reasons': ['missing_or_duplicate_name']})
            continue
        seen_names.add(name)
        if candidate.get('status', 'ok') == 'oom':
            oom_count += 1
            evaluations.append({'name': name, 'status': 'oom', 'reasons': ['oom']})
            continue
        memory = candidate.get('peak_memory_mb')
        throughput = candidate.get('samples_per_s')
        eval_loss = candidate.get('eval_loss', candidate.get('val_loss'))
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (memory, throughput, eval_loss)):
            invalid_count += 1
            evaluations.append({'name': name, 'status': 'invalid', 'reasons': ['missing_or_non_numeric_metric']})
            continue
        memory_ok = memory <= budget['memory_cap_mb']
        speed_ok = throughput >= budget['min_samples_per_s']
        quality_ok = eval_loss <= quality_floor['max_val_loss']
        if not quality_ok:
            quality_failed += 1
        reasons = []
        if not memory_ok:
            reasons.append('memory_over_budget')
        if not speed_ok:
            reasons.append('throughput_below_floor')
        if not quality_ok:
            reasons.append('quality_over_floor')
        evaluations.append({
            'name': name,
            'status': 'feasible' if not reasons else 'rejected',
            'reasons': reasons,
        })
        if memory_ok and speed_ok and quality_ok:
            feasible.append(candidate)

    # 优先选择峰值显存更低的方案，再用吞吐和质量处理并列情况。
    feasible.sort(key=lambda x: (x['peak_memory_mb'], -x['samples_per_s'], x.get('eval_loss', x.get('val_loss'))))
    best_candidate = feasible[0]['name'] if feasible else None
    baseline = next((item for item in candidates if item.get('name') == 'baseline' and item.get('status', 'ok') == 'ok' and all(isinstance(item.get(key), (int, float)) and math.isfinite(item.get(key)) for key in ('peak_memory_mb', 'samples_per_s'))), None)
    best = feasible[0] if feasible else None
    return {
        'candidate_count': len(candidates),
        'measured_count': len(candidates) - oom_count - invalid_count,
        'oom_count': oom_count,
        'invalid_count': invalid_count,
        'feasible_count': len(feasible),
        'best_candidate': best_candidate,
        'quality_failed_count': quality_failed,
        'feasible_names': [item['name'] for item in feasible],
        'baseline_peak_memory_mb': baseline['peak_memory_mb'] if baseline else None,
        'baseline_available': baseline is not None,
        'best_peak_memory_mb': best['peak_memory_mb'] if best else None,
        'memory_saving_mb': (baseline['peak_memory_mb'] - best['peak_memory_mb']) if baseline and best else 0.0,
        'throughput_ratio': (best['samples_per_s'] / baseline['samples_per_s']) if baseline and best else None,
        'min_memory_saving_mb': budget['min_memory_saving_mb'],
        'min_throughput_ratio': budget['min_throughput_ratio'],
        'evaluations': evaluations,
    }


def decide_memory_strategy_project(summary: Dict[str, object]) -> Dict[str, object]:
    """根据基线、可行候选和预算阈值输出 accept、tune 或 reject。"""
    # 决策顺序固定为：基线完整性、可行候选、显存收益和吞吐保留率。
    # ==========================================
    # TODO 3 对应变量 baseline_available / feasible_count / best_candidate：确定决策输入。
    # 提示：meaningful_memory_gain / acceptable_throughput 根据 summary 派生，再按顺序输出结论。
    # ==========================================
    feasible_count = summary['feasible_count']  # 可行候选数量
    best_candidate = summary['best_candidate']
    quality_failed_count = summary['quality_failed_count']

    if not summary.get('baseline_available', False):
        return {
            'decision': 'reject',
            'reason': 'baseline_missing_or_invalid',
            'next_action': 'rerun_baseline_before_comparing_candidates',
        }
    if feasible_count == 0:
        return {
            'decision': 'reject',
            'reason': 'no_strategy_meets_budget_and_quality',
            'next_action': 'rework_checkpoint_or_offload_scope',
        }
    # 显存收益和吞吐保留率都达标，才允许进入 accept 分支。
    meaningful_memory_gain = summary.get('memory_saving_mb', 0.0) >= summary['min_memory_saving_mb']
    acceptable_throughput = summary.get('throughput_ratio') is not None and summary['throughput_ratio'] >= summary['min_throughput_ratio']
    if best_candidate != 'baseline' and meaningful_memory_gain and acceptable_throughput:
        # 只有显存收益和吞吐保留率同时达标，才接受策略变更。
        return {
            'decision': 'accept',
            'reason': 'strategy_is_best_feasible_option',
            'next_action': 'promote_to_training_run',
        }
    if quality_failed_count > 0:
        return {
            'decision': 'tune',
            'reason': 'strategy_needs_quality_recovery',
            'next_action': 'adjust_checkpoint_granularity_or_offload_scope',
        }
    if not meaningful_memory_gain:
        return {
            'decision': 'tune',
            'reason': 'memory_saving_below_meaningful_threshold',
            'next_action': 'test_pressure_or_offload_scope',
        }
    if not acceptable_throughput:
        return {
            'reason': 'throughput_loss_exceeds_budget',
            'decision': 'tune',
            'next_action': 'reduce_checkpoint_or_offload_scope',
        }
    return {
        'decision': 'tune',
        'reason': 'baseline_still_best_under_current_budget',
        'next_action': 'revisit_strategy_mix',
    }

```

### 解析

**1. TODO 1: 检查预算与质量阈值**
- **实现方式**：先把显存上限、吞吐下限和验证损失上限检查齐，再进入方案比较。
- **关键点**：没有统一预算口径时，checkpoint / offload / hybrid 之间的比较都没有解释力。
- **项目意义**：这一步把 `76` 固定成预算约束下的显存策略对比页，而不是泛技巧列表。

**2. TODO 2: 汇总显存策略候选**
- **实现方式**：按 peak memory、samples/s 和 val loss 统一过滤候选，再选出最省显存的可行方案。
- **关键点**：显存收益只有在质量和吞吐都没有跌出边界时，才值得被保留。
- **项目意义**：这一步把 `19 / 42 / 43 / 73` 的机制与测量知识收成真正可比较的工程候选。

**3. TODO 3: 输出项目结论**
- **实现方式**：把候选可行性和最优方案统一收成 `accept / tune / reject`。
- **关键点**：项目结论必须回答“当前预算下哪种显存策略值得继续采用”，而不是只输出一个峰值显存最小值。
- **项目意义**：这一步把 `76` 收成显存优化路线中的正式策略对比项目。

### Step 5（可选）：真实 GPU 显存策略 benchmark

#### 5.1 实验目标与条件

本 Step 复用 73 的训练口径，在真实 GPU 上比较四种激活显存策略。表中先固定实验条件，再说明只改变什么以及要保存哪些结果；随后按 5.2–5.5 的顺序完成检查、配置、运行和记录。环境依赖按维护文档准备；自动安装只补普通 Python 依赖，不替代 CUDA 驱动或 PyTorch wheel。

| 实验要素 | 固定或设置的内容 | 形成的证据 |
|:---|:---|:---|
| 实验对象 | baseline、checkpoint、offload、hybrid | 四种策略的可比较集合 |
| 共同条件 | `Qwen/Qwen2.5-0.5B-Instruct`、FP32、AdamW、batch=1、seq_len=768、seed=42 | `73_real_gpu_training.json` 与当前配置 |
| 变化变量 | 只改变激活值的保存、重算或搬运策略 | 每个策略的重复运行结果 |
| 输出指标 | step time（ms）、吞吐（samples/s）、peak allocated / reserved（MiB）、eval loss、OOM | `76_real_gpu_memory.json` |

**策略与代价口径**

| 策略 | 用什么换显存 | 主要观察点 |
|:---|:---|:---|
| baseline | 不引入额外重算或搬运 | 显存和速度参照 |
| checkpoint | 用额外 forward 计算换激活值驻留 | 重算时间与峰值显存 |
| offload | 用 CPU-GPU 搬运和同步换 GPU 驻留空间 | 搬运时间与峰值显存 |
| hybrid | 同时承担部分重算和搬运代价 | 折中后的速度与显存 |

<div align="center"><strong>先校验 73 的条件，再采集策略指标并保存 JSON。</strong></div>

![76 GPU 策略对比流程](../public/02_PyTorch_Algorithms/76_gpu_strategy_benchmark.svg)


#### 5.2 环境预检

先检查项目路径、Python、PyTorch 和 CUDA。这个单元只报告环境状态，不加载模型、不开始测量；预检未通过时，先根据输出修复环境。

```python
"""GPU benchmark 的独立环境预检：只确认路径和运行时，不安装依赖、不加载模型。"""
# 只检查项目路径、Python、PyTorch 和 CUDA；不安装依赖、不加载模型。
from pathlib import Path
import os
import subprocess
import sys

GPU_PROJECT_ROOT = Path(os.environ.get('LLM_ALGO_PROJECT_ROOT', Path.cwd())).expanduser().resolve()
if not (GPU_PROJECT_ROOT / 'tools/project_runtime.py').is_file():
    colab_root = Path('/content/llm-algo-leetcode')
    if (colab_root / 'tools/project_runtime.py').is_file():
        GPU_PROJECT_ROOT = colab_root
    elif Path('/content').is_dir() and not colab_root.exists():
        subprocess.run(['git', 'clone', 'https://github.com/datawhalechina/llm-algo-leetcode.git', str(colab_root)], check=True)
        GPU_PROJECT_ROOT = colab_root
if not (GPU_PROJECT_ROOT / 'tools/project_runtime.py').is_file():
    raise RuntimeError('找不到项目根目录：请设置 LLM_ALGO_PROJECT_ROOT，或先把仓库放到 /content/llm-algo-leetcode。')
os.chdir(GPU_PROJECT_ROOT)
if str(GPU_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(GPU_PROJECT_ROOT))
print(f'project_root: {GPU_PROJECT_ROOT}')
try:
    import torch
except ImportError:
    print('未检测到 PyTorch；请按维护文档安装匹配的 torch CUDA wheel 后再运行 GPU benchmark。')
else:
    print(f'torch: {torch.__version__}')
    print(f'cuda_build: {torch.version.cuda}; cuda_available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f'gpu: {props.name}; memory_gib: {props.total_memory / 2**30:.1f}')

```

#### 5.3 配置变量

先用 `smoke` 检查流程，再用 `pressure` 进行重复测量。通常只修改运行开关、workload、策略集合和预算阈值；模型、输入和指标计算逻辑保持不变。

```python
# 通常只修改本 cell 的运行开关、WORKLOAD、STRATEGIES 和预算阈值。
# 先用 smoke 检查流程，再用 pressure 进行重复测量；不要修改下一 cell 的测量和 JSON 写入逻辑。
from pathlib import Path

AUTO_INSTALL_REAL_DEPS = True  # 真实 GPU 开启时，只安装当前内核缺失的普通依赖。
AUTO_INSTALL_ALLOW_BREAK_SYSTEM_PACKAGES = True  # 云端 PEP 668 环境允许安装普通依赖；不会重装 PyTorch。
RUN_REAL_GPU = False  # 默认先运行 CPU 正确性检查；GPU benchmark 时显式改为 True。
DTYPE_MODE = 'fp32'  # fp32：主线；bf16：显式扩展实验，启用 CUDA autocast。
MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'  # 固定基座模型。
MODEL_SOURCE = 'auto'  # 模型来源：auto / huggingface / modelscope / local。
MODEL_CACHE_DIR = 'model_cache'  # 模型缓存目录。
BATCH_SIZE = 1  # 实际值由 WORKLOADS 覆盖；增大它会提高 activation 压力。
WORKLOAD = 'pressure'  # pressure(seq_len=768)：与 73 主线对齐；pressure_1024 是长序列扩展。
WORKLOADS = {
    'smoke': {'batch_size': 1, 'seq_len': 512, 'warmup': 2, 'iters': 5},
    'pressure': {'batch_size': 1, 'seq_len': 768, 'warmup': 2, 'iters': 5},
    'pressure_1024': {'batch_size': 1, 'seq_len': 1024, 'warmup': 2, 'iters': 5},
}
WARMUP = 2  # 预热轮数，不计入正式平均值；正式报告建议至少 2。
ITERS = 5  # 每种策略的正式测量轮数；这是教学 smoke，稳定性要求高时应增加。
SEED = 42  # 固定输入，保证策略间 workload 一致；不代表真实数据集质量。
STRATEGIES = ['baseline', 'checkpoint', 'offload', 'hybrid']  # 四种策略在同一 workload 下比较；资源不足时可先删 offload，但要记录。

# 完成 smoke 后，可改为：['baseline', 'checkpoint', 'offload', 'hybrid']
# BF16 / 长序列扩展允许先比较子集，但报告必须保留实际 strategies。
REPEATS = 1 if WORKLOAD == 'smoke' else 3  # smoke 快速检查；pressure 默认重复 3 次，形成可比较的稳定性证据。
MEMORY_CAP_MB = 11200.0  # 硬显存预算；需为系统和桌面进程留余量。
MIN_SAMPLES_PER_S = 1.0  # 最低吞吐；低于此值的策略判为不可行。
# None 表示按当前固定 workload 的 baseline eval_loss 自动生成质量上限
MAX_VAL_LOSS = None  # 只是质量代理门槛，不是完整任务质量门槛。
# 主线与 73 的 FP32 / pressure 报告对齐；BF16 / seq1024 扩展时显式修改这两条路径。
BASELINE_73_RELATIVE_PATH = Path('benchmarks/results/73_real_gpu_training.json')
OUTPUT_RELATIVE_PATH = Path('benchmarks/results/76_real_gpu_memory.json')

```

#### 5.4 运行与保存

运行 cell 会按当前配置加载模型、测量四种策略并写入 JSON；每个策略使用相同 workload，正式结论应查看重复运行、稳定性和 OOM 状态，而不是只看一次结果。若口径不一致或预检失败，程序会在加载模型前停止，相关记录统一放在 5.5 的证据状态中。

```python
# 按配置完成环境预检、模型准备、策略测量、重复运行和 JSON 写入。
# RUN_REAL_GPU=False 时只输出跳过信息；运行后查看 environment_preflight、config、candidates[*].runs、stability 和 decision。
import json
import gc
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

# 先定位仓库并加入 sys.path，再导入项目工具；避免 Colab 直接打开时找不到 tools。
PROJECT_ROOT = Path(os.environ.get('LLM_ALGO_PROJECT_ROOT', Path.cwd())).expanduser().resolve()
if not (PROJECT_ROOT / 'tools/project_runtime.py').is_file():
    colab_root = Path('/content/llm-algo-leetcode')
    if (colab_root / 'tools/project_runtime.py').is_file():
        PROJECT_ROOT = colab_root
    else:
        for candidate in (PROJECT_ROOT, *PROJECT_ROOT.parents):
            if (candidate / 'tools/project_runtime.py').is_file():
                PROJECT_ROOT = candidate
                break
        else:
            if Path('/content').is_dir() and not colab_root.exists():
                subprocess.run(['git', 'clone', 'https://github.com/datawhalechina/llm-algo-leetcode.git', str(colab_root)], check=True)
            if (colab_root / 'tools/project_runtime.py').is_file():
                PROJECT_ROOT = colab_root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from tools.project_runtime import ensure_output_path, resolve_project_root, environment_preflight, runtime_snapshot, standard_experiment_config, standard_training_metrics, validate_training_config
from tools.training_memory_runtime import measure_training_run
from tools.memory_strategy_runtime import (
    summarize_memory_strategy_candidates,
    decide_memory_strategy_project,
)

PROJECT_ROOT = resolve_project_root(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_PATH = ensure_output_path(PROJECT_ROOT, OUTPUT_RELATIVE_PATH)
print(f'项目根目录: {PROJECT_ROOT}')
print(f'结果保存路径: {OUTPUT_PATH}')

if RUN_REAL_GPU:
    import torch
    if AUTO_INSTALL_REAL_DEPS:
        import importlib.util
        missing = [name for name in ('transformers',) if importlib.util.find_spec(name) is None]
        if missing:
            install_cmd = [sys.executable, '-m', 'pip', 'install', '-U', *missing]
            markers = [Path(sys.prefix) / 'EXTERNALLY-MANAGED', Path(sys.executable).parent.parent / 'EXTERNALLY-MANAGED']
            if any(marker.is_file() for marker in markers):
                if not AUTO_INSTALL_ALLOW_BREAK_SYSTEM_PACKAGES:
                    raise RuntimeError('检测到 PEP 668 受管 Python，请启用 AUTO_INSTALL_ALLOW_BREAK_SYSTEM_PACKAGES 或改用独立虚拟环境。')
                install_cmd[3:3] = ['--break-system-packages']
            print('使用当前 Notebook 内核安装缺失依赖：', missing)
            subprocess.check_call(install_cmd)
            print('依赖安装完成；如当前内核仍找不到 transformers，请重启内核后继续。')
    valid_dtype_modes = {'fp32', 'bf16'}
    valid_strategies = {'baseline', 'checkpoint', 'offload', 'hybrid'}
    if DTYPE_MODE not in valid_dtype_modes:
        raise ValueError(f'DTYPE_MODE 必须是 {sorted(valid_dtype_modes)} 之一，当前为 {DTYPE_MODE!r}')
    if not STRATEGIES or len(set(STRATEGIES)) != len(STRATEGIES) or not set(STRATEGIES).issubset(valid_strategies):
        raise ValueError(f'STRATEGIES 必须是 valid_strategies 的非空子集且不能重复，当前为 {STRATEGIES!r}')
    if MEMORY_CAP_MB <= 0 or MIN_SAMPLES_PER_S < 0:
        raise ValueError('MEMORY_CAP_MB 必须大于 0，MIN_SAMPLES_PER_S 不能小于 0。')
    if REPEATS < 1:
        raise ValueError('REPEATS 必须至少为 1。')
    from tools.model_runtime import resolve_model
    from transformers import AutoConfig, AutoModelForCausalLM

    if WORKLOAD not in WORKLOADS:
        raise ValueError(f'未知 workload: {WORKLOAD}，可选值：{sorted(WORKLOADS)}')
    workload_config = WORKLOADS[WORKLOAD]
    BATCH_SIZE = workload_config['batch_size']
    SEQ_LEN = workload_config['seq_len']
    WARMUP = workload_config['warmup']
    ITERS = workload_config['iters']
    validate_training_config({'batch_size': BATCH_SIZE, 'seq_len': SEQ_LEN, 'warmup': WARMUP, 'iters': ITERS, 'seed': SEED})
    preflight = environment_preflight(torch, required_packages=('transformers',), require_gpu=True, output_path=OUTPUT_PATH)
    print({'environment_preflight': preflight})
    if not preflight['ready']:
        raise RuntimeError('环境预检未通过，请先按 next_actions 修复；没有加载模型。')
    print({'runtime': runtime_snapshot(torch)})
    if not torch.cuda.is_available():
        raise RuntimeError('RUN_REAL_GPU=True 但 CUDA 不可用。')

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device('cuda')
    baseline_report_path = PROJECT_ROOT / BASELINE_73_RELATIVE_PATH
    if not baseline_report_path.exists():
        raise FileNotFoundError(f'找不到 73 baseline：{baseline_report_path}，请先运行 73 的相同 workload。')
    baseline_report = json.loads(baseline_report_path.read_text(encoding='utf-8'))
    baseline_config = baseline_report.get('config', {})
    expected_baseline = {
        'model_id': MODEL_ID, 'batch_size': BATCH_SIZE, 'seq_len': SEQ_LEN,
        'dtype': 'float32', 'optimizer': 'AdamW', 'workload': WORKLOAD,
        'warmup': WARMUP, 'iters': ITERS,
    }
    if DTYPE_MODE == 'bf16':
        expected_baseline['amp_dtype'] = 'torch.bfloat16'
        if baseline_config.get('mode') != 'bf16_probe':
            raise ValueError('BF16 扩展实验需要 73 的 bf16_probe 报告。')
    mismatches = {
        key: {'baseline': baseline_config.get(key), 'current': value}
        for key, value in expected_baseline.items()
        if baseline_config.get(key) != value
    }
    if mismatches:
        raise ValueError(f'73 与 76 的 workload 口径不一致，请先重跑 73：{mismatches}')
    model_path = resolve_model(MODEL_ID, source=MODEL_SOURCE, cache_dir=MODEL_CACHE_DIR)
    print(f'模型路径: {model_path}')
    model_config = AutoConfig.from_pretrained(model_path)
    generator = torch.Generator(device='cpu').manual_seed(SEED)
    shared_input_ids_cpu = torch.randint(
        0, model_config.vocab_size, (BATCH_SIZE, SEQ_LEN), generator=generator
    )
    eval_input_ids_cpu = torch.randint(
        0, model_config.vocab_size, (BATCH_SIZE, SEQ_LEN), generator=generator
    )

    def strategy_context(name):
        if name in {'offload', 'hybrid'}:
            return torch.autograd.graph.save_on_cpu(pin_memory=False)
        return nullcontext()

    def run_strategy(name):
        torch.manual_seed(SEED)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
        model.config.use_cache = False
        if name in {'checkpoint', 'hybrid'}:
            model.gradient_checkpointing_enable()
        model.to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        input_ids = shared_input_ids_cpu.to(device)
        labels = input_ids.clone()
        eval_input_ids = eval_input_ids_cpu.to(device)
        eval_labels = eval_input_ids.clone()

        def train_step():
            optimizer.zero_grad(set_to_none=True)
            with strategy_context(name):
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=DTYPE_MODE == 'bf16'):
                    loss = model(input_ids=input_ids, labels=labels).loss
                loss.backward()
            optimizer.step()
            return float(loss.detach().item())

        result = measure_training_run(
            train_step, torch_module=torch, batch_size=BATCH_SIZE,
            warmup=WARMUP, iters=ITERS,
        )
        model.eval()
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=DTYPE_MODE == 'bf16'):
                eval_loss = float(model(input_ids=eval_input_ids, labels=eval_labels).loss.item())
        result = {
            'name': name,
            'status': 'ok',
            'step_time_ms': result['step_time_ms'],
            'samples_per_s': result['samples_per_s'],
            'loss': result['loss'],
            'eval_loss': round(eval_loss, 6),
            'peak_memory_mb': result['peak_mem_mb'],
            'peak_reserved_mb': result['peak_reserved_mb'],
        }
        del optimizer, model, input_ids, labels, eval_input_ids, eval_labels
        gc.collect()
        torch.cuda.empty_cache()
        return result

    raw_run_strategy = run_strategy
    def run_strategy(name):
        try:
            return raw_run_strategy(name)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            return {
                'name': name,
                'status': 'oom',
                'error': str(exc).split('\n')[0],
            }

    def run_strategy_repeated(name):
        runs = [run_strategy(name) for _ in range(REPEATS)]
        successful = [item for item in runs if item.get('status') == 'ok']
        if not successful:
            return {**runs[0], 'runs': runs}
        numeric_keys = ('step_time_ms', 'samples_per_s', 'loss', 'eval_loss', 'peak_memory_mb', 'peak_reserved_mb')
        aggregated = {
            key: round(sum(item[key] for item in successful) / len(successful), 3)
            for key in numeric_keys
        }
        def value_range(key):
            values = [item[key] for item in successful]
            return round(max(values) - min(values), 3)
        stability = {
            'step_time_range_ms': value_range('step_time_ms'),
            'throughput_range_samples_per_s': value_range('samples_per_s'),
            'peak_memory_range_mb': value_range('peak_memory_mb'),
            'successful_runs': len(successful),
            'oom_runs': len(runs) - len(successful),
        }
        status = 'ok' if len(successful) == len(runs) else 'partial_oom'
        return {**successful[0], **aggregated, 'status': status, 'runs': runs, 'successful_runs': len(successful), 'oom_runs': len(runs) - len(successful), 'stability': stability}

    candidates = [run_strategy_repeated(name) for name in STRATEGIES]
    budget = {'memory_cap_mb': MEMORY_CAP_MB, 'min_samples_per_s': MIN_SAMPLES_PER_S, 'min_memory_saving_mb': 512.0, 'min_throughput_ratio': 0.70}
    baseline_candidate = next((item for item in candidates if item.get('name') == 'baseline' and item.get('status', 'ok') == 'ok'), None)
    if MAX_VAL_LOSS is None and baseline_candidate is not None:
        max_eval_loss = baseline_candidate['eval_loss'] * 1.02
    elif MAX_VAL_LOSS is None:
        max_eval_loss = float('inf')
    else:
        max_eval_loss = MAX_VAL_LOSS
    quality_floor = {'max_val_loss': round(max_eval_loss, 6)}
    summary = summarize_memory_strategy_candidates(candidates, budget, quality_floor)
    decision = decide_memory_strategy_project(summary)
    result = {
        'task': 'task3_training_memory_optimization',
        'environment_preflight': preflight,
        'stage': 'activation_checkpoint_offload_benchmark',
        'source_baseline': str(baseline_report_path.relative_to(PROJECT_ROOT)),
        'baseline_validation': {'status': 'matched', 'config': expected_baseline},
        'evidence_level': 'fixed_workload_strategy_smoke' if DTYPE_MODE == 'fp32' else 'bf16_capacity_strategy_smoke',
        'config': {
            'model_id': MODEL_ID, 'workload': WORKLOAD, 'batch_size': BATCH_SIZE, 'seq_len': SEQ_LEN,
            'dtype': 'float32', 'amp_dtype': 'torch.bfloat16' if DTYPE_MODE == 'bf16' else None, 'mode': DTYPE_MODE, 'optimizer': 'AdamW',
            'warmup': WARMUP, 'iters': ITERS, 'strategies': STRATEGIES,
            'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
            'device': torch.cuda.get_device_name(0), 'seed': SEED,
        },
        'budget': budget, 'quality_floor': quality_floor,
        'candidates': candidates, 'summary': summary, 'decision': decision,
        'repeat_summary': {
            item['name']: {
                'status': item.get('status'),
                'successful_runs': item.get('successful_runs', 0),
                'oom_runs': item.get('oom_runs', 0),
                'stability': item.get('stability'),
            }
            for item in candidates
        },
    }
    result['experiment'] = standard_experiment_config(result['config'])
    result['standard_metrics'] = {item['name']: standard_training_metrics(item) for item in candidates}
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
else:
    print('跳过真实 GPU benchmark：保持 CPU-first 模式。')

```

#### 5.5 读取与复测

先运行上一个 benchmark cell，再运行下面的读取 cell。它只读取结果文件，不会启动模型、不重复测量，也不会覆盖 JSON。读取 JSON 后，先核对实际运行环境和统一口径，再查看已保存结果和学习者复测位置。


```python
# 结果读取 cell：把 JSON 中的配置、重复运行和聚合指标打印成可抄录的表格。
# 如果文件不存在，先回到配置 cell 和 benchmark cell；不要手动创建空 JSON。
import json
from pathlib import Path

result_path = PROJECT_ROOT / OUTPUT_RELATIVE_PATH
if not result_path.exists():
    raise FileNotFoundError(f'找不到结果文件：{result_path}。请先运行 GPU benchmark cell。')
report = json.loads(result_path.read_text(encoding='utf-8'))
print('结果文件:', result_path)
print('实验配置:', report.get('config', {}))
print('决策:', report.get('decision', {}))

rows = []
for candidate in report.get('candidates', []):
    rows.append({
        '策略': candidate.get('name'),
        '状态': candidate.get('status'),
        'runs': candidate.get('runs', []),
        '平均 step time(ms)': candidate.get('step_time_ms'),
        'step time 范围(ms)': candidate.get('stability', {}).get('step_time_range_ms'),
        '平均吞吐(samples/s)': candidate.get('samples_per_s'),
        '吞吐范围(samples/s)': candidate.get('stability', {}).get('throughput_range_samples_per_s'),
        '峰值显存(MiB)': candidate.get('peak_memory_mb'),
        '成功次数': candidate.get('successful_runs', 0),
        'OOM 次数': candidate.get('oom_runs', 0),
    })
for row in rows:
    print(row)

```

**实测环境与统一口径**

| 项目 | 实际配置 |
|:---|:---|
| 系统 / GPU | Linux `6.8.0-138-generic`、RTX 5070 Ti Laptop GPU / 12227 MiB |
| 驱动 | `570.211.01` |
| PyTorch / CUDA | `2.11.0+cu128` / `12.8` |
| 模型 | `Qwen/Qwen2.5-0.5B-Instruct` |
| workload | `pressure`：batch=1、seq_len=768 |
| 训练设置 | FP32、AdamW、learning rate=1e-5、warmup=2、iters=5、seed=42 |
| 预算 | 显存上限 11200 MiB、吞吐下限 1 sample/s |
| 输出文件 | `benchmarks/results/76_real_gpu_memory.json` |

**主线结果与复测记录**

本地历史结果使用 FP32 / `seq_len=768`；下面展示 JSON 中的汇总值，并保留 Run 1–3 和学习者复测位置。原始文件为 `benchmarks/results/76_real_gpu_memory.json`。

Run 1–3 用于填写逐次结果，均值和范围来自 JSON 中的 `runs` 字段；当前均值标注为历史值，少于 3 次完整运行时应标记为 `gpu_smoke`。

| 策略 | Run 1 | Run 2 | Run 3 | 平均 step time（ms） | step time 范围（ms） | 平均吞吐（samples/s） | 吞吐范围（samples/s） | 峰值显存（MiB） | eval loss | 状态 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| baseline | 待填写 | 待填写 | 待填写 | 497.839（历史） | 待填写 | 2.009（历史） | 待填写 | 9782.74 | 12.356503 | 待复核 |
| checkpoint | 待填写 | 待填写 | 待填写 | 567.740（历史） | 待填写 | 1.761（历史） | 待填写 | 9450.76 | 12.356503 | 待复核 |
| offload | 待填写 | 待填写 | 待填写 | 1877.559（历史） | 待填写 | 0.533（历史） | 待填写 | 9448.33 | 12.356503 | 待复核 |
| hybrid | 待填写 | 待填写 | 待填写 | 783.478（历史） | 待填写 | 1.276（历史） | 待填写 | 9454.64 | 12.356503 | 待复核 |

下表用于填写自己的硬件或其他 workload；策略比较必须复用同一模型、输入、训练条件和重复次数。

| 策略 | GPU / 显存 | 模型 | dtype | batch | seq_len | repeats | step time (ms) | throughput (samples/s) | peak allocated (MiB) | peak reserved (MiB) | eval loss | 状态 | evidence level |
|:---|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---|:---|
| baseline | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| checkpoint | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| offload | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| hybrid | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |


#### 5.6 解释与决策

**证据状态与失败记录**

口径检查会在加载模型前执行；以下记录说明一次 workload 不一致，不属于 benchmark 结果。

| 检查项 | 73 已保存结果 | 76 当前配置 | 处理动作 |
|:---|:---|:---|:---|
| workload | `pressure_1024` | `pressure` | 报错并停止，先让两节使用同一 workload |
| 检查目的 | 序列长度和压力条件可能不同 | 需要复用 73 baseline | 重跑 73，或同步修改 76 后再运行 |
| 结论状态 | 不能直接作为对照 | 尚未开始测量 | 不生成本次策略比较结论 |

**主线策略判断**

本地结果显示：checkpoint 比 baseline 少占 331.98 MiB 显存，但吞吐下降约 12.3%；offload 和 hybrid 的速度代价更高，四种策略的 eval loss 一致。当前没有方案达到 512 MiB 显存收益阈值，因此结论为 `tune`；这些结果只适用于本模型、序列长度和全参数训练口径。

**扩展实测：BF16 / seq_len=1024**

该组实验改变了 dtype 和序列长度，因此单独作为扩展证据，不与主线结果合并。配置为 `Qwen/Qwen2.5-0.5B-Instruct`、batch=1、BF16 autocast、AdamW，仅比较 baseline 与 checkpoint。
这组实测表明 BF16 使 `seq_len=1024` 在当前 12GB GPU 上成功运行；checkpoint 额外节省约 27.16 MiB，吞吐保留约 86.7%。它只说明当前 workload 下的结果，不代表所有长序列任务。76 提供策略实测证据，75 再据此进行预算决策。

| 策略 | step time | throughput | peak allocated | peak reserved | eval loss | 状态 |
|:---|---:|---:|---:|---:|---:|:---|
| baseline | 297.060 ms | 3.366 samples/s | 10037.52 MiB | 10542.00 MiB | 12.202896 | ok |
| checkpoint | 342.564 ms | 2.919 samples/s | 10010.36 MiB | 10876.00 MiB | 12.202565 | ok |


---
## 相关阅读

以下资料按“训练显存机制 → 官方实现 → 项目决策”排列，用于把 checkpoint、offload 和重算代价连接到实际训练系统。

- [Training Deep Nets with Sublinear Memory Cost 论文](https://arxiv.org/abs/1604.06174)
- [ZeRO-Offload 论文：民主化大模型训练](https://arxiv.org/abs/2101.06840)
- [PyTorch `torch.utils.checkpoint` 文档](https://pytorch.org/docs/stable/checkpoint.html)
- [43. Unified Memory Management | 统一内存管理](./43_Unified_Memory_Management.md)
- [73 训练性能分析](./73_Training_Performance_Analysis.md)
- [75 显存预算压缩项目](./75_Memory_Budget_Compression_Project.md)
- [74 Profiling 驱动的端到端优化](./74_Profiling_Driven_End_to_End_Optimization.md)
