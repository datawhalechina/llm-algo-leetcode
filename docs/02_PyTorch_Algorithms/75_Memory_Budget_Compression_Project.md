# 75. Memory Budget Compression Project | 显存预算压缩项目

**难度：** Hard | **环境：** CPU-only；读取 GPU 项目报告 | **标签：** `显存优化`, `预算规划`, `压缩策略` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/75_Memory_Budget_Compression_Project.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

显存预算不是单纯追求峰值越低，而是在固定模型、数据和 workload 下，判断 checkpoint、offload 等方案能否在显存、吞吐和质量之间达到目标。本节带你从训练报告中提取可比较字段，检查证据是否适用，分析预算阈值变化对候选方案的影响，最后形成 `accept / tune / reject` 决策。

**关键词：** `memory`, `budget`, `checkpoint`, `offload`, `project`

---
## 前置阅读

**导语：** 先了解激活检查点和卸载如何改变训练状态，再查看训练测量与 checkpoint / offload 对照结果，最后进入本节的显存预算筛选与决策。
- [19. Activation Checkpointing | 激活检查点](./19_Activation_Checkpointing.md)
- [42. Activation Offload | 激活卸载](./42_Activation_Offload.md)
- [73. Training Performance Analysis | 训练性能分析](./73_Training_Performance_Analysis.md)
- [76. Activation Checkpoint Offload Benchmark | Activation / Checkpoint / Offload 对比项目](./76_Activation_Checkpoint_Offload_Benchmark.md)
---

### Step 1：定义预算问题与比较输入
在有限显存下，哪些策略仍能满足吞吐和质量要求？先明确要比较哪些候选、什么条件算可行，以及报告中必须有哪些字段。

| 先固定什么 | 具体填写 | 用来回答什么 |
|:---|:---|:---|
| 输入证据 | GPU 项目报告 JSON 以及对应的模型、数据、dtype、batch、seq_len、硬件 | 候选是否来自同一 workload |
| 预算门槛 | `memory_cap_mb`、`min_samples_per_s`、`min_memory_saving_mb`、`min_throughput_ratio`、`max_val_loss` | 什么条件下方案才算可行 |
| 候选集合 | `baseline`、`checkpoint`、`offload`、`hybrid` | 这次准备比较哪些方案 |
| 输出字段 | `feasible_names`、`best_candidate`、`memory_saving_mb`、`throughput_ratio` | 为后续比较准备统一字段 |


预算不足时，不应直接把所有策略叠加，而要按峰值来源和代价安排裁剪顺序：先处理可解释、可回退的 activation 策略，再评估搬运、batch / sequence length 或模型规模变化。

| 候选动作 | 优先观察 | 主要代价 |
|:---|:---|:---|
| checkpoint | activation 峰值与额外重算时间 | 计算时间增加 |
| offload | 可迁移状态与传输路径 | 搬运、同步和调度开销 |
| batch / sequence length | workload 是否允许缩小 | 吞吐、质量或服务能力变化 |
| 模型规模 / dtype | 是否仍满足任务质量门槛 | 能力、精度或部署兼容性变化 |

![75 显存预算决策流程](../public/02_PyTorch_Algorithms/75_budget_decision_flow.svg)

### Step 2：判断证据是否适用
先把手里的证据按用途整理，判断它能否用于当前比较：`measured` 支持报告对应硬件和 workload 下的判断，`projected` 用来规划 batch、seq_len 和候选策略的下一次实验。遇到配置变化时，先标记证据适用范围，再决定是否补测。

下表把“当前能回答什么”和“还要补什么”放在一起。

| 证据模式 | 现在能回答什么 | 还需要补什么 |
|:---|:---|:---|
| `measured` | 支持报告对应硬件和 workload 下的预算比较 | 记录报告中的模型、dtype、batch、seq_len 和硬件，避免误用 |
| `projected` | 根据显存账本规划 batch、seq_len 或候选策略 | 不能证明吞吐、kernel 或 OOM 边界；需要补 GPU 实测 |
| 跨硬件或跨 workload | 只能把旧报告当作规划参考 | GPU、模型、dtype、数据或 seq_len 改变时，重新建立同口径比较 |



### Step 3：用阈值敏感性检查预算结论
固定一份有效报告，只改变预算阈值，观察可行候选集合、最佳候选和项目结论是否变化。这样可以判断结论是稳定收益，还是只在某个阈值下成立。



| 检查项 | 调整方式 | 观察结果 |
|:---|:---|:---|
| 显存上限 | 逐步收紧 `memory_cap_mb` | 可行候选和最佳候选是否改变 |
| 吞吐下限 | 逐步提高 `min_throughput_ratio` | 哪些策略因速度门槛被淘汰 |
| 质量上限 | 调整 `max_val_loss` | 质量约束是否改变候选排序 |
| 结论稳定性 | 综合比较不同阈值下的结果 | 判断预算结论是否稳定 |

![75 预算敏感性流程](../public/02_PyTorch_Algorithms/75_sensitivity_flow.svg)

### Step 4：实现预算筛选与项目决策
现在进入代码实现。请补全 3 个函数：先校验预算字段，再筛选候选并计算收益，最后根据摘要输出项目决策。代码围绕已有报告字段完成预算判断，把实验结果转换成可复查的项目结论。

| 类型 | 函数 | 需要补全的逻辑 | 运行后检查 |
|:---|:---|:---|:---|
| TODO 1 | `validate_memory_budget` | 检查字段是否缺失、数值是否合法 | `is_valid`、`missing_keys`、`invalid_keys` |
| TODO 2 | `summarize_memory_strategies` | 处理 OOM、重复和无效指标，筛选可行候选并计算相对 baseline 的收益 | `evaluations`、淘汰计数、`feasible_names`、`best_candidate`、显存节省和吞吐保留率 |
| TODO 3 | `decide_memory_budget_project` | 依次检查 baseline、可行候选和显存/吞吐阈值 | `decision`、`reason`、`next_action` |



```python
from typing import Dict, List

```


```python
# 3 个核心 TODO：预算检查、候选汇总、项目结论
# 目标：把 baseline 与压缩策略的显存预算比较收束成一份项目报告

import math
from typing import Dict, List

def validate_memory_budget(budget: Dict[str, float], quality_floor: Dict[str, float]) -> Dict[str, object]:
    """检查预算字段和值，返回校验状态、缺失字段和非法字段。"""
    # ==========================================
    # TODO 1：完成字段完整性、数值有限性和取值范围校验。
    # 提示：先列出两组必需字段，再分别记录缺失字段和非法字段；不要为缺失字段补默认值。
    # 显存上限、最低收益必须非负，吞吐保留率必须位于 0～1。
     # 缺失字段不能补默认值；显存上限、最低收益必须非负，吞吐保留率必须位于 0～1。
    # ==========================================
    # required_budget_keys = ???
    # missing_keys = ???
    # invalid_keys = ???
    raise NotImplementedError("请先完成 TODO 代码！")

def summarize_memory_strategies(candidates: List[Dict[str, object]], budget: Dict[str, float], quality_floor: Dict[str, float]) -> Dict[str, object]:
    """按显存、吞吐和质量门槛汇总候选策略，并保留淘汰原因。"""
    # ==========================================
    # TODO 2：完成候选筛选、可行候选排序和相对 baseline 的收益计算。
    # 提示：先处理 OOM、重复名称和无效指标，再用三个布尔门槛筛选；最后按显存、吞吐和质量排序。
    # 先区分 OOM、重复名称和无效指标，再检查显存、吞吐、质量三个门槛。
    # 候选字段：name、status、peak_memory_mb、samples_per_s、eval_loss / val_loss。
    # baseline 只作为收益参照，不要把它当成排序键。
    # ==========================================

    # evaluations = ???
    # feasible = ???
    # memory_ok / speed_ok / quality_ok = ???
    raise NotImplementedError("请先完成 TODO 代码！")

def decide_memory_budget_project(summary: Dict[str, object]) -> Dict[str, object]:
    """根据可行候选、最低显存收益和吞吐保留率输出项目决策。"""
    # ==========================================
    # TODO 3：按 baseline、可行候选和收益阈值的顺序输出决策。
    # 提示：先检查 baseline 和可行候选，再比较显存收益与吞吐保留率；返回三个项目结论字段。
    # 不要硬编码具体策略名称。
    # ==========================================
    # feasible_count = ???
    # meaningful_memory_gain / acceptable_throughput = ???
    # decision / reason / next_action = ???
    raise NotImplementedError("请先完成 TODO 代码！")

```


```python
# 测试你的实现
def test_memory_budget_project():
    try:
        budget = {'memory_cap_mb': 12000.0, 'min_samples_per_s': 6.0, 'min_memory_saving_mb': 512.0, 'min_throughput_ratio': 0.70}
        quality_floor = {'max_val_loss': 1.15}
        check = validate_memory_budget(budget, quality_floor)
        assert check['is_valid'] is True, '预算检查应通过'
        assert check['missing_keys'] == [], '完整预算不应缺字段'
        missing = validate_memory_budget({'memory_cap_mb': 12000.0}, quality_floor)
        assert missing['is_valid'] is False, '缺少吞吐门槛时应拒绝预算配置'
        assert 'min_samples_per_s' in missing['missing_keys'], '应指出缺少的预算字段'
        invalid = validate_memory_budget(
            {'memory_cap_mb': -1.0, 'min_samples_per_s': 6.0, 'min_memory_saving_mb': 512.0, 'min_throughput_ratio': 1.2},
            {'max_val_loss': 1.15},
        )
        assert invalid['is_valid'] is False, '非法预算值应被拒绝'
        assert {'memory_cap_mb', 'min_throughput_ratio'} <= set(invalid['invalid_keys'])

        candidates = [
            {'name': 'baseline', 'peak_memory_mb': 18000.0, 'samples_per_s': 8.0, 'val_loss': 1.06},
            {'name': 'checkpoint', 'peak_memory_mb': 11800.0, 'samples_per_s': 6.5, 'val_loss': 1.09},
            {'name': 'offload', 'peak_memory_mb': 9800.0, 'samples_per_s': 4.5, 'val_loss': 1.08},
        ]
        summary = summarize_memory_strategies(candidates, budget, quality_floor)
        assert summary['feasible_count'] == 1, '只应有一个方案满足预算与质量'
        assert summary['best_candidate'] == 'checkpoint', 'checkpoint 应成为最优可行方案'
        evaluations = {item['name']: item for item in summary['evaluations']}
        assert evaluations['baseline']['reasons'] == ['memory_over_budget'], '应记录 baseline 的显存淘汰原因'
        assert evaluations['offload']['reasons'] == ['throughput_below_floor'], '应记录 offload 的吞吐淘汰原因'

        decision = decide_memory_budget_project(summary)
        assert decision['decision'] == 'accept', '可行且最优的方案应被接受'
        custom_summary = summarize_memory_strategies(
            [
                {'name': 'baseline', 'peak_memory_mb': 18000.0, 'samples_per_s': 8.0, 'val_loss': 1.06},
                {'name': 'activation_checkpoint', 'peak_memory_mb': 11800.0, 'samples_per_s': 6.5, 'val_loss': 1.09},
            ],
            budget,
            quality_floor,
        )
        assert decide_memory_budget_project(custom_summary)['decision'] == 'accept', '自定义压缩策略也应可被接受'

        hard_summary = summarize_memory_strategies(
            [
                {'name': 'checkpoint', 'peak_memory_mb': 13000.0, 'samples_per_s': 6.2, 'val_loss': 1.10},
                {'name': 'offload', 'peak_memory_mb': 11000.0, 'samples_per_s': 5.0, 'val_loss': 1.20},
            ],
            budget,
            quality_floor,
        )
        hard_decision = decide_memory_budget_project(hard_summary)
        assert hard_decision['decision'] == 'reject', '没有满足预算与质量时应 reject'

        edge_summary = summarize_memory_strategies(
            [
                {'name': 'oom_candidate', 'status': 'oom'},
                {'name': 'incomplete', 'peak_memory_mb': 10000.0},
            ],
            budget,
            quality_floor,
        )
        assert edge_summary['oom_count'] == 1, 'OOM 候选应单独计数'
        assert edge_summary['invalid_count'] == 1, '缺少指标的候选应标记为 invalid'
        assert decide_memory_budget_project(edge_summary)['decision'] == 'reject', '没有可行策略时应 reject'

        no_baseline = summarize_memory_strategies(
            [{'name': 'checkpoint', 'peak_memory_mb': 10000.0, 'samples_per_s': 7.0, 'eval_loss': 1.05}],
            budget,
            quality_floor,
        )
        assert decide_memory_budget_project(no_baseline)['reason'] == 'baseline_missing_or_invalid'

        invalid_baseline = summarize_memory_strategies(
            [
                {'name': 'baseline', 'peak_memory_mb': float('nan'), 'samples_per_s': 8.0, 'eval_loss': 1.06},
                {'name': 'activation_checkpoint', 'peak_memory_mb': 11800.0, 'samples_per_s': 6.5, 'eval_loss': 1.09},
            ],
            budget,
            quality_floor,
        )
        assert invalid_baseline['baseline_available'] is False, 'NaN baseline 不应进入收益计算'

        malformed = summarize_memory_strategies(
            [
                {'name': 'checkpoint', 'peak_memory_mb': float('nan'), 'samples_per_s': 7.0, 'eval_loss': 1.05},
                {'name': 'checkpoint', 'peak_memory_mb': 10000.0, 'samples_per_s': 7.0, 'eval_loss': 1.05},
            ],
            budget,
            quality_floor,
        )
        assert malformed['invalid_count'] == 2, 'NaN 和重复候选都应被标记为 invalid'

        small_gain = summarize_memory_strategies(
            [
                {'name': 'baseline', 'peak_memory_mb': 12000.0, 'samples_per_s': 8.0, 'eval_loss': 1.06},
                {'name': 'checkpoint', 'peak_memory_mb': 11700.0, 'samples_per_s': 7.0, 'eval_loss': 1.07},
            ],
            budget,
            quality_floor,
        )
        assert decide_memory_budget_project(small_gain)['decision'] == 'tune', '显存收益不足时应 tune'
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


test_memory_budget_project()

```

🛑 **STOP HERE** 🛑

## 参考代码与解析

### 完整参考实现
答案区保留一份可独立运行的内联实现，用于对照题目区 TODO；Step 5 的项目执行代码会调用同逻辑的公共 runtime。


```python
# 完整参考实现：与题目区的 3 个 TODO 一一对应；不替代 Step 5 的公共 runtime。
import math
from typing import Dict, List

def validate_memory_budget(budget: Dict[str, float], quality_floor: Dict[str, float]) -> Dict[str, object]:
    """检查预算字段和值，返回校验状态、缺失字段和非法字段。

    不为缺失字段补默认值；显存、吞吐、收益和质量阈值的单位由输入报告约定。
    """
    # ==========================================
    # TODO 1 对应变量 required_budget_keys：收集预算必需字段。
    # TODO 1 对应变量 missing_keys / invalid_keys：分别记录缺失字段和非法字段。
    # 提示：required_budget_keys / required_quality_keys 定义输入契约；
    # missing_keys 记录缺失字段，numeric_values 汇总待检查值，invalid_keys 记录非法字段。
    # ==========================================
    required_budget_keys = ['memory_cap_mb', 'min_samples_per_s', 'min_memory_saving_mb', 'min_throughput_ratio']  # 预算必需字段
    required_quality_keys = ['max_val_loss']
    missing_keys = [key for key in required_budget_keys if key not in budget]  # 缺失字段不参与数值校验
    missing_keys += [key for key in required_quality_keys if key not in quality_floor]
    numeric_values = {key: budget.get(key) for key in required_budget_keys}
    numeric_values.update({key: quality_floor.get(key) for key in required_quality_keys})
    # numeric_values 统一收集待检查值，invalid_keys 只记录字段名，便于测试和报告读取。
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


def summarize_memory_strategies(candidates: List[Dict[str, object]], budget: Dict[str, float], quality_floor: Dict[str, float]) -> Dict[str, object]:
    """筛选候选策略，保留淘汰原因，并计算相对 baseline 的收益。

    OOM、重复名称和缺失指标分别记录，不把失败实验伪装成零值。
    """
    # ==========================================
    # TODO 2 对应变量 evaluations / feasible：记录候选状态并保留可行候选。
    # TODO 2 对应派生判断 memory_ok / speed_ok / quality_ok：对应显存、吞吐和质量门槛。
    # 提示：先处理 OOM、重复名称和无效指标，再用 memory_ok / speed_ok / quality_ok 筛选。
    # baseline 只用于计算显存节省和吞吐保留率，不参与候选排序。
    # ==========================================
    feasible: List[Dict[str, object]] = []  # 通过全部门槛的候选
    quality_failed = 0
    invalid_count = 0
    oom_count = 0
    evaluations = []
    seen_names = set()

    for candidate in candidates:
        # 先处理 OOM、重复名称和无效指标，再检查三个预算门槛。
        name = candidate.get('name')
        if not isinstance(name, str) or not name or name in seen_names:
            invalid_count += 1
            evaluations.append({'name': name, 'status': 'invalid', 'reasons': ['missing_or_duplicate_name']})
            continue
        seen_names.add(name)
        if candidate.get('status', 'ok') == 'oom':
            oom_count += 1
            evaluations.append({'name': candidate.get('name'), 'status': 'oom', 'reasons': ['oom']})
            continue
        memory = candidate.get('peak_memory_mb')
        throughput = candidate.get('samples_per_s')
        eval_loss = candidate.get('eval_loss', candidate.get('val_loss'))
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (memory, throughput, eval_loss)):
            invalid_count += 1
            evaluations.append({'name': candidate.get('name'), 'status': 'invalid', 'reasons': ['missing_or_non_numeric_metric']})
            continue
        # 三个布尔变量分别对应显存、吞吐和质量门槛，便于生成可解释的 reasons。
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
        evaluations.append({'name': candidate.get('name'), 'status': 'feasible' if not reasons else 'rejected', 'reasons': reasons})
        if memory_ok and speed_ok and quality_ok:
            feasible.append(candidate)

    # baseline 只用于计算节省和吞吐保留率；候选排序不依赖 baseline。
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
        'throughput_ratio': (best['samples_per_s'] / baseline['samples_per_s']) if baseline and best else None,
        'baseline_available': baseline is not None,
        'best_peak_memory_mb': best['peak_memory_mb'] if best else None,
        'memory_saving_mb': (baseline['peak_memory_mb'] - best['peak_memory_mb']) if baseline and best else 0.0,
        'min_memory_saving_mb': budget['min_memory_saving_mb'],
        'min_throughput_ratio': budget['min_throughput_ratio'],
        'evaluations': evaluations,
    }


def decide_memory_budget_project(summary: Dict[str, object]) -> Dict[str, object]:
    """根据候选摘要和收益阈值输出 accept、tune 或 reject。

    返回 decision、reason 和 next_action，供项目报告记录下一步动作。
    """
    # ==========================================
    # TODO 3 对应变量 feasible_count：确认是否存在可行候选。
    # TODO 3 对应派生判断 meaningful_memory_gain / acceptable_throughput：检查收益门槛。
    # TODO 3 对应结果 decision / reason / next_action：收束为项目结论。
    # 提示：先读取 baseline_available / feasible_count，再判断收益布尔量，最后返回三个结论字段。
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
            'next_action': 'tighten_batch_or_rework_memory_plan',
        }
    # 两个布尔量分别判断显存收益和吞吐保留率是否达到预算门槛。
    meaningful_memory_gain = summary.get('memory_saving_mb', 0.0) >= summary['min_memory_saving_mb']
    acceptable_throughput = summary.get('throughput_ratio') is not None and summary['throughput_ratio'] >= summary['min_throughput_ratio']
    if best_candidate != 'baseline' and meaningful_memory_gain and acceptable_throughput:
        return {
            'decision': 'accept',
            'reason': 'compression_strategy_is_best_feasible_option',
            'next_action': 'promote_to_training_run',
        }
    if quality_failed_count > 0:
        return {
            'decision': 'tune',
            'reason': 'compression_needs_quality_recovery',
            'next_action': 'adjust_checkpoint_scope_or_batch_plan',
        }
    if not meaningful_memory_gain:
        return {
            'decision': 'tune',
            'reason': 'memory_saving_below_meaningful_threshold',
            'next_action': 'test_larger_pressure_or_optimizer_compression',
        }
    if not acceptable_throughput:
        return {
            'decision': 'tune',
            'reason': 'throughput_loss_exceeds_budget',
            'next_action': 'reduce_compression_scope',
        }
    return {
        'decision': 'tune',
        'reason': 'baseline_still_best_under_current_budget',
        'next_action': 'revisit_memory_strategy_mix',
    }

```

### 解析

**1. TODO 1: 检查预算与质量阈值**
- **实现方式**：先把显存上限、吞吐下限、最低显存收益、最低吞吐保留率和验证损失上限检查齐，再进入方案比较。
- **关键点**：没有统一预算口径时，显存压缩方案之间的比较都没有解释力。
- **项目意义**：这一步把 `75` 固定成预算约束下的显存决策页，而不是泛显存技巧页。

**2. TODO 2: 汇总显存压缩策略**
- **实现方式**：按 peak memory、samples/s 和 eval loss 统一过滤候选，记录每个候选的淘汰原因，再选出最省显存的可行方案。
- **关键点**：显存收益只有在质量和吞吐都没有跌出边界时，才值得被保留。
- **项目意义**：这一步把 `19 / 42 / 73 / 76` 的机制与 benchmark 证据收成真正可比较的工程候选。

**3. TODO 3: 输出项目结论**
- **实现方式**：把候选可行性、最低显存收益、吞吐保留率和最优方案统一收成 `accept / tune / reject`。
- **关键点**：项目结论必须回答“当前预算下哪种显存压缩方案值得继续采用”，而不是只输出一个峰值显存最小值。
- **项目意义**：这一步把 `75` 收成显存优化路线中的正式预算项目。

### Step 5（可选）：读取 GPU 报告并生成预算决策

**5.1 报告来源与预算条件**

本节读取 76 节的 GPU 报告，并在固定报告口径下生成预算决策 JSON。先核对报告来源、实验条件和预算字段；具体策略结果与敏感性分析统一放在 5.5。

| 报告来源与预算条件 | 配置 |
|:---|:---|
| 来源报告 | `benchmarks/results/76_real_gpu_memory.json` |
| GPU / 显存 | RTX 5070 Ti Laptop GPU / 待填写 |
| 模型与 workload | `Qwen/Qwen2.5-0.5B-Instruct`、FP32、batch=1、seq_len=768 |
| 预算条件 | 显存上限、吞吐下限、最低显存收益、质量阈值：待填写 |
| evidence level | `fixed_workload_strategy_benchmark` / 待复核 |


5.1 的产物是可核对的输入口径，不在这里预先下结论。

**5.2 输入核验与执行流程**

Step 5 读取 76 已完成的 GPU 报告，把实测证据转换为 CPU 预算决策；不下载模型、不重新训练，也不采集 GPU。按下表运行配置和结果代码。

| 实验阶段 | 学习者操作 | 固定 / 修改内容 | 输出证据 |
|:---|:---|:---|:---|
| 输入核验 | 确认 73 baseline 和 76 策略报告存在且字段完整 | 模型、workload、dtype、optimizer、重复运行 | 可进入决策的 measured 证据 |
| 配置预算 | 只修改配置 cell | `measured / projected`、硬件 profile、显存上限、吞吐下限、质量阈值 | 当前预算条件 |
| 运行决策 | 执行结果代码，不启动模型 | measured 读取报告；projected 只做容量规划 | 候选集合与 `accept / tune / reject` |
| 敏感性分析 | 改变预算阈值，不重新训练 | 显存、吞吐和质量门槛组合 | 可行集合与最佳候选是否稳定 |
| 记录结果 | 读取 JSON 并填写结论 | 输入报告路径、预算、证据模式和复核入口 | `75_memory_budget_decision.json` |

![75 GPU 报告到预算决策的收口流程](../public/02_PyTorch_Algorithms/75_report_decision_flow.svg)
<div align="center"><strong>75 读取已有 GPU 证据，负责预算判断，不重新测量性能。</strong></div>


```python
"""预算决策的输入预检：确认项目路径和上游 GPU 报告，不启动 GPU、不加载模型。"""
from pathlib import Path
import os
import subprocess
import sys

PROJECT_ROOT = Path(os.environ.get('LLM_ALGO_PROJECT_ROOT', Path.cwd())).expanduser().resolve()
if not (PROJECT_ROOT / 'tools/project_runtime.py').is_file():
    colab_root = Path('/content/llm-algo-leetcode')
    if (colab_root / 'tools/project_runtime.py').is_file():
        PROJECT_ROOT = colab_root
    elif Path('/content').is_dir() and not colab_root.exists():
        subprocess.run(['git', 'clone', 'https://github.com/datawhalechina/llm-algo-leetcode.git', str(colab_root)], check=True)
        PROJECT_ROOT = colab_root
if not (PROJECT_ROOT / 'tools/project_runtime.py').is_file():
    raise RuntimeError('找不到项目根目录：请设置 LLM_ALGO_PROJECT_ROOT，或先把仓库放到 /content/llm-algo-leetcode。')
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REPORT_PATH = PROJECT_ROOT / 'benchmarks/results/76_real_gpu_memory.json'
print(f'project_root: {PROJECT_ROOT}')
print(f'report_path: {REPORT_PATH}')
if not REPORT_PATH.is_file():
    print('尚未找到 76 的 GPU 报告；请先完成 76 的实验，或把报告路径改为已有 JSON。')
else:
    print(f'report_size_bytes: {REPORT_PATH.stat().st_size}')

```

**5.3 配置预算与阈值**

配置 cell 只选择证据模式、硬件预算和阈值；`measured` 才能生成基于已有报告的决策，`projected` 只保留规划标签，不能生成未经验证的性能预测。

```python
from pathlib import Path

# 75 只读取 76 的 JSON，不重新训练模型，也不需要 GPU。
RUN_REAL_PROJECT = True  # 设为 True 后运行预算决策。
EVIDENCE_MODE = 'measured'  # 当前可执行模式；projected 仅用于规划标记，不产生性能预测。
MODEL_PROFILE = {
    'model_id': 'Qwen/Qwen2.5-0.5B-Instruct',
    'task': 'causal_lm_training',
    'dataset_name': '固定小型教学数据集',
    'dataset_samples': 4,
}
HARDWARE_PROFILES = {
    'laptop_12gb': {'name': '12GB Laptop GPU', 'total_memory_mb': 12227, 'usable_memory_mb': 11200},
    'rtx4090_24gb': {'name': 'RTX 4090 24GB', 'total_memory_mb': 24576, 'usable_memory_mb': 22500},
    'datacenter_40gb': {'name': '40GB Data-center GPU', 'total_memory_mb': 40960, 'usable_memory_mb': 38000},
}
HARDWARE_PROFILE = 'laptop_12gb'  # 规划目标；measured 模式仍以 76 报告中的 GPU 为证据。
USE_PROFILE_MEMORY_CAP = True  # True：自动使用硬件档案的 usable_memory_mb；False：使用下面手动填写的预算。
BUDGET = {
    'memory_cap_mb': 11200.0,  # 当前实验允许的 GPU 显存上限。
    'min_samples_per_s': 1.0,  # 可接受的最低训练吞吐。
    'min_memory_saving_mb': 512.0,  # 相对 baseline 的最低显存收益；教学阈值，可调整。
    'min_throughput_ratio': 0.70,  # 相对 baseline 的最低吞吐保留率。
}
if USE_PROFILE_MEMORY_CAP:
    BUDGET['memory_cap_mb'] = float(HARDWARE_PROFILES[HARDWARE_PROFILE]['usable_memory_mb'])
# 这里的 cap 是决策预算，不是整张卡可分配的理论总显存。
MAX_VAL_LOSS = None  # None：使用 76 baseline eval_loss 的 1.02 倍作为教学代理门槛，不等于真实任务质量。
SENSITIVITY_MEMORY_CAPS = [9600.0, 11200.0]  # 只改变预算约束，不重新训练；可按设备显存调整。
SENSITIVITY_THROUGHPUT_FLOORS = [1.0, 1.5, 2.0]  # 观察 accept / tune / reject 是否依赖吞吐门槛。
SENSITIVITY_QUALITY_FLOORS = [12.20, 12.45, 12.80]  # 质量代理阈值；不是完整任务评测。
# 主线读取 76 的 FP32 / pressure 报告；BF16 或其他 workload 必须改成对应的独立文件。
RESULT_76_RELATIVE_PATH = Path('benchmarks/results/76_real_gpu_memory.json')
OUTPUT_RELATIVE_PATH = Path('benchmarks/results/75_memory_budget_decision.json')

```


```python
import json
import os
import sys
from pathlib import Path

# 5.4 读取报告并生成预算决策：75 不需要 GPU，只读取 76 报告。
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from tools.project_runtime import ensure_output_path, require_input_file, resolve_project_root, standard_experiment_config, standard_training_metrics
from tools.memory_budget_runtime import (
    summarize_memory_strategies,
    decide_memory_budget_project,
)

PROJECT_ROOT = resolve_project_root(PROJECT_ROOT)
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))
RESULT_76_PATH = PROJECT_ROOT / RESULT_76_RELATIVE_PATH

if RUN_REAL_PROJECT:
    if EVIDENCE_MODE != 'measured':
        raise ValueError('当前 75 代码只执行 measured 决策；projected 仅记录规划标签，不能生成性能预测。')
    require_input_file(RESULT_76_PATH, '76 结果')
    raw = json.loads(RESULT_76_PATH.read_text(encoding='utf-8'))
    source_config = raw.get('config', {})
    required_source_fields = ('model_id', 'workload', 'batch_size', 'seq_len', 'dtype', 'optimizer')
    missing_source_fields = [key for key in required_source_fields if source_config.get(key) in (None, '')]
    if missing_source_fields:
        raise ValueError(f'76 结果缺少统一实验字段：{missing_source_fields}；不能进入预算决策。')
    if raw.get('candidates') is None or not raw['candidates']:
        raise ValueError('76 没有候选策略结果；不能把空报告当作预算证据。')
    if EVIDENCE_MODE == 'measured':
        missing_repeats = [item.get('name', '<unknown>') for item in raw['candidates'] if len(item.get('runs', [])) < 3]
        if missing_repeats:
            raise ValueError(f'76 仍缺少每个候选的 3 次重复运行：{missing_repeats}；请先重跑 pressure workload。')
    candidates = []
    for item in raw['candidates']:
        candidates.append({
            'name': item['name'],
            'status': item.get('status', 'ok'),
            'step_time_ms': item.get('step_time_ms'),
            'samples_per_s': item.get('samples_per_s'),
            'peak_memory_mb': item.get('peak_memory_mb', item.get('peak_mem_mb')),
            'peak_reserved_mb': item.get('peak_reserved_mb'),
            'loss': item.get('loss'),
            'eval_loss': item.get('eval_loss', item.get('val_loss')),
            'val_loss': item.get('val_loss', item.get('eval_loss')),
            'error': item.get('error'),
        })
    baseline = next((x for x in candidates if x['name'] == 'baseline'), None)
    if baseline is None:
        raise RuntimeError('76 没有可用 baseline，无法形成预算决策。')
    max_val_loss = baseline['eval_loss'] * 1.02 if MAX_VAL_LOSS is None else MAX_VAL_LOSS
    quality_floor = {'max_val_loss': round(max_val_loss, 6)}
    summary = summarize_memory_strategies(candidates, BUDGET, quality_floor)
    decision = decide_memory_budget_project(summary)
    sensitivity = []
    for memory_cap_mb in SENSITIVITY_MEMORY_CAPS:
        for throughput_floor in SENSITIVITY_THROUGHPUT_FLOORS:
            for quality_floor_value in SENSITIVITY_QUALITY_FLOORS:
                sensitivity_budget = {
                    'memory_cap_mb': memory_cap_mb,
                    'min_samples_per_s': throughput_floor,
                    'min_memory_saving_mb': BUDGET['min_memory_saving_mb'],
                    'min_throughput_ratio': BUDGET['min_throughput_ratio'],
                }
                sensitivity_quality_floor = {'max_val_loss': quality_floor_value}
                sensitivity_summary = summarize_memory_strategies(candidates, sensitivity_budget, sensitivity_quality_floor)
                sensitivity_decision = decide_memory_budget_project(sensitivity_summary)
                sensitivity.append({
                    'memory_cap_mb': memory_cap_mb,
                    'min_samples_per_s': throughput_floor,
                    'max_val_loss': quality_floor_value,
                    'feasible_names': sensitivity_summary['feasible_names'],
                    'best_candidate': sensitivity_summary['best_candidate'],
                    'decision': sensitivity_decision['decision'],
                })
    decision_counts = {}
    best_candidate_counts = {}
    feasible_sets = {}
    for row in sensitivity:
        decision = row['decision']
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
        best_name = row['best_candidate'] or '<none>'
        best_candidate_counts[best_name] = best_candidate_counts.get(best_name, 0) + 1
        feasible_key = ','.join(row['feasible_names']) or '<none>'
        feasible_sets[feasible_key] = feasible_sets.get(feasible_key, 0) + 1
    sensitivity_summary = {
        'scenario_count': len(sensitivity),
        'decision_counts': decision_counts,
        'best_candidate_counts': best_candidate_counts,
        'feasible_set_counts': feasible_sets,
        'best_candidate_stable': len(best_candidate_counts) == 1,
        'feasible_set_stable': len(feasible_sets) == 1,
        'decision_stable': len(decision_counts) == 1,
        'interpretation': 'stable' if len(best_candidate_counts) == 1 and len(feasible_sets) == 1 else 'sensitive_to_thresholds',
    }
    project_result = {
        'task': 'task3_training_memory_optimization',
        'stage': 'memory_budget_decision',
        'source': str(RESULT_76_PATH.relative_to(PROJECT_ROOT)),
        'budget': BUDGET,
        'quality_floor': quality_floor,
        'evidence_mode': EVIDENCE_MODE,
        'planning_context': {
            'model': MODEL_PROFILE,
            'hardware_target': HARDWARE_PROFILES[HARDWARE_PROFILE],
            'note': '硬件目标用于预算规划；只有 76 报告中的硬件和 workload 属于 measured evidence。',
        },
        'candidates': candidates,
        'summary': summary,
        'decision': decision,
        'sensitivity': sensitivity,
        'sensitivity_summary': sensitivity_summary,
    }
    project_result['planning_context']['measured_hardware'] = source_config.get('device')
    project_result['planning_context']['hardware_comparison'] = 'same_hardware_required_for_performance_claim'
    project_result['experiment'] = standard_experiment_config({**source_config, 'dtype': source_config.get('dtype', source_config.get('amp_dtype'))})
    project_result['standard_metrics'] = {item['name']: standard_training_metrics(item) for item in candidates}
    output_path = ensure_output_path(PROJECT_ROOT, OUTPUT_RELATIVE_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(project_result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(project_result, ensure_ascii=False, indent=2))
    print('\n预算敏感性：')
    for row in sensitivity:
        print(f"memory_cap={row['memory_cap_mb']:.0f} MB, throughput_floor={row['min_samples_per_s']:.1f}, feasible={row['feasible_names']}, decision={row['decision']}")
    print('预算敏感性汇总：', sensitivity_summary)
else:
    print('跳过真实项目决策：保持 CPU-first 模式。')

```

**5.5 结果记录与敏感性分析**

运行 5.4 的预算决策代码后，先保留完整的策略指标，再填写汇总结果。历史结果只用于示范记录格式；学习者应把自己的 GPU、模型、workload 和阈值写入复测行，不能直接把历史数值迁移到其他硬件。

**本地历史结果**

| 策略 | step time（ms） | 吞吐（samples/s） | peak allocated（MB） | peak reserved（MB） | eval loss | 状态 |
|:---|---:|---:|---:|---:|---:|:---|
| baseline | 497.839 | 2.009 | 9782.74 | 10750.00 | 12.356503 | ok |
| checkpoint | 567.740 | 1.761 | 9450.76 | 10896.00 | 12.356503 | ok |
| offload | 1877.559 | 0.533 | 9448.33 | 10404.00 | 12.356503 | ok |
| hybrid | 783.478 | 1.276 | 9454.64 | 10444.00 | 12.356503 | ok |

`checkpoint` 是当前显存与吞吐门槛下的候选方案：显存节省约 332 MB、吞吐保留约 87.7%，但低于 512 MB 的最低显存收益阈值，因此结论是 `tune`。这表示当前收益不足以定版，不表示实验失败。

**学习者复测记录**

| 硬件 / 显存 | 模型与 workload | 预算条件 | 可行策略 | 最佳策略 | 显存节省 | 吞吐保留率 | 决策 | 下一步动作 |
|:---|:---|:---|:---|:---|---:|---:|:---|:---|
| 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |

**示例敏感性分析**

敏感性分析只改变预算阈值，不重新测量；如果最佳策略或可行集合随阈值变化，应记录为“对预算条件敏感”，并写明下一次复验动作。

| 实验 | 预算 | 可行策略 | 最佳策略 | 显存节省 | 决策 |
|:---|:---|:---|:---|---:|:---|
| FP32 / seq768 | 11200 MiB | baseline / checkpoint / hybrid | checkpoint | 331.98 MiB | tune |
| FP32 / seq768 | 9600 MiB | checkpoint / hybrid | checkpoint | 331.98 MiB | tune |
| BF16 / seq1024 | 9600 MiB | 无 | — | — | reject |
| BF16 / seq1024 | 11200 MiB | baseline / checkpoint | checkpoint | 27.16 MiB | tune |

这些示例用于展示记录方法，不替代目标硬件上的实测。BF16 在该 workload 下比 checkpoint 更直接地缓解容量问题；checkpoint 是否值得保留，仍需结合更高 activation 压力或 profiling 证据判断。
---
## 相关阅读

- [ZeRO 论文：Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [QLoRA 论文：Efficient Finetuning of Quantized Language Models](https://arxiv.org/abs/2305.14314)
- [DeepSpeed 官方仓库](https://github.com/microsoft/DeepSpeed)
- [PyTorch Activation Checkpointing 官方文档](https://pytorch.org/docs/stable/checkpoint.html)
- [PyTorch FSDP 官方文档](https://pytorch.org/docs/stable/fsdp.html)
- [DeepSpeed ZeRO-3 官方文档](https://deepspeed.readthedocs.io/en/latest/zero3.html)
- [PyTorch CUDA 内存管理文档](https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)
- [73 训练性能分析](./73_Training_Performance_Analysis.md)
- [76 Activation / Checkpoint / Offload 对比项目](./76_Activation_Checkpoint_Offload_Benchmark.md)
- [74 Profiling 驱动的端到端优化](./74_Profiling_Driven_End_to_End_Optimization.md)
