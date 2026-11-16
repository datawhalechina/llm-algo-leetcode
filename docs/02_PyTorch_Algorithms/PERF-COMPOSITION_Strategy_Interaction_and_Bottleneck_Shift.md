# PERF-COMPOSITION. Strategy Interaction and Bottleneck Shift | 策略相互作用与瓶颈转移

**难度：** Hard | **环境：** [CPU基础练习，GPU可选](#runtime-environment) | **标签：** `策略组合`、`依赖冲突`、`边际收益` | **目标人群：** 具备性能测量基础的读者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/PERF-COMPOSITION_Strategy_Interaction_and_Bottleneck_Shift.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

降低计算精度可能缩短矩阵运算，提前搬入下一批数据可能减少等待。两项分别有效时，一起使用是否还能继续变快？计算变短后，原本不明显的搬运或调度也可能成为主要开销。

本节先检查策略能否一起执行，再比较基线、两个单项和组合。重点是组合比已经较好的单项还能省多少时间，以及数值、资源或延迟要求不满足时恢复哪种配置，而不是把单项收益直接相加。

**关键词：** 组合依赖、资源竞争、新增收益、回退选择、重新定位瓶颈

<a id="runtime-environment"></a>

## 运行环境

- **基础练习：** 使用[BASE-CPU](../guide.md#base-cpu)，运行[Step 4的题目与测试](#mechanism-practice)。
- **GPU实测：** 环境和固定工作负载见[Step 5](#gpu-experiment)。

## 前置阅读

先判断单项减少了什么工作或等待，再检查组合的先后条件与测量范围。

- [PERF-CANDIDATE. 约束、候选与代价转移](./PERF-CANDIDATE_Constraint_to_Candidate_and_Cost_Transfer.md)：拆开减少量与新增开销。
- [PERF-REGRESSION. 配对基准与性能回归阈值](./PERF-REGRESSION_Paired_Benchmark_and_Regression_Thresholds.md)：比较同负载的重复运行。
- [PERF-TRACE. 执行轨迹依赖与关键路径](./PERF-TRACE_Trace_Dependencies_and_Critical_Path.md)：检查耗时变化是否影响最终完成。

## Step 1：哪些依赖与冲突会改变组合结果

组合中的依赖表示“必须先有某种结果，下一项才能执行”。例如，格式转换必须先完成，下游计算内核才能消费对应布局；两项配置对同一布局提出不相容要求时，先解决格式冲突，再评估组合。资源竞争则表示两项同时使用带宽、算力或临时空间；它们仍可能一起运行，只是净收益要通过测量判断。

| 关系 | 如何识别 | 怎样处理 |
| --- | --- | --- |
| 前置依赖 | 下游需要上游生成的格式、状态或能力 | 依赖项先执行，缺失或循环依赖拒绝 |
| 明确不兼容 | 两项配置对同一对象要求互斥状态 | 任一方声明不兼容即拒绝组合 |
| 资源竞争 | 同时使用带宽、计算资源或临时空间 | 保留合法组合，测量争用与峰值 |
| 顺序敏感 | 先后改变最终布局、精度或执行路径 | 将顺序写入实验配置；仅有真实语义差异时比较反向顺序 |

![组合前检查与同基线复测](../public/02_PyTorch_Algorithms/perf_composition_flow.svg)
## Step 2：比较组合比可用单项还能快多少

新增收益也称边际收益，回答“在已有方案上再加一项，能继续减少多少时间”。设同负载基线延迟为 $T_0$，实测组合延迟为 $T_{AB}$。先从通过正确性、质量与资源检查、并优于基线的单项中选择最低延迟 $T_{\ast}$；没有这样的单项时以基线作为恢复配置和比较参考。

$$
g_0=\frac{T_0-T_{AB}}{T_0},\qquad
g_{\ast}=\frac{T_{\ast}-T_{AB}}{T_{\ast}}.
$$

$g_0$表示相对基线的延迟下降， $g_{\ast}$表示相对参考的额外下降，分母不同。比如基线100ms、可用单项80ms、组合76ms，相对基线下降24%，但在该单项上只再下降5%。两个单项可能减少了同一段工作，也可能新增争用，因此组合值必须实际测量。

| 对照 | 回答什么问题 | 记录什么 |
| --- | --- | --- |
| 基线与各单项 | 每个改动是否独立有效 | 检查结果、时间与资源 |
| 可用单项与组合 | 新增复杂性是否换来额外改善 | 同负载样本、下降比例与要求 |
| 基线与组合 | 整体是否优于最初配置 | 完整处理时间与失败 |
| 组合与恢复配置 | 组合不适合时使用哪个方案 | 已检查的配置；没有可用单项时保留基线 |

## Step 3：组合后重新检查时间花在哪里

阶段占比由自身耗时与总耗时共同决定。例如计算从40ms缩短到15ms，固定等待仍是20ms，等待占比就会增加。瓶颈转移指限制总完成速度的因素发生改变；判断时，先看各阶段的绝对耗时，再结合事件依赖、重叠与请求竞争，确认下一项改动能否继续缩短总时间。

图中四行是构造的串行例子，阶段可以直接相加；真实流水线可能重叠，仍要重新测总时间。下一步先看组合是否满足目标，再沿时间线检查剩余等待，决定保留组合、使用单项还是恢复基线。

| 比较结果 | 当前选择 | 下一项检查 |
| --- | --- | --- |
| 组合通过要求，且相对参考的新增收益足够 | 保留组合与恢复配置 | 缩短的工作是否影响完成链 |
| 组合快于基线，但额外收益不足 | 使用已检查的单项，或基线 | 新增调度、传输与资源竞争 |
| 组合数值或资源要求失败 | 恢复可用单项，或基线 | 哪项检查失败、能否调整 |
| 组合与参考相等，或更慢 | 不增加当前组合 | 改动是否生效，以及剩余等待 |

![阶段缩短后，固定等待的占比怎样改变](../public/02_PyTorch_Algorithms/perf_composition_bottleneck_shift.svg)

<a id="mechanism-practice"></a>

## Step 4：实现依赖检查、组合边际与合格回退

题目接收候选规格、顺序、越低越好的实测指标和各单项的检查状态。先验证顺序，再计算组合边际，最后从通过检查且优于基线的单项中选择回退参考。`measured_margins`只统计给定单项的数值关系，`decide_composition`再筛选合格参考。没有合格单项时使用基线；组合与参考相等时，即使最低边际门槛为零也不保留组合。

| TODO / 函数 | 学习者完成的机制 | 必须满足的约束 | 测试证据 |
| --- | --- | --- | --- |
| TODO 1：验证组合顺序<br>`validate_order` | 检查前置依赖、双向冲突与重复项 | 候选和引用已定义；单向声明的冲突也有效 | 合法顺序、倒置、循环、重复和两种冲突顺序 |
| TODO 2：计算实测边际<br>`measured_margins` | 选择最低单项值，计算两个分母下的下降比例 | 测量有限为正；结果来自组合实测值 | 非加和例子、并列最优、非法测量 |
| TODO 3：选择保留或回退<br>`decide_composition` | 筛选合格参考，比较质量与新增收益 | 回退项必须合格且更快；无合格项用基线 | 不合格最快项、组合失败、收益相等与三类动作 |

```python
import math
from collections.abc import Sequence
```


```python
def validate_order(order: Sequence[str], specs: dict[str, dict[str, set[str]]]) -> list[str]:
    """检查显式依赖与互斥声明，不把普通资源竞争当作禁止共存。"""
    if isinstance(order, (str, bytes)):
        raise ValueError("组合顺序必须为候选序列")
    for name, spec in specs.items():
        for field in ("requires", "conflicts"):
            refs = spec.get(field, set())
            if not isinstance(refs, (set, list, tuple)) or any(not isinstance(ref, str) for ref in refs):
                raise ValueError("依赖和冲突必须为候选名称集合")
            if set(refs) - set(specs) or name in refs:
                raise ValueError("存在未知引用或自身依赖／冲突")
    # TODO 1：验证组合顺序
    # seen = ???
    # validated = ???
    raise NotImplementedError("TODO 1：请验证组合顺序")
    for name in order:
        if name not in specs or name in seen:
            raise ValueError("候选未知或重复")
        # requires = ???
        # conflicts = ???
        # reverse_conflicts = ???
        if not requires <= seen:
            raise ValueError(f"候选 {name} 的依赖尚未满足")
        if conflicts & seen or reverse_conflicts:
            raise ValueError(f"候选 {name} 与已有候选冲突")
        seen.add(name)
        validated.append(name)
    return validated


def measured_margins(baseline: float, singles: dict[str, float], combined: float) -> dict[str, object]:
    """计算同一越低越好指标的实测边际，不判定单项质量。"""
    if not singles or any(not isinstance(name, str) or not name for name in singles):
        raise ValueError("至少提供一个具名单项")
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) or v <= 0 for v in (baseline, combined, *singles.values())):
        raise ValueError("测量值必须有限为正")
    # TODO 2：计算实测边际
    # best_single_name = ???
    # best_single_value = ???
    # gain_vs_baseline = ???
    # gain_vs_best_single = ???
    raise NotImplementedError("TODO 2：请计算实测边际")
    return {"best_single": best_single_name, "gain_vs_baseline": gain_vs_baseline,
            "gain_vs_best_single": gain_vs_best_single}


def decide_composition(baseline, singles, combined, guardrails_passed, minimum_marginal_gain, *,
                       single_guardrails: dict[str, bool]):
    """只从合格且更快单项选择回退，空集时使用基线。"""
    measured_margins(baseline, singles, combined)
    if type(guardrails_passed) is not bool:
        raise ValueError("组合检查状态必须为布尔值")
    if set(single_guardrails) != set(singles) or any(type(v) is not bool for v in single_guardrails.values()):
        raise ValueError("每个单项必须明确提供检查状态")
    if isinstance(minimum_marginal_gain, bool) or not isinstance(minimum_marginal_gain, (int, float)) or not math.isfinite(minimum_marginal_gain) or not 0 <= minimum_marginal_gain < 1:
        raise ValueError("最低边际下降比例必须在 [0,1) 内")
    # TODO 3：选择保留或回退
    # eligible = ???
    # best_single = ???
    # reference_value = ???
    # reference_name = ???
    # gain_vs_baseline = ???
    # gain_vs_reference = ???
    # combination_passed = ???
    # worthwhile = ???
    raise NotImplementedError("TODO 3：请选择合格回退")
    if combination_passed and worthwhile:
        decision = "keep_combination"
    else:
        decision = "keep_single" if best_single else "rollback_baseline"
    return {"best_single": best_single, "reference": reference_name,
            "gain_vs_baseline": gain_vs_baseline, "gain_vs_reference": gain_vs_reference,
            "decision": decision, "rollback_to": reference_name}
```

## 测试

核对组合依赖、两个收益分母以及不合格单项不能成为回退目标。

```python
def _specs():
    return {"format": {"requires": set(), "conflicts": set()},
            "kernel": {"requires": {"format"}, "conflicts": {"eager_only"}},
            "eager_only": {"requires": set(), "conflicts": set()}}


def _expect_composition_error(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("非法组合未被拒绝")


def test_order_and_asymmetric_conflicts():
    assert validate_order(["format", "kernel"], _specs()) == ["format", "kernel"]
    for order in (["kernel", "format"], ["format", "kernel", "eager_only"],
                  ["eager_only", "format", "kernel"], ["format", "format"], ["unknown"]):
        _expect_composition_error(validate_order, order, _specs())


def test_missing_references_and_cycles():
    _expect_composition_error(validate_order, ["a"], {"a": {"requires": {"missing"}}})
    _expect_composition_error(validate_order, ["a"], {"a": {"requires": {"a"}}})
    _expect_composition_error(validate_order, ["a", "b"],
                              {"a": {"requires": {"b"}}, "b": {"requires": {"a"}}})
    _expect_composition_error(validate_order, "format", _specs())


def test_measured_margins_and_ties():
    margins = measured_margins(100, {"a": 80, "b": 85}, 76)
    assert margins["best_single"] == "a"
    assert math.isclose(margins["gain_vs_baseline"], 0.24)
    assert math.isclose(margins["gain_vs_best_single"], 0.05)
    assert measured_margins(100, {"b": 80, "a": 80}, 76)["best_single"] == "a"


def test_nonfinite_measurements():
    for bad in (0, -1, True, float("nan"), float("inf")):
        _expect_composition_error(measured_margins, bad, {"a": 80}, 72)
        _expect_composition_error(measured_margins, 100, {"a": bad}, 72)
        _expect_composition_error(measured_margins, 100, {"a": 80}, bad)
    _expect_composition_error(measured_margins, 100, {}, 72)


def test_three_decisions():
    singles, guards = {"a": 80, "b": 85}, {"a": True, "b": True}
    assert decide_composition(100, singles, 72, True, .05, single_guardrails=guards)["decision"] == "keep_combination"
    assert decide_composition(100, singles, 79, True, .05, single_guardrails=guards)["decision"] == "keep_single"
    failed = decide_composition(100, singles, 72, False, .05, single_guardrails=guards)
    assert failed["decision"] == "keep_single" and failed["rollback_to"] == "a"
    assert decide_composition(100, {"a": 105}, 110, False, .05,
                              single_guardrails={"a": True})["decision"] == "rollback_baseline"


def test_ineligible_fastest_single():
    singles, guards = {"fast_bad": 50, "safe": 80}, {"fast_bad": False, "safe": True}
    report = decide_composition(100, singles, 70, False, .03, single_guardrails=guards)
    assert report["rollback_to"] == "safe" and report["decision"] == "keep_single"
    none = decide_composition(100, singles, 70, False, .03,
                              single_guardrails={"fast_bad": False, "safe": False})
    assert none["best_single"] is None and none["rollback_to"] == "baseline"
    assert none["decision"] == "rollback_baseline"
    recovered = decide_composition(100, singles, 70, True, .03,
                                   single_guardrails={"fast_bad": False, "safe": False})
    assert recovered["decision"] == "keep_combination" and recovered["reference"] == "baseline"


def test_marginal_equalities():
    args = dict(single_guardrails={"a": True})
    assert decide_composition(100, {"a": 80}, 76, True, .05, **args)["decision"] == "keep_combination"
    assert decide_composition(100, {"a": 80}, 80, True, 0, **args)["decision"] == "keep_single"
    assert decide_composition(100, {"a": 80}, 100, True, 0, **args)["decision"] == "keep_single"


def test_guardrail_and_threshold_inputs():
    for bad in (-.1, 1, True, float("nan"), float("inf")):
        _expect_composition_error(decide_composition, 100, {"a": 80}, 72, True, bad,
                                  single_guardrails={"a": True})
    for guards in ({}, {"a": 1}, {"a": True, "b": True}):
        _expect_composition_error(decide_composition, 100, {"a": 80}, 72, True, .03,
                                  single_guardrails=guards)


for test in (test_order_and_asymmetric_conflicts, test_missing_references_and_cycles,
             test_measured_margins_and_ties, test_nonfinite_measurements, test_three_decisions,
             test_ineligible_fastest_single, test_marginal_equalities, test_guardrail_and_threshold_inputs):
    test()
print("✅ 组合依赖、实测边际与合格回退测试通过")
```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

## 参考代码与解析

### 代码


```python
def validate_order(order: Sequence[str], specs: dict[str, dict[str, set[str]]]) -> list[str]:
    """检查显式依赖与互斥声明，不把普通资源竞争当作禁止共存。"""
    if isinstance(order, (str, bytes)):
        raise ValueError("组合顺序必须为候选序列")
    for name, spec in specs.items():
        for field in ("requires", "conflicts"):
            refs = spec.get(field, set())
            if not isinstance(refs, (set, list, tuple)) or any(not isinstance(ref, str) for ref in refs):
                raise ValueError("依赖和冲突必须为候选名称集合")
            if set(refs) - set(specs) or name in refs:
                raise ValueError("存在未知引用或自身依赖／冲突")
    # TODO 1：验证组合顺序
    seen, validated = set(), []
    for name in order:
        if name not in specs or name in seen:
            raise ValueError("候选未知或重复")
        requires = set(specs[name].get("requires", set()))
        conflicts = set(specs[name].get("conflicts", set()))
        reverse_conflicts = {prior for prior in seen if name in specs[prior].get("conflicts", set())}
        if not requires <= seen:
            raise ValueError(f"候选 {name} 的依赖尚未满足")
        if conflicts & seen or reverse_conflicts:
            raise ValueError(f"候选 {name} 与已有候选冲突")
        seen.add(name)
        validated.append(name)
    return validated


def measured_margins(baseline: float, singles: dict[str, float], combined: float) -> dict[str, object]:
    """计算同一越低越好指标的实测边际，不判定单项质量。"""
    if not singles or any(not isinstance(name, str) or not name for name in singles):
        raise ValueError("至少提供一个具名单项")
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) or v <= 0 for v in (baseline, combined, *singles.values())):
        raise ValueError("测量值必须有限为正")
    # TODO 2：计算实测边际
    best_single_name = min(singles, key=lambda name: (singles[name], name))
    best_single_value = singles[best_single_name]
    gain_vs_baseline = (baseline - combined) / baseline
    gain_vs_best_single = (best_single_value - combined) / best_single_value
    return {"best_single": best_single_name, "gain_vs_baseline": gain_vs_baseline,
            "gain_vs_best_single": gain_vs_best_single}


def decide_composition(baseline, singles, combined, guardrails_passed, minimum_marginal_gain, *,
                       single_guardrails: dict[str, bool]):
    """只从合格且更快单项选择回退，空集时使用基线。"""
    measured_margins(baseline, singles, combined)
    if type(guardrails_passed) is not bool:
        raise ValueError("组合检查状态必须为布尔值")
    if set(single_guardrails) != set(singles) or any(type(v) is not bool for v in single_guardrails.values()):
        raise ValueError("每个单项必须明确提供检查状态")
    if isinstance(minimum_marginal_gain, bool) or not isinstance(minimum_marginal_gain, (int, float)) or not math.isfinite(minimum_marginal_gain) or not 0 <= minimum_marginal_gain < 1:
        raise ValueError("最低边际下降比例必须在 [0,1) 内")
    # TODO 3：选择保留或回退
    eligible = {name: value for name, value in singles.items()
                if single_guardrails[name] and value < baseline}
    best_single = min(eligible, key=lambda name: (eligible[name], name)) if eligible else None
    reference_value = eligible[best_single] if best_single else baseline
    reference_name = best_single if best_single else "baseline"
    gain_vs_baseline = (baseline - combined) / baseline
    gain_vs_reference = (reference_value - combined) / reference_value
    combination_passed = guardrails_passed and gain_vs_baseline > 0
    worthwhile = gain_vs_reference > 0 and gain_vs_reference >= minimum_marginal_gain
    if combination_passed and worthwhile:
        decision = "keep_combination"
    else:
        decision = "keep_single" if best_single else "rollback_baseline"
    return {"best_single": best_single, "reference": reference_name,
            "gain_vs_baseline": gain_vs_baseline, "gain_vs_reference": gain_vs_reference,
            "decision": decision, "rollback_to": reference_name}
```

### 解析

#### TODO 1：分别检查依赖与冲突

- 加入候选前，所需前置项必须已经出现；互斥关系只要任一方声明就有效，不要求两侧重复填写。
- 测试覆盖未知引用、依赖倒置、循环、重复项，以及两种顺序下的单向冲突；普通带宽竞争仍允许进入测量。

#### TODO 2：用对应的参考值作分母

- `measured_margins`只按给定数值取最低单项，再计算相对基线和最低单项的下降比例，不在这里判断质量。
- 已知数值与并列测试检查比例和稳定排序，非法数值测试阻止无效输入；这些计算不能由单项收益相加替代。

#### TODO 3：选择保留或恢复的配置

- `decide_composition`先按给定检查状态筛选更快的单项，再选择参考；没有可用单项时回到基线。
- 组合必须严格快于参考且新增收益达到要求。测试用“最快但不合格”的单项检查恢复目标，也检查等于阈值可通过、等于参考则不增加组合。

<a id="gpu-experiment"></a>

## Step 5：GPU比较精度与双缓冲的组合

四条路径分别为FP32串行基线、BF16串行、FP32双缓冲和BF16双缓冲。FP32、BF16分别是32位与16位浮点格式；双缓冲交替使用两块设备输入区，让搬入下一批数据有机会与当前计算重叠。相同种子生成同源数据，再按各路径精度转换。

实验记录每次处理8批输入的时间，比较组合相对基线与可用单项的下降比例。数值检查另走独立前向计算余弦相似度，即比较两份输出向量的方向是否接近；完整双缓冲路径还需逐元素对照。

<a id="gpu-experiment-setup"></a>

### 5.1 环境与固定工作负载

使用`PERF-CUDA-MECHANISM`，环境准备见[性能机制GPU环境](../topic_discussion/performance_optimization/gpu_quickstart.md#perf-cuda-mechanism-quickstart)，登记见[实验资产表](../topic_discussion/performance_optimization/experiment_assets.md)。输入为本机合成数据。

| 对象 | 固定设置 | 解释方式 |
| --- | --- | --- |
| 网络与输入 | SwiGLU门控多层感知机，隐藏维2048、中间维5504；每次8批，32／256词元 | 门控支路与另一支路逐元素相乘后投影；四路径在同次运行中比较 |
| 重复协议 | 12轮轮换预热，24轮轮换四路径 | 无反向传播；计时路径未显式禁用自动微分 |
| 速度要求 | 相对可用参考再降至少3%；批处理P95比不超过1.05 | P95是本次8批处理时间的分位值，不是在线请求尾延迟 |
| 数值要求 | 独立前向余弦相似度至少0.999 | 检查输出方向，不能替代双缓冲全路径逐元素误差 |
| 恢复配置 | 单项相似度与P95符合要求，并比基线更快 | 在这些已检查条件内选择；没有时保留基线 |
| 显存与阶段时间 | 多候选共同驻留的峰值，独立搬运／计算服务时间 | 共享峰值不作各路径独立容量排名；服务时间用于继续排查 |
| 版本 | 工作负载v2；新保存schema v2记录单项检查 | 历史schema v1保持原样，新决策分支尚未做GPU复测 |


```python
RUN_MODE = "dry_run"  # 默认值 dry_run；改为 real_gpu 执行真实组合矩阵。
RUN_LABEL = "default"  # 新实验填写唯一标签；读取历史结果时填写已有标签。
PROFILE_ID = "PERF-CUDA-MECHANISM"  # 默认性能机制环境。
SEMANTIC_ID = "PERF-COMPOSITION"  # 稳定机制 ID。
WORKLOAD_ID = "perf-decoder-mlp-composition-matrix"  # 稳定工作负载 ID。
WORKLOAD_VERSION = "v2"  # v2 使用逐轮轮换顺序，减少四条路径的先后偏差。
GPU_CONFIG = {
    "hidden_size": 2048,          # 默认值 2048；与候选页保持一致。
    "intermediate_size": 5504,   # 默认值 5504；与候选页保持一致。
    "token_counts": [32, 256],   # 默认两档 Token 负载；逐负载比较边际收益。
    "pipeline_batches": 8,       # 默认值 8；提供稳定的重叠窗口。
    "warmup_steps": 12,          # 默认值 12；四条路径轮换预热且不计入统计。
    "repeat_steps": 24,          # 默认值 24；每轮轮换路径顺序并统计中位数与 P95。
    "minimum_marginal_gain": 0.03, # 默认值 3%；组合相对最佳单项的最低边际收益。
    "minimum_cosine": 0.999,     # 默认值 0.999；组合输出质量门槛。
    "maximum_p95_ratio": 1.05,   # 默认值 1.05；组合 P95 最多允许比最佳单项回归 5%。
    "seed": 42,                  # 默认值 42；固定权重与输入。
}
RESULTS_RELATIVE_DIR = "benchmarks/results/perf_composition"
```

### 5.2 执行组合矩阵并保存 JSON

预检通过后执行四路径矩阵。保存前先逐项检查单项的相似度、尾延迟和基线收益，再选择回退参考；默认 `dry_run` 不导入 GPU 运行时，不覆盖已有结果。共享计时实现保持不变。

```python
from pathlib import Path
import json
import math
import platform
import subprocess
import sys

if RUN_MODE not in {"dry_run", "real_gpu"}:
    raise ValueError("RUN_MODE 只能为 dry_run 或 real_gpu")
for key in ("hidden_size", "intermediate_size", "pipeline_batches", "warmup_steps", "repeat_steps"):
    value = GPU_CONFIG[key]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} 必须为正整数")
tokens = GPU_CONFIG["token_counts"]
if not tokens or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in tokens) or len(set(tokens)) != len(tokens):
    raise ValueError("Token 负载必须为不重复的正整数")
for key in ("minimum_marginal_gain", "minimum_cosine", "maximum_p95_ratio"):
    value = GPU_CONFIG[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{key} 必须为有限数值")
if not 0 <= GPU_CONFIG["minimum_marginal_gain"] < 1 or not -1 <= GPU_CONFIG["minimum_cosine"] <= 1 or GPU_CONFIG["maximum_p95_ratio"] <= 0:
    raise ValueError("组合判断门槛范围非法")

if RUN_MODE == "real_gpu":
    import torch
    from tools.performance_optimization_runtime import run_candidate_composition_matrix
    from tools.project_runtime import resolve_project_root, save_json_record_once

    if not RUN_LABEL or RUN_LABEL == "default" or Path(RUN_LABEL).name != RUN_LABEL:
        raise ValueError("正式运行需要不含目录分隔符的唯一 RUN_LABEL")
    project_root = resolve_project_root()
    RESULTS_DIR = project_root / RESULTS_RELATIVE_DIR
    result_path = RESULTS_DIR / f"{SEMANTIC_ID.lower()}_{RUN_LABEL}.json"
    if result_path.exists():
        raise FileExistsError("同标签结果已存在：读取已有结果，或更换 RUN_LABEL")
    subprocess.run([sys.executable, str(project_root / "tools/environment_preflight.py"),
                    "--profile", PROFILE_ID], check=True)
    matrix = run_candidate_composition_matrix(
        hidden_size=GPU_CONFIG["hidden_size"], intermediate_size=GPU_CONFIG["intermediate_size"],
        token_counts=GPU_CONFIG["token_counts"], pipeline_batches=GPU_CONFIG["pipeline_batches"],
        warmup_steps=GPU_CONFIG["warmup_steps"], repeat_steps=GPU_CONFIG["repeat_steps"],
        seed=GPU_CONFIG["seed"],
        variants=["baseline_fp32_serial", "bf16_serial", "fp32_overlap", "bf16_overlap"],
    )
    if matrix.get("protocol", {}).get("measurement_order") != "rotating_order_within_each_round":
        raise RuntimeError("当前内核仍缓存旧版性能运行时；请重启内核后从 5.1 开始重新运行。")
    loads = {name: {item["token_count"]: item for item in data["loads"]} for name, data in matrix["variants"].items()}
    comparisons = []
    for token_count in GPU_CONFIG["token_counts"]:
        base = loads["baseline_fp32_serial"][token_count]
        singles = {name: loads[name][token_count] for name in ("bf16_serial", "fp32_overlap")}
        combo = loads["bf16_overlap"][token_count]
        base_value = base["latency"]["median_ms"]
        combo_value = combo["latency"]["median_ms"]
        single_checks = {
            name: {
                "quality_passed": item["cosine_vs_fp32"] >= GPU_CONFIG["minimum_cosine"],
                "tail_passed": item["latency"]["p95_ms"] / base["latency"]["p95_ms"] <= GPU_CONFIG["maximum_p95_ratio"],
                "faster_than_baseline": item["latency"]["median_ms"] < base_value,
            } for name, item in singles.items()
        }
        eligible = {name: item for name, item in singles.items() if all(single_checks[name].values())}
        best_single = min(eligible, key=lambda name: (eligible[name]["latency"]["median_ms"], name)) if eligible else None
        reference = eligible[best_single] if best_single else base
        reference_name = best_single if best_single else "baseline_fp32_serial"
        best_value = reference["latency"]["median_ms"]
        gain_vs_baseline = (base_value - combo_value) / base_value
        gain_vs_reference = (best_value - combo_value) / best_value
        gain_vs_best_single = gain_vs_reference if best_single else None
        quality_passed = combo["cosine_vs_fp32"] >= GPU_CONFIG["minimum_cosine"]
        combo_p95_ratio = combo["latency"]["p95_ms"] / reference["latency"]["p95_ms"]
        tail_passed = combo_p95_ratio <= GPU_CONFIG["maximum_p95_ratio"]
        if (quality_passed and tail_passed and gain_vs_baseline > 0
                and gain_vs_reference > 0 and gain_vs_reference >= GPU_CONFIG["minimum_marginal_gain"]):
            decision = "keep_combination"
        else:
            decision = "keep_single" if best_single else "rollback_baseline"
        rollback_to = reference_name
        services = combo["service_times"]
        dominant_service = "compute" if services["compute_median_ms"] >= services["h2d_median_ms"] else "h2d"
        comparisons.append({
            "token_count": token_count, "best_single": best_single,
            "single_checks": single_checks, "reference_name": reference_name,
            "gain_vs_reference": gain_vs_reference,
            "gain_vs_baseline": gain_vs_baseline, "gain_vs_best_single": gain_vs_best_single,
            "combination_p95_ratio_vs_best_single": combo_p95_ratio if best_single else None,
            "combination_p95_ratio_vs_reference": combo_p95_ratio, "tail_passed": tail_passed,
            "combination_cosine_vs_fp32": combo["cosine_vs_fp32"],
            "combination_service_times": services, "dominant_service_time": dominant_service,
            "decision": decision, "rollback_to": rollback_to,
        })
    record = {
        "schema_version": "performance-composition-mechanism/v2", "profile_id": PROFILE_ID,
        "semantic_id": SEMANTIC_ID, "run_label": RUN_LABEL,
        "framework": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda},
        "hardware": {"device": torch.cuda.get_device_name(0), "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory},
        "workload": {"id": WORKLOAD_ID, "version": WORKLOAD_VERSION, **GPU_CONFIG},
        "matrix": matrix, "comparisons": comparisons, "failure": None,
        "decision_version": "guarded_fallback/v2",
        "evidence_level": "local_mechanism_benchmark", "decision": "review_per_load",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json_record_once(result_path, record)
    print(json.dumps({"result_path": str(result_path), "comparisons": comparisons}, ensure_ascii=False, indent=2))
else:
    print({"run_mode": RUN_MODE, "profile_id": PROFILE_ID, "semantic_id": SEMANTIC_ID, "config": GPU_CONFIG})
```

### 5.3 读取结果

按已有`RUN_LABEL`独立读取，从四路径样本重算中位数、P95、收益与恢复目标，再按记录版本核对判断。旧版最快单项与当前已检查单项的选择规则分别展示，原始结果不改写。


```python
from pathlib import Path
import json
from tools.project_runtime import resolve_project_root
from tools.performance_result_review import read_composition_result

result_path = resolve_project_root() / RESULTS_RELATIVE_DIR / f"{SEMANTIC_ID.lower()}_{RUN_LABEL}.json"
if not result_path.is_file():
    if RUN_MODE == "real_gpu":
        raise FileNotFoundError(f"未找到结果：{result_path}")
    composition_record = {"status": "not_run", "result_path": str(result_path)}
    print(composition_record)
else:
    composition_record, result_review = read_composition_result(result_path, run_label=RUN_LABEL)
    print(json.dumps(result_review, ensure_ascii=False, indent=2))
    print(json.dumps({key: composition_record.get(key) for key in (
        "schema_version", "run_label", "framework", "hardware", "workload", "decision_version",
        "failure", "evidence_level", "decision", "comparisons",
    )}, ensure_ascii=False, indent=2))
    print("协议：", composition_record.get("matrix", {}).get("protocol"))
    for name, variant in composition_record.get("matrix", {}).get("variants", {}).items():
        for load in variant["loads"]:
            print("路径／Token：", name, load["token_count"])
            print({key: load.get(key) for key in ("first_cuda_ms", "first_wall_ms",
                "service_times", "model_bytes", "buffer_bytes", "matrix_peak_allocated_bytes",
                "cosine_vs_fp32", "checksum")})
            print("轮次 | 时间 ms")
            for trial, value in enumerate(load["latency"]["samples_ms"], 1):
                print(trial, value)
```

### 5.4 解释证据与决策

当前登记为 [rtx5070ti_composition_rotating_v2](../../benchmarks/results/perf_composition/perf-composition_rtx5070ti_composition_rotating_v2.json)，工作负载 v2、结果 schema v1。24 轮轮换四路径，BF16 串行是两档更快单项，其相似度和相对基线的批处理P95也通过本页回退条件；组合在两档均进一步降低时间。

按当前单项检查规则复核历史样本，两档均保留组合。已有数值检查使用独立前向相似度，完整双缓冲流水线的逐元素输出仍待核对。接下来先补这一检查，再扩大负载并重复测量；组合后计算阶段的时间较长，可沿时间线检查它是否限制总完成速度。

| 观察项 | 读取字段 | 当前结果 | 解释与动作 |
| --- | --- | --- | --- |
| 运行与数值范围 | `failure`、协议与组合相似度 | 无失败，24 轮，独立前向相似度通过 | 保留局部性能结果，完整流水线数值仍需补验 |
| 合格参考 | 单项中位数、相似度及 P95 | 两档均为 BF16 串行，检查条件通过 | 已满足本页相似度与 P95 条件，可保留为当前回退配置 |
| 新增收益 | `gain_vs_best_single` 与同轮样本 | 32／256 Token 延迟分别再降约 5.6%／12.2%，两档均 24/24 轮更快 | 这是延迟下降比例，不是吞吐提升百分比 |
| 尾延迟 | 组合／参考的 P95 比值 | 约 0.73／0.89 | 两档均低于 1.05，满足现有尾延迟条件 |
| 再归因 | `dominant_service_time` | 两档均标记计算服务时间较长 | 沿时间线核对计算依赖，再用单因素复测检查缩短计算能否改善总时间 |
| 历史动作 | `decision`、`rollback_to` | 两档 `keep_combination`，回退 BF16 串行 | 保留当前局部组合结论，补齐完整路径正确性后再交给项目复测 |
## 相关阅读

需要把组合放回训练时，比较真实保存与恢复代价；需要解释剩余时间时，再查看依赖与等待。

- [76. 激活重算、卸载与预算比较](./76_Activation_Checkpoint_Offload_Benchmark.md)：在训练更新中比较重算、卸载及其组合。
- [PERF-TRACE. 执行轨迹依赖与关键路径](./PERF-TRACE_Trace_Dependencies_and_Critical_Path.md)：检查阶段变化是否影响最终完成。
