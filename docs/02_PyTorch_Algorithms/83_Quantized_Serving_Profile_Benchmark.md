# 83. Quantized Serving Profile Benchmark | 量化服务 Profile 基准

**难度：** Hard | **环境：** [CPU 主线，GPU 可选](#runtime-environment) | **标签：** `量化压缩`、`服务压力`、`服务目标` | **目标人群：** 理解推理指标、需要比较服务配置的读者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/83_Quantized_Serving_Profile_Benchmark.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

同一量化模型在单请求下响应很快，增加并发后却可能让首个回答或后续内容到达更晚。交互式请求重视等待，离线任务更看重单位时间完成的总输出，两者适合的配置可能不同。

固定已经验证的量化产物和执行路径，再分别测延迟优先、均衡与吞吐优先三类场景。每类场景都有自己的请求压力和目标，结果按场景判断，得到这份产物适用的范围。

<a id="runtime-environment"></a>

## 运行环境

- **CPU 练习：** 使用 [BASE-CPU](../guide.md#base-cpu)，检查来源引用、指标方向和场景内选择。
- **真实实验：** 使用 [INF-VLLM-011-CU128](../topic_discussion/quantization_compression/quickstart.md#q-vllm-quickstart)，需要兼容 GPU 与已通过部署对照的量化产物。
- **执行阶段：** `prepare → measure → close`；准备和收尾阶段读取磁盘记录，不启动服务。

## 前置阅读

先确认使用同一产物和低精度路径，再改变请求压力。

- [量化推理与部署](./67_Quantized_Inference_and_Deployment.md)：提供本项目必需的部署引用。
- [请求选择、准入与公平性](./36_Decode_Scheduling.md)：解释排队、并发与尾部等待。

## Step 1：固定部署对象后再改变请求压力

服务场景（Profile）是一组请求压力、主指标和服务目标的约定，与环境 Profile 的依赖配置是不同对象。场景比较先沿用已接受的量化产物、基座版本和执行路径，再调整请求数、并发与生成上限，观察服务表现怎样变化。

来源引用包含前序结果的位置和内容摘要，使每个场景能追溯到同一产物。运行中同时记录后端、硬件和精度，环境发生变化时重新测量相应场景。

| 固定对象 | 读取记录 | 用途 |
| --- | --- | --- |
| 模型与产物 | 产物编号、版本、文件摘要和来源 | 确定三类场景加载同一对象 |
| 执行路径 | 格式、后端、回退和日志 | 保持低精度路径一致 |
| 设备环境 | 硬件、运行时与后端版本 | 区分负载变化和环境变化 |
| 请求来源 | 请求内容、分词和生成规则 | 从相同来源构造各类压力 |

![固定量化部署后测量三类服务场景](../public/02_PyTorch_Algorithms/83_serving_profile_flow.svg)

## Step 2：三类服务目标怎样选择不同指标

服务等级目标（SLO）为当前场景规定可接受的上限或下限。延迟优先观察客户端首次收到内容的时间，均衡场景约束后续输出间隔并保留吞吐下限，吞吐优先则观察每秒输出量。历史字段使用 TTFT 和 TPOT 名称，具体内容分块与流结束如何计时，沿用当前采集协议。

P95 表示单轮请求延迟中约 95% 不超过的位置；本项目对多轮 P95 求均值，得到重复运行的汇总，而非全部请求合并后的 P95。同一场景内按固定请求压力比较，跨场景分别保留结论。

| 场景 | 请求压力 | 主指标 | 其他条件 |
| --- | --- | --- | --- |
| 延迟优先 | 低并发、短输出 | TTFT P95，越低越好 | 首响应上限、失败率 |
| 均衡 | 中等并发和生成长度 | TPOT P95，越低越好 | 输出吞吐下限、失败率 |
| 吞吐优先 | 较高并发、较长输出 | 输出词元／秒，越高越好 | 吞吐下限、失败率 |

![同一场景先筛选再按指标方向比较](../public/02_PyTorch_Algorithms/83_serving_profile_metrics.svg)

## Step 3：场景内选择与三场景汇总怎样分开

先筛除产物、场景或请求负载不一致的记录，再按全部服务目标过滤。主指标相同时，可以比较各项约束的归一化余量之和：上限指标越低、下限指标越高，余量越大。这个量用于当前规则下的并列候选比较，具体延迟和吞吐仍分别读取。

CPU 选择器支持同一场景的多个候选，当前真实矩阵则每类执行一份固定配置并重复测量。三类都取得完整记录后，再列出各自是否达标；汇总接受只适用于达标场景，缺少场景时先补测。

| 判断层 | 需要满足什么 | 未满足时的动作 |
| --- | --- | --- |
| 单条记录可比 | 同产物、同场景、同请求负载与真实运行 | 排除并保留原因 |
| 单场景过滤 | 所有指标有效且各自达标 | 调整当前场景配置 |
| 单场景排序 | 主指标方向正确，持平时比较余量 | 保留选中记录及其指标 |
| 三场景完整性 | 三类均完成，必要指标齐全 | 缺测则待验证 |
| 汇总结论 | 完整测量后至少一类达标 | 只登记实际达标范围 |

## Step 4：实现来源检查、场景条件与候选选择

题目提供部署来源、服务场景和多份候选记录，骨架负责失败筛选及逐项服务目标判断。请组合来源条件，检查场景是否声明请求压力与指标方向，再选择主指标最优的合格候选。测试涵盖缺来源、非法方向、低延迟、高吞吐和主指标持平。

| 任务与目的 | 已知条件 | 完成标准 | 测试验证 |
| --- | --- | --- | --- |
| TODO 1：检查部署来源 | 接受状态、产物身份和前序引用 | 三项均存在且通过 | 合格记录与缺来源记录 |
| TODO 2：检查场景条件 | 请求数、并发、输出上限、主指标与服务目标 | 数量为正，指标方向合法，主指标有门槛 | 非法方向与缺字段 |
| TODO 3：选择合格候选 | 已完成可比性和服务目标筛选的记录 | 按主指标方向选择，持平时比较余量 | 延迟、吞吐与持平样例 |


```python
from typing import Any, Dict, List

from tools.quantization_project_runtime import _finite_number, _positive_int, _text

```


```python
SUPPORTED_PROFILES = {"latency", "balanced", "throughput"}
SUPPORTED_DIRECTIONS = {"max", "min"}


def validate_deployment_lineage(deployment: Dict[str, Any]) -> Dict[str, Any]:
    """检查 67 部署记录是否可以作为服务 Profile 的唯一上游资产。"""
    artifact = deployment.get("artifact", {})
    lineage = deployment.get("lineage", {})
    # 给定实现：提取 67 接受状态、产物身份和 82 血缘事实。
    accepted = (
        deployment.get("stage") == "quantized_deployment"
        and deployment.get("decision", {}).get("decision") == "accept"
    )
    artifact_ready = bool(artifact.get("artifact_id") and artifact.get("path"))
    gate_lineage_ready = bool(
        lineage.get("artifact_gate_reference") and lineage.get("artifact_gate_sha256")
    )
    # TODO 1（部署血缘）：组合三个不可互相替代的准入条件。
    # lineage_ready = ???  # 接受状态、产物身份和 82 血缘必须同时成立。
    if "lineage_ready" not in locals():
        raise NotImplementedError("TODO 1：请完成部署血缘合取规则")
    issues = []
    if not accepted:
        issues.append("deployment_not_accepted")
    if not artifact_ready:
        issues.append("artifact_identity_or_path_missing")
    if not gate_lineage_ready:
        issues.append("artifact_gate_lineage_missing")
    return {"ready": lineage_ready, "issues": issues}


def _positive_workload(workload: Dict[str, Any]) -> bool:
    """给定实现：检查请求数、并发、生成上限和重复次数是否为正。"""
    return all(
        _positive_int(workload.get(key))
        for key in ("num_prompts", "concurrency", "max_tokens", "repeats")
    )


def _valid_slo_rules(slo: Dict[str, Any]) -> bool:
    """给定实现：检查每项 SLO 是否声明合法方向和数值阈值。"""
    return all(
        isinstance(rule, dict) and rule.get("direction") in SUPPORTED_DIRECTIONS
        and _finite_number(rule.get("limit")) and rule["limit"] >= 0
        for rule in slo.values()
    )


def validate_serving_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """检查 workload、主指标和带方向 SLO 是否构成完整 Profile。"""
    workload = profile.get("workload", {})
    slo = profile.get("slo", {})
    # TODO 2（Profile 契约）：组合 workload、SLO 和主指标约束。
    # workload_ready = ???  # workload ID 存在，且四项规模参数均为正。
    # slo_ready = ???  # SLO 非空，且每项规则都通过方向与阈值检查。
    # primary_ready = ???  # 主指标必须出现在当前 Profile 的 SLO 中。
    if not all(name in locals() for name in ("workload_ready", "slo_ready", "primary_ready")):
        raise NotImplementedError("TODO 2：请完成三个 Profile 契约判断")
    issues = []
    if profile.get("name") not in SUPPORTED_PROFILES:
        issues.append("unsupported_profile")
    if not workload_ready:
        issues.append("invalid_workload")
    if not slo_ready:
        issues.append("invalid_slo")
    if not primary_ready:
        issues.append("primary_metric_missing_from_slo")
    return {"ready": not issues, "issues": issues}


def _meets_slo(metrics: Dict[str, Any], slo: Dict[str, Any]) -> bool:
    """按每项指标声明的 max/min 方向检查 SLO。"""
    for metric, rule in slo.items():
        value = metrics.get(metric)
        if not _finite_number(value) or value < 0:
            return False
        if rule["direction"] == "max" and value > rule["limit"]:
            return False
        if rule["direction"] == "min" and value < rule["limit"]:
            return False
    return True


def _slo_headroom(metrics: Dict[str, Any], slo: Dict[str, Any]) -> float:
    """给定实现：计算已合格记录相对各项 SLO 的归一化余量之和。"""
    headroom = 0.0
    for metric, rule in slo.items():
        value, limit = metrics[metric], rule["limit"]
        scale = max(abs(float(limit)), 1.0)
        headroom += ((limit - value) if rule["direction"] == "max" else (value - limit)) / scale
    return headroom


def choose_primary_record(
    records: List[Dict[str, Any]], primary_metric: str, direction: str, slo: Dict[str, Any]
) -> Dict[str, Any] | None:
    """按主指标方向选择候选；过滤和 SLO 检查已由调用方完成。"""
    if not records:
        return None
    best = records[0]
    for candidate in records[1:]:
        candidate_value = candidate["metrics"][primary_metric]
        best_value = best["metrics"][primary_metric]
        # TODO 3（候选排序）：先比主指标；主指标持平时比较整体 SLO 余量。
        # candidate_is_better = ???
        if "candidate_is_better" not in locals():
            raise NotImplementedError("TODO 3：请完成方向化比较规则")
        if candidate_is_better:
            best = candidate
    return best


def select_serving_candidate(
    records: List[Dict[str, Any]], profile: Dict[str, Any], *, expected_artifact_id: str | None = None
) -> Dict[str, Any]:
    """在同一 Profile 内过滤不可比记录、检查 SLO 并选择主指标最优候选。"""
    if not validate_serving_profile(profile)["ready"]:
        raise ValueError("Profile 契约不完整")
    expected_artifact = expected_artifact_id or next((record.get("artifact_id") for record in records if record.get("artifact_id")), None)
    comparable = [
        record for record in records
        if record.get("status") == "ok"
        and record.get("artifact_id") == expected_artifact
        and record.get("profile_name") == profile["name"]
        and record.get("workload_id") == profile["workload"]["id"]
        and record.get("evidence_level") == "real_project_benchmark"
    ]
    feasible = [
        record for record in comparable
        if _meets_slo(record.get("metrics", {}), profile["slo"])
    ]
    primary_metric = profile["primary_metric"]
    direction = profile["slo"][primary_metric]["direction"]
    selected = choose_primary_record(feasible, primary_metric, direction, profile["slo"])
    excluded = [record.get("name", "unnamed") for record in records if record not in feasible]
    return {
        "decision": "accept" if selected else "tune",
        "selected": selected,
        "comparable_count": len(comparable),
        "feasible_count": len(feasible),
        "excluded": excluded,
    }

```

## 测试

核对部署来源、指标方向和场景内的候选选择。


```python
def _deployment_record():
    """构造已通过 67、且携带 82 血缘的部署记录。"""
    return {
        "stage": "quantized_deployment",
        "artifact": {"artifact_id": "qwen-awq-r1", "path": "/tmp/qwen-awq-r1"},
        "lineage": {"artifact_gate_reference": "82_ref.json", "artifact_gate_sha256": "abc123"},
        "decision": {"decision": "accept"},
    }


def _latency_profile():
    """构造使用上限型延迟 SLO 的交互式 Profile。"""
    return {
        "name": "latency",
        "workload": {"id": "fixed-chat", "num_prompts": 8, "concurrency": 1, "max_tokens": 64, "repeats": 3},
        "primary_metric": "ttft_p95_ms",
        "slo": {
            "ttft_p95_ms": {"direction": "max", "limit": 500.0},
            "failure_rate": {"direction": "max", "limit": 0.0},
        },
    }


def test_deployment_lineage():
    """验证 67 接受状态和 82 血缘缺一不可。"""
    deployment = _deployment_record()
    assert validate_deployment_lineage(deployment)["ready"]
    assert not validate_deployment_lineage({**deployment, "lineage": {}})["ready"]


def test_serving_profile_contract():
    """验证 workload、主指标和 SLO 方向组成完整契约。"""
    profile = _latency_profile()
    assert validate_serving_profile(profile)["ready"]
    bad = {**profile, "slo": {"ttft_p95_ms": {"direction": "smaller", "limit": 500.0}}}
    assert not validate_serving_profile(bad)["ready"]
    for value in (True, 0, -1, 1.5):
        invalid = {**profile, 'workload': {**profile['workload'], 'repeats': value}}
        assert not validate_serving_profile(invalid)['ready']


def test_profile_candidate_selection():
    """验证上限型延迟和下限型吞吐使用相反选择方向。"""
    latency = _latency_profile()
    records = [
        {"name": "c1", "status": "ok", "artifact_id": "qwen-awq-r1", "profile_name": "latency", "workload_id": "fixed-chat", "evidence_level": "real_project_benchmark", "metrics": {"ttft_p95_ms": 420.0, "failure_rate": 0.0}},
        {"name": "c2", "status": "ok", "artifact_id": "qwen-awq-r1", "profile_name": "latency", "workload_id": "fixed-chat", "evidence_level": "real_project_benchmark", "metrics": {"ttft_p95_ms": 360.0, "failure_rate": 0.0}},
        {"name": "failed", "status": "failed", "artifact_id": "qwen-awq-r1", "profile_name": "latency", "workload_id": "fixed-chat", "evidence_level": "real_project_benchmark", "metrics": {}},
    ]
    contaminated = [{**records[0], "name": "wrong-artifact", "artifact_id": "other"}, *records]
    selected = select_serving_candidate(contaminated, latency, expected_artifact_id="qwen-awq-r1")
    assert selected["decision"] == "accept" and selected["selected"]["name"] == "c2"
    for value in (float('nan'), float('inf'), True, -1.0):
        invalid = {**records[0], 'metrics': {'ttft_p95_ms': value, 'failure_rate': 0.0}}
        assert select_serving_candidate([invalid], latency)['decision'] != 'accept'

    throughput = {
        **latency, "name": "throughput", "primary_metric": "throughput_tps",
        "slo": {"throughput_tps": {"direction": "min", "limit": 70.0}},
    }
    throughput_records = [
        {**record, "profile_name": "throughput", "metrics": {"throughput_tps": value}}
        for record, value in zip(records[:2], (80.0, 95.0))
    ]
    assert select_serving_candidate(
        throughput_records, throughput, expected_artifact_id="qwen-awq-r1"
    )["selected"]["name"] == "c2"

    tie_slo = {
        "tpot_p95_ms": {"direction": "max", "limit": 80.0},
        "throughput_tps": {"direction": "min", "limit": 20.0},
    }
    tied = [
        {"name": "narrow", "metrics": {"tpot_p95_ms": 60.0, "throughput_tps": 22.0}},
        {"name": "roomy", "metrics": {"tpot_p95_ms": 60.0, "throughput_tps": 30.0}},
    ]
    assert choose_primary_record(tied, "tpot_p95_ms", "max", tie_slo)["name"] == "roomy"


def run_serving_profile_tests():
    """按项目顺序运行血缘、Profile 契约与候选选择测试。"""
    test_deployment_lineage()
    test_serving_profile_contract()
    test_profile_candidate_selection()
    print("✅ 部署血缘、Profile 契约和方向化 SLO 选择均已验证。")


run_serving_profile_tests()

```

---

## 参考代码与解析

### 代码


```python
SUPPORTED_PROFILES = {"latency", "balanced", "throughput"}
SUPPORTED_DIRECTIONS = {"max", "min"}


def validate_deployment_lineage(deployment: Dict[str, Any]) -> Dict[str, Any]:
    """检查 67 部署记录是否可以作为服务 Profile 的唯一上游资产。"""
    artifact = deployment.get("artifact", {})
    lineage = deployment.get("lineage", {})
    # 给定实现：提取 67 接受状态、产物身份和 82 血缘事实。
    accepted = (
        deployment.get("stage") == "quantized_deployment"
        and deployment.get("decision", {}).get("decision") == "accept"
    )
    artifact_ready = bool(artifact.get("artifact_id") and artifact.get("path"))
    gate_lineage_ready = bool(
        lineage.get("artifact_gate_reference") and lineage.get("artifact_gate_sha256")
    )
    # TODO 1（部署血缘）：组合三个不可互相替代的准入条件。
    lineage_ready = accepted and artifact_ready and gate_lineage_ready
    issues = []
    if not accepted:
        issues.append("deployment_not_accepted")
    if not artifact_ready:
        issues.append("artifact_identity_or_path_missing")
    if not gate_lineage_ready:
        issues.append("artifact_gate_lineage_missing")
    return {"ready": lineage_ready, "issues": issues}


def _positive_workload(workload: Dict[str, Any]) -> bool:
    """给定实现：检查请求数、并发、生成上限和重复次数是否为正。"""
    return all(
        _positive_int(workload.get(key))
        for key in ("num_prompts", "concurrency", "max_tokens", "repeats")
    )


def _valid_slo_rules(slo: Dict[str, Any]) -> bool:
    """给定实现：检查每项 SLO 是否声明合法方向和数值阈值。"""
    return all(
        isinstance(rule, dict) and rule.get("direction") in SUPPORTED_DIRECTIONS
        and _finite_number(rule.get("limit")) and rule["limit"] >= 0
        for rule in slo.values()
    )


def validate_serving_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """检查 workload、主指标和带方向 SLO 是否构成完整 Profile。"""
    workload = profile.get("workload", {})
    slo = profile.get("slo", {})
    # TODO 2（Profile 契约）：组合 workload、SLO 和主指标约束。
    workload_ready = bool(workload.get("id")) and _positive_workload(workload)
    slo_ready = bool(slo) and _valid_slo_rules(slo)
    primary_ready = profile.get("primary_metric") in slo
    issues = []
    if profile.get("name") not in SUPPORTED_PROFILES:
        issues.append("unsupported_profile")
    if not workload_ready:
        issues.append("invalid_workload")
    if not slo_ready:
        issues.append("invalid_slo")
    if not primary_ready:
        issues.append("primary_metric_missing_from_slo")
    return {"ready": not issues, "issues": issues}


def _meets_slo(metrics: Dict[str, Any], slo: Dict[str, Any]) -> bool:
    """按每项指标声明的 max/min 方向检查 SLO。"""
    for metric, rule in slo.items():
        value = metrics.get(metric)
        if not _finite_number(value) or value < 0:
            return False
        if rule["direction"] == "max" and value > rule["limit"]:
            return False
        if rule["direction"] == "min" and value < rule["limit"]:
            return False
    return True


def _slo_headroom(metrics: Dict[str, Any], slo: Dict[str, Any]) -> float:
    """给定实现：计算已合格记录相对各项 SLO 的归一化余量之和。"""
    headroom = 0.0
    for metric, rule in slo.items():
        value, limit = metrics[metric], rule["limit"]
        scale = max(abs(float(limit)), 1.0)
        headroom += ((limit - value) if rule["direction"] == "max" else (value - limit)) / scale
    return headroom


def choose_primary_record(
    records: List[Dict[str, Any]], primary_metric: str, direction: str, slo: Dict[str, Any]
) -> Dict[str, Any] | None:
    """TODO 3：按主指标方向选择已经满足 SLO 的候选。"""
    if not records:
        return None
    best = records[0]
    for candidate in records[1:]:
        candidate_value = candidate["metrics"][primary_metric]
        best_value = best["metrics"][primary_metric]
        primary_is_better = candidate_value < best_value if direction == "max" else candidate_value > best_value
        primary_is_tied = candidate_value == best_value
        candidate_is_better = primary_is_better or (
            primary_is_tied
            and _slo_headroom(candidate["metrics"], slo) > _slo_headroom(best["metrics"], slo)
        )
        if candidate_is_better:
            best = candidate
    return best


def select_serving_candidate(
    records: List[Dict[str, Any]], profile: Dict[str, Any], *, expected_artifact_id: str | None = None
) -> Dict[str, Any]:
    """在同一 Profile 内过滤不可比记录、检查 SLO 并选择主指标最优候选。"""
    if not validate_serving_profile(profile)["ready"]:
        raise ValueError("Profile 契约不完整")
    expected_artifact = expected_artifact_id or next((record.get("artifact_id") for record in records if record.get("artifact_id")), None)
    comparable = [
        record for record in records
        if record.get("status") == "ok"
        and record.get("artifact_id") == expected_artifact
        and record.get("profile_name") == profile["name"]
        and record.get("workload_id") == profile["workload"]["id"]
        and record.get("evidence_level") == "real_project_benchmark"
    ]
    feasible = [
        record for record in comparable
        if _meets_slo(record.get("metrics", {}), profile["slo"])
    ]
    primary_metric = profile["primary_metric"]
    direction = profile["slo"][primary_metric]["direction"]
    selected = choose_primary_record(feasible, primary_metric, direction, profile["slo"])
    excluded = [record.get("name", "unnamed") for record in records if record not in feasible]
    return {
        "decision": "accept" if selected else "tune",
        "selected": selected,
        "comparable_count": len(comparable),
        "feasible_count": len(feasible),
        "excluded": excluded,
    }

```

### 解析

#### TODO 1：检查部署来源

- 已接受部署、产物身份和前序产物引用共同确定本次输入；三者各自提供必要信息。
- 答案取三个事实的合取，测试删除嵌入来源记录，核对缺失状态会阻止后续比较。

#### TODO 2：检查场景条件

- 请求数、并发、生成上限和重复次数为正；服务目标各自声明 `max` 或 `min`，主指标也需出现在其中。
- 答案分别判断工作负载、服务目标和主指标。测试使用非法方向和缺字段，核对条件不会静默通过。

#### TODO 3：选择合格候选

- 骨架已过滤失败、错配与未达标记录；主指标按上限方向取更小值、下限方向取更大值。
- 答案在主指标持平时比较归一化余量之和。测试分别构造延迟、吞吐和持平情况，核对排序方向。

## Step 5：测量量化服务压力

在三类固定场景中分别测量同一量化部署，并按各自服务目标判断。三类记录齐全后才汇总：至少一类达标可接受对应场景，其余场景仍保留调整结论。

| 路径 | 核对对象 | 输出 |
| --- | --- | --- |
| CPU | 部署来源、指标方向与场景内选择 | 规则与边界测试 |
| 真实后端 | 同一产物、三类固定请求压力 | 请求结果、日志及运行身份 |
| 收尾 | 来源、文件摘要与测量完整性 | 各场景结论、汇总报告和回执 |

### 5.1 环境与运行配置

使用 [INF-VLLM-011-CU128](../topic_discussion/quantization_compression/quickstart.md#q-vllm-quickstart)，填写唯一标签、阶段和部署结果标签。默认 `prepare`，固定模型、请求与门槛由下一步加载。


```python
RUN_LABEL = 'replace_with_unique_label'  # 本次唯一标签；不覆盖既有结果。
RUN_STAGE = 'prepare'  # prepare / measure / close；默认只读预检。
UPSTREAM_RUN_LABEL = 'replace_with_upstream_label'  # 67部署已接受结果的唯一标签。

```

### 5.2 环境与唯一上游预检

从67部署引用读取来源，核对SHA256、接受状态、嵌入的候选清单、实际产物文件与固定请求集。只有首次measure会检查GPU环境；prepare、close和已有测量的读取不初始化GPU。


```python
import importlib
import json

from tools.project_runtime import resolve_project_root
import tools.quantized_deployment_runtime as deployment_runtime

deployment_runtime = importlib.reload(deployment_runtime)
plan = deployment_runtime.project_plan(resolve_project_root(), '83', RUN_LABEL, UPSTREAM_RUN_LABEL, RUN_STAGE)
prepared = deployment_runtime.preflight_project(plan)
print({'ready': prepared['ready'], 'issues': prepared['issues'],
       'profile': plan['contract']['environment_profile'], 'inputs': prepared.get('snapshot')})

```

### 5.3 核对压力矩阵与固定条件

三类场景共用产物、后端、硬件和请求来源；每类明确请求数、并发、输出上限、重复次数、主指标和SLO。重复实验的p95采用各轮p95均值，不能解释为合并所有请求后的p95。


```python
print({'fixed_contract': plan['contract'],
       'artifact': prepared.get('artifact'),
       'quality_source': '67_deployment_evidence'})

```

### 5.4 真实执行并保存原始证据

measure按固定矩阵执行，逐项保存来源身份、原始请求结果、完整日志和失败。完整成员可在中断后复用；后端输出已存在但成员记录未完成时保留现场并使用新标签，避免覆盖。此处不生成稳定引用。


```python
if RUN_STAGE == 'measure':
    if not prepared['ready']:
        raise RuntimeError('预检未通过：' + '; '.join(prepared['issues']))
    measured = deployment_runtime.measure_project(plan)
    print({'status': measured['status'], 'decision': measured['decision']})
else:
    print({'status': 'skipped', 'stage': RUN_STAGE})

```

### 5.5 只读核对已保存证据

重新定位本标签的记录，逐项检查来源、固定合同、成员数量、身份、实际请求数及文件哈希，再重算决策。已有报告、交接和回执必须与原始证据一致；读取不会启动后端或补写结果。


```python
if prepared['ready']:
    checked = deployment_runtime.read_project(plan)
else:
    checked = {'status': 'blocked', 'decision': {'decision': 'pending'}, 'issues': prepared['issues']}
print({'status': checked['status'], 'decision': checked['decision'],
       'profiles': checked.get('report', {}).get('profile_decisions'),
       'issues': checked.get('issues', [])})

```

### 5.6 形成决策、登记与交接

close只消费已保存证据，生成报告和SHA256回执；内容一致时复用，中断时补齐缺失资产，冲突时停止。先核对三类压力均完成，再分别记录各场景是否达标；缺测时保持待验证。

| 已核对证据 | 决策 | 下一动作 |
|---|---|---|
| 测量不完整、失败或关键指标缺失 | pending | 补齐证据；失败复测使用新标签 |
| 场景间的后端、硬件或精度身份发生变化 | reject | 保留报告，修复比较条件 |
| 三类均完成，但没有一类达到自身SLO | tune | 调整后用新标签复测 |
| 三类均完成，至少一类达到自身SLO | accept | 仅登记实际达标的服务场景 |

**当前登记结果**：尚无本页新协议的真实 GPU 结果。完成固定协议的真实测量后登记标签与结论；当前状态见[实验资产表](../topic_discussion/quantization_compression/experiment_assets.md)。


```python
if RUN_STAGE == 'close':
    closed = deployment_runtime.close_project(plan) if prepared['ready'] else checked
    print(closed)
else:
    print({'status': 'skipped', 'stage': RUN_STAGE})

```

## 相关阅读

需要解释压测曲线或增加后端时，再核对相应计时协议。

- [服务调度基准](./70_Serving_Scheduler_Benchmark.md)：请求选择对响应顺序与尾延迟的影响。
- [vLLM 基准测试](https://docs.vllm.ai/en/latest/contributing/benchmarks.html)：请求并发、重复运行与指标入口。
