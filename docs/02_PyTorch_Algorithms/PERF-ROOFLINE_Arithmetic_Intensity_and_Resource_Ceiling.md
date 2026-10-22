# PERF-ROOFLINE. Arithmetic Intensity and Resource Ceiling | 算术强度与资源上界

**难度：** Medium | **环境：** [CPU 主线，GPU 可选](#runtime-environment) | **标签：** `性能分析` `Roofline` `算术强度` | **目标人群：** 算子性能分析学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/PERF-ROOFLINE_Arithmetic_Intensity_and_Resource_Ceiling.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

矩阵乘法算得多、搬得少时，可能受计算能力限制；搬运多、复用少时，可能受带宽限制。Roofline 把这两种资源上界放在同一张图上。

先从矩阵形状估算运算量和最低搬运量，再把设备参考与实测放到同一图中解释差距。

**关键词：** 算术强度、最低搬运量、机器平衡点、经验吞吐、Roofline

<a id="runtime-environment"></a>

## 运行环境

- **基础练习：** 使用 [BASE-CPU](../guide.md#base-cpu)，运行 [Step 4 的题目与测试](#mechanism-practice)。
- **GPU 实测：** 阅读 [Step 5](#gpu-experiment)；环境和固定工作负载见其中 5.1。
## 前置阅读

先掌握参数量、FLOPs 和基本性能分析，再用矩阵乘法建立 Roofline 模型。

- [Part 01 · 02 参数量与 FLOPs](../01_Hardware_Math_and_Systems/02_LLM_Params_and_FLOPs.md)
- [Part 01 · 13 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
## Step 1：矩阵形状怎样决定运算量与最低搬运量

对于矩阵乘法 $C=AB$，设 $A$ 的形状为 $m\times k$，$B$ 为 $k\times n$，输出 $C$ 为 $m\times n$。正整数 $m,n,k$ 表示矩阵维度，$d$ 表示每个元素占用的字节数；乘法与加法各计一次浮点运算。用每个输出元素约 $k$ 次乘加估算运算量 $F$，并假设两个输入各读取一次、输出写入一次，得到最低搬运量 $Q_{\min}$ 与模型算术强度 $I_{\mathrm{model}}$：

$$
F\approx 2mnk,\qquad
Q_{\min}=d(mk+kn+mn),\qquad
I_{\mathrm{model}}=\frac{F}{Q_{\min}}.
$$

$F$ 的单位为 FLOPs，$Q_{\min}$ 为 Byte，强度为 FLOPs/Byte。该模型假设输入来自所分析的存储层、输出需要写回，且不额外读取旧输出。真实显存流量可能因重复读取而增加，也可能因跨次运行命中缓存而减少；因此不能把模型强度直接当作硬件计数器测得的实际强度。

| 数量 | 计算假设 | 对判断的影响 |
| --- | --- | --- |
| 运算次数 | 稠密矩阵乘法约为 $2mnk$，不包含其他算子 | 融合、稀疏或额外逐元素运算后需重算 |
| 最低搬运量 | 两个输入与输出使用相同元素字节数，各搬运一次 | 分块复用影响真实流量；不同存储层不能混用带宽 |
| 模型算术强度 | 同一次运算的 FLOPs 除以上述字节数 | 用于提出资源假设，不直接测得缓存行为 |

![从工作量模型到资源判断](../public/02_PyTorch_Algorithms/perf_roofline_overview.svg)
## Step 2：计算吞吐与带宽怎样共同限制执行时间

设 $P$ 为所选精度路径的计算吞吐上界，单位 FLOPs/s；$B$ 为所分析存储层的带宽，单位 Byte/s；$Q$ 为同一层的数据搬运字节数，$I=F/Q$ 为对应算术强度。计算与搬运即使完全重叠，执行时间 $T$ 也不能小于两者所需时间的较大值，因此吞吐上界 $R$ 和机器平衡点 $I_*$ 为：

$$
T\ge\max\left(\frac{F}{P},\frac{Q}{B}\right),\qquad
R(I)=\min(P,IB),\qquad I_*=\frac{P}{B}.
$$

$I_*$ 的单位同样为 FLOPs/Byte。实际可持续能力取决于数据类型、精度策略、形状与缓存条件；用一次矩阵乘法和复制测量替代 $P,B$ 时，得到的是经验参考曲线，不再是严格硬件上界。使用最低搬运模型时，区间判断也依赖其复用假设。

| 强度位置 | 模型限制项 | 优先核对什么 |
| --- | --- | --- |
| $I<I_*$ | $IB$，带宽侧斜线 | 实际搬运、复用和融合是否改变耗时 |
| $I>I_*$ | $P$，计算侧平台 | 运算路径、并行度和计算内核效率 |
| $I=I_*$ | 两项相等 | 两类资源同时达到分界；代码约定归入计算侧标签 |
## Step 3：实测点偏离屋顶线时怎样排查

设实测时间为 $T_{\mathrm{meas}}$，实测吞吐为 $P_{\mathrm{meas}}=F/T_{\mathrm{meas}}$，相对屋顶线的比值为 $\eta$。计算时必须使用同一个工作负载的运算量与时间：

$$
\eta=\frac{P_{\mathrm{meas}}}{R(I)}.
$$

使用理想上界时，明显大于 1 的比值提示运算计数、计时或资源假设需要复核；使用经验参考时，比值可以超过 1，可能来自形状更适合、缓存命中或采样波动。比值低于 1 也不能单凭差距定位瓶颈，启动、同步、排队和执行效率都可能增加时间。下图使用理想化双对数坐标示意两类区间，不代表本机实测结果。

| 观察 | 可以支持的判断 | 下一项验证 |
| --- | --- | --- |
| 低强度且接近斜线 | 搬运可能限制当前算子 | 减少搬运或提高复用后，时间是否下降 |
| 高强度但远低于平台 | 计算侧模型仍有较大差距 | 检查计时范围、计算内核和启动等待 |
| 超过经验参考 | 本次吞吐高于参考工作负载 | 重测参考与候选，核对形状、精度和缓存 |
| 局部吞吐改善 | 当前算子的执行发生变化 | 用执行时间线判断端到端关键路径是否缩短 |

![算术强度与两类资源区间](../public/02_PyTorch_Algorithms/perf_roofline_regimes.svg)
<a id="mechanism-practice"></a>

## Step 4：计算矩阵工作量、资源区间与相对吞吐

输入矩阵维度、元素字节数、计算吞吐和带宽，依次建立工作量模型、取两类资源限制中的较小值，再计算实测吞吐的相对比值。骨架负责输入校验；你将补全三处计算。输入的资源值既可以是理想上界，也可以是经验参考，报告保留大于 1 的比值并提示复核，不将其截断为 1。

| TODO / 函数 | 学习者完成的机制 | 必须满足的约束 | 测试证据 |
| --- | --- | --- | --- |
| TODO 1：计算矩阵工作量<br>`matmul_workload` | 计算 FLOPs、最低搬运量与算术强度 | 维度和元素字节数为正整数，不接受布尔值 | 矩形、方阵和不同字节数；零、负数与非整数拒绝 |
| TODO 2：选择资源限制项<br>`roofline_bound` | 比较计算吞吐与强度乘带宽 | 输入有限且为正；平衡点两项相等时采用计算侧标签 | 带宽侧、计算侧、分界点及非有限值 |
| TODO 3：计算相对吞吐<br>`roofline_report` | 计算实测吞吐与资源参考的比值 | 允许零吞吐；大于 1 不截断，返回复核提示 | 零、正常与超过参考的吞吐；负值和非有限值拒绝 |

```python
from math import isfinite
from typing import Dict
```


```python
def matmul_workload(m: int, n: int, k: int, dtype_bytes: int) -> Dict[str, float]:
    """计算稠密矩阵乘法的运算量与单次读写模型。"""
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
           for value in (m, n, k, dtype_bytes)):
        raise ValueError("矩阵维度与元素字节数必须为正整数")
    # TODO 1：计算矩阵工作量
    # flops = ???
    # minimum_bytes = ???
    raise NotImplementedError("TODO 1：请计算矩阵工作量")
    return {"flops": flops, "minimum_bytes": minimum_bytes,
            "arithmetic_intensity": flops / minimum_bytes}


def roofline_bound(arithmetic_intensity: float, peak_flops: float,
                   bandwidth_bytes_per_s: float) -> Dict[str, object]:
    """使用统一单位计算资源限制项；平衡点约定归入计算侧。"""
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not isfinite(value) or value <= 0
           for value in (arithmetic_intensity, peak_flops, bandwidth_bytes_per_s)):
        raise ValueError("强度、计算吞吐与带宽必须为有限正数")
    bandwidth_bound = arithmetic_intensity * bandwidth_bytes_per_s
    if not isfinite(bandwidth_bound):
        raise ValueError("强度乘带宽超出有限数值范围")
    # TODO 2：选择资源限制项
    # bound_flops = ???
    # regime = ???
    raise NotImplementedError("TODO 2：请选择资源限制项")
    return {"machine_balance": peak_flops / bandwidth_bytes_per_s,
            "bandwidth_bound_flops": bandwidth_bound,
            "bound_flops": bound_flops, "regime": regime}


def roofline_report(workload: Dict[str, float], machine: Dict[str, object],
                    measured_flops: float) -> Dict[str, object]:
    """保留真实相对比值，超过参考时提示核对测量条件。"""
    values = (workload["arithmetic_intensity"], machine["bound_flops"], measured_flops)
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not isfinite(value) for value in values):
        raise ValueError("强度、资源参考与实测吞吐必须有限")
    if values[0] <= 0 or values[1] <= 0 or measured_flops < 0:
        raise ValueError("强度与资源参考为正，实测吞吐不能为负")
    if machine["regime"] not in {"bandwidth_bound", "compute_bound"}:
        raise ValueError("未知资源区间")
    # TODO 3：计算相对吞吐
    # efficiency = ???
    raise NotImplementedError("TODO 3：请计算相对吞吐")
    return {"arithmetic_intensity": workload["arithmetic_intensity"],
            "regime": machine["regime"], "bound_flops": machine["bound_flops"],
            "measured_flops": measured_flops, "roofline_efficiency": efficiency,
            "measurement_review_required": efficiency > 1.0}
```

## 测试

```python
def _expect_roofline_value_error(function, *args):
    """非法输入必须明确拒绝，不能生成看似有效的比例。"""
    try:
        function(*args)
    except ValueError:
        return
    raise AssertionError("非法输入未被拒绝")


def test_matmul_workload():
    report = matmul_workload(2, 3, 4, 2)
    assert report["flops"] == 48
    assert report["minimum_bytes"] == 52
    assert abs(report["arithmetic_intensity"] - 48 / 52) < 1e-12
    square = matmul_workload(8, 8, 8, 4)
    assert square["flops"] == 1024 and square["minimum_bytes"] == 768


def test_dtype_scaling():
    narrow = matmul_workload(8, 8, 8, 2)
    wide = matmul_workload(8, 8, 8, 4)
    assert narrow["flops"] == wide["flops"]
    assert narrow["minimum_bytes"] * 2 == wide["minimum_bytes"]
    assert narrow["arithmetic_intensity"] == 2 * wide["arithmetic_intensity"]


def test_invalid_workload_inputs():
    for index in range(4):
        for invalid in (0, -1, 1.5, True, float("nan"), float("inf")):
            args = [2, 3, 4, 2]
            args[index] = invalid
            _expect_roofline_value_error(matmul_workload, *args)


def test_resource_regimes():
    bandwidth = roofline_bound(2.0, 1000.0, 100.0)
    compute = roofline_bound(20.0, 1000.0, 100.0)
    assert bandwidth["machine_balance"] == 10
    assert bandwidth["bound_flops"] == 200 and bandwidth["regime"] == "bandwidth_bound"
    assert compute["bound_flops"] == 1000 and compute["regime"] == "compute_bound"


def test_balance_point():
    balanced = roofline_bound(10.0, 1000.0, 100.0)
    assert balanced["bandwidth_bound_flops"] == balanced["bound_flops"] == 1000
    assert balanced["regime"] == "compute_bound"


def test_invalid_resource_inputs():
    for index in range(3):
        for invalid in (0, -1, True, float("nan"), float("inf")):
            args = [2.0, 1000.0, 100.0]
            args[index] = invalid
            _expect_roofline_value_error(roofline_bound, *args)
    _expect_roofline_value_error(roofline_bound, 1e308, 1000.0, 1e308)


def test_relative_throughput():
    workload = {"arithmetic_intensity": 2.0}
    machine = roofline_bound(2.0, 1000.0, 100.0)
    for measured, expected in ((0.0, 0.0), (100.0, 0.5), (200.0, 1.0), (202.0, 1.01)):
        report = roofline_report(workload, machine, measured)
        assert abs(report["roofline_efficiency"] - expected) < 1e-12
        assert report["measurement_review_required"] == (expected > 1)
        assert report["regime"] == machine["regime"]


def test_invalid_report_inputs():
    workload = {"arithmetic_intensity": 2.0}
    machine = roofline_bound(2.0, 1000.0, 100.0)
    for invalid in (-1, True, float("nan"), float("inf")):
        _expect_roofline_value_error(roofline_report, workload, machine, invalid)
    _expect_roofline_value_error(roofline_report, {"arithmetic_intensity": 0}, machine, 1)
    _expect_roofline_value_error(roofline_report, workload, {**machine, "bound_flops": 0}, 1)
    _expect_roofline_value_error(roofline_report, workload, {**machine, "regime": "unknown"}, 1)


def run_roofline_tests():
    """核对工作量、资源分界与相对吞吐的八组行为。"""
    for test in (test_matmul_workload, test_dtype_scaling, test_invalid_workload_inputs,
                 test_resource_regimes, test_balance_point, test_invalid_resource_inputs,
                 test_relative_throughput, test_invalid_report_inputs):
        test()
    print("✅ 工作量、资源分界与相对吞吐测试通过")


run_roofline_tests()
```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果遇到困难，再向下查看参考代码与解析。
<br><br><br><br><br><br><br><br><br><br>

---
## 参考代码与解析

### 代码

```python
def matmul_workload(m: int, n: int, k: int, dtype_bytes: int) -> Dict[str, float]:
    """计算稠密矩阵乘法的运算量与单次读写模型。"""
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
           for value in (m, n, k, dtype_bytes)):
        raise ValueError("矩阵维度与元素字节数必须为正整数")
    # TODO 1：计算矩阵工作量
    flops = float(2 * m * n * k)
    minimum_bytes = float((m * k + k * n + m * n) * dtype_bytes)
    return {"flops": flops, "minimum_bytes": minimum_bytes,
            "arithmetic_intensity": flops / minimum_bytes}


def roofline_bound(arithmetic_intensity: float, peak_flops: float,
                   bandwidth_bytes_per_s: float) -> Dict[str, object]:
    """使用统一单位计算资源限制项；平衡点约定归入计算侧。"""
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not isfinite(value) or value <= 0
           for value in (arithmetic_intensity, peak_flops, bandwidth_bytes_per_s)):
        raise ValueError("强度、计算吞吐与带宽必须为有限正数")
    bandwidth_bound = arithmetic_intensity * bandwidth_bytes_per_s
    if not isfinite(bandwidth_bound):
        raise ValueError("强度乘带宽超出有限数值范围")
    # TODO 2：选择资源限制项
    bound_flops = min(peak_flops, bandwidth_bound)
    regime = "bandwidth_bound" if bandwidth_bound < peak_flops else "compute_bound"
    return {"machine_balance": peak_flops / bandwidth_bytes_per_s,
            "bandwidth_bound_flops": bandwidth_bound,
            "bound_flops": bound_flops, "regime": regime}


def roofline_report(workload: Dict[str, float], machine: Dict[str, object],
                    measured_flops: float) -> Dict[str, object]:
    """保留真实相对比值，超过参考时提示核对测量条件。"""
    values = (workload["arithmetic_intensity"], machine["bound_flops"], measured_flops)
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not isfinite(value) for value in values):
        raise ValueError("强度、资源参考与实测吞吐必须有限")
    if values[0] <= 0 or values[1] <= 0 or measured_flops < 0:
        raise ValueError("强度与资源参考为正，实测吞吐不能为负")
    if machine["regime"] not in {"bandwidth_bound", "compute_bound"}:
        raise ValueError("未知资源区间")
    # TODO 3：计算相对吞吐
    efficiency = measured_flops / machine["bound_flops"]
    return {"arithmetic_intensity": workload["arithmetic_intensity"],
            "regime": machine["regime"], "bound_flops": machine["bound_flops"],
            "measured_flops": measured_flops, "roofline_efficiency": efficiency,
            "measurement_review_required": efficiency > 1.0}
```

### 解析

#### TODO 1：计算矩阵工作量

- 每个输出约需 $k$ 次乘加，因此运算量约为 $2mnk$；字节模型累加两个输入和一个输出，不把运算次数误当搬运量。
- 参考实现分别计算两者，再相除得到强度；矩形与方阵测试检查形状关系，元素字节数测试要求运算量不变、搬运量按比例变化。

#### TODO 2：选择资源限制项

- 计算吞吐与强度乘带宽共同限制结果，取较小值；两项相等时只是标签约定，不意味着带宽限制消失。
- 参考实现保留平衡点与两个限制项；两侧区间和分界点测试检查比较方向，非法输入测试拒绝非有限值和乘积溢出。

#### TODO 3：计算相对吞吐

- 相对比值直接用实测吞吐除以资源参考；不能截断超过 1 的值来掩盖经验参考不够高或测量条件不一致。
- 参考实现同时返回比值与复核提示；测试覆盖零吞吐、等于参考、超过参考和非法值，防止无效数据进入判断。
<a id="gpu-experiment"></a>

## Step 5：GPU 机制实验——测量本机带宽与计算参考

CPU 练习计算工作量与资源区间；GPU 实验使用设备内复制和两个不同尺寸的矩阵乘法，测量经验带宽、参考计算吞吐与被评估算子的吞吐。复制字节数按读取和写入各一次计算，不是只用源张量大小。

| 学习问题 | CPU 已确认 | GPU 要确认与后续使用 |
| --- | --- | --- |
| 模型强度相对本机参考落在哪类资源区间 | 运算量、单次读写字节模型、平衡点与相对比值 | 同环境下的带宽、计算参考与候选吞吐；当前实验不记录硬件实际流量，也不验证端到端收益 |

<a id="gpu-experiment-setup"></a>

### 5.1 环境与固定工作负载

本实验使用 `PERF-CUDA-MECHANISM`，首次运行按[专题 Quickstart](../topic_discussion/performance_analysis/quickstart.md#perf-cuda-mechanism-quickstart)创建和预检环境，结果登记见[实验资产表](../topic_discussion/performance_analysis/experiment_assets.md)。

| 配置层 | 本节固定设置 | 修改规则 |
| --- | --- | --- |
| 运行环境 | 单张 CUDA GPU；`PERF-CUDA-MECHANISM`；参考显存 8 GiB | Profile 预检未通过时停止 |
| 数据来源 | 本机随机 FP32 张量，无外部下载；未固定随机种子 | 复测不保证输入逐元素相同 |
| 带宽参考 | 256 MiB 源张量与同尺寸目标张量，设备内复制 | 流量模型为读＋写共 512 MiB；不是硬件计数器读数 |
| 计算参考 | 4096 方阵乘法；`float32_matmul_precision=high` | 这是 FP32 存储及指定精度策略，不据此宣称全程使用完整 FP32 乘法精度 |
| 被评估算子 | 2048 方阵乘法，与参考分开采样 | 改变形状、元素类型或精度策略后使用新标签 |
| 测量 | 10 次预热、每类 20 次正式测量；CUDA 事件时间中位数 | 当前按复制、参考、候选分组采样，不是轮换配对；不包括数值误差核对 |
| 结果资产 | `benchmarks/results/perf_roofline/<workload_id>_<run_label>.json` | 防止覆盖；5.3 可独立读取历史结果 |

```python
RUN_MODE = "dry_run"  # 默认值 dry_run；改为 real_gpu 后建立本机经验 Roofline。
RUN_LABEL = "default"  # 默认值 default；正式复测使用唯一标签并保留历史结果。
ENVIRONMENT_PROFILE = "PERF-CUDA-MECHANISM"  # 运行前必须通过的环境配置 ID。
SEMANTIC_ID = "PERF-ROOFLINE"  # 机制页稳定标识。
WORKLOAD_ID = "perf-empirical-roofline"  # 经验计算与带宽上界的稳定标识。
WORKLOAD_VERSION = "v2"  # v2 固定 FP32 matmul 精度策略并统一结果身份。
MATMUL_PRECISION = "high"  # 默认值 high；改变后经验计算上界不可与旧结果直接比较。
COPY_ELEMENTS = 64 * 1024 * 1024  # FP32 元素数；默认复制约 256 MiB。
PEAK_MATMUL_SIZE = 4096  # 建立经验计算上界的方阵边长；显存不足时下调并使用新标签。
CANDIDATE_MATMUL_SIZE = 2048  # 被评估矩阵乘法的方阵边长，必须与上界测量分开。
WARMUP_STEPS = 10  # 不进入正式结果的预热次数。
REPEAT_STEPS = 20  # 复制与矩阵乘法各自的正式重复次数。
RESULT_RELATIVE_DIR = "benchmarks/results/perf_roofline"  # 结果 JSON 的仓库相对目录。
```

### 5.2 执行实验并保存 JSON

先验证配置与环境，再分组测量复制、参考矩阵乘法和被评估矩阵乘法。已有同标签记录时停止；默认 `dry_run` 不导入 CUDA 框架，也不分配设备张量。

```python
from pathlib import Path
import json
import math
import statistics
import sys
import subprocess


def _gpu_matmul_workload(size, dtype_bytes=4):
    """按相同公式计算方阵模型，使 GPU 实验可独立运行。"""
    flops = float(2 * size ** 3)
    minimum_bytes = float(3 * size ** 2 * dtype_bytes)
    return {"flops": flops, "minimum_bytes": minimum_bytes,
            "arithmetic_intensity": flops / minimum_bytes}


def _gpu_roofline_report(workload, peak_flops, bandwidth_bytes_per_s, measured_flops):
    """使用经验参考生成比值，允许候选超过参考。"""
    intensity = workload["arithmetic_intensity"]
    bandwidth_bound = intensity * bandwidth_bytes_per_s
    bound_flops = min(peak_flops, bandwidth_bound)
    regime = "bandwidth_bound" if bandwidth_bound < peak_flops else "compute_bound"
    return {
        "arithmetic_intensity": intensity,
        "machine_balance": peak_flops / bandwidth_bytes_per_s,
        "regime": regime,
        "bound_flops": bound_flops,
        "measured_flops": measured_flops,
        "roofline_efficiency": measured_flops / bound_flops,
    }

if RUN_MODE not in {"dry_run", "real_gpu"}:
    raise ValueError("RUN_MODE 只能为 dry_run 或 real_gpu")
for name, value in {"COPY_ELEMENTS": COPY_ELEMENTS, "PEAK_MATMUL_SIZE": PEAK_MATMUL_SIZE,
                    "CANDIDATE_MATMUL_SIZE": CANDIDATE_MATMUL_SIZE,
                    "WARMUP_STEPS": WARMUP_STEPS, "REPEAT_STEPS": REPEAT_STEPS}.items():
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} 必须为正整数")
if MATMUL_PRECISION not in {"highest", "high", "medium"}:
    raise ValueError("未知矩阵乘法精度策略")

if RUN_MODE == "real_gpu":
    import torch
    from tools.project_runtime import resolve_project_root, save_json_record_once

    if not RUN_LABEL or RUN_LABEL == "default" or Path(RUN_LABEL).name != RUN_LABEL:
        raise ValueError("正式运行需要不含目录分隔符的唯一 RUN_LABEL")
    project_root = resolve_project_root()
    result_dir = project_root / RESULT_RELATIVE_DIR
    result_path = result_dir / f"{WORKLOAD_ID}_{RUN_LABEL}.json"
    if result_path.exists():
        raise FileExistsError("同标签结果已存在：读取已有结果，或更换 RUN_LABEL")
    subprocess.run([sys.executable, str(project_root / "tools/environment_preflight.py"),
                    "--profile", ENVIRONMENT_PROFILE], check=True)
    if not torch.cuda.is_available():
        raise RuntimeError("PERF-ROOFLINE 真实实验需要 CUDA GPU")

    device = torch.device("cuda")
    torch.set_float32_matmul_precision(MATMUL_PRECISION)
    copy_source = torch.randn(COPY_ELEMENTS, device=device, dtype=torch.float32)
    copy_target = torch.empty_like(copy_source)
    peak_left = torch.randn(PEAK_MATMUL_SIZE, PEAK_MATMUL_SIZE, device=device, dtype=torch.float32)
    peak_right = torch.randn_like(peak_left)
    candidate_left = torch.randn(CANDIDATE_MATMUL_SIZE, CANDIDATE_MATMUL_SIZE, device=device, dtype=torch.float32)
    candidate_right = torch.randn_like(candidate_left)

    def elapsed_ms(operation):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end))

    copy_op = lambda: copy_target.copy_(copy_source)
    peak_matmul_op = lambda: torch.mm(peak_left, peak_right)
    candidate_matmul_op = lambda: torch.mm(candidate_left, candidate_right)
    for _ in range(WARMUP_STEPS):
        elapsed_ms(copy_op)
        elapsed_ms(peak_matmul_op)
        elapsed_ms(candidate_matmul_op)
    copy_samples_ms = [elapsed_ms(copy_op) for _ in range(REPEAT_STEPS)]
    peak_matmul_samples_ms = [elapsed_ms(peak_matmul_op) for _ in range(REPEAT_STEPS)]
    candidate_matmul_samples_ms = [elapsed_ms(candidate_matmul_op) for _ in range(REPEAT_STEPS)]
    copy_median_s = statistics.median(copy_samples_ms) / 1000.0
    peak_matmul_median_s = statistics.median(peak_matmul_samples_ms) / 1000.0
    candidate_matmul_median_s = statistics.median(candidate_matmul_samples_ms) / 1000.0
    copy_bytes = 2 * COPY_ELEMENTS * 4
    measured_bandwidth = copy_bytes / copy_median_s
    peak_workload = _gpu_matmul_workload(PEAK_MATMUL_SIZE)
    candidate_workload = _gpu_matmul_workload(CANDIDATE_MATMUL_SIZE)
    empirical_compute_ceiling = peak_workload["flops"] / peak_matmul_median_s
    candidate_measured_compute = candidate_workload["flops"] / candidate_matmul_median_s
    report = _gpu_roofline_report(
        candidate_workload, empirical_compute_ceiling, measured_bandwidth, candidate_measured_compute
    )

    result_dir.mkdir(parents=True, exist_ok=True)
    valid_measurement = all(math.isfinite(value) and value > 0 for value in (measured_bandwidth, empirical_compute_ceiling, candidate_measured_compute))
    record = {
        "schema_version": "performance-roofline-mechanism/v2",
        "profile_id": ENVIRONMENT_PROFILE,
        "semantic_id": SEMANTIC_ID,
        "run_label": RUN_LABEL,
        "framework": {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda},
        "hardware": {"device": torch.cuda.get_device_name(0), "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory},
        "workload": {"id": WORKLOAD_ID, "version": WORKLOAD_VERSION, "copy_elements": COPY_ELEMENTS, "peak_matmul_size": PEAK_MATMUL_SIZE, "candidate_matmul_size": CANDIDATE_MATMUL_SIZE, "matmul_precision": MATMUL_PRECISION, "warmup_steps": WARMUP_STEPS, "repeat_steps": REPEAT_STEPS},
        "empirical_ceilings": {"bandwidth_bytes_per_s": measured_bandwidth, "compute_flops_per_s": empirical_compute_ceiling},
        "metrics": {"copy_samples_ms": copy_samples_ms, "peak_matmul_samples_ms": peak_matmul_samples_ms, "candidate_matmul_samples_ms": candidate_matmul_samples_ms, "candidate_workload": candidate_workload, "roofline": report},
        "failure": None,
        "evidence_level": "local_mechanism_benchmark",
        "decision": "ready_for_bottleneck_hypothesis" if valid_measurement else "reject_invalid_measurement",
    }
    save_json_record_once(result_path, record)
else:
    print("⏭️ RUN_MODE=dry_run；未执行 GPU Roofline 实验。")
```

### 5.3 读取结果

修改已有 `RUN_LABEL` 后直接读取，不需要开启 `real_gpu`。先检查版本与测量条件，再查看三类样本及参考比值；历史字段 `empirical_ceilings` 保留原名，按经验参考解释。

```python
from pathlib import Path
import json
from tools.project_runtime import resolve_project_root

result_path = resolve_project_root() / RESULT_RELATIVE_DIR / f"{WORKLOAD_ID}_{RUN_LABEL}.json"
if not result_path.is_file():
    if RUN_MODE == "real_gpu":
        raise FileNotFoundError(f"未找到结果：{result_path}")
    roofline_record = {"status": "not_run", "result_path": str(result_path)}
    print(roofline_record)
else:
    roofline_record = json.loads(result_path.read_text(encoding="utf-8"))
    print(json.dumps({key: roofline_record.get(key) for key in (
        "schema_version", "run_label", "framework", "hardware", "workload",
        "failure", "evidence_level", "decision", "empirical_ceilings",
    )}, ensure_ascii=False, indent=2))
    print("模型与相对吞吐：", roofline_record.get("metrics", {}).get("roofline"))
    print("测量对象 | 轮次 | CUDA 时间 ms")
    for name in ("copy_samples_ms", "peak_matmul_samples_ms", "candidate_matmul_samples_ms"):
        for index, value in enumerate(roofline_record.get("metrics", {}).get(name, []), 1):
            print(name, index, value)
```

### 5.4 解释证据与决策

当前登记为 [rtx5070ti_roofline_v1](../../benchmarks/results/perf_roofline/perf-empirical-roofline_rtx5070ti_roofline_v1.json)，工作负载版本为 v2。RTX 5070 Ti 实测建立了同一环境下的经验带宽和经验计算参考。按该参考构建的 Roofline 模型将 2048 方阵候选划入计算侧约束区间，实测吞吐接近参考值。候选与参考使用不同矩阵尺寸，现有记录不足以区分形状、采样波动等因素对差异的贡献。

| 观察项 | 读取字段 | 当前结果 | 解释与动作 |
| --- | --- | --- | --- |
| 经验资源参考 | 经验带宽、经验计算吞吐与机器平衡点 | 394.76 GB/s；20.04 TFLOP/s；50.75 FLOPs/Byte | 这些值作为当前环境与测量形状的经验参考，不是严格硬上界 |
| 资源区间 | 算术强度与 `regime` | 341.33 FLOPs/Byte；`compute_bound` | 候选强度明显高于机器平衡点，按模型先检查计算路径，同时用实际流量验证搬运假设 |
| 相对参考吞吐 | 实测吞吐 / 经验计算参考 | 20.12 TFLOP/s；比值 1.004 | 候选比参考高约 0.4%；接近经验参考值，差异来源尚未单独验证 |
| 端到端一致性 | 算子区间与关键路径位置 | 本页只形成算子级资源假设 | 仍需结合执行时间线确认该矩阵乘法是否位于目标工作负载关键路径 |
## 相关阅读

需要查原始模型、分析工具或从实测定位瓶颈时，可继续阅读以下资料。

- [Roofline: An Insightful Visual Performance Model](https://doi.org/10.1145/1498765.1498785)
- [NVIDIA Nsight Compute Roofline Charts](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline-charts)
- [性能分析 · 05 瓶颈归因](../topic_discussion/performance_analysis/05_bottleneck_attribution.md)
