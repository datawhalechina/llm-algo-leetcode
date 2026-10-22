# 73. Training Performance Analysis | 训练性能分析

**难度：** Hard | **环境：** CPU 可完成代码与模板验证；有 GPU 时可补充真实运行数据 | **标签：** `显存优化`, `训练剖析`, `性能分析` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/73_Training_Performance_Analysis.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

前面的训练与显存内容介绍了参数、梯度、激活值和优化器状态如何影响训练成本。本节从一次完整训练 step 出发，学习如何把这些显存对象与耗时、吞吐等信息整理成训练成本记录，并进一步判断主要成本来自哪里。

完成后，你应能读懂一份训练成本报告，区分静态显存账本和运行时观测结果，并提出下一步分析方向。

**关键词：** `training`, `profiling`, `memory`, `step time`

---

## 前置阅读

**导语：** 先回顾训练循环和优化器如何共同完成一次参数更新，再进入本节记录一次 training step 的时间、吞吐和显存成本。
- [13 SFT 训练循环](./13_SFT_Training_Loop.md)
- [Part 00 · 11 Optimizers and Loss | 优化器与损失](../00_Prerequisites/11_PyTorch_Optimizers_and_Loss.md)
- [Part 01 · 13 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)

---
### Step 1：从固定训练任务建立基线证据

训练性能分析从固定模型、数据批次和更新配置开始，同时建立显存对象账本与完整训练步基线。账本解释容量由哪些对象构成，计时与吞吐记录描述运行现象，阶段 trace 用来定位等待发生在哪里；三者共同形成可交给候选实验的基线证据。

| 输入资产 | 分析操作 | 证据输出 | 后续用途 |
|:---|:---|:---|:---|
| 固定训练任务 | 核对模型、数据、精度、批次、序列长度和更新配置 | 可复现的负载身份 | 保证候选与基线可配对 |
| 显存对象 | 统计参数、梯度、优化器状态、激活和临时工作区 | 对象账本与生命周期 | 解释峰值容量来源 |
| 完整训练步 | 测量输入准备、前向、反向和参数更新 | 耗时、吞吐、loss 与峰值显存 | 建立性能基线 |
| 阶段活动 | 记录阶段时间和必要的 trace | 主要等待阶段与候选瓶颈假设 | 指定下一次单变量实验 |

![训练显存对象、影响因素与账本字段](../public/02_PyTorch_Algorithms/73_memory_ledger_map.svg)

### Step 2：定义训练 step 与测量边界

一次训练 step 可以理解为“准备一个 batch，并完成一次参数更新”。为了比较两次训练的成本，必须让两次测量从同一个位置开始，到同一个位置结束。下面的表格说明每个阶段做什么，以及它是否计入一次完整训练 step。

| 训练阶段 | 这一步发生什么 | 是否计入一次完整 step | 主要观察 |
|:---|:---|:---:|:---|
| 输入准备 | 准备 batch，并把数据放到计算设备 | 是 | 输入准备和搬运时间 |
| 梯度清理 | 清除上一轮遗留的梯度 | 是 | 梯度状态维护的成本 |
| 前向与损失计算 | 根据 batch 计算模型输出和损失 | 是 | 激活值规模与前向耗时 |
| 梯度计算 | 根据损失求出各参数的梯度 | 是 | 梯度、重算和反向耗时 |
| 参数更新 | 使用梯度更新参数及训练状态 | 是 | 更新耗时与优化器状态 |


![73 training step 测量边界](../public/02_PyTorch_Algorithms/73_training_step_boundary.svg)
### Step 3：把测量结果转成可验证的瓶颈假设

基线报告不能只给出一个训练步耗时。先检查运行是否成功、重复测量是否稳定，再把时间热点、容量峰值和对象账本对应起来，形成一个可以由单变量候选实验证伪的假设。证据不足时保留待补测状态，不根据单次热点直接指定优化方案。

| 观察结果 | 结合的证据 | 可验证假设 | 交给下一阶段的动作 |
|:---|:---|:---|:---|
| 反向传播占时最高 | 阶段 trace、算子活动和精度路径 | 反向计算或保存张量位于关键路径 | 比较检查点、精度或算子候选 |
| 峰值接近容量上限 | 峰值时刻、对象账本和分配时间线 | 激活或训练状态导致容量压力 | 比较重算、卸载或状态压缩候选 |
| 设备空闲且输入等待明显 | 输入准备时间与主机—设备搬运活动 | 输入流水线限制训练步 | 固定模型后比较输入候选 |
| 重复运行波动过大 | 多次训练步样本和失败记录 | 当前证据不足以确定瓶颈 | 先修复测量条件并补测 |
### Step 4：形成训练基线与瓶颈假设

计时与训练状态账本作为给定工具。请完成 3 个项目级任务：先核验基线报告能否用于归因，再把重复测量、阶段耗时和对象账本合成证据，最后提出可由下一项单变量实验验证的瓶颈假设。CPU 示例验证证据流，不产生 GPU 峰值结论；真实指标由 Step 5 采集。

| 任务与目的 | 已知条件 | 完成标准 | 测试验证 |
|:---|:---|:---|:---|
| TODO 1：核验基线契约 | 固定任务身份、运行状态和重复样本 | 缺失身份、失败运行或样本不足时停止归因 | 完整与缺失报告均有断言 |
| TODO 2：构造基线证据 | 重复样本、阶段耗时、容量和对象账本 | 输出波动、主导阶段与容量压力 | 时间热点和容量热点分别可见 |
| TODO 3：提出瓶颈假设 | 基线证据与判定阈值 | 输出假设、依据和下一项单变量实验 | 不稳定证据优先要求复测 |

```python
import math
import statistics
import time
import torch

```


```python
# 完成训练性能统计和基础显存账本函数
# 目标：完成 measure -> compare -> decide，并统计训练状态 tensor 的实际字节数

def measure_train_step(train_step_fn, warmup=2, iters=8, batch_size=1):
    """测量 CPU training step；不读取或模拟 GPU 峰值显存。

    输入是训练函数和测量配置，返回 step_time_ms、samples_per_s 和 peak_mem_mb。
    peak_mem_mb 固定为 0.0；真实 GPU 峰值由 Step 5 单独测量。
    """
    if warmup < 0 or iters <= 0 or batch_size <= 0:
        raise ValueError('warmup / iters / batch_size 配置不合法')
    for _ in range(warmup):
        train_step_fn()

    start = time.perf_counter()
    for _ in range(iters):
        train_step_fn()
    end = time.perf_counter()
    elapsed = (end - start) / iters

    peak_mem_mb = 0.0

    return {
        'step_time_ms': round(elapsed * 1000, 2),
        'samples_per_s': round(batch_size * iters / (end - start), 3),
        'peak_mem_mb': round(peak_mem_mb, 2),
    }

def summarize_training_result(base_metrics, tuned_metrics):
    """按统一方向计算 baseline / tuned 的时间、显存和吞吐差值。"""
    time_delta = base_metrics['step_time_ms'] - tuned_metrics['step_time_ms']
    mem_delta = base_metrics['peak_mem_mb'] - tuned_metrics['peak_mem_mb']
    throughput_delta = tuned_metrics['samples_per_s'] - base_metrics['samples_per_s']
    return {
        'step_time_delta_ms': round(time_delta, 2),
        'peak_mem_delta_mb': round(mem_delta, 2),
        'time_improved': time_delta > 0,
        'memory_improved': mem_delta > 0,
        'throughput_delta': round(throughput_delta, 2),
        'throughput_improved': throughput_delta > 0,
    }

def recommend_training_decision(summary, min_time_delta_ms=10.0, min_memory_delta_mb=512.0):
    """给定实现：根据速度和显存收益输出训练项目结论。"""
    strong_time_gain = summary['step_time_delta_ms'] >= min_time_delta_ms
    strong_memory_gain = summary['peak_mem_delta_mb'] >= min_memory_delta_mb
    if strong_time_gain and strong_memory_gain:
        return {'decision': 'accept', 'reason': '训练速度和显存收益都达标。'}
    if summary['time_improved'] or summary['memory_improved']:
        return {'decision': 'tune', 'reason': '至少有一项收益，但仍需继续验证。'}
    return {'decision': 'reject', 'reason': '速度和显存都没有形成有效收益。'}

def summarize_training_memory_ledger(model, optimizer):
    """统计参数、梯度和 optimizer state 的实际 tensor 字节数。

    activation、临时 workspace 和 CUDA allocator reserved 不在本函数内计算。
    """
    # 给定实现：统计参数、梯度和 optimizer state 的实际字节数。
    def tensor_bytes(tensor):
        return tensor.numel() * tensor.element_size()

    parameter_bytes = sum(tensor_bytes(parameter) for parameter in model.parameters())
    gradient_bytes = sum(tensor_bytes(parameter.grad) for parameter in model.parameters() if parameter.grad is not None)
    optimizer_state_bytes = sum(tensor_bytes(value) for state in optimizer.state.values() for value in state.values() if torch.is_tensor(value))
    first_parameter = next(model.parameters(), None)
    return {
        'parameter_bytes': parameter_bytes,
        'gradient_bytes': gradient_bytes,
        'optimizer_state_bytes': optimizer_state_bytes,
        'parameter_dtype': str(first_parameter.dtype) if first_parameter is not None else None,
    }

def validate_training_baseline(report, minimum_samples=3):
    """确认训练基线具备归因所需的身份、状态和重复样本。"""
    # ==========================================
    # TODO 1：核验基线契约。
    # 检查 workload_id、status、samples，以及每个样本的阶段耗时和峰值字段。
    # issues = ???
    # 返回 is_valid、issues 和 sample_count。
    # ==========================================
    raise NotImplementedError('请先完成 TODO 代码！')

def build_training_baseline_evidence(report, memory_ledger, capacity_mb):
    """把重复样本、阶段耗时与对象账本合成为基线证据。"""
    # ==========================================
    # TODO 2：构造基线证据。
    # 计算 step time 中位数与变异系数、各阶段中位数、主导阶段和容量占用率。
    # stage_medians = ???
    # dominant_stage = ???
    # 返回稳定性、容量压力和账本摘要。
    # ==========================================
    raise NotImplementedError('请先完成 TODO 代码！')

def classify_training_bottleneck(evidence, capacity_alert_ratio=0.90, hotspot_share=0.40):
    """依据基线证据提出下一项可验证实验。"""
    # ==========================================
    # TODO 3：提出瓶颈假设。
    # 证据不稳定时先复测；容量压力优先进入 76；否则根据主导阶段占比选择下一项实验。
    # 返回 status、hypothesis、evidence 和 next_experiment。
    # ==========================================
    raise NotImplementedError('请先完成 TODO 代码！')

```


```python
# 测试你的实现
def test_training_project_template():
    try:
        torch.manual_seed(42)
        counter = {'n': 0}
        model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 4))
        # AdamW 在第一次 optimizer.step() 后创建状态，便于检查 optimizer state 账本。
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        inputs = torch.randn(2, 8)
        targets = torch.randn(2, 4)

        def train_step():
            counter['n'] += 1
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(inputs), targets)
            loss.backward()
            optimizer.step()
            return loss.detach()

        timing_calls = {'n': 0}
        def count_step():
            timing_calls['n'] += 1
        warmup_result = measure_train_step(count_step, warmup=2, iters=3)
        assert timing_calls['n'] == 5, 'warmup 应执行但不能混入正式迭代次数'
        assert warmup_result['samples_per_s'] > 0.0

        before = [parameter.detach().clone() for parameter in model.parameters()]
        result = measure_train_step(train_step, warmup=0, iters=2)
        assert counter['n'] == 2, "measure_train_step 没有正确执行训练迭代次数！"
        assert {'step_time_ms', 'samples_per_s', 'peak_mem_mb'} <= result.keys(), "训练统计字段不完整！"
        assert result['samples_per_s'] > 0.0, "samples_per_s 应为正数！"
        assert result['step_time_ms'] >= 0.0, "step_time_ms 应为非负数！"
        assert result['peak_mem_mb'] == 0.0, "CPU 测量不能虚构 GPU 峰值显存！"
        assert all(parameter.grad is not None for parameter in model.parameters()), "训练 step 必须完成 backward 并产生梯度！"
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters()), "参数出现 NaN 或 Inf！"
        assert any(not torch.equal(previous, parameter) for previous, parameter in zip(before, model.parameters())), "optimizer.step 必须更新至少一个参数！"
        ledger = summarize_training_memory_ledger(model, optimizer)
        expected_parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
        assert ledger['parameter_bytes'] == expected_parameter_bytes, "参数字节数账本不正确！"
        assert ledger['gradient_bytes'] > 0, "完成 backward 后应能统计梯度字节数！"
        assert ledger['optimizer_state_bytes'] > 0, "完成 AdamW step 后应能统计 optimizer state 字节数！"
        assert ledger['parameter_dtype'] == 'torch.float32', "参数 dtype 记录不正确！"
        for invalid in ({'warmup': -1, 'iters': 2}, {'warmup': 0, 'iters': 0}):
            try:
                measure_train_step(train_step, **invalid)
            except ValueError:
                pass
            else:
                raise AssertionError('非法 warmup / iters 应明确拒绝！')

        baseline = {'step_time_ms': 120.0, 'peak_mem_mb': 8192.0, 'samples_per_s': 8.0}
        tuned = {'step_time_ms': 98.0, 'peak_mem_mb': 6144.0, 'samples_per_s': 10.0}
        summary = summarize_training_result(baseline, tuned)
        assert summary['step_time_delta_ms'] == 22.0, "step_time_delta_ms 计算不正确！"
        assert summary['peak_mem_delta_mb'] == 2048.0, "peak_mem_delta_mb 计算不正确！"
        assert summary['time_improved'] is True, "time_improved 判断不正确！"
        assert summary['memory_improved'] is True, "memory_improved 判断不正确！"
        assert summary['throughput_delta'] == 2.0, "throughput_delta 计算不正确！"
        assert summary['throughput_improved'] is True, "throughput_improved 判断不正确！"
        decision = recommend_training_decision(summary, min_time_delta_ms=10.0, min_memory_delta_mb=512.0)
        assert decision['decision'] == 'accept', "速度和显存收益都达标时应建议 accept！"

        weak_summary = {'step_time_delta_ms': 6.0, 'peak_mem_delta_mb': 256.0, 'time_improved': True, 'memory_improved': True}
        assert recommend_training_decision(weak_summary, min_time_delta_ms=10.0, min_memory_delta_mb=512.0)['decision'] == 'tune', "收益不够稳时应建议 tune！"

        bad_summary = {'step_time_delta_ms': -4.0, 'peak_mem_delta_mb': 0.0, 'time_improved': False, 'memory_improved': False}
        assert recommend_training_decision(bad_summary, min_time_delta_ms=10.0, min_memory_delta_mb=512.0)['decision'] == 'reject', "没有形成有效收益时应建议 reject！"

        baseline_report = {
            'workload_id': 'train-b2-s1024', 'status': 'ok',
            'samples': [
                {'step_time_ms': 100.0, 'samples_per_s': 20.0, 'peak_memory_mb': 9200.0, 'stages_ms': {'input': 10.0, 'forward': 25.0, 'backward': 50.0, 'update': 15.0}},
                {'step_time_ms': 102.0, 'samples_per_s': 19.6, 'peak_memory_mb': 9250.0, 'stages_ms': {'input': 11.0, 'forward': 25.0, 'backward': 51.0, 'update': 15.0}},
                {'step_time_ms': 98.0, 'samples_per_s': 20.4, 'peak_memory_mb': 9180.0, 'stages_ms': {'input': 9.0, 'forward': 24.0, 'backward': 50.0, 'update': 15.0}},
            ],
        }
        contract = validate_training_baseline(baseline_report)
        assert contract['is_valid'] is True
        assert validate_training_baseline({**baseline_report, 'workload_id': ''})['is_valid'] is False
        evidence = build_training_baseline_evidence(baseline_report, ledger, capacity_mb=10000.0)
        assert evidence['stable'] is True and evidence['dominant_stage'] == 'backward'
        assert evidence['capacity_ratio'] > 0.90
        hypothesis = classify_training_bottleneck(evidence)
        assert hypothesis['hypothesis'] == 'training_capacity_pressure'
        assert hypothesis['next_experiment'] == '76_activation_checkpoint_offload'
        noisy = {**evidence, 'stable': False}
        assert classify_training_bottleneck(noisy)['status'] == 'rerun_required'
        print("✅ 训练性能分析项目模板代码通过基础校验。")

    except NotImplementedError:
        print("请先完成 TODO 代码！")
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError, RuntimeError) as e:
        if isinstance(e, AttributeError):
            print("代码未完成，无法找到必要的属性")
        elif isinstance(e, NameError):
            print("代码可能未完成，导致了变量未定义")
        elif isinstance(e, TypeError):
            print("代码可能未完成，导致了操作错误")
        elif isinstance(e, ValueError):
            print("代码可能未完成，导致了数值错误")
        elif isinstance(e, AssertionError):
            print(f"❌ 测试失败: {e}")
        elif isinstance(e, RuntimeError):
            print("代码可能未完成，导致了运行时错误")
        else:
            print("代码可能未完成，导致了断言失败")
        raise NotImplementedError("请先完成 TODO 代码！") from e
    except Exception as e:
        print(f"❌ 发生未知异常: {e}")
        raise


test_training_project_template()

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
import time
import torch

def measure_train_step(train_step_fn, warmup=2, iters=8, batch_size=1):
    """测量 CPU training step 的平均耗时；不采集或模拟 GPU 显存。

    先执行 warmup，再用正式迭代计算平均 step time 和 samples/s；peak_mem_mb
    固定为 0.0，真实 GPU 显存由后续 GPU 实验单独测量。
    """
    if warmup < 0 or iters <= 0 or batch_size <= 0:
        raise ValueError('warmup / iters / batch_size 配置不合法')
    for _ in range(warmup):
        train_step_fn()

    # TODO 1 提示：只包住正式迭代；start / end 是总耗时边界，elapsed 是平均 step time。
    start = time.perf_counter()  # TODO 1 对应挖空：开始正式计时
    for _ in range(iters):
        train_step_fn()
    end = time.perf_counter()  # TODO 1 对应挖空：结束正式计时
    elapsed = (end - start) / iters  # TODO 1 对应挖空：计算平均 step time

    peak_mem_mb = 0.0

    return {
        'step_time_ms': round(elapsed * 1000, 2),
        'samples_per_s': round(batch_size * iters / (end - start), 3),
        'peak_mem_mb': round(peak_mem_mb, 2),
    }

def summarize_training_result(base_metrics, tuned_metrics):
    """按统一方向计算 baseline / tuned 的时间、显存和吞吐差值。

    正值表示 tuned 相对 baseline 有改善：耗时和显存取 baseline - tuned，
    吞吐取 tuned - baseline。
    """
    # TODO 2 提示：time_delta / mem_delta 用 baseline - tuned；throughput_delta 反向相减。
    time_delta = base_metrics['step_time_ms'] - tuned_metrics['step_time_ms']  # TODO 2 对应挖空：耗时差值
    mem_delta = base_metrics['peak_mem_mb'] - tuned_metrics['peak_mem_mb']  # TODO 2 对应挖空：显存差值
    throughput_delta = tuned_metrics['samples_per_s'] - base_metrics['samples_per_s']  # TODO 2 对应挖空：吞吐差值
    return {
        'step_time_delta_ms': round(time_delta, 2),
        'peak_mem_delta_mb': round(mem_delta, 2),
        'time_improved': time_delta > 0,
        'memory_improved': mem_delta > 0,
        'throughput_delta': round(throughput_delta, 2),
        'throughput_improved': throughput_delta > 0,
    }

# 给定实现：输出训练项目结论
def recommend_training_decision(summary, min_time_delta_ms=10.0, min_memory_delta_mb=512.0):
    """根据配置阈值输出轻量的 accept / tune / reject 建议。"""
    strong_time_gain = summary['step_time_delta_ms'] >= min_time_delta_ms
    strong_memory_gain = summary['peak_mem_delta_mb'] >= min_memory_delta_mb
    if strong_time_gain and strong_memory_gain:
        decision = 'accept'
        reason = '训练速度和显存收益都达标，值得继续保留当前优化。'
    elif summary['time_improved'] or summary['memory_improved']:
        decision = 'tune'
        reason = '至少有一项收益成立，但还没形成稳定项目结论，先继续微调。'
    else:
        decision = 'reject'
        reason = '速度和显存都没有形成有效收益，当前改动不值得保留。'
    return {'decision': decision, 'reason': reason}

# 给定实现：统计训练状态账本；optimizer.step() 后 optimizer state 才会出现
def summarize_training_memory_ledger(model, optimizer):
    """返回参数、梯度和 optimizer state 的实际字节数。

    activation、临时 workspace 和 CUDA allocator reserved 不在本函数内计算。
    """
    def tensor_bytes(tensor):
        return tensor.numel() * tensor.element_size()

    parameter_bytes = sum(tensor_bytes(parameter) for parameter in model.parameters())
    gradient_bytes = sum(
        tensor_bytes(parameter.grad)
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    optimizer_state_bytes = sum(
        tensor_bytes(value)
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )
    first_parameter = next(model.parameters(), None)
    return {
        'parameter_bytes': parameter_bytes,
        'gradient_bytes': gradient_bytes,
        'optimizer_state_bytes': optimizer_state_bytes,
        'parameter_dtype': str(first_parameter.dtype) if first_parameter is not None else None,
    }

def validate_training_baseline(report, minimum_samples=3):
    issues = []
    if not isinstance(report.get('workload_id'), str) or not report.get('workload_id'):
        issues.append('workload_id_missing')
    if report.get('status') != 'ok':
        issues.append('baseline_failed')
    samples = report.get('samples')
    if not isinstance(samples, list) or len(samples) < minimum_samples:
        issues.append('insufficient_samples')
        samples = samples if isinstance(samples, list) else []
    required = ('step_time_ms', 'samples_per_s', 'peak_memory_mb', 'stages_ms')
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or any(key not in sample for key in required):
            issues.append(f'sample_{index}:incomplete')
    return {'is_valid': not issues, 'issues': issues, 'sample_count': len(samples)}

def build_training_baseline_evidence(report, memory_ledger, capacity_mb):
    samples = report['samples']
    step_times = [sample['step_time_ms'] for sample in samples]
    step_median = statistics.median(step_times)
    step_cv = statistics.pstdev(step_times) / statistics.fmean(step_times)
    stage_names = samples[0]['stages_ms'].keys()
    stage_medians = {name: statistics.median(sample['stages_ms'][name] for sample in samples) for name in stage_names}
    dominant_stage = max(stage_medians, key=stage_medians.get)
    peak_memory_mb = statistics.median(sample['peak_memory_mb'] for sample in samples)
    return {
        'workload_id': report['workload_id'], 'step_time_median_ms': step_median,
        'step_time_cv': step_cv, 'stable': step_cv <= 0.10,
        'stage_medians_ms': stage_medians, 'dominant_stage': dominant_stage,
        'dominant_stage_share': stage_medians[dominant_stage] / sum(stage_medians.values()),
        'peak_memory_mb': peak_memory_mb, 'capacity_ratio': peak_memory_mb / capacity_mb,
        'memory_ledger': memory_ledger,
    }

def classify_training_bottleneck(evidence, capacity_alert_ratio=0.90, hotspot_share=0.40):
    if not evidence.get('stable', False):
        return {'status': 'rerun_required', 'hypothesis': 'measurement_unstable', 'evidence': ['step_time_cv'], 'next_experiment': 'repeat_training_baseline'}
    if evidence['capacity_ratio'] >= capacity_alert_ratio:
        return {'status': 'ready', 'hypothesis': 'training_capacity_pressure', 'evidence': ['peak_memory_mb', 'capacity_ratio', 'memory_ledger'], 'next_experiment': '76_activation_checkpoint_offload'}
    if evidence['dominant_stage_share'] >= hotspot_share:
        stage = evidence['dominant_stage']
        return {'status': 'ready', 'hypothesis': f'{stage}_critical_path', 'evidence': ['stage_medians_ms', 'dominant_stage_share'], 'next_experiment': f'test_{stage}_candidate'}
    return {'status': 'needs_more_evidence', 'hypothesis': 'no_dominant_bottleneck', 'evidence': ['stage_medians_ms'], 'next_experiment': 'collect_trace'}


```

### 解析

基线是否稳定、时间热点和容量压力是否有对应证据，决定瓶颈假设能否进入下一项单变量实验。

**TODO 1：核验基线契约**
固定工作负载身份、成功状态和重复样本数量；身份缺失、运行失败或样本不足时停止归因。测试用完整报告与身份缺失报告验证准入条件。

**TODO 2：构造基线证据**
用重复样本的中位数和变异系数判断计时稳定性，以阶段耗时定位时间热点，并结合峰值容量和对象账本判断容量压力。测试分别覆盖时间热点与容量预警。

**TODO 3：提出瓶颈假设**
证据不稳定时先补测；容量接近上限时比较检查点或卸载策略；否则围绕主导阶段设计单变量实验。测试验证容量压力与测量不稳定时会给出不同的下一步动作。

### Step 5（项目实验）：测量训练基线

在相同模型、输入和训练步骤下，各重复三次 FP32 与 BF16，比较单步耗时、吞吐和峰值显存。运行顺序是 `prepare → measure → close`。

| 核对内容 | CPU 练习 | GPU 测量与结果 |
|:---|:---|:---|
| 报告可用性 | 检查任务身份、重复样本和失败输入 | 5.5 检查模型与工作负载、重复次数和有限数值 |
| 训练与资源 | 小网络完成反向传播和参数更新，统计训练状态字节 | 固定模型测量 FP32 与自动混合精度的整步耗时、吞吐和峰值显存 |
| 后续实验 | 用给定阶段耗时提出可验证假设 | 形成 76 的比较基线；实际阶段耗时由 74 的剖析实验补充 |

#### 5.1 本次运行配置

运行环境：`PERF-CUDA-MECHANISM`，单卡显存至少 8 GiB；环境创建与验证见[性能分析 Quickstart](../topic_discussion/performance_analysis/quickstart.md#perf-cuda-mechanism-quickstart)。

只需填写本次标签和运行阶段。首次打开保持 `prepare`；新测量使用同一唯一标签依次运行 `measure`、`close`。


```python
# 修改这两个值后，按 5.1 到 5.6 顺序运行。
RUN_LABEL = 'replace_with_unique_label'  # 正式测量时改为唯一标签，measure 和 close 使用同一值。
RUN_STAGE = 'prepare'  # 可选 prepare / measure / close；prepare 只读取和检查。

```

#### 5.2 固定实验配置与环境预检

检查项目路径、PyTorch、CUDA 和模型文件，并加载固定训练配置。`prepare` 不加载模型。


```python
"""检查运行环境并加载固定实验配置。"""
from pathlib import Path
import os
import sys

PROJECT_ID = '73'
SEMANTIC_ID = 'PERF-TRAINING-BASELINE'
SCHEMA_VERSION = 'training-performance-project/v1'
MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
MODEL_SOURCE = 'local'
MODEL_CACHE_DIR = 'model_cache'
WORKLOAD = 'pressure'
WORKLOADS = {'pressure': {'batch_size': 1, 'seq_len': 768, 'warmup': 2, 'iters': 5}}
REPEATS = 3
LEARNING_RATE = 1e-5
SEED = 42
REAL_RUN_MODE = 'paired'
APPROVED_RESULT_RELATIVE_PATH = Path('benchmarks/results/73_real_gpu_training_rtx5070ti_training_baseline_v1.json')
APPROVED_DECISION_RELATIVE_PATH = Path('benchmarks/results/73_training_baseline_decision_rtx5070ti_training_baseline_v1.json')

if RUN_STAGE not in {'prepare', 'measure', 'close'}:
    raise ValueError("RUN_STAGE 只能是 'prepare'、'measure' 或 'close'。")
if RUN_STAGE != 'prepare' and RUN_LABEL == 'replace_with_unique_label':
    raise ValueError('执行测量或生成决策前，请设置唯一 RUN_LABEL。')

start_dir = Path(os.environ.get('LLM_ALGO_PROJECT_ROOT', Path.cwd())).expanduser().resolve()
PROJECT_ROOT = next(
    (candidate for candidate in (start_dir, *start_dir.parents)
     if (candidate / 'tools/project_runtime.py').is_file()),
    None,
)
if PROJECT_ROOT is None and (Path('/content/llm-algo-leetcode') / 'tools/project_runtime.py').is_file():
    PROJECT_ROOT = Path('/content/llm-algo-leetcode')
if PROJECT_ROOT is None:
    raise RuntimeError('找不到项目根目录，请设置 LLM_ALGO_PROJECT_ROOT。')
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
except ImportError as exc:
    raise RuntimeError('当前环境缺少 PyTorch，请先按性能实验 Quickstart 准备环境。') from exc

print({'project_root': str(PROJECT_ROOT), 'torch': torch.__version__,
       'cuda_build': torch.version.cuda, 'cuda_available': torch.cuda.is_available()})
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print({'gpu': props.name, 'memory_gib': round(props.total_memory / 2**30, 2)})

```

#### 5.3 对照合同与结果路径

确认 FP32 与 BF16 使用相同模型、输入、训练步骤和重复次数，只改变计算精度。


```python
"""派生本次运行的工作负载和结果路径。"""
workload_config = dict(WORKLOADS[WORKLOAD])
BATCH_SIZE = workload_config['batch_size']
SEQ_LEN = workload_config['seq_len']
WARMUP = workload_config['warmup']
ITERS = workload_config['iters']
OUTPUT_RELATIVE_PATH = Path(f'benchmarks/results/{PROJECT_ID}_real_gpu_training_{RUN_LABEL}.json')
DECISION_OUTPUT_RELATIVE_PATH = Path(f'benchmarks/results/{PROJECT_ID}_training_baseline_decision_{RUN_LABEL}.json')
READ_RESULT_RELATIVE_PATH = APPROVED_RESULT_RELATIVE_PATH if RUN_STAGE == 'prepare' else OUTPUT_RELATIVE_PATH
EXPERIMENT_IDENTITY = {
    'model_id': MODEL_ID, 'workload': WORKLOAD, 'batch_size': BATCH_SIZE,
    'seq_len': SEQ_LEN, 'repeats': REPEATS, 'seed': SEED,
    'run_label': RUN_LABEL, 'run_stage': RUN_STAGE,
    'read_result': str(READ_RESULT_RELATIVE_PATH),
}
print(EXPERIMENT_IDENTITY)

```

#### 5.4 采集训练测量

`measure` 各运行三次 FP32 与 BF16 并保存原始结果；其他阶段跳过 GPU。


```python
"""按固定 workload 运行 FP32 / AMP 对照，汇总指标并保存 JSON 报告。"""
import json
import os
import sys
import time
from pathlib import Path

# 先把 Notebook 所在仓库加入 sys.path，再导入项目工具；适配本地、Colab 和 ModelScope。
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
            raise RuntimeError('找不到项目根目录；请设置 LLM_ALGO_PROJECT_ROOT。')
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from tools.project_runtime import ensure_output_path, resolve_project_root, environment_preflight, runtime_snapshot, save_json_record_once, standard_experiment_config, standard_training_metrics, validate_training_config
from tools.training_memory_runtime import measure_training_run

PROJECT_ROOT = resolve_project_root(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_PATH = PROJECT_ROOT / OUTPUT_RELATIVE_PATH
print(f'项目根目录: {PROJECT_ROOT}')
print(f'结果保存路径: {OUTPUT_PATH}')

def summarize_training_result(base_metrics, tuned_metrics):
    time_delta = base_metrics['step_time_ms'] - tuned_metrics['step_time_ms']
    mem_delta = base_metrics['peak_mem_mb'] - tuned_metrics['peak_mem_mb']
    throughput_delta = tuned_metrics['samples_per_s'] - base_metrics['samples_per_s']
    return {
        'step_time_delta_ms': round(time_delta, 2),
        'peak_mem_delta_mb': round(mem_delta, 2),
        'time_improved': time_delta > 0,
        'memory_improved': mem_delta > 0,
        'throughput_delta': round(throughput_delta, 2),
        'throughput_improved': throughput_delta > 0,
    }


if 'recommend_training_decision' not in globals():
    def recommend_training_decision(summary, min_time_delta_ms=10.0, min_memory_delta_mb=512.0):
        time_gain = summary['step_time_delta_ms'] >= min_time_delta_ms
        memory_gain = summary['peak_mem_delta_mb'] >= min_memory_delta_mb
        if time_gain and memory_gain:
            return {'decision': 'accept', 'reason': '训练速度和显存收益都达到门槛。'}
        if summary['time_improved'] or summary['memory_improved']:
            return {'decision': 'tune', 'reason': '至少有一项收益，但仍需继续验证。'}
        return {'decision': 'reject', 'reason': '速度和显存都没有形成有效收益。'}

if RUN_STAGE == 'measure':
    OUTPUT_PATH = ensure_output_path(PROJECT_ROOT, OUTPUT_RELATIVE_PATH)
    if OUTPUT_PATH.exists():
        raise FileExistsError(f'结果已存在，请更换 RUN_LABEL：{OUTPUT_PATH}')
    # 阶段 1：检查 workload、依赖和 CUDA；检查失败时不加载模型。
    import torch
    from tools.model_runtime import resolve_model
    from transformers import AutoConfig, AutoModelForCausalLM

    if WORKLOAD not in WORKLOADS:
        raise ValueError(f'未知 workload: {WORKLOAD}，可选值：{sorted(WORKLOADS)}')
    workload_config = WORKLOADS[WORKLOAD]
    BATCH_SIZE = workload_config['batch_size']
    SEQ_LEN = workload_config['seq_len']
    WARMUP = workload_config['warmup']
    ITERS = workload_config['iters']
    if REPEATS < 1:
        raise ValueError('REPEATS 必须至少为 1。')
    validate_training_config({'batch_size': BATCH_SIZE, 'seq_len': SEQ_LEN, 'warmup': WARMUP, 'iters': ITERS, 'seed': SEED, 'learning_rate': LEARNING_RATE})
    preflight = environment_preflight(torch, required_packages=('transformers',), require_gpu=True, output_path=OUTPUT_PATH)
    print({'environment_preflight': preflight})
    if not preflight['ready']:
        raise RuntimeError('环境预检未通过，请先按 next_actions 修复；没有加载模型。')
    print({'runtime': runtime_snapshot(torch)})
    if not torch.cuda.is_available():
        raise RuntimeError('measure 阶段需要 CUDA，请先完成 GPU 环境预检。')

    # 阶段 2：准备固定输入、选择可验证的 AMP dtype，并记录运行环境。
    device = torch.device('cuda')
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model_path = resolve_model(MODEL_ID, source=MODEL_SOURCE, cache_dir=MODEL_CACHE_DIR)
    print(f'模型路径: {model_path}')
    model_config = AutoConfig.from_pretrained(model_path)
    input_generator = torch.Generator(device='cpu').manual_seed(SEED)
    shared_input_ids_cpu = torch.randint(
        0, model_config.vocab_size, (BATCH_SIZE, SEQ_LEN),
        generator=input_generator,
    )
    capability = torch.cuda.get_device_capability()
    native_bf16 = capability[0] >= 8 and torch.cuda.is_bf16_supported(including_emulation=False)
    amp_dtype = torch.bfloat16 if native_bf16 else torch.float16
    common = {
        'model_id': MODEL_ID, 'batch_size': BATCH_SIZE, 'seq_len': SEQ_LEN,
        'dtype': 'float32', 'optimizer': 'AdamW', 'workload': WORKLOAD,
        'warmup': WARMUP, 'iters': ITERS, 'amp_dtype': str(amp_dtype),
        'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
        'device': torch.cuda.get_device_name(0),
        'compute_capability': list(capability), 'native_bf16': native_bf16,
    }

    def run_train_mode(use_amp, repeat_index=0):
        """在一次独立模型实例上执行 warmup 和正式训练 step 测量。"""
        torch.manual_seed(SEED + repeat_index)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
        model.config.use_cache = False
        model.to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        input_ids = shared_input_ids_cpu.to(device)
        labels = input_ids.clone()

        def train_step():
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                loss = model(input_ids=input_ids, labels=labels).loss
            loss.backward()
            optimizer.step()
            return float(loss.detach().item())

        result = measure_training_run(
            train_step, torch_module=torch, batch_size=BATCH_SIZE,
            warmup=WARMUP, iters=ITERS,
        )
        del optimizer, model, input_ids, labels
        torch.cuda.empty_cache()
        return result

    def aggregate_runs(runs):
        """计算多次 GPU 运行的均值，并保留单次结果供复核。"""
        return {
            key: round(sum(item[key] for item in runs) / len(runs), 3)
            for key in ('step_time_ms', 'samples_per_s', 'loss', 'peak_mem_mb', 'peak_reserved_mb')
        }

    # 阶段 3：分别运行 baseline 和 AMP candidate，每次都重新初始化模型。
    if REAL_RUN_MODE == 'bf16_probe':
        tuned_runs = [run_train_mode(use_amp=True, repeat_index=i) for i in range(REPEATS)]
        tuned = aggregate_runs(tuned_runs)
        result = {
            'task': 'task3_training_memory_optimization',
            'environment_preflight': preflight,
            'stage': 'bf16_capacity_probe',
            'config': {**common, 'mode': REAL_RUN_MODE, 'seed': SEED, 'repeats': REPEATS},
            'candidate': {**tuned, 'runs': tuned_runs},
            'measurement_status': 'capacity_probe_only',
            'evidence_level': 'fixed_workload_capacity_probe',
        }
    else:
        baseline_runs = [run_train_mode(use_amp=False, repeat_index=i) for i in range(REPEATS)]
        tuned_runs = [run_train_mode(use_amp=True, repeat_index=i) for i in range(REPEATS)]
        baseline = aggregate_runs(baseline_runs)
        tuned = aggregate_runs(tuned_runs)
        summary = summarize_training_result(baseline, tuned)
        time_delta = baseline['step_time_ms'] - tuned['step_time_ms']
        memory_delta = baseline['peak_mem_mb'] - tuned['peak_mem_mb']
        summary.update({
            'time_improvement_pct': round(time_delta / baseline['step_time_ms'] * 100, 2),
            'memory_improvement_pct': round(memory_delta / baseline['peak_mem_mb'] * 100, 2),
            'meaningful_memory_improved': memory_delta >= 512.0,
        })
        result = {
            'task': 'task3_training_memory_optimization',
            'environment_preflight': preflight,
            'stage': 'measurement_baseline',
            'next_stage': '76_activation_checkpoint_offload_benchmark',
            'config': {**common, 'mode': REAL_RUN_MODE, 'seed': SEED, 'repeats': REPEATS},
            'baseline': {**baseline, 'runs': baseline_runs},
            'tuned': {**tuned, 'runs': tuned_runs},
            'summary': summary,
            'loss_delta_tuned_minus_baseline': round(tuned['loss'] - baseline['loss'], 6),
            'evidence_level': 'fixed_workload_performance_smoke',
        }
    # 阶段 4：补充统一报告字段，并将单次结果和均值写入 JSON。
    result.update({'schema_version': SCHEMA_VERSION, 'semantic_id': SEMANTIC_ID, 'run_mode': 'real_gpu', 'run_label': RUN_LABEL})
    result['experiment'] = standard_experiment_config(result['config'])
    if 'baseline' in result:
        result['standard_metrics'] = {name: standard_training_metrics(result[name]) for name in ('baseline', 'tuned')}
    elif 'candidate' in result:
        result['standard_metrics'] = {'candidate': standard_training_metrics(result['candidate'])}
    output_path = Path(OUTPUT_PATH)
    save_json_record_once(output_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
else:
    print(f'跳过训练测量：当前 RUN_STAGE={RUN_STAGE!r}。')

```

#### 5.5 核对测量结果

读取保存的结果，核对实验配置、重复次数和两组指标。本单元不重新训练。


```python
import hashlib
result_path = PROJECT_ROOT / READ_RESULT_RELATIVE_PATH
if not result_path.is_file():
    raise FileNotFoundError(f'找不到训练 baseline：{result_path}')
report = json.loads(result_path.read_text(encoding='utf-8'))
is_paired = report.get('config', {}).get('mode') == 'paired'
evidence_review = {
    'path': str(result_path.relative_to(PROJECT_ROOT)),
    'sha256': hashlib.sha256(result_path.read_bytes()).hexdigest(),
    'schema_version': report.get('schema_version'), 'semantic_id': report.get('semantic_id'),
    'run_label': report.get('run_label'), 'config': report.get('config'),
    'baseline': report.get('baseline'), 'candidate': report.get('tuned', report.get('candidate')),
    'summary': report.get('summary'), 'evidence_level': report.get('evidence_level'),
}
def validate_measured_training_report(report):
    """核对整步测量的身份、样本与数值，不推断阶段热点或质量等价。"""
    import math
    issues = []
    config = report.get('config') or {}
    repeats = config.get('repeats')
    if type(repeats) is not int or repeats < 3:
        issues.append('repeats_must_be_at_least_three')
    expected = {
        'model_id': MODEL_ID, 'workload': WORKLOAD,
        'batch_size': BATCH_SIZE, 'seq_len': SEQ_LEN,
        'warmup': WARMUP, 'iters': ITERS, 'seed': SEED,
        'optimizer': 'AdamW', 'dtype': 'float32',
    }
    for key, value in expected.items():
        if config.get(key) != value:
            issues.append(f'config_mismatch:{key}')
    mode = config.get('mode')
    if mode not in {'paired', 'bf16_probe'}:
        issues.append('unknown_measurement_mode')
    positive = ('step_time_ms', 'samples_per_s', 'peak_mem_mb', 'peak_reserved_mb')
    for name in (('baseline', 'tuned') if mode == 'paired' else ('candidate',)):
        group = report.get(name)
        if not isinstance(group, dict):
            issues.append(f'{name}:missing')
            continue
        runs = group.get('runs')
        if not isinstance(runs, list) or type(repeats) is not int or len(runs) != repeats:
            issues.append(f'{name}:repeat_count_mismatch')
        for index, row in enumerate([group] + (runs if isinstance(runs, list) else [])):
            if not isinstance(row, dict):
                issues.append(f'{name}:{index}:invalid_row')
                continue
            if row.get('status', 'ok') != 'ok':
                issues.append(f'{name}:{index}:failed')
            for key in (*positive, 'loss'):
                value = row.get(key)
                if (type(value) not in (int, float) or not math.isfinite(value)
                        or (key in positive and value <= 0)):
                    issues.append(f'{name}:{index}:invalid_{key}')
    return issues

measurement_issues = validate_measured_training_report(report)
evidence_review['issues'] = measurement_issues
evidence_review['ready_for_decision'] = (
    evidence_review['schema_version'] == SCHEMA_VERSION
    and evidence_review['semantic_id'] == SEMANTIC_ID
    and not measurement_issues
    and report.get('run_label') == ('rtx5070ti_training_baseline_v1' if RUN_STAGE == 'prepare' else RUN_LABEL)
)
print(evidence_review)

```

#### 5.6 形成训练基线结论

`close` 保存决策记录，并核对标签、schema、结论、来源文件和 SHA256。

| 结果 | 条件 | 下一步 |
| --- | --- | --- |
| `accept` | 单步耗时和显存收益均达到门槛 | 将结果交给 76 |
| `tune` | 只有一项改善 | 保留测量并继续比较显存策略 |
| `reject` | 两项均未改善 | 保留 FP32 基线 |
| `measure` | 缺少完整 FP32/BF16 对照 | 补齐测量 |

**当前登记结果**

| 字段 | 登记内容 |
| --- | --- |
| 项目与标签 | 73 节训练性能基线；`rtx5070ti_training_baseline_v1` |
| 运行条件 | RTX 5070 Ti Laptop GPU；固定完整训练 workload |
| 原始证据 | 正式结果内保存重复测量、运行时和显存记录 |
| 正式证据 | [训练基线](../../benchmarks/results/73_real_gpu_training_rtx5070ti_training_baseline_v1.json) |
| 清单与交接 | 无独立 handoff；74、75、76 节按同一 workload 身份引用 |
| 结论 | `tune`：真实单卡基线已形成，可继续比较显存与 profiling 候选 |

```python
if RUN_STAGE == 'close':
    if not evidence_review['ready_for_decision']:
        raise RuntimeError(f'训练结果核对未通过：{evidence_review}')
    training_decision = (
        recommend_training_decision(report['summary'])
        if report.get('config', {}).get('mode') == 'paired'
        else {'decision': 'measure', 'reason': 'bf16_probe_without_fp32_control'}
    )
    decision_asset = {
        'schema_version': 'training-baseline-decision/v1', 'semantic_id': SEMANTIC_ID,
        'run_label': RUN_LABEL,
        'source_evidence': {'path': evidence_review['path'], 'sha256': evidence_review['sha256']},
        'decision': training_decision,
        'handoff': {'consumer': '76_activation_checkpoint_offload',
                    'workload': report.get('config', {}).get('workload')},
    }
    decision_output_path = PROJECT_ROOT / DECISION_OUTPUT_RELATIVE_PATH
    if decision_output_path.exists():
        existing = json.loads(decision_output_path.read_text(encoding='utf-8'))
        if existing != decision_asset:
            raise RuntimeError('同名决策与当前结果不一致，请更换 RUN_LABEL。')
        print({'status': 'already_complete', 'path': str(decision_output_path)})
    else:
        save_json_record_once(decision_output_path, decision_asset)
        print({'status': 'completed', 'path': str(decision_output_path), 'decision': training_decision})
    registered_decision = json.loads(decision_output_path.read_text(encoding='utf-8'))
    registered_evidence_path = PROJECT_ROOT / registered_decision['source_evidence']['path']
    registration_receipt = {
        'run_label_matches': registered_decision.get('run_label') == RUN_LABEL,
        'schema_matches': registered_decision.get('schema_version') == 'training-baseline-decision/v1',
        'decision_recorded': isinstance(registered_decision.get('decision', {}).get('decision'), str),
        'registered_files_exist': decision_output_path.is_file() and registered_evidence_path.is_file(),
        'source_hash_matches': hashlib.sha256(registered_evidence_path.read_bytes()).hexdigest() == registered_decision['source_evidence']['sha256'],
    }
    if not all(registration_receipt.values()):
        raise RuntimeError(f'训练基线登记核对失败：{registration_receipt}')
    print({'registration_receipt': registration_receipt, 'decision_path': str(decision_output_path.relative_to(PROJECT_ROOT))})
    registered_decision = json.loads(decision_output_path.read_text(encoding='utf-8'))
    registered_evidence_path = PROJECT_ROOT / registered_decision['source_evidence']['path']
    registration_receipt = {
        'run_label_matches': registered_decision.get('run_label') == RUN_LABEL,
        'schema_matches': registered_decision.get('schema_version') == 'training-baseline-decision/v1',
        'decision_recorded': isinstance(registered_decision.get('decision', {}).get('decision'), str),
        'registered_files_exist': decision_output_path.is_file() and registered_evidence_path.is_file(),
        'source_hash_matches': hashlib.sha256(registered_evidence_path.read_bytes()).hexdigest() == registered_decision['source_evidence']['sha256'],
    }
    if not all(registration_receipt.values()):
        raise RuntimeError(f'训练基线登记核对失败：{registration_receipt}')
    print({'registration_receipt': registration_receipt, 'decision_path': str(decision_output_path.relative_to(PROJECT_ROOT))})
else:
    print(f'跳过决策写入：当前 RUN_STAGE={RUN_STAGE!r}。')

```

---
## 相关阅读

- [PyTorch Profiler 官方文档](https://pytorch.org/docs/stable/profiler.html)
- [PyTorch CUDA 内存管理](https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)
- [PyTorch Automatic Mixed Precision](https://pytorch.org/docs/stable/amp.html)
- [76 Activation / Checkpoint / Offload 对比项目](./76_Activation_Checkpoint_Offload_Benchmark.md)
- [74 Profiling 驱动的端到端优化](./74_Profiling_Driven_End_to_End_Optimization.md)
