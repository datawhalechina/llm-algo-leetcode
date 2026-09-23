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
- [Part 02 · 09 SFT 训练循环](./09_SFT_Training_Loop.md)
- [Part 00 · 11 Optimizers and Loss | 优化器与损失](../00_Prerequisites/11_PyTorch_Optimizers_and_Loss.md)
- [Part 01 · 13 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)

---
### Step 1：建立训练显存账本与基线

训练成本首先来自几类显存对象：长期驻留的参数与优化器状态、反向传播需要的激活值，以及算子执行时的临时工作区。本 Step 建立“对象—影响因素—账本字段”的静态视图，为后续测量提供解释基础。

| 显存对象 | 典型内容与生命周期 | 主要影响因素 | 账本中先记录什么 |
|:---|:---|:---|:---|
| 参数与梯度 | 权重长期驻留；梯度在 backward 后产生 | 模型规模、可训练参数量、dtype、训练方式 | 参数字节数、梯度是否已产生 |
| optimizer state | 第一次更新后产生并持续保存的动量、二阶矩等状态 | optimizer、参数量、状态精度 | state 字节数、参数 dtype |
| activation | forward 产生，backward 使用完成后释放或被策略保存 | batch、seq_len、hidden size、checkpoint | 生命周期和是否被保存 |
| 临时张量与 workspace | 算子执行期间的短时分配和 loss 中间量 | 算子实现、kernel、运行时 | 区分长期对象与短时对象 |

![训练显存对象、影响因素与账本字段](../public/02_PyTorch_Algorithms/73_memory_ledger_map.svg)

### Step 2：定义训练 step 与测量口径

一次训练 step 可以理解为“准备一个 batch，并完成一次参数更新”。为了比较两次训练的成本，必须让两次测量从同一个位置开始，到同一个位置结束。下面的表格说明每个阶段做什么，以及它是否计入一次完整训练 step。

| 训练阶段 | 这一步发生什么 | 是否计入一次完整 step | 主要观察 |
|:---|:---|:---:|:---|
| 输入准备 | 准备 batch，并把数据放到计算设备 | 是 | 输入准备和搬运时间 |
| `zero_grad` | 清理上一轮留下的梯度 | 是 | 清理梯度的成本 |
| forward / loss | 根据输入计算输出和 loss | 是 | activation 和前向时间 |
| backward | 根据 loss 计算参数梯度 | 是 | 梯度、重算和反向时间 |
| `optimizer.step` | 用梯度更新参数，并维护优化器状态 | 是 | 参数更新和 optimizer state |


![73 training step 测量边界](../public/02_PyTorch_Algorithms/73_training_step_boundary.svg)
### Step 3：固定实验条件并生成基线报告

要比较两次训练是否真的更快、更省显存，首先要让它们使用相同的模型、数据和训练条件。然后先用小规模运行检查流程，再进行多次正式测量。每次只改变一个因素，这样才能判断结果变化来自哪项改动。

| 实验内容 | 学习者要固定或完成的事项 | 为什么需要记录 |
|:---|:---|:---|
| 模型与数据 | 模型、训练数据、batch、seq_len 和 seed 保持一致 | 保证两次实验可比较 |
| 训练条件 | dtype、optimizer、learning rate 保持一致；实验组只改变目标变量 | 判断收益来自哪项改动 |
| 测量过程 | 先运行 `smoke` 检查流程，再用 `pressure` 正式测量；正式测量包含多次重复 | 排除流程错误并观察波动 |
| 比较指标 | 记录 step time、samples/s、peak allocated、peak reserved、loss 和 OOM 状态 | 同时判断速度、显存和训练结果 |
| 结果报告 | 保存模型、环境、配置、指标和证据等级 | 让别人能够复查这次实验 |
### Step 4（CPU 代码练习）：实现测量、比较与账本统计

这一 Step 开始把前面的测量口径写成代码。你需要完成两个 TODO：第一个函数测量一次训练 step 的平均耗时和吞吐，第二个函数比较 baseline 与 tuned 的结果。其余两个函数已经给出，用来把比较结果转成项目判断，并统计参数、梯度和优化器状态的字节数。完成后先运行 CPU 测试，检查返回字段和判断方向是否正确。
这里的 CPU 代码用于验证测量和比较逻辑；`peak_mem_mb=0.0` 是明确的 CPU 占位字段，不代表真实 GPU 显存。真实 GPU 数据由 Step 5 产生。

| 类型 | 函数 | 你要完成或阅读的内容 | 运行后检查什么 |
|:---|:---|:---|:---|
| TODO 1 | `measure_train_step` | 先 warmup，再用计时器测量正式迭代，计算平均 step time 和吞吐 | `step_time_ms`、`samples_per_s`、`peak_mem_mb=0.0` |
| TODO 2 | `summarize_training_result` | 比较 baseline 与 tuned 的时间、显存和吞吐差值，并统一正负方向 | 三个差值和对应的改进判断 |
| 无需修改 | `recommend_training_decision` | 阅读阈值如何把结果转成 `accept / tune / reject` | `decision`、`reason` |
| 无需修改 | `summarize_training_memory_ledger` | 阅读参数、梯度和 optimizer state 如何换算为字节数 | 三类字节数和参数 dtype |

```python
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
    # ==========================================
    # TODO 1: 记录 CPU 平均 step time 和 samples/s；不要读取或虚构 CUDA 显存。
    # 提示：先 warmup，再测正式迭代；warmup 不计入 elapsed。
    # ==========================================
    for _ in range(warmup):
        train_step_fn()

    # start = ???
    for _ in range(iters):
        train_step_fn()
    # end = ???
    # elapsed = ???

    peak_mem_mb = 0.0

    return {
        'step_time_ms': round(elapsed * 1000, 2),
        'samples_per_s': round(batch_size * iters / (end - start), 3),
        'peak_mem_mb': round(peak_mem_mb, 2),
    }

def summarize_training_result(base_metrics, tuned_metrics):
    """按统一方向计算 baseline / tuned 的时间、显存和吞吐差值。"""
    # ==========================================
    # TODO 2: 比较 baseline 和 tuned 的指标差值
    # 提示：step time / peak memory 用 baseline - tuned；samples/s 用 tuned - baseline。
    # ==========================================
    # time_delta = ???
    # mem_delta = ???
    # throughput_delta = ???
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


```

### 解析

**1. TODO 1：测量 CPU training step**

- **实现方式**：先执行 warmup，再用 `perf_counter()` 统计正式迭代的总耗时，最后换算平均 step time 和吞吐。
- **关键点**：warmup 不计入正式结果，`step_time_ms` 表示一次正式迭代的平均耗时。
- **技术细节**：`samples_per_s` 使用 `batch_size × iters ÷ 总耗时` 计算；CPU 代码中的 `peak_mem_mb=0.0` 是占位字段，不代表 GPU 峰值。

| 输出字段 | 计算含义 | 阅读方式 |
|:---|:---|:---|
| `step_time_ms` | 正式迭代平均耗时 | 越小表示单步越快 |
| `samples_per_s` | `batch_size × iters ÷ 总耗时` | 越大表示吞吐越高 |
| `peak_mem_mb` | CPU 接口中的固定占位字段 | `0.0` 不代表 GPU 峰值 |

**2. TODO 2：统一 baseline 与 tuned 的差值方向**

- **实现方式**：使用相同的模型、数据、设备和统计口径，再计算时间、显存和吞吐差值。
- **关键点**：把差值统一成“正数代表优化有效”：时间和显存使用 `baseline - tuned`，吞吐使用 `tuned - baseline`。

- **技术细节**：时间差为 `baseline.step_time_ms - tuned.step_time_ms`；显存差为 `baseline.peak_mem_mb - tuned.peak_mem_mb`；吞吐差为 `tuned.samples_per_s - baseline.samples_per_s`。

**3. 给定实现：训练项目决策**

- **实现方式**：`recommend_training_decision` 将速度和显存差值与阈值比较，输出 `accept / tune / reject`。
- **关键点**：两项收益都达到阈值才是 `accept`；只有部分收益时返回 `tune`；没有有效收益时返回 `reject`。
- **项目意义**：把数值比较转换成可以继续执行的训练项目判断。

**4. 给定实现：训练状态账本**
- **实现方式**：`summarize_training_memory_ledger` 统计参数、已产生的梯度和 optimizer state 的实际 tensor bytes。
- **关键点**：activation、临时 workspace 和 allocator reserved 不属于静态账本。
- **项目意义**：帮助学习者把显存对象与训练成本联系起来，并为后续 GPU 测量提供参照。


### Step 5（GPU 项目实验，可选）：采集真实 training baseline

#### 5.1 环境、输入与固定条件

本次实验回答一个具体问题：在同一训练任务上，改变计算 dtype 是否会改变训练成本。使用 `Qwen/Qwen2.5-0.5B-Instruct`、固定输入和 `pressure` workload，对比 FP32 baseline 与 AMP candidate；GPU 支持 BF16 时使用 BF16，否则使用 FP16。
表中只记录本轮实验契约；checkpoint、offload、模型规模、数据集、kernel / backend 和并行策略留到后续对照实验。


| 实验要素 | 本轮设置 | 两组如何保持一致 | 输出 |
|:---|:---|:---|:---|
| 实验对象 | FP32 baseline 与 AMP candidate | 使用同一模型、输入和 training step | 两组可比较结果 |
| 固定条件 | `pressure`：batch=1、seq_len=768、AdamW、learning rate=1e-5、seed=42 | 模型、输入、workload 和训练配置完全一致 | 统一实验配置 |
| 本轮变量 | 计算 dtype：FP32 对比 BF16；不支持 BF16 时使用 FP16 | 只改变 dtype | dtype 对照结果 |
| 测量方式 | 先运行 `smoke`，再运行 `pressure`；warmup 不计入结果，正式运行重复多次 | 两组使用相同测量流程 | step time、吞吐、峰值显存和 loss |
| 结果保存 | 记录硬件、配置、汇总值、波动范围和 evidence level | 不手动修改原始测量值 | JSON 报告，供后续项目读取 |

![GPU training baseline 实验流程](../public/02_PyTorch_Algorithms/73_gpu_baseline_flow.svg)

#### 5.2 环境启动检查（可独立运行）

从文档链接打开 Colab 时，Notebook 文件和项目仓库不一定同时存在。先运行下面的单元，它会准备项目根目录、把项目加入 `sys.path`，并检查当前 Python 是否真的使用 CUDA 版 PyTorch。它不会静默重装 PyTorch；如果检测到 CPU 版，会给出安装命令。

```python
"""检查 GPU 实验所需的项目路径、Python 解释器和 CUDA 运行时。"""
from pathlib import Path
import os
import subprocess
import sys

PROJECT_ROOT = Path('/content/llm-algo-leetcode') if Path('/content').is_dir() else Path.cwd()
if not (PROJECT_ROOT / 'tools/project_runtime.py').is_file():
    if PROJECT_ROOT.exists() and any(PROJECT_ROOT.iterdir()):
        raise RuntimeError(f'项目目录存在但不是完整仓库：{PROJECT_ROOT}')
    subprocess.run([
        'git', 'clone',
        'https://github.com/datawhalechina/llm-algo-leetcode.git',
        str(PROJECT_ROOT),
    ], check=True)
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
print('项目根目录:', PROJECT_ROOT)
print('tools 存在:', (PROJECT_ROOT / 'tools').is_dir())
print('PyTorch:', torch.__version__)
print('PyTorch CUDA:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    print('当前是 CPU 版 PyTorch 或 CUDA 未连接。GPU 实验前请安装 CUDA wheel，并重启 Colab runtime：')
    print('%pip install --force-reinstall --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128')
else:
    print('GPU:', torch.cuda.get_device_name(0))
    print('GPU memory GB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
    capability = torch.cuda.get_device_capability()
    native_bf16 = capability[0] >= 8 and torch.cuda.is_bf16_supported(including_emulation=False)
    print('Compute capability:', f'{capability[0]}.{capability[1]}')
    print('BF16 allocatable:', torch.cuda.is_bf16_supported())
    print('BF16 native acceleration:', native_bf16)

```

#### 5.3 配置实验条件

下面的配置单元只选择模型、workload、数据类型和重复次数，不加载模型、不启动训练。baseline 与 candidate 只改变数据类型，其他条件保持一致。

```python
"""定义 73 的 GPU baseline 配置；本 cell 只改配置，不加载模型或启动训练。"""
RUN_REAL_GPU = False  # CPU-first 默认关闭；在 Colab / 本地 GPU 实测时显式改为 True。
AUTO_INSTALL_REAL_DEPS = True  # 真实 GPU 开启时，只安装当前内核缺失的普通依赖。
AUTO_INSTALL_ALLOW_BREAK_SYSTEM_PACKAGES = True  # 云端 PEP 668 环境允许安装普通依赖；不会重装 PyTorch。
#REAL_RUN_MODE = 'paired'  # paired：FP32/BF16 对比；bf16_probe：只探测 BF16 容量。
REAL_RUN_MODE = 'paired'
MODEL_PROFILES = {
    'qwen25_small': 'Qwen/Qwen2.5-0.5B-Instruct',
    'deepseek_r1_small': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
}
MODEL_PROFILE = 'qwen25_small'  # 先用小模型建立基线。
MODEL_ID = MODEL_PROFILES[MODEL_PROFILE]  # 实际加载的模型 ID。
MODEL_SOURCE = 'auto'  # 模型来源：auto / huggingface / modelscope / local。
MODEL_CACHE_DIR = 'model_cache'  # 模型缓存目录。
WORKLOADS = {
    'smoke': {'batch_size': 1, 'seq_len': 256, 'warmup': 2, 'iters': 5},
    'pressure': {'batch_size': 1, 'seq_len': 768, 'warmup': 2, 'iters': 5},
    'pressure_1024': {'batch_size': 1, 'seq_len': 1024, 'warmup': 2, 'iters': 5},
}
WORKLOAD = 'pressure'  # 主线使用 seq_len=768；pressure_1024 是扩展 workload。
REPEATS = 3  # 正式采集可改为 3；每次重复都会重新初始化模型。
BATCH_SIZE = 1  # 由 WORKLOADS 覆盖；增大它通常会提高吞吐和 activation 压力。
SEQ_LEN = 768  # 由 WORKLOADS 覆盖；它是压力变量，不要脱离 workload 单独修改。
WARMUP = 3  # 由 WORKLOADS 覆盖；用于 kernel / allocator 预热，不计入平均值。
ITERS = 10  # 由 WORKLOADS 覆盖；数值越小越接近 smoke，重复性较弱。
LEARNING_RATE = 1e-5  # baseline 与 tuned 必须一致；本节不据此判断收敛。
SEED = 42  # 固定输入和初始化，降低随机差异；不能消除 GPU 调度噪声。
from pathlib import Path
OUTPUT_RELATIVE_PATH = Path('benchmarks/results/73_real_gpu_training.json')

```

#### 5.4 执行训练并保存 JSON

运行下面的单元，按已配置的 workload 完成 baseline / candidate 对照，汇总时间、吞吐、显存和 loss，并将原始结果写入 JSON。

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
            colab_root = Path('/content/llm-algo-leetcode')
            if Path('/content').is_dir() and not colab_root.exists():
                subprocess.run(['git', 'clone', 'https://github.com/datawhalechina/llm-algo-leetcode.git', str(colab_root)], check=True)
            if (colab_root / 'tools/project_runtime.py').is_file():
                PROJECT_ROOT = colab_root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from tools.project_runtime import ensure_output_path, resolve_project_root, environment_preflight, runtime_snapshot, standard_experiment_config, standard_training_metrics, validate_training_config
from tools.training_memory_runtime import measure_training_run

PROJECT_ROOT = resolve_project_root(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_PATH = ensure_output_path(PROJECT_ROOT, OUTPUT_RELATIVE_PATH)
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

if RUN_REAL_GPU:
    # 阶段 1：检查 workload、依赖和 CUDA；检查失败时不加载模型。
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
        raise RuntimeError('RUN_REAL_GPU=True 但 CUDA 不可用，请先完成 GPU 环境预检。')

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
            'decision': {'decision': 'measure', 'reason': 'BF16 probe does not compare against FP32 baseline.'},
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
        decision = recommend_training_decision(summary)
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
            'decision': decision,
        }
    # 阶段 4：补充统一报告字段，并将单次结果和均值写入 JSON。
    result['experiment'] = standard_experiment_config(result['config'])
    if 'baseline' in result:
        result['standard_metrics'] = {name: standard_training_metrics(result[name]) for name in ('baseline', 'tuned')}
    elif 'candidate' in result:
        result['standard_metrics'] = {'candidate': standard_training_metrics(result['candidate'])}
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
else:
    print('跳过真实 GPU 训练：保持 CPU-first 模式。')

```

#### 5.5 实测记录与结果表

本节保留一组已保存的历史 GPU 记录作为示例；读者应在自己的环境中重新运行。先核对实测环境与统一口径，再比较时间、吞吐、峰值显存和 loss。逐次运行数据保存在 JSON 中，页面展示汇总值和波动范围。

**实测环境与统一口径**

| 项目 | 实际配置 |
|:---|:---|
| 系统 / GPU | Linux `6.8.0-138-generic`、RTX 5070 Ti Laptop GPU / 12227 MiB |
| 驱动 | `570.211.01` |
| PyTorch / CUDA | `2.11.0+cu128` / `12.8` |
| 模型 | `Qwen/Qwen2.5-0.5B-Instruct` |
| workload | `pressure`：batch=1、seq_len=768 |
| 训练设置 | AdamW、learning rate=1e-5、warmup=2、iters=5、repeats=3、seed=42 |
| 对照变量 | FP32 baseline 与 AMP BF16 candidate；不支持 BF16 时记录实际使用的 FP16 |

**主线结果与复测记录**

读取结果后，先比较两组在固定 workload 下的时间、吞吐、峰值显存和 loss；下表保留本地历史结果，随后提供学习者复测位置。

| 指标 | FP32 baseline | AMP BF16 candidate | 变化 / 波动范围 |
|:---|---:|---:|:---|
| step time | 482.753 ms | 237.998 ms | 提升约 50.7%；范围 4.409 / 6.393 ms（历史结果） |
| throughput | 2.072 samples/s | 4.202 samples/s | 提升约 102.8%；范围 0.019 / 0.113 samples/s（历史结果） |
| peak allocated | 9782.56 MiB | 9477.25 MiB | 下降 305.31 MiB（3.12%）；范围 0.25 / 0 MiB |
| peak reserved | 10766 MiB | 10624 MiB | allocator 预留变化 |
| 最后一步 loss | 11.477 | 11.716 | 差值约 +0.239 |
| 状态 / 证据级别 | ok / repeated benchmark | ok / repeated benchmark | 仅支持固定 workload 下的对比结论 |

下表用于填写自己的 GPU 或其他配置；每行对应一次固定 workload 下的 baseline / candidate 结果。

| 配置 / 策略 | GPU / 显存 | 模型 | dtype | batch | seq_len | repeats | step time (ms) | throughput (samples/s) | peak allocated (MiB) | peak reserved (MiB) | loss | 状态 | evidence level |
|:---|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---|:---|
| baseline | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| candidate | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |

#### 5.6 解释结果与形成决策

本轮历史结果显示：AMP candidate 的 step time 从 482.753 ms 降至 237.998 ms，吞吐从 2.072 samples/s 提升至 4.202 samples/s；peak allocated 仅下降 305.31 MiB，最后一步 loss 增加约 0.239。速度收益明确，但显存收益未达到 512 MiB 阈值，因此当前判断为 `tune`。

这组结果只支持固定模型、`pressure` workload 和当前训练设置下的对比结论。下一步进入 76，比较 checkpoint / offload / hybrid；再由 75 根据 76 的结果进行显存预算决策。

---
## 相关阅读

以下资料按“训练成本测量 → 显存观测 → 后续项目验证”排列，用于把本节的训练成本记录连接到后续策略实验。

- [PyTorch Profiler 官方文档](https://pytorch.org/docs/stable/profiler.html)
- [PyTorch 内存管理与 CUDA 缓存分配器](https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)
- [PyTorch Automatic Mixed Precision 官方文档](https://pytorch.org/docs/stable/amp.html)
- [PyTorch 官方仓库](https://github.com/pytorch/pytorch)
- [76 Activation / Checkpoint / Offload 对比项目](./76_Activation_Checkpoint_Offload_Benchmark.md)
- [75 显存预算压缩项目](./75_Memory_Budget_Compression_Project.md)
- [74 Profiling 驱动的端到端优化](./74_Profiling_Driven_End_to_End_Optimization.md)
