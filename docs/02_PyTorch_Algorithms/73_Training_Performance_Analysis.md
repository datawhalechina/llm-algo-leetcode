# 73. Training Performance Analysis | 训练性能分析

**难度：** Hard | **环境：** CPU 可完成模板验证；GPU 用于正式 baseline | **标签：** `显存优化`, `训练剖析`, `性能分析` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/73_Training_Performance_Analysis.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

前面的训练与显存小节已经说明了 activation、梯度和 optimizer state 为什么会影响训练成本。这一节把这些机制落到一次完整 training step 的测量上：先固定 workload 建立 baseline，再用统一字段记录时间、吞吐、显存和训练状态。

> 运行提示：先查看[使用指南中的项目环境预检与安装说明](../docs/guide.md#项目环境预检与安装)，再打开真实 GPU 开关。CPU-first 路径不要求 GPU；真实 GPU 路径必须先通过预检。

CPU-first 路径用于检查计时、统计字段和报告逻辑；真实 GPU 路径才用于验证 step time、吞吐、peak / reserved memory 和 OOM。GPU smoke 只能证明流程可运行，重复 benchmark 才能比较当前 workload 的平均成本。

| 内容 | 建议比重 | 作用 |
|---|---:|---|
| CPU 模板与正确性测试 | 约 40% | 验证计时、吞吐、账本和异常边界 |
| GPU baseline 与结果解释 | 约 60% | 采集真实训练成本，并为 76 提供可复用基线 |

本节不比较 checkpoint / offload，也不解释具体 kernel 或阶段瓶颈；它的产出是一份可被 76 复用的 baseline 报告。策略比较进入 76，预算判断进入 75，阶段归因进入 74。扩展实验可以改变 batch、seq_len 或 dtype，但必须单独保存并写明条件。
**主责与复用边界：** 本项目主责是训练 baseline 和统一测量口径；76 负责 checkpoint / offload 策略比较，75 负责预算筛选，74 负责 trace 归因。其他路线可以复用本节的计时与环境字段，但不能把 baseline 直接当成优化收益。

**关键词：** `training`, `profiling`, `memory`, `step time`

---

## 前置阅读

**导语：** 先理解反向传播、显存对象和训练侧显存手段，再进入本节建立统一测量口径；本节重点不是重复讲机制，而是为后续 76 的方案 benchmark 提供可靠 baseline。
- [09. SFT Training Loop | SFT 训练循环](./09_SFT_Training_Loop.md)
- [17. Autograd Basics | Autograd 基础](./17_Autograd_Basics.md)
- [19. Activation Checkpointing and Activation Offload | 激活检查点与激活卸载](./19_Activation_Checkpointing_and_Activation_Offload.md)
- [42. Activation Offload | 激活卸载](./42_Activation_Offload.md)
- [P1: 13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)

## 相关阅读

**导语：** 完成本节后，显存路线先进入 76 比较 checkpoint / offload，再进入 75 形成预算决策，最后由 74 做 profiling 驱动的端到端验证；如果关注训练项目交付，可回到 60 核对训练成本。
- [76. Activation Checkpoint Offload Benchmark | Activation / Checkpoint / Offload 对比项目](./76_Activation_Checkpoint_Offload_Benchmark.md)
- [75. Memory Budget Compression Project | 显存预算压缩项目](./75_Memory_Budget_Compression_Project.md)
- [74. Profiling-Driven End-to-End Optimization | profiling 驱动的端到端优化](./74_Profiling_Driven_End_to_End_Optimization.md)
- [60. LoRA Fine-Tuning Project | LoRA 微调项目](./60_LoRA_Fine_Tuning_Project.md)
---
### Step 1：显存账本与可测假设（CPU / GPU 共用）

训练峰值可以先用一个不追求精确的账本表达：`参数 + 梯度 + optimizer state + activation + 临时张量 / workspace`。本节把反向传播和显存生命周期知识转成测量假设，不重新实现这些机制。

| 观察对象 | 变化因素 | 需要验证的假设 |
|:---|:---|:---|
| activation | `batch_size`、`seq_len`、hidden size | 序列或 batch 增大时峰值是否更敏感 |
| 参数 / 梯度 | 模型规模、dtype、训练方式 | workload 改变后是否仍接近固定占用 |
| optimizer state | optimizer、参数量、状态精度 | 是否已经占据主要预算 |
| 临时张量 / workspace | kernel、算子和运行时 | 峰值是否来自短时分配而非长期状态 |

主线固定模型、输入、batch、seq_len、硬件和后端，只改变一个变量；探索性 workload 必须单独保存。73 只记录未优化基线，checkpoint 的重算和 offload 的搬运留给 76 验证。

### Step 2：训练 step 与测量边界（CPU / GPU 共用）

训练性能分析必须先确认 baseline 可复现。一次完整 training step 是 `zero_grad → forward → loss → backward → optimizer.step`；当前模板测量整个生命周期，不单独测量阶段。

| 阶段 | 主要状态 / 开销 | 73 的观测范围 | 深入分析入口 |
|:---|:---|:---|:---|
| forward | activation、临时张量、算子执行 | 包含在总 step time 和峰值中 | 74 profiler |
| backward | saved tensors、梯度、重算 | 包含在总 step time 和峰值中 | 76 / 74 |
| optimizer.step | optimizer state 更新 | 包含在总 step time 中 | 阶段计时 / 74 |
| 数据与同步 | CPU 预处理、拷贝、等待 | 当前不单独测量 | 74 profiler |

因此，73 能回答完整 step 的平均成本，不能单独判断哪个阶段是瓶颈，也不能从 `peak allocated` 拆出各类显存对象。
### Step 3：实验协议与 baseline 报告（CPU 检查 / GPU 采集）

先定义统一的实验协议和报告字段，再分别用 CPU 模板检查逻辑、用 GPU Step 5 采集真实训练成本。

| 顺序 | 操作 | 目的 |
|:---|:---|:---|
| 1 | 运行 `smoke` | 确认模型、依赖和报告流程可运行 |
| 2 | 固定模型、dtype、optimizer、batch、seq_len、输入、seed | 保证 baseline 可复现 |
| 3 | 运行 `pressure`，warm-up 后重复测量 | 获得正式 GPU baseline |
| 4 | 一次只改变一个变量，并使用独立输出文件 | 保持结果可归因 |
| 5 | 对比 step time、samples/s、显存、loss 和 OOM | 判断收益是否伴随代价 |

第一张表规定“怎么做”：先确认流程，再固定口径，最后比较指标；每一步都要把配置或观测结果写入第二张表，而不是只在屏幕上查看。

dtype 改变、activation 生命周期改变和数据/同步改变属于不同假设，不能一次性叠加。完成实验流程后，将结果按下面的类别保存为可复用报告。

| 报告类别 | 必填字段 | 用途 |
|:---|:---|:---|
| workload | model、dtype、batch、seq_len、warmup、iters、seed | 判断结果是否可复现 |
| 环境 | GPU、PyTorch、CUDA、运行后端 | 判断硬件证据边界 |
| 性能 | step time、samples/s | 衡量训练成本 |
| 显存 | peak allocated、peak reserved、OOM | 衡量容量压力 |
| 训练状态 | loss / eval_loss | 防止只追求速度或显存 |
| 证据等级 | CPU、GPU smoke、repeated benchmark | 限定结论强度 |

73 的交付物是可被 76 复用的 baseline 报告；它不替代 75 的预算裁决，也不替代 74 的阶段归因。
### Step 4：CPU 正确性检查（模板验证）

**要求：** 请补全下方训练性能统计函数和静态显存账本函数，完成 warmup、正式迭代、平均 step time、samples/s、设备相关峰值显存，以及参数、梯度和 optimizer state 字节数的统计。先通过 CPU 测试，再将训练测量口径用于 Step 5 的真实 GPU baseline。固定 baseline 配置，一次只改变一个变量；CPU 不读取或虚构 GPU 显存，GPU 计时需要正确处理 CUDA 同步。CPU 结果只验证函数逻辑和账本关系，不代表 GPU 训练速度或显存峰值。


```python
import time
import torch

```


```python
# 完成训练性能统计和静态显存账本函数
# 目标：完成 measure -> compare -> decide，并建立可检查的训练状态账本

def measure_train_step(train_step_fn, warmup=2, iters=8, device='cpu', batch_size=1):
    """测量一次完整训练 step 的平均耗时；CPU 不采集 GPU 显存。"""
    if warmup < 0 or iters <= 0 or batch_size <= 0:
        raise ValueError('warmup / iters / batch_size 配置不合法')
    use_cuda = str(device).startswith('cuda')
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError('device=cuda 但当前 CUDA 不可用')
    # ==========================================
    # TODO 1: 记录平均 step time 和 peak memory
    # 提示：先 warmup，再测正式迭代；device=cuda 时同步后读取 peak allocated，
    #      device=cpu 时不要读取或虚构 CUDA 显存。
    # ==========================================
    for _ in range(warmup):
        train_step_fn()

    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # start = ???
    for _ in range(iters):
        train_step_fn()
    # end = ???
    # elapsed = ???

    if use_cuda:
        torch.cuda.synchronize()

    peak_mem_mb = 0.0
    if use_cuda:
        # peak_mem_mb = ???
        pass

    return {
        'step_time_ms': round(elapsed * 1000, 2),
        'samples_per_s': round(batch_size * iters / (end - start), 3),
        'peak_mem_mb': round(peak_mem_mb, 2),
    }

def summarize_training_result(base_metrics, tuned_metrics):
    """按 baseline - tuned 计算速度和显存差值。"""
    # ==========================================
    # TODO 2: 比较 baseline 和 tuned 的指标差值
    # 提示：delta = baseline - tuned，正数表示 tuned 更省或更快。
    # ==========================================
    # time_delta = ???
    # mem_delta = ???
    return {
        'step_time_delta_ms': round(time_delta, 2),
        'peak_mem_delta_mb': round(mem_delta, 2),
        'time_improved': time_delta > 0,
        'memory_improved': mem_delta > 0,
    }

def recommend_training_decision(summary, min_time_delta_ms=10.0, min_memory_delta_mb=512.0):
    """根据速度和显存收益给出训练项目结论。"""
    # ==========================================
    # TODO 3: 输出训练项目结论
    # 规则：达到配置阈值才算强收益；只有轻微正收益时继续 tune；
    # - 速度和显存收益都达标：accept
    # - 至少一项有正收益但未同时达标：tune
    # - 没有正收益：reject
    # ==========================================
    # strong_time_gain = ???
    # strong_memory_gain = ???
    # if ???:
    #     decision = ???
    #     reason = ???
    # elif ???:
    #     decision = ???
    #     reason = ???
    # else:
    #     decision = ???
    #     reason = ???
    # return {'decision': decision, 'reason': reason}

def summarize_training_memory_ledger(model, optimizer):
    """统计训练后参数、梯度和 optimizer state 的实际字节数。"""
    # ==========================================
    # TODO 4: 统计参数、梯度和 optimizer state 的实际字节数
    # 提示：遍历 tensor 的 numel() * element_size()；optimizer.step() 后再统计 state。
    # ==========================================
    # parameter_bytes = ???
    # gradient_bytes = ???
    # optimizer_state_bytes = ???
    # return {"parameter_bytes": ..., "gradient_bytes": ...,
    #         "optimizer_state_bytes": ..., "parameter_dtype": ...}

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
        if not torch.cuda.is_available():
            try:
                measure_train_step(train_step, warmup=0, iters=1, device='cuda')
            except RuntimeError:
                pass
            else:
                raise AssertionError('CUDA 不可用时，device=cuda 应明确报错！')
        for invalid in ({'warmup': -1, 'iters': 2}, {'warmup': 0, 'iters': 0}):
            try:
                measure_train_step(train_step, **invalid)
            except ValueError:
                pass
            else:
                raise AssertionError('非法 warmup / iters 应明确拒绝！')

        baseline = {'step_time_ms': 120.0, 'peak_mem_mb': 8192.0}
        tuned = {'step_time_ms': 98.0, 'peak_mem_mb': 6144.0}
        summary = summarize_training_result(baseline, tuned)
        assert summary['step_time_delta_ms'] == 22.0, "step_time_delta_ms 计算不正确！"
        assert summary['peak_mem_delta_mb'] == 2048.0, "peak_mem_delta_mb 计算不正确！"
        assert summary['time_improved'] is True, "time_improved 判断不正确！"
        assert summary['memory_improved'] is True, "memory_improved 判断不正确！"
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

# TODO 1: 测量训练 step 的平均耗时和峰值显存
def measure_train_step(train_step_fn, warmup=2, iters=8, device='cpu', batch_size=1):
    """测量一次完整训练 step 的平均耗时；CPU 不采集 GPU 显存。"""
    if warmup < 0 or iters <= 0 or batch_size <= 0:
        raise ValueError('warmup / iters / batch_size 配置不合法')
    use_cuda = str(device).startswith('cuda')
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError('device=cuda 但当前 CUDA 不可用')
    for _ in range(warmup):
        train_step_fn()

    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(iters):
        train_step_fn()
    if use_cuda:
        torch.cuda.synchronize()
    end = time.perf_counter()
    elapsed = (end - start) / iters

    peak_mem_mb = 0.0
    if use_cuda:
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        'step_time_ms': round(elapsed * 1000, 2),
        'samples_per_s': round(batch_size * iters / (end - start), 3),
        'peak_mem_mb': round(peak_mem_mb, 2),
    }

# TODO 2: 汇总 baseline 和 tuned 的差异
def summarize_training_result(base_metrics, tuned_metrics):
    """按 baseline - tuned 计算速度和显存差值。"""
    time_delta = base_metrics['step_time_ms'] - tuned_metrics['step_time_ms']
    mem_delta = base_metrics['peak_mem_mb'] - tuned_metrics['peak_mem_mb']
    return {
        'step_time_delta_ms': round(time_delta, 2),
        'peak_mem_delta_mb': round(mem_delta, 2),
        'time_improved': time_delta > 0,
        'memory_improved': mem_delta > 0,
    }

# TODO 3: 输出训练项目结论
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

# TODO 4: 统计训练状态账本；optimizer.step() 后 optimizer state 才会出现
def summarize_training_memory_ledger(model, optimizer):
    """返回参数、梯度和 optimizer state 的实际字节数，不包含 activation 或 CUDA 峰值。"""
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


counter = {'n': 0}
def train_step():
    counter['n'] += 1
print(measure_train_step(train_step, warmup=0, iters=2))

```

### 解析

**TODO 1：测量完整 training step**
- 先执行 warmup，避免把首次初始化成本混入正式结果。
- GPU 计时前后都要 `synchronize()`，否则只测到 CPU 发起 kernel 的时间。
- 只有 `device='cuda'` 时读取 `max_memory_allocated()`；CPU 模式用 `0.0` 表示本次没有采集 GPU 显存。

**TODO 2：计算 baseline 与 tuned 的差值**
- 使用 `baseline - tuned`，因此 step time 或 peak memory 的正差值表示 tuned 更低。
- 差值只适用于 workload、设备和统计口径一致的两次测量。

**TODO 3：输出轻量决策**
- 同时达到速度和显存阈值时为 `accept`。
- 只有轻微或单项正收益时为 `tune`；没有正收益时为 `reject`。
- 这里的决策只用于练习比较逻辑，不能替代 75 的预算裁决。
## Step 5：真实 GPU baseline（主实验，可选运行）

前面的 Step 1-4 是 CPU-first 的性能分析模板；本 Step 把同一套完整 training step 测量口径接到真实 causal LM 的 forward / backward / optimizer.step。默认关闭，只有在 GPU、Transformers 和模型依赖准备好时才运行。成功后会保存 `benchmarks/results/73_real_gpu_training.json`；只有当 73 与 76 使用相同 workload 配置时，它才能作为 76 的 baseline 输入。

本示例只比较一个变量：FP32 baseline 与 AMP candidate。AMP 只在设备具备原生 BF16 Tensor Core 路径时选择 BF16，否则回退 FP16；`torch.cuda.is_bf16_supported()` 的默认结果包含模拟支持，不能单独作为硬件加速依据。模型、固定 batch、batch size、序列长度、optimizer 和迭代次数保持一致。固定 batch 用于保证两种模式的输入可比，不替代 60 节的真实 SFT 数据质量与收敛实验。这里的 AMP candidate 只用于观察 dtype 对训练成本的影响，不是 checkpoint 或 offload 策略。`smoke` 用于先验证流程，`pressure` 用于生成与 76 对齐的高压力 baseline；正式采集可将 `REPEATS` 改为 3，报告会同时保存每次结果和均值。
### Colab / GPU 启动检查（可独立运行）

从文档链接打开 Colab 时，Notebook 文件和项目仓库不一定同时存在。先运行下面的单元，它会准备项目根目录、把项目加入 `sys.path`，并检查当前 Python 是否真的使用 CUDA 版 PyTorch。它不会静默重装 PyTorch；如果检测到 CPU 版，会给出安装命令。

```python
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


```python
RUN_REAL_GPU = False  # CPU-first 默认关闭；在 Colab / 本地 GPU 实测时显式改为 True。
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


```python
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
    return {
        'step_time_delta_ms': round(time_delta, 2),
        'peak_mem_delta_mb': round(mem_delta, 2),
        'time_improved': time_delta > 0,
        'memory_improved': mem_delta > 0,
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
        raise RuntimeError('RUN_REAL_GPU=True 但 CUDA 不可用，请先完成 GPU 环境预检。')

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
        return {
            key: round(sum(item[key] for item in runs) / len(runs), 3)
            for key in ('step_time_ms', 'samples_per_s', 'loss', 'peak_mem_mb', 'peak_reserved_mb')
        }

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

## 实测记录：本地 RTX 5070 Ti GPU

本记录保存一次真实 GPU 验证结果，作为 76 的 baseline 样例；读者仍应在自己的环境中重新运行。

| 项目 | 配置 |
|:---|:---|
| 内核 / GPU | Linux `6.8.0-138-generic` / NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| 驱动 / 显存 | `570.211.01` / 12227 MiB |
| PyTorch / CUDA | `2.11.0+cu128` / `12.8` |
| 模型 | `Qwen/Qwen2.5-0.5B-Instruct` |
| batch / seq len | `1 / 768` |
| warmup / iters / repeats | `2 / 5 / 3` |
| 对比模式 | FP32 baseline / AMP BF16 candidate |

| 指标 | FP32 baseline | AMP BF16 candidate | 变化 |
|:---|---:|---:|---:|
| step time | 482.753 ms | 237.998 ms | 提升约 50.7% |
| throughput | 2.072 samples/s | 4.202 samples/s | 提升约 102.8% |
| peak allocated | 9782.56 MiB | 9477.25 MiB | 下降 305.31 MiB（3.12%） |
| peak reserved | 10766 MiB | 10624 MiB | allocator 预留变化 |
| 最后一步 loss | 11.477 | 11.716 | 差值约 +0.239 |
| 状态 / 证据级别 | ok / repeated benchmark | ok / repeated benchmark | 73 只说明固定 workload 的对比结果 |

结论：AMP 带来了明显速度收益，但没有形成实质显存收益，当前自动决策为 `tune`。`peak reserved` 的下降不能直接当作模型显存节省；loss 差异还需要更长训练和固定验证集复核。

73 建立 baseline 与测量口径；76 在相同任务上比较 checkpoint / offload / hybrid；75 再根据 76 的结果形成显存预算决策。

### 解析

**1. TODO 1: 统计训练 step 耗时和峰值显存**
- **实现方式**：先执行 `warmup` 轮训练 step 预热，再用 `time.perf_counter()` 记录正式测量阶段的起点和终点，最后用 `(end - start) / iters` 得到平均 step time。
- **关键点**：warmup 不计入结果，避免首次运行的数据加载、kernel 初始化或缓存状态影响平均耗时。
- **显存统计**：GPU 场景下先调用 `torch.cuda.reset_peak_memory_stats()` 清空历史峰值，再用 `torch.cuda.max_memory_allocated()` 读取本轮训练的峰值显存。CPU 场景下返回 `0.0`，保证模板可以在无 GPU 环境中运行。

**2. TODO 2: 汇总 baseline 和 tuned 的差异**
- **实现方式**：`time_delta = baseline_step_time - tuned_step_time`，`mem_delta = baseline_peak_mem - tuned_peak_mem`。
- **关键点**：这里统一用 `baseline - tuned`，所以 delta 为正表示优化后更快或更省显存。
- **技术细节**：`time_improved` 和 `memory_improved` 只是快速判断标记，真正复盘时还要结合 loss、吞吐和收敛稳定性一起看。

**3. TODO 3: 输出训练项目结论**
- **accept**：速度和显存收益都达标，说明当前改动值得保留并继续推进。
- **tune**：至少有一项收益成立，但还没达到稳定项目结论，适合继续围绕当前方向微调。
- **reject**：速度和显存都没有形成有效收益，说明当前改动不值得继续保留。

**4. TODO 4：建立训练状态账本**
- 统计参数、已产生的梯度和 optimizer state 的 tensor bytes；`optimizer.step()` 之前可能还没有完整的 state。
- 这份账本不包含 activation、临时 workspace 或 allocator reserved；它用于解释对象规模，不等于 GPU 峰值显存。

**训练性能分析的实验原则**
- **固定 baseline**：同一轮对比中固定模型、数据、batch size、seq len、优化器和评测方式。
- **一次只改一个变量**：例如只改 batch size、混合精度、gradient checkpointing 或数据加载方式，避免结果不可归因。
- **指标一起看**：step time 变快但 peak memory、loss 或稳定性变差时，要把取舍写清楚。
- **瓶颈归因**：如果 step time 没有改善，需要回到 profiling 结果，判断瓶颈来自数据等待、前向 / 反向算子，还是显存压力。
- **工程产物**：建议保存对比表、profiling 截图、瓶颈结论和下一轮计划，形成可复用的训练性能排障记录。
