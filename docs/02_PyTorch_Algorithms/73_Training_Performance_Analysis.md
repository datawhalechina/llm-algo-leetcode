# 73. Training Performance Analysis | 训练性能分析

**难度：** Hard | **环境：** CPU-first | **标签：** `显存优化`, `训练剖析`, `性能分析` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/73_Training_Performance_Analysis.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

本节是 Task 3 的测量入口。你需要分析一次训练性能回归或优化是否真实存在。先固定训练任务、收敛约束和显存预算，再把 training step 拆成数据加载、forward、backward 和 optimizer step，分别测量耗时、吞吐与峰值显存。最终输出瓶颈判断、优化前后差异和是否保留改动的建议。

> 环境提示：运行 CPU-first 主线不需要 GPU；运行真实 GPU Step 6 前，请先阅读 [Part02 Intro 的环境说明](./intro.md#environment-notes-环境说明)。Colab、ModelScope 和本地 GPU 默认都使用当前 runtime，不需要先搭建两套虚拟环境。

如果把它放回显存优化路线，一个更直接的读法是：这页不只是泛泛地比较训练快慢，而是优先帮助你判断 `checkpointing / offload / mixed precision / batch` 这些训练侧显存手段到底把峰值显存压下去了多少，又把 step time 和稳定性拉坏了多少。也就是说，它在显存路线里承担的是“证据链页”，不是另一个纯机制页。

**关键词：** `training`, `profiling`, `memory`, `step time`

---

## 前置阅读

**导语：** 先完成 Task 0 的反向传播基础、Task 1 的显存与性能认知，再了解 Task 2 的训练侧显存手段，最后进入本节建立统一测量口径；本节重点不是重复讲机制，而是为后续 76 的方案 benchmark 提供可靠 baseline。
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
### Step 1: 定义训练性能分析目标

- 固定模型、输入数据、batch size、seq len、硬件环境和运行后端，保证 baseline 与 tuned 只差一个变量。
- 明确优化目标，例如降低 step time、提升 samples/s、降低 peak memory 或减少同步等待。
- 同时写清约束条件：训练任务要保留 loss / 收敛约束，不能只追求单项速度收益。
- Baseline 需要能稳定复现，不能只跑一次；建议先 warm-up，再测多轮平均值。
- 这一步的目标是让后面的性能分析有判断标准，而不是只得到一组孤立数字。

### Step 2: 先确认 baseline 合法，再决定是否需要拆解

训练性能分析必须先确认 baseline 可复现。当前最小模板只测完整 training step；如果要把一个 step 拆成能归因的几段，还需要进入 profiler 或额外的阶段计时，不能把总耗时直接当成瓶颈定位。

- 数据加载：DataLoader、CPU 预处理、CPU -> GPU 拷贝是否让 GPU 等待（需要阶段计时或 profiler 才能确认）。
- 前向计算：Attention、Linear、LayerNorm 等 forward kernel 是否占主要时间（当前模板不单独测量）。
- 反向计算：backward kernel、梯度计算和梯度累积是否成为瓶颈（当前模板不单独测量）。
- 优化器更新：optimizer step 是否占用明显时间（需要阶段计时或 profiler 才能确认）。
- 显存峰值：激活、梯度、优化器状态和临时 buffer 是否接近上限。
- 同步开销：是否存在不必要的 CPU/GPU 同步或多卡通信等待。

这一步的目标是把“训练慢”具体化成某一类瓶颈，而不是只得到一个模糊结论。
### Step 3: 用统一口径比较收益与代价

训练优化项目必须同时看 step time、samples/s、peak memory 和 loss，不能只挑单项速度收益下结论。

- 一次只改一个变量，例如 batch size、混合精度、gradient checkpointing、数据加载或同步点。
- 改完后重新测同样的指标，比较 baseline / tuned 的差异。
- 如果 step time 变快但 loss 异常、显存更高或稳定性变差，要把取舍写清楚。
- 这一轮修改的目标是建立因果关系，而不是一次性把所有开关都打开。

这一步的目标是回答：这次改动是把瓶颈解决了，还是只是把瓶颈挪走了。
### Step 4: 输出训练性能结论

训练性能分析最终不是输出“总耗时有没有降”，而是输出这次改动在当前训练任务下是否值得继续保留、微调或回退。

- 输出 baseline / tuned 对比表，至少包含 step time、samples/s、peak memory、loss 和备注。
- 附上 profiling 截图或关键统计，说明瓶颈来自数据、forward、backward、optimizer 还是显存。
- 写清楚本次改动、收益、代价和是否满足 loss / 收敛约束。
- 如果还有后续优化空间，就列出下一轮优先级。

这一步的目标是把训练性能分析收成一份可复用的项目报告。

**本节交付物：** 一份可被 76 复用的 baseline 记录，至少包含 workload、运行环境、模型、batch、seq_len、warmup、iters、step_time_ms、samples_per_s、peak_mem_mb 和 loss。本节的轻量 `accept / tune / reject` 只用于对照，不替代 75 的预算裁决。
### Step 5: 最小代码模板

上面的 Step 1-4 是完整训练性能分析流程。下面的代码只实现其中最小、可复用的三块：测量训练 step 的平均耗时与峰值显存、汇总 baseline / tuned 的差异，以及把结果收成 `accept / tune / reject` 的轻量项目决策。真实项目中的 forward / backward / optimizer 拆解和 loss 约束，需要在 profiling 报告中继续补充。

### 提示

- 先固定 baseline，再看 tuned，不要把环境变量、batch、seq len 一起改掉。
- GPU 场景下要关注 peak memory，CPU 场景下至少要保证计时口径一致。
- 一次只改一个变量，才能把 step time 和显存变化归因到具体修改。
### 测试

运行下面的测试单元，确认 `measure_train_step` 和 `summarize_training_result` 的输出字段完整且口径一致。

```python
import time
import torch

```


```python
# 完成训练性能统计的两个函数
# 目标：完成 measure -> compare 的最小训练性能分析链路

def measure_train_step(train_step_fn, warmup=2, iters=8):
    # ==========================================
    # TODO 1: 记录平均 step time 和 peak memory
    # 提示：先 warmup，再测正式迭代；GPU 场景下记录 peak memory。
    # ==========================================
    for _ in range(warmup):
        train_step_fn()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # start = ???
    for _ in range(iters):
        train_step_fn()
    # end = ???
    # elapsed = ???

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    peak_mem_mb = 0.0
    if torch.cuda.is_available():
        # peak_mem_mb = ???
        pass

    return {
        'step_time_ms': round(elapsed * 1000, 2),
        'peak_mem_mb': round(peak_mem_mb, 2),
    }

def summarize_training_result(base_metrics, tuned_metrics):
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
    # 规则：
    # - 速度和显存收益都达标：accept
    # - 至少一项有正收益：tune
    # - 否则：reject
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

```


```python
# 测试你的实现
def test_training_project_template():
    try:
        counter = {'n': 0}

        def train_step():
            counter['n'] += 1

        result = measure_train_step(train_step, warmup=0, iters=2)
        assert counter['n'] == 2, "measure_train_step 没有正确执行训练迭代次数！"
        assert 'step_time_ms' in result and 'peak_mem_mb' in result, "训练统计字段不完整！"
        assert result['step_time_ms'] >= 0.0, "step_time_ms 应为非负数！"
        assert result['peak_mem_mb'] >= 0.0, "peak_mem_mb 应为非负数！"

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
def measure_train_step(train_step_fn, warmup=2, iters=8):
    for _ in range(warmup):
        train_step_fn()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(iters):
        train_step_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()
    elapsed = (end - start) / iters

    peak_mem_mb = 0.0
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        'step_time_ms': round(elapsed * 1000, 2),
        'peak_mem_mb': round(peak_mem_mb, 2),
    }

# TODO 2: 汇总 baseline 和 tuned 的差异
def summarize_training_result(base_metrics, tuned_metrics):
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

counter = {'n': 0}
def train_step():
    counter['n'] += 1
print(measure_train_step(train_step, warmup=0, iters=2))

```

## Step 6（可选）：真实 GPU 训练 step 对比

前面的 Step 1-5 是 CPU-first 的性能分析模板；本 Step 把同一套完整 training step 测量口径接到真实 causal LM 的 forward / backward / optimizer.step。默认关闭，只有在 GPU、Transformers 和模型依赖准备好时才运行。成功后会保存 `benchmarks/results/73_real_gpu_training.json`；只有当 73 与 76 使用相同 workload 配置时，它才能作为 76 的 baseline 输入。

本示例只比较一个变量：FP32 baseline 与 AMP（优先 BF16，否则 FP16）tuned。模型、固定 batch、batch size、序列长度、optimizer 和迭代次数保持一致。固定 batch 用于保证两种模式的输入可比，不替代 60 节的真实 SFT 数据质量与收敛实验。`smoke` 用于先验证流程，`pressure` 用于生成与 76 对齐的高压力 baseline；正式采集可将 `REPEATS` 改为 3，报告会同时保存每次结果和均值。

```python
RUN_REAL_GPU = True  # 是否运行真实 GPU 测量；实测时显式改为 True。
#REAL_RUN_MODE = 'paired'  # paired：FP32/BF16 对比；bf16_probe：只探测 BF16 容量。
REAL_RUN_MODE =  'bf16_probe'
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
WORKLOAD = 'pressure_1024'  # pressure 保留 768；pressure_1024 与新的 76 高压力实验对齐。
REPEATS = 3  # 正式采集可改为 3；每次重复都会重新初始化模型。
BATCH_SIZE = 1  # 由 WORKLOADS 自动覆盖；保留变量便于阅读配置。
SEQ_LEN = 1024  # 由 WORKLOADS 自动覆盖；显著影响 activation 显存。
WARMUP = 3  # 由 WORKLOADS 自动覆盖；不计入正式平均值。
ITERS = 10  # 由 WORKLOADS 自动覆盖；正式测量轮数。
LEARNING_RATE = 1e-5  # baseline 与 tuned 必须保持一致。
SEED = 42  # 固定输入和初始化，减少方案间随机差异。
from pathlib import Path
OUTPUT_RELATIVE_PATH = Path('benchmarks/results/73_real_gpu_training_bf16.json')

```


```python
import json
import os
import sys
import time
from pathlib import Path

def resolve_project_root():
    override = os.environ.get('LLM_ALGO_PROJECT_ROOT')
    if override:
        return Path(override).expanduser().resolve()
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / 'benchmarks').is_dir() and (candidate / '02_PyTorch_Algorithms').is_dir():
            return candidate
    return current

PROJECT_ROOT = resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from tools.project_runtime import ensure_output_path, runtime_snapshot, validate_training_config
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
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    common = {
        'model_id': MODEL_ID, 'batch_size': BATCH_SIZE, 'seq_len': SEQ_LEN,
        'dtype': 'float32', 'optimizer': 'AdamW', 'workload': WORKLOAD,
        'warmup': WARMUP, 'iters': ITERS, 'amp_dtype': str(amp_dtype),
        'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
        'device': torch.cuda.get_device_name(0),
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

        for _ in range(WARMUP):
            train_step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        losses = [train_step() for _ in range(ITERS)]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        result = {
            'step_time_ms': round(elapsed * 1000 / ITERS, 3),
            'samples_per_s': round(BATCH_SIZE * ITERS / elapsed, 3),
            'loss': round(losses[-1], 6),
            'peak_mem_mb': round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
            'peak_reserved_mb': round(torch.cuda.max_memory_reserved() / (1024 ** 2), 2),
        }
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
| 对比模式 | FP32 baseline / AMP BF16 tuned |

| 指标 | FP32 baseline | AMP BF16 tuned | 变化 |
|:---|---:|---:|---:|
| step time | 482.753 ms | 237.998 ms | 提升约 50.7% |
| throughput | 2.072 samples/s | 4.202 samples/s | 提升约 102.8% |
| peak allocated | 9782.56 MiB | 9477.25 MiB | 下降 305.31 MiB（3.12%） |
| peak reserved | 10766 MiB | 10624 MiB | allocator 预留变化 |
| 最后一步 loss | 11.477 | 11.716 | 差值约 +0.239 |

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

**训练性能分析的实验原则**
- **固定 baseline**：同一轮对比中固定模型、数据、batch size、seq len、优化器和评测方式。
- **一次只改一个变量**：例如只改 batch size、混合精度、gradient checkpointing 或数据加载方式，避免结果不可归因。
- **指标一起看**：step time 变快但 peak memory、loss 或稳定性变差时，要把取舍写清楚。
- **瓶颈归因**：如果 step time 没有改善，需要回到 profiling 结果，判断瓶颈来自数据等待、前向 / 反向算子，还是显存压力。
- **工程产物**：建议保存对比表、profiling 截图、瓶颈结论和下一轮计划，形成可复用的训练性能排障记录。
