# INF-PROPOSAL. Dynamic Proposal and Verification Budget | 动态候选生成与验证预算

**难度：** Hard | **环境：** [CPU 主线，GPU 可选](#runtime-environment) | **标签：** `推理优化`, `投机解码`, `动态候选` | **目标人群：** 理解自回归生成的学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/INF-PROPOSAL_Dynamic_Proposal_and_Verification_Budget.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

投机解码的收益不仅取决于草稿器是否更快，还取决于每轮提出多少候选、目标模型能连续接受多少，以及拒绝后需要付出多少修正成本。例如同样提出五个候选，连续接受四个与只接受一个，会摊出完全不同的验证成本。

本节先根据近期接受情况调整下一轮候选数量，再把草稿、验证和修正的成本分开计算，解释为什么调用减少不一定意味着生成更快。

<a id="runtime-environment"></a>

## 运行环境

- **基础练习：** 使用 [BASE-CPU](../guide.md#base-cpu)，运行本页的 CPU 题目与测试。
- **GPU 实验：** 使用 [INF-CUDA-MECHANISM](../topic_discussion/inference_optimization/quickstart.md#inf-cuda-mechanism-quickstart)，模型与工作负载见 5.1。

## 前置阅读

先理解投机验证如何得到连续接受前缀，再比较候选长度的取舍；原生多词元预测可在相关阅读中对照。

- [Part 02 · 23：投机解码](./23_Speculative_Decoding.md)
- [Part 02 · 35：Multi-Token Decoding](./35_Multi_Token_Decoding.md)
## Step 1：候选预算改变工作量，验证决定最终提交

草稿器是提出候选的模型或检索路径，目标模型负责验证；候选预算 $k$ 是本轮最多提出的词元数，不是最终提交数。验证仅保留连续接受前缀，首次拒绝就丢弃后缀；全部接受时还可追加一个目标词元，最终提交仍受 EOS 和剩余生成长度限制。

贪心验证比较候选与同位置目标最大分词元，拒绝时使用目标选择；随机验证则使用接受概率和正残差修正，不能只比较两个词元是否相同。两条路径都必须从已提交上下文继续，撤销未提交候选状态。

| 阶段 | 使用的信息 | 输出 | 与预算的关系 |
| --- | --- | --- | --- |
| 提出候选 | 当前上下文与上限 $k$ | 一段候选 | 提议数量可以小于上限 |
| 目标验证 | 对齐位置的目标证据 | 连续接受前缀或首个拒绝 | 提出更多不保证接受更多 |
| 修正或追加 | 拒绝位置或额外目标位置 | 本轮最终提交词元 | 该词元不计入草稿接受长度 |
| 更新状态 | 已提交词元与有效缓存 | 下一轮上下文 | 拒绝后的候选状态必须撤销 |

![动态候选与验证反馈](../public/02_PyTorch_Algorithms/inf_proposal_loop.svg)

## Step 2：近期接受情况怎样改变下一轮候选长度

候选过短，目标模型一次验证推进的位置可能太少；候选过长，则可能为最终丢弃的后缀付出草稿和验证成本。动态策略可以使用滑动接受率、连续接受长度、草稿不确定性或当前队列压力调整下一轮候选上限 $k$。近期接受率可按每轮已接受草稿数除以已提议数计算，空提议轮单列；它与模型对下一个候选的置信度不是同一个量。置信度提前停止还可能使实际提议数小于预算上限。

| 信号 | 增大预算的条件 | 减小预算的条件 |
| --- | --- | --- |
| 接受率 | 连续多轮高接受 | 拒绝频繁 |
| 草稿置信 | 分布集中 | 不确定性升高 |
| 系统负载 | 验证有余量 | 目标模型队列或显存压力高 |
## Step 3：接受收益怎样覆盖草稿与验证成本

设接受的草稿数为 $A$，总提议数为 $K$，普通解码每词元成本为 $t_b$，草稿每词元成本为 $t_d$，本轮验证和拒绝修正耗时分别为 $t_v$、$t_c$。先只计算接受草稿这一部分的时间差：$G_A=A t_b-[K t_d+t_v+\mathbf{1}_{A<K}t_c]$。例如 $A=K=4$、$t_b=2$ ms、$t_d=0.2$ ms、$t_v=1$ ms，得到 6.2 ms；四个草稿全部失败时，仍需支付草稿和验证成本。

这项估算不计修正或追加词元的产出，因此不能单凭它为负就判定完整生成更慢。完整比较应固定输出协议，用总生成时间除以最终提交词元数，并单独记录草稿接受长度；队列、缓存和重叠执行会进一步改变实际耗时。

![候选预算的收益与浪费](../public/02_PyTorch_Algorithms/inf_proposal_budget.svg)

## Step 4：实现动态预算、连续验证与收益账本

本题用近期接受率、固定的草稿/目标贪心词元序列和时间成本观察预算、连续前缀和时间估算。输入校验、预算边界与报告结构已经给出；你将补全预算调整、首个拒绝位置和净收益的两项成本。前缀函数检查序列相同的位置，不实现随机接受或模型缓存。测试分别覆盖阈值、全部/部分接受、空候选、目标序列不足与修正成本。

| TODO / 函数 | 学习者完成的机制 | 必须满足的约束 | 测试证据 |
| --- | --- | --- | --- |
| TODO 1 · `choose_proposal_budget` | 根据近期接受率调整候选预算 | 在 `min_k` / `max_k` 内按 high / low 阈值增大、保持或减小 | 高、中、低接受率和预算边界 |
| TODO 2 · `verify_candidate_prefix` | 找到连续接受边界 | 在首个不同 Token 处停止；全部一致时无拒绝位置 | 全部接受、中途拒绝与空候选 |
| TODO 3 · `estimate_net_gain` | 估算接受草稿部分的时间差 | 基线节省与 draft、verify、correction 额外成本分项记录 | 正收益、负收益与拒绝修正 |

```python
from statistics import mean

```


```python
def choose_proposal_budget(
    recent_acceptance,
    current_k,
    min_k=1,
    max_k=8,
    high=0.8,
    low=0.4,
):
    """根据近期接受率证据调整下一轮候选长度。"""
    if not (1 <= min_k <= current_k <= max_k):
        raise ValueError('候选预算边界非法')
    if not (0.0 <= low < high <= 1.0):
        raise ValueError('接受率阈值非法')
    if any(not 0.0 <= value <= 1.0 for value in recent_acceptance):
        raise ValueError('接受率必须位于 [0, 1]')
    if not recent_acceptance:
        return current_k
    rate = mean(recent_acceptance)
    # ==========================================
    # TODO 1：按 high / low 阈值调整下一轮预算。
    # ==========================================
    # adjustment = ???  # 根据 high/low 区间取 +1、-1 或 0。
    raise NotImplementedError("请完成 TODO 1：adjustment")
    return min(max(current_k + adjustment, min_k), max_k)


def verify_candidate_prefix(draft_tokens, target_tokens):
    """比较草稿与目标 token，返回连续接受数和首个拒绝位置。"""
    # ==========================================
    # TODO 2：找到首个不同词元的位置。
    # ==========================================
    # rejected_at = ???  # 首个不匹配索引；可比较前缀完全一致时为 None。
    raise NotImplementedError("请完成 TODO 2：rejected_at")
    accepted = min(len(draft_tokens), len(target_tokens)) if rejected_at is None else rejected_at
    if rejected_at is None and len(draft_tokens) > len(target_tokens):
        rejected_at = accepted
    return {'accepted': accepted, 'rejected_at': rejected_at}


def estimate_net_gain(
    accepted,
    proposed,
    baseline_token_ms,
    draft_token_ms,
    verify_ms,
    correction_ms=0.0,
):
    """返回候选路径节省时间、额外成本与净收益。"""
    if not 0 <= accepted <= proposed:
        raise ValueError('accepted 必须位于 0..proposed')
    if min(baseline_token_ms, draft_token_ms, verify_ms, correction_ms) < 0:
        raise ValueError('时间成本不得为负')
    # ==========================================
    # TODO 3：计算接受草稿的基线时间与额外成本。
    # ==========================================
    # saved_ms = ???  # 已接受 token 按基线逐 token 执行可节省的时间。
    # extra_cost_ms = ???  # 全部草稿、一次验证和必要修正的成本。
    raise NotImplementedError("请完成 TODO 3：saved_ms、extra_cost_ms")
    return {
        'saved_ms': saved_ms,
        'extra_cost_ms': extra_cost_ms,
        'net_gain_ms': saved_ms - extra_cost_ms,
    }

```

## 测试

测试动态预算边界、连续接受语义和净收益正负。

```python
def test_budget_adaptation():
    """预算应随近期接受率增减，并保持上下界。"""
    assert choose_proposal_budget([1, 1, 1], 4) == 5
    assert choose_proposal_budget([0, 0, 1], 4) == 3
    assert choose_proposal_budget([], 4) == 4
    assert choose_proposal_budget([1], 8) == 8
    assert choose_proposal_budget([0.8], 4) == 5
    assert choose_proposal_budget([0.4], 4) == 3
    assert choose_proposal_budget([0.6], 4) == 4
    assert choose_proposal_budget([0], 1) == 1


def test_budget_rejects_invalid_acceptance():
    """接受率超出 [0, 1] 时应拒绝输入。"""
    for invalid in ([1.2], [-0.1]):
        try:
            choose_proposal_budget(invalid, 4)
        except ValueError:
            pass
        else:
            raise AssertionError('接受率必须位于 [0, 1]')


def test_contiguous_verification():
    """验证只能接受首个分歧之前的连续前缀。"""
    assert verify_candidate_prefix([1, 2, 3, 4], [1, 2, 9, 4]) == {'accepted': 2, 'rejected_at': 2}
    assert verify_candidate_prefix([1, 2], [1, 2]) == {'accepted': 2, 'rejected_at': None}
    assert verify_candidate_prefix([1, 2, 3], [1, 2]) == {'accepted': 2, 'rejected_at': 2}
    assert verify_candidate_prefix([], [1]) == {'accepted': 0, 'rejected_at': None}


def test_net_gain_ledger():
    """收益账本应区分有效推进与拒绝修正成本。"""
    assert estimate_net_gain(4, 4, 2.0, 0.2, 1.0)['net_gain_ms'] == 6.2
    assert estimate_net_gain(0, 4, 2.0, 0.5, 2.0, 1.0)['net_gain_ms'] < 0
    assert estimate_net_gain(4, 4, 2.0, 0.2, 1.0, 99.0)['net_gain_ms'] == 6.2
    partial = estimate_net_gain(2, 4, 2.0, 0.5, 1.0, 0.25)
    assert partial == {'saved_ms': 4.0, 'extra_cost_ms': 3.25, 'net_gain_ms': 0.75}


test_budget_adaptation()
test_budget_rejects_invalid_acceptance()
test_contiguous_verification()
test_net_gain_ledger()
print('✅ 动态候选机制测试通过')

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
# TODO 1：根据近期接受率调整候选预算。
def choose_proposal_budget(
    recent_acceptance,
    current_k,
    min_k=1,
    max_k=8,
    high=0.8,
    low=0.4,
):
    """根据近期接受率证据调整下一轮候选长度。"""
    if not (1 <= min_k <= current_k <= max_k):
        raise ValueError('候选预算边界非法')
    if not (0.0 <= low < high <= 1.0):
        raise ValueError('接受率阈值非法')
    if any(not 0.0 <= value <= 1.0 for value in recent_acceptance):
        raise ValueError('接受率必须位于 [0, 1]')
    if not recent_acceptance:
        return current_k
    rate = mean(recent_acceptance)
    adjustment = 1 if rate >= high else -1 if rate <= low else 0
    return min(max(current_k + adjustment, min_k), max_k)


# TODO 2：找到连续接受边界。
def verify_candidate_prefix(draft_tokens, target_tokens):
    """比较草稿与目标 token，返回连续接受数和首个拒绝位置。"""
    rejected_at = next(
        (index for index, pair in enumerate(zip(draft_tokens, target_tokens)) if pair[0] != pair[1]),
        None,
    )
    accepted = min(len(draft_tokens), len(target_tokens)) if rejected_at is None else rejected_at
    if rejected_at is None and len(draft_tokens) > len(target_tokens):
        rejected_at = accepted
    return {'accepted': accepted, 'rejected_at': rejected_at}


# TODO 3：计算候选路径净收益。
def estimate_net_gain(
    accepted,
    proposed,
    baseline_token_ms,
    draft_token_ms,
    verify_ms,
    correction_ms=0.0,
):
    """返回候选路径节省时间、额外成本与净收益。"""
    if not 0 <= accepted <= proposed:
        raise ValueError('accepted 必须位于 0..proposed')
    if min(baseline_token_ms, draft_token_ms, verify_ms, correction_ms) < 0:
        raise ValueError('时间成本不得为负')
    saved_ms = accepted * baseline_token_ms
    extra_cost_ms = proposed * draft_token_ms + verify_ms + (correction_ms if accepted < proposed else 0.0)
    return {
        'saved_ms': saved_ms,
        'extra_cost_ms': extra_cost_ms,
        'net_gain_ms': saved_ms - extra_cost_ms,
    }

```

### 解析

#### TODO 1：按近期接受情况调整预算

- 平均接受率达到高阈值时加一步、降至低阈值时减一步，中间区间保持；上下界限制最终预算。
- 测试检查两端阈值、中间区间、空历史和预算饱和，区别“大于”与“大于等于”的边界。

#### TODO 2：找到可验证的连续前缀

- 贪心前缀在首个不同词元处结束，后面重新相同也不能补回；目标序列先结束时，该位置是验证上限，而非概率拒绝的证据。
- 测试分别覆盖中途分歧、全部相同、目标较短和空候选，检查接受数量与首个未通过位置。

#### TODO 3：核对接受草稿的时间账本

- 草稿成本按总提议数计，基线收益按接受数计，发生拒绝才加入修正成本；这项估算未计额外提交词元。
- 测试比较正负时间差，并单独改变修正成本，检查全接受时不扣修正、部分接受时只扣一次。

## Step 5：GPU 机制实验——Transformers 动态辅助生成

| 学习问题 | CPU 已确认 | GPU 要确认与后续使用 |
| --- | --- | --- |
| 固定、启发式和动态候选预算如何改变目标模型调用与端到端延迟 | 预算调整、连续接受和净收益账本 | 在同一目标/助手模型与提示集上确认输出一致性、目标前向次数、median/P95、峰值显存与加速边界；结果进入 68 的投机后端复测 |

### 5.1 环境与固定工作负载

使用 `INF-CUDA-MECHANISM` Profile；参考环境为单卡 12 GB 以上显存，首次运行需要下载目标模型与助手模型，不需要外部数据集。环境创建和预检见[推理优化实验 Quickstart](../topic_discussion/inference_optimization/quickstart.md#inf-cuda-mechanism-quickstart)，实验登记见[推理优化实验资产表](../topic_discussion/inference_optimization/experiment_assets.md)。运行前的 Profile 预检必须返回 `status=ok`。

| 配置层 | 本实验固定口径 | 修改规则 |
| --- | --- | --- |
| 模型 | Qwen2.5 1.5B 目标模型 + 0.5B 助手模型；同 tokenizer 与 dtype | 更换任一模型或 revision 后不得与旧结果合并 |
| 工作负载 | 三条固定提示；贪心生成 64 token | 提示、长度或采样语义变化后递增 workload version |
| 配对变量 | 无助手基线 → constant / heuristic / dynamic | 共用输入、目标模型、生成参数、预热与重复规则 |
| 测量 | 输出一致性、目标/助手前向次数、median/P95、峰值显存 | 先过一致性，再解释调用次数和延迟 |
| 结果 | 候选矩阵与 JSON 路径 | 正式复测使用唯一 `RUN_LABEL` |


```python
RUN_MODE = 'dry_run'  # 默认值 dry_run：只登记计划；改为 real_gpu 执行真实辅助生成矩阵。
RUN_LABEL = 'dry_run'  # 默认值 default；正式复测改为唯一标签，避免覆盖旧结果。
ENVIRONMENT_PROFILE = 'INF-CUDA-MECHANISM'  # 默认单卡推理机制 Profile；预检必须返回 status=ok。
WORKLOAD_ID = 'inf-dynamic-assisted-generation'  # 稳定实验身份；不要在 ID 中编码运行标签。
WORKLOAD_VERSION = 'v1'  # 默认 v1：比较无助手、固定、启发式与动态预算。
TARGET_MODEL_ID = 'Qwen/Qwen2.5-1.5B-Instruct'  # 默认目标模型；负责最终验证和输出。
TARGET_MODEL_REVISION = 'main'  # 默认 main；5.2 自动解析并记录实际 commit。
ASSISTANT_MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'  # 默认助手模型；与目标模型共享 tokenizer。
ASSISTANT_MODEL_REVISION = 'main'  # 默认 main；5.2 自动解析并记录实际 commit。
MODEL_DTYPE = 'float16'  # 默认 float16；更换 dtype 后结果不可直接合并。
PROMPTS = (  # 默认三类短提示；修改文本后递增 workload version。
    '请用三点解释 KV Cache 为什么随上下文增长。',
    'Write a Python function that merges overlapping intervals.',
    'A serving system has high TPOT but low queue wait. Diagnose likely causes.',
)
MAX_NEW_TOKENS = 64  # 默认生成 64 token；增大会提高复测时间与显存占用。
INITIAL_ASSISTANT_TOKENS = 5  # 默认初始候选预算 5 token。
DYNAMIC_CONFIDENCE_THRESHOLD = 0.4  # 默认动态停止阈值 0.4；只用于置信度候选。
ASSISTED_CANDIDATES = (  # 三个候选分别隔离固定预算、启发式预算和置信度提前停止。
    {'name': 'constant', 'schedule': 'constant', 'confidence_threshold': 0.0},
    {'name': 'heuristic_transient', 'schedule': 'heuristic_transient', 'confidence_threshold': 0.0},
    {'name': 'dynamic_confidence', 'schedule': 'constant', 'confidence_threshold': DYNAMIC_CONFIDENCE_THRESHOLD},
)
WARMUP_STEPS = 1  # 默认每条路径预热 1 次；不进入统计。
REPEAT_STEPS = 5  # 默认重复 5 次；用于计算 median 与 P95。
SEED = 42  # 默认随机种子 42；贪心生成仍记录以固定环境状态。
OUTPUT_RELATIVE_PATH = f'benchmarks/results/inf_proposal_{RUN_LABEL}.json'  # 默认仓库相对结果路径。

```

### 5.2 执行并保存 JSON

执行单元只消费 5.1 配置。它先生成无助手参考输出，再依次运行三种助手预算策略；每次测量都重置候选预算，记录目标/助手前向次数、输出一致性、median/P95、峰值显存和失败原因。


```python
from tools.notebook_runtime import package_version, resolve_model_commit
from tools.project_runtime import resolve_project_root

import copy
import json
import math
import os
import platform
import statistics
import time

import torch


REPO_ROOT = resolve_project_root()
OUTPUT_PATH = REPO_ROOT / OUTPUT_RELATIVE_PATH


def percentile_95(values):
    """返回离散样本的最近秩 P95。"""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


def greedy_generation_config(model, tokenizer):
    """从模型默认配置复制一份无采样、无重复惩罚的贪心配置。"""
    config = copy.deepcopy(model.generation_config)
    config.do_sample = False
    config.temperature = None
    config.top_p = None
    config.top_k = None
    config.repetition_penalty = 1.0
    config.max_new_tokens = MAX_NEW_TOKENS
    config.min_new_tokens = 0
    config.use_cache = True
    config.pad_token_id = tokenizer.eos_token_id
    return config


def run_generation(model, tokenizer, prompts, assistant_model=None, generation_config=None):
    """运行完整提示集并记录输出、前向次数与 CUDA 延迟。"""
    target_calls = 0
    assistant_calls = 0

    def count_target(_module, _inputs):
        nonlocal target_calls
        target_calls += 1

    def count_assistant(_module, _inputs):
        nonlocal assistant_calls
        assistant_calls += 1

    target_hook = model.register_forward_pre_hook(count_target)
    assistant_hook = assistant_model.register_forward_pre_hook(count_assistant) if assistant_model is not None else None
    outputs = []
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats()
    try:
        started.record()
        with torch.inference_mode():
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
                generated = model.generate(
                    **inputs,
                    assistant_model=assistant_model,
                    generation_config=generation_config,
                )
                outputs.append(generated[0].detach().cpu().tolist())
        finished.record()
        torch.cuda.synchronize()
    finally:
        target_hook.remove()
        if assistant_hook is not None:
            assistant_hook.remove()
    generated_tokens = sum(max(0, len(ids) - len(tokenizer(prompt).input_ids)) for ids, prompt in zip(outputs, prompts))
    return {
        'outputs': outputs,
        'latency_ms': started.elapsed_time(finished),
        'target_forward_calls': target_calls,
        'assistant_forward_calls': assistant_calls,
        'generated_tokens': generated_tokens,
        'tokens_per_target_call': generated_tokens / max(1, target_calls),
        'peak_memory_bytes': int(torch.cuda.max_memory_allocated()),
    }


result = {
    'schema_version': 'inference-mechanism-gpu/v1',
    'semantic_id': 'INF-PROPOSAL',
    'environment_profile': ENVIRONMENT_PROFILE,
    'run_mode': RUN_MODE,
    'run_label': RUN_LABEL,
    'framework': {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'transformers': package_version('transformers'),
    },
    'hardware': {
        'cuda_available': torch.cuda.is_available(),
        'cuda': torch.version.cuda,
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    },
    'workload': {
        'id': WORKLOAD_ID,
        'version': WORKLOAD_VERSION,
        'target_model_id': TARGET_MODEL_ID,
        'target_revision': TARGET_MODEL_REVISION,
        'target_commit': None,
        'assistant_model_id': ASSISTANT_MODEL_ID,
        'assistant_revision': ASSISTANT_MODEL_REVISION,
        'assistant_commit': None,
        'dtype': MODEL_DTYPE,
        'prompt_count': len(PROMPTS),
        'max_new_tokens': MAX_NEW_TOKENS,
        'assisted_candidates': list(ASSISTED_CANDIDATES),
        'initial_assistant_tokens': INITIAL_ASSISTANT_TOKENS,
        'dynamic_confidence_threshold': DYNAMIC_CONFIDENCE_THRESHOLD,
        'warmup_steps': WARMUP_STEPS,
        'repeat_steps': REPEAT_STEPS,
        'seed': SEED,
    },
    'baseline': None,
    'candidates': [],
    'failure': None,
    'evidence_level': 'dry_run',
    'decision': 'not_measured',
}

if RUN_MODE == 'real_gpu':
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('RUN_MODE=real_gpu，但 CUDA 不可用')
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cache_dir = os.getenv('HF_HUB_CACHE') or None
        local_only = os.getenv('HF_HUB_OFFLINE') == '1'
        target_commit, _ = resolve_model_commit(TARGET_MODEL_ID, TARGET_MODEL_REVISION, cache_dir, local_only)
        assistant_commit, _ = resolve_model_commit(ASSISTANT_MODEL_ID, ASSISTANT_MODEL_REVISION, cache_dir, local_only)
        result['workload']['target_commit'] = target_commit
        result['workload']['assistant_commit'] = assistant_commit

        dtype = getattr(torch, MODEL_DTYPE)
        torch.manual_seed(SEED)
        tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID, revision=target_commit, cache_dir=cache_dir, local_files_only=local_only)
        target = AutoModelForCausalLM.from_pretrained(TARGET_MODEL_ID, revision=target_commit, cache_dir=cache_dir, local_files_only=local_only, dtype=dtype).to('cuda').eval()
        target_generation_config = greedy_generation_config(target, tokenizer)
        target.generation_config = copy.deepcopy(target_generation_config)
        for _ in range(WARMUP_STEPS):
            run_generation(target, tokenizer, PROMPTS[:1], generation_config=target_generation_config)
        baseline_runs = [run_generation(target, tokenizer, PROMPTS, generation_config=target_generation_config) for _ in range(REPEAT_STEPS)]
        reference_outputs = baseline_runs[0]['outputs']
        result['baseline'] = {
            'name': 'target_only',
            'median_latency_ms': statistics.median(item['latency_ms'] for item in baseline_runs),
            'p95_latency_ms': percentile_95([item['latency_ms'] for item in baseline_runs]),
            'median_target_forward_calls': statistics.median(item['target_forward_calls'] for item in baseline_runs),
            'peak_memory_bytes': max(item['peak_memory_bytes'] for item in baseline_runs),
        }

        assistant = AutoModelForCausalLM.from_pretrained(ASSISTANT_MODEL_ID, revision=assistant_commit, cache_dir=cache_dir, local_files_only=local_only, dtype=dtype).to('cuda').eval()
        assistant_generation_config = greedy_generation_config(assistant, tokenizer)
        assistant.generation_config = copy.deepcopy(assistant_generation_config)

        for candidate in ASSISTED_CANDIDATES:
            assistant.generation_config.num_assistant_tokens_schedule = candidate['schedule']
            assistant.generation_config.num_assistant_tokens = INITIAL_ASSISTANT_TOKENS
            assistant.generation_config.assistant_confidence_threshold = candidate['confidence_threshold']
            runs = []
            for _ in range(REPEAT_STEPS):
                assistant.generation_config.num_assistant_tokens = INITIAL_ASSISTANT_TOKENS
                runs.append(run_generation(target, tokenizer, PROMPTS, assistant_model=assistant, generation_config=target_generation_config))
            median_ms = statistics.median(item['latency_ms'] for item in runs)
            outputs_match = all(item['outputs'] == reference_outputs for item in runs)
            result['candidates'].append({
                'candidate': candidate['name'],
                'schedule': candidate['schedule'],
                'confidence_threshold': candidate['confidence_threshold'],
                'outputs_match_baseline': outputs_match,
                'median_latency_ms': median_ms,
                'p95_latency_ms': percentile_95([item['latency_ms'] for item in runs]),
                'latency_ratio_vs_baseline': median_ms / result['baseline']['median_latency_ms'],
                'median_target_forward_calls': statistics.median(item['target_forward_calls'] for item in runs),
                'median_assistant_forward_calls': statistics.median(item['assistant_forward_calls'] for item in runs),
                'median_tokens_per_target_call': statistics.median(item['tokens_per_target_call'] for item in runs),
                'peak_memory_bytes': max(item['peak_memory_bytes'] for item in runs),
                'final_num_assistant_tokens': assistant.generation_config.num_assistant_tokens,
                'path_confirmed': True,
            })

        valid = [item for item in result['candidates'] if item['outputs_match_baseline'] and item['path_confirmed']]
        faster = [item for item in valid if item['latency_ratio_vs_baseline'] < 0.95]
        result['evidence_level'] = 'local_mechanism_benchmark'
        result['decision'] = 'accept' if faster else ('tune' if valid else 'reject')
    except Exception as error:
        result['failure'] = {'type': type(error).__name__, 'message': str(error)}
        result['evidence_level'] = 'failed_real_gpu_attempt'
        result['decision'] = 'reject_until_fixed'

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
result

```

### 5.3 读取结果

只读取 5.2 保存的 JSON。先展示环境、工作负载、失败、证据等级和决策，再比较三种预算策略的输出一致性、目标/助手前向次数、median/P95、峰值显存和相对延迟。


```python
import json
from tools.project_runtime import resolve_project_root

result_path = resolve_project_root() / OUTPUT_RELATIVE_PATH
report = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else {'failure': {'type': 'not_run', 'message': str(result_path)}, 'candidates': []}
summary = {
    'schema_version': report.get('schema_version'),
    'framework': report.get('framework'),
    'hardware': report.get('hardware'),
    'workload': report.get('workload'),
    'failure': report.get('failure'),
    'evidence_level': report.get('evidence_level'),
    'decision': report.get('decision'),
}
matrix = [
    {
        'candidate': item.get('candidate'),
        'schedule': item.get('schedule'),
        'confidence_threshold': item.get('confidence_threshold'),
        'outputs_match_baseline': item.get('outputs_match_baseline'),
        'latency_ratio_vs_baseline': item.get('latency_ratio_vs_baseline'),
        'median_latency_ms': item.get('median_latency_ms'),
        'p95_latency_ms': item.get('p95_latency_ms'),
        'median_target_forward_calls': item.get('median_target_forward_calls'),
        'median_assistant_forward_calls': item.get('median_assistant_forward_calls'),
        'median_tokens_per_target_call': item.get('median_tokens_per_target_call'),
        'peak_memory_bytes': item.get('peak_memory_bytes'),
        'path_confirmed': item.get('path_confirmed'),
    }
    for item in report.get('candidates', [])
]
summary, report.get('baseline'), matrix

```

### 5.4 解释结果并作出决策

先检查目标/助手模型 commit、提示、生成长度和贪心语义是否一致，再确认候选路径已启用且输出与无助手基线完全一致。随后比较目标前向次数、每次目标调用的有效 token、median/P95 和峰值显存；调用更少但延迟更高时，说明助手成本或同步开销抵消了收益。

`rtx5070ti_proposal_v2` 已在 RTX 5070 Ti Laptop 12 GiB 上完成配对实验。三种候选均保持贪心输出与仅目标模型基线一致。固定预算把目标模型前向次数中位数从 192 降至 61，但延迟比为 1.059；启发式预算降至 68 次，延迟比为 1.006，两者的助手成本抵消了调用减少。置信度动态候选为 73 次目标前向和 186 次助手前向，延迟比为 0.854，Median 从 2594.6 ms 降至 2216.9 ms，成为本机候选。

| 观察项 | 读取字段 | 当前结果 | 解释与动作 |
| --- | --- | --- | --- |
| 运行与口径 | `failure`、模型 commit、工作负载、生成参数 | 无失败；目标 1.5B、助手 0.5B、贪心生成 64 词元 | 口径变化后不得合并旧结果 |
| 固定预算 | 目标/助手前向次数、相对延迟 | 61 / 289；1.059 | 调用减少但端到端更慢，拒绝当前配置 |
| 启发式预算 | 目标/助手前向次数、相对延迟 | 68 / 256；1.006 | 尚无稳定收益，继续调参 |
| 置信度动态候选 | 输出一致性、目标/助手前向次数、相对延迟 | 一致；73 / 186；0.854 | 接受为本机机制候选 |
| 证据与下一步 | `evidence_level`、`decision` | `local_mechanism_benchmark`、`accept` | 扩大输入和负载后再作后端结论 |

## 相关阅读

继续比较原生候选与真实后端选型，或查阅框架辅助生成协议。

- [原生多词元预测](./35_Multi_Token_Decoding.md)：模型内部如何提出多个候选。
- [投机解码机制选型基准](./68_Speculative_Decoding_Benchmark.md)：能力、资产、输出一致性与净收益的联合判断。

- [Transformers Assisted Decoding](https://huggingface.co/docs/transformers/main/assisted_decoding)
- [Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [vLLM Speculators：算法选择](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/decision_guide/)
