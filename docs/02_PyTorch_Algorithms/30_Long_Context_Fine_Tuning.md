# 30. Long Context Fine Tuning | 长上下文微调
**难度：** Medium | **环境：** CPU-first | **标签：** `训练微调`, `Long Context`, `数据组织` | **目标人群：** 训练机制学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/30_Long_Context_Fine_Tuning.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

长上下文微调最容易犯的错，不是“上下文不够长”，而是还没把数据长度、显存预算和评测口径说清楚，就直接把 `seq_len` 拉大。结果通常是训练更慢、显存更高、收益却不稳定。

这一节先不碰真实训练框架，而是把项目里最先该做的三件事拆出来：
- 统计长度分布，确认问题到底出在少量超长样本，还是整体长度迁移。
- 规划长上下文批次，估算哪些样本真的能进目标窗口。
- 用统一口径比较 baseline 和 long-context run，判断是否值得继续投入。

**关键词：** `context budget`, `packing`, `length distribution`, `sparse attention`, `linear attention`

本节还用 CPU 小张量对照 Dense、Sparse 和 Linear Attention 的连接数、状态形状与长度账本，帮助你把长上下文的数据问题和架构选择放到同一张检查表里。

---

## 前置阅读

**导语：** 进入本节前，先能读出训练循环中的 batch、activation 和 KV Cache 成本，再比较样本长度分布与目标窗口是否匹配。

- [11. KV Cache and Memory Growth | KV Cache 与显存增长](../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md)
- [12. Gradient Accumulation | 梯度累积](./12_Gradient_Accumulation.md)
- [19. Activation Checkpointing | 激活检查点](./19_Activation_Checkpointing.md)

---

### Step 1: 先确认长度问题长什么样

- 先统计长度分布，区分短样本、过渡样本和超长样本。
- 不要只看最大长度，要看落在不同区间的样本占比。
- 如果超长样本只占很少比例，优先考虑重写数据组织和 packing，而不是盲目扩大训练窗口。
- 先区分是扩大 dense Attention 的窗口，还是改变 Attention 连接/计算形式；这两类方案的质量和实现风险不同。

### Step 2: 把上下文预算写成显式规则

![Long Context Fine-Tuning Budget](/02_PyTorch_Algorithms/30_long_context_budget.svg)

- 先给 response 预留固定 token 空间，避免 prompt 把监督区全部吃掉。
- 再判断样本是否能在目标窗口内完整放下。
- 若放不下，要明确是截断、切块，还是延后到更大窗口实验。
- 对 Sparse Attention 记录 mask 的非零率或连接数量；对 Linear Attention 记录状态大小和递推维度。

### Step 3: 用统一口径比较 baseline 和 candidate run

- 对齐 `fit_rate`、`peak_memory_gb`、`step_time_s` 和 `eval_score`。
- 如果窗口更大，但 fit rate 只提升一点点、显存和步时却显著上升，就不该直接推进。
- 真正应该保留的是“长度收益明显，且成本可接受”的方案。
- CPU 结果只能确认预算计算和 toy 机制；真实 kernel、带宽、吞吐和质量仍需单独 benchmark。

### Step 4: 动手实战

1. 补全 `bucket_length_distribution`，把样本长度分成 `short / medium / long` 三档。
2. 补全 `plan_long_context_batches`，判断样本在预留 response token 后能否完整装入上下文窗口。
3. 补全 `summarize_long_context_experiment`，输出 baseline 和 candidate 的对比结论。
4. 用小张量写出 dense mask 与 sparse mask 的连接数量，并记录 linear attention 状态 shape；只比较结构，不比较 GPU 速度。

### 提示

- `TODO 1` 先按 `short_threshold / long_threshold` 把长度分成 `short / medium / long` 三档。
- `TODO 2` 先算 `available_prompt_tokens`，再判断每个样本是能完整装入还是溢出。
- `TODO 3` 先分别比较 `fit_rate / memory / step_time / eval` 的变化，再决定 `keep_candidate`。


```python
from typing import Dict, List

```


```python
def summarize_sparse_connections(seq_len: int, window_size: int) -> Dict[str, float]:
    """TODO 4: 统计局部窗口 Sparse Attention 的连接数量和非零率。"""
    # 提示：每个 token 最多连接左右 window_size 范围内的位置；先计算总连接数，再除以 seq_len ** 2。
    # total_connections = ???
    # density = ???
    raise NotImplementedError


def summarize_linear_attention_state(seq_len: int, feature_dim: int, value_dim: int) -> Dict[str, int]:
    """TODO 5: 计算 Linear Attention 递推状态的形状和元素数。"""
    # 提示：简化教学模型维护 [feature_dim, value_dim] 状态，不构造 [seq_len, seq_len] 矩阵。
    # state_shape = ???
    # state_elements = ???
    raise NotImplementedError


def bucket_length_distribution(lengths: List[int], short_threshold: int, long_threshold: int) -> Dict[str, int]:
    """
    TODO 1: 把样本长度分成 `short / medium / long` 三档。
    """
    # 提示：先创建 counts，再逐个长度判断落在哪个区间。
    # counts = ???
    # if ???:
    #     counts['short'] += 1
    # elif ???:
    #     counts['medium'] += 1
    # else:
    #     counts['long'] += 1
    raise NotImplementedError


def plan_long_context_batches(samples: List[Dict[str, int]], target_context_len: int, reserved_response_tokens: int) -> Dict[str, object]:
    """
    TODO 2: 规划长上下文批次。
    """
    # 提示：先算 available_prompt_tokens，再把样本分到 fit_names 和 overflow_names。
    # available_prompt_tokens = ???
    # fit_names = ???
    # overflow_names = ???
    # fit_rate = ???
    raise NotImplementedError


def summarize_long_context_experiment(baseline_run: Dict[str, float], candidate_run: Dict[str, float]) -> Dict[str, object]:
    """
    TODO 3: 输出 baseline 和 candidate 的对比结论。
    """
    # 提示：先算 fit_rate_gain、memory_delta_gb、step_time_delta_s、eval_gain，再判断 keep_candidate。
    # fit_rate_gain = ???
    # memory_delta_gb = ???
    # step_time_delta_s = ???
    # eval_gain = ???
    # keep_candidate = ???
    raise NotImplementedError

```


```python
def test_long_context_template():
    try:
        counts = bucket_length_distribution([64, 128, 300, 900], short_threshold=128, long_threshold=512)
        assert counts == {'short': 2, 'medium': 1, 'long': 1}
        sparse = summarize_sparse_connections(seq_len=8, window_size=1)
        assert sparse['total_connections'] == 22
        assert 0 < sparse['density'] < 1
        state = summarize_linear_attention_state(seq_len=128, feature_dim=16, value_dim=8)
        assert state['state_shape'] == (16, 8)
        assert state['state_elements'] == 128

        samples = [
            {'name': 'short_doc', 'prompt_tokens': 300},
            {'name': 'medium_doc', 'prompt_tokens': 1200},
            {'name': 'too_long_doc', 'prompt_tokens': 1900},
        ]
        batch_plan = plan_long_context_batches(samples, target_context_len=2048, reserved_response_tokens=256)
        assert batch_plan['available_prompt_tokens'] == 1792
        assert batch_plan['fit_count'] == 2 and batch_plan['overflow_count'] == 1
        assert abs(batch_plan['fit_rate'] - 2 / 3) < 1e-8

        baseline = {'fit_rate': 0.45, 'peak_memory_gb': 14.0, 'step_time_s': 0.8, 'eval_score': 0.61}
        candidate = {'fit_rate': 0.80, 'peak_memory_gb': 18.5, 'step_time_s': 1.1, 'eval_score': 0.64}
        summary = summarize_long_context_experiment(baseline, candidate)
        assert abs(summary['fit_rate_gain'] - 0.35) < 1e-8
        assert summary['memory_delta_gb'] == 4.5
        assert summary['keep_candidate'] is True
        print('测试通过：长上下文微调模板可以工作。')
    except NotImplementedError:
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError) as e:
        raise NotImplementedError('请先完成 TODO 代码！') from e


test_long_context_template()

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
def summarize_sparse_connections(seq_len: int, window_size: int) -> Dict[str, float]:
    """TODO 4: 统计局部窗口 Sparse Attention 的连接数量和非零率。"""
    total_connections = sum(min(seq_len, i + window_size + 1) - max(0, i - window_size) for i in range(seq_len))
    return {'total_connections': total_connections, 'density': total_connections / (seq_len ** 2)}


def summarize_linear_attention_state(seq_len: int, feature_dim: int, value_dim: int) -> Dict[str, int]:
    """TODO 5: 计算 Linear Attention 递推状态的形状和元素数。"""
    state_shape = (feature_dim, value_dim)
    return {'state_shape': state_shape, 'state_elements': feature_dim * value_dim, 'seq_len': seq_len}


def bucket_length_distribution(lengths: List[int], short_threshold: int, long_threshold: int) -> Dict[str, int]:
    """
    TODO 1: 把样本长度分成 `short / medium / long` 三档。
    """
    # 提示：先创建 counts，再逐个长度判断落在哪个区间。
    # counts = ???
    # if ???:
    #     counts['short'] += 1
    # elif ???:
    #     counts['medium'] += 1
    # else:
    #     counts['long'] += 1
    counts = {'short': 0, 'medium': 0, 'long': 0}
    for length in lengths:
        if length <= short_threshold:
            counts['short'] += 1
        elif length <= long_threshold:
            counts['medium'] += 1
        else:
            counts['long'] += 1
    return counts


def plan_long_context_batches(samples: List[Dict[str, int]], target_context_len: int, reserved_response_tokens: int) -> Dict[str, object]:
    """
    TODO 2: 规划长上下文批次。
    """
    # 提示：先算 available_prompt_tokens，再把样本分到 fit_names 和 overflow_names。
    # available_prompt_tokens = ???
    # fit_names = ???
    # overflow_names = ???
    # fit_rate = ???
    fit_names: List[str] = []
    overflow_names: List[str] = []
    available_prompt_tokens = max(target_context_len - reserved_response_tokens, 0)
    for sample in samples:
        if sample.get('prompt_tokens', 0) <= available_prompt_tokens:
            fit_names.append(sample.get('name', 'sample'))
        else:
            overflow_names.append(sample.get('name', 'sample'))
    total = len(samples)
    return {
        'available_prompt_tokens': available_prompt_tokens,
        'fit_count': len(fit_names),
        'overflow_count': len(overflow_names),
        'fit_rate': len(fit_names) / total if total else 0.0,
        'fit_names': fit_names,
        'overflow_names': overflow_names,
    }


def summarize_long_context_experiment(baseline_run: Dict[str, float], candidate_run: Dict[str, float]) -> Dict[str, object]:
    """
    TODO 3: 输出 baseline 和 candidate 的对比结论。
    """
    # 提示：先算 fit_rate_gain、memory_delta_gb、step_time_delta_s、eval_gain，再判断 keep_candidate。
    # fit_rate_gain = ???
    # memory_delta_gb = ???
    # step_time_delta_s = ???
    # eval_gain = ???
    # keep_candidate = ???
    fit_rate_gain = candidate_run.get('fit_rate', 0.0) - baseline_run.get('fit_rate', 0.0)
    memory_delta_gb = candidate_run.get('peak_memory_gb', 0.0) - baseline_run.get('peak_memory_gb', 0.0)
    step_time_delta_s = candidate_run.get('step_time_s', 0.0) - baseline_run.get('step_time_s', 0.0)
    eval_gain = candidate_run.get('eval_score', 0.0) - baseline_run.get('eval_score', 0.0)
    return {
        'fit_rate_gain': fit_rate_gain,
        'memory_delta_gb': memory_delta_gb,
        'step_time_delta_s': step_time_delta_s,
        'eval_gain': eval_gain,
        'keep_candidate': fit_rate_gain > 0 and eval_gain >= 0,
    }

```

### 解析

**1. TODO 1：把样本长度分成 `short / medium / long` 三档**
- 先按 `short_threshold` 和 `long_threshold` 做长度分桶，避免把“少量超长样本”误判成“整体都需要更长上下文”。
- 长度分布是长上下文项目的第一张账本，因为它决定了问题到底是整体迁移，还是局部极端样本驱动。

**2. TODO 2：规划长上下文批次**
- 先从 `target_context_len` 中扣掉 `reserved_response_tokens`，得到真正可用于 prompt 的 `available_prompt_tokens`。
- 再把样本分成能装入窗口的 `fit_names` 和溢出的 `overflow_names`，最后算出 `fit_rate`。
- 这一步回答的是“哪些样本真的能进目标窗口”，而不是只看理论最大上下文长度。

**3. TODO 3：输出 baseline 和 candidate 的对比结论**
- 同时比较 `fit_rate_gain`、`memory_delta_gb`、`step_time_delta_s` 和 `eval_gain`，再决定 `keep_candidate`。
- 长上下文方案不是窗口越大越好，只有当覆盖率收益和效果收益都成立时，才值得接受更高成本。

**4. TODO 4：统计 Sparse Attention 连接**
- **实现方式**：按局部窗口逐行统计允许连接的 token 数，再除以 `seq_len ** 2` 得到 mask density。
- **边界**：连接数量下降不等于 kernel 时间按相同比例下降；实际收益取决于稀疏布局和实现。

**5. TODO 5：统计 Linear Attention 状态**
- **实现方式**：记录简化递推模型的 `[feature_dim, value_dim]` 状态，而不是构造 `[seq_len, seq_len]` 矩阵。
- **边界**：状态规模趋势不等于质量、吞吐或真实显存结论。

**6. 这页的定位**
- 先做长度分桶，避免把“少量极长样本”误判成“整体都需要更长上下文”。
- 上下文预算要扣除 response 预留区，真正能用来装 prompt 的空间才是关键。
- 对比实验至少要同时看覆盖率增益和效果增益，不能只看窗口是否变大。

## 相关阅读

完成长度分布、上下文预算和 CPU 观察后，可以继续阅读长上下文方法、显存动作与真实微调项目。

- [LongLoRA 原论文：Efficient Fine-tuning of Long-Context Large Language Models](https://arxiv.org/abs/2309.12307)
- [Hugging Face Transformers：长文本生成文档](https://huggingface.co/docs/transformers/main/en/llm_tutorial)
- [32. Data Engineering for SFT | SFT 数据工程](./32_Data_Engineering_for_SFT.md)
- [42. Activation Offload | 激活卸载](./42_Activation_Offload.md)
- [62. Instruction Fine-Tuning Project | 指令微调项目](./62_Instruction_Fine_Tuning_Project.md)
