# 28. Pipeline Parallelism MicroBatch | Pipeline 并行微批次

**难度：** Hard | **环境：** CPU-first | **标签：** `分布式训练`, `Pipeline Parallelism`, `调度` | **目标人群：** 分布式训练工程师

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/28_Pipeline_Parallelism_MicroBatch.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


先把流水线分段和微批次调度的关系理清，再看 1F1B 的气泡分析会更顺。

**关键词：** `Pipeline Parallelism`, `Micro-batch`, `bubble ratio`

## 前置阅读

**导语：** 先看 ZeRO，再看 Pipeline 的微批次调度会更容易理解流水线并行。
- [27. ZeRO Optimizer Sim | ZeRO 优化器模拟](./27_ZeRO_Optimizer_Sim.md)
- [25. Quantization W8A16 | W8A16 量化](./25_Quantization_W8A16.md)

## 相关阅读

**导语：** Pipeline 之后，建议继续看 Tensor Parallelism 和项目实战。
- [29. Tensor Parallelism Sim | Tensor 并行模拟](./29_Tensor_Parallelism_Sim.md)
- [30. LoRA Fine-Tuning Project | LoRA 微调项目](./30_LoRA_Fine_Tuning_Project.md)


### Step 1: 为什么会有 Bubble

**核心手段：微批次 (Micro-batch)**
为了填补气泡，我们不能等整个巨大的 Batch Size 都走完再交给下一个 GPU。我们把一个大 Batch 切分成 $m$ 个 Micro-batch。GPU 1 算完 Micro-batch 1 后立马丢给 GPU 2，自己接着算 Micro-batch 2。
从调度角度看，1F1B 之所以会产生 bubble，是因为流水线在“灌满”和“排空”阶段无法始终保持每个 Stage 都有活干；增加 micro-batch 的数量，本质上就是把这些空档尽量填起来。

### Step 2: Bubble Ratio 公式怎么来

> **大厂高频面试题：推导 1F1B 调度的气泡占比 (Bubble Ratio)**
> 假设：
> - 模型切分在 $p$ 张 GPU (即 Pipeline Stage) 上。
> - 一个全局 Batch 被切分成了 $m$ 个 Micro-batch。
> - 忽略通信延迟，假设前向和反向计算时间恒定。
> 
> **理想满载时间 (Ideal Compute)**：$m \times p$ 个单元的时间。
> **实际耗时 (Actual Time)**：$(p - 1) + m$ 个单位时间的流水线排空。
> 
> **Bubble Ratio 公式**：
> $\text{Bubble} = \frac{p - 1}{m + p - 1} \approx \frac{p - 1}{m}$ (当 m 足够大时)

### Step 3: 代码实现框架

这一步不是再推一遍公式，而是把上面的调度直觉落成可执行的时间轴：先生成每个时间步里各个 stage 的活跃状态，再统计 active slots，最后得到 bubble ratio。测试会用 `p=8, m=32` 检查结果是否落在合理区间。


```python
import torch
```


```python

def build_pipeline_timeline(p, m):
    """
    构造一个简化的流水线时间轴。

    timeline[t] 记录第 t 个时间步里，哪些 stage 正在处理哪些 micro-batch。
    """
    # ==========================================
    # TODO 1: 构造 Pipeline 的时间轴
    # 提示: 每个时间步里，记录 stage 与 micro-batch 的对应关系
    # timeline = ???
    # ==========================================
    pass


def compute_bubble_ratio(p, m):
    """
    计算流水线并行的气泡率。
    
    Args:
        p: Pipeline Stage 数量 (GPU 数量)
        m: Micro-batch 的数量
        
    Returns:
        float: 气泡占比 [0, 1]
    """
    # ==========================================
    # TODO 2: 基于时间轴统计 active slots，并计算 Bubble Ratio
    # 提示: Bubble Ratio = 1 - active_slots / total_slots
    # timeline = ???
    # active_slots = ???
    # bubble = ???
    # ==========================================
    pass


```


```python
def test_pipeline_bubble():
    try:
        # 测试 8 个 Stage，切分为 32 个 micro-batch
        ratio = compute_bubble_ratio(p=8, m=32)
        # 精确公式结果为 7 / (32 + 7) = 7/39 = 0.179...
        # 近似公式为 7 / 32 = 0.218...
        # 为了兼容两种答案，我们只检查上限
        assert ratio is not None, "未实现计算！"
        assert 0.15 < ratio < 0.25, f"计算错误，结果应该在 0.17~0.22 左右，实际为 {ratio}"
        
        print("✅ 测试通过！在大规模集群训练时，为了降低流水线气泡，Micro-batch 的数量 m 必须远远大于 Stage 的数量 p。")
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError, RuntimeError) as e:
        if isinstance(e, AttributeError):
            print("代码未完成，无法找到必要的属性")
        elif isinstance(e, NameError):
            print("代码可能未完成，导致了变量未定义")
        elif isinstance(e, TypeError):
            print("代码可能未完成，导致了类型错误")
        elif isinstance(e, ValueError):
            print("代码可能未完成，导致了张量维度错误")
        elif isinstance(e, AssertionError):
            print("代码可能未完成，导致了断言失败")
        else:
            print("代码可能未完成，导致了运行时错误")
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise

test_pipeline_bubble()


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
def build_pipeline_timeline(p, m):
    # TODO 1: 构造 Pipeline 的时间轴
    timeline = []
    total_steps = p + m - 1
    for t in range(total_steps):
        active = []
        for stage in range(p):
            micro_idx = t - stage
            if 0 <= micro_idx < m:
                active.append((stage, micro_idx))
        timeline.append(active)
    return timeline


def compute_bubble_ratio(p, m):
    # TODO 2: 基于时间轴统计活跃槽位并计算 Bubble Ratio
    timeline = build_pipeline_timeline(p, m)
    active_slots = sum(len(step) for step in timeline)
    total_slots = len(timeline) * p
    bubble = 1 - active_slots / total_slots
    return bubble


```

### 解析

**1. TODO 1（构造时间轴）**
- 通过 `timeline[t]` 记录每个时间步里哪些 stage 正在处理哪些 micro-batch。
- `micro_idx = t - stage` 体现了 1F1B 的对角线式调度关系。
- 只要 `0 <= micro_idx < m`，就说明该 stage 在这个时间步是活跃的。

**2. TODO 2（统计活跃槽位并计算 Bubble Ratio）**
- `active_slots` 统计时间轴里所有活跃的 stage/micro-batch 组合数。
- `total_slots = len(timeline) * p` 表示如果每个时间步都满载时的槽位总数。
- `bubble = 1 - active_slots / total_slots` 就是气泡占比。

**3. 进阶思考**
- 当 `m` 远大于 `p` 时，`Bubble Ratio` 会快速下降。
- 这正是大规模训练里要尽量使用更多 micro-batch 的原因。
- 实际系统里还会叠加通信、重计算和显存约束，调度会更复杂。
