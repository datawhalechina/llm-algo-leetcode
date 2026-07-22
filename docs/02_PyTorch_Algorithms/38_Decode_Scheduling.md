# 38. Decode Scheduling | 解码调度
**难度：** Hard | **环境：** GPU required | **标签：** `解码`, `Scheduling`, `推理优化` | **目标人群：** 推理系统与服务调度

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/38_Decode_Scheduling.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


先把投机解码、前缀缓存和多 token 生成看清，再看 decode scheduling 如何组织 prefill / decode / batch 才会更顺。

这里的关键不是“谁先跑”这么简单，而是要同时处理不同请求的阶段差异、缓存命中和等待时间。一个调度器如果只顾着轮询顺序，很容易让 GPU 在预填充和解码之间来回切换；如果只顾着 decode 吞吐，又可能把长前缀请求拖得太久。

**关键词：** `decode scheduling`, `prefill`, `batch reordering`

## 前置阅读

**导语：** 先把解码策略、投机解码和前缀缓存理顺，再看调度规则会更容易。

- [21. Decoding Strategies | 解码策略](../02_PyTorch_Algorithms/21_Decoding_Strategies.md)
- [23. Speculative Decoding | 投机解码](../02_PyTorch_Algorithms/23_Speculative_Decoding.md)
- [36. Prefix Caching and Chunked Prefill | 前缀缓存与分块预填充](../02_PyTorch_Algorithms/36_Prefix_Caching_and_Chunked_Prefill.md)
- [37. Multi-Token Decoding | 多 Token 解码](../02_PyTorch_Algorithms/37_Multi_Token_Decoding.md)

## 相关阅读

**导语：** 解码调度之后，可以继续看 PagedAttention 和 RadixAttention。

- [22. vLLM PagedAttention | vLLM PagedAttention](../02_PyTorch_Algorithms/22_vLLM_PagedAttention.md)
- [24. SGLang RadixAttention | SGLang 基数注意力](../02_PyTorch_Algorithms/24_SGLang_RadixAttention.md)
- [P1: 17. CUDA Stream and Asynchrony | CUDA Stream 与异步执行](../01_Hardware_Math_and_Systems/17_CUDA_Stream_and_Asynchrony.md)

### Step 1: 原理与痛点

Decode scheduling 的问题不只是“谁先跑”，而是如何同时兼顾 prefill 的吞吐、decode 的延迟和 batch 重组的公平性。不同请求在不同阶段对 GPU 的需求不同，调度器要做的就是把这些阶段穿起来，减少空转和等待。

从机制上看，decode scheduling 其实在回答三个问题：

- **先调谁**：短请求、长请求、已命中 cache 的请求，谁应该先进入 GPU；
- **怎么调**：prefill 和 decode 是否要分队列，batch 是否要重组；
- **何时切换**：当某个请求完成 prefill 后，是否立刻进入 decode，还是要等更多请求一起凑 batch。

### Step 2: 代码实现框架

下面的代码会模拟一个最小调度器：输入多个请求后，先区分 prefill 与 decode，再按优先级和缓存命中情况安排执行顺序。这里更重要的是理解“调度决策”本身，而不是某个复杂服务框架的全部细节。

你可以把它拆成三层：

- **请求层**：记录每个请求当前处于 prefill 还是 decode；
- **策略层**：决定每一步优先执行哪个请求；
- **执行层**：把调度结果落到具体的 batch / queue / step 上。

### Step 3: 核心机制

一个好的 decode scheduler 往往要同时看队列长度、cache 命中、请求阶段和等待时间。它不是单纯把任务排队，而是尽量让 GPU 既不空转，也不被长请求拖住。

换句话说，调度的本质不是“先来先服务”，而是“在吞吐、延迟和公平性之间找一个可解释的平衡点”。如果 batch 一味求大，延迟会被拉长；如果只顾着短请求，长请求又会饿死。

### Step 4: 动手实战

**要求**：请补全下方 `DecodeSchedulerSim`，实现一个极简版的 prefill / decode 调度器。先把请求阶段、优先级和队列切换逻辑跑通，再考虑更复杂的服务策略。

```python
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Sequence


Phase = Literal['prefill', 'decode']


@dataclass
class RequestState:
    request_id: int
    prompt_len: int
    generated_len: int = 0
    phase: Phase = 'prefill'
    priority: int = 0
    cache_hit: bool = False

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.generated_len

    @property
    def done(self) -> bool:
        return self.generated_len >= self.prompt_len


class DecodeSchedulerSim:
    """极简版 prefill / decode 调度器。"""

    def __init__(self):
        self.queue: List[RequestState] = []
        self.timeline: List[Dict[str, int | str]] = []

    # TODO 1: 入队一个请求，记录其初始阶段
    def enqueue(self, request_id: int, prompt_len: int, priority: int = 0, cache_hit: bool = False) -> None:
        raise NotImplementedError

    # TODO 2: 为请求定义调度排序键
    def _schedule_key(self, req: RequestState):
        raise NotImplementedError

    # TODO 3: 执行一个最小调度步
    def step(self) -> Dict[str, int | str] | None:
        raise NotImplementedError

    # TODO 4: 持续调度直到完成或达到步数上限
    def run(self, max_steps: int = 100) -> List[Dict[str, int | str]]:
        raise NotImplementedError
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
# TODO：下面是题目区的参考实现。

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Sequence


Phase = Literal['prefill', 'decode']


@dataclass
class RequestState:
    request_id: int
    prompt_len: int
    generated_len: int = 0
    phase: Phase = 'prefill'
    priority: int = 0
    cache_hit: bool = False

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.generated_len

    @property
    def done(self) -> bool:
        return self.generated_len >= self.prompt_len


class DecodeSchedulerSim:
    """极简版 prefill / decode 调度器。"""

    def __init__(self):
        self.queue: List[RequestState] = []
        self.timeline: List[Dict[str, int | str]] = []

    def enqueue(self, request_id: int, prompt_len: int, priority: int = 0, cache_hit: bool = False) -> None:
        """加入一个请求，默认先处于 prefill 阶段。"""
        self.queue.append(
            RequestState(
                request_id=request_id,
                prompt_len=prompt_len,
                priority=priority,
                cache_hit=cache_hit,
            )
        )

    def _schedule_key(self, req: RequestState):
        """优先挑选 cache 命中高、等待少、阶段更适合推进的请求。"""
        phase_rank = 0 if req.phase == 'prefill' else 1
        cache_rank = 0 if req.cache_hit else 1
        return (phase_rank, cache_rank, -req.priority, req.total_len, req.request_id)

    def step(self) -> Dict[str, int | str] | None:
        """执行一个最小调度步，返回本轮被调度的请求信息。"""
        active = [req for req in self.queue if not req.done]
        if not active:
            return None

        chosen = min(active, key=self._schedule_key)
        if chosen.phase == 'prefill':
            chosen.phase = 'decode'
            event = {
                'request_id': chosen.request_id,
                'phase': 'prefill',
                'action': 'prefill_to_decode',
                'prompt_len': chosen.prompt_len,
                'generated_len': chosen.generated_len,
            }
        else:
            chosen.generated_len += 1
            if chosen.done:
                event = {
                    'request_id': chosen.request_id,
                    'phase': 'decode',
                    'action': 'finish',
                    'prompt_len': chosen.prompt_len,
                    'generated_len': chosen.generated_len,
                }
            else:
                event = {
                    'request_id': chosen.request_id,
                    'phase': 'decode',
                    'action': 'decode_one_step',
                    'prompt_len': chosen.prompt_len,
                    'generated_len': chosen.generated_len,
                }

        self.timeline.append(event)
        return event

    def run(self, max_steps: int = 100) -> List[Dict[str, int | str]]:
        """持续调度到所有请求完成，或达到最大步数。"""
        steps = 0
        while steps < max_steps:
            event = self.step()
            if event is None:
                break
            steps += 1
        return self.timeline


def _demo_scheduler() -> List[Dict[str, int | str]]:
    sim = DecodeSchedulerSim()
    sim.enqueue(request_id=1, prompt_len=2, priority=2, cache_hit=True)
    sim.enqueue(request_id=2, prompt_len=3, priority=1, cache_hit=False)
    sim.enqueue(request_id=3, prompt_len=1, priority=3, cache_hit=True)
    return sim.run(max_steps=12)

```

### 解析
- Decode scheduling 的重点是同时兼顾 prefill 吞吐、decode 延迟和队列公平性。
- 本页把调度拆成请求层、策略层、执行层，便于理解“先调谁、怎么调、何时切换”。
- 只要能观察到调度事件序列，就能继续往更复杂的策略扩展。
### 测试

```python
sim = DecodeSchedulerSim()
sim.enqueue(request_id=1, prompt_len=2, priority=2, cache_hit=True)
sim.enqueue(request_id=2, prompt_len=3, priority=1, cache_hit=False)
events = sim.run(max_steps=10)
assert len(events) > 0
assert all('request_id' in e for e in events)
assert any(e['action'] in {'prefill_to_decode', 'decode_one_step', 'finish'} for e in events)
print('✅ DecodeSchedulerSim 测试通过')
```
