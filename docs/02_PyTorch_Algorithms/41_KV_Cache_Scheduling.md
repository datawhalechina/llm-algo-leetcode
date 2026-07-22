# 41. KV Cache Scheduling | KV Cache 调度
**难度：** Hard | **环境：** GPU required | **标签：** `KV Cache`, `Scheduling`, `推理优化` | **目标人群：** 推理系统与缓存工程

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/41_KV_Cache_Scheduling.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


先把前缀缓存、FP8 量化和解码调度看清，再看 KV Cache Scheduling 如何在复用、驱逐、优先级和队列切换之间做平衡。

**关键词：** `KV cache scheduling`, `cache reuse`, `eviction`

## 前置阅读

**导语：** 先看前缀缓存、解码调度和 KV Cache 量化，再看缓存调度会更容易。

- [36. Prefix Caching and Chunked Prefill | 前缀缓存与分块预填充](../02_PyTorch_Algorithms/36_Prefix_Caching_and_Chunked_Prefill.md)
- [38. Decode Scheduling | 解码调度](../02_PyTorch_Algorithms/38_Decode_Scheduling.md)
- [40. FP8 and KV Cache Quantization | FP8 与 KV Cache 量化](../02_PyTorch_Algorithms/40_FP8_and_KV_Cache_Quantization.md)
- [22. vLLM PagedAttention | vLLM PagedAttention](../02_PyTorch_Algorithms/22_vLLM_PagedAttention.md)

## 相关阅读

**导语：** KV Cache 调度之后，可以继续看并行策略和通信 profiling。

- [27. ZeRO Optimizer Sim | ZeRO 优化器模拟](../02_PyTorch_Algorithms/27_ZeRO_Optimizer_Sim.md)
- [42. Communication Profiling with NCCL | NCCL 通信性能剖析](../02_PyTorch_Algorithms/42_Communication_Profiling_with_NCCL.md)
- [P1: 11. KV Cache and Memory Growth | KV Cache 与显存增长](../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md)

### Step 1: 原理与痛点

KV Cache Scheduling 的问题，是如何在多请求、多前缀、多缓存块的条件下，决定哪些缓存继续复用、哪些缓存该延后、哪些缓存可以驱逐。它关注的是“缓存怎么排、怎么切换、怎么回收”，而不是单纯“缓存怎么存”。

### Step 2: 代码实现框架

下面的代码会先模拟一个最小调度器，再把复用、驱逐、优先级和队列切换分成几个小动作。你需要关注的是：不同请求来了以后，缓存资源如何在系统里被动态分配，并尽量让高复用前缀留在热路径上。

### Step 3: 核心机制

这一页的重点是理解 cache-aware scheduling：如果一个前缀已经被很多请求复用，那么它应该更“保守”地被保留；如果某些请求长时间不活跃，就应该允许驱逐、降级处理，或者把它排到更低优先级的队列里。

### Step 4: 动手实战

**要求**：请补全下方 `KVCacheSchedulerSim`，实现一个极简版的 KV Cache 调度器。先把复用、驱逐、优先级和队列切换的基本逻辑跑通，再考虑更复杂的缓存策略。

```python
import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(order=True)
class CacheEntry:
    priority: float
    last_used: int
    prefix: str = field(compare=False)
    hits: int = field(default=0, compare=False)
    bytes: int = field(default=0, compare=False)


class KVCacheSchedulerSim:
    """A tiny KV cache scheduler simulator.

    It models the teaching loop around:
    reuse -> priority -> eviction -> queue switching.
    """

    def __init__(self, capacity_bytes: int = 1024):
        self.capacity_bytes = capacity_bytes
        self.current_bytes = 0
        self.time = 0
        self.entries: Dict[str, CacheEntry] = {}
        self.queue: List[Tuple[float, int, str]] = []
        self.log: List[str] = []

    # TODO 1: 为 cache entry 计算优先级
    def _score(self, hits: int, size: int, last_used: int) -> float:
        raise NotImplementedError

    # TODO 2: 刷新优先级队列
    def _refresh_queue(self, prefix: str):
        raise NotImplementedError

    # TODO 3: 持续驱逐直到容量足够
    def _evict_until_fit(self, needed: int):
        raise NotImplementedError

    # TODO 4: 访问一个 prefix，记录复用或新增
    def touch(self, prefix: str, bytes_: int):
        raise NotImplementedError

    # TODO 5: 执行一串调度请求
    def schedule(self, requests: List[Tuple[str, int]]):
        raise NotImplementedError

    # TODO 6: 导出当前 cache 状态
    def snapshot(self) -> List[Tuple[str, int, float, int]]:
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

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(order=True)
class CacheEntry:
    priority: float
    last_used: int
    prefix: str = field(compare=False)
    hits: int = field(default=0, compare=False)
    bytes: int = field(default=0, compare=False)


class KVCacheSchedulerSim:
    """A tiny KV cache scheduler simulator.

    It models the teaching loop around:
    reuse -> priority -> eviction -> queue switching.
    """

    def __init__(self, capacity_bytes: int = 1024):
        self.capacity_bytes = capacity_bytes
        self.current_bytes = 0
        self.time = 0
        self.entries: Dict[str, CacheEntry] = {}
        self.queue: List[Tuple[float, int, str]] = []
        self.log: List[str] = []

    def _score(self, hits: int, size: int, last_used: int) -> float:
        recency = 1.0 / (1.0 + max(self.time - last_used, 0))
        reuse_bonus = float(hits)
        size_penalty = size / max(self.capacity_bytes, 1)
        return reuse_bonus + 0.5 * recency - 0.25 * size_penalty

    def _refresh_queue(self, prefix: str):
        entry = self.entries[prefix]
        entry.priority = self._score(entry.hits, entry.bytes, entry.last_used)
        heapq.heappush(self.queue, (-entry.priority, entry.last_used, prefix))

    def _evict_until_fit(self, needed: int):
        while self.current_bytes + needed > self.capacity_bytes and self.entries:
            while self.queue:
                neg_p, last_used, prefix = heapq.heappop(self.queue)
                entry = self.entries.get(prefix)
                if entry is None:
                    continue
                if (-neg_p, last_used) != (entry.priority, entry.last_used):
                    continue
                break
            else:
                prefix = min(self.entries.values(), key=lambda e: (e.priority, e.last_used)).prefix
                entry = self.entries[prefix]

            self.current_bytes -= entry.bytes
            self.entries.pop(prefix, None)
            self.log.append(f"evict:{prefix}")

    def touch(self, prefix: str, bytes_: int):
        self.time += 1
        if prefix in self.entries:
            entry = self.entries[prefix]
            entry.hits += 1
            entry.last_used = self.time
            self._refresh_queue(prefix)
            self.log.append(f"reuse:{prefix}")
            return

        self._evict_until_fit(bytes_)
        entry = CacheEntry(priority=0.0, last_used=self.time, prefix=prefix, hits=1, bytes=bytes_)
        entry.priority = self._score(entry.hits, entry.bytes, entry.last_used)
        self.entries[prefix] = entry
        self.current_bytes += bytes_
        heapq.heappush(self.queue, (-entry.priority, entry.last_used, prefix))
        self.log.append(f"add:{prefix}")

    def schedule(self, requests: List[Tuple[str, int]]) -> List[str]:
        """requests: list of (prefix, bytes)."""
        for prefix, bytes_ in requests:
            self.touch(prefix, bytes_)
        return list(self.log)

    def snapshot(self) -> List[Tuple[str, int, float, int]]:
        return [
            (e.prefix, e.bytes, round(e.priority, 4), e.hits)
            for e in sorted(self.entries.values(), key=lambda e: (-e.priority, e.last_used, e.prefix))
        ]

```

### 解析
- KV cache scheduling 的重点是复用、驱逐、优先级和队列切换。
- 它解决的不是“缓存能不能存”，而是“哪些缓存该留、该换、该回收”。
- 这一页更像 cache 管理策略，而不是单纯的量化或算子实现。
### 测试

```python
sim = KVCacheSchedulerSim(capacity_bytes=128)
events = [
    ('a', 40),
    ('b', 48),
    ('a', 40),
    ('c', 56),
    ('d', 48),
    ('a', 40),
]
log = sim.schedule(events)
snap = sim.snapshot()
assert len(log) >= len(events)
assert isinstance(snap, list)
assert all(len(item) == 4 for item in snap)
print('✅ KVCacheSchedulerSim 测试通过')
```
