# 22. vLLM PagedAttention | vLLM 分页注意力
**难度：** Hard | **环境：** [CPU 主线](#runtime-environment) | **标签：** `推理优化`, `KV Cache`, `PagedAttention` | **目标人群：** 理解注意力与 KV 缓存的学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/22_vLLM_PagedAttention.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

在线推理服务里，请求不会整齐地一起开始、一起结束：有的提示很短，有的上下文很长，有的已生成完，有的还在逐词元解码。如果仍然按最大长度给每个请求预留整块 KV Cache，未使用的预留容量会限制同时容纳的请求数。

分页注意力（PagedAttention）把键值缓存（KV Cache）放入固定大小的物理块，并用块表（block table）记录请求位置到物理存储的映射。理解按需分配、跨块扩容与释放后，就能计算尾块余量，并判断空间收益来自哪里。

**关键词：** `PagedAttention`, `KV cache`, `block table`

---

<a id="runtime-environment"></a>

## 运行环境

- **基础练习：** 使用 [BASE-CPU](../guide.md#base-cpu)，运行本页题目与测试。

## 前置阅读

先理解 KV Cache 随序列长度增长的容量变化，以及 MHA/GQA 如何决定每个 token 保存的 K/V 状态；再观察分页 Block 如何管理这些状态。
- [Part 01 · 11. KV Cache and Memory Growth | KV Cache 与显存增长](../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md)
- [Part 02 · 04. MHA / GQA | 多头注意力与分组查询注意力](./04_Attention_MHA_GQA.md)

---

## Step 1： PagedAttention 如何组织逻辑与物理 KV Cache
在线请求的长度和结束时间并不相同，如果每个请求都按最大长度预留连续 KV Cache，就会同时产生未使用尾部和难以复用的空闲空间。PagedAttention 的输入是请求 token 与缓存预算，处理方式是按需分配固定大小的物理 Block，并用 Block Table 记录逻辑位置到物理 Block 的映射；输出是可继续增长、可回收的缓存布局。
这里的块是容纳若干词元 K/V 的存储单位，不是 Transformer 模型块，也不是 CUDA 线程块。页内逻辑位置连续，物理块编号可以不连续。

| 管理方式 | 如何分配 KV Cache | 主要问题或收益 |
|---|---|---|
| 连续预留 | 按 `max_len` 一次分配连续空间 | 长度未知时产生尾部浪费和空洞 |
| 虚拟内存类比 | 用逻辑位置访问物理存储 | 逻辑地址与物理地址可以分离 |
| 固定 Block | 每个 Block 容纳固定数量 token，例如 16 个 | 分配粒度统一，便于回收和复用 |
| 分页管理 | 按需分配固定大小的物理 Block | 物理块可以不连续，空间更容易复用 |
| Block Table | 记录逻辑 Block 到物理 Block 的映射 | 计算时仍能按逻辑 token 顺序访问缓存 |

![PagedAttention 的 KV Cache 管理总览](../public/02_PyTorch_Algorithms/22_paged_attention_overview.svg)

## Step 2：有效缓存长度怎样触发块分配

预填充（Prefill）处理提示词并建立初始缓存；解码（Decode）处理新位置并追加它的 K/V。设每块容纳四个词元，六个有效位置占两块，处理第七、第八个位置仍使用尾块，处理第九个位置才需要第三块。采样出一个新词元不等于它的 K/V 已经写入，二者在生成循环中相差一次处理时机。

块表负责逻辑位置到物理存储的映射，分配器负责取得与归还物理块。一次扩容若因容量不足失败，请求长度和块表应保持原样，不能只更新其中一个。

| 阶段 | 输入 | 状态变化 | 需要检查的边界 |
| --- | --- | --- | --- |
| 预填充 | 已处理提示长度 | 按块大小向上取整分配 | 资源不足不留下半分配 |
| 解码追加 | 新处理的位置 | 尾块满后再申请新块 | 失败不增加有效长度 |
| 释放 | 不再使用的请求 | 清空映射并归还专有块 | 重复释放应报错 |
| 再分配 | 新请求 | 使用池内空闲物理块 | 物理编号不要求连续 |

## Step 3： Block Table 如何把逻辑访问映射到物理缓存

分页缓存需要把逻辑 token 顺序与非连续物理空间对应起来。物理 Block 池保存缓存，映射表保存逻辑位置到物理位置的关系，分配器负责按需取得和释放物理 Block；例如块大小为四、有效长度为六，即使块表是 `[7,2]`，前四个位置也来自物理块 7，后两个位置来自物理块 2；不能按物理编号排序。这里描述每请求独占块，跨请求共享还要增加引用管理。

| 数据结构 | 作用 | 应保持的不变量 |
|---|---|---|
| 物理 Block 池 | 保存物理 KV Block | 一个物理索引对应一个池内 Block |
| 逻辑—物理映射表 | 记录逻辑 Block 到物理 Block 的映射 | 本页独占分配中不出现无效或重复索引 |
| 空闲 Block 集合 | 管理尚未分配的物理 Block | 已分配 Block 不应同时出现在其中 |
| 逻辑序列长度 | 判断尾块和 Decode 扩容边界 | 与当前逻辑 token 数一致 |

![PagedAttention 块表图](../public/02_PyTorch_Algorithms/22_paged_attention_blocks.svg)

## Step 4： 实现并验证分页缓存管理器

本题用 CPU 张量模拟固定块池；理论容量函数计入完整 K/V 维度，存储池则只保留用于检查位置顺序的二维教学数据。你需要根据容量、边界和逻辑顺序，完成 Prefill 分配、Decode 跨块扩容、逻辑缓存恢复；释放骨架已提供，并通过测试检查状态是否保持一致。

| TODO / 函数 | 学习者完成的机制 | 必须满足的约束 | 测试证据 |
|---|---|---|---|
| TODO 1<br>`estimate_kv_cache_bytes` | 计算单 token、单 Block 和总容量 | 同时计入 K/V、层数、KV Head 和数据类型 | 容量账本测试 |
| TODO 2<br>`allocate_for_prefill` | 向上取整并选择待提交物理块 | OOM 与重复分配不改变请求和空闲池 | Prefill 与原子 OOM 测试 |
| TODO 3<br>`allocate_for_decode` | 识别追加 token 是否跨越 Block 边界 | 仅跨界时申请；失败时长度和块表不变 | Decode 边界与回滚测试 |
| TODO 4<br>`get_physical_cache` | 按逻辑映射表恢复缓存顺序 | 不按物理索引排序，并截断尾块 | 非连续物理块恢复测试 |



```python
import torch
from typing import Dict, List, Tuple
```


```python
def estimate_kv_cache_bytes(num_blocks: int, block_size: int, num_layers: int, num_kv_heads: int, head_dim: int, dtype_bytes: int = 2) -> Dict[str, int]:
    """估算 KV Cache 理论容量；不代表 vLLM allocator 的实际峰值。"""
    values = (num_blocks, block_size, num_layers, num_kv_heads, head_dim, dtype_bytes)
    if any(value <= 0 for value in values):
        raise ValueError('所有容量维度和 dtype_bytes 都必须为正数')
    # TODO 1（容量账本）：依次计算单 token、单 Block 和整个池的理论容量。
    # kv_bytes_per_token = ???
    # kv_bytes_per_block = ???
    # total_cache_bytes = ???
    raise NotImplementedError('TODO 1：请完成 KV Cache 容量账本')
    layout_shape = [num_layers, 2, num_kv_heads, block_size, head_dim]
    return {'kv_bytes_per_token': kv_bytes_per_token, 'kv_bytes_per_block': kv_bytes_per_block, 'total_cache_bytes': total_cache_bytes, 'layout_shape': layout_shape}


class Request:
    """记录逻辑序列长度，以及逻辑 Block 到物理 Block 的映射。

    `block_table[i]` 是第 i 个逻辑 Block 对应的物理索引；物理索引不要求连续。"""

    def __init__(self, request_id: int, prompt_len: int):
        if request_id < 0 or prompt_len <= 0:
            raise ValueError('request_id 必须非负，prompt_len 必须为正数')
        self.request_id = request_id
        self.seq_len = prompt_len
        # 记录此请求占据的物理 Block 索引
        self.block_table: List[int] = []

class KVCacheManager:
    """用空闲块列表模拟 KV Cache 的分配、扩容、释放和恢复。

    这里的物理池只保存教学用张量，不等同于 vLLM 的真实 KV Cache layout。"""
    def __init__(self, num_blocks: int, block_size: int, head_dim: int):
        if num_blocks <= 0 or block_size <= 0 or head_dim <= 0:
            raise ValueError('num_blocks、block_size 和 head_dim 必须为正数')
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.head_dim = head_dim

        # 模拟预分配一块大显存池 (vLLM 会在 GPU 上分配几 GB)
        # 形状: [num_blocks, block_size, head_dim]
        self.physical_kv_cache = torch.zeros(num_blocks, block_size, head_dim)

        # 跟踪哪些物理块被占用了
        self.free_blocks: List[int] = list(range(num_blocks))
    def allocate_for_prefill(self, req: Request):
        """
        请求刚进来时 (Prefill阶段)，为它的 Prompt 长度分配所需的全部 Block
        """
        if req.block_table:
            raise ValueError('请求已经完成 Prefill，不能重复分配')
        # TODO 2（Prefill 分配）：先计算目标块数，再选择待提交的物理块。
        # needed_blocks = ???
        # allocated_blocks = ???
        raise NotImplementedError('TODO 2：请完成 Prefill 原子分配')
        if len(self.free_blocks) < needed_blocks:
            raise RuntimeError('OOM')
        del self.free_blocks[:needed_blocks]
        req.block_table.extend(allocated_blocks)

    def allocate_for_decode(self, req: Request):
        """
        自回归生成时 (Decode阶段)，检查序列长度。
        如果当前最后一个 Block 满了，则按需分配 1 个新 Block。
        """
        # TODO 3（Decode 扩容）：计算追加后的长度，并判断是否进入新 Block。
        # new_seq_len = ???
        # is_new_block_needed = ???
        raise NotImplementedError('TODO 3：请判断 Decode 是否跨越 Block 边界')
        if is_new_block_needed:
            if not self.free_blocks:
                raise RuntimeError('OOM')
            req.block_table.append(self.free_blocks.pop(0))
        req.seq_len = new_seq_len

    def allocation_report(self, req: Request) -> Dict[str, float]:
        """报告逻辑 token、已分配 token、尾块浪费和利用率。"""
        allocated_tokens = len(req.block_table) * self.block_size
        unused_tail_tokens = allocated_tokens - req.seq_len
        utilization = req.seq_len / allocated_tokens if allocated_tokens else 0.0
        return {'logical_tokens': req.seq_len, 'allocated_tokens': allocated_tokens, 'unused_tail_tokens': unused_tail_tokens, 'utilization': utilization}

    def release_request(self, req: Request):
        """释放请求占用的物理块，并清空其块表。"""
        released_block_ids = list(req.block_table)
        if not released_block_ids:
            raise ValueError('请求没有可释放的物理块，可能已经释放')
        if len(set(released_block_ids)) != len(released_block_ids):
            raise ValueError('block_table 不能包含重复物理块')
        if any(block_id < 0 or block_id >= self.num_blocks for block_id in released_block_ids):
            raise ValueError('请求包含越界物理块')
        if any(block_id in self.free_blocks for block_id in released_block_ids):
            raise ValueError('请求包含已释放的物理块')
        self.free_blocks.extend(released_block_ids)
        self.free_blocks.sort()
        req.block_table.clear()

    def get_physical_cache(self, req: Request) -> torch.Tensor:
        """根据块表恢复逻辑连续的 KV Cache。"""
        if not req.block_table:
            raise ValueError('请求没有可读取的物理块')
        if any(block_id < 0 or block_id >= self.num_blocks for block_id in req.block_table):
            raise ValueError('block_table 包含越界物理块')
        # TODO 4（逻辑恢复）：按 Block Table 顺序读取并沿 token 维拼接。
        # blocks = ???
        # cat_blocks = ???
        raise NotImplementedError('TODO 4：请恢复逻辑连续的 KV Cache')
        return cat_blocks[:req.seq_len]


```

## 测试

运行下面的测试单元，确认 prefill / decode / cache 拼装三段链路都正确。

```python
# 机制测试：容量、跨块扩容、逻辑映射、原子失败与释放。
def test_request_and_ledger_contract():
    """验证请求输入契约与 KV Cache 容量账本。"""
    for bad_request in ((-1, 2), (1, 0)):
        try:
            Request(request_id=bad_request[0], prompt_len=bad_request[1])
        except ValueError:
            continue
        raise AssertionError('非法请求参数应被拒绝')
    ledger = estimate_kv_cache_bytes(4, 4, 2, 8, 16, 2)
    assert ledger['kv_bytes_per_token'] == 1024
    assert ledger['total_cache_bytes'] == 16384

def test_prefill_and_decode_growth():
    """验证 Prefill 分配和 Decode 跨 Block 边界扩容。"""
    manager = KVCacheManager(num_blocks=4, block_size=4, head_dim=8)
    req = Request(request_id=1, prompt_len=4)
    manager.allocate_for_prefill(req)
    assert len(req.block_table) == 1
    manager.allocate_for_decode(req)
    assert req.seq_len == 5 and len(req.block_table) == 2

def test_logical_cache_reconstruction():
    """验证非连续物理块按逻辑 Block 顺序恢复。"""
    manager = KVCacheManager(num_blocks=4, block_size=2, head_dim=2)
    req = Request(request_id=2, prompt_len=3)
    req.block_table = [3, 1]
    manager.physical_kv_cache[3].fill_(3.0)
    manager.physical_kv_cache[1].fill_(1.0)
    cache = manager.get_physical_cache(req)
    assert cache.shape == (3, 2)
    assert torch.all(cache[:2] == 3.0) and torch.all(cache[2:] == 1.0)

def test_oom_is_atomic():
    """验证 Prefill OOM 不会留下半分配状态。"""
    manager = KVCacheManager(num_blocks=1, block_size=4, head_dim=8)
    req = Request(request_id=3, prompt_len=5)
    free_before = list(manager.free_blocks)
    try:
        manager.allocate_for_prefill(req)
    except RuntimeError as error:
        assert 'OOM' in str(error)
    else:
        raise AssertionError('资源不足时应抛出 OOM')
    assert req.block_table == [] and manager.free_blocks == free_before

def test_release_and_reuse():
    """验证尾块账本、释放和物理 Block 复用。"""
    manager = KVCacheManager(num_blocks=3, block_size=4, head_dim=8)
    req = Request(request_id=4, prompt_len=5)
    manager.allocate_for_prefill(req)
    report = manager.allocation_report(req)
    assert report['unused_tail_tokens'] == 3
    manager.release_request(req)
    assert req.block_table == [] and manager.free_blocks == [0, 1, 2]

def test_single_token_blocks_and_decode_rollback():
    """块大小为一时每个位置都扩容；OOM 不改变长度与映射。"""
    manager = KVCacheManager(num_blocks=2, block_size=1, head_dim=2)
    req = Request(request_id=6, prompt_len=1)
    manager.allocate_for_prefill(req)
    manager.allocate_for_decode(req)
    assert req.seq_len == 2 and len(req.block_table) == 2
    before = (req.seq_len, list(req.block_table), list(manager.free_blocks))
    try:
        manager.allocate_for_decode(req)
    except RuntimeError:
        pass
    else:
        raise AssertionError("跨块扩容应报告 OOM")
    assert (req.seq_len, req.block_table, manager.free_blocks) == before


def test_paged_attention_manager():
    """汇总运行分页缓存的容量、扩容、映射、OOM 与释放测试。"""
    test_request_and_ledger_contract()
    test_prefill_and_decode_growth()
    test_logical_cache_reconstruction()
    test_oom_is_atomic()
    test_release_and_reuse()
    test_single_token_blocks_and_decode_rollback()
    print('✅ PagedAttention 机制测试通过：账本、扩容、映射、OOM 和回收均通过。')


test_paged_attention_manager()

```

---

🛑 **STOP HERE**

先完成题目并运行测试，再展开参考代码。

<br><br><br><br><br><br><br><br><br><br>
## 参考代码与解析

### 代码

```python
class Request:
    """记录逻辑序列长度，以及逻辑 Block 到物理 Block 的映射。

    `block_table[i]` 是第 i 个逻辑 Block 对应的物理索引；物理索引不要求连续。"""

    def __init__(self, request_id: int, prompt_len: int):
        if request_id < 0 or prompt_len <= 0:
            raise ValueError('request_id 必须非负，prompt_len 必须为正数')
        self.request_id = request_id
        self.seq_len = prompt_len
        self.block_table: List[int] = []

def estimate_kv_cache_bytes(num_blocks: int, block_size: int, num_layers: int, num_kv_heads: int, head_dim: int, dtype_bytes: int = 2) -> Dict[str, int]:
    """估算 KV Cache 理论容量；不代表 vLLM allocator 的实际峰值。"""
    values = (num_blocks, block_size, num_layers, num_kv_heads, head_dim, dtype_bytes)
    if any(value <= 0 for value in values):
        raise ValueError('所有容量维度和 dtype_bytes 都必须为正数')
    kv_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
    kv_bytes_per_block = block_size * kv_bytes_per_token
    total_cache_bytes = num_blocks * kv_bytes_per_block
    layout_shape = [num_layers, 2, num_kv_heads, block_size, head_dim]
    return {'kv_bytes_per_token': kv_bytes_per_token, 'kv_bytes_per_block': kv_bytes_per_block, 'total_cache_bytes': total_cache_bytes, 'layout_shape': layout_shape}


class KVCacheManager:
    """用空闲块列表模拟 KV Cache 的分配、扩容、释放和恢复。

    这里的物理池只保存教学用张量，不等同于 vLLM 的真实 KV Cache layout。"""
    def __init__(self, num_blocks: int, block_size: int, head_dim: int):
        if num_blocks <= 0 or block_size <= 0 or head_dim <= 0:
            raise ValueError('num_blocks、block_size 和 head_dim 必须为正数')
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.head_dim = head_dim

        # 给定实现：模拟预分配一块大显存池
        self.physical_kv_cache = torch.zeros(num_blocks, block_size, head_dim)

        # 跟踪哪些物理块被占用了
        self.free_blocks: List[int] = list(range(num_blocks))
    def allocate_for_prefill(self, req: Request):
        """
        请求刚进来时 (Prefill阶段)，为它的 Prompt 长度分配所需的全部 Block
        """
        if req.block_table:
            raise ValueError('请求已经完成 Prefill，不能重复分配')
        # TODO 2（Prefill 分配）：先计算目标块数，再选择待提交的物理块。
        needed_blocks = (req.seq_len + self.block_size - 1) // self.block_size

        if len(self.free_blocks) < needed_blocks:
            raise RuntimeError("OOM")

        allocated_blocks = self.free_blocks[:needed_blocks]
        del self.free_blocks[:needed_blocks]
        req.block_table.extend(allocated_blocks)

    def allocate_for_decode(self, req: Request):
        """
        自回归生成时 (Decode阶段)，检查序列长度。
        如果当前最后一个 Block 满了，则按需分配 1 个新 Block。
        """
        if not req.block_table:
            raise ValueError("Decode 追加前必须先分配请求块")
        new_seq_len = req.seq_len + 1

        # TODO 3（Decode 扩容）：计算追加后的长度，并判断是否进入新 Block。
        is_new_block_needed = (new_seq_len + self.block_size - 1) // self.block_size > len(req.block_table)

        if is_new_block_needed:
            if not self.free_blocks:
                raise RuntimeError("OOM")
            block_id = self.free_blocks.pop(0)
            req.block_table.append(block_id)
        req.seq_len = new_seq_len

    def allocation_report(self, req: Request) -> Dict[str, float]:
        """报告有效长度、分配位置数、尾块余量与利用率。"""
        allocated_tokens = len(req.block_table) * self.block_size
        unused_tail_tokens = allocated_tokens - req.seq_len
        utilization = req.seq_len / allocated_tokens if allocated_tokens else 0.0
        return {'logical_tokens': req.seq_len, 'allocated_tokens': allocated_tokens, 'unused_tail_tokens': unused_tail_tokens, 'utilization': utilization}

    def release_request(self, req: Request):
        """归还请求专有块；拒绝无效索引和重复释放。"""
        released_block_ids = list(req.block_table)
        if not released_block_ids:
            raise ValueError('请求没有可释放的物理块，可能已经释放')
        if len(set(released_block_ids)) != len(released_block_ids):
            raise ValueError('block_table 不能包含重复物理块')
        if any(block_id < 0 or block_id >= self.num_blocks for block_id in released_block_ids):
            raise ValueError('请求包含越界物理块')
        if any(block_id in self.free_blocks for block_id in released_block_ids):
            raise ValueError('请求包含已释放的物理块')
        self.free_blocks.extend(released_block_ids)
        self.free_blocks.sort()
        req.block_table.clear()

    def get_physical_cache(self, req: Request) -> torch.Tensor:
        """根据块表恢复逻辑连续的 KV Cache。"""
        # TODO 4（逻辑恢复）：按 Block Table 顺序读取并沿 token 维拼接。
        if not req.block_table:
            raise ValueError('请求没有可读取的物理块')
        if any(block_id < 0 or block_id >= self.num_blocks for block_id in req.block_table):
            raise ValueError('block_table 包含越界物理块')
        blocks = [self.physical_kv_cache[block_id] for block_id in req.block_table]
        cat_blocks = torch.cat(blocks, dim=0)

        # 只截取真实 seq_len 长度返回
        return cat_blocks[:req.seq_len]


```

### 解析

#### TODO 1：KV Cache 容量账本

- 单 token 的容量同时包含 K 和 V，再乘以层数、KV Head 数、Head 维度和元素字节数。单 token、单 Block 和整个物理池的容量应逐级相等。
- 容量账本测试检查三个层级的结果。

#### TODO 2：Prefill 原子分配

- 请求长度按 Block 大小向上取整，再一次性取得所需物理 Block。容量不足时，请求映射和空闲 Block 集合都不能留下半分配状态。
- Prefill 分配和原子 OOM 测试检查成功与失败两条路径。

#### TODO 3：Decode 跨块扩容

- 追加 token 后只有跨过当前 Block 边界时才申请新的物理 Block。普通追加不改变 Block 数；扩容失败时长度和映射保持不变。
- 尾块内追加、跨边界追加和无空闲 Block 测试分别覆盖这些条件。

#### TODO 4：恢复逻辑缓存

- 按照逻辑映射表的顺序读取物理 Block，再截断到真实逻辑序列长度。物理索引可以不连续，但恢复后的 token 顺序必须连续。
- 非连续物理映射和尾块截断测试检查恢复结果。
## 相关阅读

继续观察共享引用如何限制回收，以及前缀索引怎样找到可复用状态。

- [Efficient Memory Management for Large Language Model Serving with PagedAttention 原论文](https://arxiv.org/abs/2309.06180)
- [vLLM 官方仓库](https://github.com/vllm-project/vllm)
- [vLLM 官方文档](https://docs.vllm.ai/en/latest/)
- [24. Prefix Cache Matching and Reuse | Prefix Cache 匹配与复用](./24_Prefix_Cache_Matching_and_Reuse.md)
- [34. SGLang RadixAttention | SGLang 基数注意力](./34_SGLang_RadixAttention.md)
- [37. KV Cache Lifecycle and Capacity Control | KV Cache 生命周期与容量控制](./37_KV_Cache_Lifecycle_and_Capacity_Control.md)
