# 22. vLLM PagedAttention | vLLM 分页注意力
**难度：** Hard | **环境：** CPU-first | **标签：** `推理优化`, `KV Cache`, `PagedAttention` | **目标人群：** 推理优化学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/22_vLLM_PagedAttention.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

在线推理服务里，请求不会整齐地一起开始、一起结束：有的 prompt 很短，有的上下文很长，有的很快生成完，有的还在继续 decode。如果仍然按最大长度给每个请求预留整块 KV Cache，显存会被大量浪费，GPU 也会因为静态 batch 里的空洞而等待。

PagedAttention 的思路是把 KV Cache 像分页内存一样管理：物理显存切成固定大小的 block，每个请求只维护一张逻辑到物理的 block table。prefill 时按需申请，decode 跨块时再补新 block，真正计算时再按表拼回逻辑连续的缓存。本节会用一个简化管理器模拟这条链路，看清 vLLM 为什么能支撑高吞吐在线 serving。

**关键词：** `PagedAttention`, `KV cache`, `block table`

---

## 前置阅读

**导语：** 先理解单步解码如何读取 KV Cache，再把连续缓存和分页 block 的管理方式放到同一条请求链路中比较。
- [21. Decoding Strategies | 解码策略](./21_Decoding_Strategies.md)
- [P1: 11. KV Cache and Memory Growth | KV Cache 与显存增长](../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md)
- [20. FlashAttention Sim | FlashAttention 模拟](./20_FlashAttention_Sim.md)

---

### Step 1: PagedAttention 如何组织逻辑与物理 KV Cache
在线请求的长度和结束时间并不相同，如果每个请求都按最大长度预留连续 KV Cache，就会同时产生未使用尾部和难以复用的空闲空间。PagedAttention 的输入是请求 token 与缓存预算，处理方式是按需分配固定大小的物理 Block，并用 Block Table 记录逻辑位置到物理 Block 的映射；输出是可继续增长、可回收的缓存布局。
先通过表格建立连续分配、固定 Block 和分页映射的整体关系，再观察主图中的逻辑访问路径。

| 管理方式 | 如何分配 KV Cache | 主要问题或收益 |
|---|---|---|
| 连续预留 | 按 `max_len` 一次分配连续空间 | 长度未知时产生尾部浪费和空洞 |
| 虚拟内存类比 | 用逻辑位置访问物理存储 | 逻辑地址与物理地址可以分离 |
| 固定 Block | 每个 Block 容纳固定数量 token，例如 16 个 | 分配粒度统一，便于回收和复用 |
| 分页管理 | 按需分配固定大小的物理 Block | 物理块可以不连续，空间更容易复用 |
| Block Table | 记录逻辑 Block 到物理 Block 的映射 | 计算时仍能按逻辑 token 顺序访问缓存 |

![PagedAttention 的 KV Cache 管理总览](../public/02_PyTorch_Algorithms/22_paged_attention_overview.svg)

### Step 2: 物理 Block 如何分配、扩容与释放
`BlockTable` 的状态会随着请求推进而变化：prefill 建立初始映射，decode 只在跨过边界时扩容，请求完成后释放物理块。沿着这个生命周期观察“请求长度变化”如何转化为“物理块状态变化”。

| 阶段 | 输入 | 状态变化 | 需要检查的边界 |
|---|---|---|---|
| Prefill | prompt 长度 | 申请多个物理 Block | 空闲 Block 是否足够 |
| Decode | 新增 token | 跨边界时追加 Block | 扩容失败时状态是否回滚 |
| Release | 已完成请求 | 归还 Block 并清空块表 | 是否重复释放 |
| Reuse | 新请求 | 使用已释放的物理 Block | 物理 Block 是否允许不连续 |

### Step 3: Block Table 如何把逻辑访问映射到物理缓存

分页缓存需要把逻辑 token 顺序与非连续物理空间对应起来。物理池保存 Block，Block Table 保存逻辑到物理的映射，管理器负责分配、释放和按需扩容；下面的表格用不变量说明三者如何协作。

| 数据结构 | 作用 | 应保持的不变量 |
|---|---|---|
| `physical_pool` | 保存物理 KV Block | 一个物理索引对应一个池内 Block |
| `block_table` | 逻辑 Block 到物理 Block 的映射 | 不应出现无效或重复的物理索引 |
| `free_blocks` | 管理尚未分配的物理 Block | 已分配 Block 不应同时出现在其中 |
| `seq_len` | 判断尾块和 decode 扩容边界 | 与当前逻辑 token 数一致 |

![PagedAttention 块表图](../public/02_PyTorch_Algorithms/22_paged_attention_blocks.svg)

### Step 4: 实现并验证分页缓存管理器

本 Step 把前面的布局和生命周期落到 `KVCacheManager`。输入是 Block 容量、请求长度和逻辑 token，输出是分配状态、Block Table 和恢复后的逻辑缓存。题目区先完成理论容量、Block 分配与回收、跨边界扩容和逻辑缓存拼装；`acquire_prefix` / `release_prefix` 的 Prefix Cache 共享只是可选扩展，不属于核心 TODO。

| 实现对象 | 需要完成的机制 | 验证重点 |
|---|---|---|
| 输入契约 | 校验请求长度、物理块数量、Block 大小和 `head_dim` 为正数 | 非法参数应尽早拒绝，不污染缓存状态 |
| 容量账本 | 计算单 token、单 Block 和总 KV Cache 容量 | K/V、层数、KV heads 和 dtype 字节数 |
| Block 生命周期 | 实现 prefill 分配、decode 扩容、release 回收 | OOM 时状态不被部分修改 |
| 逻辑缓存读取 | 按 Block Table 恢复逻辑 token 顺序 | 物理块不连续时仍能正确拼装 |


```python
import torch
from typing import Dict, List, Tuple
```


```python
# TODO 1：计算 KV Cache 理论容量
def estimate_kv_cache_bytes(num_blocks: int, block_size: int, num_layers: int, num_kv_heads: int, head_dim: int, dtype_bytes: int = 2) -> Dict[str, int]:
    """估算 KV Cache 理论容量；不代表 vLLM allocator 的实际峰值。"""
    # TODO 1：计算单 token 的 K/V 字节数、单 block 字节数和总容量。
    # kv_bytes_per_token = ???  # 单 token 的 K/V 字节数
    # kv_bytes_per_block = ???  # 一个物理 Block 的 K/V 字节数
    # total_cache_bytes = ???  # 全部物理 Block 的理论容量
    # layout_shape 使用 [num_layers, 2(K/V), num_kv_heads, block_size, head_dim]，这里作为给定的布局描述。
    layout_shape = [num_layers, 2, num_kv_heads, block_size, head_dim]
    raise NotImplementedError("请先完成 TODO 代码！")


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
        # 可选扩展：prefix_key -> {'block_ids': [...], 'refcount': n}
        self.prefix_cache: Dict[Tuple[int, ...], dict] = {}

    def acquire_prefix(self, prefix_tokens: List[int]) -> List[int]:
        """可选扩展：为相同 token 前缀共享物理 Block。"""
        # 扩展 A（Prefix Cache）：规范化 prefix_key，命中时增加引用计数；
        # 未命中时按需申请 Block，并记录 block_ids 与 refcount。
        raise NotImplementedError("请先完成 Prefix Cache 扩展！")

    def release_prefix(self, prefix_tokens: List[int]):
        # 扩展 B（Prefix Cache）：引用计数归零后归还物理 Block。
        raise NotImplementedError("请先完成 Prefix Cache 扩展！")

    def allocate_for_prefill(self, req: Request):
        """
        请求刚进来时 (Prefill阶段)，为它的 Prompt 长度分配所需的全部 Block
        """
        # ==========================================
        # TODO 2: 计算需要的 block 数量
        # 提示：按向上取整规则计算需要的 Block 数量。
        # 已有 block_table 的请求不能重复 Prefill；prompt_len 必须为正数。
        # 如果 block_table 非空，应明确拒绝重复分配，而不是继续追加。
        # 先计算目标状态，再检查资源；失败时 free_blocks 和 block_table 都不能改变。
        # 这是一个状态转换：先计算并检查资源，再一次性提交块表。
        # ==========================================
        # needed_blocks = ???
        
        # ==========================================
        # TODO 3: 从 free_blocks 中弹出对应数量的 block 索引，
        # 并追加到请求的 block_table 中
        # 如果 free_blocks 不够了，抛出 RuntimeError("OOM")
        # ==========================================
        # allocated_blocks = ???  # 本次申请的物理 Block 列表；按顺序提交到 req.block_table
        # 先确认空闲块足够，再修改 free_blocks 和 block_table，避免 OOM 后半分配。
        # 物理块可以不连续，但逻辑顺序必须与 block_table 顺序一致。
        pass

    def allocate_for_decode(self, req: Request):
        """
        自回归生成时 (Decode阶段)，检查序列长度。
        如果当前最后一个 Block 满了，则按需分配 1 个新 Block。
        """
        # ==========================================
        # TODO 4: 判断是否刚好需要跨入新的一块 Block？
        # 条件：加 1 后的 seq_len 除以 block_size 余数是多少？
        # ==========================================
        # new_seq_len 由 req.seq_len 加 1 得到；请填写 is_new_block_needed = ??? 以判断是否跨入新物理 Block。
        # 先检查资源，再提交 seq_len 和 block_table；OOM 时请求状态必须保持不变。
        
        # 如果需要，尝试分配 1 个新的物理 Block 放入块表
        # if is_new_block_needed:
        #    if not self.free_blocks: ...
        #    从 free_blocks 取出一个物理索引并追加到 req.block_table。
        # 注意：OOM 时不能修改 req.seq_len 或 req.block_table。
        pass

    # TODO 5：计算请求的 Block 占用报告
    def allocation_report(self, req: Request) -> Dict[str, float]:
        """报告逻辑 token、已分配 token、尾块浪费和利用率。"""
        # TODO 5：分别计算以下变量：
        # allocated_tokens = ???  # 已分配物理块可容纳的 token 数
        # unused_tail_tokens = ???  # 尾块尚未使用的 token 数
        # utilization = ???  # 逻辑 token / 已分配容量
        raise NotImplementedError("请先完成 TODO 代码！")

    def release_request(self, req: Request):
        """释放请求占用的物理块，并清空其块表。"""
        # TODO 6：先记录并校验 released_block_ids，再归还 free_blocks。
        # released_block_ids = ???  # 本次释放的物理 Block 列表；释放后空闲池数量可直接由 free_blocks 得到
        # 重复释放应报错或明确拒绝，不能把同一物理块加入 free_blocks 两次。
        # 同时检查物理索引范围，避免非法状态污染空闲池。
        pass

    # TODO 7：恢复逻辑连续 Cache
    def get_physical_cache(self, req: Request) -> torch.Tensor:
        """根据块表恢复逻辑连续的 KV Cache。"""
        # TODO 7：只读取 req.block_table 中的物理块，并截断到 req.seq_len。
        # ==========================================
        # blocks = ???  # 按 block_table 顺序提取物理块
        # cat_blocks = ???  # 沿 token 维拼接物理块，并在返回时截断到 req.seq_len
        # 必须按 block_table 顺序读取并截断尾块，不能按物理索引排序。
        return cat_blocks[:req.seq_len]


```

### 测试

运行下面的测试单元，确认 prefill / decode / cache 拼装三段链路都正确。

```python
# 运行此单元格以测试你的实现
def test_paged_attention_manager():
    try:
        # Case 1: 典型 Prefill + Decode + Cache 拼装
        manager = KVCacheManager(num_blocks=10, block_size=4, head_dim=64)
        print("初始化内存池...")

        for bad_request in ((-1, 2), (1, 0)):
            try:
                Request(request_id=bad_request[0], prompt_len=bad_request[1])
                raise AssertionError('非法请求参数未被拒绝')
            except ValueError:
                pass
        print("✅ Request 输入校验通过！")

        req1 = Request(request_id=1, prompt_len=6)
        manager.allocate_for_prefill(req1)
        assert len(req1.block_table) == 2, "长度 6 的请求应分配 2 个 Block！"
        assert len(manager.free_blocks) == 8, "池中应该剩下 8 个空闲块！"
        print(f"✅ Prefill 测试通过！Req1 分配的块表: {req1.block_table}")

        try:
            manager.allocate_for_prefill(req1)
        except ValueError:
            print("✅ 重复 Prefill 已拒绝！")
        else:
            raise AssertionError('同一请求不能重复 Prefill 并追加 Block')

        manager.allocate_for_decode(req1)
        assert len(req1.block_table) == 2, "生成第 7 个 token 时不应该分配新块！"

        manager.allocate_for_decode(req1)
        manager.allocate_for_decode(req1)
        assert len(req1.block_table) == 3, "生成第 9 个 token 时应当分配了第 3 个新块！"
        assert len(manager.free_blocks) == 7, "池中应该剩下 7 个空闲块！"
        print(f"✅ Decode 动态分配测试通过！Req1 最新块表: {req1.block_table}")

        for block_id, value in zip(req1.block_table, [1.0, 2.0, 3.0]):
            manager.physical_kv_cache[block_id].fill_(value)
        cache = manager.get_physical_cache(req1)
        assert cache.shape == (9, 64), f"拼装出来的连续 Cache 形状不对，应为 (9, 64)，实为 {cache.shape}"
        assert torch.all(cache[:4] == 1.0), "第 1 个 Block 未正确拼装！"
        assert torch.all(cache[4:8] == 2.0), "第 2 个 Block 未正确拼装！"
        assert torch.all(cache[8:] == 3.0), "第 3 个 Block 的截断拼装不正确！"
        print("✅ Cache 拼装测试通过！多块物理缓存被正确恢复为逻辑连续序列。")

        # Case 1b: 物理块可以不连续，但恢复必须遵循逻辑 Block 顺序
        scattered_manager = KVCacheManager(num_blocks=4, block_size=2, head_dim=2)
        scattered_req = Request(request_id=10, prompt_len=3)
        scattered_req.block_table = [3, 1]
        scattered_manager.physical_kv_cache[3].fill_(3.0)
        scattered_manager.physical_kv_cache[1].fill_(1.0)
        scattered_cache = scattered_manager.get_physical_cache(scattered_req)
        assert torch.all(scattered_cache[:2] == 3.0)
        assert torch.all(scattered_cache[2:] == 1.0)
        print("✅ 非连续物理块按逻辑顺序恢复通过！")

        # Case 2: 恰好跨越 block 边界时，Decode 应该分配新块，并正确截断最后一块
        manager2 = KVCacheManager(num_blocks=4, block_size=4, head_dim=8)
        req2 = Request(request_id=2, prompt_len=4)
        manager2.allocate_for_prefill(req2)
        assert len(req2.block_table) == 1, "长度 4 的请求应只分配 1 个 Block！"
        manager2.allocate_for_decode(req2)
        assert len(req2.block_table) == 2, "长度 5 的请求应分配第 2 个 Block！"
        manager2.physical_kv_cache[req2.block_table[0]].fill_(7.0)
        manager2.physical_kv_cache[req2.block_table[1]].fill_(8.0)
        cache2 = manager2.get_physical_cache(req2)
        assert cache2.shape == (5, 8), f"拼装出来的连续 Cache 形状不对，应为 (5, 8)，实为 {cache2.shape}"
        assert torch.all(cache2[:4] == 7.0), "边界块的前 4 个 token 不正确！"
        assert torch.all(cache2[4:] == 8.0), "边界块的最后 1 个 token 不正确！"
        print("✅ 边界分配与截断测试通过！")

        # Case 3: OOM 分支必须抛出 RuntimeError
        oom_manager = KVCacheManager(num_blocks=1, block_size=4, head_dim=8)
        oom_req = Request(request_id=3, prompt_len=5)
        try:
            oom_manager.allocate_for_prefill(oom_req)
        except RuntimeError as e:
            assert "OOM" in str(e), "OOM 异常信息不正确！"
            print("✅ OOM 测试通过！")
        else:
            raise AssertionError('显存池不足时应该抛出 RuntimeError("OOM")！')
        # Case 4: Prefill OOM 不应留下半分配状态
        atomic_manager = KVCacheManager(num_blocks=2, block_size=4, head_dim=8)
        atomic_req = Request(request_id=4, prompt_len=9)
        free_before = list(atomic_manager.free_blocks)
        try:
            atomic_manager.allocate_for_prefill(atomic_req)
        except RuntimeError as e:
            assert 'OOM' in str(e), '应明确报告 OOM'
            assert atomic_req.block_table == [], 'OOM 后请求不应保留半分配块'
            assert atomic_manager.free_blocks == free_before, 'OOM 后空闲池不应被部分消耗'
            print('✅ Prefill 原子 OOM 测试通过！')
        else:
            raise AssertionError('资源不足时必须抛出 OOM')

        ledger = estimate_kv_cache_bytes(num_blocks=4, block_size=4, num_layers=2, num_kv_heads=8, head_dim=16, dtype_bytes=2)
        assert ledger['kv_bytes_per_token'] == 1024, 'K/V 两份缓存账本计算不正确'
        assert ledger['total_cache_bytes'] == 16384, 'Block 池总容量计算不正确'
        reusable_manager = KVCacheManager(num_blocks=3, block_size=4, head_dim=8)
        reusable_req = Request(request_id=5, prompt_len=5)
        reusable_manager.allocate_for_prefill(reusable_req)
        allocated_before_release = list(reusable_req.block_table)
        report = reusable_manager.allocation_report(reusable_req)
        assert report['allocated_tokens'] == 8, '已分配 token 账本不正确'
        assert report['unused_tail_tokens'] == 3, '尾块浪费计算不正确'
        assert round(report['utilization'], 3) == 0.625, 'Block 利用率计算不正确'
        reusable_manager.release_request(reusable_req)
        assert reusable_req.block_table == [], '释放后请求块表应为空'
        assert reusable_manager.free_blocks == [0, 1, 2], '释放后的 Block 应可复用'
        try:
            reusable_manager.release_request(reusable_req)
        except ValueError:
            pass
        else:
            raise AssertionError('重复释放请求应明确报错')
        rollback_manager = KVCacheManager(num_blocks=1, block_size=4, head_dim=8)
        rollback_req = Request(request_id=6, prompt_len=4)
        rollback_manager.allocate_for_prefill(rollback_req)
        old_seq_len = rollback_req.seq_len
        try:
            rollback_manager.allocate_for_decode(rollback_req)
        except RuntimeError:
            assert rollback_req.seq_len == old_seq_len, 'Decode OOM 后 seq_len 不应改变'
            print('✅ Decode OOM 回滚测试通过！')
        else:
            raise AssertionError('Decode 跨块时应触发 OOM')
        print('✅ KV Cache 账本与释放复用测试通过！')

        print("\n✅ All Tests Passed! PagedAttention 内存管理逻辑验证通过。")

    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
        raise
    except Exception as e:
        # 只把 TODO 未完成单独提示；断言、类型和状态错误保留原始异常，便于定位。
        print(f"❌ 测试失败: {type(e).__name__}: {e}")
        raise


test_paged_attention_manager()

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
class Request:
    """记录逻辑序列长度，以及逻辑 Block 到物理 Block 的映射。

    `block_table[i]` 是第 i 个逻辑 Block 对应的物理索引；物理索引不要求连续。"""

    def __init__(self, request_id: int, prompt_len: int):
        if request_id < 0 or prompt_len <= 0:
            raise ValueError('request_id 必须非负，prompt_len 必须为正数')
        self.request_id = request_id
        self.seq_len = prompt_len
        self.block_table: List[int] = []

# TODO 1：计算 KV Cache 理论容量
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
        # prefix_key -> {'block_ids': [...], 'refcount': n}
        self.prefix_cache: Dict[Tuple[int, ...], dict] = {}

    def acquire_prefix(self, prefix_tokens: List[int]) -> List[int]:
        """为相同 token 前缀共享物理 Block；这是简化的引用计数模型。"""
        key = tuple(int(token) for token in prefix_tokens)
        if not key:
            raise ValueError('prefix_tokens 不能为空')
        entry = self.prefix_cache.get(key)
        if entry is not None:
            entry['refcount'] += 1
            return list(entry['block_ids'])
        needed = (len(key) + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < needed:
            raise RuntimeError('OOM')
        block_ids = self.free_blocks[:needed]
        del self.free_blocks[:needed]
        self.prefix_cache[key] = {'block_ids': block_ids, 'refcount': 1}
        return list(block_ids)

    def release_prefix(self, prefix_tokens: List[int]):
        key = tuple(int(token) for token in prefix_tokens)
        entry = self.prefix_cache.get(key)
        if entry is None:
            raise KeyError('prefix 不存在或已释放')
        entry['refcount'] -= 1
        if entry['refcount'] == 0:
            self.free_blocks.extend(entry['block_ids'])
            self.free_blocks.sort()
            del self.prefix_cache[key]

    def allocate_for_prefill(self, req: Request):
        """
        请求刚进来时 (Prefill阶段)，为它的 Prompt 长度分配所需的全部 Block
        """
        if req.block_table:
            raise ValueError('请求已经完成 Prefill，不能重复分配')
        # TODO 2: 计算需要的 block 数量（向上取整）
        needed_blocks = (req.seq_len + self.block_size - 1) // self.block_size
        
        # TODO 3: 从 free_blocks 中弹出对应数量的 block 索引
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
        new_seq_len = req.seq_len + 1
        
        # TODO 4: 判断是否需要新的 Block
        is_new_block_needed = (new_seq_len % self.block_size) == 1
        
        if is_new_block_needed:
            if not self.free_blocks:
                raise RuntimeError("OOM")
            block_id = self.free_blocks.pop(0)
            req.block_table.append(block_id)
        req.seq_len = new_seq_len

    # TODO 5：计算请求的 Block 占用报告
    def allocation_report(self, req: Request) -> Dict[str, float]:
        allocated_tokens = len(req.block_table) * self.block_size
        unused_tail_tokens = allocated_tokens - req.seq_len
        utilization = req.seq_len / allocated_tokens if allocated_tokens else 0.0
        return {'logical_tokens': req.seq_len, 'allocated_tokens': allocated_tokens, 'unused_tail_tokens': unused_tail_tokens, 'utilization': utilization}

    def release_request(self, req: Request):
        # TODO 6：校验释放列表、归还物理块，并避免重复释放。
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

    # TODO 7：恢复逻辑连续 Cache
    def get_physical_cache(self, req: Request) -> torch.Tensor:
        """根据块表恢复逻辑连续的 KV Cache。"""
        # TODO 7: 根据 req.block_table 的索引，从物理池中提取对应的块
        if not req.block_table:
            raise ValueError('请求没有可读取的物理块')
        if any(block_id < 0 or block_id >= self.num_blocks for block_id in req.block_table):
            raise ValueError('block_table 包含越界物理块')
        blocks = [self.physical_kv_cache[block_id] for block_id in req.block_table]
        cat_blocks = torch.cat(blocks, dim=0)
        
        # 只截取真实 seq_len 长度返回
        return cat_blocks[:req.seq_len]

def run_prefix_cache_extension_check():
    """答案区扩展测试：验证相同前缀的 Block 共享和引用计数释放。"""
    manager = KVCacheManager(num_blocks=4, block_size=4, head_dim=8)
    prefix = [101, 102, 103, 104, 105]
    first_ids = manager.acquire_prefix(prefix)
    second_ids = manager.acquire_prefix(prefix)
    assert first_ids == second_ids, '相同 prefix 应共享同一组物理 Block'
    assert len(manager.free_blocks) == 2, 'Prefix Cache 命中不应重复申请 Block'
    manager.release_prefix(prefix)
    assert len(manager.free_blocks) == 2, '仍有引用时不能释放共享 Block'
    manager.release_prefix(prefix)
    assert manager.free_blocks == [0, 1, 2, 3], '引用计数归零后应归还 Block'
    print('✅ Prefix Cache 扩展测试通过！')


run_prefix_cache_extension_check()
```

### 解析

**1. TODO 1：KV Cache 显存账本**
- **实现方式**：单 token 的理论容量为 `2 * num_layers * num_kv_heads * head_dim * dtype_bytes`；再乘以 `block_size` 和 `num_blocks`。
- **关键点**：前面的 `2` 表示 K、V 两份缓存；账本只描述容量，不包含 workspace、allocator reserved 或临时张量。
- **证据边界**：这是 CPU 可验证的理论估算，不是 vLLM 的真实 GPU 显存峰值。

**2. TODO 2：Prefill 计算所需 Block 数**
- **实现方式**：`needed_blocks = (req.seq_len + self.block_size - 1) // self.block_size`。
- **关键点**：向上取整，确保最后一个不满的 Block 也能容纳剩余 token。
- **关键点**：已有 `block_table` 的请求不能重复 Prefill；先检查空闲块数量，再修改请求和池状态。

**3. TODO 3：分配物理 Block**
- **实现方式**：从 `free_blocks` 取出 `allocated_blocks`，按顺序写入 `req.block_table`。
- **关键点**：`block_table[i]` 表示第 `i` 个逻辑 Block 对应的物理索引；物理索引不要求连续。
- **边界**：资源不足时抛出 OOM，且不能留下半分配状态；物理索引可以不连续。

**4. TODO 4：Decode 跨块扩容**
- **实现方式**：长度增加后，使用 `is_new_block_needed = (new_seq_len % block_size) == 1` 判断是否进入新块。
- **关键点**：只在跨过 Block 边界时申请一个物理块，体现按需增长。
- **边界**：本题只模拟块表更新，不实现真实 KV 写入和 Attention kernel。

**5. TODO 5：占用报告**
- **实现方式**：用 `len(req.block_table) * block_size` 得到已分配 token 容量，再计算尾块浪费和利用率。
- **关键点**：报告同时保留逻辑 token 数和物理分配容量，不能把两者混成一个显存数字。

**6. TODO 6：请求释放与 Block 复用**
- **实现方式**：复制 `req.block_table`，校验物理块没有重复、越界或已经回到空闲池，再归还 `free_blocks` 并清空块表。
- **测试对应**：原子 OOM 测试检查失败不改变状态，释放复用测试检查池状态恢复。

**7. TODO 7：按块表恢复逻辑 Cache**
- **实现方式**：按照 `req.block_table` 读取物理块，沿 token 维拼接，再截断到 `req.seq_len`。
- **关键点**：这展示了逻辑地址与物理地址解耦；真实 PagedAttention 会在 kernel 中按块读取，不需要先拼成连续张量。

**本节能得出的结论**
- 可以解释 Block Table、按需分配、跨块扩容、释放复用和尾块碎片。
- 可以计算 KV Cache 的理论容量，并观察 block_size、dtype 和 KV heads 的影响。

**本节不能得出的结论**
- 不能根据 CPU 模拟推断 vLLM 的 TTFT、吞吐、GPU 峰值显存或 kernel 效率。
- Continuous Batching、Prefix Cache 和真实 serving 调度分别在 70、69 及真实 backend 项目中验证。
### Step 5: 可选 GPU 机制探针

这一组实验只把 Block 池和连续预留策略放到 GPU 上，观察理论账本与 CUDA 分配结果的关系。它不启动 vLLM，也不代表真实 PagedAttention kernel、TTFT 或吞吐收益。

| 对照 | GPU 上分配的对象 | 观察指标 |
|---|---|---|
| Block 池 | 按请求实际长度向上取整后的共享池 | 物理分配 token、尾块浪费、peak allocated |
| 连续预留 | 每个请求都预留相同最大长度 | 预留 token、peak allocated |

只有 `RUN_MODE = 'real_gpu'` 才会创建 CUDA 张量；默认 `dry_run` 只检查配置，不占用 GPU。

```python
# GPU 可选实验配置：先运行 dry_run 检查参数，再改为 real_gpu。
RUN_MODE = 'dry_run'  # dry_run / real_gpu
GPU_PROBE = {
    'request_lengths': [17, 33, 65, 129],
    'max_reserved_len': 256,
    'block_size': 16,
    'head_dim': 64,
    'dtype': 'float16',
}

def run_paged_memory_probe(run_mode='dry_run', config=None):
    """比较合成 Block 池和连续预留的 GPU 分配量。

    这里的张量形状只有 [token, head_dim]，用于观察分配量差异；
    它不是包含 layer/KV-head/block-table 的真实 vLLM cache layout。
    """
    import json
    import torch
    config = dict(config or GPU_PROBE)
    lengths = config['request_lengths']
    block_size = config['block_size']
    dtype = getattr(torch, config['dtype'])
    paged_tokens = sum((length + block_size - 1) // block_size * block_size for length in lengths)
    contiguous_tokens = len(lengths) * config['max_reserved_len']
    plan = {
        'request_lengths': lengths,
        'paged_allocated_tokens': paged_tokens,
        'contiguous_reserved_tokens': contiguous_tokens,
        'tail_waste_tokens': paged_tokens - sum(lengths),
        'synthetic_tensor_shape': [paged_tokens, config['head_dim']],
        'layout_note': '仅用于 CUDA 分配探针，不代表真实 PagedAttention kernel',
        'evidence_level': 'synthetic_gpu_memory_probe',
    }
    if run_mode == 'dry_run':
        print(json.dumps({'mode': run_mode, 'plan': plan}, ensure_ascii=False, indent=2))
        return plan
    if run_mode != 'real_gpu':
        raise ValueError("RUN_MODE 只能是 dry_run 或 real_gpu")
    if not torch.cuda.is_available():
        raise RuntimeError('real_gpu 模式需要 CUDA')
    device = torch.device('cuda')
    results = {}
    for name, tokens in [('paged', paged_tokens), ('contiguous', contiguous_tokens)]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        tensor = torch.empty((tokens, config['head_dim']), dtype=dtype, device=device)
        torch.cuda.synchronize(device)
        results[name] = {
            'allocated_tokens': tokens,
            'peak_allocated_mb': round(torch.cuda.max_memory_allocated(device) / 2**20, 2),
        }
        del tensor
    torch.cuda.empty_cache()
    result = {'mode': run_mode, 'plan': plan, 'results': results, 'device': torch.cuda.get_device_name(0)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result

run_paged_memory_probe(RUN_MODE, GPU_PROBE)
```

## 相关阅读

完成本节的 Block Table 和物理块管理模拟后，可以沿着“分页缓存机制 → 推理引擎实现 → 缓存复用与调度”继续阅读。

- [Efficient Memory Management for Large Language Model Serving with PagedAttention 原论文](https://arxiv.org/abs/2309.06180)
- [vLLM 官方仓库](https://github.com/vllm-project/vllm)
- [vLLM 官方文档](https://docs.vllm.ai/en/latest/)
- [24. SGLang RadixAttention | SGLang 基数注意力](./24_SGLang_RadixAttention.md)
- [34. Prefix Cache Matching and Reuse | Prefix Cache 匹配与复用](./34_Prefix_Cache_Matching_and_Reuse.md)
- [37. KV Cache Scheduling | KV Cache 调度](./37_KV_Cache_Scheduling.md)
