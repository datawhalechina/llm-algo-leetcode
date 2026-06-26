# 19. Distributed Communication Primitives | 分布式进阶：多机通信原语实战 (All-Reduce, All-Gather)
**难度：** Hard | **标签：** `Distributed Training`, `NCCL`, `Communication Primitives` | **目标人群：** 核心 Infra 与算子开发

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/04_CUDA_and_System_Optimization/19_Distributed_Communication_Primitives.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


在前几节的 ZeRO-1 和 TP (张量并行) 中，我们只是通过**数组切片**逻辑上模拟了分布式计算。
但在真实的工业界集群中（如 8 张 A100 甚至千卡集群），GPU 之间必须通过 NCCL (Nvidia Collective Communication Library) 进行真实的物理数据交换。
本节我们将深入 `torch.distributed`，实战最核心的两大通信原语：`All-Reduce` 和 `All-Gather`。这也是面试极其高频的考点（如何计算通信量？Ring-AllReduce 怎么跑的？）。

这一节会把 `dist.all_reduce` 和 `dist.all_gather` 放到真实分布式训练语境里理解。

虽然本节不直接编写 Triton kernel，但它是后续 ZeRO、Offload 和 CUDA Streams 的通信底座。

## 前置

**导语：** 这一节先看 Part 1 的通信边界相关 Group，把 all-reduce / all-gather 依赖的前提先立起来。
- [Part 1: 1C 多卡通信与显存共享](../01_Hardware_Math_and_Systems/1C.md)
- [Part 1: 20 NCCL 与 AllReduce 基础](../01_Hardware_Math_and_Systems/20_NCCL_and_AllReduce_Basics.md)

### Step 1: 集合通信原语的本质

> **All-Reduce (全归约)：**
> 假设每张 GPU 上都有一个相同形状的梯度张量。你想把所有 GPU 的梯度加起来，然后再把总和发还给每张 GPU（在 DDP 数据并行中更新权重必备）。
> - **底层逻辑：** 通常通过 Ring-AllReduce 算法，将数据分为 N 份（N为GPU数），在环形拓扑上传输。
> - **通信量：** 大约是 $2 \times \frac{N-1}{N} \times 	ext{Size}$，它不受 GPU 数量激增的影响，极其高效。

> **All-Gather (全收集)：**
> 假设每张 GPU 算出了模型的一部分输出（如 TP 列切分），你需要把所有 GPU 的这些片段拼装成一个完整的大张量，分发给所有人。
> - **底层逻辑：** 每张卡把自己的那块数据广播给其他所有人。
> - **ZeRO-3 中的应用：** 每张卡只有自己负责的 $\frac{1}{N}$ 权重，在前向传播时，必须通过 `All-Gather` 临时把完整权重拼出来才能算矩阵乘法。

### Step 2: torch.distributed 代码框架
利用 `torch.distributed.init_process_group(backend='nccl')` 初始化通信后端。获取 `dist.get_rank()` (当前 GPU 编号) 和 `dist.get_world_size()` (总 GPU 数) 后，执行 `dist.all_reduce(tensor)` 或 `dist.all_gather(tensor_list, local_tensor)` 进行原语调用。

### Step 3: 动手实战

**要求**：请补全下方 `simulate_distributed_primitives`，使用 PyTorch 的多进程包 `torch.multiprocessing` 模拟 2 张卡的真实通信环境，并在其中实现 `all_reduce` 和 `all_gather` 的调用。


```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
```


```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run_worker(rank, world_size):
    """
    在子进程中执行的代码。代表单张 GPU 的视角。
    """
    # TODO 1: 模拟 DDP 的梯度同步场景
    # 场景：每张卡先得到自己的局部梯度，再用 all_reduce 求平均
    # gradient = ???
    # dist.all_reduce(...)
    # gradient /= world_size

    # TODO 2: 模拟 TP 的特征拼接场景
    # 场景：每张卡只负责一段特征，all_gather 后再拼成完整向量
    # local_feature = ???
    # gathered_list = ???
    # dist.all_gather(...)
    # full_feature = torch.cat(...)
    raise NotImplementedError("请先完成 TODO 1 和 TODO 2")

```


```python
# 运行分布式模拟测试

def simulate_distributed_primitives(num_gpus=2):
    raise NotImplementedError("请先完成 TODO 1 和 TODO 2")


def test_distributed():
    print("启动多进程分布式通信模拟 (模拟 2 个节点/显卡)...")
    raise NotImplementedError("请先完成 TODO 1 和 TODO 2")


test_distributed()

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
import os
import time
import ast
import inspect
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run_worker(rank, world_size):
    """
    在子进程中执行的代码。代表单张 GPU 的视角。
    """
    # 1. 初始化进程组 (Backend 推荐 nccl，但如果本地无多卡或只是 CPU 测试，则使用 gloo)
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    backend = 'nccl' if use_cuda else 'gloo'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend, rank=rank, world_size=world_size)

    device = torch.device(f'cuda:{rank}') if use_cuda else torch.device('cpu')
    if use_cuda:
        torch.cuda.set_device(rank)

    def _time_op(fn):
        if use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end)
        start = time.perf_counter()
        fn()
        return (time.perf_counter() - start) * 1000

    try:
        # ==========================================
        # TODO 1: 模拟 All-Reduce (梯度同步)
        # 场景：每张卡先算出自己的局部梯度，随后做求和并平均
        # ==========================================
        gradient = torch.full((1024,), float(rank + 1), device=device)
        reduce_ms = _time_op(lambda: dist.all_reduce(gradient, op=dist.ReduceOp.SUM))
        gradient /= world_size

        # ==========================================
        # TODO 2: 模拟 All-Gather (特征拼接)
        # 场景：每张卡只负责一段特征，all_gather 后再拼成完整向量
        # ==========================================
        local_feature = torch.full((128,), float(rank), device=device)
        gathered_list = [torch.zeros_like(local_feature) for _ in range(world_size)]
        gather_ms = _time_op(lambda: dist.all_gather(gathered_list, local_feature))
        full_feature = torch.cat(gathered_list, dim=0)

        expected_grad = torch.full_like(gradient, float(sum(range(1, world_size + 1))) / world_size)
        expected_feature = torch.cat([torch.full_like(local_feature, float(r)) for r in range(world_size)], dim=0)

        assert torch.allclose(gradient, expected_grad), 'All-Reduce 结果不正确！'
        assert torch.allclose(full_feature, expected_feature), 'All-Gather 结果不正确！'

        if rank == 0:
            print(f'✅ Rank 0 All-Reduce 后得到均值梯度: {gradient[:4].tolist()}...')
            print(f'✅ Rank 0 All-Gather 后得到完整特征: {full_feature[:8].tolist()}...')
            print(f'⏱️ All-Reduce 通信时间: {reduce_ms:.2f} ms')
            print(f'⏱️ All-Gather 通信时间: {gather_ms:.2f} ms')

    finally:
        dist.destroy_process_group()

def simulate_distributed_primitives(num_gpus=2):
    # ==========================================
    # 检测是否实现了分布式通信原语
    # ==========================================
    source = inspect.getsource(run_worker)
    tree = ast.parse(source)
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_worker'), None)
    assert fn is not None, '缺少 run_worker 函数'

    calls = {n.func.attr for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert 'all_reduce' in calls, 'TODO 1: 必须调用 dist.all_reduce'
    assert 'all_gather' in calls, 'TODO 2: 必须调用 dist.all_gather'

    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() >= num_gpus
    if not use_cuda:
        print(f'当前机器可用 GPU 数量少于 {num_gpus}，将使用 CPU (gloo 后端) 模拟多进程通信。')

    mp.spawn(
        run_worker,
        args=(num_gpus,),
        nprocs=num_gpus,
        join=True,
    )

def test_distributed():
    print('启动多进程分布式通信模拟 (模拟 2 个节点/显卡)...')
    if not torch.cuda.is_available():
        print('⏭️ 无 GPU，完成结构检查；运行级验证需要 GPU。')
        return True
    simulate_distributed_primitives(num_gpus=2)
    print('\n✅ 分布式通信原语测试通过。')

if __name__ == '__main__':
    test_distributed()

```

### 解析

**1. TODO 1: All-Reduce求和操作**
- **实现方式**: `dist.all_reduce(tensor_to_reduce, op=dist.ReduceOp.SUM)`
- **关键点**: 
  - 原位修改张量，所有进程得到相同的归约结果
  - 支持多种归约操作（SUM, PRODUCT, MIN, MAX等）
  - rank 0上的 [1.0, 2.0] + rank 1上的 [3.0, 4.0] = [4.0, 6.0]
- **技术细节**: 
  - 底层使用 Ring-AllReduce 算法；完整推导与 N=4 例子已单独整理到 [09.1 Ring-AllReduce Deep Dive](./09_1_Ring_AllReduce_Deep_Dive.md)
  - 通信开销与 GPU 数量无关，可扩展性强
  - DDP（分布式数据并行）中用于梯度同步：每个 GPU 计算不同 batch 的梯度，通过 All-Reduce 求平均

**2. TODO 2: All-Gather收集操作**
- **实现方式**: `dist.all_gather(gathered_list, local_tensor)`
- **关键点**:
  - 每个进程贡献一个张量，收集到预分配的列表中
  - 所有进程得到完整的收集结果
  - 需要预先分配接收缓冲区（`torch.zeros_like`）
- **技术细节**:
  - 通信量为 $(N-1) \times \text{Size}$，每个 GPU 需要接收其他 N-1 个 GPU 的数据
  - 张量并行（TP）中用于特征拼接：每个 GPU 计算部分列，All-Gather 拼成完整特征
  - ZeRO-3 中用于权重重组：每个 GPU 只存 1/N 权重，前向传播时 All-Gather 临时重组完整权重

**工程优化要点**

- **和后续主线的关系**: 虽然这一节不直接写 Triton kernel，但 DDP / TP / ZeRO 的通信原语会直接影响后续 CUDA Streams、Offload 和多 GPU 调度的性能。
- **Ring-AllReduce 深度分析**: 如果你想看通信量推导、分阶段例子和参数服务器对比，请前往 [09.1 Ring-AllReduce Deep Dive](./09_1_Ring_AllReduce_Deep_Dive.md)。
- **NCCL vs gloo 性能对比**: NVIDIA GPU 必须使用 nccl 后端，利用 NVLink/PCIe 拓扑优化，性能远超 gloo。NCCL 针对 GPU 间通信优化，支持 GPUDirect RDMA（跨节点 GPU 直接通信，无需 CPU 中转），延迟低、带宽高。gloo 是 CPU 通信库，适合 CPU 训练或调试
- **通信与计算重叠**: DDP 中使用 `no_sync()` 上下文管理器延迟梯度同步，在梯度累积阶段跳过 All-Reduce，累积多个 micro-batch 后再同步，减少通信次数。同时，DDP 会在反向传播时自动将梯度 All-Reduce 与后续层的反向计算重叠，隐藏通信延迟
- **梯度累积优化**: 多个 micro-batch 累积后再调用 All-Reduce，通信次数从 K 次降为 1 次（K 为累积步数）。例如，batch_size=32 但显存只够 8，可以累积 4 个 micro-batch，通信开销降为原来的 1/4
- **混合精度通信**: 梯度可以用 fp16 传输，通信量减少 50%。PyTorch AMP 会自动处理精度转换，All-Reduce 前将 fp32 梯度转为 fp16，接收后转回 fp32 更新权重。注意：权重更新必须用 fp32，否则累积误差会导致训练不稳定
- **分层通信拓扑**: 多机训练中，先机内 All-Reduce（利用 NVLink 高带宽），再机间 All-Reduce（利用 InfiniBand）。NCCL 会自动检测拓扑并优化通信路径。例如，8 机 64 卡训练，先在每台机器内 8 卡 All-Reduce，再在 8 台机器间 All-Reduce（每台机器派一个代表），总通信量更少
- **ZeRO-3 权重分片应用**: 每个 GPU 只存储 1/N 权重，前向传播时需要 All-Gather 临时重组完整权重，计算完成后立即释放。反向传播时再次 All-Gather，计算梯度后用 Reduce-Scatter 切分梯度，减少显存占用

- **通信时间测量**: 可以用 `torch.cuda.Event` 在 `all_reduce` / `all_gather` 前后计时，先区分通信耗时，再决定是否需要重叠或分片优化。

### 思考与讨论

**1. Ring-AllReduce 的通信量为什么与 GPU 数量无关？**

完整的通信量证明、N=4 轮次示例和 Parameter Server 对比，已经单独整理到 [09.1 Ring-AllReduce Deep Dive](./09_1_Ring_AllReduce_Deep_Dive.md)。这里先记住结论：Ring-AllReduce 将一次通信拆成 `Reduce-Scatter` 和 `All-Gather` 两阶段，总通信量为 $2 \times \frac{N-1}{N} \times \text{Size}$，当 $N$ 很大时趋近于 $2 \times \text{Size}$。

**2. All-Reduce vs Reduce-Scatter + All-Gather：如何节省显存？**

更完整的显存/通信量对比也在 [09.1 Ring-AllReduce Deep Dive](./09_1_Ring_AllReduce_Deep_Dive.md)。简化地说，`Reduce-Scatter` 先把张量切成分片，`All-Gather` 再把分片重组回完整张量；ZeRO-2 / ZeRO-3 正是利用这组原语在“显存占用”和“通信开销”之间做权衡。

**3. 通信带宽瓶颈：如何分析和优化？**

- 通信时间近似等于 `数据量 / 带宽`，所以需要先看瓶颈是在 NVLink、PCIe 还是 InfiniBand。
- 优先使用 `NCCL`，并利用通信与计算重叠、梯度累积、混合精度和分层通信来隐藏延迟。
- 如果多机通信仍然过慢，再考虑 ZeRO / Offload 这类更激进的显存与通信优化。
