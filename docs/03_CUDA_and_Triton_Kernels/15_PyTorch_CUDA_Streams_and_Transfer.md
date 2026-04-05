# 15 PyTorch CUDA Streams and Transfer

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/03_CUDA_and_Triton_Kernels/15_PyTorch_CUDA_Streams_and_Transfer.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 15. 突破 PCIe 瓶颈：CPU-GPU 锁页内存与 CUDA 异步流通信

**难度：** Hard | **标签：** `System`, `CUDA Streams`, `Memory Transfer` | **目标人群：** 核心 Infra 与算子开发

在训练大模型的数据加载 (Data Loading) 或者推理时 Offload 权重/KV Cache 到内存中时，CPU 到 GPU 的 PCIe 传输带宽往往会成为严重的性能瓶颈。
在 PyTorch 中，简单的 `.cuda()` 或 `.to('cuda')` 是一把双刃剑：它在底层是同步阻塞的。如果等待数据传完才开始计算，GPU 就会处于饥饿状态 (Compute Starvation)。
本节我们将深入 PyTorch 底层系统调用，通过 **锁页内存 (Pinned Memory)** 和 **CUDA 多流 (Streams)** 实现数据传输与计算的完全重叠 (Overlap)。


### Step 1: 锁页内存与异步流机制

> **锁页内存 (Pinned / Page-Locked Memory)：**
> 操作系统的虚拟内存会将不常用的数据交换到硬盘 (Swap)。当 GPU 通过 PCIe 请求 CPU 内存数据时，如果内存是可分页的 (Pageable)，CPU 必须先将其锁定，或者拷贝到一个临时的锁页缓冲区，然后再传输给 GPU。这导致了双倍的内存开销和极慢的速度。
> **优化：** 使用 `tensor.pin_memory()` 直接在主机分配物理上被锁定的内存，GPU 可以通过 DMA (直接内存访问) 高速、无 CPU 干预地拉取数据。

> **CUDA 异步流 (Streams) 与重叠计算：**
> PyTorch 默认所有的操作都在同一个默认流 (Default Stream) 上排队串行执行。
> 想要在计算当前 Batch 的同时，把下一个 Batch 传到 GPU？我们需要：
> 1. 数据必须在锁页内存中。
> 2. 使用 `tensor.to('cuda', non_blocking=True)` 发起异步传输。
> 3. 创建一个新的 `torch.cuda.Stream`，在这个分支流上专门负责传输或计算，让它们在时间轴上并行！


### Step 2: 锁页内存 与异步流隐藏延迟
默认的 CPU 到 GPU 数据拷贝会导致 GPU 计算引擎闲置。通过将张量放置在 `Pinned Memory` (锁页内存) 中，系统保证这块内存不会被换出，从而允许 GPU 利用 DMA 引擎在后台异步“偷取”数据。如果把异步读取与计算操作放在不同的 `CUDA Stream` 中，就能彻底隐藏 IO 耗时。


### Step 3: 多流计算代码框架
使用 `tensor.pin_memory()` 锁定内存，利用 `tensor.to(device, non_blocking=True)` 发起异步传输。通过 `with torch.cuda.stream(stream1):` 开辟非默认执行流（Stream），将计算任务抛入。最后使用 `torch.cuda.synchronize()` 等待任务完成。


###  Step 4: 动手实战

**要求**：请补全下方 `overlap_transfer_and_compute` 函数。我们将使用双缓冲 (Double Buffering) 和两个 CUDA 流，在流 A 上执行耗时的矩阵乘法，同时在流 B 上异步把下一块数据传到 GPU。


```python
import torch
import time

def overlap_transfer_and_compute(cpu_tensors, compute_stream, transfer_stream, compute_func):
    """
    使用双缓冲和双流实现传输与计算的重叠。
    
    参数:
    - cpu_tensors: 一个列表，包含多个位于 CPU Pinned Memory 的张量
    - compute_stream: 负责执行计算的 CUDA 流
    - transfer_stream: 负责执行 H2D 传输的 CUDA 流
    - compute_func: 一个耗时的 GPU 计算函数，接收一个 GPU 张量
    """
    num_batches = len(cpu_tensors)
    if num_batches == 0:
        return
        
    # 我们使用两个变量交替作为 GPU 端的缓冲区 (Double Buffering)
    gpu_buffer_0 = torch.empty_like(cpu_tensors[0], device='cuda')
    gpu_buffer_1 = torch.empty_like(cpu_tensors[0], device='cuda')
    buffers = [gpu_buffer_0, gpu_buffer_1]
    
    # ==========================================
    # 预先传输第一个 Batch 到 buffer_0
    # ==========================================
    with torch.cuda.stream(transfer_stream):
        buffers[0].copy_(cpu_tensors[0], non_blocking=True)
        
    for i in range(num_batches):
        current_buffer = buffers[i % 2]
        next_buffer = buffers[(i + 1) % 2]
        
        # ==========================================
        # TODO 1: 在 transfer_stream 确保当前 buffer 数据已经传完
        # 提示: 使用 compute_stream.wait_stream(transfer_stream)
        # ==========================================
        # ???
        compute_stream.wait_stream(transfer_stream)
        
        # ==========================================
        # TODO 2: 在 compute_stream 上发起当前 batch 的计算
        # 提示: with torch.cuda.stream(compute_stream): ...
        # ==========================================
        with torch.cuda.stream(compute_stream):
            compute_func(current_buffer)
            
        # ==========================================
        # TODO 3: 在 transfer_stream 上异步传输下一个 batch (如果还有的话)
        # 提示: 必须告诉 transfer_stream，它在覆盖 next_buffer 之前，
        # 需要等待 compute_stream 把上一次对 next_buffer 的计算做完！
        # 1. transfer_stream.wait_stream(compute_stream)
        # 2. 发起拷贝: next_buffer.copy_(cpu_tensors[i+1], non_blocking=True)
        # ==========================================
        if i + 1 < num_batches:
            transfer_stream.wait_stream(compute_stream)
            with torch.cuda.stream(transfer_stream):
                next_buffer.copy_(cpu_tensors[i+1], non_blocking=True)
                
    # 确保所有计算在退出前完成
    torch.cuda.synchronize()

```

```python
# 测试并对比纯串行与异步重叠的性能
def test_overlap():
    if not torch.cuda.is_available():
        print("⏭️ 忽略测试：无 GPU。")
        return
        
    try:
        # 构造大型数据，使其传输和计算耗时相当
        dim = 8192
        num_batches = 10
        
        # 1. 分配锁页内存 (Pinned Memory)
        print("分配锁页内存...")
        cpu_tensors = [torch.randn(dim, dim).pin_memory() for _ in range(num_batches)]
        
        # 模拟一个耗时的计算 (做多次矩阵乘法)
        weight = torch.randn(dim, dim, device='cuda')
        def compute_func(x):
            for _ in range(3):
                x = x @ weight
            return x
            
        # ==========================================
        # 纯串行执行 (Baseline)
        # ==========================================
        torch.cuda.synchronize()
        start_time = time.time()
        for i in range(num_batches):
            # 隐式阻塞的传输
            gpu_tensor = cpu_tensors[i].cuda(non_blocking=False)
            compute_func(gpu_tensor)
        torch.cuda.synchronize()
        serial_time = time.time() - start_time
        
        # ==========================================
        # 异步流重叠执行 (Overlap)
        # ==========================================
        compute_stream = torch.cuda.Stream()
        transfer_stream = torch.cuda.Stream()
        
        torch.cuda.synchronize()
        start_time = time.time()
        overlap_transfer_and_compute(cpu_tensors, compute_stream, transfer_stream, compute_func)
        overlap_time = time.time() - start_time
        
        print(f"串行耗时: {serial_time:.4f} 秒")
        print(f"重叠耗时: {overlap_time:.4f} 秒")
        print(f"性能提升: {(serial_time / overlap_time - 1) * 100:.1f}%")
        
        assert overlap_time < serial_time * 0.95, "异步重叠没有带来显著的性能提升，请检查流同步逻辑！"
        print("\n✅ 恭喜！你成功用 CUDA Streams 隐藏了 PCIe 的传输延迟！这是写高性能推理服务器的最核心技能。")
        
    except NotImplementedError:
        print("请先完成 TODO 代码！")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

test_overlap()

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

::: details 💡 点击查看官方解析与参考代码


### 💡 核心实现原理解析

在深度学习计算中，PCIe 带宽经常成为瓶颈。这段代码实现了一个非常经典的高性能计算模式：**双缓冲 (Double Buffering) 和 计算/通信重叠 (Computation/Communication Overlap)**。

1.  **:** 这一步确保在计算  前，从 CPU 传到 GPU  的数据确实已经传输完毕。计算流在此处等待传输流的事件。
2.  **:** 这一步将计算任务提交到独立的计算流上，让它在 GPU 上执行。
3.  **:** 在把下一个 batch 的数据拷入  时，我们必须保证  上一次的计算已经被消耗完毕！否则新的数据会覆盖掉还没来得及计算的旧数据（产生 Race Condition 数据竞争）。
4.  **:** 终于安全了，我们在传输流上异步发起对下一块缓冲区的填充。

由于我们只有两块 Buffer 交替使用，所以在向 buffer_1 传数据时，如果计算还在用 buffer_1，就必须等；如果计算在用 buffer_0，因为 buffer 独立，传输就可以和 buffer_0 的计算完美并行！这就实现了时间轴上的重叠，极大地隐藏了传输延迟。


```python
def overlap_transfer_and_compute(cpu_tensors, compute_stream, transfer_stream, compute_func):
    num_batches = len(cpu_tensors)
    if num_batches == 0:
        return
        
    gpu_buffer_0 = torch.empty_like(cpu_tensors[0], device="cuda")
    gpu_buffer_1 = torch.empty_like(cpu_tensors[0], device="cuda")
    buffers = [gpu_buffer_0, gpu_buffer_1]
    
    with torch.cuda.stream(transfer_stream):
        buffers[0].copy_(cpu_tensors[0], non_blocking=True)
        
    for i in range(num_batches):
        current_buffer = buffers[i % 2]
        next_buffer = buffers[(i + 1) % 2]
        
        # 1. 确保在计算流上计算当前 buffer 时，当前 buffer 已经传完了
        compute_stream.wait_stream(transfer_stream)
        
        # 2. 发起计算 (在 compute_stream 上)
        with torch.cuda.stream(compute_stream):
            compute_func(current_buffer)
            
        # 3. 准备下一个 buffer 的传输
        if i + 1 < num_batches:
            # 确保用来装下一批数据的 next_buffer 上一次的计算结果已经被消费掉，否则会被覆盖！
            transfer_stream.wait_stream(compute_stream)
            with torch.cuda.stream(transfer_stream):
                next_buffer.copy_(cpu_tensors[i+1], non_blocking=True)
                
    torch.cuda.synchronize()
```

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
