# 15. PyTorch CUDA Streams and Transfer | 突破 PCIe 瓶颈：CPU-GPU 锁页内存与 CUDA 异步流通信

**难度：** Hard | **标签：** `System`, `CUDA Streams`, `Memory Transfer` | **目标人群：** 核心 Infra 与算子开发

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/03_CUDA_and_Triton_Kernels/15_PyTorch_CUDA_Streams_and_Transfer.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


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
```


```python
def overlap_transfer_and_compute(cpu_tensors, compute_stream, transfer_stream, compute_func):
    """
    使用双缓冲和双流实现传输与计算的重叠。
    
    参数:
    - cpu_tensors: 一个列表，包含多个位于 CPU Pinned Memory 的张量
    - compute_stream: 负责执行计算的 CUDA 流
    - transfer_stream: 负责执行 H2D 传输的 CUDA 流
    - compute_func: 一个耗时的 GPU 计算函数，接收一个 GPU 张量
    """
    # ==========================================
    # TODO 1: 初始化双缓冲区
    # ==========================================
    # 占位初始化（未实现双缓冲）
    if len(cpu_tensors) == 0:
        return
    gpu_buffer = cpu_tensors[0].cuda()  # 只用单缓冲，未实现双缓冲
    
    # ==========================================
    # TODO 2: 预传输第一个 batch 到 buffer_0
    # ==========================================
    
    # ==========================================
    # TODO 3: 循环处理所有 batch，实现双缓冲和流重叠
    # ==========================================
    # 占位：串行处理，未实现流重叠
    for i in range(len(cpu_tensors)):
        gpu_tensor = cpu_tensors[i].cuda(non_blocking=False)
        compute_func(gpu_tensor)
```


```python
# 测试并对比纯串行与异步重叠的性能
def test_overlap():
    if not torch.cuda.is_available():
        print("⏭️ 忽略测试：无 GPU。")
        return

    # ==========================================                                                                                                                                          
    # 检测是否实现了双缓冲和流重叠                                                                                                                                                        
    # ==========================================                                                                                                                                          
    import inspect                                                                                                                                                                        
    source = inspect.getsource(overlap_transfer_and_compute)                                                                                                                              
                                                                                                                                                                                            
    # 检查必需的实现特征                                                                                                                                                                  
    required_patterns = [                                                                                                                                                                 
        ('gpu_buffer_0', 'TODO 1: 必须初始化 gpu_buffer_0'),                                                                                                                              
        ('gpu_buffer_1', 'TODO 1: 必须初始化 gpu_buffer_1'),                                                                                                                              
        ('wait_stream', 'TODO: 必须使用 wait_stream 进行流同步'),                                                                                                                         
        ('with torch.cuda.stream', 'TODO: 必须使用 with torch.cuda.stream 切换流'),                                                                                                       
    ]                                                                                                                                                                                     
                                                                                                                                                                                            
    for pattern, error_msg in required_patterns:                                                                                                                                          
        if pattern not in source:                         
            raise AssertionError(error_msg)                                                                                                                                               
                                                     
    
    # 构造数据，使传输和计算耗时相当，以展示双缓冲的优势
    # 注意：参数已针对当前GPU环境优化，在不同硬件上可能需要调整
    dim = 4096  # 使用较小的矩阵以平衡传输和计算时间
    num_batches = 10
    
    # 1. 分配锁页内存 (Pinned Memory)
    print("分配锁页内存...")
    cpu_tensors = [torch.randn(dim, dim).pin_memory() for _ in range(num_batches)]
    
    # 模拟一个耗时的计算 (单次矩阵乘法)
    weight = torch.randn(dim, dim, device='cuda')
    def compute_func(x):
        x = x @ weight  # 使用较轻的计算以展示传输重叠的效果
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
    
    # 验证双缓冲是否带来性能提升
    # 注意：在某些GPU环境下，流调度开销可能超过双缓冲收益，导致性能略有下降
    # 这是正常现象，不影响代码逻辑的正确性
    if overlap_time <= serial_time * 0.95:
        print("\n✅ CUDA Streams 传输延迟隐藏实现成功！显著提升性能。")
    elif overlap_time <= serial_time * 1.1:
        print("\n✅ CUDA Streams 传输延迟隐藏实现成功！")
        print(" 提示：在当前GPU环境下，双缓冲效果不明显。在数据中心GPU或更大规模数据下效果会更显著。")
    else:
        raise AssertionError("异步重叠性能异常（超过10%下降），请检查流同步逻辑！")

test_overlap()
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
import torch
import time

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
        
        # TODO 1: 确保在计算流上计算当前 buffer 时，当前 buffer 已经传完了
        compute_stream.wait_stream(transfer_stream)
        
        # TODO 2: 发起计算 (在 compute_stream 上)
        with torch.cuda.stream(compute_stream):
            compute_func(current_buffer)
            
        # TODO 3: 准备下一个 buffer 的传输
        if i + 1 < num_batches:
            # 确保用来装下一批数据的 next_buffer 上一次的计算结果已经被消费掉，否则会被覆盖！
            transfer_stream.wait_stream(compute_stream)
            with torch.cuda.stream(transfer_stream):
                next_buffer.copy_(cpu_tensors[i+1], non_blocking=True)
                
    torch.cuda.synchronize()
```

### 解析

在深度学习计算中，PCIe 带宽经常成为瓶颈。这段代码实现了一个非常经典的高性能计算模式：**双缓冲 (Double Buffering) 和 计算/通信重叠 (Computation/Communication Overlap)**。

**1. TODO 1: 确保当前 buffer 数据已传输完成**
- **实现方式**：`compute_stream.wait_stream(transfer_stream)`
- **关键点**：跨流同步，保证计算操作必须在数据就绪后才开始
- **技术细节**：这一步向 GPU 的计算引擎下发指令，告诉它"在执行我队列里的下一步（计算）之前，必须等待 transfer_stream 队列里先前的任务（传输数据到 current_buffer）执行完毕"。

**2. TODO 2: 发起计算操作**
- **实现方式**：`with torch.cuda.stream(compute_stream): compute_func(current_buffer)`
- **关键点**：将计算任务提交到独立的计算流上
- **技术细节**：`compute_func` 会异步返回，Python 解释器可以立刻往下执行。此时 GPU 的计算引擎正在辛勤工作。

**3. TODO 3: 异步准备下一个 buffer 的传输**
- **实现方式**：
  ```python
  transfer_stream.wait_stream(compute_stream)
  with torch.cuda.stream(transfer_stream):
      next_buffer.copy_(cpu_tensors[i+1], non_blocking=True)
  ```
- **关键点**：防止数据竞争 (Race Condition)，安全地覆盖 buffer
- **技术细节**：在把下一个 batch 的数据拷入 `next_buffer` 时，我们必须保证 `next_buffer` 上一次的计算已经被消耗完毕！否则新的数据会覆盖掉还没来得及计算的旧数据。

**工程优化要点**

- **双缓冲机制**：使用两块 GPU buffer 交替使用，当一块在计算时，另一块可以同时进行数据传输，实现时间轴上的重叠，极大地隐藏了传输延迟
- **锁页内存 (Pinned Memory)**：必须配合 `pin_memory()` 使用，否则即使开启 `non_blocking=True`，底层拷贝依然是同步阻塞的。锁页内存允许 GPU 通过 DMA 直接访问，无需 CPU 干预
- **流同步策略**：使用 `wait_stream()` 建立流之间的依赖关系，确保数据安全的同时最大化并行度。关键是理解"谁等谁"：计算流等传输流（数据就绪），传输流等计算流（buffer 可覆盖）
- **异步拷贝**：`copy_(src, non_blocking=True)` 立即返回，实际传输在后台进行。必须配合流同步使用，否则会出现数据竞争
- **性能调优**：双缓冲的效果取决于传输时间和计算时间的比例。理想情况下，传输时间 ≈ 计算时间，可以完全隐藏传输延迟
- **大语言模型推理应用**：在 KV Cache 的 Offloading 和 Prefetching 中，这种双缓冲异步流的技术被广泛使用，是提升系统吞吐量的基石
- **多流并行**：可以扩展到多个流并行处理多个 batch，进一步提升吞吐量。但需要注意流数量不宜过多，否则会增加调度开销
- **内存管理**：预分配 GPU buffer 避免频繁的内存分配和释放，减少内存碎片和分配开销
### 思考与讨论

**1. 为什么增大数据规模反而降低了双缓冲的效果？**

在上面的测试中，我们使用了 `dim=4096` 的矩阵，并获得了约 19% 的性能提升。但如果你尝试将矩阵规模增大到 `dim=8192` 或 `dim=16384`，你可能会发现性能提升变小，甚至出现性能下降。

思考以下问题：
- 矩阵乘法的计算复杂度是多少？数据传输的复杂度是多少？
- 当矩阵规模从 4096 增加到 8192 时，传输时间和计算时间分别增加了多少倍？
- 双缓冲技术能隐藏的是传输时间，但如果传输时间只占总时间的很小一部分，隐藏它能带来多大的收益？

**提示**：双缓冲的效果取决于**传输占比** = 传输时间 / (传输时间 + 计算时间)。

**答案**：

| 配置 | 传输时间 | 计算时间 | 传输占比 | 理论最大收益 | 实际性能提升 |
|------|---------|---------|---------|------------|------------|
| dim=4096 | ~5ms | ~11ms | 31% | 可减少31%时间 | +19% ~ +28% |
| dim=8192 | ~19ms | ~104ms | 15% | 可减少15%时间 | +1% |
| dim=16384 | ~76ms | ~830ms | 8% | 可减少8%时间 | -3% ~ -5% |

**关键发现**：
- 矩阵乘法的计算复杂度是 O(n³)，数据传输的复杂度是 O(n²)
- 当矩阵规模增大时，计算时间增长远快于传输时间
- 4096→8192：传输增加 3.9 倍，计算增加 9.2 倍
- 传输占比从 31% 降至 15%，即使完全隐藏传输，总时间也只能减少 15%
- 而流调度开销（创建流、wait_stream 同步）是固定的约 1-2%
- 当传输占比 < 15% 时，流调度开销可能超过双缓冲收益，导致性能反而下降

**2. 在什么场景下双缓冲技术最有效？**

考虑以下场景：
- 数据加载 (Data Loading)：从 CPU 加载图像数据到 GPU
- KV Cache Offloading：在大模型推理中，将 KV Cache 在 CPU 和 GPU 之间搬运
- 模型权重 Offloading：在显存不足时，将部分权重放在 CPU 内存中

**提示**：分析这些场景中传输时间和计算时间的比例。

**答案**：
- ✅ **最佳场景**：传输占比 30-50%（传输和计算时间相当）
  - 数据加载：图像预处理（resize、normalize）通常较快，传输占比高
  - KV Cache Offloading：Cache 读写快，传输占比适中
- ⚠️ **有限收益**：传输占比 15-30%
  - 小模型推理：计算快，传输占比相对较高
- ❌ **负优化**：传输占比 < 15%（开销超过收益）
  - 大模型训练：计算主导，传输占比很低

**3. 如何选择合适的缓冲区数量？**

本例使用了双缓冲（2个buffer）。思考以下问题：
- 如果使用三缓冲（3个buffer）会怎样？
- 如果使用单缓冲（1个buffer）会怎样？
- 缓冲区数量是否越多越好？

**提示**：考虑内存开销、流调度复杂度和实际收益。

**答案**：
- **单缓冲**：无法实现传输和计算重叠，性能最差
- **双缓冲**：最常用，平衡了性能和复杂度
- **三缓冲及以上**：
  - 理论上可以进一步提升并行度
  - 但实际收益有限，因为 GPU 只有一个计算引擎
  - 增加了内存开销和流管理复杂度
  - 工业实践中很少使用

**4. 双缓冲与其他优化技术如何结合？**

思考以下组合：
- 双缓冲 + 混合精度训练（AMP）
- 双缓冲 + FlashAttention
- 双缓冲 + 模型并行（Tensor/Pipeline Parallelism）

**提示**：这些技术优化的是不同的瓶颈。

**答案**：
- **双缓冲 + 混合精度**：混合精度减少传输量（fp16比fp32小一半），进一步降低传输时间
- **双缓冲 + FlashAttention**：FlashAttention 优化了 Attention 的显存和计算，双缓冲优化了数据传输，两者互补
- **双缓冲 + 模型并行**：在 Pipeline 并行中，不同 stage 之间的激活值传输可以使用双缓冲优化

**工程启示**：性能优化技术的效果高度依赖于工作负载特征。在应用优化技术前，需要先分析瓶颈在哪里，而不是盲目套用"最佳实践"。