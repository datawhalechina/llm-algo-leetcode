# 18. CUDA Custom Kernel Intro | 硬核降维打击：原生 CUDA C++ 编程与 PyTorch C++ 扩展 (JIT)

**难度：** Hard | **标签：** `CUDA C++`, `JIT Extension`, `Vector Add` | **目标人群：** 核心 Infra 与算子开发

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/03_CUDA_and_Triton_Kernels/18_CUDA_Custom_Kernel_Intro.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


如果你在面试中只懂写 Triton，虽然能解决日常的算子融合问题，但面试官一定会问你：“Triton 底层的 Grid/Block 是怎么映射到 CUDA 线程的？”
本节我们将打破 Python 的温室，使用 `torch.utils.cpp_extension.load_inline`，直接在 Jupyter Notebook 的字符串里**手撕原生 CUDA C++ 核函数**，并且在后台即时 (JIT) 编译成 PyTorch 可用的模块。
我们将从最经典的 Vector Add (向量加法) 开始，直观对比 Triton 的 `pid` 和 CUDA 的 `blockIdx.x * blockDim.x + threadIdx.x`。

### Step 1: CUDA 编程模型核心

> **线程层级 (Thread Hierarchy)：**
> - **Grid (网格)**：整个计算任务的总称。由多个 Block 组成。
> - **Block (线程块)**：由多个 Thread 组成。同一个 Block 内的线程可以共享 SRAM (Shared Memory) 并进行同步。
> - **Thread (线程)**：GPU 计算的最小单位。
> 
> **计算全局索引的公式 (必考！)：**
> 如果我们把数据分为多个 Block，每个 Block 里有 `blockDim.x` 个线程。当前线程在整个大数组中的一维索引是：
> `int index = blockIdx.x * blockDim.x + threadIdx.x;`

> **与 Triton 的区别：**
> - 在 **Triton** 中，我们通常让一个 Program (相当于 CUDA 的一个 Block) 负责处理一个长度为 `BLOCK_SIZE` 的张量切片 (通过循环和 Mask)。
> - 在 **CUDA** 中，我们是为每一个标量元素 (Element) 分配一个 Thread！这种**细粒度**带来了极高的开发门槛，但也赋予了对硬件绝对的控制权。

### Step 2: CUDA 线程层级详解
从纯 Python 跨越到底层，最重要的是理解硬件线程层级：
1. **Thread (线程)**：底层的计算单元，负责单个数据元素。
2. **Block (线程块)**：一组协作的 Thread，共享极速小内存 (Shared Memory)，可在块内同步步调。通常包含 128 或 256 个线程。
3. **Grid (网格)**：一堆独立的 Block 组成 Grid，彼此无法直接通信。这种设计非常适合矩阵并行计算的需求。
### Step 3: 原生 CUDA 与 PyTorch JIT 扩展框架
用 `__global__` 修饰 C++ CUDA Kernel，利用 `threadIdx.x` 和 `blockIdx.x` 定位数组下标。使用 PyTorch 的 `torch.utils.cpp_extension.load_inline`，它能在 Jupyter 运行时，唤起 `nvcc` 将 C++ 字符串编译成 Python 可直接引用的模块。

###  Step 4: 动手实战

**要求**：请补全下方 `cuda_source` 字符串中的 C++ CUDA 内核代码。你需要实现全局索引计算和越界检查。


```python
import torch
from torch.utils.cpp_extension import load_inline
```


```python
# ==========================================
# 1. 编写原生 CUDA C++ 代码
# 我们定义一个 __global__ 关键字修饰的核函数
# 以及一个 C++ 的包裹函数用于给 PyTorch 调用
# ==========================================
cuda_source = '''
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA 核函数 (Kernel)
__global__ void vector_add_kernel(const float* x, const float* y, float* out, int size) {
    // ==========================================
    // TODO 1: 计算当前线程的全局一维索引
    // 提示: 使用 blockIdx.x, blockDim.x, threadIdx.x 计算
    // ==========================================
    // int index = ???;
    
    // ==========================================
    // TODO 2: 越界检查 (防止访问超过 size 的内存)
    // 提示: 检查 index 是否小于 size
    // ==========================================
    // if (???) {
    //     out[index] = x[index] + y[index];
    // }
}

// PyTorch 调用的 C++ Wrapper 函数
torch::Tensor vector_add_cuda(torch::Tensor x, torch::Tensor y) {
    auto size = x.numel();
    auto out = torch::empty_like(x);
    
    // 设置线程块大小 (每个 Block 包含 256 个线程)
    const int threads = 256;
    
    // 计算需要的 Grid 大小 (向上取整)
    const int blocks = (size + threads - 1) / threads;
    
    // 调用核函数: kernel<<<blocks, threads>>>(...)
    vector_add_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), 
        y.data_ptr<float>(), 
        out.data_ptr<float>(), 
        size
    );
    
    // 确保 CUDA 执行完成 (同步)
    cudaDeviceSynchronize();
    
    return out;
}
'''

# 用于向 PyTorch 注册我们写的 C++ 函数
cpp_source = '''
torch::Tensor vector_add_cuda(torch::Tensor x, torch::Tensor y);
'''

# ==========================================
# 2. 编译并加载 CUDA 扩展 (JIT)
# 这一步可能会耗时 10-30 秒，它会在后台调用 nvcc 编译器
# ==========================================
print("⏳ 正在后台使用 NVCC 编译 CUDA C++ 代码... 请耐心等待！")
import time
start = time.time()

try:
    vector_add_extension = load_inline(
        name='vector_add_ext',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['vector_add_cuda'],
        with_cuda=True,
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3']
    )
    print(f"✅ 编译成功！耗时: {time.time() - start:.2f} 秒")
except Exception as e:
    print(f"❌ 编译失败，请确保环境中已安装 CUDA Toolkit (nvcc)。报错: {e}")
```


```python
# 测试你的 CUDA C++ 算子
def test_cuda_vector_add():
    # 检查TODO是否完成
    if '// int index = ???;' in cuda_source or '// if (???)' in cuda_source:
        raise NotImplementedError("请先完成 TODO 1 和 TODO 2")
    
    if not torch.cuda.is_available():
        print("⏭️ 无 GPU，跳过测试")
        return
    
    if 'vector_add_extension' not in globals():
        raise RuntimeError("CUDA 扩展编译失败，请检查 nvcc 是否安装")
    
    torch.manual_seed(42)
    # 故意选一个不能被 256 整除的 size，测试边界检查是否正确
    size = 10000 
    
    x = torch.randn(size, device='cuda', dtype=torch.float32)
    y = torch.randn(size, device='cuda', dtype=torch.float32)
    
    # 1. PyTorch 原生计算
    out_pt = x + y
    
    # 2. 我们手写的 CUDA C++ 计算
    out_cu = vector_add_extension.vector_add_cuda(x, y)
    
    diff = torch.max(torch.abs(out_pt - out_cu))
    assert diff < 1e-5, "CUDA 核函数计算结果错误！"
    
    print("✅ CUDA C++ 向量加法核函数验证通过。")
    print("工程实践：全局索引公式 index = blockIdx.x * blockDim.x + threadIdx.x 是CUDA编程基础。")

test_cuda_vector_add()
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
from torch.utils.cpp_extension import load_inline

cuda_source = '''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void vector_add_kernel(const float* x, const float* y, float* out, int size) {
    // TODO 1: 计算当前线程的全局一维索引
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO 2: 越界检查
    if (index < size) {
        out[index] = x[index] + y[index];
    }
}

torch::Tensor vector_add_cuda(torch::Tensor x, torch::Tensor y) {
    auto size = x.numel();
    auto out = torch::empty_like(x);
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    vector_add_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), 
        y.data_ptr<float>(), 
        out.data_ptr<float>(), 
        size
    );
    
    cudaDeviceSynchronize();
    return out;
}
'''
```


```python
# 编译和测试
cpp_source = '''
torch::Tensor vector_add_cuda(torch::Tensor x, torch::Tensor y);
'''

print("⏳ 正在编译 CUDA 扩展...")
import time
start = time.time()

try:
    vector_add_extension = load_inline(
        name='vector_add_ext_answer',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['vector_add_cuda'],
        with_cuda=True,
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3']
    )
    print(f"✅ 编译成功！耗时: {time.time() - start:.2f} 秒")
except Exception as e:
    print(f"❌ 编译失败: {e}")

def test_cuda_vector_add():
    if not torch.cuda.is_available():
        print("⏭️ 无 GPU，跳过测试")
        return
    
    if 'vector_add_extension' not in globals():
        raise RuntimeError("CUDA 扩展编译失败")
    
    torch.manual_seed(42)
    size = 10000
    
    x = torch.randn(size, device='cuda', dtype=torch.float32)
    y = torch.randn(size, device='cuda', dtype=torch.float32)
    
    out_pt = x + y
    out_cu = vector_add_extension.vector_add_cuda(x, y)
    
    diff = torch.max(torch.abs(out_pt - out_cu))
    assert diff < 1e-5, "CUDA 核函数计算结果错误！"
    
    print("✅ CUDA C++ 向量加法核函数验证通过。")

test_cuda_vector_add()
```

### 解析

**1. TODO 1: 计算全局线程索引**
- **实现方式**: `int index = blockIdx.x * blockDim.x + threadIdx.x;`
- **关键点**: 
  - blockIdx.x: 当前Block在Grid中的索引
  - blockDim.x: 每个Block包含的线程数（256）
  - threadIdx.x: 当前线程在Block中的索引（0-255）
- **技术细节**: 
  - 这是CUDA编程的核心公式，必须掌握
  - 适用于一维向量的并行计算
  - 每个线程负责一个数组元素

**2. TODO 2: 越界检查**
- **实现方式**: `if (index < size)`
- **关键点**: 
  - Grid总线程数通常大于数组大小
  - 必须检查边界，防止内存越界
  - 未检查会导致Segmentation Fault
- **技术细节**: 
  - 例如：size=1000, threads=256, blocks=4
  - 总线程数 = 4 × 256 = 1024
  - 多出24个线程需要被过滤掉

**工程优化要点**

- **Grid/Block配置策略**:
  - threads通常选择256或512（warp的倍数）
  - blocks = (size + threads - 1) / threads（向上取整）
  - 过小的threads浪费SM资源，过大的threads超出硬件限制

- **内存访问模式**:
  - 连续线程访问连续内存（coalesced access）
  - 本例中：thread 0访问x[0], thread 1访问x[1]，完美合并
  - 合并访问可提升10-100倍带宽利用率

- **线程分支发散**:
  - 同一warp（32个线程）内的分支会串行执行
  - 本例中：if (index < size) 在边界处会导致分支发散
  - 影响较小，因为只有最后一个Block受影响

- **Shared Memory使用**:
  - 本例未使用Shared Memory（简单向量加法不需要）
  - 复杂算法（如矩阵乘法）需要Shared Memory优化
  - Shared Memory带宽远高于Global Memory

- **CUDA vs Triton性能对比**:
  - CUDA：完全控制，性能极致，开发复杂
  - Triton：自动优化，开发简单，性能接近CUDA
  - 选择：简单算子用Triton，复杂算子用CUDA

- **JIT编译优化**:
  - load_inline适合快速原型和测试
  - 生产环境建议预编译（setup.py）
  - 编译选项：-O3优化，--use_fast_math加速

- **常见错误排查**:
  - 编译失败：检查CUDA Toolkit安装
  - 运行时错误：使用cudaGetLastError()检查
  - 性能问题：使用nvprof或Nsight分析
### 思考与讨论

**1. 为什么CUDA需要手动计算全局索引？**

在Triton中，我们使用`pid = tl.program_id(0)`获取Program ID，然后通过`offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`计算偏移量。而在CUDA中，我们需要手动计算`index = blockIdx.x * blockDim.x + threadIdx.x`。

思考以下问题：
- CUDA的线程模型与Triton的Program模型有什么区别？
- 为什么CUDA采用Block/Thread两层结构？
- 手动计算索引有什么优势和劣势？

**提示**: 考虑硬件架构、编程灵活性、性能优化等因素。

**答案**:

**CUDA vs Triton线程模型对比**:

| 维度 | CUDA | Triton |
|------|------|--------|
| 抽象层级 | 低（直接映射硬件） | 高（编译器自动优化） |
| 线程粒度 | 每个线程处理1个元素 | 每个Program处理BLOCK_SIZE个元素 |
| 索引计算 | 手动计算全局索引 | 自动生成偏移量 |
| 灵活性 | 完全控制 | 受限于编译器 |
| 开发难度 | 高 | 低 |

**CUDA两层结构的原因**:
1. **硬件映射**: 
   - Block映射到SM（Streaming Multiprocessor）
   - Thread映射到CUDA Core
   - 一个SM可以同时运行多个Block

2. **Shared Memory**: 
   - 同一Block内的线程共享Shared Memory
   - 不同Block之间无法通信
   - 这种设计简化了硬件实现

3. **同步机制**: 
   - Block内可以使用__syncthreads()同步
   - Block间无法同步（除非使用Global Memory）
   - 适合数据并行计算

**手动计算索引的优劣**:

优势：
- 完全控制内存访问模式
- 可以实现复杂的索引逻辑（如2D/3D索引）
- 性能优化空间大

劣势：
- 开发复杂，容易出错
- 需要深入理解硬件架构
- 代码可读性差

**工程启示**: 
- 简单算子用Triton（开发效率高）
- 复杂算子用CUDA（性能极致）
- 理解CUDA有助于优化Triton代码

**2. Grid/Block配置如何影响性能？**

在本例中，我们使用`threads=256`和`blocks=(size+threads-1)/threads`。不同的配置会显著影响性能。

思考以下问题：
- threads=128 vs threads=256 vs threads=512，哪个更好？
- 如果size很小（如100），性能会怎样？
- 如何选择最优的Grid/Block配置？

**提示**: 考虑SM占用率、warp调度、内存带宽等因素。

**答案**:

**threads选择的影响**:

| threads | SM占用率 | warp数量 | 性能 | 适用场景 |
|---------|---------|---------|------|---------|
| 128 | 低 | 4 | 差 | 不推荐 |
| 256 | 中 | 8 | 好 | 通用选择 |
| 512 | 高 | 16 | 最好 | 计算密集型 |
| 1024 | 很高 | 32 | 可能更差 | 寄存器压力大 |

**关键因素**:
1. **Warp调度**: 
   - GPU以warp（32个线程）为单位调度
   - threads应该是32的倍数
   - 256 = 8 warps，512 = 16 warps

2. **SM占用率**: 
   - 每个SM最多1024个线程
   - threads=256时，每个SM可以运行4个Block
   - threads=512时，每个SM可以运行2个Block
   - 更多Block可以隐藏内存延迟

3. **寄存器压力**: 
   - 每个线程使用的寄存器数量有限
   - threads过大会导致寄存器溢出到Local Memory
   - 性能急剧下降

**小数据集问题**（size=100）:
- blocks = (100 + 256 - 1) / 256 = 1
- 只有1个Block，无法充分利用GPU
- 性能远低于CPU
- 解决：批处理多个小任务

**最优配置策略**:
1. 使用nvprof或Nsight Compute分析
2. 测试不同配置的性能
3. 通常256或512是安全选择
4. 复杂kernel需要根据寄存器使用量调整

**3. CUDA vs Triton：何时使用原生CUDA？**

Triton提供了高层抽象，大多数情况下性能接近手写CUDA。但某些场景下，原生CUDA仍然是必需的。

思考以下问题：
- 什么情况下Triton无法满足需求？
- 原生CUDA的学习成本是否值得？
- 如何在Triton和CUDA之间做选择？

**提示**: 考虑算法复杂度、性能要求、开发时间等因素。

**答案**:

**必须使用CUDA的场景**:
1. **复杂的线程同步**: 
   - 需要__syncthreads()、warp shuffle等
   - Triton不支持Block内同步

2. **动态并行**: 
   - Kernel内部启动新的Kernel
   - Triton不支持

3. **特殊硬件特性**: 
   - Tensor Core的精细控制
   - Shared Memory的bank conflict优化
   - Triton自动处理，但可能不是最优

4. **极致性能优化**: 
   - 需要手动优化每一个细节
   - Triton的自动优化有时不够

**Triton的优势场景**:
1. **简单的element-wise操作**: 
   - 向量加法、激活函数等
   - Triton代码简洁，性能接近CUDA

2. **快速原型开发**: 
   - 测试新算法
   - 验证想法

3. **自动优化**: 
   - Autotune自动选择最优配置
   - 减少手动调优工作

**学习成本分析**:
- CUDA学习曲线陡峭（需要1-3个月）
- Triton学习曲线平缓（需要1-2周）
- 但理解CUDA有助于优化Triton代码

**选择策略**:
1. 优先尝试Triton
2. 性能不满足时，分析瓶颈
3. 如果是Triton限制，切换到CUDA
4. 混合使用：简单部分用Triton，复杂部分用CUDA

**工程启示**:
- 掌握CUDA是高级优化的必备技能
- Triton适合日常开发
- 理解底层有助于写出更好的高层代码