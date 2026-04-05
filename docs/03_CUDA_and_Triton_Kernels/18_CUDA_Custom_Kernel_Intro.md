# 18 CUDA Custom Kernel Intro

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/03_CUDA_and_Triton_Kernels/18_CUDA_Custom_Kernel_Intro.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 18. 硬核降维打击：原生 CUDA C++ 编程与 PyTorch C++ 扩展 (JIT)

**难度：** Hard | **标签：** `CUDA C++`, `JIT Extension`, `Vector Add` | **目标人群：** 核心 Infra 与算子开发

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


### Step 2: CUDA 编程模型核心
从纯 Python 跨越到底层，最重要的是理解硬件线程层级：
1. **Thread (线程)**：底层的计算单元，负责单个数据元素。
2. **Block (线程块)**：一组协作的 Thread，共享极速小内存 (Shared Memory)，可在块内同步步调。通常包含 128 或 256 个线程。
3. **Grid (网格)**：一堆独立的 Block 组成 Grid，彼此无法直接通信。这极度贴合了矩阵并行的需求。


### Step 3: 原生 CUDA 与 PyTorch JIT 扩展框架
用 `__global__` 修饰 C++ CUDA Kernel，利用 `threadIdx.x` 和 `blockIdx.x` 定位数组下标。使用 PyTorch 的 `torch.utils.cpp_extension.load_inline`，它能在 Jupyter 运行时，唤起 `nvcc` 将 C++ 字符串编译成 Python 可直接引用的模块。


###  Step 4: 动手实战

**要求**：请补全下方 `cuda_source` 字符串中的 C++ CUDA 内核代码。你需要实现全局索引计算和越界检查。


```python
import torch
from torch.utils.cpp_extension import load_inline

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
    // ==========================================
    // int index = ???;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // ==========================================
    // TODO 2: 越界检查 (防止访问超过 size 的内存)
    // ==========================================
    // if (???) {
    if (index < size) {
        out[index] = x[index] + y[index];
    }
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
    // 通常在 wrapper 里我们不需要手动 sync，因为 PyTorch 会管理流，但这里为了确保没报错加上
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
    if not torch.cuda.is_available() or 'vector_add_extension' not in globals():
        print("⏭️ 忽略测试：无 GPU 或编译失败。")
        return
        
    try:
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
        
        print("✅ 太硬核了！你成功使用 JIT 编译并运行了原生的 CUDA C++ 代码！")
        print("💡 面试官如果问你 blockIdx.x 和 threadIdx.x，现在你可以把这个公式倒背如流了。")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

test_cuda_vector_add()

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

Triton 虽然好用，但它是在 Python 层面封装的一层高阶抽象。深入底层的 CUDA C++，我们才能真正理解 GPU 是如何工作的：

1.  **全局一维索引的魔法公式 ()**:
    *   这就是 GPU **SIMT (单指令多线程)** 的灵魂所在。
    *   想象一个巨大的数组。我们把这个数组切成了很多个块 (Block)。
    *    是当前线程所在 Block 的编号（比如第 3 个 Block）。
    *    是每个 Block 里包含了多少个线程（我们在 C++ wrapper 里定义了 ，所以它是 256）。
    *    是当前线程在自己 Block 里的局部编号（比如第 10 个线程，范围 0~255）。
    *   那么这个线程在整个大数组里的绝对位置就是：。这个线程就负责计算 ！
2.  **越界检查 ()**:
    *   我们分配 Block 时，为了方便通常是向上取整的，比如 。
    *   如果 ，我们需要  个 Block。
    *   这 4 个 Block 总共有  \times 256 = 1024$ 个线程。
    *   但是我们的数组只有 1000 个元素！多出来的 24 个线程如果不加  判断，就会访问到数组外面的内存（Segment Fault / 显存越界），导致程序崩溃。
3.  **PyTorch JIT 编译 ()**:
    *   传统写 CUDA 算子需要写  文件、配置  和 ，非常繁琐。
    *    直接在运行时调用 NVIDIA 的  编译器，把 C++ 字符串编译成动态链接库，直接映射为 Python 函数。这是测试和验证 CUDA 代码极好的工具。


```python
cuda_source = '''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void vector_add_kernel(const float* x, const float* y, float* out, int size) {
    // 1. 计算全局一维索引： 当前 Block 编号 * Block 大小 + Block 内线程编号
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 2. 越界检查。因为总线程数 (blocks * threads) 通常大于或等于 size
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

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
