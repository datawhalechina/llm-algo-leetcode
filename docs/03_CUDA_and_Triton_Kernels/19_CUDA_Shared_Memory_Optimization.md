# 19 CUDA Shared Memory Optimization

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/03_CUDA_and_Triton_Kernels/19_CUDA_Shared_Memory_Optimization.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 19. 榨干硬件极限：CUDA Shared Memory (共享内存) 优化与 GEMM

**难度：** Hard | **标签：** `CUDA C++`, `Shared Memory`, `GEMM` | **目标人群：** 核心 Infra 与算子开发

在 Triton 中，我们在 Python 里通过 `tl.load()` 拉取数据到 SRAM 并在上面算点积。Triton 编译器帮我们自动处理了所有的底层痛点：共享内存的分配、跨线程的数据同步、甚至 Bank Conflict (存储体冲突) 的避免。
然而，要真正理解 GPU 的优化极限，你必须手写一次 **Shared Memory Tiling (共享内存分块矩阵乘法)**。
本节我们将继续使用 C++ JIT 编译，在原生 CUDA 中显式声明 `__shared__` 数组，让 Block 内的 Threads 协作搬运数据，并通过 `__syncthreads()` 完成极速矩阵乘！


### Step 1: 共享内存在 CUDA 中的使用

> **物理位置：**
> 共享内存位于 SM (流多处理器) 芯片内部，比全局显存 (HBM/VRAM) 快近 100 倍！但容量极小。

> **协作搬运 (Collaborative Loading)：**
> 要计算 $C$ 的一个 $16 \times 16$ 小块。我们需要从 HBM 把 $A$ 对应行的一块和 $B$ 对应列的一块搬进 SRAM。
> - 在 CUDA 里，有 $16 \times 16 = 256$ 个线程负责这块 $C$。
> - 我们可以让每个线程只负责搬运 $A$ 块里的 1 个元素和 $B$ 块里的 1 个元素！(这是最优雅的协作)。
> - 搬完后，**必须**调用 `__syncthreads();`，保证整个 Block 的 256 个线程都搬完了，然后大家再一起开开心心地在 SRAM 里算点积累加。

> **变量修饰符：**
> 在 C++ 代码中，只需在变量前加上 `__shared__` 关键字，GPU 就会将其分配到 SRAM。


### Step 2: 共享内存 与分块乘法的究极奥义
如果直接写 CUDA 版的矩阵乘法，将遭受灾难性的 Global Memory（HBM）带宽惩罚。为了榨干硬件，必须借用 Block 级共享内存（Shared Memory，访问延迟仅约10周期）。将原矩阵划分为二维 Tile（瓷砖）。每次外层循环中，整个 Block 内的线程通力合作，把瓷砖搬进 Shared Memory，调用 `__syncthreads()` 后利用内部点积计算累加和。


### Step 3: 共享内存矩阵乘代码框架
声明 `__shared__ float s_A[TILE_SIZE][TILE_SIZE];` 和 `s_B`。在迭代过程中，利用二维的 `threadIdx.x` 和 `threadIdx.y` 将数据从内存拖拽到 SRAM 中。完成同步屏障后，用一个很短的 for 循环计算该子块的点积，最后覆盖写入结果矩阵。


###  Step 4: 动手实战

**要求**：请补全 `cuda_shared_gemm_source`。实现著名的 Shared Memory Tiled GEMM（共享内存分块矩阵乘法）。


```python
import torch
from torch.utils.cpp_extension import load_inline

# ==========================================
# 编写基于 Shared Memory 优化的矩阵乘法 (Square Matrix)
# 假设矩阵维度都是 TILE_SIZE 的倍数
# ==========================================
cuda_shared_gemm_source = '''
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void shared_gemm_kernel(const float* A, const float* B, float* C, int N) {
    // 1. 确定当前线程负责计算 C 矩阵中的哪个元素的行号和列号
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // 2. 声明位于超高速 SRAM 中的共享内存数组
    // 这些数组是被同一个 Block 中的 256 个线程 (16x16) 共享的！
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];
    
    float sum = 0.0f;
    
    // 3. 在 K 维度上切块 (Tiling) 进行循环
    int num_tiles = N / TILE_SIZE;
    
    for (int t = 0; t < num_tiles; ++t) {
        // ==========================================
        // TODO 1: 协作搬运数据 (从 HBM 读到 Shared Memory)
        // 每个线程负责从 A 搬运一个元素到 s_A，从 B 搬运一个元素到 s_B
        // 提示:
        // A 的行是 row，列是 t * TILE_SIZE + threadIdx.x
        // B 的行是 t * TILE_SIZE + threadIdx.y，列是 col
        // ==========================================
        // s_A[threadIdx.y][threadIdx.x] = ???
        // s_B[threadIdx.y][threadIdx.x] = ???
        s_A[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        s_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        
        // ==========================================
        // TODO 2: 线程同步！绝对不能少！
        // 必须等 Block 里所有人都搬完了，才能开始算。
        // ==========================================
        // ???
        __syncthreads();
        
        // ==========================================
        // TODO 3: SRAM 内部极速矩阵点积累加
        // 遍历这一个 Tile (长度为 TILE_SIZE) 进行累加
        // ==========================================
        for (int k = 0; k < TILE_SIZE; ++k) {
            // sum += ???
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }
        
        // ==========================================
        // TODO 4: 再次同步！
        // 等大家都算完这个 Tile 了，才能进入下一次 t 循环去覆盖 s_A 和 s_B
        // ==========================================
        // ???
        __syncthreads();
    }
    
    // 4. 将累加结果写回全局显存 (HBM)
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor shared_gemm_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0); // 为了教学简化，假设是 NxN 矩阵
    auto C = torch::empty_like(A);
    
    // 线程块大小：16x16 = 256 threads
    dim3 threads(TILE_SIZE, TILE_SIZE);
    
    // Grid 大小
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    shared_gemm_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>(), 
        N
    );
    
    return C;
}
'''

cpp_shared_source = '''
torch::Tensor shared_gemm_cuda(torch::Tensor A, torch::Tensor B);
'''

# 编译扩展
print("⏳ 正在编译带有 Shared Memory 的 CUDA GEMM，请稍候...")
try:
    shared_gemm_extension = load_inline(
        name='shared_gemm_ext',
        cpp_sources=cpp_shared_source,
        cuda_sources=cuda_shared_gemm_source,
        functions=['shared_gemm_cuda'],
        with_cuda=True,
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3']
    )
    print("✅ 编译成功！")
except Exception as e:
    print(f"❌ 编译失败: {e}")

```

```python
# 测试并体会极速 SRAM 的魅力
def test_shared_gemm():
    if not torch.cuda.is_available() or 'shared_gemm_extension' not in globals():
        print("⏭️ 忽略测试：无 GPU 或编译失败。")
        return
        
    try:
        torch.manual_seed(42)
        # N 必须是 16 的倍数以契合此简单 Kernel 的假设
        N = 1024 
        
        A = torch.randn(N, N, device='cuda', dtype=torch.float32)
        B = torch.randn(N, N, device='cuda', dtype=torch.float32)
        
        # 1. PyTorch 原生计算
        C_ref = A @ B
        
        # 2. 我们手写的高级 Shared Memory CUDA 计算
        C_cu = shared_gemm_extension.shared_gemm_cuda(A, B)
        
        diff = torch.max(torch.abs(C_ref - C_cu))
        assert diff < 1e-2, "Shared Memory GEMM 计算结果错误！"
        
        print("✅ 恭喜！你成功用最硬核的原生 CUDA C++ 管理了 GPU 的 Shared Memory！")
        print("💡 面试官一定会问：为什么要用 __syncthreads() 两次？如果你能解释出'防脏读'和'防早覆盖'的读写同步逻辑，这把稳了！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

test_shared_gemm()

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

Shared Memory (SRAM) 是突破 GPU 算力瓶颈的关键。比起每次都从慢速的显存 (HBM) 读数据，把一个小块数据搬到 SRAM 供大家复用，能带来数量级的性能提升。

1.  **数据搬运公式 ()**:
    *    是当前 Block 内部 6 \times 16$ 的小区域。
    *   全局矩阵  的行固定为当前线程负责的全局 。
    *   全局矩阵  的列则是在每次  循环中平移：。
    *   大家各自搬一个数，瞬间把这个 Tile 装满了！
2.  **第一次 **:
    *   **防脏读 (Read-after-Write Hazard)**。
    *   如果不加这句，有的线程跑得快，已经开始执行后面的计算循环  了。
    *   但是！别的跑得慢的线程可能**还没把数据写进 **！这会导致跑得快的线程读到了未初始化的脏数据。
3.  **计算循环 ()**:
    *   在极速的 SRAM 内部跑一个长度为 16 的点积循环。因为数据都在芯片内，没有 VRAM 带宽瓶颈。
4.  **第二次 **:
    *   **防覆盖 (Write-after-Read Hazard)**。
    *   算完这个 Tile 后，要进入下一次  循环，覆盖写入新的 Tile 数据到  中。
    *   如果不加这句，跑得快的线程已经进入下一次循环并覆盖了 ，而跑得慢的线程**还在算上一个 Tile 的点积**！慢线程就会读到被覆盖的新数据，导致计算错误。


```python
cuda_shared_gemm_source = '''
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void shared_gemm_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];
    
    float sum = 0.0f;
    int num_tiles = N / TILE_SIZE;
    
    for (int t = 0; t < num_tiles; ++t) {
        // 1. 协作搬运：每个线程从 HBM 搬 1 个元素到 SRAM
        s_A[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        s_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        
        // 2. 第一次同步：等所有人都搬完，防止有人提前去读没写好的数据 (防脏读)
        __syncthreads();
        
        // 3. 极速点积
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }
        
        // 4. 第二次同步：等所有人都算完，防止有人提前进入下一轮循环覆盖掉老数据 (防覆盖)
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor shared_gemm_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::empty_like(A);
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    shared_gemm_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>(), 
        N
    );
    
    cudaDeviceSynchronize();
    return C;
}
'''
```

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
