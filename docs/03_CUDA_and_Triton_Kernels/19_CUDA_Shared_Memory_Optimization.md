# 19. CUDA Shared Memory Optimization | 榨干硬件极限：CUDA Shared Memory (共享内存) 优化与 GEMM

**难度：** Hard | **标签：** `CUDA C++`, `Shared Memory`, `GEMM` | **目标人群：** 核心 Infra 与算子开发

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/03_CUDA_and_Triton_Kernels/19_CUDA_Shared_Memory_Optimization.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


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
> - 搬完后，**必须**调用 `__syncthreads();`，保证整个 Block 的 256 个线程都搬完了，然后所有线程就可以在 SRAM 里进行点积累加计算。

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
```


```python
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
        // 提示: A的行是row，列是t*TILE_SIZE+threadIdx.x；B的行是t*TILE_SIZE+threadIdx.y，列是col
        // ==========================================
        // s_A[threadIdx.y][threadIdx.x] = ???;
        // s_B[threadIdx.y][threadIdx.x] = ???;
        
        // ==========================================
        // TODO 2: 线程同步 (防脏读)
        // 提示: 使用 __syncthreads() 确保所有线程都完成数据搬运
        // ==========================================
        // ???
        
        // ==========================================
        // TODO 3: SRAM 内部极速矩阵点积累加
        // 提示: 遍历TILE_SIZE，累加 s_A[threadIdx.y][k] * s_B[k][threadIdx.x]
        // ==========================================
        // for (int k = 0; k < TILE_SIZE; ++k) {
        //     sum += ???;
        // }
        
        // ==========================================
        // TODO 4: 再次同步 (防覆盖)
        // 提示: 使用 __syncthreads() 确保所有线程都完成计算
        // ==========================================
        // ???
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
print(" 正在编译带有 Shared Memory 的 CUDA GEMM，请稍候...")
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
    # 检查TODO是否完成
    if '// s_A[threadIdx.y][threadIdx.x] = ???;' in cuda_shared_gemm_source or '// ???' in cuda_shared_gemm_source:
        raise NotImplementedError("请先完成 TODO 1-4")
    
    if not torch.cuda.is_available():
        print("⏭️ 无 GPU，跳过测试")
        return
    
    if 'shared_gemm_extension' not in globals():
        raise RuntimeError("CUDA 扩展编译失败，请检查 nvcc 是否安装")
    
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
    
    print("✅ Shared Memory GEMM 验证通过。")
    print("工程实践：两次__syncthreads()分别防止脏读和数据覆盖，是Shared Memory编程的关键。")

test_shared_gemm()
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
        // TODO 1: 协作搬运数据 (从 HBM 读到 Shared Memory)
        s_A[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        s_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        
        // TODO 2: 第一次同步 (防脏读)
        __syncthreads();
        
        // TODO 3: SRAM 内部极速矩阵点积累加
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }
        
        // TODO 4: 第二次同步 (防覆盖)
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

### 解析

**1. TODO 1: 协作搬运数据**
- **实现方式**: 
  ```cpp
  s_A[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
  s_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
  ```
- **关键点**: 
  - 每个线程负责搬运一个元素到Shared Memory
  - 256个线程协作，瞬间填满16×16的Tile
  - A的行固定为row，列在每次循环中平移
  - B的列固定为col，行在每次循环中平移
- **技术细节**: 
  - Shared Memory位于SM芯片内部，比HBM快100倍
  - 协作搬运是Tiling优化的核心
  - 每个线程只搬运1个元素，负载均衡

**2. TODO 2: 第一次同步（防脏读）**
- **实现方式**: `__syncthreads();`
- **关键点**: 
  - 确保所有线程都完成数据搬运
  - 防止快线程读到慢线程未写入的脏数据
  - Block内的全局同步屏障
- **技术细节**: 
  - Read-after-Write Hazard（RAW）
  - 如果不同步，快线程可能读到未初始化的s_A/s_B
  - __syncthreads()是Block级同步，不能跨Block

**3. TODO 3: SRAM内部极速矩阵点积累加**
- **实现方式**: 
  ```cpp
  for (int k = 0; k < TILE_SIZE; ++k) {
      sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
  }
  ```
- **关键点**: 
  - 在Shared Memory中进行点积计算
  - 避免重复访问HBM
  - 每个线程计算C矩阵的一个元素
- **技术细节**: 
  - Shared Memory访问延迟约10周期
  - HBM访问延迟约400-800周期
  - 性能提升来自数据复用

**4. TODO 4: 第二次同步（防覆盖）**
- **实现方式**: `__syncthreads();`
- **关键点**: 
  - 确保所有线程都完成计算
  - 防止快线程提前进入下一轮循环覆盖s_A/s_B
  - 保护慢线程正在使用的数据
- **技术细节**: 
  - Write-after-Read Hazard（WAR）
  - 如果不同步，快线程会覆盖慢线程正在读的数据
  - 导致计算结果错误

**工程优化要点**

- **Shared Memory大小限制**:
  - 每个SM的Shared Memory有限（48KB-164KB）
  - TILE_SIZE=16时，两个Tile占用2KB
  - 过大的TILE_SIZE会限制Block并发数
  - 需要权衡Tile大小和SM占用率

- **Bank Conflict避免**:
  - Shared Memory分为32个Bank
  - 同一warp内的线程同时访问同一Bank会冲突
  - 本例中：s_A[threadIdx.y][k]和s_B[k][threadIdx.x]
  - 访问模式良好，无Bank Conflict

- **Tiling策略优化**:
  - TILE_SIZE选择：8, 16, 32是常见值
  - 过小：Shared Memory利用率低
  - 过大：寄存器压力大，SM占用率低
  - 最优值需要实验确定

- **性能对比分析**:
  - Naive GEMM：每个元素从HBM读N次
  - Shared Memory GEMM：每个元素从HBM读1次
  - 理论加速比：接近N/TILE_SIZE倍
  - 实际加速比：5-10倍（受其他因素影响）

- **内存访问模式优化**:
  - 协作搬运实现了coalesced access
  - 连续线程访问连续内存
  - 最大化内存带宽利用率

- **寄存器使用优化**:
  - sum变量存储在寄存器中
  - 避免频繁访问Shared Memory
  - 减少延迟

- **Double Buffering技术**:
  - 高级优化：使用两组Shared Memory
  - 在计算当前Tile时，预取下一个Tile
  - 进一步隐藏内存延迟
  - 需要更多Shared Memory

```python
cpp_shared_source = '''
torch::Tensor shared_gemm_cuda(torch::Tensor A, torch::Tensor B);
'''

# 编译扩展
print("⏳ 正在编译带有 Shared Memory 的 CUDA GEMM，请稍候...")
try:
    shared_gemm_extension = load_inline(
        name='shared_gemm_ext_answer',
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
    shared_gemm_extension = None

# 测试函数
def test_shared_gemm():
    if not torch.cuda.is_available():
        print("⏭️ 无 GPU，跳过测试")
        return
    
    if shared_gemm_extension is None:
        raise RuntimeError("CUDA 扩展编译失败，请检查 nvcc 是否安装")
    
    torch.manual_seed(42)
    N = 1024
    
    A = torch.randn(N, N, device='cuda', dtype=torch.float32)
    B = torch.randn(N, N, device='cuda', dtype=torch.float32)
    
    C_ref = A @ B
    C_cu = shared_gemm_extension.shared_gemm_cuda(A, B)
    
    diff = torch.max(torch.abs(C_ref - C_cu))
    assert diff < 1e-2, f"Shared Memory GEMM 计算结果错误！差异: {diff}"
    
    print("✅ Shared Memory GEMM 验证通过。")
    print("工程实践：两次__syncthreads()分别防止脏读和数据覆盖，是Shared Memory编程的关键。")

test_shared_gemm()
```

### 思考与讨论

**1. 为什么需要两次__syncthreads()？**

在Shared Memory编程中，两次同步是必需的。缺少任何一次都会导致数据竞争。

思考以下问题：
- 第一次同步保护什么？
- 第二次同步保护什么？
- 如果只用一次同步会怎样？

**提示**: 考虑读写依赖关系（RAW和WAR）。

**答案**:

**第一次__syncthreads()：防脏读（Read-after-Write Hazard）**

| 时间 | 快线程 | 慢线程 | 问题 |
|------|--------|--------|------|
| T1 | 写s_A[0][0]完成 | 还在写s_A[15][15] | - |
| T2 | 开始读s_A[0][15] | 还在写s_A[15][15] | ❌ 快线程读到未初始化的数据 |

**解决**: 第一次__syncthreads()确保所有线程都完成写入，才允许任何线程开始读取。

**第二次__syncthreads()：防覆盖（Write-after-Read Hazard）**

| 时间 | 快线程 | 慢线程 | 问题 |
|------|--------|--------|------|
| T1 | 计算完成 | 还在读s_A[15][15] | - |
| T2 | 进入下一轮，写s_A[0][0] | 还在读s_A[0][0] | ❌ 快线程覆盖了慢线程正在读的数据 |

**解决**: 第二次__syncthreads()确保所有线程都完成读取，才允许任何线程开始覆盖。

**工程启示**: 
- Shared Memory编程必须仔细分析读写依赖
- 缺少同步会导致难以调试的数据竞争
- 过多同步会降低性能，需要精确控制

**2. 什么是Bank Conflict？如何避免？**

Shared Memory分为32个Bank，同一warp内的线程同时访问同一Bank会导致串行化。

思考以下问题：
- Bank Conflict如何影响性能？
- 本例中是否存在Bank Conflict？
- 如何设计访问模式避免冲突？

**提示**: 考虑Shared Memory的硬件组织方式。

**答案**:

**Bank Conflict原理**:
- Shared Memory分为32个Bank（对应warp大小）
- 每个Bank宽度4字节（一个float）
- 地址映射：`Bank ID = (address / 4) % 32`
- 同一warp内多个线程访问同一Bank → 串行执行

**性能影响**:
- 无冲突：1个周期
- 2-way冲突：2个周期
- 32-way冲突：32个周期（性能降低32倍！）

**本例分析**:

读取s_A时：
```cpp
s_A[threadIdx.y][k]  // 同一warp内，threadIdx.y相同，k相同
// 所有线程访问同一个元素 → Broadcast，无冲突
```

读取s_B时：
```cpp
s_B[k][threadIdx.x]  // threadIdx.x = 0,1,2,...,31
// 连续线程访问连续地址 → 无冲突
```

**结论**: 本例访问模式良好，无Bank Conflict。

**避免策略**:
1. **Padding**: 在数组维度上加1，改变Bank映射
2. **转置**: 调整数据布局
3. **访问模式设计**: 确保连续线程访问不同Bank

**3. 如何选择最优的TILE_SIZE？**

TILE_SIZE影响性能的多个方面，需要权衡。

思考以下问题：
- TILE_SIZE如何影响Shared Memory使用？
- TILE_SIZE如何影响SM占用率？
- 如何实验确定最优值？

**提示**: 考虑硬件限制和性能指标。

**答案**:

**TILE_SIZE的影响**:

| TILE_SIZE | Shared Memory | SM占用率 | 数据复用 | 性能 |
|-----------|---------------|---------|---------|------|
| 8 | 1KB | 高 | 低 | 中 |
| 16 | 4KB | 中 | 中 | 好 |
| 32 | 8KB | 低 | 高 | 最好（如果SM够用） |
| 64 | 32KB | 很低 | 很高 | 可能更差（SM不够） |

**权衡因素**:

1. **Shared Memory限制**:
   - 每个SM有48KB-164KB Shared Memory
   - TILE_SIZE=32时，两个Tile占用8KB
   - 如果每个SM只能运行1个Block，占用率低

2. **寄存器压力**:
   - 更大的TILE_SIZE需要更多寄存器
   - 寄存器不足会溢出到Local Memory
   - 性能急剧下降

3. **数据复用**:
   - TILE_SIZE越大，每个元素被复用次数越多
   - 减少HBM访问次数
   - 提升性能

**实验方法**:
```python
for tile_size in [8, 16, 32, 64]:
    # 修改TILE_SIZE并重新编译
    # 测试性能
    # 使用nvprof分析SM占用率、Bank Conflict等
```

**最优值**（经验）:
- 小矩阵（N<512）：TILE_SIZE=16
- 中等矩阵（512≤N<2048）：TILE_SIZE=32
- 大矩阵（N≥2048）：TILE_SIZE=32或64

**工程启示**:
- 没有万能的TILE_SIZE
- 需要根据硬件和问题规模调优
- 使用Profiling工具指导优化