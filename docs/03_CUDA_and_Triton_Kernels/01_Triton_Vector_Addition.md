# 01 Triton Vector Addition

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/03_CUDA_and_Triton_Kernels/01_Triton_Vector_Addition.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 01. Triton 入门与 Hello World：向量加法 (Vector Addition)

**难度：** Easy | **标签：** `入门`, `Triton` | **目标人群：** 通用基础 (算法/Infra)

这是 Triton 编程的“Hello World”。在这里，你将抛弃平时在 PyTorch 里理所当然的 `z = x + y`，而是站在 GPU 硬件的视角，亲自控制“指针”、“内存偏移”和“线程块（Block）”，体会最原始的数据搬运。


### Step 1: 核心思想与编程模型

> **Triton 的核心理念：基于 Block 的编程**
> 在原生的 CUDA C++ 中，你需要精确控制几万个“线程 (Thread)”该去读哪个数组元素。
> 而在 Triton 中，你只需要控制一个“线程块 (Block)”。编译器会自动帮你把这个 Block 拆分给底层的 128 或 256 个线程去执行。

> **程序的维度 (Grid)**
> 假设我们要把两个长度为 10 万的向量相加。GPU 一次处理不了这么多。
> 我们把任务切分成很多个“Block”（比如每个 Block 处理 1024 个元素）。
> 那么我们需要启动 `100000 / 1024` 取整 个 Program。这就是所谓的 `Grid` 大小。


### Step 2: Triton 内核与寻址掩码框架
在 Triton 中，你需要使用 `tl.program_id` 确定当前线程块的 ID，并计算出所负责的数据指针偏移 `offs`。在加载和存储数据时，由于数组总长度不一定是 `BLOCK_SIZE` 的整数倍，必须生成一个类似 `offs < N` 的掩码变量 `mask` 并在 `tl.load` 和 `tl.store` 时传入以防止内存越界（Segmentation Fault）。


###  Step 3: 核心 API

Triton 的语法非常固定，通常包含以下三板斧：
1. **定位自己**：`pid = tl.program_id(0)` 获取当前是第几个 Block。
2. **计算偏移**：`offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` 计算当前 Block 应该去读显存里的哪些位置。
3. **边界保护 (Mask)**：如果向量长度不是 `BLOCK_SIZE` 的整数倍，最后一个 Block 就会越界报错。我们需要一个 `mask = offsets < N` 来保护显存读写。


###  Step 4: 动手实战

**要求**：请补全下方 `add_kernel`，实现向量 `Z = X + Y` 的底层计算。


```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,  # 输入向量 X 的首地址指针
    y_ptr,  # 输入向量 Y 的首地址指针
    z_ptr,  # 输出向量 Z 的首地址指针
    n_elements, # 向量的总长度 N
    BLOCK_SIZE: tl.constexpr, # 每个 Block 处理的元素个数
):
    # 1. 获取当前 Block 的 ID
    pid = tl.program_id(axis=0)
    
    # ==========================================
    # TODO 1: 计算当前 Block 需要处理的内存偏移量 (offsets)
    # 提示: block_start = pid * BLOCK_SIZE
    # 偏移量 = block_start + tl.arange(0, BLOCK_SIZE)
    # ==========================================
    # block_start = ???
    # offsets = ???
    
    # ==========================================
    # TODO 2: 计算边界保护 Mask
    # 提示: 只有当 offsets 小于 n_elements 时，才是合法的访问
    # ==========================================
    # mask = ???
    
    # ==========================================
    # TODO 3: 从 HBM (显存) 中将 x 和 y 加载到 SRAM (片上内存) 中
    # 提示: 使用 tl.load，传入带 offsets 的指针，并使用 mask
    # ==========================================
    # x = tl.load(x_ptr + offsets, mask=mask)
    # y = ???
    
    # ==========================================
    # TODO 4: 在 SRAM 中进行向量加法，并将结果写回 HBM 的 z_ptr
    # 提示: 使用 tl.store(指针, 值, mask)
    # ==========================================
    # z = ???
    # tl.store(???, z, mask=mask)
    pass

def triton_add(x: torch.Tensor, y: torch.Tensor):
    assert x.is_cuda and y.is_cuda and x.shape == y.shape
    n_elements = x.numel()
    z = torch.empty_like(x)
    
    # 定义 BLOCK_SIZE，通常为 2 的幂次方，如 1024
    BLOCK_SIZE = 1024
    # 计算需要启动多少个 Block (向上取整)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # 启动 Kernel
    add_kernel[grid](x, y, z, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return z

```

```python
# 测试你的实现
def test_vector_add():
    try:
        torch.manual_seed(0)
        size = 98432  # 故意不用 2 的幂次方，测试 mask 是否生效
        x = torch.rand(size, device='cuda')
        y = torch.rand(size, device='cuda')
        
        # PyTorch Native
        z_torch = x + y
        
        # Triton 
        z_triton = triton_add(x, y)
        
        assert torch.allclose(z_torch, z_triton), "Triton 算子输出不正确！"
        print("✅ Vector Addition (Hello World) 算子测试通过！Mask 边界处理正确！")
        
    
        print("\n--- ⚡ 性能基准测试 (Benchmark) ---")
        quantiles = [0.5, 0.2, 0.8]
        ms_pt, min_ms_pt, max_ms_pt = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
        ms_tr, min_ms_tr, max_ms_tr = triton.testing.do_bench(lambda: add_triton(x, y), quantiles=quantiles)
        print(f"PyTorch Time: {ms_pt:.4f} ms")
        print(f"Triton Time:  {ms_tr:.4f} ms")
        print(f"Speedup:      {ms_pt / ms_tr:.2f}x")
    except NotImplementedError:
        print("请先完成 TODO 代码！")
    except Exception as e:
        print(f"❌ 测试失败 (可能需要 CUDA 环境): {e}")

test_vector_add()

```

::: details 💡 点击查看官方解析与参考代码

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---


## 官方解析与参考代码

**解析：**
1. **TODO 1 (Offsets)：** Triton 是基于 Block 计算的（而不是 CUDA C++ 的线程计算）。每个 `pid` 处理连续的一块数据。`tl.arange(0, BLOCK_SIZE)` 会生成一个 `[0, 1, 2, ..., BLOCK_SIZE-1]` 的向量，加上基址 `block_start`，我们就得到了当前 block 需要处理的所有元素的全局索引。
2. **TODO 2 (Mask)：** 因为数组的总长度 `n_elements` 未必是 `BLOCK_SIZE` 的整数倍，最后一个 block 算出来的 offsets 会超出数组边界！因此需要一个布尔型掩码 `mask = offsets < n_elements` 保护内存，防止引发段错误 (Segmentation Fault)。
3. **TODO 3 (Load)：** SRAM 读取。直接利用刚才算出来的全局指针地址 `x_ptr + offsets`，配合 `mask` 加载到片上。
4. **TODO 4 (Compute & Store)：** Triton 自动做矢量化计算 `z = x + y`。最后使用 `tl.store` 加 `mask` 写回到 HBM。


```python
@triton.jit
def add_kernel_solution(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    # TODO 1: 计算内存偏移量
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # TODO 2: 边界保护 Mask
    mask = offsets < n_elements
    
    # TODO 3: 加载数据到 SRAM
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # TODO 4: 计算并存回 HBM
    z = x + y
    tl.store(z_ptr + offsets, z, mask=mask)

```

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
