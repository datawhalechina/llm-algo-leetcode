# 01. Triton Vector Addition | Triton 入门与 Hello World：向量加法 (Vector Addition)

**难度：** Easy | **标签：** `入门`, `Triton` | **目标人群：** 通用基础 (算法/Infra)

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/03_CUDA_and_Triton_Kernels/01_Triton_Vector_Addition.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


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

```


```python

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
    # 提示: 使用 pid 和 BLOCK_SIZE 计算块的起始位置
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
    # ==========================================
    # x = ???
    # y = ???
    
    # ==========================================
    # TODO 4: 在 SRAM 中进行向量加法，并将结果写回 HBM 的 z_ptr
    # ==========================================
    # z = ???
    pass

def triton_add(x: torch.Tensor, y: torch.Tensor):
    # 检查 kernel 是否已实现（通过检查源码中是否还有 pass）
    import inspect
    source = inspect.getsource(add_kernel.fn)
    if 'pass' in source and '# block_start = ???' in source:
        raise NotImplementedError("请完成 add_kernel 中的 TODO 1-4")
    
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
    if not torch.cuda.is_available():
        print("⏭️ 忽略测试：无 GPU。本节编译和运行 Triton/CUDA 算子必须依赖 NVIDIA GPU。")
        return
        
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
        
        # 边界测试
        print("\n---  边界情况测试 ---")
        
        # 测试1: 单元素向量
        x1 = torch.tensor([1.0], device='cuda')
        y1 = torch.tensor([2.0], device='cuda')
        z1 = triton_add(x1, y1)
        assert torch.allclose(z1, torch.tensor([3.0], device='cuda')), "单元素测试失败"
        print("✅ 单元素向量测试通过")
        
        # 测试2: 小向量（小于BLOCK_SIZE）
        x2 = torch.rand(100, device='cuda')
        y2 = torch.rand(100, device='cuda')
        z2 = triton_add(x2, y2)
        assert torch.allclose(x2 + y2, z2), "小向量测试失败"
        print("✅ 小向量测试通过")
        
        # 测试3: 恰好对齐BLOCK_SIZE
        x3 = torch.rand(1024, device='cuda')
        y3 = torch.rand(1024, device='cuda')
        z3 = triton_add(x3, y3)
        assert torch.allclose(x3 + y3, z3), "对齐测试失败"
        print("✅ BLOCK_SIZE对齐测试通过")
    
        print("\n---  性能基准测试 (Benchmark) ---")
        quantiles = [0.5, 0.2, 0.8]
        ms_pt, min_ms_pt, max_ms_pt = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
        ms_tr, min_ms_tr, max_ms_tr = triton.testing.do_bench(lambda: triton_add(x, y), quantiles=quantiles)
        print(f"PyTorch Time: {ms_pt:.4f} ms")
        print(f"Triton Time:  {ms_tr:.4f} ms")
        print(f"Speedup:      {ms_pt / ms_tr:.2f}x")
    except NotImplementedError:
        print("请先完成 TODO 代码！")
    except Exception as e:
        print(f"❌ 测试失败 (可能需要 CUDA 环境): {e}")

test_vector_add()
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
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
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

### 解析

**1. TODO 1: 计算内存偏移量**
- **实现方式**：`block_start = pid * BLOCK_SIZE`，`offsets = block_start + tl.arange(0, BLOCK_SIZE)`
- **关键点**：每个Block处理连续的BLOCK_SIZE个元素，通过pid确定起始位置
- **技术细节**：`tl.arange(0, BLOCK_SIZE)` 生成 [0, 1, ..., BLOCK_SIZE-1] 的向量，加上block_start得到全局索引

**2. TODO 2: 边界保护Mask**
- **实现方式**：`mask = offsets < n_elements`
- **关键点**：防止最后一个Block越界访问，避免Segmentation Fault
- **技术细节**：mask是布尔向量，标记哪些位置是合法的内存访问。当向量长度不是BLOCK_SIZE的整数倍时，最后一个Block会有部分元素超出边界

**3. TODO 3: 加载数据到SRAM**
- **实现方式**：`x = tl.load(x_ptr + offsets, mask=mask)`，`y = tl.load(y_ptr + offsets, mask=mask)`
- **关键点**：从HBM（显存）加载数据到SRAM（片上内存），使用mask保护越界访问
- **技术细节**：指针算术 `x_ptr + offsets` 计算每个元素的地址。Triton会自动将这个向量化的加载操作映射到GPU的内存事务

**4. TODO 4: 计算并存回HBM**
- **实现方式**：`z = x + y`，`tl.store(z_ptr + offsets, z, mask=mask)`
- **关键点**：在SRAM中进行向量化计算，然后写回HBM
- **技术细节**：Triton自动进行SIMD向量化，无需手动循环。`tl.store` 同样需要mask保护，确保只写入合法位置

**工程优化要点**
- **BLOCK_SIZE选择**：通常为2的幂次方（256、512、1024），需要根据GPU架构调优。过小会增加kernel启动开销，过大会降低occupancy
- **内存访问模式**：连续访问（coalesced access）可最大化带宽利用率。本例中每个Block访问连续的内存区域，符合最佳实践
- **Mask开销**：虽然mask增加了分支判断，但避免了越界访问的严重后果（程序崩溃）。现代GPU对predicated execution有良好支持
- **性能对比**：简单向量加法是memory-bound操作，Triton与PyTorch性能接近，因为瓶颈在内存带宽而非计算
- **Kernel融合**：向量加法通常不单独使用，而是与其他操作融合（kernel fusion）以减少内存往返次数
- **工业实践**：Triton的优势在于复杂kernel的开发效率，对于简单操作如向量加法，PyTorch的优化已经足够好