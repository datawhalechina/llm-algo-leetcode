# 03. GPU Architecture and Memory | GPU 物理架构、内存层级与核心硬件单元

**难度：** Hard | **环境：** GPU optional | **标签：** `硬件架构`, `GPU`, `内存层级` | **目标人群：** 核心 Infra 与算子开发

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


先抓住 GPU 的物理层级、Tensor Core 和 HBM / SRAM 之间的数量级差异，再去看 Triton/CUDA 算子和 FlashAttention，硬件直觉会更稳。

**关键词：** `Tensor Core`, `SRAM`, `HBM`, `memory hierarchy`, `arithmetic intensity`

这一页会进一步拆开 GPU 的物理架构、内存结构 (SRAM vs HBM)，以及它们在现代大模型算法（如 FlashAttention）中的实际应用。

## 前置
**导语：** 这一页主要承接单卡硬件、访存优化和性能分析，所以最好先把 1B、1D 和 profiling 的直觉接上，再看物理架构细节。

- [Part 1: 1B 单卡硬件与访存优化](./1B.md)
- [Part 1: 1D 异构调度与算子编程](./1D.md)
- [Part 1: 13 Profiling and Bottleneck Analysis](./13_Profiling_and_Bottleneck_Analysis.md)

## 相关阅读
**导语：** 如果想把“硬件层级 -> 访存瓶颈 -> 优化方法”这条线继续往前推，可以接着看：
- [Part 1: 19 Operator Fusion Introduction](./19_Operator_Fusion_Introduction.md)：先看算子融合怎么减少搬运。
- [Part 1: 24 SRAM Optimization Techniques](./24_SRAM_Optimization_Techniques.md)：再看片上缓存和 SRAM 如何影响性能。
- [Part 1: 04 Attention Variants & Memory Optimization](./04_Attention_Memory_Optimization.md)：最后回到注意力变体和显存优化。
## Q1：简述自 V100 以来 NVIDIA GPU 架构的演进，以及为了适应大模型计算做出了哪些核心改变？

<details>
<summary>点击展开查看解析</summary>

NVIDIA 的 GPU 架构代际演进，本质上是为了适应深度学习（尤其是 Transformer）对**混合精度矩阵计算**和**极高显存带宽**的持续攀升需求。

*   **Volta 架构 (V100 - 2017)**: 
    *   **关键引入**：首次引入了专为深度学习矩阵乘加 (MMA) 设计的 **Tensor Core (张量核心)**，支持 FP16 混合精度计算。
*   **Ampere 架构 (A100 - 2020)**:
    *   **常见能力**：支持了 **TF32 (Tensor Float 32)** 和更广泛的 FP16/BF16。
    *   **架构升级**：提升了 HBM2e (High Bandwidth Memory) 的带宽，并扩大了片上缓存容量（例如 L2 Cache 可达 40MB）。A100 时代的官方规格已经把 FP16 Tensor Core 算力推到 312 TFLOPS 量级，同时把 HBM 带宽推到 1.5 TB/s 级别。它还引入了 MIG (多实例 GPU) 和非对称稀疏化 (Sparse Tensor Core)。
*   **Hopper 架构 (H100 - 2022)**:
    *   **专为 LLM 而生**：引入了原生的 **FP8 数据格式**和 **Transformer Engine**。
    *   **内存与调度**：加入了 Thread Block Cluster 和 TMA (Tensor Memory Accelerator)，允许在不经过寄存器的情况下直接进行 HBM 到 SRAM 的异步数据搬运，进一步缓解了带宽压力。H100 的官方规格把 FP8 Tensor Core 算力推到 1,979 TFLOPS，HBM3 带宽可达 3.35 TB/s，NVLink 带宽可达 900 GB/s。
*   **Blackwell 架构 (B100/B200 - 2024)**:
    *   **针对生成式 AI 的进一步优化**：引入了第二代 Transformer Engine，原生支持更低精度的 **FP4 计算格式**，为单卡推理吞吐提升提供了更高上限。
    *   **通信与互连升级**：第五代 NVLink 双向带宽提升到 1.8 TB/s 级别（不同平台实现会有差异），为大规模模型集群提供了更宽的互连带宽上限。
</details>
### Q1小验证：GPU 内存层级分析

把 shared memory、L2 cache 和 HBM 的数量级差异算清楚，再看瓶颈主要出在哪一层。

```python
import torch
from typing import Dict

def bytes_to_gb(bytes_val: float) -> float:
    return bytes_val / 1e9

# GPU 内存层级的带宽（字节/秒）
MEMORY_BANDWIDTH = {
    'shared_memory': 19e12,  # 19 TB/s (A100)
    'l2_cache': 1.5e12,      # 1.5 TB/s
    'hbm': 1.5e12,           # 1.5 TB/s (A100)
}

def analyze_memory_hierarchy() -> Dict[str, Dict[str, float]]:
    """
    分析 GPU 内存层级的性能特性。
    """
    result = {}
    
    for mem_type, bandwidth in MEMORY_BANDWIDTH.items():
        # ==========================================
        # TODO 1.1: 计算访问延迟（访问 1 KB 数据）
        # latency_ns = (1024 / bandwidth) * 1e9
        # ==========================================
        latency_ns = (1024 / bandwidth) * 1e9
        
        result[mem_type] = {
            'bandwidth_tb_s': bandwidth / 1e12,
            'latency_ns': latency_ns,
        }
    
    return result

# 测试
mem_analysis = analyze_memory_hierarchy()
for mem_type, stats in mem_analysis.items():
    print(f"{mem_type:15s}: {stats['bandwidth_tb_s']:6.1f} TB/s, Latency: {stats['latency_ns']:6.2f} ns")
```

### 数量级速览

| 代际 | 关键变化 | 代表性指标 |
| --- | --- | --- |
| V100 | 首次引入 Tensor Core | 支持 FP16 MMA，开启了深度学习混合精度时代 |
| A100 | 带宽和片上缓存显著增强 | FP16 Tensor Core 可达 312 TFLOPS，HBM 带宽约 1.5 TB/s 级别 |
| H100 | FP8 + 更强调度与搬运机制 | FP8 Tensor Core 可达 1,979 TFLOPS，HBM 带宽可达 3.35 TB/s，NVLink 可达 900 GB/s |
| Blackwell | 更低精度与更强互连 | 原生 FP4，NVLink 提升到 1.8 TB/s 级别（平台实现有差异） |

这一张表是为了把 GPU 架构代际演进的数量级直觉和后面的内存层级、Attention 访存模式接起来。
## Q2：什么是 Tensor Core？它与普通的 CUDA Core 有何本质区别，为什么能明显加速矩阵计算？

<details>
<summary>点击展开查看解析</summary>

**普通 CUDA Core vs Tensor Core**
*   **CUDA Core (FP32/INT32)**: 每次时钟周期只能执行一个标量的 FMA (Fused Multiply-Add，乘加) 操作：`d = a * b + c`。
*   **Tensor Core (FP16/BF16/FP8)**: 专为矩阵乘法设计。在单个时钟周期内，它可以执行一个完整的 $4 \times 4$ 矩阵的 MMA (Matrix Multiply-Accumulate) 操作：`D = A * B + C`。

**为什么它这么快？**
Tensor Core 利用了半精度 (FP16) 或更低精度 (FP8) 来加速乘法，同时使用单精度 (FP32) 的累加器来保证加法精度。由于 Transformer 的自注意力和 MLP 几乎全是密集的矩阵乘法 (GEMM)，Tensor Core 的算力在这类场景下通常会显著高于普通 CUDA Core（例如 A100 的 FP16 Tensor Core 算力可达 312 TFLOPs）。

更直白地说，Tensor Core 不是“把标量 FMA 做快一点”，而是把一批矩阵乘加打包成更大的 MMA 一次完成，所以它在 GEMM 这种高复用、密集计算任务上特别占优。
</details>
### Q2小验证：Attention 显存占用计算

先拆出 Q/K/V、Attention 矩阵和输出的显存，再看为什么标准 Attention 会迅速变成 OOM 热点。

```python
def calculate_attention_vram(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype_bytes: int = 2
) -> Dict[str, float]:
    """
    计算标准 Attention 的显存占用。
    """
    # ==========================================
    # TODO 2.1: 计算 Q、K、V 的显存占用
    # qkv_vram = 3 * seq_len * num_heads * head_dim * dtype_bytes
    # ==========================================
    qkv_vram = 3 * seq_len * num_heads * head_dim * dtype_bytes
    
    # ==========================================
    # TODO 2.2: 计算 Attention 矩阵的显存占用
    # attention_matrix_vram = seq_len * seq_len * dtype_bytes
    # ==========================================
    attention_matrix_vram = seq_len * seq_len * dtype_bytes
    
    # ==========================================
    # TODO 2.3: 计算输出的显存占用
    # output_vram = seq_len * num_heads * head_dim * dtype_bytes
    # ==========================================
    output_vram = seq_len * num_heads * head_dim * dtype_bytes
    
    # ==========================================
    # TODO 2.4: 总显存占用
    # total_vram = qkv_vram + attention_matrix_vram + output_vram
    # ==========================================
    total_vram = qkv_vram + attention_matrix_vram + output_vram
    
    return {
        'qkv': bytes_to_gb(qkv_vram),
        'attention_matrix': bytes_to_gb(attention_matrix_vram),
        'output': bytes_to_gb(output_vram),
        'total': bytes_to_gb(total_vram),
    }

# 测试
print("标准 Attention 显存占用:")
for seq_len in [512, 4096, 128*1024]:
    vram = calculate_attention_vram(seq_len, 32, 128)
    print(f"  seq_len={seq_len:6d}: {vram['total']:10.2f} GB")
```

## Q3：请描述 GPU 的内存层级结构 (Memory Hierarchy)，并解释为什么大模型推理通常是 Memory Bound (访存受限) 的？

<details>
<summary>点击展开查看解析</summary>

GPU 的内存结构像一个金字塔，越靠近计算单元的速度越快，但容量越小：

1.  **Registers (寄存器)**：
    *   速度最快（<1 个周期），容量极小（每个线程几十个 32-bit 寄存器）。
    *   如果变量太多发生 **Register Spilling (寄存器溢出)**，数据会被回退到较慢的 Local Memory (物理上位于 HBM)。
2.  **Shared Memory (SRAM / 片上共享内存)**：
    *   速度极快（~19 TB/s），每个 SM (流多处理器) 只有几百 KB。
    *   **很关键**：它是同一个 Block 内所有线程协作、交换数据的主要高速通道。**Triton 的一个重要作用，就是帮你自动化管理 SRAM 的分配和调度。**
3.  **L2 Cache**: 
    *   所有 SM 共享，几十 MB，用于缓冲 HBM 的读写。
4.  **HBM (全局显存 / Global Memory)**:
    *   容量大 (40GB ~ 80GB)，但速度相对极慢 (1.5 TB/s ~ 3 TB/s)。
    *   如果算子的每一次计算都需要去 HBM 走一遭（如 PyTorch 原生的多次小操作），就会触发严重的 **Memory Bound (访存受限)**。

更好记的判断方式是看**算术强度**：

```text
Arithmetic Intensity = FLOPs / Bytes
```

如果一个算子的算术强度很低，就说明它每搬一次数据，只做了很少的计算，通常更容易被 HBM 带宽卡住；如果算术强度足够高，计算单元才更容易跑满。
</details>
### Q3小验证：FlashAttention 的显存节省

对比标准 Attention 和 FlashAttention，看看 Tiling + Online Softmax 如何把显存复杂度压回到 O(N)。

```python
def calculate_flash_attention_vram(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype_bytes: int = 2
) -> Dict[str, float]:
    """
    计算 FlashAttention 的显存占用。
    """
    # ==========================================
    # TODO 3.1: 计算 Q、K、V 的显存占用
    # qkv_vram = 3 * seq_len * num_heads * head_dim * dtype_bytes
    # ==========================================
    qkv_vram = 3 * seq_len * num_heads * head_dim * dtype_bytes
    
    # ==========================================
    # TODO 3.2: FlashAttention 只需存储 Online Softmax 的中间值
    # online_softmax_vram = seq_len * num_heads * 2 * 4
    # ==========================================
    online_softmax_vram = seq_len * num_heads * 2 * 4
    
    # ==========================================
    # TODO 3.3: 计算输出的显存占用
    # output_vram = seq_len * num_heads * head_dim * dtype_bytes
    # ==========================================
    output_vram = seq_len * num_heads * head_dim * dtype_bytes
    
    # ==========================================
    # TODO 3.4: 总显存占用
    # total_vram = qkv_vram + online_softmax_vram + output_vram
    # ==========================================
    total_vram = qkv_vram + online_softmax_vram + output_vram
    
    return {
        'qkv': bytes_to_gb(qkv_vram),
        'online_softmax': bytes_to_gb(online_softmax_vram),
        'output': bytes_to_gb(output_vram),
        'total': bytes_to_gb(total_vram),
    }

# 测试
print("\nFlashAttention 显存占用:")
for seq_len in [512, 4096, 128*1024]:
    vram_std = calculate_attention_vram(seq_len, 32, 128)
    vram_flash = calculate_flash_attention_vram(seq_len, 32, 128)
    ratio = vram_std['total'] / vram_flash['total'] if vram_flash['total'] > 0 else float('inf')
    print(f"  seq_len={seq_len:6d}: 标准={vram_std['total']:10.2f} GB, Flash={vram_flash['total']:10.2f} GB, 节省={ratio:8.0f}x")

def test_gpu_memory_practice():
    mem = analyze_memory_hierarchy()
    assert 'shared_memory' in mem and 'hbm' in mem
    assert mem['shared_memory']['bandwidth_tb_s'] > mem['hbm']['bandwidth_tb_s']

    attn = calculate_attention_vram(512, 32, 128)
    flash = calculate_flash_attention_vram(512, 32, 128)
    assert attn['total'] > flash['total']
    assert attn['qkv'] > 0 and flash['online_softmax'] > 0
    print('✅ 03 GPU Architecture and Memory tests passed')

test_gpu_memory_practice()
```

## Q4：结合 GPU 的内存结构，解释 FlashAttention 是如何利用 SRAM 解决传统 Attention 的访存瓶颈的？

<details>
<summary>点击展开查看解析</summary>

在标准的自注意力机制中，$S = QK^T$ 产生了一个尺寸为 $N \times N$ 的巨大矩阵。
*   **PyTorch 原生**：计算出 $S$，把它**写回 HBM**；读取 $S$ 计算 Softmax，再**写回 HBM**；读取 Softmax 结果和 $V$，计算出最终结果。这种反复读写 $O(N^2)$ 大小数据的行为，直接导致了显存溢出 (OOM) 和速度极慢。

*   **FlashAttention 的底层逻辑 (Tiling + SRAM)**：
    1.  **切块 (Tiling)**：将巨大的 $Q, K, V$ 切成小块 (Blocks)，使得这些小块**刚好能塞进容量只有几百 KB 的 SRAM 中**。
    2.  **在 SRAM 内完成一切 (Fusion)**：把 $Q_{block}$ 和 $K_{block}$ 加载到 SRAM，利用 Tensor Core 算出 $S_{block}$。
    3.  **在线归约 (Online Softmax)**：在 SRAM 内部直接更新局部最大值和指数和，避免写回 $S$。
    4.  最后再乘以 $V_{block}$，把最终结果写回 HBM。
    
**结论**：把 $O(N^2)$ 的 HBM 读写明显压缩到接近 $O(N)$ 的读写。**FlashAttention 不是减少了计算量，而是通过 SRAM 缓解了 Memory Bound 的影响。**

对学习者来说，最重要的不是死记某个固定 GB 数，而是记住它把 attention 的 IO 模式从“反复搬运大矩阵”改成了“分块在 SRAM 中完成”。FlashAttention-2 再进一步优化了 work partitioning，因此在长序列场景里会更有优势。
</details>
### Q4小验证：FlashAttention 的分块尺度

```python
def attention_score_bytes(seq_len, dtype_bytes=2):
    # 这里只估算 attention score 矩阵的体积，便于和 SRAM tile 做量级对比。
    return seq_len * seq_len * dtype_bytes


def tile_bytes(block_size, dtype_bytes=2):
    return block_size * block_size * dtype_bytes

seq_len = 4096
block_size = 128
score_mb = attention_score_bytes(seq_len) / 1024 / 1024
tile_kb = tile_bytes(block_size) / 1024
reduction = attention_score_bytes(seq_len) / tile_bytes(block_size)

print(f'Naive score matrix: {score_mb:.1f} MB')
print(f'One {block_size}x{block_size} tile: {tile_kb:.1f} KB')
print(f'IO reduction factor (rough): {reduction:.0f}x')

```

## Q5：在多卡分布式集群中，节点内通信的 PCIe 和 NVLink 有什么区别？

<details>
<summary>点击展开查看解析</summary>

当单卡装不下模型时，我们需要分布式训练。GPU 之间的物理连接方式决定了通信带宽 (Communication Bound)：

*   **PCIe (外围组件互连)**：
    *   传统的插槽，带宽有限 (PCIe Gen4 双向 64 GB/s)。
    *   **拓扑痛点**：跨 GPU 通信通常需要经过 PCIe Switch 甚至 CPU，延迟高、带宽低。
*   **NVLink (NVIDIA 私有互连)**：
    *   专为 GPU-to-GPU 设计的高速通道。
    *   **A100 的 NVLink 3.0**：每条链路 50 GB/s，单卡 12 条，总双向带宽高达 **600 GB/s**。这比 PCIe 快了近 10 倍。
    *   **H100 的 NVLink 4.0**：总双向带宽可达 **900 GB/s**。
    *   **Blackwell / NVLink 5**：带宽进一步提升到 **1.8 TB/s** 级别。
    *   **NVSwitch**：允许同一台物理机内的 8 张 GPU 实现全互连 (All-to-All) 的无阻塞通信，这是跑满 `All-Reduce` 和 `All-Gather` 极限带宽的硬件基础。

</details>
## ⚠️ 常见误区

- `Shared Memory` 比 `L2` 快，不代表可以把所有数据都塞进去；它更适合做局部块内复用。
- `HBM` 带宽已经很高，不代表就不会 `Memory Bound`；在高算力 GPU 上，带宽反而更容易成为瓶颈。
- `FlashAttention` 主要减少的是 HBM 访问，不是把主要计算量“变没了”。
- `NVLink` 很快，但仍然需要正确的通信库、拓扑和并行策略配合，否则并不会自动接近跑满。


```python
def pcie_vs_nvlink(payload_mb, pcie_gbps=64, nvlink_gbps=900):
    # 带宽差异真正影响的是把一块数据搬过去要花多少时间。
    pcie_ms = payload_mb * 8 / pcie_gbps
    nvlink_ms = payload_mb * 8 / nvlink_gbps
    return {'pcie_ms': round(pcie_ms, 2), 'nvlink_ms': round(nvlink_ms, 2), 'speedup': round(pcie_ms / nvlink_ms, 1)}

for payload in [64, 256, 1024]:
    print(payload, 'MB ->', pcie_vs_nvlink(payload))
print('higher bandwidth only matters when the transfer is on the critical path')

```
