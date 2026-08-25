# 14. FlashAttention Memory Model | FlashAttention 显存模型

**难度：** Medium | **环境：** CPU-first | **标签：** `推理优化`, `FlashAttention`, `显存模型` | **目标人群：** 系统性能入门者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

标准 attention 在长序列下的压力，往往不只来自公式本身，也来自中间结果和反复搬运；在特定形状和硬件上，HBM 带宽可能成为瓶颈。如果 `QK^T`、softmax 和后续乘法之间需要频繁落到显存，attention 就可能从“矩阵乘法问题”转成“访存组织问题”。

这一页在整个教程的纵向主线里属于 `Part 01` 的 attention 访存基础页，优先服务 `推理优化路线`，也给后续 kernel 实现页建立 SRAM 复用视角。学完这里，后面再看 `20 / 34 / 66` 以及相关 kernel 实现时，你会更容易把“长序列 attention 为什么慢”改写成“哪一段数据搬运最贵”；如果这里没学明白，后面容易把 prefill 慢误看成单纯算力问题，而忽略中间结果落地和 HBM 往返可能带来的影响。按专题归类，这一页主要属于 `推理优化专题`，也和 `编译与图优化专题` 共享一部分访存优化视角。本节是 **CPU-first；GPU 用于扩展验证**：CPU 模拟可以验证 attention 中间矩阵的规模、分块计算和访存次数的变化；真实 GPU 才能验证 kernel 融合、HBM 带宽、实际峰值显存、prefill 吞吐和不同序列长度下的收益。显存路线在这里重点观察 Attention 工作集和中间张量的增长，不把模拟结果直接写成 73–76 的项目结论。

**关键词：** `FlashAttention`, `tiling`, `SRAM`

对应显存优化路线的 Task1（显存与性能认知）以及推理优化路线的 Task2。它通常不直接进入 73–76；如果把 attention 访存策略放进训练 workload，应由 73 / 76 测量，由 74 用 profiler 解释，而不是把模拟节的结论直接当成项目结论。

---
## 前置阅读

**导语：** 先确认显存模型和 Attention 访存直觉，再看 FlashAttention 的分块改法会更顺；如果你正在走 `推理优化路线`，这一页会直接承接 `20 / 34 / 66`，因为后面 prefill 为什么会慢、chunked prefill 为什么有意义，本质上都要先回到这里的搬运模型。

- [Group 1B: Single-GPU Hardware and Memory Optimization | 1B: 单卡硬件与访存优化](./1B.md)
- [Group 1C: Distributed Communication and Memory Sharing | 1C: 多卡通信与显存共享](./1C.md)

## 相关阅读

**导语：** 如果想把 FlashAttention 的显存模型继续接到实现和推理验证上，可以沿这三页往下看。

- [20. FlashAttention Sim | FlashAttention 模拟](../02_PyTorch_Algorithms/20_FlashAttention_Sim.md)
- [08. Triton Flash Attention | 真正的 Flash Attention 前向算子](../03_Triton_Kernels/08_Triton_Flash_Attention.md)
- [34. Prefix Caching and Chunked Prefill | 前缀缓存与分块预填充](../02_PyTorch_Algorithms/34_Prefix_Caching_and_Chunked_Prefill.md)
---
## Q1：为什么标准 Attention 会在长序列下迅速变成显存瓶颈？

<details>
<summary>点击展开查看解析</summary>

标准 Attention 的问题，不是“算不动”，而是中间结果太大。

在长序列场景里，$QK^T$ 会生成一个接近 $N 	imes N$ 的注意力矩阵。如果这个矩阵要频繁写回 HBM，再读出来做 Softmax 和后续乘法，显存访问次数就会非常多。

这意味着两件事：
- 中间结果占用的显存会迅速膨胀；
- 数据搬运会比计算本身更容易成为瓶颈。

所以 FlashAttention 要解决的，不是“让矩阵更小”，而是“不要让大矩阵长期落到 HBM 上”。
</details>
### Q1小验证：大矩阵为什么危险

先从 $N 	imes N$ 的规模直觉开始，记住中间矩阵一旦落到 HBM，代价就会很高。

```python
def attention_score_bytes(seq_len, dtype_bytes=2):
    # 只估算 attention score 矩阵的体积，便于和 tile 的工作集做对比。
    return seq_len * seq_len * dtype_bytes

for n in [1024, 2048, 4096]:
    naive_gb = attention_score_bytes(n) / 1024 / 1024 / 1024
    print(f'seq_len={n:4d} -> score matrix ≈ {naive_gb:6.2f} GB')

```

## Q2：FlashAttention 为什么要用 tiling 和 online softmax？

<details>
<summary>点击展开查看解析</summary>

FlashAttention 的核心是把大问题拆成小块处理。

- **Tiling**：把 $Q$、$K$、$V$ 切成能放进片上 SRAM 的小块。
- **Online softmax**：在处理每个块时，就把局部最大值和指数和更新掉，避免把完整注意力矩阵写回 HBM。

这样做的本质是把“先生成完整矩阵，再统一做归约”的流程，改成“边算边归约”。

好处有两个：
1. 中间结果不需要长期停留在 HBM；
2. 计算和归约可以在更近的存储层完成，减少大规模搬运。

所以 FlashAttention 的关键不是算法名，而是它把处理顺序和存储层级重新安排了一遍。
</details>
### Q2小验证：分块之后为什么更稳

把一个大矩阵拆成多个小块，再区分一维 tile 数、二维 score tile 数和单个 score tile 的理论大小。这里的数值是工作集模型，不是完整 kernel 的 shared memory 或寄存器占用。

```python
def num_1d_tiles(seq_len, tile_size):
    """返回一条序列维度上的 tile 数；不代表 Attention 的二维 tile 总数。"""
    if seq_len <= 0 or tile_size <= 0:
        raise ValueError('seq_len 和 tile_size 必须为正数')
    return (seq_len + tile_size - 1) // tile_size

def num_score_tiles(seq_len, tile_size):
    """返回 Q tile 与 K tile 组合形成的二维 score tile 数。"""
    tiles_1d = num_1d_tiles(seq_len, tile_size)
    return tiles_1d * tiles_1d

def score_tile_bytes(tile_size, dtype_bytes=2):
    """只估算一个 score tile，不代表完整 FlashAttention 工作集。"""
    if tile_size <= 0 or dtype_bytes <= 0:
        raise ValueError('tile_size 和 dtype_bytes 必须为正数')
    return tile_size * tile_size * dtype_bytes


seq_len = 4096
for tile in [64, 128, 256]:
    tiles_1d = num_1d_tiles(seq_len, tile)
    score_tiles = num_score_tiles(seq_len, tile)
    score_tile_kb = score_tile_bytes(tile) / 1024
    print(f'tile={tile:3d} -> 1D tiles={tiles_1d:3d}, score tiles={score_tiles:4d}, score tile ≈ {score_tile_kb:6.1f} KB')

```

## Q3：为什么说 FlashAttention 是在把压力从 HBM 挪到 SRAM？

<details>
<summary>点击展开查看解析</summary>

HBM 容量大，但访问代价高；SRAM 容量小，但离计算更近、访问更快。

FlashAttention 的设计就是尽量让中间数据停留在 SRAM 里，把 HBM 主要留给必要的输入输出。这样一来，虽然计算量没有本质减少，但大规模中间矩阵反复读写 HBM 的情况被明显压缩了。

这就是为什么 FlashAttention 常常被描述为“IO-aware”的实现：它不是单纯追求更多算术，而是通过更好的存储层级利用，减少最贵的数据搬运。

理解这一点之后，后面看 Triton 或更底层实现时，就能明白为什么 tile size、block 组织和 shared memory 这么重要。
</details>
### Q3小验证：存储层级的直觉

把“HBM 负责大容量，SRAM 负责局部复用”这条链记住，再看优化思路会更顺。

```python
def score_materialization_ratio(seq_len, tile_size, dtype_bytes=2):
    """比较完整 score 矩阵与单个 score tile 的理论存储规模。

    这不是实际 HBM 流量、kernel 加速比或端到端性能指标。
    """
    full = attention_score_bytes(seq_len, dtype_bytes)
    tile = score_tile_bytes(tile_size, dtype_bytes)
    return full / tile

for tile in [64, 128, 256]:
    print(f'tile={tile:3d} -> score materialization ratio ≈ {score_materialization_ratio(4096, tile):.0f}x')
print('smaller tile => smaller score tile, but more tiles and more scheduling work')

```

## ⚠️ 常见误区

- FlashAttention 不是把计算量消掉了，而是减少了中间结果落到 HBM 的次数。
- `tiling` 不是为了形式更复杂，而是为了让中间状态能留在更合适的存储层。
- `online softmax` 不是额外技巧，而是把归约过程前移到块内处理。
- 如果只盯着 FLOPs，不看数据搬运，通常会误判 FlashAttention 的收益。