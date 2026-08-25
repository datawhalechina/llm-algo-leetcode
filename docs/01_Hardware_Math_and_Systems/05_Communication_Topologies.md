# 05. Communication Topologies | 通信拓扑与分布式基石

**难度：** Medium | **环境：** CPU-first | **标签：** `并行通信`, `分布式训练`, `通信拓扑` | **目标人群：** 通信机制入门者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/05_Communication_Topologies.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

大模型训练里，算力并不是唯一瓶颈。很多任务从单卡扩到多卡后，吞吐并不会线性提升，问题往往不在矩阵乘法，而在通信路径本身：数据并行怎么同步，张量并行怎么切分，流水线并行为什么会产生等待，不同拓扑下这些通信会不会直接压过计算收益。

这是一节**并行前置节**：它从 DP、TP、PP 和通信拓扑出发，解释多卡训练为什么会受到带宽和同步影响，主要服务 `监督微调路线` 与 `通信与并行专题`。

**关键词：** `DP`, `TP`, `PP`

---

## 前置阅读
**导语：** 先复习 GPU 拓扑和显存切分，再用本页的示例比较并行方式、链路带宽与通信代价。
- [03. GPU Architecture and Memory | GPU 物理架构与内存层级](./03_GPU_Architecture_and_Memory.md)
- [06. VRAM Calculation and ZeRO | 显存计算与 ZeRO 优化](./06_VRAM_Calculation_and_ZeRO.md)

## 相关阅读
**导语：** 后续页面把本页的拓扑判断接到 NCCL、并行策略和通信调度。
- [20. NCCL and AllReduce Basics | NCCL 与 AllReduce 基础](./20_NCCL_and_AllReduce_Basics.md)
- [26. Parallel Strategy Decision Framework | 并行策略决策框架](./26_Parallel_Strategy_Decision_Framework.md)
- [27. Communication Scheduling Optimization | 通信调度优化](./27_Communication_Scheduling_Optimization.md)
---
## Q1：什么是大模型训练中的 3D 并行 (3D Parallelism)？

<details><summary>点击展开查看解析</summary>

3D 并行通常指把数据并行 (DP)、张量并行 (TP) 和流水线并行 (PP) 组合起来使用。

- **DP**：不同卡处理不同数据批次，再同步梯度。
- **TP**：把单层中的大张量切到多卡上共同计算。
- **PP**：把不同层切到不同设备或设备组上形成流水线。

这三者的目标不是“越多越好”，而是让模型、算力和通信拓扑能一起匹配。
</details>
### Q1小验证：三种并行分别切什么

先把“切数据 / 切张量 / 切层”记住。

```python
def three_d_parallel(dp, tp, pp):
    # 3D 并行的核心不是三个名词，而是三种切分方式是否能同时成立。
    return {
        'dp_groups': dp,
        'tp_shards': tp,
        'pp_stages': pp,
        'effective_workers': dp * tp * pp,
    }

cases = [
    three_d_parallel(8, 1, 1),
    three_d_parallel(4, 2, 2),
    three_d_parallel(2, 4, 4),
]
for case in cases:
    print(case)
print('3D parallelism = DP × TP × PP')

```

## Q2：用一组示例带宽比较机内与机外通信

<details><summary>点击展开查看解析</summary>

机内通常可以通过 NVLink / NVSwitch 获得更高带宽，而机外则常常受限于 PCIe 或网络链路。下面的数值只是教学用的链路带宽假设，不代表所有 A100/H100、主板、驱动或网络配置；真实实验应以 `nvidia-smi topo -m`、NCCL 测试或厂商规格为准。

这意味着：
- 机内通信更适合高频同步；
- 机外通信更容易成为瓶颈；
- 只看 GPU 数量不看拓扑，很容易高估扩展收益。

所以通信拓扑不是背景信息，而是并行策略能否成立的前提。
</details>
### Q2小验证：带宽差距会带来什么

带宽差距越大，越需要谨慎决定通信放在哪里。

```python
def bandwidth_ratio(intra=900, inter=64):
    """Return an illustrative link-bandwidth ratio in the same unit (Gb/s)."""
    if intra <= 0 or inter <= 0:
        raise ValueError('intra and inter bandwidths must be positive')
    return intra / inter

print(f'ratio ≈ {bandwidth_ratio():.1f}x')
```

## Q3：带宽悬崖如何决定 TP 与 PP 的部署边界？

<details><summary>点击展开查看解析</summary>

当硬件带宽出现明显断层时，跨断层的通信成本会骤增。

如果张量并行需要频繁跨很慢的链路同步，那它的扩展效果就会受限；如果流水线并行能够把通信切在更合适的边界上，它就可能更合适。

所以部署边界不是拍脑袋定的，而是由带宽悬崖、同步频率和模型切分方式共同决定。
</details>
### Q3小验证：带宽悬崖会放大通信代价

```python
def comm_time_ms(size_mb, bandwidth_gbps):
    """Estimate ideal one-way payload transfer time in milliseconds.

    This is a bandwidth-only lower-bound proxy: it excludes latency,
    protocol overhead, collective algorithms, contention, directionality
    and topology. ``size_mb`` is decimal MB and bandwidth is Gb/s.
    """
    if size_mb < 0:
        raise ValueError('size_mb must be non-negative')
    if bandwidth_gbps <= 0:
        raise ValueError('bandwidth_gbps must be positive')
    # MB -> Mb, divide by Gb/s, then convert seconds to milliseconds.
    return size_mb * 8 / bandwidth_gbps

payload_mb = 256
nvlink_gbps = 900
pcie_gbps = 64
nvlink_time = comm_time_ms(payload_mb, nvlink_gbps)
pcie_time = comm_time_ms(payload_mb, pcie_gbps)
ratio = pcie_time / nvlink_time

print(f'{payload_mb} MB over NVLink: {nvlink_time:.2f} ms')
print(f'{payload_mb} MB over PCIe: {pcie_time:.2f} ms')
print(f'PCIe / NVLink time ratio: {ratio:.1f}x')

```

## Q4：All-Reduce、All-Gather、Reduce-Scatter 分别有什么区别？

<details><summary>点击展开查看解析</summary>

这三种集合通信原语分别对应不同的数据流：

- **All-Reduce**：所有设备做归约，并把结果广播回每一张卡。
- **All-Gather**：把各卡局部数据收集到一起，形成完整结果。
- **Reduce-Scatter**：先归约，再把结果切分发回各卡。

它们在数据并行、张量并行和流水线并行中会以不同方式出现，影响同步成本和通信模式；实际耗时还取决于消息大小、world size、拓扑、collective 算法与通信库实现。下面的代码只做概念映射，不是在模拟 NCCL。
</details>
### Q4小验证：通信原语各自做什么

先把“聚合”“收集”“切分再发回”记清楚。

```python
def collective(kind):
    table = {
        'allreduce': 'reduce + broadcast',
        'allgather': 'gather all pieces',
        'reducescatter': 'reduce then scatter',
    }
    return table.get(kind, 'unknown')

for k in ['allreduce', 'allgather', 'reducescatter']:
    print(k, '->', collective(k))
```

## ⚠️ 常见误区

- 3D 并行不是把三种并行简单相加。
- 机内和机外通信不是同一个成本级别。
- 带宽悬崖会直接改变部署策略。
- 选并行方案时，通信原语和拓扑要一起看。