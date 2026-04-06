# 25 Pipeline Parallelism MicroBatch

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/25_Pipeline_Parallelism_MicroBatch.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 25. 分布式进阶：流水线并行与微批次气泡 (Pipeline Parallelism)

**难度：** Hard | **标签：** `Distributed Training`, `Pipeline Parallelism` | **目标人群：** 核心 Infra 开发

在前面我们学习了 ZeRO (切分优化器/梯度/状态) 和 Tensor Parallelism (把矩阵乘法切成小块)。
但当模型达到几千亿参数时（例如 Llama 3 400B），单台拥有 8 张 GPU 的服务器连存放切分后的切片都做不到！必须使用跨服务器的大规模集群。跨服务器的网络带宽非常慢，此时只能引入 **3D 混合并行的最后一块拼图：流水线并行 (Pipeline Parallelism, PP)**。

> **流水线并行的本质**：
> 把一个拥有 80 层的 Transformer 模型按深度切开。GPU 1 负责第 1~10 层，GPU 2 负责 11~20 层，依次类推。
>
> **气泡痛点 (Pipeline Bubble)**：
> 前向传播时，必须等 GPU 1 算完，GPU 2 才能开工。反向传播时，必须等 GPU 8 算完，GPU 7 才能接收到梯度。这种严重的等待空窗期被称为 **气泡 (Bubble)**。

本节我们将推导计算气泡比例的工业级公式，并了解经典的 1F1B 调度。


### 理论与公式计算

**核心手段：微批次 (Micro-batch)**
为了填补气泡，我们不能等整个巨大的 Batch Size 都走完再交给下一个 GPU。我们把一个大 Batch 切分成 $m$ 个 Micro-batch。GPU 1 算完 Micro-batch 1 后立马丢给 GPU 2，自己接着算 Micro-batch 2。

> **大厂高频面试题：推导 1F1B 调度的气泡占比 (Bubble Ratio)**
> 假设：
> - 模型切分在 $p$ 张 GPU (即 Pipeline Stage) 上。
> - 一个全局 Batch 被切分成了 $m$ 个 Micro-batch。
> - 忽略通信延迟，假设前向和反向计算时间恒定。
> 
> **理想满载时间 (Ideal Compute)**：$m 	imes p$ 个单元的时间。
> **实际耗时 (Actual Time)**：$(p - 1) + m$ 个单位时间的流水线排空。
> 
> **Bubble Ratio 公式**：
> $\text{Bubble} = \frac{p - 1}{m + p - 1} \approx \frac{p - 1}{m}$ (当 m 足够大时)


```python
def compute_bubble_ratio(p, m):
    """
    计算流水线并行的气泡率。
    
    Args:
        p: Pipeline Stage 数量 (GPU 数量)
        m: Micro-batch 的数量
        
    Returns:
        float: 气泡占比 [0, 1]
    """
    # ==========================================
    # TODO: 使用近似公式或精确公式计算 Bubble Ratio
    # ==========================================
    pass


```

```python
def test_pipeline_bubble():
    try:
        # 测试 8 个 Stage，切分为 32 个 micro-batch
        ratio = compute_bubble_ratio(p=8, m=32)
        # 精确公式结果为 7 / (32 + 7) = 7/39 = 0.179...
        # 近似公式为 7 / 32 = 0.218...
        # 为了兼容两种答案，我们只检查上限
        assert ratio is not None, "未实现计算！"
        assert 0.15 < ratio < 0.25, f"计算错误，结果应该在 0.17~0.22 左右，实际为 {ratio}"
        
        print("✅ 测试通过！在大规模集群训练时，为了降低流水线气泡，Micro-batch 的数量 m 必须远远大于 Stage 的数量 p。")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

test_pipeline_bubble()


```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---


::: details 💡 点击查看官方解析与参考代码

流水线并行（Pipeline Parallelism）是千亿级大模型跨物理机训练的必备组件。通过切分模型深度，配合 Micro-batch 流水作业以及 1F1B（One-Forward-One-Backward）调度算法，可以极大地掩盖设备之间的相互等待时间。气泡比率的公式揭示了一个核心工程实践：微批次的数量必须远大于流水线深度。


```python
def compute_bubble_ratio(p, m):
    # 实际工业界使用的精确气泡占比公式
    return (p - 1) / (m + p - 1)


```

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
