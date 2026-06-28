# 28. Pipeline Parallelism MicroBatch | 分布式并行：Pipeline Parallelism 微批次模拟

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/28_Pipeline_Parallelism_MicroBatch.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 前置

**导语：** 先看 ZeRO，再看 Pipeline 的微批次调度会更容易理解流水线并行。
- [Part 2: 25 ZeRO Optimizer Sim](./27_ZeRO_Optimizer_Sim.md)
- [Part 2: 24 Quantization W8A16](./25_Quantization_W8A16.md)

## 相关阅读

**导语：** Pipeline 之后，建议继续看 Tensor Parallelism 和项目实战。
- [Part 2: 27 Tensor Parallelism Sim](./29_Tensor_Parallelism_Sim.md)
- [Part 2: 28 LoRA Fine-Tuning Project](./30_LoRA_Fine_Tuning_Project.md)


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
import torch
```


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

## 参考代码与解析

### 代码

```python
def compute_bubble_ratio(p, m):
    # 实际工业界使用的精确气泡占比公式
    return (p - 1) / (m + p - 1)


```

### 解析

流水线并行（Pipeline Parallelism）是千亿级大模型跨物理机训练的必备组件。通过切分模型深度，配合 Micro-batch 流水作业以及 1F1B（One-Forward-One-Backward）调度算法，可以极大地掩盖设备之间的相互等待时间。气泡比率的公式揭示了一个核心工程实践：微批次的数量必须远大于流水线深度。
