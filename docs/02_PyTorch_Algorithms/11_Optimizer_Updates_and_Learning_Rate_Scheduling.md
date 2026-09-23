# 11. Optimizer Updates and Learning Rate Scheduling | 优化器更新与学习率调度

**难度：** Medium | **环境：** CPU-first | **标签：** `训练微调`, `学习率调度`, `WSD/Cosine` | **目标人群：** 训练机制学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/11_Optimizer_Updates_and_Learning_Rate_Scheduling.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

训练不是只选一个固定学习率然后一路跑到底。刚开始参数和优化器状态都不稳定，学习率太大容易把 loss 冲飞；中期如果过早衰减，模型又会失去继续吸收数据的能力；到了末期，还需要把更新幅度收下来帮助收敛。

WSD 把这个训练节奏拆成三段：先 warmup 把学习率抬起来，中间 stable 保持学习能力，最后 decay 做收敛退火。这一节在训练微调路线里和 `12` 一起组成 `13` 的直接前置：`11` 负责更新幅度怎么变，`12` 负责一次有效更新怎样凑出来。学完这里，后面再看 `13` 和 `60` 时，你会更容易把学习率曲线和训练稳定性、收敛速度、项目报告里的调参结论连起来；如果这里没学明白，后面很容易只把 scheduler 当现成配置项，而说不清训练为什么在 warmup、stable 和 decay 三段表现不同。

**关键词：** `warmup`, `stable`, `decay`

---
## 前置阅读

**导语：** 先补齐优化器、最小训练接口和训练循环，再看学习率如何按阶段控制更新幅度。
- [P0: 11. PyTorch Optimizers and Loss | 优化器与损失](../00_Prerequisites/11_PyTorch_Optimizers_and_Loss.md)
- [P0: 12. PyTorch Minimal Training Interface | 最小训练接口](../00_Prerequisites/12_PyTorch_Minimal_Training_Interface.md)
- [P0: 13. Simple Neural Network Training | 简单神经网络训练](../00_Prerequisites/13_Simple_Neural_Network_Training.md)

---
### Step 1：训练更新节奏与学习率调度
WSD 调度器把训练过程拆成 Warmup、Stable 和 Decay 三段，每一段对应一种不同的训练需求。

> **为什么一定要有 Warmup (预热)？**
> 1. **训练初期**参数更新方向和幅度尚未稳定。如果直接使用最大学习率（如 3e-4），更新过猛可能导致 loss 震荡甚至发散。
> 2. **AdamW 优化器**在刚开始时，其用于分母的“二阶动量 (方差的移动平均)”还没收集够数据，非常小。除以一个极小的数会导致实际更新步长不可控。Warmup 给了优化器一段逐步稳定的时间。

> **Cosine Decay 的痛点与 WSD 的选择：**
> - **传统的 Cosine Decay** 需要在训练**一开始就定死总步数 (Total Steps)**，慢慢按照余弦曲线下降到 0。这导致一个致命问题：如果你发现数据还没训完，想加数据继续训 (Continued Pre-training)，此时学习率已经降到底了，模型失去了学习新知识的能力。
> - **WSD (Warmup-Stable-Decay) 调度器** 准确解决了这个问题。它把训练分为三段：
>   1. **Warmup (预热)**：线性增长到最大学习率。
>   2. **Stable (稳定期)**：保持最大学习率不变，吃尽海量数据。如果想加数据，无限延长这个阶段即可。
>   3. **Decay (高效退火)**：只在训练的最后阶段，用线性或余弦曲线逐步降低更新幅度，帮助模型收敛。

![学习率调度总览](../public/02_PyTorch_Algorithms/11_scheduler_overview.svg)

### Step 2：WSD 三阶段与学习率曲线
先看清三段学习率曲线分别解决什么训练问题，再理解它们如何组成完整的训练节奏。
Warmup-Stable-Decay (WSD) 是现代大模型训练里常见的三段式节奏：
1. **Warmup**: 学习率从接近 0 线性增长到基础学习率 $\eta_{max}$，避免训练初期更新过猛。
2. **Stable**: 保持 $\eta_{max}$ 训练主要数据，让模型在稳定更新幅度下继续吸收样本。
3. **Cosine Decay**: 在最后阶段使用余弦退火，将学习率平滑降至 $\eta_{min}$，帮助收敛。

本节采用 **WSD + cosine decay**：中间保持基础学习率，末段使用余弦曲线逐步退火。

| 阶段 | 学习率行为 | 训练作用 |
|:---|:---|:---|
| Warmup | 从低值逐步升高 | 降低训练初期的更新风险 |
| Stable | 保持基础学习率 | 承担主要训练过程 |
| Decay | 逐步降低 | 帮助训练后期收敛 |

![WSD 阶段判断](../public/02_PyTorch_Algorithms/11_scheduler_phase_logic.svg)

### Step 3：有效更新与 scheduler 计数
学习率调度要跟着有效的参数更新推进，而不是跟着每个 micro-batch 机械推进。梯度累积时，多个 micro-batch 先形成一个 effective batch，再完成一次参数更新。

因此，梯度累积只改变 optimizer update 的发生频率，不改变 scheduler 的计数语义；warmup、stable 和 decay 的长度都以 optimizer update 为单位，而不是把每个 micro-batch 都算作一次 scheduler update。

| 阶段 | 计数单位 | 主要动作 |
|:---|:---|:---|
| micro-batch | 每个小批次 | forward、loss、backward，累积梯度 |
| effective batch | 多个 micro-batch 汇总 | 判断是否完成一次有效更新 |
| optimizer update | 一次参数更新 | 执行 `optimizer.step()` |
| scheduler update | 一次学习率推进 | 执行 `scheduler.step()`，进入下一个 WSD step |


### Step 4：实现并验证 WSD Scheduler
接下来把三段曲线写成可运行的调度器。题目中的阶段长度按一次有效的 `optimizer.step()` 计数；如果训练使用梯度累积，多个 micro-batch 先形成一个 effective batch，随后才执行一次参数更新和一次 `scheduler.step()`。

**要求**：请补全下方 `WSD_Scheduler` 类。我们需要继承 PyTorch 原生的 `torch.optim.lr_scheduler.LRScheduler`，实现它的 `get_lr()` 方法。


```python
import torch
import math
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LRScheduler
```


```python
# 题目设计：把 WSD 的三阶段规则实现为 PyTorch Scheduler。
# 每个 TODO 只负责一个阶段的学习率计算，scheduler 的步数按 optimizer update 推进。
class WSD_Scheduler(LRScheduler):
    """按 optimizer update 执行 Warmup-Stable-Decay 学习率调度。

    Args:
        optimizer: 承载参数和基础学习率的优化器。
        num_warmup_steps: 线性升温的有效更新次数。
        num_stable_steps: 保持基础学习率的有效更新次数。
        num_decay_steps: 余弦退火的有效更新次数。
        min_lr_ratio: 退火结束时相对基础学习率的比例。

    Note:
        `step` 表示有效 optimizer update 次数，不是 micro-batch 次数；
        三个阶段的总长度由 warmup、stable 和 decay 之和确定。
    """
    def __init__(self, optimizer, num_warmup_steps, num_stable_steps, num_decay_steps, min_lr_ratio=0.1, last_epoch=-1):
        self.num_warmup_steps = num_warmup_steps
        self.num_stable_steps = num_stable_steps
        self.num_decay_steps = num_decay_steps
        self.min_lr_ratio = min_lr_ratio
        self.total_steps = num_warmup_steps + num_stable_steps + num_decay_steps
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        step = self._step_count - 1
        
        lrs = []
        for base_lr in self.base_lrs:
            min_lr = base_lr * self.min_lr_ratio
            
            # TODO 1：Warmup 阶段；计算从 0 线性增长到 base_lr 的 current_lr。
            if step < self.num_warmup_steps:
                # current_lr = ???
                pass
            
            # TODO 2：Stable 阶段；将 current_lr 设为 base_lr。
            elif step < (self.num_warmup_steps + self.num_stable_steps):
                # current_lr = ???
                pass
                
            # TODO 3：Cosine Decay 阶段；依次计算 decay_step、decay_ratio、cosine_decay 和 current_lr。
            else:
                # decay_step = ???
                # decay_ratio = ???
                # cosine_decay = ???
                # current_lr = ???
                pass
                
            lrs.append(current_lr)
            
        return lrs
```


```python
# 测试设计：验证 WSD 的 warmup、stable、cosine decay 三阶段和 update 计数语义。
# 学习率序列是机制证据；绘图只是辅助阅读，不参与测试断言。
def _collect_wsd_lrs(warmup, stable, decay, max_lr=3e-4, min_lr_ratio=0.1):
    """按 optimizer update 推进 scheduler，并收集学习率序列。"""
    dummy_model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(dummy_model.parameters(), lr=max_lr)
    scheduler = WSD_Scheduler(
        optimizer,
        num_warmup_steps=warmup,
        num_stable_steps=stable,
        num_decay_steps=decay,
        min_lr_ratio=min_lr_ratio,
    )
    lrs = []
    for _ in range(warmup + stable + decay):
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()
    return lrs

def _assert_wsd_boundaries(lrs, warmup, stable, max_lr, min_lr_ratio):
    """验证三个阶段的起点、平台、衔接点和终点。"""
    assert lrs[0] == 0.0, "warmup 起点应该从 0 开始"
    assert abs(lrs[warmup] - max_lr) < 1e-8, "warmup 结束时应该达到最大学习率"
    assert abs(lrs[warmup + stable - 1] - max_lr) < 1e-8, "stable 阶段应该维持最大学习率"
    assert abs(lrs[warmup + stable] - max_lr) < 1e-8, "decay 起点应该从最大学习率开始"
    assert abs(lrs[-1] - (max_lr * min_lr_ratio)) < 1e-8, "decay 结束时应该达到最小学习率"

def _plot_wsd_curve(lrs, warmup, stable):
    """可选地绘制 WSD 曲线，不承担机制断言。"""
    plt.figure(figsize=(10, 5))
    plt.plot(lrs, label="Learning Rate", color='blue', linewidth=2)
    plt.axvline(x=warmup, color='r', linestyle='--', alpha=0.5, label='End Warmup')
    plt.axvline(x=warmup + stable, color='g', linestyle='--', alpha=0.5, label='Start Decay')
    plt.title("WSD (Warmup-Stable-Decay) Scheduler")
    plt.xlabel("Optimizer Updates")
    plt.ylabel("Learning Rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.close()

def test_wsd_scheduler():
    """运行 CPU-first 的 WSD scheduler 机制测试。"""
    try:
        warmup, stable, decay = 1000, 7000, 2000
        max_lr, min_lr_ratio = 3e-4, 0.1
        lrs = _collect_wsd_lrs(warmup, stable, decay, max_lr, min_lr_ratio)
        _assert_wsd_boundaries(lrs, warmup, stable, max_lr, min_lr_ratio)
        _plot_wsd_curve(lrs, warmup, stable)
        print("✅ WSD 三阶段学习率曲线和 update 边界断言通过。")

    except NotImplementedError:
        print("请先完成 TODO 代码！")
        raise
    except (AttributeError, NameError, TypeError, ValueError) as e:
        print("代码可能未完成，导致变量未定义" if isinstance(e, NameError) else "代码可能未完成，导致了类型错误")
        raise NotImplementedError("请先完成 TODO 代码！") from e
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
        raise NotImplementedError("请先完成 TODO 代码！") from e
    except Exception as e:
        print(f"❌ 发生异常: {e}")
        raise

test_wsd_scheduler()
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
# 参考实现与题目区保持同一组函数、参数和阶段分支，只补全 TODO。
class WSD_Scheduler(LRScheduler):
    def __init__(self, optimizer, num_warmup_steps, num_stable_steps, num_decay_steps, min_lr_ratio=0.1, last_epoch=-1):
        self.num_warmup_steps = num_warmup_steps
        self.num_stable_steps = num_stable_steps
        self.num_decay_steps = num_decay_steps
        self.min_lr_ratio = min_lr_ratio
        self.total_steps = num_warmup_steps + num_stable_steps + num_decay_steps
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        step = self._step_count - 1
        
        lrs = []
        for base_lr in self.base_lrs:
            min_lr = base_lr * self.min_lr_ratio
            
            # 预热期先把学习率线性抬升，避免一开始更新过猛。
            # TODO 1：Warmup 阶段；current_lr 从 0 线性增长到 base_lr。
            if step < self.num_warmup_steps:
                if step == 0:
                    current_lr = 0.0
                else:
                    current_lr = base_lr * step / self.num_warmup_steps
            
            # 稳定期直接保持最大学习率，让训练过程持续吸收数据。
            # TODO 2：Stable 阶段；current_lr 保持为 base_lr。
            elif step < (self.num_warmup_steps + self.num_stable_steps):
                current_lr = base_lr
                
            # 退火期按进度把学习率从 base_lr 拉到 min_lr。
            # TODO 3：Cosine Decay 阶段；依次计算 decay_step、decay_ratio、cosine_decay 和 current_lr。
            else:
                decay_step = step - self.num_warmup_steps - self.num_stable_steps
                decay_ratio = decay_step / self.num_decay_steps
                cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_ratio))
                current_lr = min_lr + (base_lr - min_lr) * cosine_decay
                
            lrs.append(current_lr)
            
        return lrs

```

### 解析

- **这一题要解决什么**：把 WSD 的三段式学习率曲线落成一个可运行的 `LRScheduler`。
- **为什么这样做**：Warmup 防止前期冲飞，Stable 提供主要学习窗口，Cosine Decay 帮助后期平滑收敛。
- **带走的直觉**：WSD 的核心不是一条从头降到尾的曲线，而是把训练过程拆成“起步、主训、退火”三个可控阶段。

**1. TODO 1: Warmup 阶段（线性增长）**

- **实现方式**：当 `step < num_warmup_steps` 时，使用 `current_lr = base_lr * step / num_warmup_steps`。
- **核心思想**：学习率从接近 0 线性增长到 `base_lr`。
- **必要性**：训练初期，参数和优化器状态都不稳定。如果直接使用大学习率，容易导致 loss 剧烈震荡甚至发散。
- **边界处理**：当 `num_warmup_steps=0` 时，代码直接跳过 warmup，避免除以 0。

**2. TODO 2: Stable 阶段（保持恒定）**

- **实现方式**：当 `num_warmup_steps <= step < num_warmup_steps + num_stable_steps` 时，返回 `base_lr`。
- **核心思想**：学习率保持在最大值，让模型在主要训练窗口里持续吸收数据。
- **与普通 Cosine 对比**：普通 Cosine 往往 warmup 后立即衰减；WSD 的 Stable 阶段让中期训练不那么早进入小步更新。
- **接到微调实验**：如果总更新步数很少，Stable 阶段不要过长，否则可能没有足够 decay 步数做收尾。

**3. TODO 3: Cosine Decay 阶段（余弦退火）**

- **实现方式**：
  ```python
  decay_step = step - self.num_warmup_steps - self.num_stable_steps
  decay_ratio = decay_step / self.num_decay_steps
  cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_ratio))
  current_lr = min_lr + (base_lr - min_lr) * cosine_decay
  ```
- **核心思想**：学习率从 `base_lr` 平滑退火到 `base_lr * min_lr_ratio`。
- **收敛作用**：较小学习率帮助模型在训练后期做细粒度调整，减少震荡。
- **超过计划步数**：代码在超过 `total_steps` 后返回 `base_lr * min_lr_ratio`，避免学习率继续变化到异常值。

**工程要点**

- **按 update 计数**：配合梯度累积时，scheduler 通常按 `optimizer.step()` 次数推进，不按 micro-batch 次数推进。
- **调用顺序**：常见训练循环是 `loss.backward()` -> `optimizer.step()` -> `scheduler.step()` -> `optimizer.zero_grad()`。
- **微调设置**：SFT / LoRA 微调步数通常比预训练少，warmup 和 decay 比例要更保守，避免一开始冲飞或最后完全学不动。
- **验证方式**：除了画曲线，还要检查阶段边界：warmup 结束是否到达 `base_lr`，stable 是否保持不变，decay 末尾是否接近 `min_lr`。

## 相关阅读

如果想把 WSD 放回真实训练流程，可以继续阅读梯度累积、端到端微调和 LoRA 项目；下面的论文与开源实现用于了解调度器在训练框架中的实际接口。

- [12. Gradient Accumulation | 梯度累积](../02_PyTorch_Algorithms/12_Gradient_Accumulation.md)
- [13. End-to-End Fine-Tuning Experiment | 端到端微调实验](../02_PyTorch_Algorithms/13_End_to_End_Fine_Tuning_Experiment.md)
- [60. LoRA Fine-Tuning Project | LoRA 微调项目](../02_PyTorch_Algorithms/60_LoRA_Fine_Tuning_Project.md)
- [Hugging Face Transformers Scheduler API](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules)
- [PyTorch LRScheduler 文档](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
- [WSD: Warmup-Stable-Decay 论文检索入口](https://arxiv.org/search/?query=Warmup-Stable-Decay&searchtype=all)
