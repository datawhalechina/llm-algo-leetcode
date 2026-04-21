# 11. LR Schedulers WSD Cosine | 大模型训练调参难点：学习率调度器 (Warmup, Cosine, WSD)

**难度：** Medium | **标签：** `Training`, `LR Scheduler`, `Llama-3` | **目标人群：** 模型微调与工程部署

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/11_LR_Schedulers_WSD_Cosine.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


在传统的深度学习中，学习率通常是固定的或是按阶梯下降 (Step Decay)。但在大语言模型 (LLM) 的预训练和微调中，学习率的调度 (LR Schedule) 直接决定了模型会不会**崩溃 (Loss Spike)** 以及能否**收敛到平缓的最优解**。
目前大模型训练有两套绝对主流的方案：
1. **Cosine Annealing with Warmup** (带预热的余弦退火)：LLaMA-1/2, GPT-3 等绝大多数模型预训练的标配。
2. **WSD (Warmup-Stable-Decay)**：LLaMA-3 和现代持续预训练 (Continued Pre-training) 最前沿的标配。

本节我们将不依赖 HuggingFace 或 PyTorch 的现成实现，**纯手工通过数学分段函数写一个现代大模型标配的 WSD 调度器**，并将其可视化。

### Step 1: 核心机制剖析

> **为什么一定要有 Warmup (预热)？**
> 1. **模型随机初始化**时，梯度非常大规模且方向混乱。如果直接给最大的学习率（如 3e-4），大规模的梯度更新会瞬间把模型权重冲飞 (Loss 直接 NaN)。
> 2. **AdamW 优化器**在刚开始时，其用于分母的“二阶动量 (方差的移动平均)”还没收集够数据，非常小。除以一个极小的数会导致实际更新步长不可控。Warmup 给了优化器几千步的“收集方差”的时间。

> **Cosine Decay 的痛点与 WSD 的崛起 (LLaMA-3 的选择)：**
> - **传统的 Cosine Decay** 需要在训练**一开始就定死总步数 (Total Steps)**，慢慢按照余弦曲线下降到 0。这导致一个致命问题：如果你发现数据还没训完，想加数据继续训 (Continued Pre-training)，此时学习率已经降到底了，模型失去了学习新知识的能力。
> - **WSD (Warmup-Stable-Decay) 调度器** 准确解决了这个问题。它把训练分为三段：
>   1. **Warmup (预热)**：线性增长到最大学习率。
>   2. **Stable (稳定期)**：保持最大学习率不变，吃尽海量数据。如果想加数据，无限延长这个阶段即可。
>   3. **Decay (高效退火)**：只在训练的最后 10% 或 5% 阶段，用一个陡峭的函数（如线性或余弦）快速降到 0，让模型迅速收敛收拢。

### Step 2: WSD 调度器的数学曲线
Warmup-Stable-Decay (WSD) 是现代预训练（如 LLaMA-3）的标配。它的三个阶段是：
1. **Warmup**: 学习率从 0 线性增长到最大值 $\eta_{max}$。
2. **Stable**: 保持最大值 $\eta_{max}$ 训练绝大部分 Token（占整体的 70%~90%）。
3. **Decay**: 在最后阶段（如退火阶段）使用余弦退火或线性衰减，将学习率迅速降至 $\eta_{min}$。这极大地帮助了模型在最后阶段收敛。

### Step 3: 代码实现框架
继承自 `torch.optim.lr_scheduler.LRScheduler`，你需要实现核心的 `get_lr()` 方法。在其中利用 `self.last_epoch` 判断当前步数处于哪一个阶段，然后根据上述的数学公式计算并返回此时的学习率数组。

###  Step 4: 动手实战

**要求**：请补全下方 `WSD_Scheduler` 类。我们需要继承 PyTorch 原生的 `torch.optim.lr_scheduler.LRScheduler`，实现它的 `get_lr()` 方法。


```python
import torch
import math
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LRScheduler
```


```python
class WSD_Scheduler(LRScheduler):
    """
    手动实现 LLaMA-3 风格的 Warmup-Stable-Decay (WSD) 学习率调度器。
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
            
            # ==========================================
            # TODO 1: Warmup 阶段
            # 规则: 当 step < num_warmup_steps 时，学习率从 0 线性增长到 base_lr
            # ==========================================
            if step < self.num_warmup_steps:
                # current_lr = ???
                current_lr = base_lr * 0.5  # 占位初始化
            
            # ==========================================
            # TODO 2: Stable 阶段
            # 规则: 学习率保持在 base_lr
            # ==========================================
            elif step < (self.num_warmup_steps + self.num_stable_steps):
                # current_lr = ???
                current_lr = base_lr * 0.5  # 占位初始化
                
            # ==========================================
            # TODO 3: Cosine Decay 阶段
            # 规则: 学习率从 base_lr 余弦衰减到 min_lr
            # 提示: 计算 decay 阶段的进度比例，使用余弦函数
            # ==========================================
            else:
                # current_lr = ???
                current_lr = base_lr * 0.5  # 占位初始化
                
            lrs.append(current_lr)
            
        return lrs
```


```python
# 测试并可视化你的实现
def test_and_plot_wsd():
    try:
        # 1. 初始化一个假的优化器 (用来承载学习率)
        dummy_model = torch.nn.Linear(2, 2)
        max_lr = 3e-4
        optimizer = torch.optim.AdamW(dummy_model.parameters(), lr=max_lr)
        
        # 2. 设定 WSD 的三个阶段步数
        warmup = 1000   # 10%
        stable = 7000   # 70%
        decay = 2000    # 20%
        total = warmup + stable + decay
        
        # 3. 初始化我们实现的 Scheduler
        scheduler = WSD_Scheduler(
            optimizer, 
            num_warmup_steps=warmup, 
            num_stable_steps=stable, 
            num_decay_steps=decay, 
            min_lr_ratio=0.1
        )
        
        # 4. 模拟训练过程，收集学习率
        lrs = []
        for _ in range(total):
            lrs.append(optimizer.param_groups[0]['lr'])
            optimizer.step()
            scheduler.step()
            
        # 5. 断言关键点的正确性
        assert lrs[0] == 0.0, "第一步应该是 0 (或者极小值)"
        assert abs(lrs[warmup] - max_lr) < 1e-8, "Warmup 结束时应该是最大学习率"
        assert abs(lrs[warmup + stable - 1] - max_lr) < 1e-8, "Stable 阶段应该维持最大学习率"
        assert abs(lrs[-1] - (max_lr * 0.1)) < 1e-8, "Decay 结束时应该是最小学习率 (max_lr * 0.1)"
        
        print("✅ 数学逻辑断言通过！")
        
        # 6. 画出学习率曲线
        plt.figure(figsize=(10, 5))
        plt.plot(lrs, label="Learning Rate", color='blue', linewidth=2)
        plt.axvline(x=warmup, color='r', linestyle='--', alpha=0.5, label='End Warmup')
        plt.axvline(x=warmup+stable, color='g', linestyle='--', alpha=0.5, label='Start Decay')
        plt.title("LLaMA-3 Style WSD (Warmup-Stable-Decay) Scheduler")
        plt.xlabel("Training Steps")
        plt.ylabel("Learning Rate")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
        
        print(" 你成功实现并可视化了目前最先进的大模型学习率调度器。现在你不怕被面试官问到 LLaMA-3 的退火策略了！")
        
    except NotImplementedError:
        print("请先完成 TODO 代码！")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise e  # 将错误抛给测试脚本

test_and_plot_wsd()
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
            
            # TODO 1: Warmup 阶段 - 线性增长（从0开始）
            if step < self.num_warmup_steps:
                if step == 0:
                    current_lr = 0.0
                else:
                    current_lr = base_lr * step / self.num_warmup_steps
            
            # TODO 2: Stable 阶段 - 保持恒定
            elif step < (self.num_warmup_steps + self.num_stable_steps):
                current_lr = base_lr
                
            # TODO 3: Cosine Decay 阶段
            else:
                decay_step = step - self.num_warmup_steps - self.num_stable_steps
                decay_ratio = decay_step / self.num_decay_steps
                cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_ratio))
                current_lr = min_lr + (base_lr - min_lr) * cosine_decay
                
            lrs.append(current_lr)
            
        return lrs

```

### 解析

**1. TODO 1: Warmup 阶段（线性增长）**

- **实现方式**：`lr = self.max_lr * (self.current_step + 1) / self.warmup_steps`
- **核心思想**：学习率从 0 线性增长到 `max_lr`。
- **必要性**：训练初期，模型参数是随机初始化的，梯度方向不稳定。如果直接使用大学习率，容易导致梯度爆炸或发散。
- **工程细节**：`(self.current_step + 1)` 确保第一步的学习率不为 0，而是 `max_lr / warmup_steps`。
- **典型设置**：Warmup 步数通常占总训练步数的 1-5%，例如训练 100k 步，Warmup 1k-5k 步。

**2. TODO 2: Stable 阶段（保持恒定）**

- **实现方式**：`lr = self.max_lr`
- **核心思想**：学习率保持在最大值，让模型充分学习。
- **训练效果**：这是模型学习的主要阶段，Loss 下降最快。
- **时长设置**：通常占总训练步数的 60-80%，是三个阶段中最长的。
- **与 Cosine 对比**：Cosine 调度器没有 Stable 阶段，学习率在 Warmup 后立即开始衰减。WSD 的 Stable 阶段提供了更稳定的训练过程。

**3. TODO 3: Decay 阶段（线性衰减）**

- **实现方式**：
  ```python
  decay_progress = (self.current_step - self.warmup_steps - self.stable_steps) / self.decay_steps
  lr = self.max_lr - (self.max_lr - self.min_lr) * min(decay_progress, 1.0)
  ```
- **核心思想**：学习率从 `max_lr` 线性衰减到 `min_lr`。
- **收敛作用**：较小的学习率帮助模型在损失函数的局部最优附近进行精细调整，避免震荡。
- **`min(decay_progress, 1.0)` 的作用**：防止训练步数超过预期时学习率变为负数。
- **典型设置**：Decay 步数通常占总训练步数的 10-20%。

**工程要点**

- **WSD vs Cosine**：WSD 更可控，适合需要精确控制训练阶段的场景；Cosine 更平滑，适合不确定最优训练步数的场景。
- **超参数调优**：Stable 阶段的长度是关键，过短会导致训练不充分，过长会浪费计算资源。
- **动态调整**：可以根据验证集 Loss 动态调整各阶段的长度，实现自适应调度。
- **多阶段训练**：可以在 Decay 后再次进入 Warmup-Stable-Decay 循环，实现周期性学习率调度（Cyclic Learning Rate）。
