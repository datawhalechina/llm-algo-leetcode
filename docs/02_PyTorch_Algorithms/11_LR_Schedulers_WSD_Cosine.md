# 11 LR Schedulers WSD Cosine

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/11_LR_Schedulers_WSD_Cosine.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 11. 大模型训练调参难点：学习率调度器 (Warmup, Cosine, WSD)

**难度：** Medium | **标签：** `Training`, `LR Scheduler`, `Llama-3` | **目标人群：** 模型微调与工程部署

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

class WSD_Scheduler(LRScheduler):
    """
    手动实现 LLaMA-3 风格的 Warmup-Stable-Decay (WSD) 学习率调度器。
    
    参数:
    - optimizer: 绑定的 PyTorch 优化器
    - num_warmup_steps: 预热步数 (在这个阶段 LR 从 0 线性增长到 max_lr)
    - num_stable_steps: 稳定期步数 (在这个阶段 LR 保持为 max_lr)
    - num_decay_steps: 退火步数 (在这个阶段 LR 从 max_lr 余弦下降到 min_lr)
    - min_lr_ratio: 最小学习率比例 (通常是 max_lr 的 0.1 或 0.01)
    """
    def __init__(self, optimizer, num_warmup_steps, num_stable_steps, num_decay_steps, min_lr_ratio=0.1, last_epoch=-1):
        self.num_warmup_steps = num_warmup_steps
        self.num_stable_steps = num_stable_steps
        self.num_decay_steps = num_decay_steps
        self.min_lr_ratio = min_lr_ratio
        
        # 总步数 = warmup + stable + decay
        self.total_steps = num_warmup_steps + num_stable_steps + num_decay_steps
        
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        """
        每次调用 scheduler.step() 时，PyTorch 会调用此方法计算当前 step 对应的学习率。
        必须返回一个列表，对应优化器中每一组参数 (param_groups) 的学习率。
        在这个函数中，我们可以通过 self._step_count 获取当前的全局步数 (从 1 开始)。
        通过 base_lr 获取我们在 optimizer 中设置的初始最大学习率。
        """
        step = self._step_count - 1 # 转换为 0-indexed 的 step
        
        lrs = []
        for base_lr in self.base_lrs:
            min_lr = base_lr * self.min_lr_ratio
            
            # ==========================================
            # TODO 1: 阶段 1 - Warmup (预热)
            # 规则: 当 step < num_warmup_steps 时
            # 返回: 0 到 base_lr 的线性插值。注意规避除以 0 的风险，强制最小返回一个极小值。
            # ==========================================
            if step < self.num_warmup_steps:
                # current_lr = ???
                current_lr = float(step) / float(max(1, self.num_warmup_steps)) * base_lr
            
            # ==========================================
            # TODO 2: 阶段 2 - Stable (稳定)
            # 规则: 当 num_warmup_steps <= step < (num_warmup_steps + num_stable_steps)
            # 返回: 维持最大学习率 base_lr
            # ==========================================
            elif step < (self.num_warmup_steps + self.num_stable_steps):
                # current_lr = ???
                current_lr = base_lr
                
            # ==========================================
            # TODO 3: 阶段 3 - Cosine Decay (余弦退火)
            # 规则: 超过稳定期后的最后衰减阶段。
            # 提示: 计算在 decay 阶段的进度比例 progress (0.0 -> 1.0)
            # 利用公式: lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))
            # ==========================================
            else:
                decay_start_step = self.num_warmup_steps + self.num_stable_steps
                # ==========================================
                # TODO 3.1: 计算当前处于 decay 阶段的第几步 (需要注意不要超过总 decay 步数)
                # ==========================================
                # steps_in_decay = ???
                steps_in_decay = min(step - decay_start_step, self.num_decay_steps)
                
                # progress = ???
                progress = float(steps_in_decay) / float(max(1, self.num_decay_steps))
                
                # current_lr = ???
                current_lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
                
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
            # 注意顺序: optimizer.step() 后接 scheduler.step()
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
        
        print("🔥 你成功实现并可视化了目前最先进的大模型学习率调度器。现在你不怕被面试官问到 LLaMA-3 的退火策略了！")
        
    except NotImplementedError:
        print("请先完成 TODO 代码！")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

test_and_plot_wsd()

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

::: details 💡 点击查看官方解析与参考代码

学习率调度器对于稳定训练至关重要。代码中展示了经典的余弦退火和带有Warmup的调度策略，通过逐步改变学习率，确保模型在初期能够快速收敛，在后期能精细搜索最优解。

```python
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
