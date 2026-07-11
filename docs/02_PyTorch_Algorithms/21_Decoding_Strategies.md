# 21. Decoding Strategies | 解码策略
**难度：** Medium | **环境：** CPU-first | **标签：** `解码`, `Sampling`, `推理优化` | **目标人群：** 推理与部署工程师

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/21_Decoding_Strategies.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


先把生成时的采样与搜索策略理顺，再理解解码阶段为什么会影响吞吐和质量。

**关键词：** `top-k`, `top-p`, `temperature`


## 前置阅读

**导语：** 先理解张量、训练闭环和显存分析，再看解码策略会更容易。
- [11. PyTorch Optimizers and Loss | PyTorch 优化器与损失函数](../00_PyTorch_Fundamentals/11_PyTorch_Optimizers_and_Loss.md)
- [12. PyTorch Minimal Training Interface | PyTorch 最小训练接口](../00_PyTorch_Fundamentals/12_PyTorch_Minimal_Training_Interface.md)
- [13. Simple Neural Network Training | 简单神经网络训练循环](../00_PyTorch_Fundamentals/13_Simple_Neural_Network_Training.md)

## 相关阅读

**导语：** 解码策略之后，可以继续看 KV Cache、显存增长和 FlashAttention。
- [11. KV Cache and Memory Growth | KV Cache 与显存增长](../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md)
- [13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
- [14. FlashAttention Memory Model | FlashAttention 显存模型](../01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.md)

### Step 1: 核心思想与痛点

> **大模型的最后一步输出什么？**
> 模型对下一个词的预测是一串未经归一化的得分，称为 `Logits`（形状为 `[vocab_size]`，比如 32000 个数字）。
> 
> **三种常用的截断与平滑策略：**
> 1. **Temperature ($T$)**：在 Softmax 之前，将所有 `Logits` 除以 $T$。
>    - $T < 1$：拉大差距，让高分更高，低分更低（结果更确定）。
>    - $T > 1$：缩小差距，让得分变得平均（结果更随机，也就是更“胡言乱语”）。
> 2. **Top-K 截断**：只保留得分最高的 $K$ 个词的概率，把排名第 $K+1$ 之后的词全部强制剔除（概率置为 $-\infty$）。
> 3. **Top-p (Nucleus) 核采样**：动态截断。按概率从大到小排序，当累加的概率刚好超过阈值 $p$ 时，截断后面的词。它可以根据分布的平缓程度，自动决定截断的数量。


这三种策略本质上都是在修改 logits 的分布形状，下一步看它们如何串成完整的解码流水线。

### Step 2: 代码实现框架
在自回归解码中，我们获取最后一个 Token 的概率分布后：
- **Temperature**: 将 Logits 除以 $T$。
- **Top-K**: 用 `torch.topk` 找出前 K 个最大的概率，将其他的 Logits 置为 $-\infty$。
- **Top-P (Nucleus)**: 对 Logits 降序排列并计算累积概率，将累积概率超过 $P$ 的位置对应的 Logits 屏蔽掉。
最后使用 `torch.multinomial` 依概率进行采样。

这一步的顺序就是先 Temperature，再 Top-K / Top-p，最后 Softmax 重归一化并用 Multinomial 采样。

###  Step 3: 核心机制：Softmax 截断与重整化

**面试必考难点：Top-p 是如何用代码实现的？**
假设词表只有 5 个词，排序后的概率分别是：`[0.5, 0.3, 0.1, 0.05, 0.05]`。阈值 $p = 0.85$。
1. 计算累加和 (Cumulative Sum, `cumsum`)：`[0.5, 0.8, 0.9, 0.95, 1.0]`
2. 找出那些**累加和超过 0.85 的位置**（即 `[False, False, True, True, True]`）
3. 把这些位置对应的原始 Logits 强制设为 `-inf`
4. 对剩下的有效 Logits 重新执行一次 `Softmax` 进行概率归一化。

最后记住：前面三步修改的是 logits 的筛选范围，最后的 Softmax 和采样才决定真正输出哪个 token。

###  Step 4: 动手实战

**要求**：请依次补全 `apply_temperature`、`apply_top_k` 和 `apply_top_p` 三个函数的核心逻辑，并组合成最终的采样函数。


```python
import torch
import torch.nn.functional as F
```


```python
def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    应用温度调节。注意：通常 T=1.0 意味着不改变，T 越接近 0 越确定（Greedy）。
    """
    # ==========================================
    # TODO 1: 温度下限与缩放
    # ==========================================
    # temp = ???
    return logits / temp

def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Top-K 截断。只保留值最大的 top_k 个，其余置为 -inf。
    """
    if top_k <= 0 or top_k >= logits.size(-1):
        return logits
        
    # ==========================================
    # TODO 2: Top-K 截断
    # ==========================================
    # filter_value = ???
    # kth_values = ???
    # logits = ???
    return logits

def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Top-p (Nucleus) 核采样截断。
    """
    if top_p <= 0.0 or top_p >= 1.0:
        return logits
        
    # 1. 首先需要将 logits 从大到小排序
    # 注意我们需要记住原始的索引 (indices)，因为截断完了还要把它复原回原来的位置！
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    
    # 2. 对排序后的概率 (需要先算一遍 Softmax) 计算累加和 (Cumulative Probability)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # ==========================================
    # TODO 3: Top-p 核心逻辑
    # ==========================================
    # sorted_indices_to_remove = ???
    # sorted_logits[sorted_indices_to_remove] = ???
    # restored_logits = ???
    return restored_logits

def decode_next_token(logits: torch.Tensor, temperature=0.7, top_k=50, top_p=0.9):
    """
    组合以上三种策略，并通过 Multinomial 进行随机多项式采样
    """
    # 1. 调温
    logits = apply_temperature(logits, temperature)
    
    # 2. Top-K 截断 (通常先 K 后 p)
    logits = apply_top_k(logits, top_k)
    
    # 3. Top-p 截断
    logits = apply_top_p(logits, top_p)
    
    # 4. 概率重归一化
    probs = F.softmax(logits, dim=-1)
    
    # 5. 从概率分布中采样 1 个词
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token

```


```python
# 运行此单元格以测试你的实现
def test_decoding():
    try:
        # 为了保证可重复性
        torch.manual_seed(42)
        vocab_size = 10
        # 伪造一组 Logits: [0.1, 2.3, 0.4, 1.2, -0.5, 4.0, 3.1, 0.0, 1.1, -1.0]
        # 最大的两个: index 5 (4.0), index 6 (3.1)
        logits = torch.tensor([[0.1, 2.3, 0.4, 1.2, -0.5, 4.0, 3.1, 0.0, 1.1, -1.0]])
        
        print("原始 Logits (前10个单词):", logits.squeeze().tolist())
        
        # 1. 测试 Temperature
        t_logits = apply_temperature(logits.clone(), 0.5)
        # 温度 0.5 应该会让差异翻倍
        assert torch.allclose(t_logits[0, 5] - t_logits[0, 6], (logits[0, 5] - logits[0, 6]) * 2), "温度调节错误！"
        print("✅ Temperature 热量调节通过！")
        
        # 2. 测试 Top-K
        k_logits = apply_top_k(logits.clone(), 3)
        # 只保留最大的三个：5, 6, 1
        valid_count = (k_logits != float('-inf')).sum().item()
        assert valid_count == 3, f"Top-K 截断没有正确执行，保留了 {valid_count} 个值"
        print("✅ Top-K 暴力截断通过！")
        
        # 3. 测试 Top-p
        # 原始概率: [0.01, 0.10, 0.01, 0.03, 0.00, 0.54, 0.22, 0.01, 0.03, 0.00]
        # 降序: 0.54 (idx 5), 0.22 (idx 6), 0.10 (idx 1) ...
        # 累加和: 0.54, 0.76, 0.86
        # 所以只有 idx 5, 6, 1 会被保留
        p_logits = apply_top_p(logits.clone(), 0.8)
        valid_count = (p_logits != float('-inf')).sum().item()
        assert valid_count == 3, f"Top-p 核采样截断没算准，保留了 {valid_count} 个值"
        print("✅ Top-p (Nucleus) 核采样动态截断通过！")
        
        # 4. 测试完整管线
        next_token = decode_next_token(logits.clone(), temperature=0.7, top_k=50, top_p=0.9)
        assert next_token.shape == (1, 1), "解码的词张量维度不对"
        print(f"\n✅ All Tests Passed! 解码策略实现通过测试。本次采样的下一个词汇 ID 是: {next_token.item()}")
        
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError, RuntimeError) as e:
        if isinstance(e, AttributeError):
            print("代码未完成，无法找到必要的属性")
        elif isinstance(e, NameError):
            print("代码可能未完成，导致了变量未定义")
        elif isinstance(e, TypeError):
            print("代码可能未完成，导致了操作错误")
        elif isinstance(e, ValueError):
            print("代码可能未完成，导致了张量维度错误")
        elif isinstance(e, RuntimeError):
            print("代码可能未完成，导致了运行时错误")
        else:
            print("代码可能未完成，导致了断言失败")
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise

test_decoding()

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
def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    应用温度调节。注意：通常 T=1.0 意味着不改变，T 越接近 0 越确定（Greedy）。
    """
    # TODO 1: 确保 temperature 至少大于一个极小值(如 1e-6) 防止除零
    temp = max(temperature, 1e-6)
    return logits / temp

def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Top-K 截断。只保留值最大的 top_k 个，其余置为 -inf。
    """
    if top_k <= 0 or top_k >= logits.size(-1):
        return logits
        
    # TODO 2: 实现 Top-K 截断
    filter_value = float('-inf')
    
    # 找到第 K 大的值
    kth_values, _ = torch.topk(logits, top_k, dim=-1, largest=True, sorted=True)
    kth_values = kth_values[..., -1:] # 取最后一个（第 K 大的值）
    
    # 将小于第 K 大值的位置设为 -inf
    logits = torch.where(logits < kth_values, torch.tensor(filter_value, device=logits.device), logits)
    return logits

def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Top-p (Nucleus) 核采样截断。
    """
    if top_p <= 0.0 or top_p >= 1.0:
        return logits
        
    # 1. 首先需要将 logits 从大到小排序
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    
    # 2. 对排序后的概率计算累加和
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # TODO 3: 实现 Top-P 核心逻辑
    # 找到需要丢弃的掩码
    sorted_indices_to_remove = cumulative_probs > top_p
    
    # 向右平移掩码以保留最后一个刚好超过阈值的 token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0  # 确保无论如何最高概率的 token 不被丢弃
    
    # 将需要剔除的 sorted_logits 设为极小值
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    
    # 将排序后的 logits 恢复到原始顺序
    restored_logits = torch.zeros_like(logits).scatter_(
        dim=-1, index=sorted_indices, src=sorted_logits
    )
    
    return restored_logits

def decode_next_token(logits: torch.Tensor, temperature=0.7, top_k=50, top_p=0.9):
    """
    组合以上三种策略，并通过 Multinomial 进行随机多项式采样
    """
    # 1. 调温
    logits = apply_temperature(logits, temperature)
    
    # 2. Top-K 截断 (通常先 K 后 p)
    logits = apply_top_k(logits, top_k)
    
    # 3. Top-p 截断
    logits = apply_top_p(logits, top_p)
    
    # 4. 概率重归一化
    probs = F.softmax(logits, dim=-1)
    
    # 5. 从概率分布中采样 1 个词
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token
```

### 解析

**1. TODO 1: Temperature 温度调节**
- **实现方式**：`temp = max(temperature, 1e-6)`，`return logits / temp`
- **关键点**：温度越低（T < 1），分布越尖锐，模型越确定；温度越高（T > 1），分布越平缓，模型越随机
- **技术细节**：防止除零错误，确保 temperature 至少为 1e-6

**2. TODO 2: Top-K 截断**
- **实现方式**：`kth_values = torch.topk(logits, top_k)[0][..., -1:]`，`logits = torch.where(logits < kth_values, -inf, logits)`
- **关键点**：只保留概率最高的 K 个词，其余词的 logits 设为负无穷
- **技术细节**：使用 `torch.topk` 找到第 K 大的值，然后用 `torch.where` 进行条件替换

**3. TODO 3: Top-p (Nucleus) 核采样**
- **实现方式**：对 logits 降序排序，计算累积概率，找到累积概率超过 p 的位置并屏蔽，最后用 `scatter_` 恢复原始顺序
- **关键点**：动态截断——根据概率分布的形状自动决定保留多少个词
- **技术细节**：
  - 向右平移掩码（`sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()`）确保保留第一个超过阈值的词
  - 使用 `scatter_` 将排序后的 logits 恢复到原始索引顺序

**工程优化要点**
- **Temperature 先行**：温度调节应该在截断之前进行，因为它影响概率分布的形状
- **Top-K vs Top-p**：Top-K 是固定数量截断，Top-p 是动态截断，通常先应用 Top-K 再应用 Top-p
- **数值稳定性**：使用 `-float('inf')` 而非直接删除元素，保持张量形状不变
- **采样策略**：`torch.multinomial` 根据归一化后的概率分布进行多项式采样
- **工业实践**：不同任务需要不同的超参数组合——代码生成通常用低温度（0.2-0.5），创意写作用高温度（0.7-1.0）