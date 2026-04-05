# 14 Decoding Strategies

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/14_Decoding_Strategies.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 14. 大模型解码策略：Top-K, Top-p (Nucleus) 与 Temperature

**难度：** Medium | **标签：** `推理算法`, `Decoding` | **目标人群：** 模型微调与工程部署

有了 `Logits` 之后，我们如何决定大模型生成的下一个词是什么？
这就是解码策略（Decoding Strategy）解决的问题。如果只取概率最大的词（Greedy Search），模型生成的文本会非常干瘪且容易重复。为了让生成的文本既有逻辑性又有创造性，我们需要引入**温度采样（Temperature）**、**Top-K** 和 **Top-p（核采样）**。

在面试大模型算法岗（无论是微调还是部署）时，这三个算法的张量实现是必考的！


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


### Step 2: 代码实现框架
在自回归解码中，我们获取最后一个 Token 的概率分布后：
- **Temperature**: 将 Logits 除以 $T$。
- **Top-K**: 用 `torch.topk` 找出前 K 个最大的概率，将其他的 Logits 置为 $-\infty$。
- **Top-P (Nucleus)**: 对 Logits 降序排列并计算累积概率，将累积概率超过 $P$ 的位置对应的 Logits 屏蔽掉。
最后使用 `torch.multinomial` 依概率进行采样。


###  Step 3: 核心机制：Softmax 截断与重整化

**面试必考难点：Top-p 是如何用代码实现的？**
假设词表只有 5 个词，排序后的概率分别是：`[0.5, 0.3, 0.1, 0.05, 0.05]`。阈值 $p = 0.85$。
1. 计算累加和 (Cumulative Sum, `cumsum`)：`[0.5, 0.8, 0.9, 0.95, 1.0]`
2. 找出那些**累加和超过 0.85 的位置**（即 `[False, False, True, True, True]`）
3. 把这些位置对应的原始 Logits 强制设为 `-inf`
4. 对剩下的有效 Logits 重新执行一次 `Softmax` 进行概率归一化。


###  Step 4: 动手实战

**要求**：请依次补全 `apply_temperature`、`apply_top_k` 和 `apply_top_p` 三个函数的核心逻辑，并组合成最终的采样函数。


```python
import torch
import torch.nn.functional as F

def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    应用温度调节。注意：通常 T=1.0 意味着不改变，T 越接近 0 越确定（Greedy）。
    """
    # ==========================================
    # TODO 1: 确保 temperature 至少大于一个极小值(如 1e-6) 防止除零
    # 然后让 logits 统一除以 temperature
    # ==========================================
    # temp = max(temperature, 1e-6)
    # return ???
    pass

def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Top-K 截断。只保留值最大的 top_k 个，其余置为 -inf。
    """
    # 如果 top_k 是 0 或超过词表大小，不处理
    if top_k <= 0 or top_k >= logits.size(-1):
        return logits
        
    # ==========================================
    # TODO 2: 实现 Top-K 截断
    # 提示: 使用 torch.topk 找出第 K 大的值，它是个一维标量还是张量？
    # 对于每个 batch，凡是严格小于这个 K_th value 的 logit，一律设为 -float('Inf')
    # ==========================================
    # filter_value = float('-inf')
    # kth_values = torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
    
    # logits[logits < kth_values] = ???
    # return logits
    pass

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
    # TODO 3: 实现 Top-P 核心逻辑
    # 1. 找出 cumulative_probs > top_p 的所有位置。这就是我们需要剔除(丢弃)的 token
    # 2. 我们想保留第一个累加概率 > p 的词，所以需要把这个掩码向右平移一位
    # 3. 将这些被剔除位置对应的 sorted_logits 设为 -inf
    # 4. 把剔除后的 sorted_logits 按照 sorted_indices 散布 (scatter) 回原来的形状里
    # ==========================================
    
    # 找到需要丢弃的掩码
    # sorted_indices_to_remove = cumulative_probs > top_p
    
    # 向右平移掩码以保留最后一个刚好超过阈值的 token
    # sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    # sorted_indices_to_remove[..., 0] = 0  # 确保无论如何最高概率的 token 不被丢弃
    
    # 将需要剔除的 sorted_logits 设为极小值
    # sorted_logits[sorted_indices_to_remove] = float('-inf')
    
    # 将其散布回原始的顺序中 (使用 torch.scatter_)
    # restored_logits = torch.zeros_like(logits).scatter_(
    #     dim=-1, index=sorted_indices, src=sorted_logits
    # )
    
    # return restored_logits
    pass

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
        # top_p=0.8 时，0.54 + 0.22 = 0.76 < 0.8，还需要加上 0.10 才会超 0.8
        # 所以只有 idx 5, 6, 1 会被保留
        p_logits = apply_top_p(logits.clone(), 0.8)
        valid_count = (p_logits != float('-inf')).sum().item()
        assert valid_count == 3, f"Top-p 核采样截断没算准，保留了 {valid_count} 个值"
        print("✅ Top-p (Nucleus) 核采样动态截断通过！")
        
        # 4. 测试完整管线
        next_token = decode_next_token(logits.clone(), temperature=0.7, top_k=50, top_p=0.9)
        assert next_token.shape == (1, 1), "解码的词张量维度不对"
        print(f"\n✅ All Tests Passed! 恭喜你，大模型解码流水线组装完成！本次采样的下一个词汇 ID 是: {next_token.item()}")
        
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
    except TypeError as e:
        print("代码可能未完成，导致了操作错误。")
    except Exception as e:
        print(f"❌ 发生未知异常: {e}")

test_decoding()

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

::: details 💡 点击查看官方解析与参考代码

自回归模型的解码策略（如Top-K和Top-P/Nucleus Sampling）直接影响文本生成的多样性和连贯性。此实现通过在 logits 上进行排序或累积概率筛选，过滤掉低概率词汇，然后重新归一化以便进行多项式采样。

```python
def top_p_filtering(logits, top_p=0.9, filter_value=-float('Inf')):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = filter_value
    return logits
```

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
