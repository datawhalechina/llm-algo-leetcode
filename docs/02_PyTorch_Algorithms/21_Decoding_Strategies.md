# 21. Decoding Strategies | 解码策略
**难度：** Medium | **环境：** CPU-first | **标签：** `推理优化`, `解码`, `Sampling` | **目标人群：** 推理优化学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/21_Decoding_Strategies.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

模型生成下一个 token 时，会先为整张词表打分，再从候选中选择一个结果。永远选择最高分比较稳定，但可能缺少变化；完全随机又难以控制。解码策略要解决的，就是如何根据分数分布和选择规则，在确定性与多样性之间作出可解释的取舍。

学习时先用 greedy 建立确定性基线，再观察 temperature 如何改变分布形状、top-k / top-p 如何改变候选集合，最后把候选数量、熵和随机性放在一起比较。这样可以从“分数如何变”逐步理解“候选如何选”和“结果为何不同”。

**关键词：** `top-k`, `top-p`, `temperature`

---

## 前置阅读

**导语：** 先回顾注意力模块如何产生当前位置的表示，以及模型如何将表示映射为词表 logits；然后观察不同解码规则怎样改变候选集合和采样结果。
- [04. Attention MHA GQA | 注意力机制：MHA、MQA、GQA](./04_Attention_MHA_GQA.md)
- [05. LLaMA3 Block Tutorial | LLaMA3 Block 实现](./05_LLaMA3_Block_Tutorial.md)

---

### Step 1: 从词表分数到下一个 token
模型在每个生成位置输出一组词表 logits。解码的输入是这组分数和选择配置，处理过程可以改变分布形状、缩小候选范围，最后输出一个下一个 token：`greedy` 取最高分，sampling 从重整化后的概率中抽样。

| 输入或路径 | 主要处理 | 输出与观察重点 |
|---|---|---|
| 原始 logits | 为每个 token 提供分数 | 候选排序与分布形状 |
| greedy | 直接选择最高分 token | 结果确定，便于作为基线 |
| sampling | 根据重整化概率抽样 | 结果具有多样性和随机性 |
| Top-K / Top-p | 在 sampling 前缩小候选集合 | 候选空间和概率质量发生变化 |

![解码策略流程](../public/02_PyTorch_Algorithms/21_decoding_pipeline.svg)

### Step 2: 候选处理顺序如何改变分布
把最后一个位置的 logits 依次处理，再进入确定性选择或概率采样。本节采用 Temperature → Top-K → Top-p → Softmax → 选择的顺序：前两步改变分数或候选范围，Softmax 之后才得到用于 sampling 的概率。temperature 很小时仍可能采样到其他 token，不能把它当作 greedy。

| 阶段 | 输入 | 输出 / 作用 | 观察重点 |
|---|---|---|---|
| Temperature | 原始 logits、温度 $T$ | 缩放后的 logits | 排序不变，分布尖锐程度改变 |
| Top-K | 缩放后的 logits、$K$ | 只保留前 K 个候选 | 固定数量；相同分数可能保留超过 K 个 |
| Top-p | 候选 logits、$p$ | 保留累计概率达到阈值的最小集合 | 必须保留首次达到阈值的边界 token |
| 选择 | 重整化后的概率或 logits | 一个下一个 token | greedy 取最大值，sampling 按概率选择 |

### Step 3: Top-p 如何确定边界并重整化概率

Top-p 的关键是保留累计概率首次达到或超过阈值的边界 token。假设排序后的概率为 `[0.5, 0.3, 0.1, 0.05, 0.05]`，当 $p=0.85$ 时，累计概率在第三个 token 首次超过阈值，因此前三个 token 进入候选集合。

过滤后，候选之外的 logits 设为 `-inf`，再对保留的 logits 做一次 Softmax；这样采样概率会在新的候选集合内重新归一化。

| 处理步骤 | 作用 | 示例或检查点 |
|---|---|---|
| 排序与累加 | 按概率从高到低计算 `cumsum` | `[0.5, 0.3, 0.1, 0.05, 0.05]` → `[0.5, 0.8, 0.9, 0.95, 1.0]` |
| 保留边界 | 保留累计概率首次达到或超过 $p$ 的 token | $p=0.85$ 时保留前三个 token，边界 token 不能被误删 |
| 屏蔽候选 | 将边界之后的 logits 设为 `-inf` | 被屏蔽 token 不再参与选择 |
| 重整化 | 对保留 logits 再执行 Softmax | 候选概率重新归一化为 1 |

![Top-p 边界保留示意](../public/02_PyTorch_Algorithms/21_top_p_boundary.svg)
### Step 4: 实现候选过滤并验证解码流程

本 Step 将前面的解码流程落到代码：输入是最后一个位置的 logits 和解码配置，输出是下一个 token 或追加后的 token 序列。题目区只实现三个会改变候选分布的步骤：`apply_temperature`、`apply_top_k`、`apply_top_p`；`decode_next_token` 与 `autoregressive_decode` 保留为已给出的组合骨架，帮助你观察这些过滤步骤如何进入 greedy、sampling 与逐 token 循环。

测试分别检查温度缩放、Top-K ties、Top-p 边界、选择路径与输入契约。CPU 合成 logits 验证的是选择规则、概率重整化和可重复性，不代表真实模型质量、TTFT 或吞吐。

| 实现对象 | 需要完成或阅读的内容 | 验证重点 |
|---|---|---|
| TODO 1：温度缩放 | 实现 `apply_temperature` | 排序、shape、dtype/device 与非法温度 |
| TODO 2：Top-K | 实现阈值过滤 | ties、候选集合与 `-inf` 屏蔽 |
| TODO 3：Top-p | 实现排序空间边界与恢复原顺序 | 边界 token、重整化前 logits 与 batch 形状 |
| 已给出组合骨架 | 阅读 `decode_next_token` 与 `autoregressive_decode` | greedy/sampling、随机种子与输出长度 |


```python
import torch
import torch.nn.functional as F
```


```python
def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """在 Softmax 前缩放 logits。

    参数：
        logits: 最后一维为词表维度的分数张量，支持单样本或 batch。
        temperature: 有限正数；越小通常使分布更尖锐。
    """
    # ==========================================
    # TODO 1: 检查 temperature 并完成缩放
    # 要求：temperature 必须是有限正数；输出保持 logits 的 shape、dtype 和 device，
    #       只改变分数尺度，不改变 token 的排序关系。
    # ==========================================
    # if temperature <= 0 or not torch.isfinite(torch.tensor(temperature)):
    #     raise ValueError('temperature 必须是有限正数')
    # temp = ???  # 有限正数的 temperature
    raise NotImplementedError("TODO 1：请校验并使用 temperature")
    return logits / temp

def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """按第 K 大值筛选 logits，保留 ties 是本实现的明确语义。

    `top_k=0` 或不小于词表大小时保持输入不变。
    """
    if top_k is None:
        return logits
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
        raise ValueError('top_k 必须为非负整数')
    if top_k == 0 or top_k >= logits.size(-1):
        return logits
        
    # ==========================================
    # TODO 2: Top-K 截断
    # 先找到第 K 大值作为阈值，再保留所有不小于阈值的位置；ties 可能使保留数超过 K。
    # 输出仍为 [..., vocab]，被过滤位置改为 -inf。
    # ==========================================
    # filter_value = ???  # 被过滤位置使用的 -inf
    # kth_values = ???  # 第 K 大阈值，形状保留最后一维
    # logits = ???  # 小于阈值的位置置为 filter_value
    return logits

def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """按累计概率保留最小候选集合，并恢复原始词表顺序。

    边界 token 会被保留；因此这里的 top-p 是“首次达到阈值”的语义。
    """
    if top_p is None or top_p == 1.0:
        return logits
    if not 0.0 < top_p < 1.0:
        raise ValueError('top_p 必须满足 0 < top_p <= 1')
        
    # 1. 首先需要将 logits 从大到小排序
    # 注意我们需要记住原始的索引 (indices)，因为截断完了还要把它复原回原来的位置！
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    
    # 2. 对排序后的 logits 做 Softmax，再计算累加概率
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # ==========================================
    # TODO 3: Top-p 核心逻辑
    # 先在排序空间确定边界，保留首次达到阈值的 token；再屏蔽后续 token 并恢复原词表顺序。
    # 输出形状必须与 logits 一致，函数只返回过滤后的 logits，不提前 Softmax。
    # ==========================================
    # sorted_indices_to_remove = ???  # 排序空间中超过边界的 token 掩码
    # sorted_indices_to_remove[..., 1:] = ???  # 将前一位置的累计概率右移
    # sorted_indices_to_remove[..., 0] = ???  # 保留首次达到阈值的边界 token
    # sorted_logits[sorted_indices_to_remove] = ???  # 屏蔽后续 token
    # restored_logits = ???  # 按 sorted_indices 恢复原词表顺序
    return restored_logits

def decode_next_token(logits: torch.Tensor, temperature=0.7, top_k=50, top_p=0.9, do_sample=True, generator=None):
    """完成一次候选过滤和 token 选择，支持 greedy 与可复现 sampling。

    输入可以是 `[vocab]` 或 `[batch, vocab]`；内部统一为二维，输出统一为 `[batch, 1]`。
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    elif logits.dim() != 2:
        raise ValueError('logits 必须是 [vocab] 或 [batch, vocab]')
    # 1. 调温
    logits = apply_temperature(logits, temperature)
    
    # 2. Top-K 截断 (通常先 K 后 p)
    logits = apply_top_k(logits, top_k)
    
    # 3. Top-p 截断
    logits = apply_top_p(logits, top_p)
    
    # 4. 概率重归一化
    probs = F.softmax(logits, dim=-1)
    
    # 5. greedy 直接取最大值；sampling 才使用 multinomial
    if not do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)
    next_token = torch.multinomial(probs, num_samples=1, generator=generator)
    
    return next_token

def autoregressive_decode(prompt_ids, logits_fn, max_new_tokens, **decode_kwargs):
    """用 logits_fn 演示逐 token 解码；不代表真实模型性能。"""
    tokens = prompt_ids.clone()
    for _ in range(max_new_tokens):
        step_logits = logits_fn(tokens)
        if step_logits.dim() == 3:
            step_logits = step_logits[:, -1, :]
        next_token = decode_next_token(step_logits, **decode_kwargs)
        tokens = torch.cat([tokens, next_token], dim=-1)
    return tokens

```


```python
# 测试设计：分别验证温度、Top-K、Top-p、选择循环与输入契约。

def _fixture_logits():
    return torch.tensor([[0.1, 2.3, 0.4, 1.2, -0.5, 4.0, 3.1, 0.0, 1.1, -1.0]])

def test_temperature_contract():
    logits = _fixture_logits()
    scaled = apply_temperature(logits, 0.5)
    assert torch.allclose(scaled[0, 5] - scaled[0, 6], (logits[0, 5] - logits[0, 6]) * 2)
    assert scaled.shape == logits.shape and scaled.dtype == logits.dtype
    assert torch.equal(torch.argsort(logits), torch.argsort(scaled))
    try:
        apply_temperature(logits, 0.0)
    except ValueError:
        return
    raise AssertionError('非法 temperature 未被拒绝')

def test_top_k_filtering():
    logits = _fixture_logits()
    assert torch.isfinite(apply_top_k(logits, 3)).sum().item() == 3
    tied = torch.tensor([[2.0, 2.0, 2.0, 1.0]])
    assert torch.isfinite(apply_top_k(tied, 2)).sum().item() == 3

def test_top_p_boundary():
    logits = _fixture_logits()
    filtered = apply_top_p(logits, 0.8)
    assert torch.isfinite(filtered).sum().item() == 3
    assert torch.equal(filtered[0, [5, 6, 1]], logits[0, [5, 6, 1]])
    for invalid_p in (0.0, 1.1):
        try:
            apply_top_p(logits, invalid_p)
        except ValueError:
            continue
        raise AssertionError('非法 top_p 未被拒绝')

def test_selection_and_autoregressive_loop():
    logits = _fixture_logits()
    assert decode_next_token(logits, 1.0, 0, 1.0, do_sample=False).item() == 5
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    assert torch.equal(decode_next_token(logits, generator=g1), decode_next_token(logits, generator=g2))
    prompt = torch.tensor([[1, 2]])
    def fake_logits(tokens):
        result = torch.zeros(tokens.size(0), tokens.size(1), 10)
        result[..., 3] = 2.0
        return result
    generated = autoregressive_decode(prompt, fake_logits, 3, temperature=1.0, top_k=0, top_p=1.0, do_sample=False)
    assert generated.shape == (1, 5) and torch.equal(generated[0, -3:], torch.tensor([3, 3, 3]))

def test_batch_and_input_contract():
    logits = _fixture_logits().repeat(2, 1)
    assert apply_top_k(logits, 3).shape == logits.shape
    assert apply_top_p(logits, 0.8).shape == logits.shape
    assert decode_next_token(logits[0], 1.0, 0, 1.0, do_sample=False).shape == (1, 1)

def run_decoding_tests():
    test_temperature_contract()
    test_top_k_filtering()
    test_top_p_boundary()
    test_selection_and_autoregressive_loop()
    test_batch_and_input_contract()
    print('✅ 解码机制测试通过：温度、候选过滤、选择循环与输入契约均符合预期。')

# 以下旧单体测试保留到本轮迁移完成前；执行入口已切换至 run_decoding_tests。
run_decoding_tests()

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
    """在 Softmax 前缩放 logits；temperature 必须是有限正数。"""
    # TODO 1: 校验 temperature，不要用静默截断替代非法参数处理
    # 要求：必须是有限正数；输出保持 logits 的 shape、dtype 和 device，只改变分数尺度。
    if not torch.isfinite(torch.tensor(temperature)) or temperature <= 0:
        raise ValueError('temperature 必须是有限正数')
    temp = temperature
    return logits / temp

def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Top-K 截断；按第 K 大值筛选，分数相同时可能保留多个 token。
    """
    if top_k is None:
        return logits
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
        raise ValueError('top_k 必须为非负整数')
    if top_k == 0 or top_k >= logits.size(-1):
        return logits
        
    # TODO 2: 实现 Top-K 截断
    # 先找到第 K 大值作为阈值，再保留所有不小于阈值的位置；ties 可能使保留数超过 K。
    filter_value = float('-inf')
    
    # 找到第 K 大的值
    kth_values, _ = torch.topk(logits, top_k, dim=-1, largest=True, sorted=True)
    kth_values = kth_values[..., -1:] # 取最后一个（第 K 大的值）
    
    # 将小于第 K 大值的位置设为 -inf
    logits = torch.where(logits < kth_values, torch.tensor(filter_value, device=logits.device), logits)
    return logits

def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Top-p 截断，保留累计概率首次达到阈值的最小候选集合。
    """
    if top_p is None or top_p == 1.0:
        return logits
    if not 0.0 < top_p < 1.0:
        raise ValueError('top_p 必须满足 0 < top_p <= 1')
        
    # 1. 首先需要将 logits 从大到小排序
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    
    # 2. 对排序后的概率计算累加和
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # TODO 3: 实现 Top-P 核心逻辑
    # 先在排序空间确定边界，保留首次达到阈值的 token；再屏蔽后续 token 并恢复原词表顺序。
    # 找到需要丢弃的掩码，并向右平移以保留边界 token
    sorted_indices_to_remove = cumulative_probs > top_p
    
    # 向右平移掩码以保留第一个达到阈值的 token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0  # 确保无论如何最高概率的 token 不被丢弃
    
    # 将需要剔除的 sorted_logits 设为极小值
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    
    # 将排序后的 logits 恢复到原始顺序
    restored_logits = torch.zeros_like(logits).scatter_(
        dim=-1, index=sorted_indices, src=sorted_logits
    )
    
    return restored_logits

def decode_next_token(logits: torch.Tensor, temperature=0.7, top_k=50, top_p=0.9, do_sample=True, generator=None):
    """应用策略并选择一个 token；do_sample=False 时使用 greedy。

    输入可以是 `[vocab]` 或 `[batch, vocab]`；内部统一为二维，输出统一为 `[batch, 1]`。
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    elif logits.dim() != 2:
        raise ValueError('logits 必须是 [vocab] 或 [batch, vocab]')
    # 1. 调温
    logits = apply_temperature(logits, temperature)
    
    # 2. Top-K 截断 (通常先 K 后 p)
    logits = apply_top_k(logits, top_k)
    
    # 3. Top-p 截断
    logits = apply_top_p(logits, top_p)
    
    # 4. 概率重归一化
    probs = F.softmax(logits, dim=-1)
    
    # 5. greedy 与 sampling 是两条不同路径
    if not do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)
    next_token = torch.multinomial(probs, num_samples=1, generator=generator)
    
    return next_token

def autoregressive_decode(prompt_ids, logits_fn, max_new_tokens, **decode_kwargs):
    """用一个 logits_fn 演示逐 token 解码；不代表真实模型性能。"""
    tokens = prompt_ids.clone()
    for _ in range(max_new_tokens):
        step_logits = logits_fn(tokens)
        if step_logits.dim() == 3:
            step_logits = step_logits[:, -1, :]
        next_token = decode_next_token(step_logits, **decode_kwargs)
        tokens = torch.cat([tokens, next_token], dim=-1)
    return tokens
```

### 解析

**1. TODO 1: Temperature 温度调节**
- **实现方式**：先验证 `temperature` 是有限正数，再执行 `logits / temperature`。
- **关键点**：温度改变概率分布的尖锐程度，但不改变 logits 的排序；它很小时仍不是严格 greedy。
- **边界**：非法温度应报错，greedy 由 `do_sample=False` 单独表达。

**2. TODO 2: Top-K 截断**
- **实现方式**：`kth_values = torch.topk(logits, top_k)[0][..., -1:]`，`logits = torch.where(logits < kth_values, -inf, logits)`
- **关键点**：只保留分数不低于第 K 大值的候选，其余 logits 设为负无穷。
- **技术细节**：使用 `torch.topk` 找到阈值，再用 `torch.where` 广播到 batch 维；分数相同时可能保留超过 K 个候选。

**3. TODO 3: Top-p (Nucleus) 核采样**
- **实现方式**：对 logits 降序排序，计算累积概率，屏蔽首次达到阈值之后的候选，最后用 `scatter_` 恢复原始顺序。
- **关键点**：动态截断——根据概率分布的形状自动决定保留多少个词
- **技术细节**：
  - Top-p 会先用 Softmax 计算累计概率，但函数最终返回的是过滤后的 logits；最终采样前还会重新 Softmax。
  - 向右平移掩码（`sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()`）确保保留边界 token
  - 使用 `scatter_` 将排序后的 logits 恢复到原始索引顺序

**工程边界与延伸**
- 本实现采用 Temperature → Top-K → Top-p；换序会改变候选集合，生产 backend 需以实际实现为准。
- 使用 `-float('inf')` 保持张量形状不变；Softmax 后候选概率会重新归一化。
- `autoregressive_decode` 只验证逐 token 控制流和形状，不包含真实模型、KV Cache 或 backend 优化。
- 候选数、熵、重复率和采样耗时可以作为 CPU 对比指标；真实生成质量、TTFT、TPOT 和吞吐需要后续模型/服务 benchmark。
## 相关阅读

完成本节的候选过滤和采样实现后，可以沿着“采样方法 → 真实生成接口 → 解码服务策略”继续阅读。

- [The Curious Case of Neural Text Degeneration：Nucleus Sampling 原论文](https://arxiv.org/abs/1904.09751)
- [Attention Is All You Need 原论文](https://arxiv.org/abs/1706.03762)
- [Transformers 文本生成策略文档](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
- [vLLM SamplingParams 文档](https://docs.vllm.ai/en/latest/api/vllm/sampling_params.html)
- [Part 02 · 23 投机解码](./23_Speculative_Decoding.md)
- [Part 02 · 35 多 Token 解码](./35_Multi_Token_Decoding.md)
- [Part 02 · 36 解码调度](./36_Decode_Scheduling.md)
