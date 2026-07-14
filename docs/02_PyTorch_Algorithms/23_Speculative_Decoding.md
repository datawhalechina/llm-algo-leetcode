# 23. Speculative Decoding | 投机解码
**难度：** Hard | **环境：** GPU required | **标签：** `解码`, `Speculative Decoding`, `推理优化` | **目标人群：** 推理加速与系统工程

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/23_Speculative_Decoding.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


先把 PagedAttention 和生成路径看清，再理解草稿模型与验证模型如何协作。

**关键词：** `speculative decoding`, `draft model`, `verification`


## 前置阅读

**导语：** 先看 PagedAttention、KV Cache 和 FlashAttention 记忆模型，再看投机解码会更容易理解草稿模型与验证模型的协作。

- [22. vLLM PagedAttention | vLLM PagedAttention](../02_PyTorch_Algorithms/22_vLLM_PagedAttention.md)
- [P1: 11. KV Cache and Memory Growth | KV Cache 与显存增长](../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md)
- [P1: 14. FlashAttention Memory Model | FlashAttention 显存模型](../01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.md)


## 相关阅读

**导语：** 投机解码之后，可以继续看 RadixAttention 和量化。

- [P1: 13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
- [P1: 17. CUDA Stream and Asynchrony | CUDA Stream 与异步执行](../01_Hardware_Math_and_Systems/17_CUDA_Stream_and_Asynchrony.md)

### Step 1: 原理与公式

投机解码（Speculative Decoding）的核心，不是“让小模型直接替代大模型”，而是让小模型先草拟一段 token，再用大模型逐个验证。这样做的关键问题是：**如何在不改变最终分布的前提下，尽量减少大模型的逐 token 推理次数**。

> **接受概率公式**
> 对于草拟 token $x$，小模型给出的概率记为 $q(x)$，大模型给出的概率记为 $p(x)$。
> - 若 $p(x) \ge q(x)$，直接接受该 token。
> - 若 $p(x) < q(x)$，则以 $\frac{p(x)}{q(x)}$ 的概率接受它。
> 
> 等价地，接受概率可以写成：
> $$\alpha(x) = \min\left(1, \frac{p(x)}{q(x)}\right)$$

**为什么这能做到“无损加速”？**
因为在拒绝的情况下，系统会立刻停止后续草稿 token 的验证，并交还给大模型重新采样，因此最终输出的分布仍然和“只用大模型自回归生成”一致。

### Step 2: 代码实现框架

下面的代码只需要实现一个验证循环：逐个比较 `draft_probs` 和 `target_probs` 中对应 token 的概率，先按 $\alpha(x)$ 决定是否接受，再在拒绝时停止后续验证。这个过程本质上是“带终止条件的接受-拒绝采样”：前面 token 的接受与否，会直接决定后续草稿还能不能继续被验证。


```python
import torch
```


```python
def speculative_verify(draft_probs, target_probs, draft_tokens):
    """
    验证小模型生成的 K 个 Token，返回被接受的 Token 列表。
    
    Args:
        draft_probs: 小模型生成各个 token 时的概率预测分布, shape [K, vocab_size]
        target_probs: 大模型对这 K 个位置的真实概率预测分布, shape [K, vocab_size]
        draft_tokens: 小模型实际采样出的 K 个 token_id, shape [K]
        
    Returns:
        accepted_tokens: list, 最终被接受的 token_id 序列
    """
    K = len(draft_tokens)
    accepted_tokens = []
    
    for i in range(K):
        token_id = draft_tokens[i]
        
        # 提取目标概率 p 和草拟概率 q
        p = target_probs[i, token_id].item()
        q = draft_probs[i, token_id].item()
        
        # ==========================================
        # TODO 1: 判断是否 100% 接受
        # 提示: p >= q 时直接接受
        # if p >= q:
        #     accepted_tokens.append(token_id)
        # ==========================================
        # TODO 2: 以 p / q 的概率接受
        # 提示: 否则按 p/q 掷硬币，拒绝则停止验证
        # r = ???
        # if r < p / q:
        #     accepted_tokens.append(token_id)
        # else:
        #     break
        pass
    
    return accepted_tokens


```


```python
def test_speculative_decoding():
    try:
        torch.manual_seed(42)
        vocab_size = 100
        K = 4
        
        # 模拟生成
        draft_tokens = [10, 20, 30, 40]
        
        draft_probs = torch.rand(K, vocab_size)
        target_probs = torch.rand(K, vocab_size)
        
        # 强行设定：
        # 第 0 个 token: p > q (必接受)
        target_probs[0, 10] = 0.8
        draft_probs[0, 10] = 0.5
        
        # 第 1 个 token: p < q, 但随机数使得它刚好被接受 (p=0.4, q=0.5, p/q=0.8, rand设为0.5)
        target_probs[1, 20] = 0.4
        draft_probs[1, 20] = 0.5
        
        # 第 2 个 token: p 远小于 q，导致拒绝 (p=0.1, q=0.9, p/q=0.11, rand设为0.9)
        target_probs[2, 30] = 0.1
        draft_probs[2, 30] = 0.9
        
        original_rand = torch.rand
        def mock_rand(*args, **kwargs):
            # 依次返回 0.5, 0.9 供判断
            if not hasattr(mock_rand, 'call_count'):
                mock_rand.call_count = 0
            mock_rand.call_count += 1
            if mock_rand.call_count == 1:
                return torch.tensor([0.5])
            else:
                return torch.tensor([0.9])
        torch.rand = mock_rand
        
        accepted = speculative_verify(draft_probs, target_probs, draft_tokens)
        
        # 恢复
        torch.rand = original_rand
        
        assert accepted == [10, 20], f"期望只接受前两个 token，但得到 {accepted}"
        print("✅ 测试通过！投机解码逻辑实现通过测试。")
        
    except NotImplementedError:
        print("请先完成 TODO 代码。")
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
        elif isinstance(e, AssertionError):
            print("代码可能未完成，导致了断言失败")
        elif isinstance(e, RuntimeError):
            print("代码可能未完成，导致了运行时错误")
        else:
            print("代码可能未完成，导致了断言失败")
        raise NotImplementedError("请先完成 TODO 代码！") from e
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise

test_speculative_decoding()

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
def speculative_verify(draft_probs, target_probs, draft_tokens):
    K = len(draft_tokens)
    accepted_tokens = []
    
    for i in range(K):
        token_id = draft_tokens[i]
        p = target_probs[i, token_id].item()
        q = draft_probs[i, token_id].item()
        
        # TODO 1: 目标概率不小于草拟概率时，直接接受
        if p >= q:
            accepted_tokens.append(token_id)
        else:
            # TODO 2: 按 p / q 的概率决定是否接受
            r = torch.rand(1).item()
            if r < p / q:
                accepted_tokens.append(token_id)
            else:
                # 拒绝该 token，停止验证后续猜测
                break
                
    return accepted_tokens


```

### 解析

**1. TODO 1（目标概率不小于草拟概率时，直接接受）**
- 对每个草拟 token，先读取大模型概率 $p(x)$ 和小模型概率 $q(x)$。
- 当 $p(x) \ge q(x)$ 时，接受概率 $\alpha(x)$ 直接变成 1。
- 这意味着目标模型已经认为这个 token 足够合理，不需要再额外掷硬币。

**2. TODO 2（按 $p/q$ 的概率决定是否接受）**
- 当 $p(x) < q(x)$ 时，接受概率退化为 $\alpha(x) = p(x)/q(x)$。
- 这一步本质上是在校正小模型过于激进的草拟结果。
- 如果硬币没过，就必须立刻停止当前草稿链路。

**3. 进阶思考**
- 草拟模型的目标不是“替代”大模型，而是“提速候选生成”。
- 终止条件之所以重要，是因为后续草稿 token 都建立在前缀被接受的前提上。
- 这也是为什么 Speculative Decoding 能在不改变输出分布的前提下减少大模型调用次数。
