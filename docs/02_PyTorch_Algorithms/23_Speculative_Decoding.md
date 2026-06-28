# 23. Speculative Decoding | 投机解码：草稿模型与验证模型协作

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/23_Speculative_Decoding.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 前置

**导语：** 先看 PagedAttention，再看投机解码会更容易理解草稿模型与验证模型的协作。
- [Part 2: 21 vLLM PagedAttention](./22_vLLM_PagedAttention.md)
- [Part 2: 20 Decoding Strategies](./21_Decoding_Strategies.md)

## 相关阅读

**导语：** 投机解码之后，可以继续看 RadixAttention 和量化。
- [Part 2: 23 SGLang RadixAttention](./24_SGLang_RadixAttention.md)
- [Part 2: 24 Quantization W8A16](./25_Quantization_W8A16.md)


### 动手实战：核心的接受/拒绝逻辑

面试中最常考的，就是如何对比草拟概率 $q(x)$ 和目标概率 $p(x)$ 来决定是否接受该 Token。
**算法法则**：
- 如果目标概率大于草拟概率 ($p \ge q$)，**100% 接受**。
- 如果目标概率小于草拟概率 ($p < q$)，以 $p/q$ 的概率接受它（丢硬币）。


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
        # ==========================================
        # if ???:
        #     accepted_tokens.append(token_id)
        # ==========================================
        # TODO 2: 以 p / q 的概率接受
        # 如果拒绝，立即中止整个验证循环！因为一步错步步错。
        # ==========================================
        # else:
        #     if ???:
        #         accepted_tokens.append(token_id)
        #     else:
        #         break
    
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
    except Exception as e:
        print(f"❌ 测试失败: {e}")

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
        
        if p >= q:
            accepted_tokens.append(token_id)
        else:
            r = torch.rand(1).item()
            if r < p / q:
                accepted_tokens.append(token_id)
            else:
                # 拒绝该 token，停止验证后续猜测
                break
                
    return accepted_tokens


```

### 解析

投机解码（Speculative Decoding）通过小模型草拟和大模型并行验证，将原本由于 Memory Bound 导致的计算等待时间转化为并发算力。验证时采用 $p/q$ 的接受概率，在数学上准确保证了即使经过了小模型的瞎猜，最终采样的 Token 分布依然和只用大模型自回归生成的分布严格一致，实现了“无损加速”。
