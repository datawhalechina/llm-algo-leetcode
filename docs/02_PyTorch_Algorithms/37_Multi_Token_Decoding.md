# 37. Multi Token Decoding | 多 Token 解码
**难度：** Hard | **环境：** GPU required | **标签：** `解码`, `Multi-Token Decoding`, `推理优化` | **目标人群：** 推理系统与系统工程

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/37_Multi_Token_Decoding.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


先把投机解码和生成路径看清，再看多 token 解码如何减少逐 token 验证的开销。

**关键词：** `multi-token decoding`, `draft model`, `verification`

## 前置阅读

**导语：** 先看投机解码、解码策略和 PagedAttention，再看多 token 解码会更清楚。

- [23. Speculative Decoding | 投机解码](../02_PyTorch_Algorithms/23_Speculative_Decoding.md)
- [21. Decoding Strategies | 解码策略](../02_PyTorch_Algorithms/21_Decoding_Strategies.md)
- [22. vLLM PagedAttention | vLLM PagedAttention](../02_PyTorch_Algorithms/22_vLLM_PagedAttention.md)
- [P1: 11. KV Cache and Memory Growth | KV Cache 与显存增长](../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md)

## 相关阅读

**导语：** 多 token 解码之后，可以继续看前缀缓存和 RadixAttention。

- [36. Prefix Caching and Chunked Prefill | 前缀缓存与分块预填充](../02_PyTorch_Algorithms/36_Prefix_Caching_and_Chunked_Prefill.md)
- [24. SGLang RadixAttention | SGLang 基数注意力](../02_PyTorch_Algorithms/24_SGLang_RadixAttention.md)
- [P1: 17. CUDA Stream and Asynchrony | CUDA Stream 与异步执行](../01_Hardware_Math_and_Systems/17_CUDA_Stream_and_Asynchrony.md)

### Step 1: 原理与痛点

单 token 解码的瓶颈在于每次只推进一个 token，草稿模型与验证模型之间的往返很频繁。多 token 解码要解决的，就是如何让模型一次提出多个候选 token，再统一做验证或接受，从而减少频繁的逐 token 开销。

它和 `Speculative Decoding` 的关系可以这样理解：Speculative Decoding 更强调“先草拟，再逐个验证”，而 Multi-Token Decoding 更强调“在一个解码步里尽量推进多个 token”。两者都在减少 token-level 往返，但前者偏“草稿-验证协作”，后者偏“单步多 token 推进”。

从工程视角看，真正的目标不是“每次都吐更多 token”，而是在接受率可控的前提下，减少 decoder 反复进入的次数、减少 cache 更新的碎片化，并把 token 生成的节奏变得更平滑。

### Step 2: 代码实现框架

下面的代码会先模拟一个最小的多 token 提议器，再把验证、接受和回退拆成几个小动作。你需要关注的不是复杂采样，而是“先提议，再验证，再决定是否继续”的流程。

为了便于理解，可以把它拆成三层：

- **提议层**：一次给出若干候选 token；
- **验证层**：逐个检查候选 token 是否足够可靠；
- **回退层**：一旦中途拒绝，就停下来回到更保守的生成路径。

### Step 3: 核心机制

多 token 解码本质上还是围绕接受率和回退路径设计的：如果草稿 token 能被大模型连续接受，就能一次推进更多 token；如果中间被拒绝，就要立刻停止并回退到更保守的生成路径。

这里最关键的 trade-off 是：提议越激进，理论上单步推进越快，但回退概率也会更高；提议越保守，接受率更稳，但整体加速效果会下降。也就是说，多 token 解码的优化点不只是“生成更多”，而是“在接受率和推进步长之间找平衡”。

### Step 4: 动手实战

**要求**：请补全下方 `MultiTokenDecoderSim`，实现一个极简版的多 token 生成与验证模拟器。先把“提议 -> 验证 -> 接受 / 回退”这条链路跑通，再考虑更复杂的采样策略。

```python
from typing import List, Sequence, Tuple

import torch


class MultiTokenDecoderSim:
    """极简版多 token 生成与验证模拟器。"""

    def __init__(self, max_proposal_len: int = 4, min_accept_ratio: float = 0.5):
        if max_proposal_len <= 0:
            raise ValueError("max_proposal_len must be positive")
        if not (0.0 < min_accept_ratio <= 1.0):
            raise ValueError("min_accept_ratio must be in (0, 1]")
        self.max_proposal_len = max_proposal_len
        self.min_accept_ratio = min_accept_ratio
        self.history: List[dict] = []

    # TODO 1: 从草稿 token 中生成本轮提议
    def propose(self, draft_tokens: Sequence[int]) -> List[int]:
        raise NotImplementedError

    # TODO 2: 用一个简单规则模拟接受 / 拒绝
    def _accept_token(self, draft_prob: float, target_prob: float) -> bool:
        raise NotImplementedError

    # TODO 3: 逐个验证草稿 token，并返回拒绝位置
    def verify(
        self,
        draft_probs: torch.Tensor,
        target_probs: torch.Tensor,
        draft_tokens: Sequence[int],
    ) -> Tuple[List[int], int | None]:
        raise NotImplementedError

    # TODO 4: 组织一次完整的提议、验证、回退过程
    def decode(
        self,
        draft_probs: torch.Tensor,
        target_probs: torch.Tensor,
        draft_tokens: Sequence[int],
    ) -> dict:
        raise NotImplementedError
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
# TODO：下面是题目区的参考实现。

from typing import List, Sequence, Tuple

import torch


class MultiTokenDecoderSim:
    """极简版多 token 生成与验证模拟器。"""

    def __init__(self, max_proposal_len: int = 4, min_accept_ratio: float = 0.5):
        if max_proposal_len <= 0:
            raise ValueError("max_proposal_len must be positive")
        if not (0.0 < min_accept_ratio <= 1.0):
            raise ValueError("min_accept_ratio must be in (0, 1]")
        self.max_proposal_len = max_proposal_len
        self.min_accept_ratio = min_accept_ratio
        self.history: List[dict] = []

    def propose(self, draft_tokens: Sequence[int]) -> List[int]:
        """从草稿 token 中挑出本轮最多可提议的 token 序列。"""
        return list(draft_tokens)[: self.max_proposal_len]

    def _accept_token(self, draft_prob: float, target_prob: float) -> bool:
        """用一个简单的确定性规则模拟“提议-验证-接受”。"""
        if draft_prob <= 0:
            return target_prob > 0
        return target_prob >= draft_prob * self.min_accept_ratio

    def verify(
        self,
        draft_probs: torch.Tensor,
        target_probs: torch.Tensor,
        draft_tokens: Sequence[int],
    ) -> Tuple[List[int], int | None]:
        """逐个验证草稿 token，返回被接受的 token 和首次拒绝位置。"""
        proposed = self.propose(draft_tokens)
        accepted_tokens: List[int] = []
        rejected_at = None

        for i, token_id in enumerate(proposed):
            draft_prob = float(torch.as_tensor(draft_probs)[i, token_id])
            target_prob = float(torch.as_tensor(target_probs)[i, token_id])
            if self._accept_token(draft_prob, target_prob):
                accepted_tokens.append(token_id)
            else:
                rejected_at = i
                break

        return accepted_tokens, rejected_at

    def decode(
        self,
        draft_probs: torch.Tensor,
        target_probs: torch.Tensor,
        draft_tokens: Sequence[int],
    ) -> dict:
        """执行一次多 token 提议与验证，返回接受 / 回退信息。"""
        proposed = self.propose(draft_tokens)
        accepted_tokens, rejected_at = self.verify(draft_probs, target_probs, proposed)
        rejected_suffix = proposed[rejected_at:] if rejected_at is not None else []

        result = {
            "proposed_tokens": proposed,
            "accepted_tokens": accepted_tokens,
            "rejected_suffix": rejected_suffix,
            "accepted_len": len(accepted_tokens),
            "rejected_at": rejected_at,
        }
        self.history.append(result)
        return result

```

### 解析
- 多 token 解码的关键不在于一次吐多少 token，而在于“提议、验证、回退”是否稳定。
- 这个模拟器用一个接受率阈值来近似多 token 验证流程。
- 先把接受 / 拒绝的闭环跑通，再去想更复杂的采样和草稿策略。
### 测试

```python
import torch

sim = MultiTokenDecoderSim(max_proposal_len=2, min_accept_ratio=0.6)
draft_tokens = [10, 20, 30]
draft_probs = torch.zeros(3, 40)
target_probs = torch.zeros(3, 40)
for i, tok in enumerate(draft_tokens):
    draft_probs[i, tok] = 0.5
    target_probs[i, tok] = 0.8 if i < 2 else 0.2
accepted, rejected_at = sim.verify(draft_probs, target_probs, draft_tokens)
assert accepted == [10, 20]
assert rejected_at == 2 or rejected_at is None
result = sim.decode(draft_probs, target_probs, draft_tokens)
assert result['accepted_len'] == len(result['accepted_tokens'])
print('✅ MultiTokenDecoderSim 测试通过')
```
