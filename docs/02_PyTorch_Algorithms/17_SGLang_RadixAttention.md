# 17 SGLang RadixAttention

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/17_SGLang_RadixAttention.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 17. SGLang 与 RadixAttention: 突破 vLLM 多轮对话瓶颈

**难度：** Medium | **标签：** `SGLang`, `Radix Tree`, `KV Cache` | **目标人群：** 模型部署与推理引擎开发

上一节我们学习了业界广泛使用的 **vLLM (PagedAttention)**。它完美解决了单次生成长文本时的**内存碎片**问题。
但在实际的生产环境中，我们经常遇到以下场景：
- **庞大且共享的 System Prompt**（如几百字的设定语，每个用户请求都带着）。
- **多轮对话（Multi-turn Chat）**（只增加了最后一句用户提问，前面几万字的聊天记录都是相同的）。
- **Few-shot Prompting**（给所有请求都塞入相同的 few-shot 示例）。

vLLM 会对每一个请求**从头开始（重新计算）**这些共享前缀的 KV Cache，这浪费了大量的时间（导致 TTFT 首字响应极慢）和显存。

**SGLang (LMSYS, 2024)** 的提出，它的核心创新 **RadixAttention** 引入了**基数树 (Radix Tree)** 来管理系统的 KV Cache。当系统发现新的请求和旧请求有着相同的前缀（Prefix）时，它会**直接复用**树节点里的 KV Cache，彻底跳过了重复的 Prefill 阶段！

### Step 1: 核心机制对比

> **vLLM PagedAttention：线性的分页存储**
> vLLM 就像操作系统的虚拟内存。每个请求有一个页表，它能很好地解决碎片，但请求与请求之间的页表是隔离的。就算两个请求的 Prompt 一模一样，它们也会各占一份显存，各算一次。

> **SGLang RadixAttention：基于基数树的共享路由**
> 系统维护了一棵全局的树。树的每一条边代表一段 Token 序列，节点里存着这段序列对应的 KV Cache 物理块指针。
> 当新请求到来时，SGLang 会用它的 Prompt 去这棵树里做**最长前缀匹配 (Longest Prefix Match)**。
> 匹配到的部分直接拿来用，没匹配到的部分再去计算并作为新分支挂在树上。

### Step 2: 动手实战 —— 模拟 Radix Tree 前缀匹配

为了让你深刻理解 SGLang 的调度思想，我们将用 Python 原生数据结构，亲手模拟一个极其简化的 Radix Tree 路由管理器。

**要求**：完成 `match_prefix` 函数，在全局 KV 树中寻找当前请求的最长前缀，返回可以省去的重计算长度（Hit Length）。

```python
import torch

class TreeNode:
    def __init__(self, key_tokens):
        self.key_tokens = key_tokens  # 这条边上的 Token 序列 (如 [101, 532, 789])
        self.children = []            # 子节点列表
        self.kv_cache_ptr = None      # 模拟指向物理 KV Cache 的指针

class SimpleRadixCache:
    def __init__(self):
        # 根节点是空的
        self.root = TreeNode([])
        
    def insert(self, tokens):
        """简单模拟向基数树中插入完整的请求。"""
        # 简化的插入逻辑（不涉及分裂裂变，仅为演示层级添加）
        # 假设当前系统只有一个 System Prompt，我们将其挂在根节点下
        node = TreeNode(tokens)
        self.root.children.append(node)
        
    def match_prefix(self, prompt_tokens):
        """
        在现有树中，为新的 prompt_tokens 寻找最长的匹配前缀。
        如果前 N 个 token 完全一致，说明这 N 个 token 的 KV Cache 可以直接复用！
        """
        # 为了教学，我们只做单层子节点的暴力匹配
        best_match_len = 0
        
        # ==========================================
        # TODO: 遍历 self.root.children，寻找能与 prompt_tokens 匹配的最长前缀
        # 如果子节点的 token 序列是 [A, B, C]，而 prompt 是 [A, B, C, D, E]
        # 那么最佳匹配长度就是 3。
        # ==========================================
        for child in self.root.children:
            cached_tokens = child.key_tokens
            match_len = 0
            
            # 提示：用 while 或 for 循环对比 cached_tokens 和 prompt_tokens
            # match_len = ???
            
            if match_len > best_match_len:
                best_match_len = match_len
                
        return best_match_len
```

```python
# 测试你的实现
def test_radix_attention():
    try:
        cache = SimpleRadixCache()
        
        # 1. 模拟系统初始化：所有的聊天都带有一段长达 100 词的“系统人设”
        system_prompt = list(range(100)) # 用 0~99 的数字模拟 Token
        cache.insert(system_prompt)
        
        # 2. 用户 A 来了：带有系统人设，并问了一句话
        user_a_prompt = list(range(100)) + [1001, 1002, 1003]
        
        match_len_a = cache.match_prefix(user_a_prompt)
        assert match_len_a == 100, "匹配失败！系统人设的 100 个 Token 应该完全被复用！"
        print(f"✅ 用户 A 命中前缀缓存！原本需要计算 {len(user_a_prompt)} 个 token，现在只需计算最后 {len(user_a_prompt) - match_len_a} 个！")
        
        # 3. 用户 B 来了：一个完全不同的、没有系统人设的请求
        user_b_prompt = [9999, 8888, 7777]
        match_len_b = cache.match_prefix(user_b_prompt)
        assert match_len_b == 0, "错误匹配！不该匹配到任何东西。"
        print("✅ 用户 B 正常 fallback，未命中缓存。")
        
        print("\n🎉 所有测试通过！这正是 SGLang 让大模型推理首字响应（TTFT）飞升 10 倍的底层秘密！")
        
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
    except Exception as e:
        print(f"❌ 发生未知异常: {e}")

test_radix_attention()

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

::: details 💡 点击查看官方解析与参考代码

SGLang 通过基数树 (Radix Tree) 突破了多轮对话和长系统提示词的计算瓶颈。当新的请求到来时，调度器会扫描内存中的基数树，寻找最长匹配的前缀。被匹配到的 Token 将直接复用物理显存中的 KV Cache，完美避开了二次前向传播，极大降低了首字响应时间（TTFT）。

```python
class TreeNode:
    def __init__(self, key_tokens):
        self.key_tokens = key_tokens
        self.children = []
        self.kv_cache_ptr = None

class SimpleRadixCache:
    def __init__(self):
        self.root = TreeNode([])
        
    def insert(self, tokens):
        node = TreeNode(tokens)
        self.root.children.append(node)
        
    def match_prefix(self, prompt_tokens):
        best_match_len = 0
        
        for child in self.root.children:
            cached_tokens = child.key_tokens
            match_len = 0
            
            # ==========================================
            # 💡 参考实现：逐个对比 Token，计算最长连续相同前缀
            # ==========================================
            while match_len < len(cached_tokens) and match_len < len(prompt_tokens):
                if cached_tokens[match_len] == prompt_tokens[match_len]:
                    match_len += 1
                else:
                    break
            
            if match_len > best_match_len:
                best_match_len = match_len
                
        return best_match_len

```

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
