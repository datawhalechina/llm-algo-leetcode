# 24. SGLang RadixAttention | SGLang 基数注意力
**难度：** Hard | **环境：** CPU-first | **标签：** `推理优化`, `KV Cache`, `RadixAttention` | **目标人群：** 推理优化学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/24_SGLang_RadixAttention.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

真实推理服务里，请求往往不是彼此独立的：多轮对话会反复带上历史上下文，Agent 和工具调用也会共享很长的 system prompt。问题在于，如果每个请求都重新计算这些公共前缀，KV Cache 会被重复占用，首 token 延迟也会被拖高。

RadixAttention 要解决的就是前缀复用问题：把已经算过的 prompt 组织成一棵可检索的前缀树，新请求到来时先找最长公共前缀，命中的部分直接复用，没命中的后缀再交给模型计算。本节用带边压缩和分裂的简化 Python 结构模拟这件事，重点看清“共享节点”和“缓存命中长度”如何转化为少算多少 token。

**关键词：** `RadixAttention`, `prefix tree`, `multi-turn`

---
## 前置阅读

**导语：** 先理解 KV Cache 的存储和请求间的公共前缀，再观察 RadixAttention 如何组织、命中和复用这些前缀。
- [22. vLLM PagedAttention | vLLM 分页注意力](./22_vLLM_PagedAttention.md)
- [21. Decoding Strategies | 解码策略](./21_Decoding_Strategies.md)
- [P1: 11. KV Cache and Memory Growth | KV Cache 与显存增长](../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md)

---

### Step 1: RadixAttention 如何组织公共前缀复用

重复出现的 System Prompt 或历史上下文会让请求反复执行相同的 prefill。RadixAttention 的输入是已登记的 token 前缀和新请求 prompt，处理方式是沿 Radix Tree 查找最长公共前缀；输出是可复用的命中范围和需要继续计算的未命中后缀。
先通过表格区分公共前缀、树路径、命中范围和未命中后缀，再观察主图中的查找与复用过程。本节先建立“前缀登记 → 最长匹配 → 命中复用 / 后缀计算”的整体视野；实际 KV Tensor 的存储、引用计数和释放由运行时缓存管理继续承接。

| 概念 | 它记录什么 | 对请求执行的作用 |
|---|---|---|
| 公共前缀 | 多个请求开头相同的 token 序列 | 作为可复用的候选缓存范围 |
| Radix Tree | 按 token 路径组织共享边，并支持最长前缀匹配 | 快速找到当前请求可以复用的范围 |
| 未命中后缀 | 从首个不一致 token 开始的剩余序列 | 继续执行 prefill，并登记为新分支 |
| PagedAttention 对照 | 按物理 Block 管理 KV Cache；RadixAttention 按 token 前缀组织共享路径 | 前者解决分页分配，后者强调前缀查找与复用 |

![RadixAttention 前缀树图](../public/02_PyTorch_Algorithms/24_radix_attention_tree.svg)

### Step 2: Radix Tree 的插入、分裂与匹配

用两条有公共前缀的请求观察路径如何插入、分裂和匹配：先沿共享边查找最长公共前缀，再把未命中的 token 留给后续 prefill。表格把每个操作与树的变化对应起来，帮助你从请求路径理解数据结构。

| 操作 | 输入 | 树的变化 | 要观察的结果 |
|---|---|---|---|
| 插入 | 新 token 路径 | 新建边，或在首个不一致位置分裂已有边 | 公共前缀只保留一条共享路径 |
| 匹配 | 新请求 prompt | 沿边逐段比较 token | 返回最长命中长度 `hit_len` |
| 拆分 | `hit_len` 与 prompt | 切出命中前缀和未命中 suffix | suffix 进入后续 prefill |


### Step 3: 命中范围如何进入执行计划

树匹配返回 token 范围，执行器还要把它转换成“复用多少、重新计算多少”的执行计划。先用公式说明命中长度，再用表格把命中前缀、未命中后缀和缓存句柄对应到下一步动作。最长公共前缀为：
$$H = \max_j \operatorname{lcp}(prompt\_tokens, cached\_path_j)$$
| 字段 | 含义 | 下一步 |
|---|---|---|
| `hit_len` | 从 prompt 开始连续命中的 token 数 | 读取对应前缀缓存 |
| `hit_prefix` | 可复用的 token 前缀 | 跳过重复 prefill |
| `miss_suffix` | 命中位置之后的 token | 执行剩余 prefill，并登记新路径 |
| `terminal` / `kv_cache_ptr` | 标记完整缓存路径，并关联缓存句柄 | 决定命中结果是否有可复用的缓存对象 |
| 证据范围 | `hit_len` 是索引层命中指标，不等于真实显存节省量 | 真实收益需结合 backend 与 workload 测量 |

![Radix Tree 命中到执行计划](../public/02_PyTorch_Algorithms/24_radix_match_flow.svg)


### Step 4: 实现并验证前缀缓存索引

代码接收已登记的 token 路径和新请求 prompt，完成插入、`match_prefix`、`split_prompt`，输出共享边、`hit_prefix` 与 `miss_suffix`。开始实现前，先明确每个函数要维护的状态，以及它如何影响命中结果和后续执行计划。
实现时注意三点：`insert` 负责维护共享树结构，`match_prefix` 负责从 prompt 开头寻找完整命中，`split_prompt` 负责把命中部分与待计算部分交给执行器。部分重叠边必须分裂，未命中或空 prompt 不能误报命中。
| 实现部分 | 需要完成的内容 | 验证重点 |
|---|---|---|
| 公共前缀计算 | `_lcp_len` 为边分裂和路径匹配提供公共边长度 | 遇到首个不一致 token 立即停止 |
| 索引操作 | `insert`、`match_prefix`、`split_prompt` | 公共边共享，部分重叠时正确分裂 |
| 执行计划 | 根据 `hit_len` 拆出 `hit_prefix` 与 `miss_suffix` | 命中部分复用，后缀继续 prefill |
| 退化路径 | 处理未命中或空 prompt | 能回退到完整重算，不误报命中 |

```python
import torch
```


```python
class TreeNode:
    """表示一条压缩边及其子路径；`terminal` 表示完整路径在此结束。"""
    def __init__(self, key_tokens, terminal=False):
        self.key_tokens = list(key_tokens)  # 这条边上的 Token 序列
        self.children = []            # 子节点列表
        self.kv_cache_ptr = None      # 模拟指向物理 KV Cache 的指针
        self.terminal = terminal      # 是否有完整请求在此结束

class SimpleRadixCache:
    """用压缩 Radix Tree 模拟公共 token 前缀的登记与匹配。"""
    def __init__(self):
        # 根节点是空的
        self.root = TreeNode([])
        
    def insert(self, tokens):
        """插入路径；遇到部分重叠边时分裂旧边。"""
        tokens = list(tokens)
        if not tokens:
            raise ValueError('tokens 不能为空')
        node = self.root
        offset = 0
        while offset < len(tokens):
            # TODO 1: 找到首 token 相同的 child，并计算公共边长度。
            # child = ???；common = ???
            # 若没有同首 token 的 child，新增叶节点时要标记 terminal=True，
            # 因为这条完整 token 路径已经登记，可以作为后续复用候选。
            if child is None:
                node.children.append(TreeNode(tokens[offset:], terminal=True))
                return
            if common == len(child.key_tokens):
                node, offset = child, offset + common
                continue
            # TODO 2: 用 shared 边替换 child，并挂接旧后缀和新后缀。
            # shared = ???
            # old_suffix = ???
            # child.key_tokens = ???
            # shared.children = ???；同时保留旧 child 的 terminal、子节点和缓存指针。
            # node.children[...] = shared
            return
        
    def _lcp_len(self, cached_tokens, prompt_tokens):
        """计算两段 token 序列的最长公共前缀长度。"""
        match_len = 0
        # ==========================================
        # TODO 3: 逐个 token 计算最长公共前缀长度
        # 提示: 遇到不相等时立刻停止
        # ==========================================
        # i = ???
        # match_len = ???
        return match_len
        
    def match_prefix(self, prompt_tokens):
        """
        在现有树中，为新的 prompt_tokens 寻找最长的匹配前缀。
        如果前 N 个 token 完全一致，说明这 N 个 token 的 KV Cache 可以直接复用！
        """
        best_match_len = 0
        node = self.root
        offset = 0
        while offset < len(prompt_tokens):
            # TODO 4: 沿共享边查找并更新已完成缓存路径的长度
            # child = ???
            # common = ???
            if child is None or common < len(child.key_tokens):
                break
            offset += common
            node = child
            # TODO 5: 只有终止节点才算完整可复用前缀。
            # 逐条边完整命中后，才根据 node.terminal 更新 best_match_len = ???
        return best_match_len

    def split_prompt(self, prompt_tokens):
        """把 prompt 拆成可复用前缀和需要重算的后缀。

        `hit_prefix` 必须来自 prompt 的开头，`miss_suffix` 从首个未命中 token 开始。"""
        # ==========================================
        # TODO 6: 先找命中长度，再拆出前缀和后缀
        # 提示: hit_len 是可复用的前缀长度
        # ==========================================
        # hit_len = ???
        # hit_prefix = ???
        # miss_suffix = ???
        return hit_prefix, miss_suffix, hit_len
```


```python
# 测试你的实现
def test_radix_attention():
    try:
        cache = SimpleRadixCache()
        cache.insert([0, 1, 2, 3])
        cache.insert([0, 1, 2, 3, 4])
        cache.insert([9, 9, 9])

        # 1. 基础 LCP 检查
        assert cache._lcp_len([1, 2, 3], [1, 2, 4]) == 2, "LCP 计算失败！"
        assert cache._lcp_len([7, 8], [7, 8, 9, 10]) == 2, "完整前缀匹配失败！"
        print("✅ 最长公共前缀计算正确！")

        # 2. 多候选路径下，应该选择最长命中前缀
        match_len = cache.match_prefix([0, 1, 2, 3, 4, 5])
        assert match_len == 5, "匹配失败！应该命中最长的 5 个 token 前缀。"
        assert cache.match_prefix([9, 9, 9]) == 3, "新建完整路径必须标记为 terminal！"
        assert cache.match_prefix([7, 6, 5]) == 0, "错误匹配！不该匹配到任何东西。"
        assert len(cache.root.children) == 2, "公共前缀没有被合并为共享边"
        shared = next(child for child in cache.root.children if child.key_tokens[0] == 0)
        assert shared.key_tokens == [0, 1, 2, 3], "共享边的 token 序列不正确"
        assert shared.terminal is True, "已登记的完整路径应标记为 terminal"
        assert len(shared.children) == 1 and shared.children[0].key_tokens == [4], "新增后缀没有挂到共享边下"
        assert shared.children[0].terminal is True, "新增完整路径的后缀节点应标记为 terminal"
        print("✅ 多路径前缀命中选择正确！")

        # 3. 前缀拆分验证
        hit_prefix, miss_suffix, hit_len = cache.split_prompt([0, 1, 2, 3, 4, 5])
        assert hit_len == 5, "Hit Length 计算错误！"
        assert hit_prefix == [0, 1, 2, 3, 4], "可复用前缀拆分错误！"
        assert miss_suffix == [5], "待重算后缀拆分错误！"

        hit_prefix2, miss_suffix2, hit_len2 = cache.split_prompt([7, 6, 5])
        assert hit_len2 == 0, "无命中时 Hit Length 应为 0！"
        assert hit_prefix2 == [], "无命中时前缀应为空！"
        assert miss_suffix2 == [7, 6, 5], "无命中时后缀应保持原样！"
        assert cache.split_prompt([0, 1, 2, 3, 4]) == ([0, 1, 2, 3, 4], [], 5), "完整命中时 suffix 应为空"
        assert cache.match_prefix([]) == 0, "空 prompt 不应产生命中"
        assert cache.split_prompt([]) == ([], [], 0), "空 prompt 的拆分结果不正确"
        try:
            cache.insert([])
        except ValueError:
            pass
        else:
            raise AssertionError('空 token 路径不能登记')
        # 反向分裂：新路径是旧路径的前缀，shared 节点仍需成为 terminal
        reverse_cache = SimpleRadixCache()
        reverse_cache.insert([4, 5, 6, 7])
        reverse_cache.insert([4, 5])
        assert reverse_cache.match_prefix([4, 5, 8]) == 2, "反向边分裂未保留 terminal"
        print("✅ 前缀拆分与回退逻辑正确！")

        print("\n 所有测试通过：共享边、最长命中和 prompt 拆分逻辑正确。")

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
        elif isinstance(e, AssertionError):
            print("代码可能未完成，导致了断言失败")
        elif isinstance(e, RuntimeError):
            print("代码可能未完成，导致了运行时错误")
        else:
            print("代码可能未完成，导致了断言失败")
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e
    except Exception as e:
        print(f"❌ 发生未知异常: {e}")
        raise


test_radix_attention()

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
class TreeNode:
    """表示一条压缩边及其子路径；`terminal` 表示完整路径在此结束。"""
    def __init__(self, key_tokens, terminal=False):
        self.key_tokens = list(key_tokens)
        self.children = []
        self.kv_cache_ptr = None
        self.terminal = terminal

class SimpleRadixCache:
    """用压缩 Radix Tree 模拟公共 token 前缀的登记与匹配。"""
    def __init__(self):
        self.root = TreeNode([])
        
    def _find_child(self, node, token):
        """返回首 token 匹配的子边；同一节点下不应出现重复首 token。"""
        return next((child for child in node.children if child.key_tokens and child.key_tokens[0] == token), None)

    def insert(self, tokens):
        """插入路径；部分重叠时分裂边，公共边由多个请求共享。"""
        # TODO 1：参考实现——按首 token 找候选子边，并计算公共边长度。
        tokens = list(tokens)
        if not tokens:
            raise ValueError('tokens 不能为空')
        node, offset = self.root, 0
        while offset < len(tokens):
            child = self._find_child(node, tokens[offset])
            if child is None:
                node.children.append(TreeNode(tokens[offset:], terminal=True))
                return
            common = self._lcp_len(child.key_tokens, tokens[offset:])
            if common == len(child.key_tokens):
                node, offset = child, offset + common
                continue
            # TODO 2：参考实现——保留旧后缀及其 terminal、子节点和缓存指针。
            shared = TreeNode(child.key_tokens[:common])
            old_suffix = TreeNode(child.key_tokens[common:], terminal=child.terminal)
            old_suffix.children = child.children
            old_suffix.kv_cache_ptr = child.kv_cache_ptr
            shared.children.append(old_suffix)
            child_index = node.children.index(child)
            node.children[child_index] = shared
            if offset + common == len(tokens):
                shared.terminal = True
            else:
                shared.children.append(TreeNode(tokens[offset + common:], terminal=True))
            return
        
    def _lcp_len(self, cached_tokens, prompt_tokens):
        # TODO 3: 逐个 token 计算最长公共前缀长度
        match_len = 0
        while match_len < len(cached_tokens) and match_len < len(prompt_tokens):
            if cached_tokens[match_len] == prompt_tokens[match_len]:
                match_len += 1
            else:
                break
        return match_len
        
    def match_prefix(self, prompt_tokens):
        """
        在现有树中，为新的 prompt_tokens 寻找最长的匹配前缀。
        """
        # TODO 4：参考实现——只沿 prompt 开头的完整共享边向下匹配。
        prompt_tokens = list(prompt_tokens)
        node, offset, best_match_len = self.root, 0, 0
        while offset < len(prompt_tokens):
            child = self._find_child(node, prompt_tokens[offset])
            if child is None:
                break
            match_len = self._lcp_len(child.key_tokens, prompt_tokens[offset:])
            if match_len < len(child.key_tokens):
                break
            offset += match_len
            node = child
            # TODO 5：参考实现——只有 terminal 节点才更新可复用命中长度。
            if node.terminal:
                best_match_len = offset
        return best_match_len

    def split_prompt(self, prompt_tokens):
        """把 prompt 拆成可复用前缀和需要重算的后缀。"""
        # TODO 6: 先找命中长度，再拆出前缀和后缀
        hit_len = self.match_prefix(prompt_tokens)
        hit_prefix = prompt_tokens[:hit_len]
        miss_suffix = prompt_tokens[hit_len:]
        return hit_prefix, miss_suffix, hit_len
```

### 解析

**1. TODO 3（逐个 token 计算最长公共前缀长度）**
- `match_len` 的本质就是两个 token 序列的最长公共前缀长度。
- 用 `while` 逐位比较，遇到不相等就立即停止。
- 两个边界都要检查：缓存路径是否结束、当前 prompt 是否结束。

**2. 插入与边分裂**
- 每条边保存一段 token；新路径只要与旧边部分重叠，就把旧边拆成 shared、old suffix 和 new suffix。
- 这样 `[0, 1, 2, 3]` 与 `[0, 1, 2, 3, 4]` 会共享前一条边，而不是各自挂在根节点下。

**3. TODO 4/5（沿共享路径匹配）**
- `match_prefix` 沿首 token 对应的 child 向下查找，遇到不完整边或不存在的 child 就停止。
- 只有已经登记完成的 terminal 节点才计入可复用长度。

**4. TODO 6（拆出可复用前缀与待重算后缀）**
- 先调用 `match_prefix` 得到 `hit_len`。
- 再把 `prompt_tokens` 切成 `hit_prefix` 和 `miss_suffix`。
- 这一步把“命中长度”真正变成“复用前缀 + 新增后缀”的工程操作。

**5. 进阶思考**
- 多轮对话和系统提示词通常有很长的公共前缀。
- Radix Tree 可以把这些公共前缀组织成共享路径；真实 KV Tensor 是否共享还取决于 backend 的 block 和生命周期管理。
- 本节不实现 LRU、引用计数、物理 block 分配和跨 worker KV 传输。
- CPU 测试只证明结构和 token 命中逻辑，不能推出首 token 延迟或吞吐提升。

### Step 5: 可选 GPU 边界探针

Radix Tree 的匹配本身是 CPU 侧索引操作。本实验只验证命中前缀和未命中 suffix 能否被整理成 GPU 输入张量，不测 SGLang 的树调度、KV Cache 复用或吞吐收益。

```python
# GPU 可选实验配置：该实验只验证数据边界，不启动 SGLang。
RUN_MODE = 'dry_run'  # dry_run / real_gpu
RADIX_GPU_PROMPTS = {
    'cached_prefix': [101, 102, 103, 104],
    'request': [101, 102, 103, 104, 201, 202],
}

def run_radix_gpu_boundary_probe(run_mode='dry_run', prompts=None):
    """把 Radix Tree 的 token 命中结果转换为 GPU 输入边界。"""
    import json
    import torch
    prompts = prompts or RADIX_GPU_PROMPTS
    cache = SimpleRadixCache()
    cache.insert(prompts['cached_prefix'])
    hit_prefix, miss_suffix, hit_len = cache.split_prompt(prompts['request'])
    plan = {
        'hit_len': hit_len,
        'hit_tokens': len(hit_prefix),
        'suffix_tokens': len(miss_suffix),
        'evidence_level': 'synthetic_gpu_input_boundary',
    }
    if run_mode == 'dry_run':
        print(json.dumps({'mode': run_mode, 'plan': plan}, ensure_ascii=False, indent=2))
        return plan
    if run_mode != 'real_gpu':
        raise ValueError("RUN_MODE 只能是 dry_run 或 real_gpu")
    if not torch.cuda.is_available():
        raise RuntimeError('real_gpu 模式需要 CUDA')
    device = torch.device('cuda')
    hit_tensor = torch.tensor(hit_prefix, dtype=torch.long, device=device)
    suffix_tensor = torch.tensor(miss_suffix, dtype=torch.long, device=device)
    torch.cuda.synchronize(device)
    result = {**plan, 'device': torch.cuda.get_device_name(0), 'hit_tensor_shape': list(hit_tensor.shape), 'suffix_tensor_shape': list(suffix_tensor.shape)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result

run_radix_gpu_boundary_probe(RUN_MODE, RADIX_GPU_PROMPTS)
```

## 相关阅读

RadixAttention 可以继续从 SGLang 实现、前缀缓存和调度策略三个方向阅读。

- [SGLang 原论文：Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)
- [SGLang 官方仓库](https://github.com/sgl-project/sglang)
- [SGLang 官方文档](https://docs.sglang.ai/)
- [22. vLLM PagedAttention | vLLM 分页注意力](./22_vLLM_PagedAttention.md)
- [34. Prefix Cache Matching and Reuse | Prefix Cache 匹配与复用](./34_Prefix_Cache_Matching_and_Reuse.md)
- [37. KV Cache Scheduling | KV Cache 调度](./37_KV_Cache_Scheduling.md)
- [69. Prefix Caching Benchmark | 前缀缓存基准项目](./69_Prefix_Caching_Benchmark.md)
