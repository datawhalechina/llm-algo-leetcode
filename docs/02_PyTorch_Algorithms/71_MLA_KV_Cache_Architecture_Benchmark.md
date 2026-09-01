# 71. MLA KV Cache Architecture Benchmark | MLA 与 KV Cache 结构基准

**难度：** Hard | **环境：** CPU-first；GPU/backend 可选 | **标签：** `推理优化`, `MLA`, `KV Cache`

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/71_MLA_KV_Cache_Architecture_Benchmark.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

本节研究 DeepSeek 风格的 Multi-head Latent Attention（MLA）如何改变 KV Cache 的表示方式。先用 CPU 根据模型配置计算 MHA、GQA 与 MLA 的缓存账本，再把同一 workload 交给支持 MLA 的 backend 做可选验证。

MLA 不是 Prefix Cache，也不是普通量化：它改变模型内部保存的 KV 表示。真实模型候选为 `deepseek-ai/DeepSeek-V2-Lite`；如果 backend 或显存无法加载它，CPU 账本仍可完成，但不能把模拟结果写成真实速度或显存结论。

**主责与复用边界：** 71 负责 MLA 结构和 KV Cache 表示；显存优化复用缓存容量账本，74 负责 profiler trace，69 负责前缀复用，70 负责请求调度。本项目不把结构账本直接写成 backend 性能结论。
## 前置阅读

**导语：** 先理解 Attention 张量形状和 KV Cache 增长，再观察 MLA 如何改变缓存账本。
- [04. Attention / MHA / GQA](./04_Attention_MHA_GQA.md)
- [Part 01: 04. Attention Memory Optimization](../01_Hardware_Math_and_Systems/04_Attention_Memory_Optimization.md)
- [11. KV Cache and Memory Growth](../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md)
- [69. Prefix Caching Benchmark](./69_Prefix_Caching_Benchmark.md)
- [74. Profiling-Driven End-to-End Optimization](./74_Profiling_Driven_End_to_End_Optimization.md)

## 相关阅读
- [70. Serving Scheduler Benchmark](./70_Serving_Scheduler_Benchmark.md)
- [81. Distributed Inference Logic Validation](./81_Distributed_Inference_Project.md)
### Step 1：确定结构比较口径

固定 batch、序列长度、层数、dtype 和 token 数，只改变 KV Cache 表示。MHA 缓存每个 head 的 K/V，GQA 共享部分 KV head，MLA 缓存低维 latent 表示及必要的位置相关分量。

| 账本字段 | 常见 config 字段 | 用途 |
|---|---|---|
| `num_layers` | `num_hidden_layers` / `n_layer` | 缓存沿层数累加 |
| `num_attention_heads` | `num_attention_heads` / `n_head` | MHA/GQA 头数基准 |
| `num_kv_heads` | `num_key_value_heads` / `n_head_kv` | 普通 K/V 缓存头数 |
| `latent_dim` | `kv_lora_rank` 等字段 | 本节简化 MLA latent 维度 |
| `rope_dim` | `qk_rope_head_dim` 等字段 | 本节简化位置分量 |
| `dtype_bytes` | 通常不在模型 config 中 | 由实验 dtype 显式提供 |

真实 DeepSeek 配置中的字段名称、位置编码拆分和 cache layout 可能随模型版本变化；本节只把可对应的字段带入简化账本，未映射的字段必须保留为缺失，不能用默认值补齐。

### Step 2：建立 CPU KV Cache 账本

CPU 只计算元素数量和理论字节数，验证公式、比例和边界；它不能验证 DeepSeek 的实际 kernel、显存 allocator 或 decode 延迟。

### Step 3：比较结构代价

同时查看 KV Cache bytes、压缩比例和额外 latent / positional 分量。本节使用简化账本帮助理解变量关系，不把 `latent_dim + rope_dim` 当作 DeepSeek MLA 的完整实现，也不能用固定节省比例代替模型配置。

### Step 4：连接真实 backend

可选使用 `deepseek-ai/DeepSeek-V2-Lite` 和固定推理 workload；只有 backend 成功加载并提供显存、延迟或 trace 证据，才能形成 GPU 结论。

### 实验条件与证据边界

| 实验 | CPU 可验证 | GPU/backend 才能验证 |
|---|---|---|
| 结构账本 | 元素数、理论 bytes、比例 | 实际 KV Cache、kernel、allocator |
| MLA smoke | 配置读取和报告格式 | backend 支持、TTFT、TPOT、显存 |
| profiling | 不能生成 CUDA trace | kernel、访存、同步、端到端时间线 |

推荐固定 128、512、1024 token prompt，`max_tokens=64`、`temperature=0`、`top_p=1`；本节不需要训练数据集。
## 练习代码

请先完成 CPU KV Cache 账本，再运行测试。

```python
from typing import Dict

```


```python
# 6 个核心 TODO：配置提取、普通 KV 账本、MLA 账本、表示比较和对比表
# 目标：把 MHA/GQA/简化 MLA 的容量假设整理成可检查的理论账本；不实现 DeepSeek MLA kernel。
# 代码声明顺序不等于学习顺序：先完成配置提取和字段校验，再组装对比表。

def kv_cache_bytes_attention(batch_size: int, seq_len: int, num_layers: int, num_kv_heads: int, head_dim: int, dtype_bytes: int = 2) -> int:
    """计算普通 MHA/GQA KV Cache 理论字节数；不代表实际 allocator 峰值。

    K 和 V 各占一份缓存；所有维度和 dtype_bytes 都参与计算。
    """
    # TODO 0：校验所有维度为正数，使用上述变量计算 K 和 V 两份缓存。
    # kv_elements = ???；kv_bytes = ???。
    #       返回整数 bytes；不要混入模型权重、临时张量或 CUDA reserved memory。
    # 返回整数 bytes；不要混入模型权重、临时张量或 CUDA reserved memory。
    raise NotImplementedError('请先完成 TODO 代码！')

def summarize_mla_config(config: Dict[str, int]) -> Dict[str, object]:
    """整理 MHA/GQA/MLA 账本所需字段，并保留缺失字段。

    latent_dim、rope_dim 和 dtype_bytes 必须来自显式配置，不为缺失值猜默认值。
    """
    # TODO 1：读取 model、num_layers、num_attention_heads、num_kv_heads、
    # normalized = ???；missing_fields = ???。
    # latent_dim、rope_dim、head_dim、dtype_bytes；标记缺失字段。
    raise NotImplementedError('请先完成 TODO 代码！')

def build_cache_comparison_table(config: Dict[str, object]) -> list[Dict[str, object]]:
    """生成 toy MHA/GQA/MLA 理论 bytes 表，不代表真实 backend。

    表格至少保留 representation、bytes、evidence；evidence 应标记理论估算。
    """
    # TODO 5：先提取并检查配置，再分别计算 MHA、GQA 和本节简化 MLA；
    # mha_bytes = ???；gqa_bytes = ???；mla_bytes = ???；comparison_rows = ???。
    # 返回 representation、bytes、evidence 三列，缺字段时明确报错。
    #       compression_ratio 只能表示理论容量变化，不能表示质量或吞吐收益。
    raise NotImplementedError('请先完成 TODO 代码！')

def extract_attention_dimensions(config: Dict[str, object]) -> Dict[str, object]:
    """从模型配置提取账本字段；不假设缺失字段的默认值。

    返回 normalized 字段和 missing_fields；字段别名只用于兼容命名。
    """
    # TODO 4：兼容 num_hidden_layers / n_layer、num_attention_heads /
    # normalized = ???；missing_fields = ???；head_dim = ???。
    # n_head 等常见别名；hidden_size 可与 attention heads 推出 head_dim，
    # 但 dtype_bytes 必须由实验显式提供。返回 normalized 字段和 missing_fields。
    raise NotImplementedError('请先完成 TODO 代码！')

def compare_kv_representations(baseline_bytes: int, candidate_bytes: int) -> Dict[str, float]:
    """比较两种 KV 表示的理论容量；不推断质量或真实吞吐。

    compression_ratio 仅在 candidate_bytes > 0 时有定义；saving_ratio 是容量比例变化。
    """
    # TODO 2：计算 bytes_delta、compression_ratio、saving_ratio；
    # bytes_delta = ???；compression_ratio = ???；saving_ratio = ???。
    # baseline_bytes > 0，candidate_bytes >= 0。
    raise NotImplementedError('请先完成 TODO 代码！')

def mla_cache_bytes(batch_size: int, seq_len: int, num_layers: int, latent_dim: int, rope_dim: int, dtype_bytes: int = 2) -> int:
    """估算本节简化 MLA latent 与位置相关缓存的理论字节数。

    latent_dim 和 rope_dim 是账本模型变量，不等于任何特定 DeepSeek 版本的 cache layout。
    """
    # TODO 3：校验维度，按本节的简化模型计算 latent 和 positional 两部分 bytes；
    # latent_bytes = ???；positional_bytes = ???；total_bytes = ???。
    # 注意：这不是 DeepSeek MLA 的完整 kernel 或 cache layout，也不能推出真实吞吐。
    raise NotImplementedError('请先完成 TODO 代码！')

```


```python
def test_mla_kv_cache_template():
    mha = kv_cache_bytes_attention(1, 1024, 2, 32, 128, 2)
    gqa = kv_cache_bytes_attention(1, 1024, 2, 8, 128, 2)
    assert mha == 33554432
    assert gqa == 8388608
    assert compare_kv_representations(mha, gqa)['saving_ratio'] == 0.75
    summary = summarize_mla_config({
        'model': 'toy-mla', 'num_layers': 2, 'num_attention_heads': 32,
        'num_kv_heads': 8, 'head_dim': 128, 'latent_dim': 512,
        'rope_dim': 64, 'dtype_bytes': 2,
    })
    assert summary['ready_for_estimate'] is True
    extracted = extract_attention_dimensions({
        'model_type': 'toy-mla', 'num_hidden_layers': 2,
        'num_attention_heads': 32, 'num_key_value_heads': 8,
        'hidden_size': 4096, 'kv_lora_rank': 512, 'qk_rope_head_dim': 64,
    })
    assert extracted['normalized']['num_layers'] == 2
    assert extracted['normalized']['num_kv_heads'] == 8
    assert extracted['normalized']['head_dim'] == 128
    table = build_cache_comparison_table({
        'model_type': 'toy-mla', 'num_hidden_layers': 2,
        'num_attention_heads': 32, 'num_key_value_heads': 8,
        'hidden_size': 4096, 'kv_lora_rank': 512, 'qk_rope_head_dim': 64,
        'dtype_bytes': 2,
    })
    assert [row['representation'] for row in table] == ['mha', 'gqa', 'mla']
    assert all(row['evidence'] == 'cpu_theoretical_ledger' for row in table)
    assert mla_cache_bytes(1, 1024, 2, 512, 64, 2) == 2359296
    for invalid in ({'batch_size': 0}, {'dtype_bytes': 0}):
        try: kv_cache_bytes_attention(1, 4, 1, 1, 8, 2, **invalid)
        except ValueError: pass
        else: raise AssertionError('非法 KV Cache 配置应明确拒绝！')
    print('测试通过：MLA KV Cache 账本模板可以工作。')

test_mla_kv_cache_template()

```

---
🛑 **STOP HERE** 🛑
请先完成 CPU 账本，再查看参考答案。
---
## 参考代码与解析

### 代码

```python
def kv_cache_bytes_attention(batch_size: int, seq_len: int, num_layers: int, num_kv_heads: int, head_dim: int, dtype_bytes: int = 2) -> int:
    """计算 K/V 两份缓存的理论字节数。"""
    values = (batch_size, seq_len, num_layers, num_kv_heads, head_dim, dtype_bytes)
    if any(not isinstance(value, int) or value <= 0 for value in values): raise ValueError('KV Cache 配置必须为正整数')
    return batch_size * seq_len * num_layers * num_kv_heads * head_dim * dtype_bytes * 2

def summarize_mla_config(config: Dict[str, int]) -> Dict[str, object]:
    """整理 MLA 账本字段，并保留缺失字段。"""
    required = ('model','num_layers','num_attention_heads','num_kv_heads','head_dim','latent_dim','rope_dim','dtype_bytes')
    missing = [key for key in required if key not in config]
    return {'model': config.get('model'), 'fields': {key: config.get(key) for key in required}, 'missing_fields': missing, 'ready_for_estimate': not missing}

def extract_attention_dimensions(config: Dict[str, object]) -> Dict[str, object]:
    """从 Hugging Face 风格配置提取账本字段，不猜测缺失值。"""
    aliases = {
        'model': ('model', 'model_type'),
        'num_layers': ('num_layers', 'num_hidden_layers', 'n_layer'),
        'num_attention_heads': ('num_attention_heads', 'n_head'),
        'num_kv_heads': ('num_kv_heads', 'num_key_value_heads', 'n_head_kv'),
        'head_dim': ('head_dim',),
        'latent_dim': ('latent_dim', 'kv_lora_rank'),
        'rope_dim': ('rope_dim', 'qk_rope_head_dim'),
        'dtype_bytes': ('dtype_bytes',),
    }
    normalized = {
        target: next((config[key] for key in keys if config.get(key) is not None), None)
        for target, keys in aliases.items()
    }
    if normalized['head_dim'] is None and config.get('hidden_size') is not None and normalized['num_attention_heads']:
        hidden_size = config['hidden_size']
        if not isinstance(hidden_size, int) or hidden_size <= 0 or hidden_size % normalized['num_attention_heads']:
            raise ValueError('hidden_size 必须是 attention head 数的正整数倍')
        normalized['head_dim'] = hidden_size // normalized['num_attention_heads']
    missing = [key for key, value in normalized.items() if value is None]
    return {'normalized': normalized, 'missing_fields': missing, 'ready_for_estimate': not missing}

def compare_kv_representations(baseline_bytes: int, candidate_bytes: int) -> Dict[str, float]:
    """计算理论容量差异；结果不等于 GPU 实测显存。"""
    if baseline_bytes <= 0 or candidate_bytes < 0: raise ValueError('baseline_bytes 必须 > 0，candidate_bytes 不能为负数')
    delta = baseline_bytes - candidate_bytes
    return {'bytes_delta': delta, 'compression_ratio': candidate_bytes / baseline_bytes, 'saving_ratio': delta / baseline_bytes}

def mla_cache_bytes(batch_size: int, seq_len: int, num_layers: int, latent_dim: int, rope_dim: int, dtype_bytes: int = 2) -> int:
    """估算 MLA latent 与位置相关缓存的理论字节数。"""
    values = (batch_size, seq_len, num_layers, latent_dim, rope_dim, dtype_bytes)
    if any(not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError('MLA 缓存配置必须为正整数')
    return batch_size * seq_len * num_layers * (latent_dim + rope_dim) * dtype_bytes

def build_cache_comparison_table(config: Dict[str, object]) -> list[Dict[str, object]]:
    """生成 toy MHA/GQA/MLA 理论 bytes 表，不代表真实 backend。"""
    extracted = extract_attention_dimensions(config)
    if not extracted['ready_for_estimate']:
        raise ValueError(f"账本字段不完整：{extracted['missing_fields']}")
    values = extracted['normalized']
    common = (1, 1024, values['num_layers'], values['head_dim'], values['dtype_bytes'])
    mha = kv_cache_bytes_attention(*common[:3], values['num_attention_heads'], *common[3:])
    gqa = kv_cache_bytes_attention(*common[:3], values['num_kv_heads'], *common[3:])
    mla = mla_cache_bytes(1, 1024, values['num_layers'], values['latent_dim'], values['rope_dim'], values['dtype_bytes'])
    return [
        {'representation': 'mha', 'bytes': mha, 'evidence': 'cpu_theoretical_ledger'},
        {'representation': 'gqa', 'bytes': gqa, 'evidence': 'cpu_theoretical_ledger'},
        {'representation': 'mla', 'bytes': mla, 'evidence': 'cpu_theoretical_ledger'},
    ]
print('CPU 账本完成：请将相同 workload 交给支持 MLA 的 backend，再进入 74 profiling。')

```


```python
def test_mla_kv_cache_template():
    mha = kv_cache_bytes_attention(1, 1024, 2, 32, 128, 2)
    gqa = kv_cache_bytes_attention(1, 1024, 2, 8, 128, 2)
    assert mha == 33554432
    assert gqa == 8388608
    assert compare_kv_representations(mha, gqa)['saving_ratio'] == 0.75
    extracted = extract_attention_dimensions({
        'model_type': 'toy-mla', 'num_hidden_layers': 2,
        'num_attention_heads': 32, 'num_key_value_heads': 8,
        'hidden_size': 4096, 'kv_lora_rank': 512, 'qk_rope_head_dim': 64,
    })
    assert extracted['normalized']['num_layers'] == 2
    assert extracted['normalized']['num_kv_heads'] == 8
    assert mla_cache_bytes(1, 1024, 2, 512, 64, 2) == 2359296
    table = build_cache_comparison_table({
        'model_type': 'toy-mla', 'num_hidden_layers': 2,
        'num_attention_heads': 32, 'num_key_value_heads': 8,
        'hidden_size': 4096, 'kv_lora_rank': 512, 'qk_rope_head_dim': 64,
        'dtype_bytes': 2,
    })
    assert [row['representation'] for row in table] == ['mha', 'gqa', 'mla']
    assert all(row['evidence'] == 'cpu_theoretical_ledger' for row in table)

test_mla_kv_cache_template()

```

### 解析

- **TODO 0**：普通 KV Cache 包含 K 和 V 两份张量；`num_kv_heads` 区分 MHA 与 GQA。MLA 不能用固定比例代替，必须根据 latent 与位置相关分量建模。
- **TODO 1**：配置汇总只保留事实字段，缺少 `latent_dim` 或 `rope_dim` 时不猜默认值。
- **TODO 2**：压缩比例是理论容量比较，不能推出质量、TTFT、TPOT 或 allocator 峰值。

真实模型建议使用 `deepseek-ai/DeepSeek-V2-Lite`；如果 backend 不支持 MLA，71 仍可完成 CPU 账本，但 74 不能伪造 CUDA trace。