# 71. MLA KV Cache Architecture Benchmark | MLA 与 KV Cache 结构基准

**难度：** Hard | **环境：** CPU-first；GPU/backend 可选 | **标签：** `推理优化`, `MLA`, `KV Cache`

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/71_MLA_KV_Cache_Architecture_Benchmark.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

本节在推理优化中作为 Task3 的架构扩展，在模型架构演进专题中作为 Task1 的专项项目。学习者先从 `ARCH-MLA` 已验证的 latent state、解耦 RoPE 与状态重建机制出发，再用固定 workload 比较 MHA、GQA 与 MLA 的缓存账本，并把候选交给支持 MLA 的 backend 验证。

MLA 不是 Prefix Cache，也不是 PagedAttention 或普通量化：它改变模型内部保存的 KV 表示。真实模型候选为 `deepseek-ai/DeepSeek-V2-Lite`；如果 backend 或显存无法加载它，CPU 账本仍可完成，但不能把模拟结果写成真实速度或显存结论。

项目结果同时记录结构字段、状态字节、backend 支持、失败原因与证据等级。只有真实 backend 成功加载并返回显存和延迟记录时，才进入部署判断。
## 前置阅读

**导语：** 先理解 Attention 张量形状和 KV Cache 增长，再观察 MLA 如何改变缓存账本。
- [04. Attention / MHA / GQA](./04_Attention_MHA_GQA.md)
- [ARCH-MLA. Multi-head Latent Attention | 多头潜在注意力](./ARCH-MLA_Multihead_Latent_Attention.md)
- [Part 01: 04. Attention Memory Optimization](../01_Hardware_Math_and_Systems/04_Attention_Memory_Optimization.md)
- [11. KV Cache and Memory Growth](../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md)
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

可选使用 `deepseek-ai/DeepSeek-V2-Lite` 和固定推理 workload。先用 Transformers 读取 config、固定 tokenizer 与输入语义，再以 vLLM 作为支持条件满足时的 Serving 主基线，SGLang 只做同模型、同输入和同生成条件的 backend 对照；只有 backend 成功加载并提供显存、延迟或 trace 证据，才能形成 GPU 结论。执行栈版本、环境隔离和预检方式统一见[模型、执行栈与环境资产表](../gpu_environment_assets.md)。

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
    # TODO 3：校验所有维度为正数，使用上述变量计算 K 和 V 两份缓存。
    # kv_elements = ???；kv_bytes = ???。
    #       返回整数 bytes；不要混入模型权重、临时张量或 CUDA reserved memory。
    # 返回整数 bytes；不要混入模型权重、临时张量或 CUDA reserved memory。
    raise NotImplementedError('请先完成 TODO 代码！')

def summarize_mla_config(config: Dict[str, int]) -> Dict[str, object]:
    """整理 MHA/GQA/MLA 账本所需字段，并保留缺失字段。

    latent_dim、rope_dim 和 dtype_bytes 必须来自显式配置，不为缺失值猜默认值。
    """
    # TODO 2：读取 model、num_layers、num_attention_heads、num_kv_heads、
    # normalized = ???；missing_fields = ???。
    # latent_dim、rope_dim、head_dim、dtype_bytes；标记缺失字段。
    raise NotImplementedError('请先完成 TODO 代码！')

def build_cache_comparison_table(config: Dict[str, object]) -> list[Dict[str, object]]:
    """生成 toy MHA/GQA/MLA 理论 bytes 表，不代表真实 backend。

    表格至少保留 representation、bytes、evidence；evidence 应标记理论估算。
    """
    # TODO 6：先提取并检查配置，再分别计算 MHA、GQA 和本节简化 MLA；
    # mha_bytes = ???；gqa_bytes = ???；mla_bytes = ???；comparison_rows = ???。
    # 返回 representation、bytes、evidence 三列，缺字段时明确报错。
    #       compression_ratio 只能表示理论容量变化，不能表示质量或吞吐收益。
    raise NotImplementedError('请先完成 TODO 代码！')

def extract_attention_dimensions(config: Dict[str, object]) -> Dict[str, object]:
    """从模型配置提取账本字段；不假设缺失字段的默认值。

    返回 normalized 字段和 missing_fields；字段别名只用于兼容命名。
    """
    # TODO 1：兼容 num_hidden_layers / n_layer、num_attention_heads /
    # normalized = ???；missing_fields = ???；head_dim = ???。
    # n_head 等常见别名；hidden_size 可与 attention heads 推出 head_dim，
    # 但 dtype_bytes 必须由实验显式提供。返回 normalized 字段和 missing_fields。
    raise NotImplementedError('请先完成 TODO 代码！')

def compare_kv_representations(baseline_bytes: int, candidate_bytes: int) -> Dict[str, float]:
    """比较两种 KV 表示的理论容量；不推断质量或真实吞吐。

    compression_ratio 仅在 candidate_bytes > 0 时有定义；saving_ratio 是容量比例变化。
    """
    # TODO 5：计算 bytes_delta、compression_ratio、saving_ratio；
    # bytes_delta = ???；compression_ratio = ???；saving_ratio = ???。
    # baseline_bytes > 0，candidate_bytes >= 0。
    raise NotImplementedError('请先完成 TODO 代码！')

def mla_cache_bytes(batch_size: int, seq_len: int, num_layers: int, latent_dim: int, rope_dim: int, dtype_bytes: int = 2) -> int:
    """估算本节简化 MLA latent 与位置相关缓存的理论字节数。

    latent_dim 和 rope_dim 是账本模型变量，不等于任何特定 DeepSeek 版本的 cache layout。
    """
    # TODO 4：校验维度，按本节的简化模型计算 latent 和 positional 两部分 bytes；
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
    for invalid_args in ((0, 4, 1, 1, 8, 2), (1, 4, 1, 1, 8, 0)):
        try: kv_cache_bytes_attention(*invalid_args)
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
# TODO 3 对应实现：普通 MHA/GQA KV Cache 的 K/V 双份账本。
def kv_cache_bytes_attention(batch_size: int, seq_len: int, num_layers: int, num_kv_heads: int, head_dim: int, dtype_bytes: int = 2) -> int:
    """计算 K/V 两份缓存的理论字节数。"""
    values = (batch_size, seq_len, num_layers, num_kv_heads, head_dim, dtype_bytes)
    if any(not isinstance(value, int) or value <= 0 for value in values): raise ValueError('KV Cache 配置必须为正整数')
    return batch_size * seq_len * num_layers * num_kv_heads * head_dim * dtype_bytes * 2

# TODO 2 对应实现：整理模型配置字段，并保留缺失字段。
def summarize_mla_config(config: Dict[str, int]) -> Dict[str, object]:
    """整理 MLA 账本字段，并保留缺失字段。"""
    required = ('model','num_layers','num_attention_heads','num_kv_heads','head_dim','latent_dim','rope_dim','dtype_bytes')
    missing = [key for key in required if key not in config]
    return {'model': config.get('model'), 'fields': {key: config.get(key) for key in required}, 'missing_fields': missing, 'ready_for_estimate': not missing}

# TODO 1 对应实现：兼容配置别名，提取 attention / latent / rope 维度。
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

# TODO 5 对应实现：比较 baseline 与 candidate 的理论容量差异。
def compare_kv_representations(baseline_bytes: int, candidate_bytes: int) -> Dict[str, float]:
    """计算理论容量差异；结果不等于 GPU 实测显存。"""
    if baseline_bytes <= 0 or candidate_bytes < 0: raise ValueError('baseline_bytes 必须 > 0，candidate_bytes 不能为负数')
    delta = baseline_bytes - candidate_bytes
    return {'bytes_delta': delta, 'compression_ratio': candidate_bytes / baseline_bytes, 'saving_ratio': delta / baseline_bytes}

# TODO 4 对应实现：计算简化 MLA latent 与位置相关缓存账本。
def mla_cache_bytes(batch_size: int, seq_len: int, num_layers: int, latent_dim: int, rope_dim: int, dtype_bytes: int = 2) -> int:
    """估算 MLA latent 与位置相关缓存的理论字节数。"""
    values = (batch_size, seq_len, num_layers, latent_dim, rope_dim, dtype_bytes)
    if any(not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError('MLA 缓存配置必须为正整数')
    return batch_size * seq_len * num_layers * (latent_dim + rope_dim) * dtype_bytes

# TODO 6 对应实现：生成 MHA/GQA/MLA 理论容量对照表，并标注证据等级。
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

- **TODO 1**：先从不同配置命名中提取维度，缺失字段必须显式保留，不能猜默认值。
- **TODO 2**：整理模型配置和账本字段，为后续 MHA/GQA/MLA 计算准备输入。
- **TODO 3**：普通 KV Cache 包含 K 和 V 两份张量；`num_kv_heads` 区分 MHA 与 GQA。
- **TODO 4**：MLA 按 latent 与位置相关分量建模，不能用固定比例代替。
- **TODO 5**：压缩比例是理论容量比较，不能推出质量、TTFT、TPOT 或 allocator 峰值。
- **TODO 6**：将三种表示整理成带 evidence 的对照表，明确这是 CPU 理论账本。

真实模型建议使用 `deepseek-ai/DeepSeek-V2-Lite`；如果 backend 不支持 MLA，71 仍可完成 CPU 账本，但 74 不能伪造 CUDA trace。
### Step 5：可选 backend 实验——GQA 与 MLA 的观察性对照

真实 backend 实验固定请求 workload，分别读取一个 GQA 模型和一个 MLA 模型的运行结果。两者模型结构和权重不同，因此只能回答候选系统在当前环境中的质量—成本表现，不能把全部差异归因于 KV 表示。

#### 5.1 环境、模型与结果位置

| 条目 | 固定内容 | 记录目的 |
| --- | --- | --- |
| 基线 | GQA 模型与独立 backend endpoint | 提供可运行的系统参照 |
| 候选 | MLA 模型与独立 backend endpoint | 验证 MLA capability 与运行成本 |
| workload | prompt、输出长度、并发、warmup、重复次数 | 保证请求口径一致 |
| 证据 | 原始 backend JSON + architecture companion | 分离运行事实与架构审计字段 |

```python
# 默认关闭：先分别启动已确认支持对应模型的 OpenAI-compatible backend。
from datetime import datetime, timezone
from pathlib import Path
import json

RUN_MLA_BACKEND_BENCHMARK = False
MLA_BASELINE_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'  # GQA 运行参照，不是权重匹配基线
MLA_CANDIDATE_MODEL = 'deepseek-ai/DeepSeek-V2-Lite'  # MLA 候选
MLA_BASELINE_BASE_URL = 'http://127.0.0.1:8000'
MLA_CANDIDATE_BASE_URL = 'http://127.0.0.1:8001'
MLA_BASELINE_BACKEND = 'vllm'
MLA_CANDIDATE_BACKEND = 'vllm'
MLA_DTYPE = 'float16'
MLA_BASELINE_PEAK_MEMORY_MB = None  # 可从 backend 监控补入
MLA_CANDIDATE_PEAK_MEMORY_MB = None  # 可从 backend 监控补入
MLA_EXISTING_BASELINE_RESULT = ''  # 已有 JSON 时可直接填写
MLA_EXISTING_CANDIDATE_RESULT = ''
MLA_RESULT_DIR = Path('benchmarks/results/71_mla_kv_architecture')

```

#### 5.2 工具与 endpoint 预检

预检只确认教程工具、模型标识和两个 endpoint 配置完整。模型能否加载、是否命中 MLA kernel 以及显存是否足够，必须由各 backend 的启动日志和运行结果证明。

```python
import sys

MLA_REPOSITORY_ROOT = next(
    (path for path in (Path.cwd(), *Path.cwd().parents) if (path / 'tools' / 'benchmark_inference_backend.py').is_file()),
    None,
)
if MLA_REPOSITORY_ROOT is None:
    raise RuntimeError('未找到 tools/benchmark_inference_backend.py，请从教程仓库中运行本节。')
if str(MLA_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(MLA_REPOSITORY_ROOT))
if not MLA_BASELINE_MODEL or not MLA_CANDIDATE_MODEL:
    raise ValueError('必须配置 GQA 基线模型与 MLA 候选模型。')
print({'repository_root': str(MLA_REPOSITORY_ROOT), 'comparison_mode': 'observational'})

```

#### 5.3 固定请求 workload

两个 endpoint 使用相同 prompt、生成长度、并发、warmup 和重复次数。模型 tokenizer 与架构不同会继续作为观察性差异保留，不能通过表面相同的 Token 数掩盖。

```python
MLA_BACKEND_WORKLOAD = {
    'prompt': 'Explain why compact KV representations affect long-context inference cost.',
    'num_prompts': 4,
    'max_tokens': 64,
    'temperature': 0.0,
    'concurrency': 1,
    'warmup': 2,
    'repeats': 3,
}
print(MLA_BACKEND_WORKLOAD)

```

#### 5.4 执行 backend benchmark

执行单元只调用已启动的 endpoint，不负责安装或启动 vLLM/SGLang。若已有同口径原始 JSON，可保持执行开关关闭并在 5.1 填入路径。

```python
if RUN_MLA_BACKEND_BENCHMARK:
    import subprocess

    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    MLA_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    run_specs = [
        ('baseline', MLA_BASELINE_MODEL, MLA_BASELINE_BASE_URL, MLA_BASELINE_BACKEND, MLA_BASELINE_PEAK_MEMORY_MB),
        ('candidate', MLA_CANDIDATE_MODEL, MLA_CANDIDATE_BASE_URL, MLA_CANDIDATE_BACKEND, MLA_CANDIDATE_PEAK_MEMORY_MB),
    ]
    generated_paths = {}
    for role, model_id, base_url, backend_name, peak_memory_mb in run_specs:
        output_path = MLA_RESULT_DIR / f'71_{role}_{timestamp}.json'
        command = [
            sys.executable, str(MLA_REPOSITORY_ROOT / 'tools' / 'benchmark_inference_backend.py'),
            '--base-url', base_url, '--model', model_id, '--label', f'mla_{role}',
            '--project', '71', '--backend', backend_name, '--dtype', MLA_DTYPE,
            '--prompt', MLA_BACKEND_WORKLOAD['prompt'],
            '--num-prompts', str(MLA_BACKEND_WORKLOAD['num_prompts']),
            '--max-tokens', str(MLA_BACKEND_WORKLOAD['max_tokens']),
            '--temperature', str(MLA_BACKEND_WORKLOAD['temperature']),
            '--concurrency', str(MLA_BACKEND_WORKLOAD['concurrency']),
            '--warmup', str(MLA_BACKEND_WORKLOAD['warmup']),
            '--repeats', str(MLA_BACKEND_WORKLOAD['repeats']),
            '--output', str(output_path),
        ]
        if peak_memory_mb is not None:
            command.extend(['--peak-memory-mb', str(peak_memory_mb)])
        subprocess.run(command, cwd=MLA_REPOSITORY_ROOT, check=True)
        generated_paths[role] = output_path
    MLA_BASELINE_RESULT_PATH = generated_paths['baseline']
    MLA_CANDIDATE_RESULT_PATH = generated_paths['candidate']
else:
    MLA_BASELINE_RESULT_PATH = Path(MLA_EXISTING_BASELINE_RESULT) if MLA_EXISTING_BASELINE_RESULT else None
    MLA_CANDIDATE_RESULT_PATH = Path(MLA_EXISTING_CANDIDATE_RESULT) if MLA_EXISTING_CANDIDATE_RESULT else None
    print('Backend benchmark disabled; configure existing result paths or start both endpoints.')

```

#### 5.5 读取结果并核对可比条件

先核对请求数、输出上限、并发、warmup 和重复次数，再查看 TTFT、TPOT、吞吐与显存。缺少质量集或显存记录时保留缺失状态，不用零值代替。

```python
MLA_BACKEND_REPORTS = None
if MLA_BASELINE_RESULT_PATH and MLA_CANDIDATE_RESULT_PATH:
    MLA_BACKEND_REPORTS = {
        'baseline': json.loads(Path(MLA_BASELINE_RESULT_PATH).read_text(encoding='utf-8')),
        'candidate': json.loads(Path(MLA_CANDIDATE_RESULT_PATH).read_text(encoding='utf-8')),
    }
    comparable_keys = ('requests', 'max_tokens', 'concurrency', 'warmup', 'repeats')
    baseline_workload = MLA_BACKEND_REPORTS['baseline']['workload']
    candidate_workload = MLA_BACKEND_REPORTS['candidate']['workload']
    mismatches = [key for key in comparable_keys if baseline_workload.get(key) != candidate_workload.get(key)]
    if mismatches:
        raise ValueError(f'backend workload 不一致：{mismatches}')
    for role, report in MLA_BACKEND_REPORTS.items():
        metrics = report['metrics']
        normalized = report.get('normalized_result', {})
        print(role, {
            'model': report['model'],
            'ttft_ms': metrics['ttft_ms'],
            'tpot_ms': metrics['tpot_ms'],
            'throughput_tokens_per_s': metrics['output_token_throughput_per_s'],
            'peak_memory_mb': normalized.get('metrics', {}).get('peak_memory_mb'),
        })
else:
    print('No paired backend results. The CPU ledger remains theoretical evidence only.')

```

#### 5.6 保存架构 companion 并形成判断

跨模型对照固定标记为 `observational`。即使 endpoint 成功，也要补齐独立质量集、实际峰值显存与 backend capability 后才能进入架构采用判断。

```python
if MLA_BACKEND_REPORTS is not None:
    from tools.architecture_result_schema import make_producer_record, save_record

    baseline_report = MLA_BACKEND_REPORTS['baseline']
    candidate_report = MLA_BACKEND_REPORTS['candidate']
    reports_ok = all(report['metrics'].get('successful_requests', 0) > 0 for report in MLA_BACKEND_REPORTS.values())
    failure = {} if reports_ok else {
        role: report['metrics'].get('failed_requests') for role, report in MLA_BACKEND_REPORTS.items()
    }

    def _backend_cost(report):
        metrics = report['metrics']
        normalized_metrics = report.get('normalized_result', {}).get('metrics', {})
        return {
            'ttft_ms': metrics.get('ttft_ms'),
            'tpot_ms': metrics.get('tpot_ms'),
            'throughput_tokens_per_s': metrics.get('output_token_throughput_per_s'),
            'peak_memory_mb': normalized_metrics.get('peak_memory_mb'),
        }

    companion = make_producer_record(
        semantic_id='ARCH-MLA-KV',
        comparison_mode='observational',
        source={
            'notebook': '02_PyTorch_Algorithms/71_MLA_KV_Cache_Architecture_Benchmark.ipynb',
            'raw_result_paths': {
                'baseline': str(MLA_BASELINE_RESULT_PATH),
                'candidate': str(MLA_CANDIDATE_RESULT_PATH),
            },
        },
        runtime={
            'baseline_backend': MLA_BASELINE_BACKEND,
            'candidate_backend': MLA_CANDIDATE_BACKEND,
            'dtype': MLA_DTYPE,
            'device': 'cuda',
        },
        workload=baseline_report['workload'],
        baseline={'name': 'gqa_reference', 'model': baseline_report['model'], 'representation': 'gqa'},
        candidates=[{'name': 'mla_candidate', 'model': candidate_report['model'], 'representation': 'mla'}],
        quality={'status': 'not_recorded', 'reason': 'independent matched quality set is required'},
        cost={role: _backend_cost(report) for role, report in MLA_BACKEND_REPORTS.items()},
        mechanism={
            'baseline_representation': 'gqa',
            'candidate_representation': 'mla',
            'backend_capability_observed': reports_ok,
            'theoretical_ledger_is_measured_memory': False,
        },
        evidence_level='real_benchmark' if reports_ok else 'gpu_smoke',
        status='ok' if reports_ok else 'failed',
        failure=failure,
        decision={
            'status': 'tune' if reports_ok else 'reject',
            'reason': 'cross-model evidence needs matched quality and memory before adoption' if reports_ok else 'backend run failed',
            'next_action': 'add matched quality, peak memory, backend version and kernel evidence',
        },
    )
    companion_path = Path(MLA_CANDIDATE_RESULT_PATH).with_name(
        Path(MLA_CANDIDATE_RESULT_PATH).stem + '_architecture.json'
    )
    if companion_path.exists():
        print(f'Companion already exists and was not overwritten: {companion_path}')
    else:
        save_record(companion_path, companion)
        print(f'Saved companion: {companion_path}')

```
