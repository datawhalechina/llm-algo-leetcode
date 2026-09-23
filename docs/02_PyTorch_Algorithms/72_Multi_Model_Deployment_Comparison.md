# 72. Multi Model Deployment Comparison | Multi-Model Deployment Comparison

**难度：** Hard | **环境：** CPU 可完成配置与决策逻辑；GPU / backend 用于真实部署对比 | **标签：** 推理优化、模型部署、backend、benchmark | **目标人群：** 想比较模型与推理运行时的项目实践者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/72_Multi_Model_Deployment_Comparison.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

同一个模型可以通过不同 backend 部署，同一个 backend 也可能因为模型架构、权重格式、dtype 和上下文长度不同而表现不同。本项目把模型产物、运行时和 workload 放进同一张对照表，帮助你判断：某个模型能否被目标 backend 加载，以及它在当前硬件和请求条件下是否值得采用。

候选模型可以从 DeepSeek、Qwen、GLM 等系列中选择；候选名称、revision、量化格式和 backend 必须在实验记录中明确。CPU 路径先验证配置、兼容性字段和决策逻辑；真实加载、延迟、吞吐和显存结论必须来自 GPU / backend。

## 前置阅读

**导语：** 先用 66 建立统一推理 baseline，再根据量化、缓存和模型架构选择对应入口；本项目最后比较模型与 backend 的部署条件。

- [Part 02 · 66 推理性能对比实验](./66_Inference_Performance_Comparison.md)
- [Part 02 · 67 量化推理与部署项目](./67_Quantized_Inference_and_Deployment.md)
- [Part 02 · 71 MLA / KV Cache 结构基准](./71_MLA_KV_Cache_Architecture_Benchmark.md)
- [部署与异构系统专题](../topic_discussion/deployment_heterogeneous/intro.md)

### Step 1：明确部署对象和比较问题

| 对象 | 本项目记录什么 | 不能直接推出什么 |
|:---|:---|:---|
| 模型 | model ID、revision、架构、参数规模和上下文长度 | 同系列模型一定有相同性能 |
| 模型产物 | 原始权重、量化权重、格式和 tokenizer | 文件更小就一定更快 |
| backend | Transformers、vLLM、SGLang、TensorRT-LLM 或 llama.cpp | backend 名称本身代表最优方案 |
| 运行条件 | GPU、dtype、batch、并发、输入长度和输出长度 | 不同 workload 可以直接横比 |
| 项目输出 | 兼容性、资源指标、服务指标和部署决策 | 单次 smoke test 等于稳定 benchmark |
### Step 2：理解模型产物与 backend 的匹配关系

| 部署路径 | 重点检查 | 主要证据 |
|:---|:---|:---|
| Transformers | config、tokenizer、dtype、device map | 加载状态、输出稳定性、峰值显存 |
| vLLM / SGLang | 架构支持、格式、启动参数和版本 | endpoint、TTFT、TPOT、吞吐、P99、显存 |
| TensorRT-LLM | engine 构建参数、硬件和 kernel 支持 | engine 构建、加载、延迟和吞吐 |
| llama.cpp / GGUF | GGUF 文件、量化类型和 offload 配置 | 加载状态、tokens/s、内存和质量 |
| 不可行组合 | 缺少权重、格式不支持或 kernel 不匹配 | unsupported / load_failed，不填零值 |
### Step 3：固定 workload 和部署指标

比较模型和 backend 时，只改变模型、模型产物或 backend 中的一项；输入、采样参数和评测方式保持一致。

| 条件组 | 固定字段 | 结果字段 |
|:---|:---|:---|
| 模型与产物 | model、revision、format、quantization、tokenizer | load status、加载显存 |
| 请求 workload | prompt tokens、generated tokens、请求数、并发度 | TTFT、TPOT、端到端延迟、吞吐 |
| 运行环境 | GPU、driver、CUDA、PyTorch、backend、dtype | peak allocated、peak reserved、OOM |
| 质量检查 | 固定 prompt、采样参数和输出长度 | 成功率、格式通过率、任务质量 |
| 决策 | 延迟、吞吐、显存和质量门槛 | accept、tune 或 reject |
### Step 4：实现配置审计和项目决策

题目区只实现 CPU 可验证的配置、字段和决策逻辑；不会伪造模型加载或 GPU 性能。答案区和解析区保持相同的函数、参数和 TODO 顺序。


```python
from typing import Any, Dict, List

def build_deployment_case(model_id, backend, model_format, hardware, workload):
    """构造一个可比较的部署候选。"""
    # TODO 1：校验必要字段，缺失时返回 status='invalid_config'。
    raise NotImplementedError('请先完成 TODO 1！')

def compare_deployment_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总同一 workload 的成功与失败候选。"""
    # TODO 2：检查 workload_key 一致性，分离 load_success / unsupported / load_failed。
    raise NotImplementedError('请先完成 TODO 2！')

def make_deployment_decision(summary, min_quality: float, max_ttft_ms: float):
    """根据质量和 TTFT 门槛输出 accept / tune / reject。"""
    # TODO 3：缺指标时 tune；无可比较候选时 reject；满足质量和延迟才 accept。
    raise NotImplementedError('请先完成 TODO 3！')

```


```python
def test_deployment_project_template():
    try:
        case = build_deployment_case('Qwen/example', 'vllm', 'safetensors', 'single_gpu', {'prompt_tokens': 128})
        assert case['status'] == 'ready'
        invalid = build_deployment_case('', 'vllm', 'safetensors', 'single_gpu', {})
        assert invalid['status'] == 'invalid_config'
        summary = compare_deployment_records([{'name': 'baseline', 'workload_key': 'w1', 'status': 'load_success', 'quality': 0.9, 'ttft_ms': 30.0}, {'name': 'candidate', 'workload_key': 'w1', 'status': 'unsupported'}])
        assert summary['workload_consistent'] is True
        decision = make_deployment_decision(summary, min_quality=0.8, max_ttft_ms=50.0)
        assert decision['decision'] in {'accept', 'tune', 'reject'}
        print('测试通过：72 部署对比项目模板逻辑正确。')
    except NotImplementedError:
        print('请先完成 TODO 代码！')

test_deployment_project_template()

```

🛑 **STOP HERE** 🛑

先完成配置审计和决策逻辑，再进入可选 GPU / backend 实验。

## 参考代码与解析

CPU 结果只能说明配置和决策逻辑正确；真实部署结论必须来自实际模型、backend、硬件和 workload。

```python
def build_deployment_case(model_id, backend, model_format, hardware, workload):
    if not all(isinstance(x, str) and x.strip() for x in (model_id, backend, model_format, hardware)) or not isinstance(workload, dict) or not workload:
        return {'status': 'invalid_config'}
    return {'status': 'ready', 'model_id': model_id, 'backend': backend, 'model_format': model_format, 'hardware': hardware, 'workload': dict(workload), 'workload_key': tuple(sorted(workload.items()))}

def compare_deployment_records(records):
    keys = {item.get('workload_key') for item in records}
    return {'workload_consistent': len(keys) <= 1, 'load_success': sum(item.get('status') == 'load_success' for item in records), 'unsupported': sum(item.get('status') == 'unsupported' for item in records), 'load_failed': sum(item.get('status') == 'load_failed' for item in records), 'records': list(records)}

def make_deployment_decision(summary, min_quality, max_ttft_ms):
    if not summary.get('workload_consistent') or summary.get('load_success', 0) == 0:
        return {'decision': 'reject', 'reason': 'no_comparable_candidate'}
    usable = [r for r in summary['records'] if r.get('status') == 'load_success']
    if any('quality' not in r or 'ttft_ms' not in r for r in usable):
        return {'decision': 'tune', 'reason': 'missing_measurements'}
    accepted = [r for r in usable if r['quality'] >= min_quality and r['ttft_ms'] <= max_ttft_ms]
    return {'decision': 'accept' if accepted else 'reject', 'accepted': [r['name'] for r in accepted]}

```


```python
# Step 5 配置单元：只登记真实部署条件，不在此处自动启动模型。
RUN_REAL_DEPLOYMENT = False
MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
MODEL_REVISION = 'main'
BACKEND = 'vllm'
MODEL_FORMAT = 'safetensors'
RESULT_PATH = 'benchmarks/results/72_model_deployment.json'
print({
    'run_real_deployment': RUN_REAL_DEPLOYMENT,
    'model_id': MODEL_ID,
    'revision': MODEL_REVISION,
    'backend': BACKEND,
    'model_format': MODEL_FORMAT,
    'result_path': RESULT_PATH,
})
```

### Step 5（可选）：真实模型与 backend 部署对比

候选可以选择 DeepSeek、Qwen、GLM 等系列，但必须登记具体 model ID、revision、格式、dtype、backend 版本和硬件。只改变模型、模型产物或 backend 中的一项；没有可用 backend 时输出 unsupported，不要用 CPU 结果替代部署证据。

| 结果字段 | 记录内容 |
|:---|:---|
| 模型与环境 | model、revision、format、backend、dtype、GPU、CUDA、驱动 |
| 服务指标 | load status、TTFT、TPOT、E2E、吞吐、P50/P95/P99 |
| 资源指标 | peak allocated、peak reserved、KV Cache 配置、OOM |
| 质量与决策 | 输出成功率、任务质量、失败原因、accept / tune / reject |

## 相关阅读

- [推理优化专题](../topic_discussion/inference_optimization/intro.md)
- [部署与异构系统专题](../topic_discussion/deployment_heterogeneous/intro.md)
- [66 推理性能对比实验](./66_Inference_Performance_Comparison.md)
- [67 量化推理与部署项目](./67_Quantized_Inference_and_Deployment.md)