# Inference. Service Project | 推理服务交付综合项目

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/05_Integrated_Projects_and_Decisions/01_Inference_Service/Inference_Service_Project.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

本项目固定模型和短算术请求，在同一单卡环境下比较两套组合配置：基线关闭前缀缓存并设 `max_num_seqs=4`，候选开启缓存并设为 8。除延迟、吞吐和显存外，还核对答案正确率、回答完整度及文本变化；Part 02 · 72 的交接清单只补充来源信息，不作为运行前提。
## 项目执行
### 1. 配置本次运行

按需修改下方配置项；仓库内路径由项目自动解析。

```python
RUN_LABEL = 'replace_with_unique_label'  # 本轮实验唯一标签；每轮使用新值
RUN_STAGE = 'contract_check'  # contract_check 只检查；measure 才启动服务并测量
ACTIVE_CANDIDATE = 'candidate'  # baseline 或 candidate；每次只测一个候选
DEPLOYMENT_HANDOFF = None  # 可选：72 的仓库相对 JSON 路径；不使用时保持 None
```

### 2. 检查环境、模型与可选来源

本项目使用本机缓存的固定模型目录并独立执行测量。若提供 72 交接清单，只记录其路径与哈希，作为候选来源线索；其中的性能数字不代替本项目测量。

```python
from pathlib import Path
import json
import subprocess
import sys
import yaml

from tools.project_runtime import resolve_project_root
from tools.inference_delivery_runtime import (
    build_delivery_record, compare_output_fidelity, evaluate_service_candidate, load_json_record, missing_candidate_results,
    normalized_backend_result, save_delivery_record,
    select_service_candidate, validate_delivery_config, validate_deployment_handoff,
)
from tools.inference_project_runtime import run_backend_benchmark, start_optional_vllm, stop_optional_vllm
from tools.model_runtime import resolve_model_snapshot

if RUN_STAGE not in {'contract_check', 'measure'}:
    raise ValueError(f'未知 RUN_STAGE：{RUN_STAGE}')
PROJECT_ROOT = resolve_project_root()
PROJECT_DIR = PROJECT_ROOT / '05_Integrated_Projects_and_Decisions/01_Inference_Service'
config_path = PROJECT_DIR / 'project_config.example.yaml'
config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
config['run_label'] = RUN_LABEL
config['active_candidate'] = ACTIVE_CANDIDATE
config.setdefault('inputs', {})['deployment_handoff'] = DEPLOYMENT_HANDOFF
config_errors = validate_delivery_config(config, allow_placeholder=RUN_STAGE == 'contract_check')
if RUN_STAGE == 'measure' and config_errors:
    raise ValueError('项目配置未完成：' + '; '.join(config_errors))
print({'run_stage': RUN_STAGE, 'config': str(config_path), 'config_errors': config_errors})

handoff_value = config.get('inputs', {}).get('deployment_handoff')
handoff_path = PROJECT_ROOT / handoff_value if handoff_value else None
handoff = None
handoff_check = None
if handoff_path and handoff_path.is_file():
    handoff = load_json_record(handoff_path)
    handoff_check = validate_deployment_handoff(handoff)
    print(handoff_check)
else:
    print({'optional_handoff': str(handoff_path) if handoff_path else None, 'status': 'not_provided' if not handoff_path else 'missing; independent measurement continues'})

workload_path = PROJECT_ROOT / config['workload']['path']
if not workload_path.is_file():
    raise FileNotFoundError(f'请求负载不存在：{workload_path}')
workload_rows = [json.loads(line) for line in workload_path.read_text(encoding='utf-8').splitlines() if line.strip()]
if not workload_rows or any(not isinstance(row, dict) or not isinstance(row.get('expected_answer'), (int, str)) or not isinstance(row.get('chat_template_kwargs'), dict) or row['chat_template_kwargs'].get('enable_thinking') is not False for row in workload_rows):
    raise ValueError('请求负载必须为每题提供标准答案，并关闭 Qwen3 思考输出以核对短答案完成度。')

resolved_models = {}
if RUN_STAGE == 'measure':
    subprocess.run([sys.executable, str(PROJECT_ROOT / 'tools/environment_preflight.py'), '--profile', 'INF-VLLM-011-CU128'], cwd=PROJECT_ROOT, check=True)
    for candidate_name, candidate_config in config['candidates'].items():
        candidate_cache = Path(candidate_config['cache_dir'])
        if not candidate_cache.is_absolute():
            candidate_cache = PROJECT_ROOT / candidate_cache
        resolved_models[candidate_name], _ = resolve_model_snapshot(
            candidate_config['model'], candidate_config['model_revision'],
            cache_dir=candidate_cache, allow_download=False,
        )
    print({'environment': 'passed', 'local_models': resolved_models, 'download_attempted': False})
```

### 3. 确认候选与比较条件

两套配置使用同一模型快照、vLLM 版本、请求集和生成参数；基线关闭缓存并限制并发序列数，候选开启缓存并提高并发序列上限。固定负载包含重复共享前缀，请求并发固定为 4。

| 指标 | 预先设定的检查 |
|:---|:---|
| 请求成功率 | 必须为 100% |
| 答案正确率、完整正确率 | 标准答案全部正确，且响应以正常停止结束 |
| 格式有效率、截断率 | `Answer: <integer>` 格式有效率为 100%；截断率为 0 |
| 同题答案重复一致率 | 同一算术题的重复请求都解析为相同整数；作为稳定性证据 |
| 精确文本一致率 | 记录基线与候选文本差异，仅作描述；不单独判定答案质量 |
| TTFT P95、TPOT P95 | 共享前缀下预期不明显回退；同时检查绝对 SLO 与相对基线门槛 |
| 输出吞吐 | 不低于配置门槛 |
| GPU 峰值占用 | 不超过配置上限；记录包含同卡其他进程的占用 |

这是缓存与并发上限的联合配置对比，不拆分两项设置各自的收益。
### 4. 运行候选并保存结果

每次只启动一个候选服务。运行后自动采集请求级输出和 GPU 显存读数，随后关闭服务；切换 `active_candidate` 后再测另一个候选。

```python
active_name = config['active_candidate']
active = config['candidates'][active_name]
workload = config['workload']
active_output = PROJECT_ROOT / active['result_path'].format(run_label=config['run_label'])
if RUN_STAGE == 'measure':
    if active_output.exists():
        raise FileExistsError(f'候选结果已存在，请更换 run_label 或结果路径：{active_output}')
    server = log_handle = None
    log_path = PROJECT_ROOT / config['outputs']['directory'] / f"{active_name}_{config['run_label']}.server.log"
    try:
        server, log_handle, port, selected_dtype, model_path = start_optional_vllm(
            model_id=resolved_models[active_name], model_source='local', cache_dir=(PROJECT_ROOT / active['cache_dir']),
            dtype=active['dtype'], max_model_len=active['max_model_len'],
            gpu_memory_utilization=active['gpu_memory_utilization'], enforce_eager=active['enforce_eager'],
            served_model_name=active['served_model_name'],
            enable_prefix_caching=active['cache_policy'] == 'enabled',
            max_num_seqs=active['max_num_seqs'], scheduling_policy=active['scheduling_policy'], log_path=log_path,
        )
        active_report = run_backend_benchmark(
            project='part05-inference-service', base_url=f'http://127.0.0.1:{port}',
            model=active['served_model_name'], label=f"{active_name}_{config['run_label']}", output=active_output,
            workload=workload['path'], num_prompts=workload['num_prompts'],
            max_tokens=workload['max_tokens'], concurrency=workload['concurrency'],
            warmup=workload['warmup'], repeats=workload['repeats'], backend=active['backend'],
            dtype=selected_dtype, cache_policy=active['cache_policy'], role=active['role'],
            measure_device_peak_memory=True, include_requests=True,
        )
        active_report['service_execution'] = {
            'model_path': model_path, 'model_revision': active['model_revision'], 'model_source': 'local_cache',
            'enable_prefix_caching': active['cache_policy'] == 'enabled',
            'max_model_len': active['max_model_len'],
            'gpu_memory_utilization': active['gpu_memory_utilization'],
            'max_num_seqs': active['max_num_seqs'],
            'scheduling_policy': active['scheduling_policy'],
            'server_log': str(log_path),
        }
        active_output.write_text(json.dumps(active_report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print({'candidate': active_name, 'result_path': str(active_output), 'metrics': active_report['metrics'], 'service_execution': active_report['service_execution']})
    finally:
        if server is not None:
            stop_optional_vllm(server, log_handle)
else:
    print({'active_candidate': active_name, 'cache_policy': active['cache_policy'], 'output': str(active_output)})
```

### 5. 读取证据并检查候选

所有候选必须使用同一冻结请求协议。服务失败、模型快照不一致或基线不可用时直接拒绝；答案、完整性、显存或 SLO 证据缺失时保持 `tune`。精确文本差异单独报告，不替代标准答案核对。

```python
results = {}
raw_results = {}
fidelity_scores = {}
for name, candidate in config['candidates'].items():
    path = PROJECT_ROOT / candidate['result_path'].format(run_label=config['run_label'])
    if not path.is_file():
        print({'candidate': name, 'missing_result': str(path)})
        continue
    raw_results[name] = load_json_record(path)
    results[name] = normalized_backend_result(raw_results[name])
    results[name]['service_execution'] = raw_results[name].get('service_execution')

baseline_names = [name for name, item in config['candidates'].items() if item['role'] == 'baseline']
baseline_name = baseline_names[0]
task_quality_keys = ('expected_answer_accuracy', 'valid_answer_format_rate', 'response_completion_rate', 'complete_correct_rate', 'truncated_response_rate', 'unique_answer_tasks', 'answer_repeat_consistency')
def task_quality_metrics(report):
    metrics = report.get('metrics', {})
    return {key: metrics.get(key) for key in task_quality_keys}
decisions = {}
selection = {'decision': 'tune', 'selected_candidate': None, 'accepted_candidates': []}
if baseline_name in results:
    baseline = results[baseline_name]
    baseline_task_quality = task_quality_metrics(raw_results[baseline_name])
    fidelity_scores[baseline_name] = 1.0 if raw_results[baseline_name].get('results') else None
    for name, result in results.items():
        fidelity = compare_output_fidelity(raw_results[baseline_name], raw_results[name])
        fidelity_scores[name] = fidelity['score']
        model_path = (raw_results[name].get('service_execution') or {}).get('model_path')
        baseline_model_path = (raw_results[baseline_name].get('service_execution') or {}).get('model_path')
        same_model_snapshot = bool(model_path) and model_path == baseline_model_path
        baseline_metrics = results[baseline_name].get('metrics', {})
        baseline_available = baseline_metrics.get('successful_requests', 0) > 0 and baseline_metrics.get('failed_requests', 0) == 0
        decisions[name] = evaluate_service_candidate(
            result, baseline, config['gates'],
            output_fidelity_score=fidelity_scores.get(name),
            baseline_output_fidelity_score=fidelity_scores.get(baseline_name),
            task_quality=task_quality_metrics(raw_results[name]),
            baseline_task_quality=baseline_task_quality,
            artifact_verified=same_model_snapshot, rollback_ready=baseline_available,
        )
        decisions[name]['output_fidelity'] = fidelity
    selection = select_service_candidate(decisions, results)
print({'candidate_decisions': decisions, 'selection': selection})
```

### 6. 登记交付决策

只有全部候选都有结果时才写最终记录。报告保留答案与完成度指标、精确文本一致率、延迟/吞吐/显存、实际服务设置、日志位置及可选的 72 来源哈希。

```python
if RUN_STAGE == 'measure':
    missing_candidates = missing_candidate_results(config['candidates'], results)
    if missing_candidates:
        print({'delivery_record': 'pending', 'missing_candidates': missing_candidates, 'next_action': '逐个测完剩余候选后再收口'})
    else:
        record = build_delivery_record(
            run_label=config['run_label'], handoff_path=handoff_path, workload=config['workload'],
            results=results, decisions=decisions, selection=selection,
        )
        output_path = PROJECT_ROOT / config['outputs']['directory'] / f"inference_service_delivery_{config['run_label']}.json"
        saved = save_delivery_record(output_path, record)
        print({'saved': str(saved), 'decision': record['decision'], 'selected_candidate': record['selected_candidate']})
else:
    print('contract_check 完成；填写配置并启动一个候选服务后再切换 measure。')
```

## 相关阅读

- [69 · Prefix Caching Benchmark](../../02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark.md)：了解本项目缓存候选所依据的局部测量。
