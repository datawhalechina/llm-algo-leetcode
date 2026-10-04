# PAR-MODEL-INTERNAL. Model Internal Parallel Benchmark | 模型内部并行基准

**难度：** Hard | **环境：** CPU-first | **标签：** `PP`, `TP`, `CP`, `Device Mesh`

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/PAR-MODEL-INTERNAL_Model_Internal_Parallel_Benchmark.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

本项目比较 Pipeline、Tensor 与 Context Parallel 三种模型内部切分轴。学习者先把 world size 分解为 Device Mesh，再检查层、张量和序列布局是否兼容，最后用容量、通信、bubble 与吞吐门槛筛选组合策略。

## 前置阅读

先理解三种切分对象，再进入组合项目：

- [28 Pipeline Parallel](./28_Pipeline_Parallelism_MicroBatch.md)
- [29 Tensor Parallel](./29_Tensor_Parallelism_Sim.md)
- [PAR-CONTEXT Context / Sequence Parallel](./PAR-CONTEXT_Context_and_Sequence_Parallelism.md)
- [49 Parallelism Strategy Selection](./49_Parallelism_Strategy_Selection.md)

### Step 1：从切分对象建立 Device Mesh

PP 切连续层，TP 切单层矩阵与 Head，CP 切序列。组合策略必须满足 `pp × tp × cp = world size`，每个 mesh 维度对应独立进程组，不能把不同轴的 collective 混为一组。

| 切分轴 | 分摊对象 | 主要新增代价 |
| --- | --- | --- |
| PP | 层与 stage | activation P2P、bubble、stage 不均 |
| TP | 矩阵、特征维与 Head | 高频 All-Reduce / Gather / Scatter |
| CP | sequence / context | K/V 或中间状态交换、长序列通信 |

### Step 2：检查布局兼容性

层数应能映射到 PP stage，Attention Head / KV Head 应能映射到 TP rank，序列分块应能映射到 CP rank。可整除只是最低条件；负载均衡、padding 和通信组仍会改变实际效率。

### Step 3：用容量与关键路径比较组合

组合策略先满足单 rank 容量，再比较 bubble、暴露通信和吞吐。增加并行维度并不自动带来收益：TP/CP 增加层内同步，PP 增加流水空转，三者还可能在同一时间线竞争链路。

### Step 4：实现 Mesh 校验、候选准入与选择

| TODO | 机制责任 | 重点验证 |
| --- | --- | --- |
| TODO 1 | 校验 world size 与 PP/TP/CP 布局 | mesh 乘积、层、Head、序列整除 |
| TODO 2 | 应用容量、bubble 与通信门槛 | 多个失败原因、运行失败 |
| TODO 3 | 从合格候选选择组合 | 吞吐优先，通信和显存作为 tie-break |


```python
from copy import deepcopy

```


```python
def validate_mesh_layout(config):
    """校验 PP、TP、CP 的 Device Mesh 与模型布局。"""
    # TODO 1：模型内部切分机制——检查 mesh 乘积以及层、Head、序列的整除关系。
    # errors = ???
    return {'valid': not errors, 'errors': errors}


def evaluate_internal_parallel_candidate(metrics, limits):
    """按运行状态、容量、bubble 和通信占比判断候选是否准入。"""
    result = deepcopy(metrics)
    # TODO 2：候选准入机制——保留全部失败原因，不只返回布尔值。
    # failures = ???
    result['eligible'] = not failures
    result['gate_failures'] = failures
    return result


def select_internal_parallel_candidate(candidates):
    """从通过门槛的 PP/TP/CP 组合中选择候选。"""
    eligible = [item for item in candidates if item['eligible']]
    if not eligible:
        return {'decision': 'reject', 'candidate': None}
    # TODO 3：优先吞吐，其次通信占比和单 rank 峰值显存。
    # selected = ???
    return {'decision': 'accept', 'candidate': selected['name']}

```

### 测试


```python
def run_model_internal_parallel_tests():
    valid = {'world_size': 8, 'pp': 2, 'tp': 2, 'cp': 2, 'num_layers': 24, 'num_heads': 16, 'sequence_length': 4096}
    invalid = {**valid, 'world_size': 4, 'num_heads': 15}
    assert validate_mesh_layout(valid) == {'valid': True, 'errors': []}
    assert set(validate_mesh_layout(invalid)['errors']) == {'mesh_product', 'attention_heads'}
    limits = {'memory_mb': 12000, 'bubble_ratio': 0.2, 'communication_ratio': 0.3}
    tp = evaluate_internal_parallel_candidate({'name': 'tp2', 'peak_memory_mb': 11000, 'bubble_ratio': 0.0, 'communication_ratio': 0.22, 'throughput': 120, 'failure': None}, limits)
    hybrid = evaluate_internal_parallel_candidate({'name': 'pp2_tp2', 'peak_memory_mb': 9000, 'bubble_ratio': 0.25, 'communication_ratio': 0.28, 'throughput': 130, 'failure': None}, limits)
    assert tp['eligible'] is True
    assert hybrid['gate_failures'] == ['bubble']
    assert select_internal_parallel_candidate([tp, hybrid]) == {'decision': 'accept', 'candidate': 'tp2'}
    assert select_internal_parallel_candidate([])['decision'] == 'reject'
    print('✅ Device Mesh、候选准入与组合选择通过基础校验。')


run_model_internal_parallel_tests()

```

---

🛑 **STOP HERE** 🛑

> 请先完成题目区并通过测试，再查看参考代码。

---

## 参考代码与解析

### 代码


```python
def validate_mesh_layout(config):
    """校验 PP、TP、CP 的 Device Mesh 与模型布局。"""
    errors = []
    if config['pp'] * config['tp'] * config['cp'] != config['world_size']:
        errors.append('mesh_product')
    if config['num_layers'] % config['pp'] != 0:
        errors.append('pipeline_layers')
    if config['num_heads'] % config['tp'] != 0:
        errors.append('attention_heads')
    if config['sequence_length'] % config['cp'] != 0:
        errors.append('context_blocks')
    return {'valid': not errors, 'errors': errors}


def evaluate_internal_parallel_candidate(metrics, limits):
    """按运行状态、容量、bubble 和通信占比判断候选是否准入。"""
    result = deepcopy(metrics)
    failures = []
    if metrics.get('failure'):
        failures.append('runtime')
    if metrics['peak_memory_mb'] > limits['memory_mb']:
        failures.append('memory')
    if metrics['bubble_ratio'] > limits['bubble_ratio']:
        failures.append('bubble')
    if metrics['communication_ratio'] > limits['communication_ratio']:
        failures.append('communication')
    result['eligible'] = not failures
    result['gate_failures'] = failures
    return result


def select_internal_parallel_candidate(candidates):
    """从通过门槛的 PP/TP/CP 组合中选择候选。"""
    eligible = [item for item in candidates if item['eligible']]
    if not eligible:
        return {'decision': 'reject', 'candidate': None}
    selected = sorted(eligible, key=lambda item: (-item['throughput'], item['communication_ratio'], item['peak_memory_mb']))[0]
    return {'decision': 'accept', 'candidate': selected['name']}

```

### 解析

- **TODO 1** 把 Device Mesh 乘积与模型布局约束分开记录，便于定位是资源分解还是结构映射失败。
- **TODO 2** 先过运行、容量、bubble 和通信门槛；未准入的高吞吐结果不能进入最终排序。
- **TODO 3** 排序规则表达当前 workload 的目标，不能替代真实多卡时间线和正确性证据。

### Step 5（可选）：真实多卡组合验证

#### 5.1 环境与固定 workload

固定模型 revision、tokenizer、global/micro batch、序列长度、dtype、world size、拓扑、warmup 和 repeats。GPU 验证将在最终环境验证阶段完成。

#### 5.2 多卡与 backend 预检

检查 CUDA、NCCL、torchrun、Device Mesh API 与可见 GPU。

#### 5.3 配置 G0 / G1 / G2

G0 为单轴 TP 或 PP，G1 为另一单轴候选，G2 为 PP×TP 或 TP×CP 组合；每组使用独立 JSON。

#### 5.4 执行并保存 JSON

执行命令在真实环境确认后登记，不在 Notebook 打开时自动启动多进程。

#### 5.5 读取结果与记录证据

结果至少包含 workload、hardware、mesh、metrics、artifact、failure、evidence_level 与 decision。

#### 5.6 解释结果与形成决策

联合解释单 rank 显存、bubble、collective、暴露通信、吞吐、正确性和扩展效率。


```python
# 默认关闭；真实多卡验证留到统一 GPU 环境阶段。
RUN_GPU_EXPERIMENT = False
WORLD_SIZE = 4
EXPERIMENTS = [
    {'name': 'G0_tp', 'mesh': {'pp': 1, 'tp': 4, 'cp': 1}, 'command': []},
    {'name': 'G1_pp', 'mesh': {'pp': 4, 'tp': 1, 'cp': 1}, 'command': []},
    {'name': 'G2_pp_tp', 'mesh': {'pp': 2, 'tp': 2, 'cp': 1}, 'command': []},
]
RESULT_DIR = 'benchmarks/results/par_model_internal'
print({'run': RUN_GPU_EXPERIMENT, 'world_size': WORLD_SIZE, 'groups': [item['name'] for item in EXPERIMENTS]})

```
