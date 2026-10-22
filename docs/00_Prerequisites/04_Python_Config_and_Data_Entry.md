# 04. Python Config and Data Entry | Python 配置与数据入口

**难度：** Easy | **环境：** CPU-first | **标签：** `Python`, `配置`, `I/O`, `复现` | **目标人群：** Part 2-4 前置补课者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/04_Python_Config_and_Data_Entry.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

一次运行除了对象内部状态，还包含配置、输入来源、资源路径和输出产物。缺少统一记录时，相同代码也可能因为外部条件不同而无法复现。

完成本节后，你应能区分内部状态与外部运行条件，并用配置、路径、产物清单和运行身份追踪一次实验。

**关键词：** `config`, `path`, `artifact`, `run_id`

![配置、资源与运行产物](../public/00_Prerequisites/04_config_artifact_manifest_map.svg)

## 前置阅读
**导语：** 先理解函数契约和对象状态，再把它们落到配置、文件和运行产物。
- [03. Python OOP and Utility Patterns | Python 面向对象与工具模式](./03_Python_OOP_and_Utility_Patterns.md)
- [0A 组页](./0A.md)

## Q1：配置覆盖后为什么还要检查 schema？

覆盖规则决定候选参数，schema 决定它是否具备运行所需字段。合并函数应返回新对象，避免修改基础配置；缺失字段应在进入文件或训练流程前失败。

```python
import json
from pathlib import Path
from tempfile import TemporaryDirectory


def merge_and_validate_config(base, overrides, required_fields):
    """合并配置并检查必需字段，不修改输入字典。"""
    merged = {**base, **overrides}
    missing = [field for field in required_fields if field not in merged]
    if missing:
        raise KeyError(f'missing config fields: {missing}')
    return merged


base = {'model': 'TinyLM', 'precision': 'bf16', 'batch_size': 4}
config = merge_and_validate_config(base, {'batch_size': 8}, ('model', 'precision', 'batch_size'))
print(config)
```

## Q1 验证：覆盖、输入不变和缺失字段是否可检查？

```python
base = {'model': 'TinyLM', 'precision': 'bf16', 'batch_size': 4}
merged = merge_and_validate_config(base, {'precision': 'fp16'}, ('model', 'precision', 'batch_size'))
assert merged['precision'] == 'fp16'
assert base['precision'] == 'bf16'

try:
    merge_and_validate_config({}, {}, ('model',))
except KeyError as error:
    assert 'model' in str(error)
else:
    raise AssertionError('缺失 model 应触发 KeyError')
print('✅ 配置 schema 通过')
```

## Q2：路径和 `with` 怎样管理资源生命周期？

路径负责定位文件，`with` 负责资源何时创建和释放。临时目录适合一次性检查：作用域内文件必须可读，退出作用域后目录应被自动清理。

```python
def write_json(path, payload):
    """写入 UTF-8 JSON，并返回读回后的对象。"""
    path = Path(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return json.loads(path.read_text(encoding='utf-8'))


with TemporaryDirectory() as tmpdir:
    temporary_root = Path(tmpdir)
    config_path = temporary_root / 'config.json'
    restored = write_json(config_path, {'model': 'TinyLM', 'seed': 42})
    print(config_path, restored)
```

## Q2 验证：文件可读且临时目录确实被清理了吗？

```python
with TemporaryDirectory() as tmpdir:
    temporary_root = Path(tmpdir)
    payload_path = temporary_root / 'payload.json'
    assert write_json(payload_path, {'ok': True}) == {'ok': True}
    assert payload_path.exists()

assert not temporary_root.exists()
print('✅ 文件与资源生命周期通过')
```

## Q3：怎样让输入、缓存、日志和 artifact 路径保持稳定？

同一 `run_name` 应映射到确定的目录结构，使训练、验证和复查代码不用重复拼接路径。这里只约定位置，不假设文件已经生成。

```python
def build_run_paths(root, run_name):
    """由根目录和单段 run_name 生成稳定路径；不创建任何文件。"""
    if (
        not isinstance(run_name, str)
        or not run_name
        or run_name in {'.', '..'}
        or '/' in run_name
        or '\\' in run_name
    ):
        raise ValueError('run_name must be one safe, non-empty path segment')
    root = Path(root)
    run_root = root / run_name
    return {
        'run_root': run_root,
        'input': root / 'inputs' / 'train.jsonl',
        'cache': run_root / 'cache',
        'log': run_root / 'logs' / 'train.jsonl',
        'artifact': run_root / 'artifacts' / 'model.pt',
    }


paths = build_run_paths('runs', 'tinylm-seed42')
print({name: str(path) for name, path in paths.items()})
```

## Q3 验证：相同运行名是否产生相同路径？

```python
first = build_run_paths('runs', 'tinylm-seed42')
second = build_run_paths('runs', 'tinylm-seed42')
assert first == second
assert first['artifact'].as_posix().endswith('tinylm-seed42/artifacts/model.pt')
assert first['input'].as_posix() == 'runs/inputs/train.jsonl'

for invalid_name in ('', '.', '..', '../escape', 'nested/run', 'nested\\run'):
    try:
        build_run_paths('runs', invalid_name)
    except ValueError as error:
        assert 'path segment' in str(error)
    else:
        raise AssertionError(f'非法 run_name 应失败: {invalid_name!r}')
print('✅ 运行路径契约通过')
```

## Q4：怎样记录一次可复现运行的最小清单？

配置说明“怎么运行”，输入来源说明“处理什么”，artifact 指向“产出了什么”。`run_id` 由配置、随机种子和输入来源生成，使相同条件可以复算同一身份；artifact 路径作为输出位置单独记录。

```python
import hashlib


def build_run_manifest(config, seed, input_source, artifact_path):
    """生成最小运行清单；run_id 只依赖稳定运行条件。"""
    identity = {
        'config': config,
        'seed': seed,
        'input_source': str(input_source),
    }
    canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True)
    return {
        **identity,
        'artifact_path': str(artifact_path),
        'run_id': hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:12],
    }


paths = build_run_paths('runs', 'tinylm-seed42')
manifest = build_run_manifest(
    config={'model': 'TinyLM', 'batch_size': 4},
    seed=42,
    input_source=paths['input'],
    artifact_path=paths['artifact'],
)
print(json.dumps(manifest, ensure_ascii=False, indent=2))
```

## Q4 验证：相同条件与变化条件能否被区分？

```python
config = {'model': 'TinyLM', 'batch_size': 4}
first = build_run_manifest(config, 42, 'data/train.jsonl', 'artifacts/a.pt')
repeat = build_run_manifest(config, 42, 'data/train.jsonl', 'artifacts/a.pt')
changed_seed = build_run_manifest(config, 43, 'data/train.jsonl', 'artifacts/a.pt')

assert first['run_id'] == repeat['run_id']
assert first['run_id'] != changed_seed['run_id']
assert set(first) == {'config', 'seed', 'input_source', 'artifact_path', 'run_id'}
print('✅ 最小运行清单通过')
```

### 本节小结

- 配置覆盖后仍需检查运行所需字段。
- 路径和 `with` 分别负责文件定位与资源生命周期。
- 稳定路径区分输入、缓存、日志和 artifact。
- manifest 用配置、seed、输入来源和 `run_id` 标识一次可复查运行。
## 相关阅读

建立配置与运行身份后，可以继续进入张量输入、检查点和批次契约。

- [05. PyTorch Tensor Fundamentals | PyTorch 张量基础操作](./05_PyTorch_Tensor_Fundamentals.md)
- [10. PyTorch State, Checkpoint, and Recovery | PyTorch 状态、Checkpoint 与恢复](./10_PyTorch_State_dict_and_Persistence.md)
- [12. PyTorch Data Interface and Training Lifecycle | PyTorch 数据接口与训练生命周期](./12_PyTorch_Minimal_Training_Interface.md)
