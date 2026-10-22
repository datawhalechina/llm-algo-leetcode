# 01. Python Essentials for LLM | Python 基础与数据表示

**难度：** Easy | **环境：** CPU-first | **标签：** `Python`, `工程基础`, `数据结构` | **目标人群：** Part 2-4 前置补课者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/01_Python_Essentials_for_LLM.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

后续教程会反复使用有序数据、固定维度和命名字段传递 token、形状、配置与指标。数据结构选错后，字段含义、可变性和输入边界会变得含糊。

完成本节后，你应能按顺序、可变性和字段访问方式选择数据结构，并用输入输出契约尽早识别缺失字段与非法输入。

**关键词：** `list`, `tuple`, `dict`, `function contract`

![Python 数据与函数契约](../public/00_Prerequisites/01_python_contract_stream_map.svg)

## 前置阅读
**导语：** 本页是 Part 00 的代码入口；具备基础 Python 经验时，可以直接从问题和验证代码开始。
- [0A 组页](./0A.md)

## Q1：list、tuple 和 dict 分别适合表达什么？

`list` 适合会增删或按顺序处理的样本，`tuple` 适合固定含义的 shape，`dict` 适合按名称访问的配置和指标。选择数据结构时先看信息是否有顺序、是否固定、是否需要按字段读取。

```python
def describe_batch(tokens, shape, metrics):
    """汇总 token 序列、固定 shape 和命名指标。"""
    return {
        'token_count': len(tokens),
        'unique_tokens': len(set(tokens)),
        'shape': tuple(shape),
        'loss': metrics['loss'],
    }


tokens = ['llm', 'infra', 'llm']
shape = (2, 3, 4)
metrics = {'loss': 0.25, 'learning_rate': 1e-3}
print(describe_batch(tokens, shape, metrics))
```

## Q1 验证：三类数据是否保持各自语义？

这里检查顺序、固定维度和命名字段没有在汇总过程中混淆。

```python
summary = describe_batch(['a', 'b', 'a'], (2, 4), {'loss': 0.5})
assert summary == {
    'token_count': 3,
    'unique_tokens': 2,
    'shape': (2, 4),
    'loss': 0.5,
}
assert describe_batch([], (0, 4), {'loss': 0.0})['token_count'] == 0
print('✅ Python 数据结构通过')
```

## Q2：函数怎样形成可复用的输入输出契约？

函数签名应区分必需输入和可选策略。下面把 `batch_size` 与 `drop_last` 设为关键字参数，使调用位置直接表达含义；返回字典则保留完整 batch 数、余数和执行步数，避免只返回一个难以解释的数字。

```python
def build_batch_plan(sample_count, *, batch_size=4, drop_last=False):
    """根据非负样本数计算 batch 计划；返回完整批、余数和执行步数。"""
    if not isinstance(sample_count, int) or sample_count < 0:
        raise ValueError('sample_count must be a non-negative integer')
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')

    full_batches, remainder = divmod(sample_count, batch_size)
    steps = full_batches if drop_last or remainder == 0 else full_batches + 1
    return {
        'full_batches': full_batches,
        'remainder': remainder,
        'steps': steps,
        'drop_last': drop_last,
    }


print(build_batch_plan(10, batch_size=4))
```

## Q2 验证：默认策略、边界和非法输入是否清楚？

验证既覆盖有余数和丢弃尾批，也确认非法 batch size 会立即失败。

```python
assert build_batch_plan(10, batch_size=4)['steps'] == 3
assert build_batch_plan(10, batch_size=4, drop_last=True)['steps'] == 2
assert build_batch_plan(0, batch_size=4)['steps'] == 0

try:
    build_batch_plan(8, batch_size=0)
except ValueError as error:
    assert 'batch_size' in str(error)
else:
    raise AssertionError('非法 batch_size 应触发 ValueError')
print('✅ 函数契约通过')
```

## Q3：为什么要在数据入口校验必需字段？

配置、样本和结果通常以 dict 传递。越早检查字段和类型，越容易把错误定位在输入入口，而不是等到模型内部才出现难以理解的 KeyError 或 shape 错误。校验函数还应返回新对象，避免意外修改调用方的数据。

```python
def validate_record(record, required_fields):
    """检查命名字段并返回浅拷贝，不修改输入 record。"""
    if not isinstance(record, dict):
        raise TypeError('record must be a dict')
    missing = [field for field in required_fields if field not in record]
    if missing:
        raise KeyError(f'missing required fields: {missing}')
    return record.copy()


sample = {'input_ids': [1, 2, 3], 'labels': [1, 2, 3]}
validated = validate_record(sample, ('input_ids', 'labels'))
validated['source'] = 'toy'
print(validated)
```

## Q3 验证：缺失字段能否被准确定位？

这里同时检查正常输入、输入不被修改，以及缺失字段的失败信息。

```python
sample = {'input_ids': [1, 2], 'labels': [1, 2]}
validated = validate_record(sample, ('input_ids', 'labels'))
assert validated == sample and validated is not sample

try:
    validate_record({'input_ids': [1, 2]}, ('input_ids', 'labels'))
except KeyError as error:
    assert 'labels' in str(error)
else:
    raise AssertionError('缺失 labels 应触发 KeyError')
print('✅ 输入字段校验通过')
```

## Q4：怎样逐条处理样本而不一次性复制全部数据？

列表适合立即得到全部结果；生成器适合按需读取较大的样本流。生成器仍应复用入口校验，并在失败信息中标出样本位置，避免惰性执行把错误来源隐藏起来。

```python
def iter_valid_records(records, required_fields):
    """逐条生成校验后的浅拷贝；非法样本会报告其顺序位置。"""
    for position, record in enumerate(records):
        try:
            yield validate_record(record, required_fields)
        except (TypeError, KeyError) as error:
            raise ValueError(f'record {position}: {error}') from error


records = [
    {'input_ids': [1, 2], 'labels': [1, 2]},
    {'input_ids': [3], 'labels': [3]},
]
record_stream = iter_valid_records(records, ('input_ids', 'labels'))
print(list(record_stream))
```

## Q4 验证：生成器是否惰性执行，并能定位非法样本？

创建生成器时不会立即遍历输入；消费结果后才执行字段校验。这里同时检查输出副本和带位置的失败信息。

```python
records = [
    {'input_ids': [1], 'labels': [1]},
    {'input_ids': [2], 'labels': [2]},
]
record_stream = iter_valid_records(records, ('input_ids', 'labels'))
assert iter(record_stream) is record_stream
validated_records = list(record_stream)
assert validated_records == records
assert all(validated is not original for validated, original in zip(validated_records, records))

invalid_records = [
    {'input_ids': [1], 'labels': [1]},
    {'input_ids': [2]},
]
try:
    list(iter_valid_records(invalid_records, ('input_ids', 'labels')))
except ValueError as error:
    assert 'record 1' in str(error) and 'labels' in str(error)
else:
    raise AssertionError('非法样本应在被消费时触发带位置的 ValueError')
print('✅ 惰性样本处理通过')
```

## 相关阅读

掌握数据结构和函数契约后，继续学习数组维度、对象状态与实验配置。

- [02. NumPy and Einsum | NumPy 与 Einsum](./02_NumPy_and_Einsum.md)
- [03. Python OOP and Utility Patterns | Python 面向对象与工具模式](./03_Python_OOP_and_Utility_Patterns.md)
- [04. Python Config and Data Entry | Python 配置与数据入口](./04_Python_Config_and_Data_Entry.md)
