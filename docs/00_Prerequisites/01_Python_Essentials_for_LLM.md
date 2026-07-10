# 01. Python Essentials for LLM | Python 基础与数据表示

**难度：** Easy | **环境：** CPU-first | **标签：** `Python`, `工程基础`, `数据结构` | **目标人群：** Part 2-4 前置补课者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/01_Python_Essentials_for_LLM.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本页聚焦：能用 `list`、`dict`、`class`、`with` 组织后续章节里的 Python 代码；能看懂并改写后续算法页中的基础片段。

**关键词：** `list`, `dict`, `class`

## 前置阅读
**导语：** 先看 0A 组页，把 Part 0 里 Python 基础页的边界和阅读顺序对齐，再进入这一页。
- [0A 组页](./0A.md)
- [01. Data Types and Precision | 大模型的数据格式与混合精度](../01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.md)

## 相关阅读
**导语：** 本页先把 Python 的最小工程写法讲清楚；如果想继续把张量、对象和 I/O 的写法补完整，可以顺着看下面这些页。
- [02. NumPy and Einsum | NumPy 与 Einsum](./02_NumPy_and_Einsum.md)
- [03. Python OOP and Utility Patterns | Python 面向对象与工具模式](./03_Python_OOP_and_Utility_Patterns.md)
- [04. Python Config and IO Patterns | Python 配置与 I/O 模式](./04_Python_Config_and_IO_Patterns.md)

## Q1：list / dict / class / with 各自负责什么？

后续章节里，Python 不是用来写长篇说明的，而是用来把配置、状态和结果放进合适的结构里。先把四个最常见的入口分清：`list` 装顺序数据，`dict` 装命名状态，`class` 装可复用对象，`with` 管资源收尾。


```python
from dataclasses import dataclass


@dataclass
class BatchSummary:
    loss: float
    step: int
    extras: dict

    def short(self):
        return f"step={self.step}, loss={self.loss:.4f}"


def merge_model_config(base_config, overrides):
    merged = base_config.copy()
    merged.update(overrides)
    return merged


def count_token_frequency(tokens):
    freq = {}
    for token in tokens:
        freq[token] = freq.get(token, 0) + 1
    return freq


print('BatchSummary 示例：', BatchSummary(loss=0.1234, step=12, extras={'lr': 1e-4}).short())
print('配置合并示例：', merge_model_config({'model_name': 'TinyLM', 'precision': 'bf16'}, {'precision': 'fp16'}))
print('token 统计示例：', count_token_frequency(['llm', 'llm', 'algo']))

```

## Q1验证：配置合并 + 状态封装

这里直接看两件事：配置能不能覆盖，状态对象能不能把摘要稳定输出。


```python
base = {'name': 'TinyLM', 'precision': 'bf16', 'batch_size': 4}
overrides = {'precision': 'fp16', 'batch_size': 8}
config = merge_model_config(base, overrides)
summary = BatchSummary(loss=0.1234, step=12, extras={'lr': 1e-4})
print('合并后的配置：', config)
print('摘要：', summary.short())
assert config == {'name': 'TinyLM', 'precision': 'fp16', 'batch_size': 8}
assert summary.short() == 'step=12, loss=0.1234'
print('✅ Q1 通过')

```

## Q2：什么时候该把零散变量收成对象？

如果一组字段会反复被读取、覆盖、打印或传递，通常就不该继续散在多个变量里。把它们收进对象后，默认值、更新方式和输出格式都会更稳定。


```python
from dataclasses import dataclass


@dataclass
class RunConfig:
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 3

    def as_dict(self):
        return {
            'lr': self.lr,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
        }


config = RunConfig()
print('RunConfig 默认值：', config.as_dict())

```

## Q2验证：默认值和覆盖是否稳定？

这里直接检查两件事：默认值是否存在，修改字段后 `as_dict()` 是否还能正确反映更新后的状态。


```python
config = RunConfig()
assert config.lr == 1e-3
assert config.batch_size == 32
assert config.as_dict() == {'lr': 1e-3, 'batch_size': 32, 'epochs': 3}

config.batch_size = 64
print('覆盖后的配置：', config.as_dict())
assert config.as_dict()['batch_size'] == 64
print('✅ Q2 通过')

```

## Q3：什么时候该用 with 管资源？

当你只是临时检查输出、保存中间结果或做一次性调试时，临时目录比手动创建和清理目录更稳，也更不容易残留脏文件。


```python
from pathlib import Path
from tempfile import TemporaryDirectory


with TemporaryDirectory() as tmpdir:
    root = Path(tmpdir)
    report = root / 'report.txt'
    report.write_text('hello io', encoding='utf-8')
    print('临时文件路径：', report)
    print('临时文件内容：', report.read_text(encoding='utf-8'))
print('✅ with 收尾完成')

```

## Q3验证：临时目录是否自动收尾？

这里不用手动删目录，只要看到 `with` 退出后资源不再需要继续维护，就说明 I/O 的生命周期已经对齐。


```python
with TemporaryDirectory() as tmpdir:
    root = Path(tmpdir)
    report = root / 'check.txt'
    report.write_text('done', encoding='utf-8')
    assert report.read_text(encoding='utf-8') == 'done'
    print('✅ 写入和读取通过')

```
