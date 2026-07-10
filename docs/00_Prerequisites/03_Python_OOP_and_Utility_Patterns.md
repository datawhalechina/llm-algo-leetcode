# 03. Python OOP and Utility Patterns | Python 面向对象与工具模式

**难度：** Easy | **环境：** CPU-first | **标签：** `Python`, `对象封装`, `工具模式` | **目标人群：** Part 2-4 前置补课者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/03_Python_OOP_and_Utility_Patterns.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本页聚焦：会把配置和状态收进对象；会给对象补最小方法，让结果能直接看见；会判断什么时候该用 helper，什么时候该用 class。

**关键词：** `class`, `dataclass`, `config`

## 前置阅读
**导语：** 先看 0A 组页，把 Python 基础和对象封装的边界对齐，再进入这一页会更顺。
- [02. NumPy and Einsum | NumPy 与 Einsum](./02_NumPy_and_Einsum.md)
- [0A 组页](./0A.md)
- [28. Fault Tolerance and 检查点 | 容错与检查点](../01_Hardware_Math_and_Systems/28_Fault_Tolerance_and_检查点.md)

## 相关阅读
**导语：** 本页先把对象封装和工具模式的最小判断讲清楚；如果想继续把配置和 I/O 的写法补完整，可以顺着看下面这些页。
- [04. Python Config and I/O Patterns | Python 配置与 I/O 模式](./04_Python_Config_and_IO_Patterns.md)
- [33. TCO and Cost Model | TCO 与成本模型](../01_Hardware_Math_and_Systems/33_TCO_and_Cost_Model.md)

## Q1：什么时候该把配置收成对象？

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

## Q1验证：默认值和覆盖是否稳定？

这里直接检查两件事：默认值是否存在，修改字段后 `as_dict()` 是否还能正确反映更新后的状态。


```python
config = RunConfig()
assert config.lr == 1e-3
assert config.batch_size == 32
assert config.as_dict() == {'lr': 1e-3, 'batch_size': 32, 'epochs': 3}

config.batch_size = 64
print('覆盖后的配置：', config.as_dict())
assert config.as_dict()['batch_size'] == 64
print('✅ RunConfig 通过')

```

## Q2：什么时候该给对象补最小方法？

当对象不仅要存数据，还要统一输出、汇总或格式化时，给它补一个最小方法通常比到处写重复的打印代码更稳。


```python
@dataclass
class Metrics:
    loss: float = 0.0
    accuracy: float = 0.0

    def to_dict(self):
        return {'loss': self.loss, 'accuracy': self.accuracy}

    def summary(self):
        return f'loss={self.loss:.4f}, accuracy={self.accuracy:.2%}'


metrics = Metrics(loss=0.1234, accuracy=0.987)
print('Metrics 字典：', metrics.to_dict())
print('Metrics 摘要：', metrics.summary())

```

## Q2验证：对象输出是否统一？

这里要看的不是类定义得多不多，而是同一份状态能不能稳定输出成字典和摘要，方便后面日志、训练记录和实验汇总直接复用。


```python
metrics = Metrics(loss=0.2345, accuracy=0.9123)
assert metrics.to_dict() == {'loss': 0.2345, 'accuracy': 0.9123}
assert metrics.summary() == 'loss=0.2345, accuracy=91.23%'
print('✅ Metrics 通过')

```

## Q3：什么时候该把一次性 helper 收回 class？

如果函数开始反复传同一组状态、重复拼接输出、重复维护默认值，它通常已经接近对象边界了；这时把状态和方法收进 class 会更稳。


```python
from dataclasses import dataclass


def format_run_tag(name, epochs):
    return f'{name}@{epochs}'


@dataclass
class RunBuilder:
    name: str
    epochs: int

    def summary(self):
        return f'{self.name}@{self.epochs}|steps={self.epochs}'


tag = format_run_tag('TinyLM', 3)
builder = RunBuilder(name='TinyLM', epochs=3)
print('helper:', tag)
print('builder:', builder.summary())
assert tag == 'TinyLM@3'
assert builder.summary() == 'TinyLM@3|steps=3'
print('✅ helper_to_class 通过')

```
