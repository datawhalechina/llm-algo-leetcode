# 03. Python OOP and Utility Patterns | Python 面向对象与工具模式

**难度：** Easy | **环境：** CPU-first | **标签：** `Python`, `对象封装`, `状态管理` | **目标人群：** Part 2-4 前置补课者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/03_Python_OOP_and_Utility_Patterns.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

当一组字段共享约束并随调用更新时，零散变量难以保证状态始终一致。对象的价值在于把数据、约束和生命周期放到同一边界内。

完成本节后，你应能判断何时需要有状态对象，并检查初始化、更新、查询和重置是否共同维护内部不变量。

**关键词：** `dataclass`, `state`, `invariant`, `lifecycle`

![Python 对象与状态管理](../public/00_Prerequisites/03_python_state_object_map.svg)

## 前置阅读
**导语：** 先掌握 Python 数据结构和函数契约，再判断哪些状态应该收进对象。
- [01. Python Essentials for LLM | Python 基础与数据表示](./01_Python_Essentials_for_LLM.md)
- [0A 组页](./0A.md)

## Q1：什么时候对象内部字段需要共同维护不变量？

只把字段放进 dataclass 还不够。字段之间存在共同约束时，应在对象创建阶段校验，使对象一旦创建就保持合法。下面通过初始化校验、状态更新和查询操作维护对象内部契约。

```python
from dataclasses import asdict, dataclass, field


@dataclass
class TrainingOptions:
    """保存一组相关选项，并在创建时维护字段不变量。"""

    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 3

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError('learning_rate must be positive')
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError('batch_size and epochs must be positive')

    def as_dict(self):
        return asdict(self)


options = TrainingOptions(batch_size=16)
print(options.as_dict())
```

## Q1 验证：合法配置与非法配置能否在入口区分？

```python
options = TrainingOptions(learning_rate=2e-4, batch_size=8, epochs=2)
assert options.as_dict()['batch_size'] == 8

try:
    TrainingOptions(batch_size=0)
except ValueError as error:
    assert 'batch_size' in str(error)
else:
    raise AssertionError('非法 batch_size 应触发 ValueError')
print('✅ 对象内部字段不变量通过')
```

## Q2：为什么持续更新的指标适合用对象管理？

带权指标均值依赖累计总和和总权重，这些状态必须一起更新、查询和重置。把它们放进同一对象，可以避免调用方忘记同步其中一个变量。

```python
@dataclass
class MetricTracker:
    """累计带权指标；total 和 weight_sum 必须同步更新。"""

    total: float = 0.0
    weight_sum: float = 0.0

    def update(self, value, *, weight=1):
        if weight <= 0:
            raise ValueError('weight must be positive')
        self.total += float(value) * weight
        self.weight_sum += float(weight)

    @property
    def mean(self):
        return self.total / self.weight_sum if self.weight_sum else 0.0

    def reset(self):
        self.total = 0.0
        self.weight_sum = 0.0


tracker = MetricTracker()
tracker.update(0.4, weight=2)
tracker.update(0.1)
print('mean:', tracker.mean)
```

## Q2 验证：update、mean 和 reset 是否共享同一状态？

```python
tracker = MetricTracker()
tracker.update(0.4, weight=2)
tracker.update(0.1)
assert abs(tracker.mean - 0.3) < 1e-12
tracker.reset()
assert tracker.total == 0.0 and tracker.weight_sum == 0.0 and tracker.mean == 0.0

try:
    tracker.update(0.2, weight=0)
except ValueError as error:
    assert 'weight' in str(error)
else:
    raise AssertionError('非正权重应触发 ValueError')
print('✅ 指标状态通过')
```

## Q3：什么时候 helper 应升级为有状态对象？

一次格式化使用 helper 即可；当调用需要持续维护 step、历史记录和最新状态时，应使用对象承载生命周期。下面的 `RunState` 每次记录指标都会推进 step，并保留可查询历史。

```python
def format_metric(step, loss):
    """无状态 helper：只格式化一次输入。"""
    return f'step={step}, loss={loss:.4f}'


@dataclass
class RunState:
    """维护跨调用的 step 与指标事件历史。"""

    step: int = 0
    history: list = field(default_factory=list)

    def record(self, loss):
        self.step += 1
        event = {'step': self.step, 'loss': float(loss)}
        self.history.append(event)
        return event

    @property
    def latest(self):
        return self.history[-1] if self.history else None


state = RunState()
state.record(0.5)
state.record(0.25)
print(format_metric(**state.latest))
```

## Q3 验证：对象是否正确维护跨调用状态？

```python
state = RunState()
assert state.latest is None
first = state.record(0.5)
second = state.record(0.25)
assert first == {'step': 1, 'loss': 0.5}
assert second == {'step': 2, 'loss': 0.25}
assert len(state.history) == 2
assert format_metric(**state.latest) == 'step=2, loss=0.2500'
print('✅ helper 与有状态对象边界通过')
```

## 相关阅读

理解对象状态后，继续把配置、路径和运行产物组织成可复现入口。

- [04. Python Config and Data Entry | Python 配置与数据入口](./04_Python_Config_and_Data_Entry.md)
- [09. PyTorch nn.Module Basics | PyTorch nn.Module 基础](./09_PyTorch_nn_Module_Basics.md)
