# 19. Debugging and Anomaly Localization | 调试与异常定位

**难度：** Medium | **环境：** CPU-first | **标签：** `调试`, `排错`, `张量健康` | **目标人群：** Part 2-4 前置补课者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/19_Debugging_and_Anomaly_Localization.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


当程序报错或训练结果异常时，最先看到的现象不一定是最接近原因的位置。一个 loss 变成 NaN，可能早在 activation 或 gradient 阶段就已经出现问题；一个 grad 为 None，也可能来自计算图断开、梯度追踪关闭或参数未注册。本节从异常现象开始，依次检查张量契约、数值状态和梯度链路，练习把问题缩小到可复现的最小位置。


**关键词：** `shape`, `dtype`, `device`, `NaN`

![异常分类与第一检查项](../public/00_Prerequisites/19_debugging_anomaly_map.svg)


## 前置阅读
**导语：** 先完成 18 的显存观察，再把 shape、dtype、device、数值和梯度问题分成可验证的故障类型。
- [18. Memory Profiling and Optimization | 显存分析与优化](./18_Memory_Profiling_and_Optimization.md)

## Q1：异常先归到哪几类，才能开始排查？

先从常见现象开始分类：维度不匹配通常是结构问题，CPU / CUDA 混用通常是设备问题，loss 变成 NaN 是数值问题，grad 为 None 则要检查梯度链路。分类不是结论，而是帮助你决定第一项检查；如果没有发现异常，应返回空列表，而不是强行指定一个原因。


| 异常类别 | 典型表现 | 第一项检查 |
|:---|:---|:---|
| shape | 维度不匹配、广播错误 | 输入和算子的 shape |
| dtype | 类型不兼容、精度异常 | 输入与参数 dtype |
| device / layout | CPU 与 CUDA 混用、访问路径异常 | device、layout、contiguous |
| numerical | NaN / Inf | 第一个非有限值 |
| gradient | grad 为 None 或梯度异常 | 计算图和参数注册 |


```python
def classify_issue(shape_ok, dtype_ok, device_ok, has_nonfinite, grad_ok):
    """收集异常类别；列表顺序表示教学上的初步检查顺序。"""
    issues = []
    if not shape_ok:
        issues.append('shape')
    if not dtype_ok:
        issues.append('dtype')
    if not device_ok:
        issues.append('device')
    if has_nonfinite:
        issues.append('numerical')
    if not grad_ok:
        issues.append('gradient')
    return issues


sample = classify_issue(shape_ok=False, dtype_ok=True, device_ok=False, has_nonfinite=True, grad_ok=False)
primary = sample[0] if sample else 'ok'
print('issues:', sample)
print('primary:', primary)

# 输出示例: issues -> ['shape', 'device', 'numerical', 'gradient']; primary -> shape

```

## Q2：什么时候先看 shape、dtype、device 和 layout？

遇到报错时，先按由易到难的顺序检查：shape 决定张量的维度和广播方式，dtype 决定数据类型是否兼容，device 决定模型和输入是否在同一计算位置，layout / contiguous 则影响算子能否直接访问这块内存。契约没有对齐前，先不要急着改优化策略；这里用一个 CPU Tensor 演示这些字段如何被检查。


```python
import torch


def audit_tensor_contract(actual, expected):
    """逐项比较张量契约，并返回首个需要先修复的字段。"""
    report = {}
    report['shape_ok'] = actual['shape'] == expected['shape']
    report['dtype_ok'] = actual['dtype'] == expected['dtype']
    report['device_ok'] = actual['device'] == expected['device']
    report['contiguous_ok'] = actual['contiguous'] == expected['contiguous']
    first_failure = next((name for name, ok in report.items() if not ok), None)
    return {'checks': report, 'first_failure': first_failure}


tensor = torch.randn(2, 4, 8)
view = tensor.transpose(1, 2)
actual = {
    'shape': tuple(view.shape),
    'dtype': str(view.dtype),
    'device': str(view.device),
    'contiguous': view.is_contiguous(),
}
expected = {'shape': (2, 8, 4), 'dtype': 'torch.float32', 'device': 'cpu', 'contiguous': True}
report = audit_tensor_contract(actual, expected)
print('contract:', report)
print('must_fix:', [k for k, v in report['checks'].items() if not v])

# 输出示例: contract 中 contiguous_ok=False, must_fix=['contiguous_ok']

```

## Q3：什么时候先查 NaN / Inf，如何找到第一个异常来源？

如果 loss 最后变成 NaN，真正的问题可能更早出现在 activation 或梯度中。沿着 loss → activation → gradient 的计算顺序检查，并记录第一个 NaN / Inf 的阶段和对象；第一个异常位置通常比最后看到的 NaN 更接近原因。这里的记录已经按执行顺序排列，函数只负责定位，不负责解释根因。


```python
def first_nonfinite_step(records):
    """返回第一个 NaN、正无穷或负无穷所在的阶段和位置。"""
    if not isinstance(records, list):
        raise TypeError('records 必须是按执行顺序排列的列表')
    for idx, (source, value) in enumerate(records):
        if value != value or value == float("inf") or value == float("-inf"):
            return {'index': idx, 'source': source, 'value': value}
    return None


trace = [('loss', 1.0), ('activation', 0.9), ('gradient', 0.7), ('gradient', float("nan")), ('gradient', 0.2)]
failure = first_nonfinite_step(trace)
print('trace:', trace)
print('first_nonfinite_step:', failure)
print('source:', failure['source'] if failure else 'ok')

# 输出示例：first_nonfinite_step -> 3, source -> gradient

```

## Q4：梯度为什么会一直是 `None`，如何区分断开的原因？

“grad 为 None”不一定只有一个原因：可能是 detach() 切断了当前 Tensor，可能是 no_grad() 关闭了梯度追踪，也可能是非叶子 Tensor 默认不保留梯度，或参数根本没有注册进模块。还要区分“grad 为 None”和“grad 的数值为 0”：前者表示没有生成或没有保留梯度，后者表示梯度已经存在，只是当前值为零。先根据出现位置判断是哪一种，再决定检查计算图、梯度属性还是模块参数。


```python
import warnings
import torch.nn as nn

def diagnose_none_grad(case):
    """把梯度为空的现象映射回可能的计算图原因。"""
    mapping = {
        'detach': 'tensor disconnected from graph',
        'no_grad': 'gradient tracking disabled',
        'leaf': 'leaf tensor not requiring grad',
        'unregistered': 'parameter not registered in module',
    }
    if case not in mapping:
        raise ValueError(f'未知的梯度异常类型: {case}')
    return mapping[case]


cases = ['detach', 'no_grad', 'leaf', 'unregistered']
for case in cases:
    print(case + ':', diagnose_none_grad(case))

leaf = torch.tensor([1.0], requires_grad=True)
detached = leaf.detach()
with torch.no_grad():
    no_grad_value = leaf * 2
non_leaf = leaf * 3
non_leaf.sum().backward()

class BrokenModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_weight = torch.randn(2, 2, requires_grad=True)

broken = BrokenModule()
assert detached.grad_fn is None
assert not no_grad_value.requires_grad
with warnings.catch_warnings():
    warnings.simplefilter('ignore', UserWarning)
    non_leaf_grad = non_leaf.grad
assert non_leaf_grad is None
assert not any(name == 'raw_weight' for name, _ in broken.named_parameters())
print('✅ detach、no_grad、非叶子 Tensor 和未注册参数机制通过')

# 输出示例: detach/no_grad/leaf/unregistered 对应各自的原因
print('next_check:', '先复现对应 case，再检查计算图或模块注册')

```

## Q5：多个异常同时出现时，排查顺序怎么定才最省事？

先做最便宜、影响面最大的结构性检查，再查数值，最后查梯度链路：shape / dtype / device → NaN / Inf → gradient。这样可以先排掉输入契约问题，再判断是否需要修改模型或训练逻辑；这一步关注的是排查成本和优先级，不是重新分类异常。


```python
def debug_priority(shape_ok, dtype_ok, device_ok, has_nonfinite, grad_ok):
    """按结构、数值、梯度的顺序返回首个待检查项和排查原因。"""
    checks = [
        ('shape', shape_ok),
        ('dtype', dtype_ok),
        ('device', device_ok),
        ('numerical', not has_nonfinite),
        ('gradient', grad_ok),
    ]
    for rank, (name, ok) in enumerate(checks, start=1):
        if not ok:
            reasons = {
                'shape': '先检查输入和算子的维度契约',
                'dtype': '再检查输入与参数的数据类型',
                'device': '确认张量和模型位于同一设备',
                'numerical': '定位第一个 NaN 或 Inf',
                'gradient': '最后检查计算图和参数注册',
            }
            return {'primary': name, 'rank': rank, 'checked': [k for k, _ in checks[:rank]], 'reason': reasons[name]}
    return {'primary': 'ok', 'rank': None, 'checked': [k for k, _ in checks], 'reason': '当前没有发现异常'}


report = debug_priority(False, True, False, True, False)
print('report:', report)
# 输出示例: primary -> shape, rank -> 1, checked -> ['shape']

```

## Q6：如何定位 loss、activation、gradient 中最先失败的阶段？

Q3 找到的是记录中的第一个非有限值，Q6 要进一步回答它出现在训练流程的哪个阶段，并给出下一步动作。依次检查 loss、activation、gradient；如果失败点落在 gradient，再回到 Q4 区分是 detach、no_grad、非叶子 Tensor 还是参数注册导致的计算图断开。


```python
def locate_failure_zone(records):
    """定位第一个失败阶段，并给出下一项排查动作。"""
    next_checks = {
        'loss': 'inspect_loss_inputs_and_numeric_range',
        'activation': 'inspect_forward_activation',
        'gradient': 'return_to_Q4_gradient_chain',
    }
    for idx, record in enumerate(records, start=1):
        if not record['finite']:
            zone = record['stage']
            return {'zone': zone, 'step': idx, 'next_check': next_checks[zone]}
    return {'zone': 'ok', 'step': None, 'next_check': 'continue_monitoring'}


for records in [
    [{'stage': 'loss', 'finite': False}, {'stage': 'activation', 'finite': True}, {'stage': 'gradient', 'finite': True}],
    [{'stage': 'loss', 'finite': True}, {'stage': 'activation', 'finite': False}, {'stage': 'gradient', 'finite': False}],
    [{'stage': 'loss', 'finite': True}, {'stage': 'activation', 'finite': True}, {'stage': 'gradient', 'finite': False}],
]:
    print('records:', records, '->', locate_failure_zone(records))
# 输出示例：zone、step 和 next_check 一起返回

```

## 相关阅读
**导语：** 完成本节后，可以继续学习 Autograd 的梯度语义，并把异常定位结果带入显存策略和真实项目验证。
- [PyTorch Autograd 官方文档](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- [PyTorch 异常检测官方文档](https://docs.pytorch.org/docs/stable/autograd.html#debugging-and-anomaly-detection)
- [20. Profiling and Memory Ledger | 性能剖析与显存账本](./20_Profiling_and_Memory_Ledger.md)
- [76. Activation / Checkpoint / Offload Benchmark | 激活检查点与卸载对比](../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.md)
