# 19. Debugging and Anomaly Localization | 调试与异常定位

**难度：** Medium | **环境：** CPU-first | **标签：** `调试`, `排错`, `张量健康` | **目标人群：** Part 2-4 前置补课者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/19_Debugging_and_Anomaly_Localization.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本页聚焦异常定位的最小判断链：先分问题类型，再看 shape、dtype、device、数值异常和梯度边界，不把排错写成经验罗列。

**关键词：** `shape`, `dtype`, `device`, `NaN`

## 前置阅读
**导语：** 先看 0E 组页，把调试和性能排查的边界对齐，再进入这一页会更顺。
- [18. Memory Profiling and Optimization | 显存分析与优化](./18_Memory_Profiling_and_Optimization.md)
- [0E 组页](./0E.md)
- [13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)

## 相关阅读
**导语：** 如果想把异常定位接上 Part 1 的性能和硬件视角，可以顺着看下面这一页。
- [03. GPU Architecture and Memory | GPU 物理架构、内存层级与核心硬件单元](../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.md)

## Q1：异常先归到哪几类？

先按 shape、dtype、device、数值异常、梯度异常这五类去分，分类清楚了，排查顺序才不会乱。


```python
def classify_issue(shape_ok, dtype_ok, device_ok, has_nonfinite, grad_ok):
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
print('issues:', sample)
print('primary:', sample[0])

# 输出示例: issues -> ['shape', 'device', 'numerical', 'gradient']; primary -> shape

```

## Q2：什么时候先看 shape、dtype、device？

只要输入和算子的契约不明确，就先别看优化，先把 shape、dtype、device 和布局对齐。


```python
def audit_tensor_contract(actual, expected):
    report = {}
    report['shape_ok'] = actual['shape'] == expected['shape']
    report['dtype_ok'] = actual['dtype'] == expected['dtype']
    report['device_ok'] = actual['device'] == expected['device']
    report['contiguous_ok'] = actual['contiguous'] == expected['contiguous']
    return report


actual = {'shape': (2, 4, 8), 'dtype': 'float16', 'device': 'cuda', 'contiguous': False}
expected = {'shape': (2, 4, 8), 'dtype': 'float16', 'device': 'cuda', 'contiguous': True}
report = audit_tensor_contract(actual, expected)
print('contract:', report)
print('must_fix:', [k for k, v in report.items() if not v])

# 输出示例: contract 中 contiguous_ok=False, must_fix=['contiguous_ok']

```

## Q3：什么时候先查 NaN / Inf？

一旦 loss、激活或梯度开始出现非有限值，就先找第一个出现问题的环节，不要直接跳到模型结构上。


```python
def first_nonfinite_step(values):
    for idx, value in enumerate(values):
        if value != value or value == float("inf") or value == float("-inf"):
            return idx
    return None


trace = [1.0, 0.9, 0.7, float("nan"), 0.2]
idx = first_nonfinite_step(trace)
print('trace:', trace)
print('first_nonfinite_step:', idx)
print('source:', 'loss' if idx == 3 else 'ok')

# 输出示例: first_nonfinite_step -> 3, source -> loss

```

## Q4：梯度为什么会一直是 `None`？

先分清是 `detach`、`no_grad`、叶子节点未保留，还是参数根本没有注册进模块。


```python
def diagnose_none_grad(case):
    mapping = {
        'detach': 'tensor disconnected from graph',
        'no_grad': 'gradient tracking disabled',
        'leaf': 'leaf tensor not requiring grad',
        'unregistered': 'parameter not registered in module',
    }
    return mapping[case]


cases = ['detach', 'no_grad', 'leaf', 'unregistered']
for case in cases:
    print(case + ':', diagnose_none_grad(case))

# 输出示例: detach/no_grad/leaf/unregistered 对应各自的原因

```

## Q5：shape 错、dtype 错、device 错，排查顺序怎么定？

先把最便宜的结构性检查做完，再去看数值和梯度，不要一上来就改模型。


```python
def debug_priority(shape_ok, dtype_ok, device_ok, has_nonfinite, grad_ok):
    checks = [
        ('shape', shape_ok),
        ('dtype', dtype_ok),
        ('device', device_ok),
        ('numerical', not has_nonfinite),
        ('gradient', grad_ok),
    ]
    for rank, (name, ok) in enumerate(checks, start=1):
        if not ok:
            return {'primary': name, 'rank': rank, 'checked': [k for k, _ in checks[:rank]]}
    return {'primary': 'ok', 'rank': None, 'checked': [k for k, _ in checks]}


report = debug_priority(False, True, False, True, False)
print('report:', report)
# 输出示例: primary -> shape, rank -> 1, checked -> ['shape']

```

## Q6：什么时候先查 NaN / Inf，什么时候先查梯度边界？

只要损失或激活出现非有限值，就先找第一个出问题的环节；如果值是有限的但梯度异常，再看梯度边界。


```python
def locate_failure_zone(loss_ok, activation_ok, grad_ok):
    stages = [
        ('loss', loss_ok),
        ('activation', activation_ok),
        ('gradient', grad_ok),
    ]
    for idx, (name, ok) in enumerate(stages, start=1):
        if not ok:
            return {'zone': name, 'step': idx}
    return {'zone': 'ok', 'step': None}


for case in [(False, True, True), (True, False, True), (True, True, False)]:
    print('case:', case, '->', locate_failure_zone(*case))
# 输出示例: 最先失败的 zone 和 step 会一起返回

```
