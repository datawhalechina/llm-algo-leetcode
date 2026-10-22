# 05. PyTorch Tensor Fundamentals | PyTorch 张量基础操作

**难度：** Easy | **环境：** CPU-first | **标签：** `PyTorch`, `Tensor`, `dtype`, `device` | **目标人群：** 从 NumPy 进入 PyTorch 的学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/05_PyTorch_Tensor_Fundamentals.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

张量在数组之上增加了数据类型、设备、梯度和存储语义。形状相同的张量，仍可能因为数据类型、设备或共享关系不同而产生不同结果。

完成本节后，你应能读取张量的基本契约，区分复制与共享存储，并判断数值张量和索引张量是否满足输入要求。

**关键词：** `tensor`, `dtype`, `device`, `storage sharing`

![Tensor 基础概念关系](../public/00_Prerequisites/05_tensor_fundamentals_map.svg)

## 前置阅读
**导语：** 先掌握 NumPy 的 shape 和广播，再观察 Tensor 新增的 dtype、device 与存储语义。
- [02. NumPy and Einsum | NumPy 与 Einsum](./02_NumPy_and_Einsum.md)
- [0B 组页](./0B.md)

## Q1：如何读取 Tensor 的 shape、dtype 与 device？

shape 描述结构，dtype 描述数值解释和单元素大小，device 描述数据所在计算位置。检查输入时应同时记录这三项，而不是只看打印出的数值。

```python
import numpy as np
import torch


def describe_tensor(tensor):
    """返回 Tensor 的结构、数值类型、设备和单元素字节数。"""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('expected a torch.Tensor')
    return {
        'shape': tuple(tensor.shape),
        'dtype': str(tensor.dtype).replace('torch.', ''),
        'device': str(tensor.device),
        'element_size_bytes': tensor.element_size(),
        'requires_grad': tensor.requires_grad,
    }


tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print(describe_tensor(tensor))
```

## Q1 验证：结构与数值属性是否完整？

```python
tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
description = describe_tensor(tensor)
assert description == {
    'shape': (2, 2),
    'dtype': 'float32',
    'device': 'cpu',
    'element_size_bytes': 4,
    'requires_grad': False,
}

try:
    describe_tensor(np.zeros(2))
except TypeError as error:
    assert 'Tensor' in str(error)
else:
    raise AssertionError('非 Tensor 输入应触发 TypeError')
print('✅ Tensor 属性通过')
```

## Q2：哪些构造方式会共享底层数据？

`torch.tensor(array)` 会复制输入；`torch.from_numpy(array)` 与 CPU 上兼容 dtype 的 `torch.as_tensor(array)` 通常共享 NumPy 内存；`clone()` 创建独立 Tensor。是否共享决定修改一侧时另一侧会不会变化，因此转换后不能只检查数值相等。

```python
array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
copied = torch.tensor(array)
from_numpy = torch.from_numpy(array)
as_tensor = torch.as_tensor(array)
cloned = from_numpy.clone()

array[0] = 99.0
print({
    'numpy': array.tolist(),
    'torch.tensor': copied.tolist(),
    'from_numpy': from_numpy.tolist(),
    'as_tensor': as_tensor.tolist(),
    'clone': cloned.tolist(),
})
```

## Q2 验证：共享与复制关系是否符合预期？

```python
array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
copied = torch.tensor(array)
shared = torch.from_numpy(array)
wrapped = torch.as_tensor(array)
cloned = shared.clone()

array[1] = 77.0
assert copied[1].item() == 2.0
assert shared[1].item() == 77.0
assert wrapped[1].item() == 77.0
assert cloned[1].item() == 2.0
print('✅ Tensor 复制与共享语义通过')
```

## Q3：dtype 或 device 转换什么时候会复用原 Tensor？

`.to()` 在 dtype 和 device 都不变时通常返回原 Tensor；发生类型或设备变化时会创建新存储，`copy=True` 则可以显式要求复制。转换必须保持 shape 和数值语义，真实设备迁移还需要目标运行环境可用。

```python
def convert_tensor(tensor, *, dtype=None, device=None, copy=False):
    """转换 Tensor 的 dtype/device；未变化时允许复用，copy=True 强制复制。"""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('expected a torch.Tensor')
    target_dtype = tensor.dtype if dtype is None else dtype
    target_device = tensor.device if device is None else device
    return tensor.to(dtype=target_dtype, device=target_device, copy=copy)


source = torch.tensor([1.0, 2.0], dtype=torch.float32)
reused = convert_tensor(source)
converted = convert_tensor(source, dtype=torch.float64)
forced_copy = convert_tensor(source, copy=True)
print({
    'reused_object': reused is source,
    'converted_dtype': str(converted.dtype),
    'forced_copy_storage': forced_copy.data_ptr() != source.data_ptr(),
})
```

## Q3 验证：复用、类型转换和强制复制是否保持各自语义？

```python
source = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
reused = convert_tensor(source)
converted = convert_tensor(source, dtype=torch.float64)
forced_copy = convert_tensor(source, copy=True)

assert reused is source
assert converted.shape == source.shape and converted.dtype == torch.float64
assert converted.data_ptr() != source.data_ptr()
assert forced_copy.shape == source.shape and forced_copy.data_ptr() != source.data_ptr()
assert torch.equal(forced_copy, source)

try:
    convert_tensor([1.0, 2.0])
except TypeError as error:
    assert 'Tensor' in str(error)
else:
    raise AssertionError('非 Tensor 输入应触发 TypeError')
print('✅ dtype/device 转换语义通过')
```

## Q4：数值 Tensor 与索引 Tensor 的 dtype/device 契约有什么不同？

浮点 Tensor 参与模型计算；Embedding、查表和位置选择通常要求 `torch.long` 索引。参与同一算子的 Tensor 还要位于兼容 device。入口函数可以显式转换索引 dtype 和 device，但不应静默接受浮点小数索引。

```python
def prepare_indices(indices, *, device):
    """将整数索引放到目标 device，并统一为 torch.long。"""
    indices = torch.as_tensor(indices)
    if indices.dtype.is_floating_point or indices.dtype == torch.bool:
        raise TypeError('indices must contain integers')
    return indices.to(device=device, dtype=torch.long)


embedding_table = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)
indices = prepare_indices([3, 1, 4], device=embedding_table.device)
selected = embedding_table.index_select(0, indices)
print('indices:', indices.dtype, indices.device)
print('selected shape:', tuple(selected.shape))
```

## Q4 验证：索引 dtype、device 与输出 shape 是否正确？

```python
table = torch.arange(20, dtype=torch.float32).reshape(5, 4)
indices = prepare_indices(np.array([3, 1], dtype=np.int32), device=table.device)
selected = table.index_select(0, indices)
assert indices.dtype == torch.long
assert indices.device == table.device
assert selected.shape == (2, 4)
assert torch.equal(selected[0], table[3])

try:
    prepare_indices([1.5, 2.0], device=table.device)
except TypeError as error:
    assert 'integers' in str(error)
else:
    raise AssertionError('浮点索引应触发 TypeError')
print('✅ dtype、device 与索引契约通过')
```

### 本节小结

- shape、dtype、device 和单元素字节数共同描述 Tensor 的基本契约。
- `torch.tensor`、`from_numpy`、`as_tensor` 与 `clone` 的复制/共享语义不同。
- `.to()` 在条件不变时可以复用原 Tensor，类型或设备变化会形成新表示。
- 索引 Tensor 应使用整数 dtype，并与被索引对象位于兼容设备。
## 相关阅读

掌握张量入口后，继续学习布局、索引和反向传播。

- [PyTorch Tensor 官方文档](https://pytorch.org/docs/stable/tensors.html)
- [06. PyTorch Tensor Layout and Indexing | PyTorch 张量布局与索引](./06_PyTorch_Tensor_Layout_and_Indexing.md)
- [07. PyTorch Autograd and Backward | PyTorch 自动求导与反向传播](./07_PyTorch_Autograd_and_Backward.md)
