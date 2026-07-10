# 06. PyTorch Tensor Layout and Indexing | PyTorch 张量布局与索引

**难度：** Easy | **环境：** CPU-first | **标签：** `PyTorch`, `布局`, `索引` | **目标人群：** Part 2-4 前置补课者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/06_PyTorch_Tensor_Layout_and_Indexing.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本页聚焦：读懂 Tensor 的维度、步长和连续性；掌握切片、广播和维度变换的最小判断；为后面的 Attention、KV cache 和 layout 讨论打底。

**关键词：** `shape`, `stride`, `contiguous`

## 前置阅读
**导语：** 先看 0B 组页，把 Tensor 的形状和 layout 边界对齐，再进入这一页会更顺。
- [05. PyTorch Tensor Fundamentals | PyTorch 张量基础操作](./05_PyTorch_Tensor_Fundamentals.md)
- [0B 组页](./0B.md)
- [03. GPU Architecture and Memory | GPU 物理架构、内存层级与核心硬件单元](../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.md)

## 相关阅读
**导语：** 本页先把 layout、索引和连续性的最小判断讲清楚；如果想继续看自动求导和 attention 显存视角，再顺着看下面这一页。
- [07. PyTorch Autograd and Backward | PyTorch 自动求导与反向传播](./07_PyTorch_Autograd_and_Backward.md)
- [04. Attention Memory Optimization | Attention 显存优化](../01_Hardware_Math_and_Systems/04_Attention_Memory_Optimization.md)

## Q1：shape、stride 和 contiguous 分别解决什么问题？

看到一个 Tensor 时，先别急着看值，先看它的 shape、stride 和是否连续。后面只要涉及 layout、view 或切片，这三个信息就决定你能不能直接继续算。


```python
import torch


def describe_layout(x):
    return {
        'shape': tuple(x.shape),
        'stride': x.stride(),
        'contiguous': x.is_contiguous(),
    }


x = torch.arange(12).reshape(3, 4)
y = x[:, 1:3]
z = x.t()

print('x:', describe_layout(x))
print('y:', describe_layout(y))
print('z:', describe_layout(z))

```

## Q1验证：切片和转置后布局是否变化？

这里直接看三个视图：原始张量、切片张量、转置张量。重点不是它们的值，而是它们的 stride 和 contiguous 是否发生变化。


```python
x = torch.arange(12).reshape(3, 4)
y = x[:, 1:3]
z = x.t()

assert x.is_contiguous() is True
assert y.is_contiguous() is False
assert z.is_contiguous() is False
print('✅ layout 基本判断通过')

```

## Q2：什么时候该先做索引和广播判断？

当你要取局部片段、做 mask、或者把一个小张量扩展到大张量上时，先确认索引和广播会不会改变语义。很多后续错误不是算错，而是把维度关系写错了。


```python
x = torch.arange(1, 13).reshape(3, 4)
mask = x % 2 == 0
print('原始张量：')
print(x)
print('布尔 mask：')
print(mask)
print('masked 后：')
print(x.masked_fill(~mask, -1))

bias = torch.tensor([10, 20, 30, 40])
print('广播加 bias：')
print(x + bias)

```

## Q2验证：mask 和广播是否符合预期？

这里直接检查两件事：偶数元素是否保留，以及一维 bias 是否能按列广播到二维张量。


```python
x = torch.arange(1, 13).reshape(3, 4)
mask = x % 2 == 0
masked = x.masked_fill(~mask, -1)
assert masked.tolist() == [[-1, 2, -1, 4], [-1, 6, -1, 8], [-1, 10, -1, 12]]

bias = torch.tensor([10, 20, 30, 40])
broadcasted = x + bias
assert broadcasted.tolist() == [[11, 22, 33, 44], [15, 26, 37, 48], [19, 30, 41, 52]]
print('✅ mask 和广播通过')

```

## Q3：什么时候必须先保证内存连续？

只要你后面要调用 `view()`、交给更底层的算子，或者想减少额外拷贝，就要先检查连续性。连续性不是“可选信息”，它会直接决定后续操作能不能顺利进行。


```python
x = torch.arange(24).reshape(2, 3, 4)
y = x.transpose(1, 2)

print('x contiguous:', x.is_contiguous())
print('y contiguous:', y.is_contiguous())

try:
    y.view(-1)
except RuntimeError as e:
    print('view 失败：', str(e).split('\n')[0])

print('reshape 结果 shape：', y.reshape(-1).shape)

```

## Q3验证：连续性和 reshape 是否如预期？

这里直接确认：转置后通常不是连续的，`view()` 会失败，但 `reshape()` 还能给出正确结果。


```python
x = torch.arange(24).reshape(2, 3, 4)
y = x.transpose(1, 2)
assert y.is_contiguous() is False
assert y.reshape(-1).shape == (24,)
print('✅ contiguous 和 reshape 通过')

```

## Q4：什么时候 contiguous 不是性能问题，什么时候就是？

如果后续只是读，不一定非要连续；如果要 view、reshape 或进某些算子，连续性就会变成硬约束。


```python
def contiguous_priority(need_view, need_kernel, read_only):
    if need_view or need_kernel:
        return 'must_fix_contiguous'
    if read_only:
        return 'maybe_ok'
    return 'profile_first'


print('case1:', contiguous_priority(True, False, False))
print('case2:', contiguous_priority(False, False, True))
print('case3:', contiguous_priority(False, True, False))
# 输出示例: must_fix_contiguous / maybe_ok / must_fix_contiguous

```

## Q5：transpose 后为什么不能直接 view？

因为 transpose 改了 stride，shape 看起来没变，但内存布局已经不是原来的连续顺序。


```python
def view_after_transpose_ok(shape_ok, stride_ok, contiguous_ok):
    if not stride_ok or not contiguous_ok:
        return 'view_will_fail'
    if shape_ok:
        return 'view_safe'
    return 'shape_mismatch'


print('result1:', view_after_transpose_ok(True, False, False))
print('result2:', view_after_transpose_ok(True, True, True))
# 输出示例: view_will_fail / view_safe

```

## Q6：shape contract 和 layout contract 有什么区别？

shape 对的是“能不能对齐维度”，layout 对的是“这些维度是不是按你想的顺序存着”。两者不是一回事，前者对了不代表后者也对。


```python
def contract_report(shape_ok, layout_ok, need_view):
    if not shape_ok:
        return {'status': 'shape_fail', 'next': 'fix_shape_first'}
    if need_view and not layout_ok:
        return {'status': 'layout_fail', 'next': 'make_contiguous_or_use_reshape'}
    return {'status': 'ok', 'next': 'continue'}


print(contract_report(True, False, True))
print(contract_report(True, True, True))
# 输出示例: layout_fail -> make_contiguous_or_use_reshape

```

## Q7：什么时候该优先用 reshape 而不是 view？

只要你不确定内存是不是连续，或者更关心结果正确而不是零拷贝，就优先用 reshape；view 是“布局已经对齐”的更强前提。


```python
def prefer_reshape(contiguous_ok, must_zero_copy):
    if must_zero_copy and contiguous_ok:
        return 'view'
    if not contiguous_ok:
        return 'reshape'
    return 'either_but_view_preferred'


print(prefer_reshape(False, False))
print(prefer_reshape(True, True))
# 输出示例: reshape / view

```
