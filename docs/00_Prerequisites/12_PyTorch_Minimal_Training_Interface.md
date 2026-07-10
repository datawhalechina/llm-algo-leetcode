# 12. PyTorch Minimal Training Interface | PyTorch 最小训练接口

**难度：** Medium | **环境：** CPU-first | **标签：** `PyTorch`, `数据接口`, `batch 契约` | **目标人群：** Part 2-4 前置补课者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/12_PyTorch_Minimal_Training_Interface.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本页聚焦：理解 Dataset 和 DataLoader 的最小职责；掌握 batch 组织和训练接口的关系；能把输入、标签和训练循环对齐到同一套约定。

**关键词：** `Dataset`, `DataLoader`, `collate`

## 前置阅读
**导语：** 先看 0C 组页，把训练闭环和数据接口的边界对齐，再进入这一页会更顺。
- [11. PyTorch Optimizers and Loss | PyTorch 优化器与损失函数](./11_PyTorch_Optimizers_and_Loss.md)
- [0C 组页](./0C.md)
- [05. Communication Topologies | 通信拓扑](../01_Hardware_Math_and_Systems/05_Communication_Topologies.md)

## 相关阅读
**导语：** 本页先把 Dataset、DataLoader 和 batch 契约讲清楚；如果想继续看训练循环怎么把这些接口串起来，再顺着看下面这一页。
- [13. Simple Neural Network Training | 简单神经网络训练循环](./13_Simple_Neural_Network_Training.md)

## Q1：Dataset 和 DataLoader 分别解决什么问题？

`Dataset` 负责单样本怎么取，`DataLoader` 负责 batch 怎么组。先把这两个角色分清，后面训练接口才不会把样本级逻辑和 batch 级逻辑混在一起。


```python
import torch
from torch.utils.data import Dataset, DataLoader


class ToyDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return torch.tensor([idx], dtype=torch.float32), torch.tensor([idx % 2], dtype=torch.float32)


loader = DataLoader(ToyDataset(), batch_size=2, shuffle=False)
for x, y in loader:
    print('batch x shape:', x.shape, 'batch y shape:', y.shape)

```

## Q1验证：batch 输出 shape 是否稳定？

这里直接确认 `DataLoader` 取出来的 batch 维度是否符合预期。


```python
loader = DataLoader(ToyDataset(), batch_size=2, shuffle=False)
batches = list(loader)
assert len(batches) == 2
x, y = batches[0]
assert x.shape == (2, 1)
assert y.shape == (2, 1)
print('✅ Dataset / DataLoader 通过')

```

## Q2：什么时候必须关注 batch 契约？

只要模型输入、标签形状和 batch 组织方式有可能变，训练循环就要先和数据接口对齐。否则看起来是模型问题，实际是契约没对上。


```python
def collate_pair(batch):
    xs, ys = zip(*batch)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


loader = DataLoader(ToyDataset(), batch_size=2, shuffle=False, collate_fn=collate_pair)
for x, y in loader:
    print('collate 后 x:', x.squeeze().tolist(), 'y:', y.squeeze().tolist())

```

## Q2验证：自定义 collate 是否真的改变了 batch 结构？

这里直接检查自定义 `collate_fn` 是否把原来的 `(B, 1)` 变成了更适合当前接口的形状。


```python
loader = DataLoader(ToyDataset(), batch_size=2, shuffle=False, collate_fn=collate_pair)
x, y = next(iter(loader))
assert x.shape == (2,)
assert y.shape == (2,)
print('✅ collate_fn 通过')

```

## Q3：什么时候必须让训练和验证复用同一套接口？

训练、验证和推理最好共享同一份数据组织方式，区别只在于是否更新参数。接口如果不统一，后面复现和排查都会很麻烦。


```python
def train_step_like(x, y):
    return x.float().mean().item(), y.float().mean().item()


for x, y in DataLoader(ToyDataset(), batch_size=2, shuffle=False):
    train_x_mean, train_y_mean = train_step_like(x, y)
    print('train-like means:', train_x_mean, train_y_mean)

```

## Q3验证：同一批数据在训练和验证中是否能复用？

这里确认一件事：同一份 batch 可以直接被训练逻辑和验证逻辑消费，而不需要重新改数据结构。


```python
loader = DataLoader(ToyDataset(), batch_size=2, shuffle=False)
x, y = next(iter(loader))
train_like = x.mean().item() + y.mean().item()
val_like = x.mean().item() + y.mean().item()
assert train_like == val_like
print('✅ 训练/验证接口一致通过')

```

## Q4：什么时候必须把最小训练接口串成闭环？

只要你要验证一套算法页的数据流，就应该先把单样本、batch、模型输入和训练步串成闭环。先跑通接口，再谈优化。


```python
def summarize_batch(batch):
    x, y = batch
    return {
        'x_shape': tuple(x.shape),
        'y_shape': tuple(y.shape),
        'x_mean': x.float().mean().item(),
        'y_mean': y.float().mean().item(),
    }


batch = next(iter(DataLoader(ToyDataset(), batch_size=2, shuffle=False)))
print('batch summary:', summarize_batch(batch))

```

## Q4验证：接口闭环是否可观察？

这里直接输出 batch 摘要，确认单样本到 batch 的转换链路是可见、可查、可复用的。


```python
batch = next(iter(DataLoader(ToyDataset(), batch_size=2, shuffle=False)))
summary = summarize_batch(batch)
assert summary['x_shape'] == (2, 1)
assert summary['y_shape'] == (2, 1)
print('✅ 最小训练接口闭环通过')

```

## Q5：什么时候必须自定义 `collate_fn`？

只要 batch 里出现变长序列、嵌套结构或需要额外对齐的字段，默认拼接就不够了，这时应该先把 batch 组织规则写进 `collate_fn`。


```python
def need_custom_collate(has_variable_length, has_nested_structures, needs_padding):
    if has_variable_length or has_nested_structures or needs_padding:
        return {'collate': 'custom', 'reason': 'batch_contract_not_flat'}
    return {'collate': 'default', 'reason': 'batch_contract_flat'}


print(need_custom_collate(True, False, False))
print(need_custom_collate(False, False, False))
# 输出示例: custom / default 会直接对应 batch 组织方式

```

## Q6：什么时候应该先改数据契约，而不是改模型输入？

如果问题的根源是 batch 形状、字段对齐或样本组织方式不稳定，就应该先改数据契约；模型只在契约稳定后再调整。


```python
def fix_layer_for_interface(problem_at_batch, problem_at_sample, model_can_absorb):
    if problem_at_batch or problem_at_sample:
        return {'first_fix': 'data_contract', 'next': 'collate_or_sampler'}
    if model_can_absorb:
        return {'first_fix': 'model_input', 'next': 'adjust_forward_signature'}
    return {'first_fix': 'profile_more', 'next': 'inspect_end_to_end'}


print(fix_layer_for_interface(True, False, False))
print(fix_layer_for_interface(False, False, True))
# 输出示例: data_contract / model_input

```
