# 10. PyTorch State dict and Persistence | PyTorch 状态管理与持久化

**难度：** Medium | **环境：** CPU-first | **标签：** `PyTorch`, `持久化`, `checkpoint` | **目标人群：** Part 2-4 前置补课者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/10_PyTorch_State_dict_and_Persistence.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本页聚焦：理解参数、buffer 和 checkpoint 的关系；掌握 `state_dict` 的保存和加载套路；能判断什么时候该保存哪些状态。前一页已经把模型对象搭起来了，这一页就继续回答“搭好的模型怎么保存、怎么恢复、怎么迁移”。阅读顺序可以按这条线走：先分清模型状态和训练过程，再看参数、buffer 和 checkpoint 的边界，然后看保存恢复的闭环，最后看结构变化和部分加载。

**关键词：** `state_dict`, `checkpoint`, `buffer`

## 前置阅读
**导语：** 先看 0C 组页，把模型封装和状态管理的边界对齐，再进入这一页会更顺。
- [09. PyTorch nn.Module Basics | PyTorch nn.Module 基础](./09_PyTorch_nn_Module_Basics.md)
- [0C 组页](./0C.md)
- [28. Fault Tolerance and Checkpointing | 容错与检查点](../01_Hardware_Math_and_Systems/28_Fault_Tolerance_and_Checkpointing.md)

## 相关阅读
**导语：** 本页先把状态保存和恢复的最小判断讲清楚；如果想继续看优化器和损失函数，再顺着看下面这一页。
- [11. PyTorch Optimizers and Loss | PyTorch 优化器与损失函数](./11_PyTorch_Optimizers_and_Loss.md)

## Q1：`state_dict` 解决什么问题？

模型不是只由参数构成的，很多运行时状态也需要一起保存。`state_dict` 的作用，就是把参数和 buffer 统一成一个可恢复的状态入口。对 Part 2 来说，这一步主要是在帮你建立“模型状态”和“训练过程”分开看的习惯。先认清状态边界，后面的保存、恢复和部分加载才不会混。


```python
import torch
import torch.nn as nn


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        # buffer 常用来存 scale、mask、统计量或缓存这类“要保存但不训练”的状态。
        self.register_buffer('scale', torch.tensor(1.0))

    def forward(self, x):
        return self.linear(x) * self.scale


model = TinyNet()
state = model.state_dict()
print('state_dict keys：', sorted(state.keys()))

```

## Q1验证：`state_dict` 是否同时包含参数和 buffer？

这里直接看 keys，确认参数和 buffer 都在，但普通属性不在。


```python
model = TinyNet()
state = model.state_dict()
assert 'linear.weight' in state
assert 'linear.bias' in state
assert 'scale' in state
print('✅ state_dict keys 通过')

```

## Q2：什么时候必须区分参数、buffer 和 checkpoint？

参数会被优化器更新，buffer 参与前向但不更新，checkpoint 则通常还要额外带上训练进度和优化器状态。先分清这三类，保存时才不会漏。这里最重要的是：参数和 buffer 解决的是“模型当前长什么样”，checkpoint 解决的是“训练走到哪一步”。


```python
model = TinyNet()
print('parameters：', [n for n, _ in model.named_parameters()])
print('buffers：', [n for n, _ in model.named_buffers()])
# checkpoint 除了模型状态，通常还会装训练进度和优化器状态。
checkpoint = {'model': model.state_dict(), 'step': 100, 'epoch': 2}
print('checkpoint keys：', list(checkpoint.keys()))

```

## Q2验证：参数、buffer 和 checkpoint 是否分开？

这里确认三件事：参数里有权重，buffer 里有 scale，checkpoint 里还可以装训练进度。


```python
model = TinyNet()
assert 'linear.weight' in dict(model.named_parameters())
assert 'scale' in dict(model.named_buffers())
checkpoint = {'model': model.state_dict(), 'step': 100}
assert 'step' in checkpoint
print('✅ 参数 / buffer / checkpoint 通过')

```

## Q3：什么时候必须做保存和恢复的最小闭环？

只要你希望模型状态能稳定落盘、加载和复现，就要把 `save` 和 `load` 当成一对动作，而不是只写一半。这里的最小闭环，实际对应的是 Part 2 里所有“保存模型 -> 恢复模型 -> 继续跑”的页面；先把这条链跑顺，后面遇到 checkpoint 问题时通常就知道该查哪一层。


```python
model = TinyNet()
# 读出来再写回去，先确认最小闭环不丢状态。
before = {k: v.clone() for k, v in model.state_dict().items()}
buffer = model.state_dict()
model.load_state_dict(buffer)
after = model.state_dict()

for key in before:
    assert torch.allclose(before[key], after[key])
print('✅ save / load 闭环通过')

```

## Q3验证：保存和恢复是否不丢状态？

这里直接确认：把 state_dict 读出来再加载回去，参数和 buffer 的值不应变化。


```python
model = TinyNet()
state = model.state_dict()
model.load_state_dict(state)
print('恢复后 keys：', sorted(model.state_dict().keys()))
assert sorted(state.keys()) == sorted(model.state_dict().keys())

```

## Q4：什么时候要警惕结构变化导致的加载失败？

如果模型结构改了，旧 checkpoint 不一定还能直接加载。结构和状态分开保存更稳，但前提是你得知道当前结构和旧状态是否还匹配。这个判断在 Part 2 里特别常见，比如做 LoRA merge、替换头部、微调迁移或改 block 结构时。


```python
model = TinyNet()
state = model.state_dict()
print('模型结构：', model.__class__.__name__)
print('状态数量：', len(state))
# 结构变化时，先对齐 key，再谈是否能加载。
print('可恢复性检查完成')

```

## Q4验证：结构和状态是否对得上？

这里只做最小检查：结构名和 state_dict 的 key 是否能对齐，先确认加载前的边界是清楚的。


```python
model = TinyNet()
state = model.state_dict()
assert 'linear.weight' in state
assert 'linear.bias' in state
assert 'scale' in state
print('✅ 结构和状态对齐通过')

```

## Q5：什么时候只存权重不够，必须把训练状态一起存？

只存权重适合推理或纯参数迁移；如果你还要继续训练、复现实验或对比收敛过程，就要把优化器状态、学习率计划和训练步数一起带上。这里和 Part 2 的关系很直接：凡是会“中途断点续训”或“需要复现实验曲线”的页面，都不该只保存权重。


```python
def checkpoint_payload(need_resume_training, need_exact_repro):
    payload = {
        'model_state': True,
        'buffer_state': True,
    }
    if need_resume_training or need_exact_repro:
        # 继续训练时，optimizer / scheduler / step 也要一起存。
        payload['optimizer_state'] = True
        payload['lr_scheduler_state'] = True
        payload['global_step'] = True
    return payload


print(checkpoint_payload(False, False))
print(checkpoint_payload(True, False))
# 输出示例: 只推理时只保留 model/buffer；继续训练时带上 optimizer / scheduler / step

```

## Q6：什么时候应该允许部分加载，而不是强行 fail？

如果你明确知道旧 checkpoint 只复用部分层，或者是在做结构迁移实验，就可以允许部分加载；但这时必须清楚哪些层被加载、哪些层被重新初始化。这个场景在 Part 2 里很常见，比如换分类头、迁移 block 或者做局部权重复用。


```python
def partial_load_policy(is_migration_experiment, keys_match_rate):
    if is_migration_experiment and keys_match_rate >= 0.6:
        # 部分加载时，优先检查缺失层和重置层。
        return {'policy': 'allow_partial_load', 'next': 'inspect_missing_layers'}
    return {'policy': 'strict_load', 'next': 'fix_checkpoint_or_model'}


print(partial_load_policy(True, 0.8))
print(partial_load_policy(False, 0.8))
# 输出示例: allow_partial_load / strict_load

```
