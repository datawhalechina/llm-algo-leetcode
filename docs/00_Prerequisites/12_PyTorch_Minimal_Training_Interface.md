# 12. PyTorch Minimal Training Interface | PyTorch 最小训练接口

**难度：** Medium | **环境：** CPU-first | **标签：** `PyTorch`, `数据接口`, `batch 契约` | **目标人群：** Part 2-4 前置补课者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/12_PyTorch_Minimal_Training_Interface.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本页聚焦：理解 Dataset 和 DataLoader 的最小职责；掌握 batch 组织和训练接口的关系；能把输入、标签和训练循环对齐到同一套约定。前面几页已经把模型怎么更新讲清楚了，这一页开始补“数据怎么稳定地喂进去”。你可以先把它看成 Part 2 里最常回看的“数据入口”：先把样本组织好，再看 batch 怎么组、怎么对齐、怎么搬到 device。阅读顺序可以按这条线走：先看单样本和 batch 怎么分工，再看 batch 契约和 SFT 对齐，最后把 batch、device 和训练接口串起来。更贴近真实训练时，batch 往往不是简单的 `(x, y)`，而是带有 `input_ids / attention_mask / labels` 这类字段的字典。实际训练前，batch 还要先和模型放到同一个 device 上。

**关键词：** `Dataset`, `DataLoader`, `collate`

## 前置阅读
**导语：** 先看 0C 组页，把训练闭环和数据接口的边界对齐，再进入这一页会更顺。
- [11. PyTorch Optimizers and Loss | PyTorch 优化器与损失函数](./11_PyTorch_Optimizers_and_Loss.md)
- [0C 组页](./0C.md)
- [02. Communication Topologies | 通信拓扑](../01_Hardware_Math_and_Systems/05_Communication_Topologies.md)

## 相关阅读
**导语：** 本页先把 Dataset、DataLoader 和 batch 契约讲清楚；如果想继续看训练循环怎么把这些接口串起来，再顺着看下面这一页。
- [13. Simple Neural Network Training | 简单神经网络训练循环](./13_Simple_Neural_Network_Training.md)

## Q1：Dataset 和 DataLoader 分别解决什么问题？

`Dataset` 负责单样本怎么取，`DataLoader` 负责 batch 怎么组。先把这两个角色分清，后面训练接口才不会把样本级逻辑和 batch 级逻辑混在一起。这里先记住：`Dataset` 解决“单条样本长什么样”，`DataLoader` 解决“这些样本怎么组批”。在 Part 2 的真实页面里，batch 常常还会多出 `attention_mask`、`labels`、`position_ids` 这类字段。


```python
import torch
from torch.utils.data import Dataset, DataLoader


class ToyDataset(Dataset):
    def __len__(self):
        # 固定样本数，方便在 notebook 里稳定观察 batch 行为。
        return 4

    def __getitem__(self, idx):
        # 单样本直接返回字典，后面 collate / SFT 对齐时可以按字段处理。
        return {
            'input_ids': torch.tensor([idx], dtype=torch.long),
            'attention_mask': torch.tensor([1], dtype=torch.long),
            'labels': torch.tensor([idx % 2], dtype=torch.long),
        }


loader = DataLoader(ToyDataset(), batch_size=2, shuffle=False)
for batch in loader:
    print('batch keys:', list(batch.keys()))
    print('input_ids shape:', batch['input_ids'].shape, 'labels shape:', batch['labels'].shape)

```

## Q1验证：batch 输出 shape 是否稳定？

这里直接确认 `DataLoader` 取出来的 batch 维度是否符合预期。你要记住的是，batch 不一定总是 tuple，dict batch 在大模型训练里更常见。


```python
loader = DataLoader(ToyDataset(), batch_size=2, shuffle=False)
batches = list(loader)
assert len(batches) == 2
batch = batches[0]
assert batch['input_ids'].shape == (2, 1)
assert batch['attention_mask'].shape == (2, 1)
assert batch['labels'].shape == (2, 1)
print('✅ Dataset / DataLoader 通过')

```

## Q2：什么时候必须关注 batch 契约？

只要模型输入、标签形状和 batch 组织方式有可能变，训练循环就要先和数据接口对齐。否则看起来是模型问题，实际是契约没对上。这里重点是：batch 契约决定了模型输入怎么解包、标签怎么对齐。对于大模型页面，最常见的契约就是字典 batch，而不是简单元组。Q2 验证先看结构是否真的变了，再用补充例子看 SFT 场景怎么对齐。


```python
def collate_pair(batch):
    # 自定义 collate 时，先把单样本拆开，再按当前任务需求重组。
    # 这里用 torch.cat 把同字段的单样本拼成 batch 字段。
    input_ids = torch.cat([item['input_ids'] for item in batch], dim=0)
    attention_mask = torch.cat([item['attention_mask'] for item in batch], dim=0)
    labels = torch.cat([item['labels'] for item in batch], dim=0)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }


loader = DataLoader(ToyDataset(), batch_size=2, shuffle=False, collate_fn=collate_pair)
for batch in loader:
    print('collate 后 keys:', list(batch.keys()))
    print('input_ids:', batch['input_ids'].tolist(), 'labels:', batch['labels'].tolist())

```

## Q2验证：自定义 collate 是否真的改变了 batch 结构？

这里直接检查自定义 `collate_fn` 是否把默认 batch 组织方式改成了更适合当前接口的字典 batch。

## Q2补充：SFT 风格 batch 要怎么对齐？

这不是一个新的问题，而是 Q2 在 SFT 场景下的具体例子。当样本里同时有 `prompt`、`response` 和 `loss_mask` 时，batch 就不再是简单的 `(x, y)`。这里先用一个最小例子把 `prompt -> -100`、`response -> loss` 这条最常见的对齐链跑通，再用断言确认 padding、mask 和 labels 没有串位。


```python
def pad_to_max_len(tensors, pad_value):
    # 先按最长样本补齐，再把 batch 维堆起来。
    max_len = max(t.size(0) for t in tensors)
    padded = []
    for tensor in tensors:
        pad_len = max_len - tensor.size(0)
        if pad_len == 0:
            padded.append(tensor)
            continue
        pad = torch.full((pad_len,), pad_value, dtype=tensor.dtype)
        padded.append(torch.cat([tensor, pad], dim=0))
    return torch.stack(padded, dim=0)


def build_sft_sample(prompt_ids, response_ids):
    # prompt 位置用 -100 忽略，response 位置保留给 loss。
    input_ids = torch.tensor(prompt_ids + response_ids, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([-100] * len(prompt_ids) + response_ids, dtype=torch.long)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }

sft_samples = [
    build_sft_sample([101, 11, 12, 102], [7, 8]),
    build_sft_sample([101, 21, 102], [9, 10]),
]

# padding 只是把变长样本对齐到同一 batch 形状。
sft_batch = {
    'input_ids': pad_to_max_len([item['input_ids'] for item in sft_samples], pad_value=0),
    'attention_mask': pad_to_max_len([item['attention_mask'] for item in sft_samples], pad_value=0),
    'labels': pad_to_max_len([item['labels'] for item in sft_samples], pad_value=-100),
}

# 先看 padding 是否把样本补齐，再用断言确认 prompt / response / pad 的角色没串位。
assert tuple(sft_batch['input_ids'].shape) == (2, 6)
assert tuple(sft_batch['attention_mask'].shape) == (2, 6)
assert tuple(sft_batch['labels'].shape) == (2, 6)
assert sft_batch['input_ids'][0].tolist() == [101, 11, 12, 102, 7, 8]
assert sft_batch['labels'][0].tolist() == [-100, -100, -100, -100, 7, 8]
assert sft_batch['input_ids'][1].tolist() == [101, 21, 102, 9, 10, 0]
assert sft_batch['attention_mask'][1].tolist() == [1, 1, 1, 1, 1, 0]
assert sft_batch['labels'][1].tolist() == [-100, -100, -100, 9, 10, -100]

print('input_ids shape:', tuple(sft_batch['input_ids'].shape))
print('attention_mask row1:', sft_batch['attention_mask'][1].tolist())
print('labels row0:', sft_batch['labels'][0].tolist())
print('labels row1:', sft_batch['labels'][1].tolist())

```


```python
loader = DataLoader(ToyDataset(), batch_size=2, shuffle=False, collate_fn=collate_pair)
batch = next(iter(loader))
assert batch['input_ids'].shape == (2,)
assert batch['attention_mask'].shape == (2,)
assert batch['labels'].shape == (2,)
print('✅ collate_fn 通过')

```

## Q3：什么时候必须让训练和验证复用同一套接口？

训练、验证和推理最好共享同一份数据组织方式，区别只在于是否更新参数。接口如果不统一，后面复现和排查都会很麻烦。这里要记住：同一套 batch 契约应该同时服务 train / val / infer。真正变化的通常是 `model.train()` / `model.eval()` 和是否包 `no_grad()`，不是 batch 的字段结构；另外，batch 进模型前通常还会先搬到同一个 device。


```python
def batch_to_device(batch, device):
    # 真实训练里，先把 batch 搬到 device，再交给模型。
    return {k: v.to(device) for k, v in batch.items()}


def train_step_like(batch, device=torch.device('cpu')):
    # 这里模拟训练步：先对齐 device，再从 batch 里取字段。
    batch = batch_to_device(batch, device)
    x = batch['input_ids'].float()
    y = batch['labels'].float()
    return x.mean().item(), y.mean().item()


for batch in DataLoader(ToyDataset(), batch_size=2, shuffle=False):
    train_x_mean, train_y_mean = train_step_like(batch)
    print('train-like means:', train_x_mean, train_y_mean)

```

## Q3验证：同一批数据在训练和验证中是否能复用？

这里确认一件事：同一份 batch 可以直接被训练逻辑和验证逻辑消费，而不需要重新改数据结构。


```python
loader = DataLoader(ToyDataset(), batch_size=2, shuffle=False)
batch = next(iter(loader))
train_like = batch['input_ids'].float().mean().item() + batch['labels'].float().mean().item()
val_like = batch['input_ids'].float().mean().item() + batch['labels'].float().mean().item()
assert train_like == val_like
print('✅ 训练/验证接口一致通过')

```

## Q4：什么时候必须把最小训练接口串成闭环？

只要你要验证一套算法页的数据流，就应该先把单样本、batch、模型输入和训练步串成闭环。这里的闭环重点是：样本 -> batch -> 模型输入 -> 训练步。对 Part 2 来说，这一步会直接帮你检查 batch 契约和模型签名是不是一致，也能顺手确认 batch 是否已经放到正确的 device。


```python
def summarize_batch(batch):
    return {
        'input_shape': tuple(batch['input_ids'].shape),
        'mask_shape': tuple(batch['attention_mask'].shape),
        'label_shape': tuple(batch['labels'].shape),
        'input_mean': batch['input_ids'].float().mean().item(),
        'label_mean': batch['labels'].float().mean().item(),
    }


batch = next(iter(DataLoader(ToyDataset(), batch_size=2, shuffle=False)))
print('batch summary:', summarize_batch(batch))

```

## Q4验证：接口闭环是否可观察？

这里直接输出 batch 摘要，确认单样本到 batch 的转换链路是可见、可查、可复用的。


```python
batch = next(iter(DataLoader(ToyDataset(), batch_size=2, shuffle=False)))
summary = summarize_batch(batch)
assert summary['input_shape'] == (2, 1)
assert summary['mask_shape'] == (2, 1)
assert summary['label_shape'] == (2, 1)
print('✅ 最小训练接口闭环通过')

```

## Q5：什么时候必须自定义 `collate_fn`？

如果 batch 里出现变长序列、嵌套结构或需要额外对齐的字段，默认拼接就不够了，这时应该先把 batch 组织规则写进 `collate_fn`。这里可以把它理解成“批处理层的定制打包逻辑”：先决定怎么补齐，再决定怎么对齐字段。真实的大模型 batch 往往需要 padding、mask 和字段对齐，这一题就是在把 Q2 的契约判断落到实现层。


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

如果问题的根源是 batch 形状、字段对齐或样本组织方式不稳定，就应该先改数据契约；模型只在契约稳定后再调整。这里优先级很重要：先把数据打包方式理顺，再动模型签名。对 Part 2 来说，先修 batch 再修模型，通常比反过来更高效。Q6 是前面判断的收口：先看数据契约，再看模型输入，SFT 里最常见的就是先让 prompt / response / mask 对齐，再考虑模型输入形式。

这也是为什么先看 batch 契约，再看模型签名，通常比反过来更稳。


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
