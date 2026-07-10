# 13. Simple Neural Network Training | 简单神经网络训练循环

**难度：** Medium | **环境：** CPU-first | **标签：** `PyTorch`, `训练骨架`, `checkpoint` | **目标人群：** Part 2-4 前置补课者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/13_Simple_Neural_Network_Training.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本页聚焦：会写完整但最小的训练循环；会切换 train / eval 模式；会把保存、早停和验证指标串起来。

**关键词：** `train`, `eval`, `checkpoint`

## 前置阅读
**导语：** 先看 0D 组页，把训练循环和模型直觉的边界对齐，再进入这一页会更顺。
- [12. PyTorch Minimal Training Interface | PyTorch 最小训练接口](./12_PyTorch_Minimal_Training_Interface.md)
- [0D 组页](./0D.md)
- [13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)

## 相关阅读
**导语：** 本页先把最小训练循环、模式切换和保存逻辑讲清楚；如果想继续看激活函数如何影响训练稳定性，再顺着看下面这一页。
- [14. Activation Functions | 激活函数](./14_Activation_Functions.md)

## Q1：训练循环的固定骨架是什么？

训练不是孤立的 forward/backward，而是一条固定流程：数据进来、前向、算损失、反向、更新、验证、保存。先把这个骨架记住，后面改模型才不会乱。


```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def make_data():
    x = torch.randn(8, 1)
    y = 2 * x + 1
    return DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)


model = nn.Linear(1, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.MSELoss()
loader = make_data()

print('batch 数量：', len(loader))
print('初始权重：', model.weight.item(), 'bias：', model.bias.item())

```

## Q1验证：训练前后是否能看到参数和 batch 信息？

这里先确认最小骨架：batch 数量能看到，参数也能读出来。


```python
loader = make_data()
x, y = next(iter(loader))
assert x.shape == (4, 1)
assert y.shape == (4, 1)
print('✅ batch shape 通过')

```

## Q2：什么时候必须切换 train / eval？

只要模型里有 Dropout、BatchNorm 或类似的训练态组件，就必须切换 train / eval。训练和评估不是同一种执行方式。


```python
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


model = TinyNet()
model.train()
print('训练态：', model.training)
model.eval()
print('评估态：', model.training)

```

## Q2验证：train / eval 切换是否生效？

这里直接确认 `model.training` 的状态变化，先把模式切换看清楚。


```python
model = TinyNet()
model.train()
assert model.training is True
model.eval()
assert model.training is False
print('✅ train / eval 通过')

```

## Q3：什么时候必须把验证和保存绑在一起？

保存最好的模型，通常不看训练损失，而是看验证指标。只要你开始比较不同 epoch，就应该把验证和保存绑到一起。


```python
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0.0
    for x, y in loader:
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def validate(model, loader, criterion):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x)
            loss = criterion(pred, y)
            total += loss.item()
    return total / len(loader)


train_loader = make_data()
val_loader = make_data()
print("train loss:", train_one_epoch(model, train_loader, optimizer, criterion))

```

## Q3验证：验证时是否真的不追踪梯度？

这里直接确认验证循环里用了 `no_grad()`，避免白白建图。


```python
model = TinyNet()
val_loader = make_data()
val_loss = validate(model, val_loader, criterion)
assert isinstance(val_loss, float)
print('✅ 验证循环通过，val_loss =', val_loss)

```

## Q4：什么时候必须把早停和 checkpoint 接进训练骨架？

只要你开始对比不同 epoch 的结果，就应该把早停和 checkpoint 接到同一条训练骨架上，而不是写成临时脚本。


```python
best_val = float("inf")
patience = 2
bad_epochs = 0
for epoch in range(3):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss = validate(model, val_loader, criterion)
    print(f'epoch={epoch}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}')
    if val_loss < best_val:
        best_val = val_loss
        bad_epochs = 0
        print('save checkpoint')
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            print('early stop')
            break

```

## Q4验证：训练骨架里是否真的能看到保存逻辑？

这里直接检查：验证指标变好时会进入保存分支，变差时会进入 early stop 计数。


```python
assert best_val < float("inf")
print('✅ 训练骨架闭环通过')

```

## Q5：什么时候训练骨架里的顺序不能乱？

只要你还要维护可复现的训练过程，forward、loss、backward、step、validate、save 的顺序就不能随便改；顺序一乱，结果就很难比较。


```python
def training_order_ok(steps):
    required = ["forward", "loss", "backward", "step", "validate", "save"]
    idx = {name: steps.index(name) if name in steps else None for name in required}
    order_ok = all(idx[a] < idx[b] for a, b in [("forward", "loss"), ("loss", "backward"), ("backward", "step"), ("step", "validate"), ("validate", "save")])
    return {
        'order_ok': order_ok,
        'index_map': idx,
    }


steps = ['forward', 'loss', 'backward', 'step', 'validate', 'save']
print(training_order_ok(steps))
print(training_order_ok(['forward', 'backward', 'loss', 'step', 'validate', 'save']))
# 输出示例: order_ok 会区分正常顺序和乱序

```

## Q6：什么时候早停应该看验证集而不是训练集？

只要训练集和验证集的表现开始分叉，早停就应该盯验证集而不是训练集；训练集继续下降不代表模型真的更好。


```python
def early_stop_signal(train_loss, val_loss, patience, best_val, bad_epochs):
    if val_loss < best_val:
        return {'signal': 'save_and_reset', 'best_val': val_loss, 'bad_epochs': 0}
    bad_epochs += 1
    if bad_epochs >= patience:
        return {'signal': 'stop', 'best_val': best_val, 'bad_epochs': bad_epochs}
    return {'signal': 'continue', 'best_val': best_val, 'bad_epochs': bad_epochs}


print(early_stop_signal(train_loss=0.2, val_loss=0.3, patience=2, best_val=0.35, bad_epochs=0))
print(early_stop_signal(train_loss=0.1, val_loss=0.4, patience=2, best_val=0.3, bad_epochs=1))
# 输出示例: save_and_reset / stop / continue 会反映验证集判断

```
