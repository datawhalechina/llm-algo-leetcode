# 06. Simple Neural Network Training | 完整的神经网络训练循环

**难度：** Medium | **标签：** `PyTorch`, `训练循环`, `深度学习` | **目标人群：** 所有学习者

## 🎯 学习目标

- 掌握完整的训练循环结构
- 理解训练、验证、测试的区别
- 学会模型保存和加载
- 掌握早停（Early Stopping）技术

---

## 📚 前置知识

- PyTorch Tensor 基础（02 题）
- PyTorch Autograd（03 题）
- PyTorch nn.Module（04 题）
- 损失函数和优化器（05 题）

---

## 💡 核心概念

### 训练循环的标准结构

```
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    # 验证阶段
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            output = model(input)
            val_loss = criterion(output, target)
```

---

## 📖 Part 1: 完整的训练循环

### 1.1 基础训练循环

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    训练一个 epoch
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备（cpu 或 cuda）
    
    Returns:
        avg_loss: 平均损失
        accuracy: 准确率
    """
    model.train()  # 设置为训练模式
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # 数据转移到设备
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy
```

### 1.2 验证循环

```python
def validate(model, val_loader, criterion, device):
    """
    验证模型
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备
    
    Returns:
        avg_loss: 平均损失
        accuracy: 准确率
    """
    model.eval()  # 设置为评估模式
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # 不计算梯度
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy
```

### 1.3 完整训练流程

```python
def train(model, train_loader, val_loader, criterion, optimizer, 
          num_epochs, device, save_path='best_model.pt'):
    """
    完整的训练流程
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数
        device: 设备
        save_path: 模型保存路径
    """
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 打印信息
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f'  ✅ Best model saved! Val Acc: {val_acc:.2f}%')
    
    print(f'\nTraining completed! Best Val Acc: {best_val_acc:.2f}%')
```

---

## 📖 Part 2: model.train() vs model.eval()

### 2.1 为什么需要切换模式？

某些层在训练和推理时行为不同：
- **Dropout**：训练时随机丢弃，推理时不丢弃
- **BatchNorm**：训练时使用 batch 统计，推理时使用全局统计

```python
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.dropout = nn.Dropout(0.5)  # 训练时丢弃 50%
        self.bn = nn.BatchNorm1d(20)
        self.fc2 = nn.Linear(20, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)  # 训练时生效
        x = self.bn(x)       # 训练时更新统计量
        x = self.fc2(x)
        return x

model = SimpleModel()

# 训练模式
model.train()
# - Dropout 会随机丢弃
# - BatchNorm 会更新 running_mean 和 running_var

# 评估模式
model.eval()
# - Dropout 不丢弃
# - BatchNorm 使用保存的 running_mean 和 running_var
```

---

## 📖 Part 3: 早停（Early Stopping）

### 3.1 什么是早停？

当验证集性能不再提升时，提前停止训练，防止过拟合。

```python
class EarlyStopping:
    """早停工具类"""
    
    def __init__(self, patience=7, min_delta=0, mode='max'):
        """
        Args:
            patience: 容忍多少个 epoch 不提升
            min_delta: 最小改进量
            mode: 'max' 表示指标越大越好，'min' 表示越小越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        """
        Args:
            score: 当前指标（如验证准确率或验证损失）
        
        Returns:
            is_best: 是否是最佳模型
        """
        if self.best_score is None:
            self.best_score = score
            return True
        
        # 判断是否改进
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

# 使用示例
early_stopping = EarlyStopping(patience=5, mode='max')

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(...)
    val_loss, val_acc = validate(...)
    
    # 检查早停
    is_best = early_stopping(val_acc)
    if is_best:
        torch.save(model.state_dict(), 'best_model.pt')
    
    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch+1}")
        break
```

---

## 📖 Part 4: 模型保存和加载

### 4.1 保存和加载 state_dict（推荐）

```python
# 保存
torch.save(model.state_dict(), 'model.pt')

# 加载
model = SimpleModel()
model.load_state_dict(torch.load('model.pt'))
model.eval()
```

### 4.2 保存完整模型

```python
# 保存
torch.save(model, 'model_full.pt')

# 加载
model = torch.load('model_full.pt')
model.eval()
```

### 4.3 保存训练状态（Checkpoint）

```python
# 保存
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'best_acc': best_acc,
}
torch.save(checkpoint, 'checkpoint.pt')

# 加载
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
best_acc = checkpoint['best_acc']
```

---

## 📖 Part 5: 学习率调度

### 5.1 StepLR - 每隔几个 epoch 降低学习率

```python
from torch.optim.lr_scheduler import StepLR

optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(num_epochs):
    train_one_epoch(...)
    validate(...)
    scheduler.step()  # 更新学习率
```

### 5.2 ReduceLROnPlateau - 根据验证指标调整

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10)

for epoch in range(num_epochs):
    train_one_epoch(...)
    val_loss, val_acc = validate(...)
    scheduler.step(val_acc)  # 根据验证准确率调整
```

### 5.3 CosineAnnealingLR - 余弦退火

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(num_epochs):
    train_one_epoch(...)
    validate(...)
    scheduler.step()
```

---

## 📖 Part 6: 进度条和日志

### 6.1 使用 tqdm 显示进度条

```python
from tqdm import tqdm

def train_one_epoch_with_progress(model, train_loader, criterion, optimizer, device):
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    # 创建进度条
    pbar = tqdm(train_loader, desc='Training')
    
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct / total:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy
```

### 6.2 使用 TensorBoard 记录日志

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(...)
    val_loss, val_acc = validate(...)
    
    # 记录到 TensorBoard
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

writer.close()

# 启动 TensorBoard: tensorboard --logdir=runs
```

---

## 🎯 实战练习

### 练习 1: 实现完整的训练流程

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 创建简单的数据集
def create_dummy_dataset():
    """创建虚拟数据集用于测试"""
    X_train = torch.randn(1000, 10)
    y_train = torch.randint(0, 2, (1000,))
    X_val = torch.randn(200, 10)
    y_val = torch.randint(0, 2, (200,))
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader

# 定义模型
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# TODO: 实现完整的训练流程
def main():
    # 1. 准备数据
    train_loader, val_loader = create_dummy_dataset()
    
    # 2. 创建模型
    model = SimpleClassifier()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. 训练模型
    # TODO: 调用 train() 函数
    pass

if __name__ == '__main__':
    main()
```

### 练习 2: 添加早停和学习率调度

```python
# TODO: 在上面的训练流程中添加：
# 1. 早停机制（patience=5）
# 2. 学习率调度（ReduceLROnPlateau）
# 3. 保存最佳模型
```

### 练习 3: 添加进度条和日志

```python
# TODO: 在训练流程中添加：
# 1. tqdm 进度条
# 2. TensorBoard 日志记录
```

---

## 📚 参考答案

<details>
<summary>点击查看完整训练流程答案</summary>

```python
def main():
    # 准备数据
    train_loader, val_loader = create_dummy_dataset()
    
    # 创建模型
    model = SimpleClassifier()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 早停和学习率调度
    early_stopping = EarlyStopping(patience=5, mode='max')
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # TensorBoard
    writer = SummaryWriter('runs/simple_classifier')
    
    # 训练
    num_epochs = 50
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 学习率调度
        scheduler.step(val_acc)
        
        # 记录日志
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # 打印
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 早停
        is_best = early_stopping(val_acc)
        if is_best:
            torch.save(model.state_dict(), 'best_model.pt')
            print(f'  ✅ Best model saved!')
        
        if early_stopping.early_stop:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    writer.close()
    print(f'\nTraining completed!')
```
</details>

---

## 🔗 相关资源

- [PyTorch Training Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [PyTorch Learning Rate Scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
- [TensorBoard with PyTorch](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)

---

## 🎓 总结

本节学习了完整的神经网络训练流程：
- ✅ 训练循环的标准结构
- ✅ model.train() vs model.eval()
- ✅ 早停（Early Stopping）
- ✅ 模型保存和加载
- ✅ 学习率调度
- ✅ 进度条和日志记录

**下一步：** 学习 [10. PyTorch Profiling Basics](./10_PyTorch_Profiling_Basics.md)，掌握性能分析工具。
