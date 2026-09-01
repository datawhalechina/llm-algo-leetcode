# 01. 反向传播总览与计算图 | Backpropagation Overview and Computation Graph

## 页面目标

本页说明为什么需要单独分析 backward，并把它和计算图、链式法则、显存及调度边界连起来。

本页的输出是训练图总览：明确梯度沿哪里回传、哪些中间状态必须保留，以及这些保存边界为什么会影响显存和调度。

## 核心问题

### 1. backward 在训练里处于什么位置

forward 负责产生输出，backward 负责把 loss 的信号沿计算图传回去。只要训练还依赖梯度更新，backward 就不是附属步骤，而是训练主链路的一半。

### 2. 计算图为什么是 backward 的骨架

计算图回答的是“梯度沿哪里走、哪些状态要留、哪里可以重算”。它不是抽象装饰，而是训练框架组织梯度路径的方式。

### 3. backward 为什么会影响显存和调度

要让梯度算得出来，框架通常要保留一部分前向中间量。序列越长、层数越深、状态越多，backward 的显存压力就越明显；而一旦显存受限，训练调度就会被迫改变。

## 你要建立的判断框架

当训练阶段出现 OOM、吞吐下降或者收敛异常时，先不要急着改超参数，而是先判断问题到底属于哪一类：

- 是梯度本身回传太重，还是中间激活占得太多
- 是 forward / backward 的保存边界不合理，还是 step 频率和 batch 口径有问题
- 是单卡显存不足，还是跨设备搬运和通信把时间吃掉了

## 工程上的三个后果

1. 计算图决定了哪些张量可以被安全释放，哪些张量必须留到 backward。
2. 计算图也是调试入口，出了梯度异常时，最先要回头看的通常就是 graph 的分支和保存点。
3. 计算图还决定了 checkpointing 的边界，因为只有明确哪些段可以重算，显存优化才有机会成立。

## 从图到可测对象

把一次训练 step 拆成 `forward → loss → backward → optimizer.step()` 后，可以分别记录：参数是否收到梯度、哪些张量仍被引用、每个阶段耗时以及显存峰值。CPU 实验适合验证梯度路径和释放时机；GPU 实验才适合判断 activation residency、workspace 和真实 step time。图结构正确，不等于某种 GPU 优化已经有效。

## 典型误区

- backward 不是 forward 的附属函数，它常常决定训练能不能跑起来。
- 不是所有前向输出都需要保存，只有 backward 还要用到的中间量才有保存价值。
- 图上的依赖关系和代码里的执行顺序不是一回事，尤其在有分支、残差和共享参数时更明显。

## 对应来源

- [Part 00 · 07 PyTorch 自动求导与反向传播](../../00_Prerequisites/07_PyTorch_Autograd_and_Backward.md)
- [Part 02 · 17 自动求导基础](../../02_PyTorch_Algorithms/17_Autograd_Basics.md)

## 经典论文

| 文献 | 读它的理由 |
|:---|:---|
| [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0) | 反向传播的起点，先理解梯度信号如何穿过多层网络。 |
| [Automatic differentiation in machine learning: a survey](https://arxiv.org/abs/1502.05767) | 看 automatic differentiation 如何把链式法则变成可执行的图。 |
| [Automatic Differentiation in ML: Where we are and where we should be going](https://arxiv.org/abs/1810.11530) | 补充 AD 的工程化视角，适合理解 graph-based autograd。 |

## 阅读建议

- 先把它当成整条专题的导言页。
- 如果你已经知道 backward 的基本概念，可以直接进入 `02`。

## 进入下一页

进入 [02 自动微分与 Attention 反向传播](./02_autograd_and_attention_backward.md)，把计算图上的抽象梯度路径对齐到 PyTorch autograd 和 attention 算子。
