# 训练工具桥：Autograd / AMP / Checkpoint / Profiler

## 页面目标

本页不引入新的训练机制，而是回答一个实现问题：

前面讲的 backward、mixed precision、checkpointing、profiling，在 PyTorch 里分别由哪些训练工具承接？

它是 `反向传播与训练机制专题` 的桥接附录页，不替代 `01-06` 主线，只负责把“机制理解”接到“训练实现”。

## 本页解决什么

如果你已经知道：

- 梯度怎么回去
- 哪些状态必须保留
- mixed precision / checkpointing / clipping 为什么存在
- profiling 怎样验证训练收益

下一步最常见的问题就是：

- 这些事在 PyTorch 里到底该用什么库
- 它们的调用顺序是什么
- 哪些工具在解决机制问题，哪些工具只是在提供接口

## 最小工具地图

| 工具 | 主要负责什么 | 不负责什么 |
|:---|:---|:---|
| `torch.autograd` | 自动构建和执行反向传播图 | 不直接帮你优化显存或吞吐 |
| `torch.amp.autocast` | 控制前向 / 反向中的低精度计算环境 | 不负责梯度缩放策略本身 |
| `torch.amp.GradScaler` | 在 `FP16` 训练里避免小梯度下溢 | `BF16` 通常不依赖它 |
| `torch.utils.checkpoint` | 用重算换显存 | 不替代 offload |
| `torch.nn.utils.clip_grad_norm_` | 在 step 前限制梯度范数 | 不改变 accumulation 节奏 |
| `torch.optim` | 参数更新 | 不负责 scheduler 策略解释 |
| `torch.optim.lr_scheduler` | 学习率调度 | 不负责 optimizer 状态本身 |
| `torch.profiler` | 采集训练阶段的时间 / 显存证据 | 不直接做优化 |

## 按训练闭环看工具

更稳的记法不是背 API，而是把它们放回训练闭环：

`forward -> loss -> backward -> accumulation -> unscale / clip -> step -> zero_grad -> profiler validate`

对应关系可以这样看：

| 训练环节 | 常用工具 | 你应该关注什么 |
|:---|:---|:---|
| forward | `autocast` | 哪些算子处在低精度环境里 |
| backward | `autograd`、`GradScaler` | 梯度图如何回传，是否需要缩放 |
| accumulation | 训练循环逻辑 | `backward` 次数和 `step` 次数是否分离 |
| 显存控制 | `checkpoint` | 哪些中间状态值得重算 |
| step 前稳定化 | `clip_grad_norm_` | clipping 是否发生在真正更新前 |
| 参数更新 | `optimizer`、`scheduler` | 更新步长和时机是否合理 |
| 证据采集 | `profiler` | 优化前后收益是否可复现 |

## 最小训练工具顺序

下面是一条最小、最常见的 PyTorch 训练顺序。代码是调用顺序示意，运行前需要由训练循环提供 `scaler`、`use_fp16`、`should_step` 和 `max_norm`；它不是可直接复制的完整训练脚本。

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss = model_forward(...)

scaled_loss = scaler.scale(loss) if use_fp16 else loss
scaled_loss.backward()

if should_step:
    if use_fp16:
        scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    if use_fp16:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

这段顺序的重点不在语法，而在职责分层：

- `autocast` 决定低精度计算环境
- `GradScaler` 只在需要时包住 backward / step
- `clip_grad_norm_` 放在真正更新前
- accumulation 决定什么时候才进入 `step`

## 常见组合

### 1. 标准单卡训练

- `autograd`
- `optimizer`
- `scheduler`
- `profiler`

适合先把最小训练闭环跑通。

### 2. 低精度训练

- `autocast`
- `GradScaler`（主要针对 `FP16`）
- `clip_grad_norm_`

适合真实训练里更常见的数值稳定化组合。

### 3. 显存紧张训练

- `checkpoint`
- accumulation
- mixed precision

适合把“重算换显存”和“低精度换显存”放到同一训练闭环里看。

### 4. 训练取证

- `torch.profiler`
- 训练日志
- benchmark / regression 记录

适合把“感觉更快”变成“证据上更快”。

## 不在本页展开什么

以下内容不在本页展开：

- `DDP / FSDP / ZeRO`
- 完整 trainer 框架
- logging / experiment tracking 全生态
- 数据加载和分布式输入管线

原因很简单：这页只负责把 `反向传播与训练机制专题` 里的机制落到最小 PyTorch 工具桥，不负责做完整训练框架地图。

## 对应来源

- [Part 02 · 17 自动求导基础](../../02_PyTorch_Algorithms/17_Autograd_Basics.md)
- [Part 02 · 19 激活检查点与 Offload](../../02_PyTorch_Algorithms/19_Activation_Checkpointing_and_Activation_Offload.md)
- [Part 02 · 12 梯度累积](../../02_PyTorch_Algorithms/12_Gradient_Accumulation.md)
- [Part 02 · 13 端到端微调实验](../../02_PyTorch_Algorithms/13_End_to_End_Fine_Tuning_Experiment.md)
- [Part 02 · 74 Profiling 驱动的端到端优化](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.md)

## 工程资料

| 资料 | 读它的理由 |
|:---|:---|
| [torch.autograd](https://docs.pytorch.org/docs/stable/autograd) | 看反向传播图和 `grad_fn` 的官方定义。 |
| [torch.amp](https://docs.pytorch.org/docs/stable/amp.html) | 看 `autocast` 和 `GradScaler` 的官方接口。 |
| [torch.utils.checkpoint](https://docs.pytorch.org/docs/stable/checkpoint) | 看 checkpoint API 和重算边界。 |
| [clip_grad_norm_](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html) | 看梯度裁剪应该放在什么位置。 |
| [torch.profiler](https://docs.pytorch.org/docs/main/profiler) | 看训练阶段的证据采集方式。 |

## 阅读建议

- 如果你还在理解机制，先回主线 `01-05`。
- 如果你已经懂机制，但不知道 PyTorch 里该用什么工具，先看这页。
- 如果你已经开始搭训练闭环，这页更适合作为“调用顺序检查表”。
