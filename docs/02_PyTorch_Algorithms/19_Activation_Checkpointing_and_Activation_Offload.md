# 19. Activation Checkpointing and Activation Offload | 激活检查点

**难度：** Hard | **环境：** GPU-optional

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/19_Activation_Checkpointing_and_Activation_Offload.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

**标签：** `显存优化`, `激活值`, `Checkpointing` | **目标人群：** 显存优化学习者

---

## 本节导读

训练大模型时，显存压力不只来自参数本身。前向传播为了后面的反向传播，需要保存部分中间激活；层数、序列长度和 batch 增大时，这部分占用也可能增加。本节检查如何减少部分激活的驻留，同时保持反向传播正确。

本节专注 `activation checkpointing`：用重新计算换显存。默认先验证输出和梯度是否一致；显式开启 GPU 后只能观察当前 toy workload 的峰值变化，不能当作真实大模型结论。真实模型中的显存、吞吐和 step time 对比，要在固定 workload 下交给 `73 / 76` 测量。`Activation Offload` 作为搬运路线，后面单独看 `42`。

本节的代码任务是实现逐 Block 和分段 checkpoint，并对照输出与梯度；核心机制是在反向阶段重新执行区段前向，用额外计算换取部分显存空间。它不承诺固定显存节省比例，也不负责给出完整大模型收益或默认 GPU benchmark 结论；真实模型的显存、吞吐和 step time 交给 `73 / 76`。

**关键词：** `checkpointing`, `recompute`

## 前置阅读

**导语：** 先看训练闭环、反向传播和显存账本，再进入 checkpointing：本节关注的是当前反向路径需要哪些激活，哪些中间结果可以通过重算来换取显存空间。

- [P0: 13. Simple Neural Network Training | 简单神经网络训练循环](../00_Prerequisites/13_Simple_Neural_Network_Training.md)
- [18. Activation and Loss Backward | 激活与损失反向](../02_PyTorch_Algorithms/18_Activation_and_Loss_Backward.md)
- [P0: 20. Profiling and Memory Ledger | 性能分析与显存账本](../00_Prerequisites/20_Profiling_and_Memory_Ledger.md)


## 相关阅读

**导语：** 学完激活显存优化后，可以继续看训练性能分析、显存策略对比项目，也可以转向 attention 侧的显存优化，理解训练和推理里的 memory bottleneck 如何分别出现。

- [42. Activation Offload | 激活卸载](../02_PyTorch_Algorithms/42_Activation_Offload.md)
- [73. Training Performance Analysis | 训练性能分析](../02_PyTorch_Algorithms/73_Training_Performance_Analysis.md)
- [76. Activation Checkpoint Offload Benchmark | Checkpoint 与 Offload 对比项目](../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.md)

---

### Step 1: 核心思想与痛点

> **标准的前向传播与反向传播：**
> 在前向传播（Forward）时，Autograd 会为反向传播保存后续计算所需的部分中间张量。具体保存哪些张量、保存多久，取决于算子实现、是否启用 checkpoint，以及张量所在的设备；在深层、长序列或较大 batch 的训练中，激活相关的驻留空间可能成为显存压力的重要来源。
> 
> **Gradient Checkpointing 的机制：**
> 我们把连续的若干层组成一个 checkpoint 区段，减少区段内部中间激活的保存；区段边界仍会保留 Autograd 反向所需的信息。
> 等到反向传播需要区段内部的值时，checkpoint 会重新执行这一小段前向传播（Recomputation），再继续计算梯度。具体保留和重算的张量由实现决定，并不等同于“只保存一个输入张量”。
> **结果：通常会增加计算时间，但收益取决于激活在总显存中的占比、checkpoint 粒度和实现方式。本节不预设固定百分比。**
>
> ![Checkpoint vs Offload 图](/02_PyTorch_Algorithms/19_checkpoint_offload.svg)

### Step 2: 激活值重计算原理
在训练极深的模型时，为反向传播保留的大量中间激活可能消耗显存。Gradient Checkpointing 的思想是：为选定区段减少中间结果的保存，在反向传播需要时重新执行对应的前向计算。这是“时间换空间”；节省比例必须结合总显存账本和实际 workload 测量，不能从层数直接推出。

### Step 3: 代码实现框架
在 PyTorch 中，可以通过 `torch.utils.checkpoint.checkpoint` 为一段前向计算声明重算边界。调用方需要保证这段函数可重复执行，并注意随机状态、输入输出和副作用；checkpoint 会与 Autograd 协作，在反向阶段按需重算，而不是简单地替调用方管理整个激活图。
checkpoint 并不是“完全不保存激活”：区段输入、输出和 Autograd 需要的边界状态仍可能驻留，具体保存内容还取决于算子实现。它减少的是区段内部部分中间结果的保存，因此显存收益不能按层数直接推算。
这里的实现单位是 Transformer Block 级别的 checkpoint，而不是子层级别的 checkpoint。被包裹的函数应尽量没有外部副作用；如果包含 dropout 或其他随机操作，还要确认随机状态在重算前后保持一致。

### Step 4: 动手实战

**要求**：完成两个层次的实现：

1. 补全 `run_with_checkpointing`，实现逐 Block checkpoint；
2. 实现 `run_with_segment_checkpointing`，把连续的多个 Block 组成一个 segment 后再 checkpoint。

这里假定每个 Block 都满足 `block(x) -> x`，不原地修改输入；segment 函数也只接收并返回一个隐藏状态张量。除了写出能运行的代码，还要保证两种实现的输出和输入梯度与普通前向一致，并理解 `segment_size=1` 与更大 segment 在显存和重算次数上的差异。

### 提示

- 逐 Block 版本可以把 `checkpoint(...)` 包在每个 block 前向外面；不要在 TODO 中修改 block 参数或原地改写输入。
- 分段版本要先确定 `[start:end]`，再定义只接收一个 Tensor 的 segment forward 函数。
- 注意 Python 闭包不要错误捕获循环变量；每次循环都要绑定当前 segment。
- `segment_size=1` 在本实现中应与逐 Block checkpoint 对齐；segment 越大，保存点更少，但每次反向重算的片段更长。
- `use_reentrant=False` 是当前 PyTorch 文档中常用的非重入实现；实际项目仍应按当前版本文档和模型约束确认选项。
- 先保证逐层和分段两种实现都能通过 CPU correctness，再比较不同 `segment_size` 的代价。

### 工程要点

- `Gradient Checkpointing` 的本质是 **时间换空间**：减少区段内部部分中间激活的驻留，在反向时重算一小段前向；它不会删除参数、梯度或 optimizer state。
- 这和 `Activation Offload` 的思路不同：checkpointing 更偏向“重算”，offload 更偏向“搬运到别的存储层”。如果你想看机制模拟，后面看 [42 Activation Offload](./42_Activation_Offload.md)。
- 在更深的模型或更长的序列里，激活相关开销可能更突出，因此 checkpointing 可能更有价值；是否有效仍要看参数、梯度、优化器状态和临时张量在总账本中的占比。
- 本节题目区同时检查 API 使用、segment 边界、闭包绑定和梯度正确性；这里不输出通用显存收益结论，真实 GPU 峰值、吞吐和时间收益由 `73 / 76` 在固定 workload 下测量。


```python
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

RUN_REAL_GPU = False
GPU_BATCH_SIZE = 1
GPU_SEQ_LEN = 1024
GPU_DIM = 1024
GPU_NUM_LAYERS = 8

# GPU 峰值实验默认关闭，避免 Notebook 自动抢占学习者正在使用的显存。

```


```python
class SimpleTransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        # 制造一个比较大的激活值内存开销
        out = self.ffn(self.norm(x))
        return x + out

def run_without_checkpointing(blocks: nn.ModuleList, x: torch.Tensor):
    """不使用 checkpoint 执行所有 Block。

    Args:
        blocks: 按顺序排列的 Transformer Block 列表。
        x: 输入隐藏状态，形状通常为 [batch, seq, dim]。

    Returns:
        所有 Block 执行后的隐藏状态。
    """
    for block in blocks:
        x = block(x)
    return x

def run_with_checkpointing(blocks: nn.ModuleList, x: torch.Tensor):
    """对每个 Block 单独启用梯度 checkpoint。

    Args:
        blocks: 按顺序排列的 Transformer Block 列表。
        x: 输入隐藏状态，必须参与梯度计算才能观察反向传播。

    Returns:
        经过逐 Block checkpoint 的输出隐藏状态。

    Note:
        前向阶段少保存中间激活，反向阶段重新执行 Block；
        这里使用 use_reentrant=False。
    """
    for block in blocks:
        # ==========================================
        # TODO 1: 使用 checkpoint 包裹当前 block
        # 提示：使用 use_reentrant=False，并返回新的 x
        # ==========================================
        # x = ???
        pass
    return x

def run_with_segment_checkpointing(blocks: nn.ModuleList, x: torch.Tensor, segment_size: int = 2):
    """将连续 Block 分成固定大小的 segment 后启用 checkpoint。

    Args:
        blocks: 按顺序排列的 Transformer Block 列表。
        x: 输入隐藏状态，形状通常为 [batch, seq, dim]。
        segment_size: 每个 checkpoint 包含的 Block 数量，必须为正数；
            最后一个 segment 可以不足该大小。

    Returns:
        经过分段 checkpoint 的输出隐藏状态。

    Note:
        segment_size=1 接近逐 Block checkpoint；更大的 segment
        通常减少保存点，但可能扩大单次重算范围。
    """
    if segment_size <= 0:
        raise ValueError("segment_size must be positive")

    for start in range(0, len(blocks), segment_size):
        segment = blocks[start:start + segment_size]

        # ==========================================
        # TODO 2: 完成当前 segment 的前向函数并调用 checkpoint
        # 注意：闭包需要绑定本轮的 segment，不能在循环结束后才读取变量
        # ==========================================
        # def segment_forward(hidden, segment=segment): ...
        # x = checkpoint(segment_forward, x, use_reentrant=False)
        pass
    return x

```

### 测试

运行下面的测试单元：默认先验证开启 checkpointing 后的输出与反向传播 correctness。GPU 峰值观察必须显式设置 `RUN_REAL_GPU=True`，因为 Notebook 自动启动 GPU workload 可能抢占学习者已有进程的显存，也可能在不同设备上直接 OOM；显式开关能让学习者先确认 workload、空闲显存和进程状态。这里的 GPU 结果仍只是 toy observation，真实模型的显存、吞吐和 step time 对比请在 `73` 建立 baseline，并在 `76` 比较 checkpoint / offload / hybrid。

```python
# 运行此单元格以测试你的实现
def _run_cpu_correctness_check():
    torch.manual_seed(42)
    dim = 128
    num_layers = 5  # 特意让最后一个 segment 不完整
    blocks = nn.ModuleList([SimpleTransformerBlock(dim) for _ in range(num_layers)])
    x_normal = torch.randn(2, 32, dim, requires_grad=True)
    x_ckpt = x_normal.detach().clone().requires_grad_(True)

    out_normal = run_without_checkpointing(blocks, x_normal)
    loss_normal = out_normal.sum()
    loss_normal.backward()
    grad_normal = x_normal.grad.detach().clone()

    out_ckpt = run_with_checkpointing(blocks, x_ckpt)
    loss_ckpt = out_ckpt.sum()
    loss_ckpt.backward()
    grad_ckpt = x_ckpt.grad.detach().clone()

    assert torch.allclose(out_normal, out_ckpt, atol=1e-5, rtol=1e-4), "checkpoint 前后输出不一致"
    assert torch.allclose(grad_normal, grad_ckpt, atol=1e-5, rtol=1e-4), "checkpoint 前后输入梯度不一致"

    x_segment = x_normal.detach().clone().requires_grad_(True)
    out_segment = run_with_segment_checkpointing(blocks, x_segment, segment_size=2)
    out_segment.sum().backward()
    assert torch.allclose(out_normal, out_segment, atol=1e-5, rtol=1e-4), "分段 checkpoint 前后输出不一致"
    assert torch.allclose(grad_normal, x_segment.grad, atol=1e-5, rtol=1e-4), "分段 checkpoint 前后输入梯度不一致"

    x_one = x_normal.detach().clone().requires_grad_(True)
    out_one = run_with_segment_checkpointing(blocks, x_one, segment_size=1)
    assert torch.allclose(out_ckpt, out_one, atol=1e-5, rtol=1e-4), "segment_size=1 应与逐层 checkpoint 一致"
    print("✅ CPU correctness 测试通过：逐层与分段 checkpoint 的输出和梯度保持一致。")


def _run_gpu_memory_check():
    # 清空显存
    torch.cuda.empty_cache()

    # 模拟一个深度为 20 层，维度很大的网络
    dim = GPU_DIM
    num_layers = GPU_NUM_LAYERS
    blocks = nn.ModuleList([SimpleTransformerBlock(dim) for _ in range(num_layers)]).cuda()

    # 模拟一个极长的序列 (Batch=2, Seq=2048)
    x_input = torch.randn(GPU_BATCH_SIZE, GPU_SEQ_LEN, dim, device='cuda', requires_grad=True)

    print("1. 测试不开启 Checkpointing 的显存占用...")
    torch.cuda.reset_peak_memory_stats()
    out_normal = run_without_checkpointing(blocks, x_input)
    out_normal.sum().backward()
    mem_normal = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"   Peak VRAM (Normal): {mem_normal:.2f} MB")

    del out_normal
    x_input.grad = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("\n2. 测试开启 Checkpointing 的显存占用...")
    out_ckpt = run_with_checkpointing(blocks, x_input)
    out_ckpt.sum().backward()
    mem_ckpt = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"   Peak VRAM (Checkpointing): {mem_ckpt:.2f} MB")

    savings = (1 - mem_ckpt / mem_normal) * 100
    print(f"\n显存节省: {savings:.1f}%")

    if mem_ckpt <= mem_normal:
        print(f"✅ GPU 显存测试通过。显存节省 {savings:.1f}%")
        if savings < 5:
            print(" 注意：显存节省效果较小。这是因为：")
            print("   - 模型层数较少（20层），激活值占总显存比例不高")
            print("   - 在更深的模型或更长的序列中，节省效果可能更明显，但仍需实测")
            print("   - 这里只能说明当前 toy workload 的结果；真实模型需要在固定 workload 下单独测量")
        else:
            print(" 实际显存节省效果取决于模型深度、序列长度和 GPU 架构。")
            print("   在更深的模型或更长的序列中，节省效果可能更明显；具体结果仍取决于 workload。")
    else:
        raise AssertionError("显存占用反而增加了，请检查实现是否正确。")


def test_gradient_checkpointing():
    try:
        _run_cpu_correctness_check()

        if RUN_REAL_GPU and torch.cuda.is_available():
            _run_gpu_memory_check()
        else:
            print("⏭️ GPU 峰值实验默认关闭；如需观察 toy workload，请设置 RUN_REAL_GPU=True，并先确认显存预算。")

    except torch.cuda.OutOfMemoryError as e:
        print(f"⏭️ GPU toy workload 发生 OOM，未形成显存结论：{e}")
        return
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError, RuntimeError) as e:
        if isinstance(e, AttributeError):
            print("代码未完成，无法找到必要的属性")
        elif isinstance(e, NameError):
            print("代码可能未完成，导致了变量未定义")
        elif isinstance(e, TypeError):
            print("代码可能未完成，导致了类型错误")
        elif isinstance(e, ValueError):
            print("代码可能未完成，导致了张量维度错误")
        elif isinstance(e, AssertionError):
            print(f"代码可能未完成，导致了断言失败: {e}")
        else:
            print("代码可能未完成，导致了运行时错误")
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise


test_gradient_checkpointing()

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---
## 参考代码与解析

### 代码

```python
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

RUN_REAL_GPU = False
GPU_BATCH_SIZE = 1
GPU_SEQ_LEN = 1024
GPU_DIM = 1024
GPU_NUM_LAYERS = 8

class SimpleTransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        out = self.ffn(self.norm(x))
        return x + out

def run_without_checkpointing(blocks: nn.ModuleList, x: torch.Tensor):
    """
    不使用 checkpoint 执行所有 Block。

    Args:
        blocks: 按顺序排列的 Transformer Block 列表。
        x: 输入隐藏状态，形状通常为 [batch, seq, dim]。

    Returns:
        所有 Block 执行后的隐藏状态。
    """
    for block in blocks:
        x = block(x)
    return x

def run_with_checkpointing(blocks: nn.ModuleList, x: torch.Tensor):
    """
    对每个 Block 单独启用梯度 checkpoint。

    Args:
        blocks: 按顺序排列的 Transformer Block 列表。
        x: 输入隐藏状态，必须参与梯度计算才能观察反向传播。

    Returns:
        经过逐 Block checkpoint 的输出隐藏状态。

    Note:
        前向阶段少保存中间激活，反向阶段重新执行 Block；
        这里使用 use_reentrant=False。
    """
    for block in blocks:
        # ==========================================
        # TODO 1: 使用 checkpoint 包裹当前 block
        # ==========================================
        x = checkpoint(block, x, use_reentrant=False)
    return x

def run_with_segment_checkpointing(blocks: nn.ModuleList, x: torch.Tensor, segment_size: int = 2):
    """将连续 Block 分成固定大小的 segment 后启用 checkpoint。

    Args:
        blocks: 按顺序排列的 Transformer Block 列表。
        x: 输入隐藏状态，形状通常为 [batch, seq, dim]。
        segment_size: 每个 checkpoint 包含的 Block 数量，必须为正数；
            最后一个 segment 可以不足该大小。

    Returns:
        经过分段 checkpoint 的输出隐藏状态。

    Note:
        segment_size=1 接近逐 Block checkpoint；更大的 segment
        通常减少保存点，但可能扩大单次重算范围。
    """
    if segment_size <= 0:
        raise ValueError("segment_size must be positive")

    for start in range(0, len(blocks), segment_size):
        segment = blocks[start:start + segment_size]

        # ==========================================
        # TODO 2: 完成当前 segment 的前向函数并调用 checkpoint
        # 注意：每轮创建一个新的局部函数，使它绑定当前 segment
        # ==========================================
        def segment_forward(hidden, segment=segment):
            for block in segment:
                hidden = block(hidden)
            return hidden

        x = checkpoint(segment_forward, x, use_reentrant=False)
    return x

```

### 解析

**1. TODO 1：逐 Block checkpoint**
- **实现方式**：`x = checkpoint(block, x, use_reentrant=False)`
- **关键点**：将每个 block 的前向传播包裹在 checkpoint 中，避免保存中间激活值
- **技术细节**：
  - `checkpoint` 函数接收模块和输入参数
  - `use_reentrant=False` 是当前 PyTorch 文档中常用的非重入实现；实际项目仍应按版本文档确认选项
  - checkpoint 减少区段内部部分中间激活的保存；边界输入、输出和其他 Autograd 所需状态仍可能保留
  - 反向传播时从最近的 checkpoint 重新计算前向传播，恢复所需的激活值
  - 每个 block 独立设置 checkpoint，实现细粒度的显存控制

**2. TODO 2：分段 checkpoint**
- **分段方式**：使用 `blocks[start:start + segment_size]` 取出当前连续片段。
- **前向函数**：`segment_forward` 只接收一个隐藏状态，并按顺序执行当前片段中的 Block。
- **闭包绑定**：使用 `segment=segment` 绑定当前循环变量，避免反向重算时所有函数都引用最后一个 segment。
- **尾段处理**：`range(0, len(blocks), segment_size)` 会自然处理不足一个完整 segment 的尾部。
- **粒度含义**：`segment_size=1` 接近逐 Block checkpoint；更大的 segment 通常减少保存点，但单次重算范围更长。

**核心机制**
1. **前向传播阶段**：正常执行前向计算，并减少 checkpoint 区段内部为反向保存的中间激活；具体保留项由实现决定
2. **反向传播阶段**：遇到需要梯度的地方，从最近的 checkpoint 点重新执行前向传播，恢复激活值后立即计算梯度
3. **时间换空间权衡**：重计算会增加 step time，但没有通用的固定百分比；需要在相同 workload 下同时比较峰值显存、吞吐和质量。

**显存节省分析**
- **未启用 checkpoint 的简化上界**：若实现为反向保留每层主要激活，其激活账本可近似随 O(L × B × S × D) 增长，其中 L 是层数，B 是 batch size，S 是序列长度，D 是隐藏维度；这不是完整显存公式。
- **Gradient Checkpointing**：减少部分区段的中间状态驻留；具体峰值取决于 checkpoint 粒度、算子实现和其他显存对象，不能简单写成固定复杂度公式。
- **理论趋势**：segment 越大，保存点越少，但单次重算的片段更长；不能从层数直接推出固定节省比例。
- **实际效果**：取决于模型结构、序列长度、dtype、参数/梯度/优化器状态占比和实现方式；简单模型中总显存变化可能很小。

**工程优化要点**
- **粒度选择**：通常在每个 Transformer Block 级别设置 checkpoint，而非每个子层。过细的粒度会增加重计算开销，过粗的粒度显存节省有限
- **计算开销**：重计算会增加 step time，具体比例需要在相同 workload 下测量，不能预设固定百分比
- **混合策略**：可以只对部分层使用 checkpoint，以平衡显存和速度；“前半段”或“后半段”并不存在对所有模型都成立的固定选择，应由账本和测量决定。
- **长序列训练**：长序列可能放大激活压力，checkpoint 有机会帮助训练越过显存边界，但仍需和减小 micro-batch、offload 或其他显存策略比较。
- **选择性 checkpoint**：可以根据层的类型和账本选择性使用 checkpoint。例如，在 Attention 已由 FlashAttention 等机制降低临时空间后，进一步测量是否只对 FFN 区段 checkpoint 更划算；这不是普遍适用的默认方案。
- **工程实践**：DeepSpeed、Megatron-LM、HuggingFace Transformers 等生态都提供 checkpoint 相关能力，但是否启用、按什么粒度启用，要结合模型、版本和 workload 配置，不能直接视为默认行为或固定标配。
- **与其他优化结合**：checkpoint 可以与混合精度训练、ZeRO优化器、模型并行等技术结合使用，进一步降低显存占用
### 思考与讨论

**1. 为什么本例中显存节省效果不明显（约8%）？**

思考以下因素：
- 激活值占总显存的比例是多少？（总显存 = 模型权重 + 优化器状态 + 激活值 + 梯度）
- 20层模型 vs 50层模型，激活值占比有何不同？
- 序列长度2048 vs 8192，对激活值显存的影响？

**提示**：在简单模型中，权重和优化器状态可能占据大部分显存，激活值占比较小。Checkpoint只能节省激活值部分，因此整体节省比例有限。

**2. 在什么场景下 Gradient Checkpointing 可能更需要优先评估？**

考虑以下场景：
- 训练 LLaMA-70B（80层，8k隐藏维度）
- 长上下文训练（32k tokens）
- 有限的GPU显存（如单卡A100 80GB）

**提示**：当主要瓶颈是激活显存，且减小 micro-batch 或序列长度不可接受时，checkpoint 可能是重要选择之一；也可以结合 offload、混合精度或其他显存策略。

**3. Checkpoint 的粒度如何选择？**

对比以下策略：
- **细粒度**：每一层都checkpoint → 显存节省最多，但重计算开销大
- **中粒度**：每个 Transformer Block checkpoint → 可作为平衡显存和速度的测量起点
- **粗粒度**：每4-8层checkpoint → 重计算开销小，但显存节省有限

**提示**：Transformer Block 常被作为一个便于配置和测量的边界，但最佳粒度仍需结合模型结构、版本和 workload 验证。

**4. Checkpoint 与其他优化技术如何结合？**

思考以下组合：
- Checkpoint + 混合精度训练（AMP）
- Checkpoint + ZeRO优化器（分布式训练）
- Checkpoint + FlashAttention（Attention层优化）
- Checkpoint + 模型并行（Tensor/Pipeline Parallelism）

**提示**：这些技术可能叠加使用。例如，若 FlashAttention 已降低 Attention 的临时空间，可将“只对 FFN 区段 checkpoint”作为待测假设；ZeRO 主要影响参数、梯度或优化器状态，checkpoint 主要影响激活驻留，二者在对象层面可能互补。

**5. 如何评估 Checkpoint 的收益？**

计算以下指标：
- **显存节省率** = (Normal VRAM - Checkpoint VRAM) / Normal VRAM
- **时间增加率** = (Checkpoint Time - Normal Time) / Normal Time
- **性价比** = 显存节省率 / 时间增加率

**提示**：这个比值只是课堂上的辅助指标，不能单独决定是否采用。最终应同时检查 OOM 边界、总峰值显存、step time、吞吐和训练质量，并交给 73/76 的固定 workload 实验确认。
### 进阶实验（可选）

以下实验供有兴趣的学习者探索，不计入必做题目。

#### 实验1：选择性 Checkpoint

实现一个函数，每隔 N 层使用一次 checkpoint，对比不同 N 值的效果：

```python
def run_with_selective_checkpointing(blocks: nn.ModuleList, x: torch.Tensor, checkpoint_every_n: int = 2):
    """
    每隔 N 层使用一次 checkpoint
    
    Args:
        blocks: 模型层列表
        x: 输入张量
        checkpoint_every_n: 每隔多少层使用一次checkpoint
    """
    for i, block in enumerate(blocks):
        if i % checkpoint_every_n == 0:
            x = checkpoint(block, x, use_reentrant=False)
        else:
            x = block(x)
    return x

# 实验：对比 checkpoint_every_n = 1, 2, 4, 8 的显存占用和时间开销
```

**思考**：
- checkpoint_every_n = 1（每层都checkpoint）vs checkpoint_every_n = 20（不使用checkpoint）
- 找到最佳的平衡点

#### 实验2：分段 Checkpoint

将模型分成若干段，每段作为一个 checkpoint：

```python
def run_with_segment_checkpointing(blocks: nn.ModuleList, x: torch.Tensor, num_segments: int = 4):
    """
    将模型分成若干段，每段作为一个 checkpoint
    
    Args:
        blocks: 模型层列表
        x: 输入张量
        num_segments: 分成多少段
    """
    segment_size = len(blocks) // num_segments
    
    for i in range(0, len(blocks), segment_size):
        segment_blocks = blocks[i:i+segment_size]
        
        # 定义segment的前向传播函数
        def segment_forward(x, segment_blocks=segment_blocks):
            for block in segment_blocks:
                x = block(x)
            return x
        
        x = checkpoint(segment_forward, x, use_reentrant=False)
    
    return x

# 实验：对比 num_segments = 1, 2, 4, 10, 20 的效果
```

**思考**：
- 分段checkpoint vs 逐层checkpoint，哪个更高效？
- 为什么实际训练框架通常需要配置 checkpoint 的粒度？

#### 实验3：混合策略

先提出一个待验证假设：只对后半部分层使用 checkpoint，观察不同边界对显存和时间的影响：

```python
def run_with_hybrid_checkpointing(blocks: nn.ModuleList, x: torch.Tensor, checkpoint_start_layer: int = 10):
    """
    只对指定层之后的层使用 checkpoint
    
    Args:
        blocks: 模型层列表
        x: 输入张量
        checkpoint_start_layer: 从第几层开始使用checkpoint
    """
    for i, block in enumerate(blocks):
        if i >= checkpoint_start_layer:
            x = checkpoint(block, x, use_reentrant=False)
        else:
            x = block(x)
    return x

# 实验：对比 checkpoint_start_layer = 0, 5, 10, 15 的效果
```

**思考**：
- 为什么前几层可以不使用checkpoint？
- 如何确定最佳的 checkpoint_start_layer？

#### 实验4：显存-时间权衡曲线

绘制不同checkpoint策略的显存-时间权衡曲线：

```python
import matplotlib.pyplot as plt

strategies = [
    ("No Checkpoint", lambda blocks, x: run_without_checkpointing(blocks, x)),
    ("Full Checkpoint", lambda blocks, x: run_with_checkpointing(blocks, x)),
    ("Every 2 Layers", lambda blocks, x: run_with_selective_checkpointing(blocks, x, 2)),
    ("Every 4 Layers", lambda blocks, x: run_with_selective_checkpointing(blocks, x, 4)),
    ("4 Segments", lambda blocks, x: run_with_segment_checkpointing(blocks, x, 4)),
    ("Hybrid (start=10)", lambda blocks, x: run_with_hybrid_checkpointing(blocks, x, 10)),
]

memory_usage = []
time_cost = []

for name, strategy in strategies:
    # 测量显存和时间
    # ... (实现测量逻辑)
    pass

# 绘制曲线
plt.scatter(memory_usage, time_cost)
for i, (name, _) in enumerate(strategies):
    plt.annotate(name, (memory_usage[i], time_cost[i]))
plt.xlabel('Peak Memory (MB)')
plt.ylabel('Time (ms)')
plt.title('Memory-Time Tradeoff of Different Checkpoint Strategies')
plt.show()
```

**思考**：
- 哪个策略的性价比最高？
- 在不同的模型规模下，最佳策略是否相同？

---

**进阶阅读**：
- [PyTorch Checkpoint 官方文档](https://pytorch.org/docs/stable/checkpoint.html)
- [DeepSpeed Activation Checkpointing](https://www.deepspeed.ai/tutorials/megatron/)
- [Gradient Checkpointing 论文](https://arxiv.org/abs/1604.06174)