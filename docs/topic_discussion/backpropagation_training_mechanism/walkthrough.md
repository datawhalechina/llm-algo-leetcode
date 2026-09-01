# 反向传播与训练机制深入阅读

假设你接手的是一条新的训练链路：forward 能跑，loss 也有数，但你解释不清梯度到底怎么回去，为什么某些张量必须保留，为什么加了 checkpointing 以后显存下来了、训练节奏却变了。

这条线最重要的是按暴露顺序判断：先看梯度路径，再看保存点，再看显存和训练节奏怎样一起被 backward 改写。

对应专题正文：[01 反向传播总览与计算图](./01_backpropagation_and_graph.md)。先建立梯度路径和保存边界。

## 第一段：先把计算图看清

故事通常从最基础的问题开始：forward 看起来没问题，但一到 backward 就只剩 API 名字。第一步要先把计算图和梯度路径画清，知道梯度沿什么链路回传，哪些节点只是算子，哪些节点会决定后面必须保存什么状态。

这一步对应 [01 反向传播总览与计算图](./01_backpropagation_and_graph.md)。

## 第二段：再把 autograd 和 attention backward 对上

一旦进入 attention，问题就不再只是“梯度能不能回去”，而是“回去时到底保存了什么、重算了什么”。这时要去看 `grad_fn`、`saved_tensors` 和 attention 的 `dV -> dP -> dS -> dQ/dK` 链路。很多人真正卡住的不是公式，而是公式和执行路径没对上。

这一步对应 [02 自动微分与 Attention 反向传播](./02_autograd_and_attention_backward.md)。

## 第三段：loss 对齐以后，显存账本才有意义

如果 `labels / mask / shift / ignore_index` 没对齐，后面的 loss 曲线本身就不可信。只有监督口径成立了，activation、参数、梯度和 optimizer state 的显存账本才值得继续分析。

这一步对应 [03 损失对齐与显存账本](./03_loss_alignment_memory_ledger.md)。

## 第四段：checkpointing 与 offload 改的不是同一类代价

训练侧显存一高，最容易想到的就是 checkpointing 和 offload。但这两条线改的不是同一种东西：checkpointing 是重算换空间，offload 是搬运换空间。它们都能压显存，但会把训练节奏改成不同的样子。

这一步对应 [04 Checkpointing 与 Offload](./04_checkpointing_and_offload.md)。

## 第五段：最后回到训练节奏

真正的闭环结束点不是“显存降了”，而是 accumulation、optimizer step、effective batch 和 profiling 口径都统一了。把这条故事走完以后，一个更像真实结论的说法通常不是“我们用了 checkpointing”，而是：梯度路径清楚、监督口径正确、activation 保存点明确，最终训练节奏和显存代价都能被解释。

这一步对应 [05 梯度累积、训练闭环与 Profiling](./05_accumulation_decision_profiling.md)，并回到 [Part 02 · 73 训练性能分析](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.md) → [Part 02 · 76 激活检查点与 Offload benchmark](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.md) → [Part 02 · 75 显存预算压缩](../../02_PyTorch_Algorithms/75_Memory_Budget_Compression_Project.md) 验证。
