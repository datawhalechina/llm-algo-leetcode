# 06. Visual Assets | Visual Assets

## 页面目标

本页集中列出反向传播专题所需的关键图，作为机制链路的视觉索引。

## 图册

### 1. 反向传播与训练闭环总图

> 图册占位：训练闭环决策图尚未生成。

这张图负责把 backward、调度和 profiling 的关系拉直，适合作为专题总览。

### 2. Attention Backward 图

> 图册占位：Attention backward 图尚未生成。

这张图负责把 `dV -> dP -> dS -> dQ / dK` 的反向顺序固定下来。

### 2.1 Naive vs FlashAttention Backward 对照图

> 图示占位：Naive 与 FlashAttention Backward 对照图尚未纳入镜像，当前以本页文字说明为准。

这张图负责说明现代实现差异：

- 哪些中间状态被完整保存
- 哪些状态在 backward 中重算
- 哪些路径通过 fused kernel 减少了中间写回

### 3. 显存账本图

> 图册占位：显存账本图尚未生成。

这张图负责说明 backward 为什么会先吃满显存，以及 activation / parameter / optimizer state 的差别。

### 4. Checkpointing 取舍图

> 图册占位：Checkpointing 取舍图尚未生成。

这张图负责说明“重算换显存”的代价模型。

### 5. Offload 取舍图

> 图册占位：Offload 取舍图尚未生成。

这张图负责说明“搬运换显存”的代价模型。

### 6. 标签对齐图

> 图册占位：标签对齐图尚未生成。

这张图负责说明 prompt、response、mask 和 loss 的监督边界。

## 使用方式

- 先看总图，确认机制链路。
- 再看 attention 图，确认梯度回传顺序。
- 再用对照图确认现代实现里哪些状态是“保存”、哪些是“重算”、哪些是“融合带过”。
- 再看显存账本图，确认为什么要做优化。
- 最后看两类取舍图和标签对齐图，确认怎么选方案。

## 相关跳转

- 回到 [反向传播与训练机制专题入口](./intro.md)
- 回到 [反向传播与训练机制正文](./casebook.md)
- 回到 [反向传播与训练机制深入阅读](./walkthrough.md)
