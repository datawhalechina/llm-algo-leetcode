# 01. SFT Data and Loss | SFT 数据与监督口径

## 页面目标

这一页回答一个最基础的问题：一条 `prompt / response` 样本，为什么能变成可训练的监督信号，以及监督信号到底落在哪些 token 上。

## 你要先确认什么

- `input_ids` 是否正确拼接了上下文和 response。
- `attention_mask` 是否正确屏蔽 padding。
- `labels` 是否只在需要监督的区间计 loss。
- `response-only loss` 是否和训练目标一致。

## 演化路径

SFT 不是先写 loss，而是先把样本拆成可监督的三件套。

1. 先定义 `prompt / response` 的边界。
2. 再把样本编码成 `input_ids`。
3. 再用 `attention_mask` 控制 padding 不参与计算。
4. 再把 `labels` 只放在 response 区间。
5. 最后对齐 shift logits，确保监督目标和预测位置一致。

这一步的核心不是公式，而是监督口径。
一旦 `labels`、mask 和 shift 对不齐，后面的 loss 曲线即使在下降，也不一定有意义。

## 常见误区

- 把 prompt 也算进 loss，导致模型学到错误对齐。
- response 为空或被截断，但数据管道没有显式报错。
- padding token 参与训练，污染监督信号。
- EOS 位置和监督区间没有统一。

## 经典阅读入口

- [09 SFT Training Loop](../../../02_PyTorch_Algorithms/09_SFT_Training_Loop.md)
- [17 Autograd Basics](../../../02_PyTorch_Algorithms/17_Autograd_Basics.md)
- [18 Activation and Loss Backward](../../../02_PyTorch_Algorithms/18_Activation_and_Loss_Backward.md)

## 前置关系

- 先看 `model_architecture` 里的 `01-05`，确认模型主体和 block 结构。
- 再回到这里看数据和 loss，理解训练信号是怎么落进去的。

## 本节要点

这一页的目标不是“解释 loss”，而是把监督边界定清楚。
只有监督口径清楚，后面的 LoRA、训练控制和项目交付才有意义。
