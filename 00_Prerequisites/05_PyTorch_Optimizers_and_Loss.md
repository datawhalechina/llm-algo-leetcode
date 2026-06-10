# 05. PyTorch Optimizers and Loss Practice | PyTorch Optimizers and Loss - 练习

**难度：** Medium | **标签：** `PyTorch`, `Loss`, `Optimizer` | **目标人群：** PyTorch 入门学习者

> 🚀 **云端运行环境**
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/05_PyTorch_Optimizers_and_Loss_Practice.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

本练习配套导学：[Chapter 0 导学](./intro.md)

## 🎯 学习目标
- 理解 MSE / CrossEntropy 的基本形式
- 掌握 SGD / Adam 的调用方式
- 完成一个最小训练步

## 核心练习
- `mse_loss(pred, target)`
- `cross_entropy_loss(logits, target)`
- `train_one_step(model, x, target, optimizer)`

## 练习提示
- 先算 loss，再 backward，再 step
- 使用 `optimizer.zero_grad()` 清空梯度
- 在小样本上验证 loss 是否下降
