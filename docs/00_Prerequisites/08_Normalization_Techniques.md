# Normalization Techniques Practice | Normalization Techniques - 练习

**难度：** Medium | **标签：** `PyTorch`, `BatchNorm`, `LayerNorm`, `Normalization` | **目标人群：** Chapter 0 入门学习者

> 🚀 **云端运行环境**
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/08_Normalization_Techniques.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

本练习配套导学：[Chapter 0 导学](./intro.md)

## 🎯 学习目标
- 掌握 BatchNorm 和 LayerNorm 的手写实现
- 理解训练态统计量与推理态统计量的区别
- 为 Transformer 中的归一化结构打基础

## 核心练习
- `batch_norm_train`
- `batch_norm_eval`
- `layer_norm_last_dim`
- `update_running_stats`

## 练习提示
- BatchNorm 通常沿 batch 维统计均值和方差。
- LayerNorm 通常沿特征维统计均值和方差。
- 训练态和推理态不要混用同一组统计量。
