# 01. NumPy and Einsum Practice | NumPy and Einsum - 练习

**难度：** Easy | **标签：** `NumPy`, `Broadcasting`, `Einsum` | **目标人群：** Chapter 0 入门学习者

> 🚀 **云端运行环境**
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/01_NumPy_and_Einsum_Practice.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

本练习配套导学：[Chapter 0 导学](./intro.md)

## 🎯 学习目标
- 掌握 NumPy 广播与矩阵乘法
- 会用 `einsum` 表达多维张量运算
- 为 Attention 和 Tensor 形状推导打底

## 核心练习
- `build_causal_mask(seq_len)`
- `matmul_einsum(a, b)`
- `batch_attention_scores(q, k)`
- `rms_normalize(x, eps)`

## 练习提示
- 先确认张量 shape，再写公式
- 注意广播维度是否对齐
- `einsum` 适合表达复杂但规律清晰的张量运算
