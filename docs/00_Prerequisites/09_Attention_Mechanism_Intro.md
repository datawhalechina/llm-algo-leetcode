# Attention Mechanism Intro Practice | Attention Mechanism Intro - 练习

**难度：** Medium | **标签：** `PyTorch`, `Attention`, `Mask`, `Softmax` | **目标人群：** Chapter 0 入门学习者

> 🚀 **云端运行环境**
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/09_Attention_Mechanism_Intro.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

本练习配套导学：[Chapter 0 导学](./intro.md)

## 🎯 学习目标
- 手写 scaled dot-product attention 的核心步骤
- 理解 causal mask 的作用
- 为 Chapter 2 的 Attention 题目打底

## 核心练习
- `build_causal_mask`
- `masked_softmax`
- `attention_weights`
- `scaled_dot_product_attention`

## 练习提示
- softmax 前做数值稳定性处理。
- mask 应该屏蔽未来 token。
- q/k/v 的最后一维通常是特征维。
