# 04. PyTorch nn.Module Basics Practice | PyTorch nn.Module Basics - 练习

**难度：** Medium | **标签：** `PyTorch`, `nn.Module`, `Parameters` | **目标人群：** PyTorch 入门学习者

> 🚀 **云端运行环境**
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/04_PyTorch_nn_Module_Basics_Practice.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

本练习配套导学：[Chapter 0 导学](./intro.md)

## 🎯 学习目标
- 正确继承 `nn.Module`
- 理解参数注册与 `state_dict`
- 学会组合基础模块构造 MLP

## 核心练习
- `SimpleLinear`
- `TwoLayerMLP`
- `count_parameters(module)`

## 练习提示
- 所有可训练参数都应通过 `nn.Parameter` 或子模块注册
- `forward()` 只负责前向传播，不要写训练逻辑
- `state_dict()` 是保存和加载模型的标准接口
