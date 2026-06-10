# LLM-Algo-LeetCode

> 🧪 Beta公测版本提示：教程主体代码与算子已基本构建完成，正在持续优化文档细节与补充注释。欢迎大家提交 Issue 反馈问题或贡献 PR！

## 🎯 项目简介

这是一个面向大模型入门到进阶的算法刷题教程，聚焦纯 LLM 领域，把底层算法与系统设计整理成可运行、可验证、可回顾的 Jupyter Notebook 练习。

详细的环境边界、学习方式和章节 GPU 要求见 [使用指南](./guide.md)。

## 📚 章节概览

| 模块 | 简介 | 状态 |
| ---- | ---- | ---- |
| [**Chapter 0: 前置知识与环境准备**](./00_Prerequisites/intro.md) | 零基础入门路径。当前已补齐 00/01/04/05/07/08/09/12/13 的练习资产，02/03/06/10/11 的理论文档也已就绪。 | 🚧 部分完成 |
| [**Chapter 1: 硬件、算力推导与系统级理论**](./01_Hardware_Math_and_Systems/intro.md) | 包含系统架构与性能优化的高频问答题，适合作为面试前的快速复习材料。涵盖 GPU 架构、显存估算、通信拓扑与国产芯片概览。 | 🚧 理论完成 |
| [**Chapter 2: PyTorch 算法实战**](./02_PyTorch_Algorithms/intro.md) | PyTorch 级别的大模型实现。<br>组页：<br>[2.1](./02_PyTorch_Algorithms/2_1.md) · [2.2](./02_PyTorch_Algorithms/2_2.md) · [2.3](./02_PyTorch_Algorithms/2_3.md) · [2.4](./02_PyTorch_Algorithms/2_4.md) · [2.5](./02_PyTorch_Algorithms/2_5.md) · [2.6](./02_PyTorch_Algorithms/2_6.md) · [2.7](./02_PyTorch_Algorithms/2_7.md) · [2.8](./02_PyTorch_Algorithms/2_8.md) | ✅ 完成 |
| [**Chapter 3: CUDA C++ 与 Triton 算子开发**](./03_CUDA_and_Triton_Kernels/intro.md) | 高性能算子实现。<br>组页：<br>[3.1](./03_CUDA_and_Triton_Kernels/3_1.md) · [3.2](./03_CUDA_and_Triton_Kernels/3_2.md) · [3.3](./03_CUDA_and_Triton_Kernels/3_3.md) · [3.4](./03_CUDA_and_Triton_Kernels/3_4.md) · [3.5](./03_CUDA_and_Triton_Kernels/3_5.md) | ✅ 完成 |

## 🚀 快速开始

1. 在左侧侧边栏选择感兴趣的章节
2. 点击 **📖 完整导学** 了解学习路径
3. 选择具体题目开始学习
4. 环境和平台差异见 [使用指南](./guide.md)

## 📖 更多资源

- [使用指南](./guide.md) - 环境与学习方式
- [贡献指南](./contributing.md) - 如何参与项目开发和测试
- [维护与发布手册](./maintenance.md) - 章节、链接、测试与发布的维护约定
- [自动化测试脚本索引](../project_test_scripts.md) - 各类验证脚本入口

## 📄 开源协议

本作品采用 [知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议](http://creativecommons.org/licenses/by-nc-sa/4.0/) 进行许可。
