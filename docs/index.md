# LLM-Algo-LeetCode

> 🧪 Beta公测版本提示：教程主体代码与算子已基本构建完成，正在持续优化文档细节与补充注释。欢迎大家提交 Issue 反馈问题或贡献 PR！

## 🎯 项目简介

本项目旨在为大语言模型（LLM）算法工程师、AI 基础设施（AI Infra）工程师以及研究实习生等岗位的候选人，提供一个**系统性、可交互、带评测**的工程实战指南。

与传统的"文字版八股文面经"不同，本项目严格限定于**纯粹的大语言模型 (LLM) 领域**（不包含 Diffusion 或多模态）。我们提取了现代大模型架构中最核心的底层算法与系统设计，将其封装为独立的 Jupyter Notebook 填空题，并配备本地测试用例，力求提供类似 LeetCode 的刷题体验。

## 👥 项目受众

- **求职面试者**：巩固 LLM 算法工程师、AI 架构师、算子开发工程师的高频底层考点。
- **AI 研发人员**：希望从代码底层理解大模型运作机制（如分布式通信、显存优化、Triton/CUDA 算子）的从业者。
- **前置要求**：具备 Python 和深度学习基础，熟悉 PyTorch。高级章节需一定 C++/CUDA 基础。

## ✨ 项目特点

1. **高度垂直**：专注 Transformer、MoE、量化、推理加速与显存优化。
2. **工程导向**：要求使用 PyTorch、Triton 或原生的 CUDA C++ 实现核心算子和系统逻辑。
3. **测试驱动 (Test-Driven)**：每一道题都内置了与工业界开源实现（如 HuggingFace, vLLM）对齐的测试验证，确保输出张量维度与数值的完全正确。并包含详尽的 Benchmark 性能基准测试。
4. **由浅入深**：分为"理论基础"、"模型组装与训练"、"底层计算加速"三个阶段，覆盖从入门到高阶架构的完整知识链路。

## 📚 章节概览

| 模块 | 简介 | 状态 |
| ---- | ---- | ---- |
| [**Chapter 0: 前置知识与环境准备**](./00_Prerequisites/intro.md) | 为零基础学习者提供平滑的入门路径。包含 Python 基础、NumPy 操作、PyTorch 核心概念、Profiling 工具、显存优化与调试技巧。 | 🚧 部分完成 |
| [**Chapter 1: 硬件、算力推导与系统级理论**](./01_Hardware_Math_and_Systems/intro.md) | 包含系统架构与性能优化的高频问答题，适合作为面试前的快速复习材料。涵盖 GPU 架构、显存估算、通信拓扑与国产芯片概览。 | 🚧 理论完成 |
| [**Chapter 2: PyTorch 核心算法实现实战**](./02_PyTorch_Algorithms/intro.md) | 核心代码实战区。包含 MHA/GQA、RoPE、MoE、SFT、LoRA、RLHF (PPO/DPO)、推理加速 (Speculative Decoding) 与分布式模拟 (ZeRO/TP/PP) 等前沿算法。 | ✅ 完成 |
| [**Chapter 3: CUDA C++ 与 Triton 算子开发**](./03_CUDA_and_Triton_Kernels/intro.md) | 针对算子加速与高阶架构。包含 Triton Fused 算子、FlashAttention、PagedAttention、以及原生 CUDA 共享内存优化。需 GPU 环境。 | ✅ 完成 |

## 🚀 快速开始

### 在线学习（零配置，推荐新手）

1. 在左侧侧边栏选择感兴趣的章节
2. 点击 **📖 完整导学** 了解学习路径
3. 选择具体题目开始学习
4. 点击题目顶部的 **"Open In Colab"** 或 **"Open In ModelScope"** 徽章，在免费云端 GPU 环境中运行代码

### 本地学习

如果你有本地开发环境，可以克隆仓库到本地：

```bash
git clone https://github.com/datawhalechina/llm-algo-leetcode.git
cd llm-algo-leetcode
```

详细的本地环境搭建和使用方法，请查看 [使用指南](./guide.md)。

## 📖 更多资源

- [使用指南](./guide.md) - 四种学习方式详细对比
- [贡献指南](./contributing.md) - 如何参与项目开发和测试
- [GitHub 仓库](https://github.com/datawhalechina/llm-algo-leetcode) - 源代码和问题反馈

## 📄 开源协议

本作品采用 [知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议](http://creativecommons.org/licenses/by-nc-sa/4.0/) 进行许可。
