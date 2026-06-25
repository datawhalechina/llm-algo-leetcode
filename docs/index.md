<h1 align="center"> 大模型算法实战教程 / LLM Algorithm Practice Lab </h1>

本项目原名 `LLM-Algo-LeetCode`，现统一更名为“大模型算法实战教程 / LLM Algorithm Practice Lab”。

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体代码与算子已基本构建完成，正在持续优化文档细节与补充注释。欢迎大家提交 Issue 反馈问题或贡献 PR！

> 大模型算法实战教程 / LLM Algorithm Practice Lab
>
> A practical LLM algorithm tutorial with theory, walkthroughs, test cases, and solutions.

[中文版 (Chinese)](#中文版) | [English Version](#english-version)

---

# 中文版

## 🎯 项目简介

这是一个面向大模型入门到进阶的算法实战教程，聚焦 Python、PyTorch、Transformer、推理优化、显存管理与 CUDA/Triton 实战。我们把每个知识点做成可运行、可验证、可回顾的 Jupyter Notebook 练习，帮助你从“会看”走到“会写、会调、会优化”。

本项目仅限于纯粹的大语言模型（LLM）领域，不包含 Diffusion 或多模态，并配备本地测试用例，提供可反复练习和回顾的学习路径。

### ✨ 项目特点

1. **高度垂直**：专注 Transformer、MoE、量化、推理加速与显存优化。
2. **工程导向**：使用 PyTorch、Triton 或 CUDA C++ 实现核心算子和系统逻辑。
3. **测试驱动**：每道题都配套本地测试和性能验证。

### 👥 项目受众

- **求职面试者**：巩固 LLM 算法工程师、AI 架构师、算子开发工程师的高频考点。
- **AI 研发人员**：从代码底层理解分布式通信、显存优化与 Triton/CUDA 算子。
- **前置要求**：具备 Python 和深度学习基础，熟悉 PyTorch；高阶内容需要一定 C++/CUDA 基础。

## 🆕 更新时间线

- **2026-06-15**：[最新更新点]推进第零部分 / 第一部分的分组与导读收口，统一部分级导航，并完成网页底部评论区接入 GitHub Discussions，同时持续扩展第一部分的正文、桥接页与 Notebook 结构。
- **2026-06-13**：[最新更新点]：修复 dead link，并为未完成页面补充占位页，避免学习入口出现 404。
- **2026-04-21**：更新 Colab 徽章链接，统一指向官方 `datawhalechina` 仓库。
- **2026-04-20**：上线站点首页与部分导学；新增第零部分前置知识与第一部分练习内容，完善在线阅读入口与学习路径。
- **2026-04-18 ~ 2026-04-19**：集中重构第二部分 / 第三部分内容，优化 Notebook、答案区与算子实现说明。
- **2026-04-02**：完成教程核心 Notebook、文档与测试脚本的初始搭建。

## 📚 部分总览

| 部分 | 简介 | 组别 | 状态 |
| ---- | ---- | ---- | ---- |
| 第零部分 | 前置知识与环境准备。 | [`0A Python 基础`](./00_Prerequisites/0A.md) / [`0B PyTorch 基础`](./00_Prerequisites/0B.md) / [`0C 深度学习基础`](./00_Prerequisites/0C.md) / [`0D 工具与调试`](./00_Prerequisites/0D.md) | 🚧 部分完成 |
| 第一部分 | 硬件、算力推导与系统级理论。 | [`1A 数值基础与算力估算`](./01_Hardware_Math_and_Systems/1A.md) / [`1B 单卡硬件与访存优化`](./01_Hardware_Math_and_Systems/1B.md) / [`1C 多卡通信与显存共享`](./01_Hardware_Math_and_Systems/1C.md) / [`1D 异构调度与算子编程`](./01_Hardware_Math_and_Systems/1D.md) / [`1E 编译优化与算力生态`](./01_Hardware_Math_and_Systems/1E.md) | 🚧 理论完成 |
| 第二部分 | PyTorch 算法实战。 | [`2.1 基础算子`](./02_PyTorch_Algorithms/intro.md) / [`2.2 模型架构`](./02_PyTorch_Algorithms/intro.md) / [`2.3 微调与训练技术`](./02_PyTorch_Algorithms/intro.md) / [`2.4 对齐技术`](./02_PyTorch_Algorithms/intro.md) / [`2.5 反向传播与显存优化`](./02_PyTorch_Algorithms/intro.md) / [`2.6 核心推理优化`](./02_PyTorch_Algorithms/intro.md) / [`2.7 高级推理优化`](./02_PyTorch_Algorithms/intro.md) / [`2.8 分布式与扩展`](./02_PyTorch_Algorithms/intro.md) | ✅ 完成 |
| 第三部分 | Triton 算子开发。 | [`3.1 基础篇`](./03_Triton_Kernels/intro.md) / [`3.2 过渡篇`](./03_Triton_Kernels/intro.md) / [`3.3 进阶A：Attention优化`](./03_Triton_Kernels/intro.md) / [`3.4 进阶B：推理优化`](./03_Triton_Kernels/intro.md) / [`3.5 项目篇`](./03_Triton_Kernels/intro.md) | ✅ 完成 |
| 第四部分 | CUDA C++ 与系统优化。 | [`4.1 CUDA 编程基础`](./04_CUDA_and_System_Optimization/intro.md) / [`4.2 系统级性能优化`](./04_CUDA_and_System_Optimization/intro.md) / [`4.3 分布式训练工程`](./04_CUDA_and_System_Optimization/intro.md) / [`4.4 架构视野`](./04_CUDA_and_System_Optimization/intro.md) | 🚧 规划中 |

## 🚀 快速开始

如果你想开始学习，直接从在线站点或左侧部分目录进入即可。环境选择和平台差异统一看 [使用指南](./guide.md)。

### 学习路径

1. 在左侧侧边栏选择感兴趣的部分
2. 点击 **📖 完整导学** 了解学习路径
3. 选择具体题目开始学习
4. 环境和平台差异见 [使用指南](./guide.md)

### 方式 1：在线阅读

访问在线站点：

[https://datawhalechina.github.io/llm-algo-leetcode/](https://datawhalechina.github.io/llm-algo-leetcode/)

适合：
- 先看目录
- 先读部分导学
- 先用 Colab 跑练习

### 方式 2：本地学习

```bash
git clone https://github.com/datawhalechina/llm-algo-leetcode.git
cd llm-algo-leetcode
conda env create -f environment.yml
conda activate llm_algo
jupyter lab
```

更完整的环境分层、GPU/CUDA、CNB 和 Docker 说明请见 [使用指南](./guide.md)。

## 📖 更多资源

- [使用指南](./guide.md) - 环境与学习方式
- [贡献指南](./contributing.md) - 如何参与项目开发和测试
- [维护与发布手册](./maintenance.md) - 部分、链接、测试与发布的维护约定
- [自动化测试脚本索引](../project_test_scripts.md) - 各类验证脚本入口

---

# English Version

This is a practical LLM algorithm tutorial that covers Python, PyTorch, Transformers, inference optimization, VRAM management, and CUDA/Triton practice. It is not just theory: each concept is turned into a runnable, verifiable, and reviewable exercise so you can move from "reading" to "writing, debugging, and optimizing".

This project is strictly limited to the Large Language Model (LLM) domain, excluding Diffusion and multimodal content. We package the core algorithms and system design ideas of modern LLM architectures into independent Jupyter Notebook fill-in-the-blank exercises with local test cases, providing a repeatable learning path for practice and review.

### Features

1. **Highly Vertical**: Focuses on Transformers, MoE, Quantization, Inference Acceleration, and VRAM Optimization.
2. **Engineering-Oriented**: Uses PyTorch, Triton, or native CUDA C++ to implement core operators and system logic.
3. **Test-Driven**: Every exercise includes industrial-aligned tests and performance validation.
4. **Step-by-Step**: Covers fundamentals, model assembly/training, and low-level compute acceleration.

### Target Audience

- **Job Seekers**: Reinforce the most common LLM algorithm, AI infrastructure, and kernel optimization interview topics.
- **AI Practitioners**: Learn LLM mechanisms from the code level, including distributed communication, VRAM optimization, and Triton/CUDA operators.
- **Prerequisites**: Basic Python and deep learning knowledge, plus PyTorch familiarity. Advanced parts require some C++/CUDA background.

## Update Timeline

- **2026-06-15**: [Latest update] finalized the Part 0 / 1 grouping and guide cleanup, unified the part-level navigation, connected the page comments to GitHub Discussions, and continued expanding Part 1 content, bridge pages, and notebook structure.
- **2026-06-13**: [Latest update] fixed dead links and added placeholder pages for unfinished content to prevent 404s in learning entry points.
- **2026-04-21**: Updated Colab badges to point to the official `datawhalechina` repository.
- **2026-04-20**: Launched the site homepage and part guides; added Part 0 prerequisites and Part 1 practice content to unify the learning path.
- **2026-04-18 ~ 2026-04-19**: Refactored Part 2 / 3 content, polishing notebooks, answer sections, and operator implementation notes.
- **2026-04-02**: Completed the initial tutorial notebooks, docs, and test scripts.

## Part Overview

| Part | Groups | Summary | Status | Entry |
| ---- | ---- | ------- | ---- | ---- |
| Part 0 | [`0A Python 基础`](./00_Prerequisites/0A.md) / [`0B PyTorch 基础`](./00_Prerequisites/0B.md) / [`0C 深度学习基础`](./00_Prerequisites/0C.md) / [`0D 工具与调试`](./00_Prerequisites/0D.md) | Prerequisites and environment setup. | 🚧 Partial | [Guide](./00_Prerequisites/intro.md) |
| Part 1 | [`1A 数值基础与算力估算`](./01_Hardware_Math_and_Systems/1A.md) / [`1B 单卡硬件与访存优化`](./01_Hardware_Math_and_Systems/1B.md) / [`1C 多卡通信与显存共享`](./01_Hardware_Math_and_Systems/1C.md) / [`1D 异构调度与算子编程`](./01_Hardware_Math_and_Systems/1D.md) / [`1E 编译优化与算力生态`](./01_Hardware_Math_and_Systems/1E.md) | Hardware, compute estimation, and system-level theory. | 🚧 Theory complete | [Guide](./01_Hardware_Math_and_Systems/intro.md) |
| Part 2 | [`2.1 基础算子`](./02_PyTorch_Algorithms/intro.md) / [`2.2 模型架构`](./02_PyTorch_Algorithms/intro.md) / [`2.3 微调与训练技术`](./02_PyTorch_Algorithms/intro.md) / [`2.4 对齐技术`](./02_PyTorch_Algorithms/intro.md) / [`2.5 反向传播与显存优化`](./02_PyTorch_Algorithms/intro.md) / [`2.6 核心推理优化`](./02_PyTorch_Algorithms/intro.md) / [`2.7 高级推理优化`](./02_PyTorch_Algorithms/intro.md) / [`2.8 分布式与扩展`](./02_PyTorch_Algorithms/intro.md) | PyTorch algorithm practice. | ✅ Complete | [Guide](./02_PyTorch_Algorithms/intro.md) |
| Part 3 | [`3.1 Foundations`](./03_Triton_Kernels/intro.md) / [`3.2 Transition`](./03_Triton_Kernels/intro.md) / [`3.3 Advanced A: Attention Optimization`](./03_Triton_Kernels/intro.md) / [`3.4 Advanced B: Inference Optimization`](./03_Triton_Kernels/intro.md) / [`3.5 Projects`](./03_Triton_Kernels/intro.md) | Triton kernel development. | ✅ Complete | [Guide](./03_Triton_Kernels/intro.md) |
| Part 4 | [`4.1 CUDA 编程基础`](./04_CUDA_and_System_Optimization/intro.md) / [`4.2 系统级性能优化`](./04_CUDA_and_System_Optimization/intro.md) / [`4.3 分布式训练工程`](./04_CUDA_and_System_Optimization/intro.md) / [`4.4 架构视野`](./04_CUDA_and_System_Optimization/intro.md) | CUDA C++ and system optimization. | 🚧 Planned | [Guide](./04_CUDA_and_System_Optimization/intro.md) |
