<h1 align="center"> LLM-Algo-LeetCode（🧪 Beta公测版） </h1>

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体代码与算子已基本构建完成，正在持续优化文档细节与补充注释。欢迎大家提交 Issue 反馈问题或贡献 PR！

[中文版 (Chinese)](#中文版) | [English Version](#english-version)

---

# 中文版

## 🎯 项目简介

这是一个面向大模型入门到进阶的算法刷题教程，服务于 LLM 算法工程师、AI Infra 工程师和研究实习生等学习者。它严格聚焦纯 LLM 领域，不包含 Diffusion 或通用多模态，把现代大模型的底层算法与系统设计整理成可运行、可验证、可回顾的 Jupyter Notebook 练习，形成类似 LeetCode 的学习体验。

**在线阅读入口：** [https://datawhalechina.github.io/llm-algo-leetcode/](https://datawhalechina.github.io/llm-algo-leetcode/)

**项目特点：**
- **高度垂直**：专注 Transformer、MoE、量化、推理加速与显存优化。
- **工程导向**：要求使用 PyTorch、Triton 或原生 CUDA C++ 实现核心算子和系统逻辑。
- **测试驱动**：每道题都带可执行验证，便于回归和对齐工业实现。

## 📚 章节总览

| 模块 | 简介 | 状态 | 入口 |
| ---- | ---- | ---- | ---- |
| Chapter 0 | 前置知识与环境准备。 | 🚧 部分完成 | [导学](./00_Prerequisites/intro.md) |
| Chapter 1 | 硬件、算力推导与系统级理论。 | 🚧 理论完成 | [导学](./01_Hardware_Math_and_Systems/intro.md) |
| Chapter 2 | PyTorch 算法实战。 | ✅ 完成 | [导学](./02_PyTorch_Algorithms/intro.md) |
| Chapter 3 | CUDA C++ 与 Triton 算子开发。 | ✅ 完成 | [导学](./03_CUDA_and_Triton_Kernels/intro.md) |

## 🚀 快速开始

如果你想开始学习，直接从在线站点或左侧章节目录进入即可。环境与平台差异见 [使用指南](./docs/guide.md)。

### 方式 1：在线阅读

访问在线站点：

[https://datawhalechina.github.io/llm-algo-leetcode/](https://datawhalechina.github.io/llm-algo-leetcode/)

适合：
- 先看目录
- 先读章节导学
- 先用 Colab 跑练习

### 方式 2：本地学习

```bash
git clone https://github.com/datawhalechina/llm-algo-leetcode.git
cd llm-algo-leetcode
conda env create -f environment.yml
conda activate llm_algo
jupyter lab
```

## 📖 更多资源

- [使用指南](./docs/guide.md) - 环境与学习方式
- [贡献指南](./docs/contributing.md) - 如何参与项目开发和测试
- [维护与发布手册](./docs/maintenance.md) - 章节、链接、测试与发布的维护约定
- [自动化测试脚本索引](./project_test_scripts.md) - 各类验证脚本入口

## 👨‍💻 贡献者名单

| 姓名 | 职责 | 简介 |
| :----| :---- | :---- |
| lynn_jingjing | 项目发起人 | 一个算法工程师 |

*(欢迎在此留下您的名字！)*

## 📄 开源协议

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。

---

# English Version

This project aims to provide a **systematic, interactive, and test-driven** engineering practice guide for candidates preparing for roles such as Large Language Model (LLM) Algorithm Engineers, AI Infrastructure (AI Infra) Engineers, and Research Interns.

Unlike traditional "text-only interview cheat sheets," this project strictly focuses on the **pure Large Language Model (LLM) domain** (excluding Diffusion or general Multimodal generation). It adopts a **"Learn then Practice"** approach. We have extracted the core underlying algorithms and system designs of modern LLM architectures, encapsulated them into independent Jupyter Notebook fill-in-the-blank exercises, and equipped them with local test cases to provide a LeetCode-like practice experience.

## Target Audience

- **Job Seekers**: Covering high-frequency concepts for LLM Algorithm Engineers and Kernel Optimization Engineers.
- **AI Practitioners**: Developers seeking a bottom-up understanding of LLM mechanisms like Distributed Communication, VRAM Optimization, and Triton/CUDA.

## Features

1. **Highly Vertical**: Focuses exclusively on Transformers, MoE, Quantization, Inference Acceleration, and VRAM Optimization.
2. **Engineering-Oriented**: Requires implementing core operators and system logic using PyTorch, Triton, or native CUDA C++.
3. **Test-Driven**: Every exercise includes built-in test validations aligned with industrial open-source implementations (e.g., HuggingFace, vLLM).

## Quick Start

### Option 1: Read Online

Visit the online platform:

[https://datawhalechina.github.io/llm-algo-leetcode/](https://datawhalechina.github.io/llm-algo-leetcode/)

### Option 2: Local Development

```bash
git clone https://github.com/datawhalechina/llm-algo-leetcode.git
cd llm-algo-leetcode
conda env create -f environment.yml
conda activate llm_algo
jupyter lab
```

For environment details and platform differences, see the Chinese guide section or [docs/guide.md](./docs/guide.md).
