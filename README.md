<h1 align="center"> LLM-Algo-LeetCode（🧪 Beta公测版） </h1>

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体代码与算子已基本构建完成，正在持续优化文档细节与补充注释。欢迎大家提交 Issue 反馈问题或贡献 PR！

[中文版 (Chinese)](#中文版) | [English Version](#english-version)

---

# 中文版

本项目旨在为大语言模型（LLM）算法工程师、AI 基础设施（AI Infra）工程师以及研究实习生等岗位的候选人，提供一个**系统性、可交互、带评测**的工程实战指南。

与传统的“文字版八股文面经”不同，本项目严格限定于**纯粹的大语言模型 (LLM) 领域**（不包含 Diffusion 或多模态）。我们提取了现代大模型架构中最核心的底层算法与系统设计，将其封装为独立的 Jupyter Notebook 填空题，并配备本地测试用例，力求提供类似 LeetCode 的刷题体验。

## 👥 项目受众

- **求职面试者**：巩固 LLM 算法工程师、AI 架构师、算子开发工程师的高频底层考点。
- **AI 研发人员**：希望从代码底层理解大模型运作机制（如分布式通信、显存优化、Triton/CUDA 算子）的从业者。
- **前置要求**：具备 Python 和深度学习基础，熟悉 PyTorch。高级章节需一定 C++/CUDA 基础。

##  项目特点

1. **高度垂直**：专注 Transformer、MoE、量化、推理加速与显存优化。
2. **工程导向**：要求使用 PyTorch、Triton 或原生的 CUDA C++ 实现核心算子和系统逻辑。
3. **测试驱动 (Test-Driven)**：每一道题都内置了与工业界开源实现（如 HuggingFace, vLLM）对齐的测试验证，确保输出张量维度与数值的完全正确。并包含详尽的 Benchmark 性能基准测试。
4. **由浅入深**：分为“理论基础”、“模型组装与训练”、“底层计算加速”三个阶段，覆盖从入门到高阶架构的完整知识链路。

## 在线阅读
> 文档站点正在建设中，近期将支持 VitePress 在线浏览。

## 目录结构与学习路线

| 模块 | 简介 | 状态 |
| ---- | ---- | ---- |
| [**Chapter 1: 硬件、算力推导与系统级理论**](./01_Hardware_Math_and_Systems/) | 包含系统架构与性能优化的高频问答题，适合作为面试前的快速复习材料。涵盖 GPU 架构、显存估算、通信拓扑与国产芯片概览。 | ✅ |
| [**Chapter 2: PyTorch 核心算法实现实战**](./02_PyTorch_Algorithms/) | 核心代码实战区。要求在 Jupyter 中完成 TODO 填空，涵盖 MHA/GQA、RoPE、MoE、LoRA、ZeRO 优化器模拟及显存优化策略。 | ✅ |
| [**Chapter 3: CUDA C++ 与 Triton 算子开发**](./03_CUDA_and_Triton_Kernels/) | 针对算子加速与高阶架构。包含 Triton Fused 算子、FlashAttention、PagedAttention、以及原生 CUDA 共享内存优化。需 GPU 环境。 | ✅ |

*(详细文件列表请进入各个子目录查看 `README` 或 `.md` / `.ipynb` 文件)*

---

##  硬件要求与学习路线建议

| 你的环境 | 可学内容 | 覆盖面试考点 |
| ---- | ---- | ---- |
| **无 GPU (CPU only)** | Chapter 1 全部 + Chapter 2 大部分（00~18） | 约 70%，覆盖算法工程师最高频考点 |
| **有 NVIDIA GPU** | 全部内容 | 100%，额外覆盖算子开发与 Infra 岗 |

> 没有 GPU 的同学不用担心！Chapter 1（理论）和 Chapter 2（PyTorch 实现）中的绝大部分代码都可以在纯 CPU 环境下运行和测试。只有 Chapter 3（Triton / CUDA 算子）必须在 Linux + NVIDIA GPU 环境下编译运行。

##  快速开始

```bash
# 0. 系统前置依赖 (Ubuntu/Debian，仅需执行一次)
# Triton 和 CUDA 算子需要 C 编译器来进行 JIT 编译
sudo apt-get install build-essential

# 1. 克隆仓库
git clone https://github.com/your-username/llm-algo-leetcode.git
cd llm-algo-leetcode

# 2. 创建并激活虚拟环境 (建议 Python 3.10+)
conda create -n llm_algo python=3.10 -y
conda activate llm_algo

# 3. 安装依赖
pip install -r requirements.txt

# 4. 运行自动化测试检查环境 (可选)
# 脚本会遍历 02 和 03 目录下的所有 Notebook，执行其中的代码块以验证正确性。
python run_all_tests.py

# 5. 启动 Jupyter Lab 开始刷题
jupyter lab
```

## 贡献者名单

| 姓名 | 职责 | 简介 |
| :----| :---- | :---- |
| lynn_jingjing | 项目发起人 | 一个算法工程师 |
*(欢迎在此留下您的名字！)*

## 参与贡献

- 如果你发现代码实现有误、测试用例设计不够严谨，可以提 Issue 进行反馈。
- 如果你想补充更多经典大模型面试题或优化现有算子实现，非常欢迎提交 Pull Request。
- **提交代码前必读**：本项目包含自动化测试脚本 `run_all_tests.py`。提交 PR 前，请确保在本地运行此脚本以保证没有引入 Regression 退化。

## 关注我们

*(如果您有相关的公众号或交流群二维码，可以放置在这里)*

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。

---

# English Version

This project aims to provide a **systematic, interactive, and test-driven** engineering practice guide for candidates preparing for roles such as Large Language Model (LLM) Algorithm Engineers, AI Infrastructure (AI Infra) Engineers, and Research Interns.

Unlike traditional "text-only interview cheat sheets," this project strictly focuses on the **pure Large Language Model (LLM) domain** (excluding Diffusion or general Multimodal generation). It adopts a **"Learn then Practice"** approach. We have extracted the core underlying algorithms and system designs of modern LLM architectures, encapsulated them into independent Jupyter Notebook fill-in-the-blank exercises, and equipped them with local test cases (`pytest` or `assert` level) to provide a LeetCode-like practice experience.

## Target Audience
- **Job Seekers**: Covering high-frequency concepts for LLM Algorithm Engineers and Kernel Optimization Engineers.
- **AI Practitioners**: Developers seeking a bottom-up understanding of LLM mechanisms like Distributed Communication, VRAM Optimization, and Triton/CUDA.

##  Features
1. **Highly Vertical**: Focuses exclusively on Transformers, MoE, Quantization, Inference Acceleration, and VRAM Optimization.
2. **Engineering-Oriented**: Requires implementing core operators and system logic using PyTorch, Triton, or native CUDA C++.
3. **Test-Driven**: Every exercise includes built-in test validations aligned with industrial open-source implementations (e.g., HuggingFace, vLLM).

*(For detailed directory structures, please refer to the Chinese section or individual chapter folders)*