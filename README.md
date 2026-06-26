<h1 align="center">大模型算法实战教程</h1>

本项目原名 `LLM-Algo-LeetCode`，现统一更名为“大模型算法实战教程”。

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体代码与算子已基本构建完成，正在持续优化文档细节与补充注释。欢迎大家提交 Issue 反馈问题或贡献 PR！

> 大模型算法实战教程
>
> A practical tutorial with theory, walkthroughs, test cases, and solutions.

[中文版 (Chinese)](#中文版) | [English Version](#english-version)

---

# 中文版

## 🎯 项目简介

这是一个面向大模型入门到进阶的算法实战教程，当前以大语言模型（LLM）为主线，覆盖 Python、PyTorch、Transformer、推理优化、显存管理与 CUDA/Triton 实战。我们把每个知识点整理为可运行、可验证、可回顾的 Jupyter Notebook 练习，帮助你从“会看”走到“会写、会调、会优化”。

本项目配有本地测试用例，支持反复练习和回顾；当前版本以 LLM 内容为主，不展开 Diffusion 或多模态内容。

### ✨ 项目特点

1. **高度垂直**：专注 Transformer、MoE、量化、推理加速与显存优化。
2. **工程导向**：使用 PyTorch、Triton 或 CUDA C++ 实现核心算子和系统逻辑。
3. **测试驱动**：每道题都配套本地测试和性能验证。

### 👥 项目受众

- **求职面试者**：巩固 LLM 算法工程师、AI 架构师、算子开发工程师的高频考点。
- **AI 研发人员**：从代码底层理解分布式通信、显存优化与 Triton/CUDA 算子。
- **前置要求**：具备 Python 和深度学习基础，熟悉 PyTorch；高阶内容需要一定 C++/CUDA 基础。

## 🆕 更新时间线

- **2026-06-26**：[最新更新点]优化了中文版首页的导览展示，并重新梳理了第三、第四部分的学习路径，让内容入口更清晰、学习顺序更直观。
- **2026-06-15**：推进第零部分 / 第一部分的分组与导读收口，统一部分级导航，并完成网页底部评论区接入 GitHub Discussions，同时持续扩展第一部分的正文、桥接页与 Notebook 结构。
- **2026-06-13**：修复 dead link，并为未完成页面补充占位页，避免学习入口出现 404。
- **2026-04-21**：更新 Colab 徽章链接，统一指向官方 `datawhalechina` 仓库。
- **2026-04-20**：上线站点首页与部分导学；新增第零部分前置知识与第一部分练习内容，完善在线阅读入口与学习路径。
- **2026-04-18 ~ 2026-04-19**：集中重构第二部分 / 第三部分内容，优化 Notebook、答案区与算子实现说明。
- **2026-04-02**：完成教程核心 Notebook、文档与测试脚本的初始搭建。

> 路径兼容说明：第三部分已从 `03_CUDA_and_Triton_Kernels` 更名为 `03_Triton_Kernels`，CUDA / 系统优化内容拆分到第四部分。旧网页路径会保留迁移入口，建议新链接统一使用 `03_Triton_Kernels`。

## 📚 部分总览

先按“前置 -> 理论 -> 算法 -> Triton -> CUDA”的顺序理解整套教程。下面这张表会直接告诉你：每一部分学什么、包含哪些组、适合谁、当前进度如何。

| 部分 | 这部分学什么 | 组别 | 适合谁 | 状态 |
| ---- | ---- | ---- | ---- | ---- |
| 第零部分：前置知识与环境准备 | 把 Python、PyTorch、Jupyter、调试工具和基础运行环境先搭好。 | [`0A Python 基础`](./00_Prerequisites/0A.md) / [`0B PyTorch 基础`](./00_Prerequisites/0B.md) / [`0C 深度学习基础`](./00_Prerequisites/0C.md) / [`0D 工具与调试`](./00_Prerequisites/0D.md) | 第一次进入教程、需要补齐入门前置的人 | ✅ 主线已完成，持续扩展 |
| 第一部分：硬件、算力推导与系统级理论 | 理解硬件、算力、访存、通信和调度这些底层约束。 | [`1A 数值基础与算力估算`](./01_Hardware_Math_and_Systems/1A.md) / [`1B 单卡硬件与访存优化`](./01_Hardware_Math_and_Systems/1B.md) / [`1C 多卡通信与显存共享`](./01_Hardware_Math_and_Systems/1C.md) / [`1D 异构调度与算子编程`](./01_Hardware_Math_and_Systems/1D.md) / [`1E 编译优化与算力生态`](./01_Hardware_Math_and_Systems/1E.md) | 想先弄清“为什么要这样写”和“为什么要这样部署”的学习者 | ✅ 理论主线已完成，持续扩展 |
| 第二部分：PyTorch 算法实战 | 在 PyTorch 层把算法、模型和推理优化先跑通。 | [`2.1 基础算子`](./docs/02_PyTorch_Algorithms/intro.md) / [`2.2 模型架构`](./docs/02_PyTorch_Algorithms/intro.md) / [`2.3 微调与训练技术`](./docs/02_PyTorch_Algorithms/intro.md) / [`2.4 对齐技术`](./docs/02_PyTorch_Algorithms/intro.md) / [`2.5 反向传播与显存优化`](./docs/02_PyTorch_Algorithms/intro.md) / [`2.6 核心推理优化`](./docs/02_PyTorch_Algorithms/intro.md) / [`2.7 高级推理优化`](./docs/02_PyTorch_Algorithms/intro.md) / [`2.8 分布式与扩展`](./docs/02_PyTorch_Algorithms/intro.md) | 希望先用熟悉工具建立实现感的人 | ✅ 已完成，持续扩展 |
| 第三部分：Triton 算子开发 | 把前面学到的算子和优化思路落到 GPU kernel。 | [`3.1 基础篇`](./docs/03_Triton_Kernels/intro.md) / [`3.2 过渡篇`](./docs/03_Triton_Kernels/intro.md) / [`3.3 进阶A：Attention优化`](./docs/03_Triton_Kernels/intro.md) / [`3.4 进阶B：推理优化`](./docs/03_Triton_Kernels/intro.md) / [`3.5 项目篇`](./docs/03_Triton_Kernels/intro.md) | 希望从 PyTorch 走向 Triton 的学习者 | ✅ 已完成，持续扩展 |
| 第四部分：CUDA C++ 与系统优化 | 进一步下探到 CUDA、系统调优和工程化架构。 | [`4.1 CUDA 编程基础`](./docs/04_CUDA_and_System_Optimization/intro.md) / [`4.2 系统级性能优化`](./docs/04_CUDA_and_System_Optimization/intro.md) / [`4.3 分布式训练工程`](./docs/04_CUDA_and_System_Optimization/intro.md) / [`4.4 架构视野`](./docs/04_CUDA_and_System_Optimization/intro.md) | 准备做底层性能优化和工程落地的人 | ✅ 主线已建立，持续扩展 |

## 🚀 快速开始

如果你想开始学习，先从在线站点阅读导学和目录；需要运行 Notebook 时，Part 0 / 1 / 2 可以优先走 CPU-first，Part 3 / 4 需要 GPU 环境。环境与平台差异见 [使用指南](./docs/guide.md)。

### 学习路径

1. 在左侧侧边栏选择感兴趣的部分
2. 点击 **📖 完整导学** 了解学习路径
3. 选择具体题目开始学习
4. 环境和平台差异见 [使用指南](./docs/guide.md)

### 方式 1：在线阅读

访问在线站点：

[https://datawhalechina.github.io/llm-algo-leetcode/](https://datawhalechina.github.io/llm-algo-leetcode/)

适合：
- 先看目录
- 先读部分导学
- Part 0 / 1 / 2 可以直接用 Colab CPU 跑练习
- Part 3 / 4 需要 Colab GPU runtime

### 方式 2：本地学习

```bash
git clone https://github.com/datawhalechina/llm-algo-leetcode.git
cd llm-algo-leetcode
conda env create -f environment.yml
conda activate llm_algo
jupyter lab
```

适合：
- 想在本地完整跑 Part 0 / 1 / 2 的 Notebook
- 想自己控制 Python / PyTorch / CUDA 版本
- 想做更稳定的离线调试
- Part 3 / 4 需要本地 NVIDIA GPU

### 方式 3：CNB 统一环境

如果你希望和仓库当前推荐环境保持一致，可以使用 CNB 统一环境入口。

适合：
- 团队协作
- 统一实验镜像
- 需要减少本地环境差异
- Part 0 / 1 / 2 可以用 CNB CPU
- Part 3 / 4 需要 CNB GPU 会话

CNB 的具体使用方式和适用范围见 [使用指南](./docs/guide.md)。

## 📖 更多资源

- [使用指南](./docs/guide.md) - 环境与学习方式
- [贡献指南](./docs/contributing.md) - 如何参与项目开发和测试
- [维护与发布手册](./docs/maintenance.md) - 部分、链接、测试与发布的维护约定
- [自动化测试脚本索引](./project_test_scripts.md) - 各类验证脚本入口

## 👨‍💻 贡献者名单

| 姓名 | 职责 | 简介 |
| :----| :---- | :---- |
| lynn_jingjing | 项目发起人 | 一个算法工程师 |

*(欢迎在此留下您的名字！)*

## 📄 开源协议

本项目采用 [CC BY-NC-SA 4.0](./LICENSE) 协议进行许可。

---

# English Version

This is a practical LLM algorithm tutorial for learners from beginner to advanced. It focuses on Python, PyTorch, Transformers, inference optimization, VRAM management, and CUDA/Triton practice. Each concept is turned into a runnable, verifiable, and reviewable Jupyter Notebook exercise so you can move from "reading" to "writing, debugging, and optimizing".

This project currently focuses on the Large Language Model (LLM) domain and does not expand into Diffusion or multimodal content. It also provides local test cases and a repeatable learning path for practice and review.

License: [CC BY-NC-SA 4.0](./LICENSE).

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

- **2026-06-26**: [Latest update] improved the Chinese homepage overview and clarified the learning path across Parts 3 and 4, making the entry points and study order more intuitive.
- **2026-06-15**: Finalized the Part 0 / 1 grouping and guide cleanup, unified the part-level navigation, connected the page comments to GitHub Discussions, and continued expanding Part 1 content, bridge pages, and notebook structure.
- **2026-06-13**: Fixed dead links and added placeholder pages for unfinished content to prevent 404s in learning entry points.
- **2026-04-21**: Updated Colab badges to point to the official `datawhalechina` repository.
- **2026-04-20**: Launched the site homepage and part guides; added Part 0 prerequisites and Part 1 practice content to unify the learning path.
- **2026-04-18 ~ 2026-04-19**: Refactored Part 2 / 3 content, polishing notebooks, answer sections, and operator implementation notes.
- **2026-04-02**: Completed the initial tutorial notebooks, docs, and test scripts.

> Path compatibility note: Part 3 has been renamed from `03_CUDA_and_Triton_Kernels` to `03_Triton_Kernels`, and CUDA / system optimization content has moved to Part 4. Old web paths keep migration pages, but new links should use `03_Triton_Kernels`.

## Part Overview

| Part | Groups | Summary | Status | Entry |
| ---- | ---- | ------- | ---- | ---- |
| Part 0 | [`0A Python Basics`](./00_Prerequisites/0A.md) / [`0B PyTorch Basics`](./00_Prerequisites/0B.md) / [`0C Deep Learning Basics`](./00_Prerequisites/0C.md) / [`0D Tools and Debugging`](./00_Prerequisites/0D.md) | Prerequisites and environment setup. | ✅ Main line complete, continuously expanding | [Guide](./00_Prerequisites/intro.md) |
| Part 1 | [`1A Numerics and Compute Estimation`](./01_Hardware_Math_and_Systems/1A.md) / [`1B Single-GPU Memory and Access`](./01_Hardware_Math_and_Systems/1B.md) / [`1C Multi-GPU Communication and VRAM`](./01_Hardware_Math_and_Systems/1C.md) / [`1D Heterogeneous Scheduling and Operators`](./01_Hardware_Math_and_Systems/1D.md) / [`1E Compiler Optimization and Compute Ecosystem`](./01_Hardware_Math_and_Systems/1E.md) | Hardware, compute estimation, and system-level theory. | ✅ Theoretical main line complete, continuously expanding | [Guide](./01_Hardware_Math_and_Systems/intro.md) |
| Part 2 | [`2.1 Basic Operators`](./docs/02_PyTorch_Algorithms/intro.md) / [`2.2 Model Architecture`](./docs/02_PyTorch_Algorithms/intro.md) / [`2.3 Fine-Tuning and Training`](./docs/02_PyTorch_Algorithms/intro.md) / [`2.4 Alignment Methods`](./docs/02_PyTorch_Algorithms/intro.md) / [`2.5 Backpropagation and VRAM Optimization`](./docs/02_PyTorch_Algorithms/intro.md) / [`2.6 Core Inference Optimization`](./docs/02_PyTorch_Algorithms/intro.md) / [`2.7 Advanced Inference Optimization`](./docs/02_PyTorch_Algorithms/intro.md) / [`2.8 Distributed and Scaling`](./docs/02_PyTorch_Algorithms/intro.md) | PyTorch algorithm practice. | ✅ Complete, continuously expanding | [Guide](./docs/02_PyTorch_Algorithms/intro.md) |
| Part 3 | [`3.1 Foundations`](./docs/03_Triton_Kernels/intro.md) / [`3.2 Transition`](./docs/03_Triton_Kernels/intro.md) / [`3.3 Advanced A: Attention Optimization`](./docs/03_Triton_Kernels/intro.md) / [`3.4 Advanced B: Inference Optimization`](./docs/03_Triton_Kernels/intro.md) / [`3.5 Projects`](./docs/03_Triton_Kernels/intro.md) | Triton kernel development. | ✅ Complete, continuously expanding | [Guide](./docs/03_Triton_Kernels/intro.md) |
| Part 4 | [`4.1 CUDA Programming Basics`](./docs/04_CUDA_and_System_Optimization/intro.md) / [`4.2 System-Level Performance Optimization`](./docs/04_CUDA_and_System_Optimization/intro.md) / [`4.3 Distributed Training Engineering`](./docs/04_CUDA_and_System_Optimization/intro.md) / [`4.4 Architecture Perspective`](./docs/04_CUDA_and_System_Optimization/intro.md) | CUDA C++ and system optimization. | ✅ Main line established, continuously expanding | [Guide](./docs/04_CUDA_and_System_Optimization/intro.md) |

## Quick Start

### Option 1: Read Online

Visit the online platform:

[https://datawhalechina.github.io/llm-algo-leetcode/](https://datawhalechina.github.io/llm-algo-leetcode/)

Suitable for:
- Reading the table of contents first
- Reading the part guides first
- Part 0 / 1 / 2 can run on Colab CPU
- Part 3 / 4 need a Colab GPU runtime

### Option 2: Local Development

```bash
git clone https://github.com/datawhalechina/llm-algo-leetcode.git
cd llm-algo-leetcode
conda env create -f environment.yml
conda activate llm_algo
jupyter lab
```

Suitable for:
- Running Part 0 / 1 / 2 locally on CPU
- Controlling your own Python / PyTorch / CUDA versions
- More stable offline debugging
- Part 3 / 4 require a local NVIDIA GPU

For environment details and platform differences, see the Chinese guide section or [docs/guide.md](./docs/guide.md).

### Option 3: CNB Unified Delivery

If you want the same runtime style used by the repository, use the CNB unified environment.

Suitable for:
- Team collaboration
- Consistent experiment images
- Lower local environment drift
- Part 0 / 1 / 2 can use CNB CPU
- Part 3 / 4 need a CNB GPU session

See [docs/guide.md](./docs/guide.md) for the exact environment rules and scope.
