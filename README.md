<h1 align="center">llm-algo-leetcode | 大模型算法与系统教程</h1>
<p align="center">
  Runnable notebooks for LLM algorithms and systems.<br>
  面向大模型算法与系统的可运行 Notebook 教程。
</p>

<p align="center">
  <strong>主学习路线 / Core Routes</strong><br>
  <a href="./topic_discussion/inference_optimization/intro.md">推理优化 / Inference Optimization</a> ·
  <a href="./topic_discussion/memory_performance_tuning/intro.md">显存优化 / Memory Optimization</a> ·
  <a href="./topic_discussion/operator_optimization/intro.md">算子优化 / Operator Optimization（建设中 / In progress）</a> ·
  <a href="./topic_discussion/post_training_alignment/intro.md">后训练优化 / Post-Training Optimization（建设中 / In progress）</a>
</p>

<p align="center">
  <strong>专题入口 / Topic Paths</strong><br>
  <a href="./topic_discussion/quantization/intro.md">量化与压缩 / Quantization and Compression</a> ·
  <a href="./topic_discussion/profiling/intro.md">性能分析 / Performance Analysis</a> ·
  <a href="./topic_discussion/communication_parallel/intro.md">通信与并行 / Communication and Parallelism</a> ·
  <a href="./topic_discussion/intro.md">查看全部专题 / All topics</a>
</p>


[中文版 (Chinese)](#中文版) | [English Version](#english-version)

---

# 中文版

## 🎯 项目简介

这是一个以 Notebook 为载体的大模型算法与系统教程。教程从 Part 02 的 PyTorch 算法实践出发，按需回补 Part 00 / Part 01 的基础，并进一步延伸到 Part 03 的 Triton 算子开发和 Part 04 的 CUDA 与系统优化。

内容围绕可运行实现、实验验证和专题化学习路线组织，帮助学习者逐步理解模型如何计算、如何训练、如何推理，以及性能问题如何被测量和优化。

### ✨ 项目特点

1. **Notebook-first**：每节围绕可运行代码、题目区、答案区和基础验证展开，适合边学边改。
2. **按目标进入**：既支持从 Part 02 开始建立算法实践，也支持按训练、推理、显存、量化和并行专题跳读。
3. **算法到系统贯通**：沿着 `PyTorch → Triton → CUDA` 逐步下探，连接模型实现、算子优化、显存管理、通信与系统性能。
4. **用证据做项目判断**：通过 benchmark、profiling、真实 GPU 和 inference backend 实验，对吞吐、延迟、显存、质量和成本进行比较。

### 👥 适合对象

- **大模型算法学习者**：希望通过 PyTorch Notebook 理解 Transformer、训练、微调、推理和模型压缩。
- **性能与系统学习者**：希望进一步学习显存、Profiling、通信、Triton、CUDA 和 GPU 优化。
- **项目实践者**：希望通过 benchmark、真实 GPU 和 inference backend 实验，建立可复现、可比较的工程判断。

## 🌐 教程总览

教程提供两种阅读方式：按 `Part 00 -> Part 04` 逐层推进，或按训练、推理、显存、算子、后训练等主路线进入，再通过量化与压缩、性能分析、通信与并行等专题补充能力。推荐先从 [`Part 02`](./02_PyTorch_Algorithms/intro.md) 建立算法实践感，再按需要回补 `Part 00` / `Part 01`，最后进入 `Part 03` / `Part 04`；路线四目前处于建设阶段，`Part 05` 作为扩展预留。

[topic_discussion](./topic_discussion/intro.md) 用于组织跨 Part 的专题路线，[team_study](./team_study/intro.md) 用于沉淀共学记录。页面顶部提供常用专题入口，下面的资产总览和专题总览分别说明主线结构与专题覆盖范围。

![教程结构示意图](./docs/image-1.png)


<details>
<summary>📚 查看完整资产总览</summary>

这套教程不要求从 `00` 开始按顺序硬读。`00` 主要是前置补齐区，如果你已有基础，可以直接从最相关的部分开始；下面这张表会直接告诉你：每一部分学什么、包含哪些组、适合谁、当前进度如何。

| 部分 | 组别 | 内容定位 | 适合对象 | 状态 |
| ---- | ---- | ---- | ---- | ---- |
| [`第零部分：前置知识与环境准备（5 组 / 20 节，已完成，持续优化）`](./00_Prerequisites/intro.md) | [`0A Python 基础与数据表示（4 节）`](./00_Prerequisites/0A.md) / [`0B PyTorch 张量与自动求导（4 节）`](./00_Prerequisites/0B.md) / [`0C PyTorch 模型构建（4 节）`](./00_Prerequisites/0C.md) / [`0D 训练与模型直觉（4 节）`](./00_Prerequisites/0D.md) / [`0E 调试与性能（4 节）`](./00_Prerequisites/0E.md) | 把 Python、NumPy、PyTorch、训练循环、调试工具和性能意识搭好。 | 第一次进入教程、需要补齐入门前置的人。 | ✅ 已完成，持续优化 |
| [`第一部分：硬件、数学与系统（5 组 / 33 节，已完成，持续优化）`](./01_Hardware_Math_and_Systems/intro.md) | [`1A 数值基础与算力估算（4 节）`](./01_Hardware_Math_and_Systems/1A.md) / [`1B 单卡硬件与访存优化（9 节）`](./01_Hardware_Math_and_Systems/1B.md) / [`1C 多卡通信与显存共享（6 节）`](./01_Hardware_Math_and_Systems/1C.md) / [`1D 异构调度与算子编程（10 节）`](./01_Hardware_Math_and_Systems/1D.md) / [`1E 编译优化与硬件生态（5 节）`](./01_Hardware_Math_and_Systems/1E.md) | 理解硬件、算力、访存、通信和调度这些底层约束。 | 想先弄清“为什么要这样写”和“为什么要这样部署”的学习者。 | ✅ 已完成，持续优化 |
| [`第二部分：PyTorch 算法实战（10 组，已完成，持续优化）`](./02_PyTorch_Algorithms/intro.md) | [`2.1 基础算子`](./02_PyTorch_Algorithms/2_1.md) / [`2.2 模型架构`](./02_PyTorch_Algorithms/2_2.md) / [`2.3 训练与微调闭环`](./02_PyTorch_Algorithms/2_3.md) / [`2.4 偏好优化与对齐`](./02_PyTorch_Algorithms/2_4.md) / [`2.5 反向传播与显存优化`](./02_PyTorch_Algorithms/2_5.md) / [`2.6 核心推理优化`](./02_PyTorch_Algorithms/2_6.md) / [`2.7 高级推理策略`](./02_PyTorch_Algorithms/2_7.md) / [`2.8 模型压缩与量化`](./02_PyTorch_Algorithms/2_8.md) / [`2.9 分布式并行策略`](./02_PyTorch_Algorithms/2_9.md) / [`2.10 项目实战`](./02_PyTorch_Algorithms/2_10.md) | 在 PyTorch 层把算法、模型、推理、压缩、并行与项目验证先跑通。 | 希望先用熟悉工具建立实现感的人。 | ✅ 已完成，持续优化 |
| [`第三部分：Triton 算子开发（5 组 / 15 节，已完成，持续优化）`](./03_Triton_Kernels/intro.md) | [`3.1 基础篇（5 节）`](./03_Triton_Kernels/intro.md) / [`3.2 过渡篇（2 节）`](./03_Triton_Kernels/intro.md) / [`3.3 进阶A：Attention优化（3 节）`](./03_Triton_Kernels/intro.md) / [`3.4 进阶B：推理优化（2 节）`](./03_Triton_Kernels/intro.md) / [`3.5 项目篇（3 节）`](./03_Triton_Kernels/intro.md) | 把前面学到的算子和优化思路落到 GPU kernel。 | 希望从 PyTorch 走向 Triton 的学习者。 | ✅ 已完成，持续优化 |
| [`第四部分：CUDA C++ 与系统优化（4 组 / 16 节，建设中）`](./04_CUDA_and_System_Optimization/intro.md) | [`4.1 CUDA 编程基础（4 节）`](./04_CUDA_and_System_Optimization/intro.md) / [`4.2 系统级性能优化（4 节）`](./04_CUDA_and_System_Optimization/intro.md) / [`4.3 分布式训练工程（4 节）`](./04_CUDA_and_System_Optimization/intro.md) / [`4.4 架构视野（4 节）`](./04_CUDA_and_System_Optimization/intro.md) | 进一步下探到 CUDA、系统调优和工程化架构。 | 准备做底层性能优化和工程落地的人。 | 🛠 建设中 |
| [`第五部分：CUDA Rust（预留）`](./05_CUDA_Rust/intro.md) | 预留中 | 预留中 | 预留中 | 🚧 预留 |

</details>

<details>
<summary>🧭 查看完整专题总览</summary>

| 层级 | 入口 | 内容定位 | 适合对象 | 状态 |
| ---- | ---- | ---- | ---- | ---- |
| 主学习路线 | [`推理优化（Inference Optimization）`](./topic_discussion/inference_optimization/intro.md) | FlashAttention、解码、PagedAttention、cache 与 benchmark。 | 想系统理解推理加速路径的学习者。 | ✅ 持续优化 |
| 主学习路线 | [`显存优化（Memory Optimization）`](./topic_discussion/memory_performance_tuning/intro.md) | VRAM、activation、checkpointing、offload 和 trade-off。 | 想系统优化显存和端到端性能的学习者。 | ✅ 持续优化 |
| 主学习路线（建设中） | [`算子优化（Operator Optimization）`](./topic_discussion/operator_optimization/intro.md) | Triton、CUDA、访存、fusion、autotune 和 kernel 到端到端的验证；图变换与 lowering 仍见编译与图优化专题。 | 想从算子实现走向 kernel 和端到端性能优化的学习者。 | 🛠 建设中 |
| 主学习路线（建设中） | [`后训练优化（Post-Training Optimization）`](./topic_discussion/post_training_alignment/intro.md) | SFT 衔接、偏好数据、DPO、GRPO 与项目交付。 | 想从监督微调继续进入偏好优化与对齐的学习者。 | 🛠 建设中 |
| 横切支撑专题 | [`量化与压缩（Quantization and Compression）`](./topic_discussion/quantization/intro.md) | PTQ、QAT、GPTQ、AWQ、FP8 与部署决策。 | 想同时考虑精度、显存、吞吐和部署取舍的学习者。 | ✅ 持续优化 |
| 横切支撑专题 | [`通信与并行（Communication and Parallelism）`](./topic_discussion/communication_parallel/intro.md) | NCCL、AllReduce、ZeRO、PP、TP 和并行验证。 | 想理解多卡训练和通信边界的学习者。 | ✅ 持续优化 |
| 横切支撑专题 | [`性能分析（Performance Analysis）`](./topic_discussion/profiling/intro.md) | 性能取证、trace 阅读、回归验证和行动决策。 | 想系统补性能意识与排障方法的学习者。 | ✅ 持续优化 |
| 基础支撑专题 | [`监督微调与训练工程（Supervised Fine-Tuning and Training Engineering）`](./topic_discussion/fine_tuning_training/intro.md) | SFT、LoRA、训练控制、数据工程和项目准备。 | 想补齐监督微调与训练工程基础的学习者。 | ✅ 持续优化 |
| 基础支撑专题 | [`反向传播与训练机制（Backpropagation and Training Mechanics）`](./topic_discussion/backpropagation_training_mechanism/intro.md) | autograd、backward、checkpointing、offload 与训练节奏。 | 想补训练机制底座的学习者。 | ✅ 持续优化 |
| 基础支撑专题 | [`大模型架构（Model Architecture）`](./topic_discussion/model_architecture/intro.md) | 结构演进、代表模型和 MoE / 稀疏化。 | 想补模型结构背景与横向对照的学习者。 | ✅ 持续优化 |
| 基础支撑专题 | [`编译与图优化（Compiler and Graph Optimization）`](./topic_discussion/compiler_graph_optimization/intro.md) | 图变换、IR、lowering、执行计划和 backend 约束。 | 想理解计算图如何变成可执行程序的学习者。 | ✅ 持续优化 |

</details>

<details>
<summary>🤝 查看共学沉淀</summary>

| 模块 | 覆盖范围 | 内容定位 | 适合对象 | 状态 |
| ---- | ---- | ---- | ---- | ---- |
| [`组队学习专题`](./team_study/intro.md) | 不固定 | [`part2_l1_202606`](./team_study/part2_l1_202606/intro.md) / [`part2_l1_202607`](./team_study/part2_l1_202607/intro.md) / [`part2_l2_202607`](./team_study/part2_l2_202607/intro.md) | 想通过共学沉淀知识、题目与复盘记录的学习者。 | 🛠 建设中 |

</details>

<details>
<summary>🆕 查看更新时间线</summary>

- **2026-08-22**：完成 README 首页导航、项目简介与项目特点重写；收口横向专题的五层 Infra 结构与跨专题边界；完成 66、73、75、76 等真实 backend / GPU 项目的结果保存、环境说明和基础验证，并同步文档入口与链接检查。
- **2026-08-17**：统一 Part 01 导读与组页口径，收紧横向专题结构和学习路线表达，补充 Part 02 项目页与图解资产审计，并明确统一验证入口。
- **2026-07-10**：整理首页教程总览与状态列，校正 Part 00 / Part 01 的组名、节数和导航状态。
- **2026-06-26**：重构首页教程总览、状态列和学习路径，明确 Part 00-04、横向专题与共学记录的关系。
- **2026-06-15**：推进第零部分 / 第一部分的分组与导读收口，统一部分级导航，并完成网页底部评论区接入 GitHub Discussions，同时持续扩展第一部分的正文、桥接页与 Notebook 结构。
- **2026-06-13**：修复 dead link，并为未完成页面补充占位页，避免学习入口出现 404。
- **2026-04-21**：更新 Colab 徽章链接，统一指向官方 `datawhalechina` 仓库。
- **2026-04-20**：上线站点首页与部分导学；新增第零部分前置知识与第一部分练习内容，完善在线阅读入口与学习路径。
- **2026-04-18 ~ 2026-04-19**：集中重构第二部分 / 第三部分内容，优化 Notebook、答案区与算子实现说明。
- **2026-04-02**：完成教程核心 Notebook、文档与测试脚本的初始搭建。

> 路径兼容说明：第三部分已从 `03_CUDA_and_Triton_Kernels` 更名为 `03_Triton_Kernels`，CUDA / 系统优化内容拆分到第四部分。旧网页路径会保留迁移入口，建议新链接统一使用 `03_Triton_Kernels`。

</details>

## 🚀 快速开始

推荐从 [`Part 02`](./02_PyTorch_Algorithms/intro.md) 开始：先通过 PyTorch Notebook 建立算法实现感，再根据遇到的知识缺口回补 `Part 00` / `Part 01`，最后进入 `Part 03` / `Part 04` 的 GPU 底层优化。也可以按训练、推理或显存等目标直接进入对应专题。运行 Notebook 前，请先查看 [使用指南](./docs/guide.md) 和对应小节中的环境说明。

<details>
<summary>查看在线、本地与 CNB 的具体使用方式</summary>

### 方式 1：在线阅读

访问在线站点：

[https://datawhalechina.github.io/llm-algo-leetcode/](https://datawhalechina.github.io/llm-algo-leetcode/)

学习步骤：优先选择 `Part 02` 或对应专题，再从 **📖 完整导学** 进入对应 `intro.md`，最后进入目标 group；遇到知识缺口时回补 `Part 00` / `Part 01`。

适合：
- 先看目录再决定从哪一部分切入
- 先读部分导学，按目标跳转到对应 group
- Part 00 / 01 / 02 的大多数基础练习可以直接用 Colab CPU 跑
- 真实推理、训练和显存实验需要 Colab GPU 或本地 NVIDIA GPU
- Part 03 / 04 需要 Colab GPU runtime

### 方式 2：本地学习

```bash
git clone https://github.com/datawhalechina/llm-algo-leetcode.git
cd llm-algo-leetcode
conda env create -f environment.yml
conda activate llm_algo
python -m pip install -r requirements/torch-cpu.txt
jupyter lab
```

学习步骤：在仓库中优先进入 `Part 02` 或对应专题目录，先阅读 `intro.md`，再打开目标 Notebook；遇到知识缺口时回补 `Part 00` / `Part 01`。

适合：
- 想在本地完整跑 Part 00 / 01 / 02 的 Notebook
- 想自己控制 Python / PyTorch / CUDA 版本
- 想做更稳定的离线调试
- Part 03 / 04 需要本地 NVIDIA GPU

如果要运行真实训练、显存或推理项目，请改用 CUDA 环境：

```bash
conda env create -f environment-gpu.yml
conda activate llm_algo_gpu
python -m pip install -r requirements/torch-cu128.txt
jupyter lab
```

已有 CUDA PyTorch 的云端或 Colab 环境不需要执行上述安装；先运行项目环境预检，确认当前 Kernel 的 PyTorch、CUDA 和 GPU 可用。

### 方式 3：CNB 统一环境

如果你希望和仓库当前推荐环境保持一致，可以使用 CNB 统一环境入口。

适合：
- 团队协作
- 统一实验镜像
- 需要减少本地环境差异
- Part 00 / 01 / 02 可以用 CNB CPU
- Part 03 / 04 需要 CNB GPU 会话

CNB 的具体使用方式和适用范围见 [使用指南](./docs/guide.md)。

学习步骤：进入 CNB 会话后，优先从 `Part 02` 或对应专题的 `intro.md` 开始，再按导学进入目标 Notebook；基础不足时回补 `Part 00` / `Part 01`。

</details>

## 📖 更多资源

- [使用指南](./docs/guide.md) - 环境与学习方式
- [贡献指南](./docs/contributing.md) - 如何参与项目开发和测试
- [维护与发布手册](./docs/maintenance.md) - 部分、链接、测试与发布的维护约定
- [统一验证入口](./docs/maintenance.md#常用命令) - `verify.py part0_1`、`verify.py part2` 等标准验证命令
- [自动化测试脚本索引](./docs/maintenance.md#测试脚本索引) - 各类验证脚本入口

## 👨‍💻 贡献者名单

| 姓名 | 职责 | 简介 |
| :----| :---- | :---- |
| lynn_jingjing | 项目发起人 | 一个算法工程师 |


## 📄 许可声明

本仓库所有 `.ipynb` 文件中的文字内容（Markdown 单元格、公式、图示说明）采用 CC BY 4.0 协议；代码内容（Code 单元格、可执行实现）采用 Apache-2.0 协议。使用、转载、改编时，请按单元格类型分别遵守对应协议。文字协议见 [`LICENSE`](./LICENSE)，代码协议见 [`LICENSE-CODE`](./LICENSE-CODE)。

---

# English Version

## 📄 License Notice

All `.ipynb` files in this repository are mixed-content notebooks: Markdown cells (tutorial text, formulas, and figure captions) are licensed under CC BY 4.0, while Code cells (executable implementations) are licensed under Apache-2.0. Please comply with the corresponding license by cell type when using, redistributing, or adapting this repository. See [`LICENSE`](./LICENSE) for text and [`LICENSE-CODE`](./LICENSE-CODE) for code.

## 🎯 Project Introduction

This is a notebook-based tutorial on LLM algorithms and systems. It starts with PyTorch practice in Part 02, lets learners backfill the prerequisites in Part 00 / Part 01 as needed, and extends to Triton kernel development in Part 03 and CUDA and system optimization in Part 04.

The tutorial is organized around runnable implementations, experimental validation, and topic-based learning paths. It helps learners understand how models compute, train, and serve requests, and how to measure and optimize performance problems.

### ✨ Features

1. **Notebook-first**: Each lesson is organized around runnable code, exercises, answer cells, and basic validation.
2. **Goal-oriented entry**: Start from Part 02 for implementation practice, or follow topic paths for training, inference, memory, quantization, and parallelism.
3. **From algorithms to systems**: Follow the `PyTorch → Triton → CUDA` path across model implementation, kernel optimization, memory, communication, and system performance.
4. **Evidence-based projects**: Use benchmarks, profiling, real GPU runs, and inference backends to compare throughput, latency, memory, quality, and cost.

### 👥 Suitable For

- **LLM Algorithm Learners**: Use PyTorch notebooks to understand Transformers, training, fine-tuning, inference, and model compression.
- **Performance and Systems Learners**: Study memory, profiling, communication, Triton, CUDA, and GPU optimization.
- **Project Practitioners**: Build reproducible engineering judgment through benchmarks, real GPU runs, and inference backends.


## 🌐 Tutorial Overview

This tutorial offers two ways to read: follow the `Part 00 -> Part 04` progression, or enter through the main routes for training, inference, memory, and operator/compiler optimization, then use quantization, profiling, and parallelism as supporting topics. We recommend starting with [`Part 02`](./02_PyTorch_Algorithms/intro.md) to build implementation intuition, then backfilling `Part 00` / `Part 01` as needed before moving to `Part 03` / `Part 04`; Route 4 is currently under construction, and `Part 05` is reserved for future expansion.

[topic_discussion](./topic_discussion/intro.md) organizes cross-Part topic paths, while [team_study](./team_study/intro.md) stores collaborative-learning records. The top of the page provides common topic entries; the asset and topic overviews below describe the main structure and topic coverage.

![Tutorial structure overview](./docs/image-1.png)

<details>
<summary>📚 View the complete asset overview</summary>

You do not need to start from `00` in strict order. `00` is the prerequisite lane; if you already have the background, jump directly to the part that matches your goal. The table below summarizes each part, its groups, its audience, and its status.

| Part | Groups | Content Positioning | Suitable For | Status |
| ---- | ---- | ---- | ---- | ---- |
| [部分导读：前置知识与环境准备（5 groups / 20 lessons）](./00_Prerequisites/intro.md) | [组内导读：0A Python Basics and Data Representation (4 lessons)](./00_Prerequisites/0A.md) / [组内导读：0B PyTorch Tensors and Autograd (4 lessons)](./00_Prerequisites/0B.md) / [组内导读：0C PyTorch Model Construction (4 lessons)](./00_Prerequisites/0C.md) / [组内导读：0D Training and Model Intuition (4 lessons)](./00_Prerequisites/0D.md) / [组内导读：0E Debugging and Performance (4 lessons)](./00_Prerequisites/0E.md) | Prerequisites, engineering basics, and notebook-first practice. | First-time learners who need prerequisite support. | ✅ Complete, continuously refining |
| [部分导读：硬件、数学与系统（5 groups / 33 lessons）](./01_Hardware_Math_and_Systems/intro.md) | [组内导读：1A Numerics and Compute Estimation (4 lessons)](./01_Hardware_Math_and_Systems/1A.md) / [组内导读：1B Single-GPU Memory and Access (9 lessons)](./01_Hardware_Math_and_Systems/1B.md) / [组内导读：1C Multi-GPU Communication and VRAM (6 lessons)](./01_Hardware_Math_and_Systems/1C.md) / [组内导读：1D Heterogeneous Scheduling and Operators (10 lessons)](./01_Hardware_Math_and_Systems/1D.md) / [组内导读：1E Compiler Optimization and Hardware Ecosystem (5 lessons)](./01_Hardware_Math_and_Systems/1E.md) | Hardware, compute estimation, memory access, communication, and scheduling constraints. | Learners who want to understand why things are written and deployed this way. | ✅ Complete, continuously refining |
| [部分导读：PyTorch 算法实战（10 groups）](./02_PyTorch_Algorithms/intro.md) | [组内导读：2.1 Basic Operators](./02_PyTorch_Algorithms/2_1.md) / [组内导读：2.2 Model Architecture](./02_PyTorch_Algorithms/2_2.md) / [组内导读：2.3 Training and Fine-Tuning Loop](./02_PyTorch_Algorithms/2_3.md) / [组内导读：2.4 Preference Optimization and Alignment](./02_PyTorch_Algorithms/2_4.md) / [组内导读：2.5 Backpropagation and VRAM Optimization](./02_PyTorch_Algorithms/2_5.md) / [组内导读：2.6 Core Inference Optimization](./02_PyTorch_Algorithms/2_6.md) / [组内导读：2.7 Advanced Inference Strategies](./02_PyTorch_Algorithms/2_7.md) / [组内导读：2.8 Model Compression and Quantization](./02_PyTorch_Algorithms/2_8.md) / [组内导读：2.9 Distributed Parallel Strategy](./02_PyTorch_Algorithms/2_9.md) / [组内导读：2.10 Projects](./02_PyTorch_Algorithms/2_10.md) | PyTorch-level practice for algorithms, models, inference, compression, parallelism, and project validation. | Learners who want to build implementation intuition with familiar tools. | ✅ Complete, continuously refining |
| [部分导读：Triton Kernel Development (5 groups / 15 lessons)](./03_Triton_Kernels/intro.md) | [组内导读：3.1 Foundations (5 lessons)](./03_Triton_Kernels/intro.md) / [组内导读：3.2 Transition (2 lessons)](./03_Triton_Kernels/intro.md) / [组内导读：3.3 Advanced A: Attention Optimization (3 lessons)](./03_Triton_Kernels/intro.md) / [组内导读：3.4 Advanced B: Inference Optimization (2 lessons)](./03_Triton_Kernels/intro.md) / [组内导读：3.5 Projects (3 lessons)](./03_Triton_Kernels/intro.md) | Triton kernel development. | Learners who want to move from PyTorch to Triton. | ✅ Complete, continuously refining |
| [Part 04: CUDA C++ and System Optimization (4 groups / 16 lessons)](./04_CUDA_and_System_Optimization/intro.md) | [4.1 CUDA Programming Basics (4 lessons)](./04_CUDA_and_System_Optimization/intro.md) / [4.2 System-Level Performance Optimization (4 lessons)](./04_CUDA_and_System_Optimization/intro.md) / [4.3 Distributed Training Engineering (4 lessons)](./04_CUDA_and_System_Optimization/intro.md) / [4.4 Architecture Perspective (4 lessons)](./04_CUDA_and_System_Optimization/intro.md) | CUDA C++ and system optimization. | Learners preparing for low-level performance optimization and engineering deployment. | 🛠 In progress |
| [Part 05: CUDA Rust (reserved)](./05_CUDA_Rust/intro.md) | Reserved | Reserved | Reserved | 🚧 Reserved |

</details>

<details>
<summary>🧭 View the complete topic overview</summary>

| Layer | Entry | Content Positioning | Suitable For | Status |
| ---- | ---- | ---- | ---- | ---- |
| Main Study Path | [Inference Optimization Topic](./topic_discussion/inference_optimization/intro.md) | FlashAttention, decoding, PagedAttention, cache, and benchmark. | Learners who want practical inference acceleration. | ✅ Ongoing |
| Main Study Path | [Memory Optimization Topic](./topic_discussion/memory_performance_tuning/intro.md) | VRAM, activation, checkpointing, offload, and trade-offs. | Learners who want to optimize memory usage and end-to-end performance. | ✅ Ongoing |
| Main Study Path (In Progress) | [Operator Optimization Topic](./topic_discussion/operator_optimization/intro.md) | Triton, CUDA, memory access, fusion, autotuning, and kernel-to-end-to-end validation; graph rewrites and lowering remain in the compiler and graph optimization topic. | Learners who want to move from operator implementation to kernel and end-to-end optimization. | 🛠 In progress |
| Main Study Path (In Progress) | [Post-Training Optimization Topic](./topic_discussion/post_training_alignment/intro.md) | SFT transition, preference data, DPO, GRPO, and project delivery. | Learners who want to continue from supervised fine-tuning into alignment. | 🛠 In progress |
| Cross-Cutting Topic | [Quantization and Compression Topic](./topic_discussion/quantization/intro.md) | PTQ, QAT, GPTQ, AWQ, FP8, and deployment decisions. | Learners balancing accuracy, memory, throughput, and deployment cost. | ✅ Ongoing |
| Cross-Cutting Topic | [Communication and Parallelism Topic](./topic_discussion/communication_parallel/intro.md) | NCCL, AllReduce, ZeRO, PP, TP, and validation. | Learners who want to understand multi-GPU scaling and communication cost. | ✅ Ongoing |
| Cross-Cutting Topic | [Performance Analysis Topic](./topic_discussion/profiling/intro.md) | Evidence collection, trace reading, regression validation, and action decisions. | Learners who want systematic performance diagnosis and debugging methods. | ✅ Ongoing |
| Foundation Topic | [Supervised Fine-Tuning and Training Engineering Topic](./topic_discussion/fine_tuning_training/intro.md) | SFT, LoRA, training control, data engineering, and project preparation. | Learners who want stronger supervised fine-tuning and training-engineering foundations. | ✅ Ongoing |
| Foundation Topic | [Backpropagation and Training Mechanics Topic](./topic_discussion/backpropagation_training_mechanism/intro.md) | Autograd, backward, checkpointing, offload, and training rhythm. | Learners who want stronger training-mechanism foundations. | ✅ Ongoing |
| Foundation Topic | [Model Architecture Topic](./topic_discussion/model_architecture/intro.md) | Structure evolution, representative models, and MoE/sparsity. | Learners who want structural background and model comparison. | ✅ Ongoing |
| Foundation Topic | [Compiler and Graph Optimization Topic](./topic_discussion/compiler_graph_optimization/intro.md) | Graph rewrites, IR, lowering, execution plans, and backend constraints. | Learners who want to understand how computation graphs become executable programs. | ✅ Ongoing |

</details>

<details>
<summary>🤝 View collaborative study</summary>

| Module | Coverage | Content Positioning | Suitable For | Status |
| ---- | ---- | ---- | ---- | ---- |
| [Team Study Topic](./team_study/intro.md) | Not fixed | [part2_l1_202606](./team_study/part2_l1_202606/intro.md) / [part2_l1_202607](./team_study/part2_l1_202607/intro.md) / [part2_l2_202607](./team_study/part2_l2_202607/intro.md) | Learners who want to accumulate knowledge and review records through collaborative study. | 🛠 In progress |

</details>

<details>
<summary>🆕 View update timeline</summary>

- **2026-08-22**: Revised the README navigation, project introduction, and project features; finalized the five-layer Infra structure and cross-topic boundaries; completed result saving, environment notes, and baseline validation for real backend / GPU projects including 66, 73, 75, and 76, together with documentation and link checks.
- **2026-08-17**: Unified the Part 01 guides and group-page conventions, tightened the cross-topic structure and learning paths, reviewed Part 02 project pages and visual assets, and clarified the shared validation entry points.
- **2026-07-10**: Refined the homepage tutorial overview and status columns, aligning the Part 00 / Part 01 group names, lesson counts, and navigation status.
- **2026-06-26**: Reworked the homepage overview, status columns, and learning paths to clarify the relationship between Parts 00-04, cross-cutting topics, and collaborative study.
- **2026-06-15**: Finalized the Part 0 / 1 grouping and guide cleanup, unified the part-level navigation, connected the page comments to GitHub Discussions, and continued expanding Part 1 content, bridge pages, and notebook structure.
- **2026-06-13**: Fixed dead links and added placeholder pages for unfinished content to prevent 404s in learning entry points.
- **2026-04-21**: Updated Colab badges to point to the official `datawhalechina` repository.
- **2026-04-20**: Launched the site homepage and part guides; added Part 0 prerequisites and Part 1 practice content to unify the learning path.
- **2026-04-18 ~ 2026-04-19**: Refactored Part 2 / 3 content, polishing notebooks, answer sections, and operator implementation notes.
- **2026-04-02**: Completed the initial tutorial notebooks, docs, and test scripts.

> Path compatibility note: Part 03 has been renamed from `03_CUDA_and_Triton_Kernels` to `03_Triton_Kernels`, and CUDA / system optimization content has moved to Part 04. Old web paths keep migration pages, but new links should use `03_Triton_Kernels`.

</details>

## 🚀 Quick Start

We recommend starting with [`Part 02`](./02_PyTorch_Algorithms/intro.md) to build implementation intuition, then backfilling `Part 00` / `Part 01` as needed before moving to `Part 03` / `Part 04`. You can also enter directly through a training, inference, or memory topic.

<details>
<summary>View online, local, and CNB options</summary>

### Option 1: Read Online

Visit the online platform:

[https://datawhalechina.github.io/llm-algo-leetcode/](https://datawhalechina.github.io/llm-algo-leetcode/)

Suitable for:
- Skimming the table of contents first and then jumping to the part you need
- Reading the part guides first
- Part 00 / 01 / 02 can run on Colab CPU for most basic exercises
- Real inference, training, and memory experiments, as well as Part 03 / 04, need a Colab GPU runtime

### Option 2: Local Development

```bash
git clone https://github.com/datawhalechina/llm-algo-leetcode.git
cd llm-algo-leetcode
conda env create -f environment.yml
conda activate llm_algo
python -m pip install -r requirements/torch-cpu.txt
jupyter lab
```

Suitable for:
- Running Part 00 / 01 / 02 locally on CPU
- Controlling your own Python / PyTorch / CUDA versions
- More stable offline debugging
- Part 03 / 04 require a local NVIDIA GPU

For real training, memory, or inference projects, use the CUDA environment instead:

```bash
conda env create -f environment-gpu.yml
conda activate llm_algo_gpu
python -m pip install -r requirements/torch-cu128.txt
jupyter lab
```

Cloud runtimes or Colab sessions that already provide CUDA PyTorch do not need this installation. Run the project preflight first and reuse the current Kernel when its PyTorch, CUDA, and GPU checks pass.

For environment details and platform differences, see [docs/guide.md](./docs/guide.md).

### Option 3: CNB Unified Delivery

If you want the same runtime style used by the repository, use the CNB unified environment.

Suitable for:
- Team collaboration
- Consistent experiment images
- Lower local environment drift
- Part 00 / 01 / 02 can use CNB CPU
- Part 03 / 04 need a CNB GPU session

See [docs/guide.md](./docs/guide.md) for the exact environment rules and scope.

</details>

## 📖 More Resources

- [docs/guide.md](./docs/guide.md) - environment and learning modes
- [docs/contributing.md](./docs/contributing.md) - how to contribute to development and testing
- [docs/maintenance.md](./docs/maintenance.md) - maintenance rules for parts, links, tests, and releases
- [Automated Test Script Index](./docs/maintenance.md#测试脚本索引) - entry points for automated verification scripts

## 👨‍💻 Contributors

| Name | Role | Description |
| :---- | :---- | :---- |
| lynn_jingjing | Project initiator | An algorithm engineer |

*(Feel free to add your name here! )*

## 📄 License

Tutorial text in this repository is licensed under [CC BY 4.0](./LICENSE), and code is licensed under [Apache-2.0](./LICENSE-CODE). `.ipynb` files are mixed-content notebooks, so please follow the corresponding license by cell type.
