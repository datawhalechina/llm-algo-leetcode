<h1 align="center">llm-algo-leetcode | 大模型算法与系统教程</h1>
<p align="center">
  Runnable notebooks for LLM algorithms and systems.<br>
  面向大模型算法与系统的可运行 Notebook 教程。
</p>

<p align="center">
  <strong>主学习路线 / Core Routes</strong><br>
  <a href="./topic_discussion/llm_architecture_evolution/intro.md">大模型架构演进 / Architecture</a> ·
  <a href="./topic_discussion/post_training_optimization/intro.md">后训练优化 / Post-Training</a> ·
  <a href="./topic_discussion/inference_optimization/intro.md">推理优化 / Inference</a> ·
  <a href="./topic_discussion/performance_optimization/intro.md">性能优化 / Performance</a> ·
  <a href="./topic_discussion/operator_optimization/intro.md">算子优化 / Operator</a>
</p>

<p align="center">
  <strong>专题导航 / Topic Navigation</strong><br>
  <a href="./topic_discussion/intro.md#基础与诊断入口">基础与诊断 / Foundations</a> ·
  <a href="./topic_discussion/intro.md#横向工程专题">横向工程 / Cross-Cutting</a> ·
  <a href="./topic_discussion/intro.md#领域专题">领域专题 / Domains</a> ·
  <a href="./topic_discussion/intro.md">全部专题 / All Topics</a>
</p>


[中文版 (Chinese)](#中文版) | [English Version](#english-version)

---

# 中文版

## 🎯 项目简介

这是一个以 Notebook 为载体的大模型算法与系统教程。教程从 Part 02 的 PyTorch 算法实践出发，按需回补 Part 00 / Part 01 的基础，并进一步延伸到 Part 03 的 Triton 算子开发和 Part 04 的 CUDA 与系统优化。

内容围绕可运行实现、实验验证和专题化学习路线组织，帮助学习者逐步理解模型如何计算、如何训练、如何推理，以及性能问题如何被测量和优化。

### ✨ 项目特点

1. **Notebook-first**：每节围绕可运行代码、题目区、答案区和基础验证展开，适合边学边改。
2. **按目标进入**：既支持从 Part 02 开始建立算法实践，也支持按五条主学习路线或基础、横向、领域专题跳读。
3. **算法到系统贯通**：沿着 `PyTorch → Triton → CUDA` 逐步下探，连接模型实现、算子优化、显存管理、通信与系统性能。
4. **用证据做项目判断**：通过 benchmark、profiling、真实 GPU 和 inference backend 实验，对吞吐、延迟、显存、质量和成本进行比较。

### 👥 适合对象

- **大模型算法学习者**：希望通过 PyTorch Notebook 理解 Transformer、训练、微调、推理和模型压缩。
- **性能与系统学习者**：希望进一步学习显存、Profiling、通信、Triton、CUDA 和 GPU 优化。
- **项目实践者**：希望通过 benchmark、真实 GPU 和 inference backend 实验，建立可复现、可比较的工程判断。

## 🌐 教程总览

教程提供两种阅读方式：按 `Part 00 -> Part 04` 逐层推进，或从大模型架构演进、后训练优化、推理优化、性能优化和算子优化五条主学习路线进入，再按问题回补基础与横向专题。五条路线的排列表示从模型结构、能力形成到服务运行、性能诊断和底层实现的对象层次，不是严格的前置依赖。推荐先从 [`Part 02`](./02_PyTorch_Algorithms/intro.md) 建立算法实践感，再按需要回补 `Part 00` / `Part 01`，最后进入 `Part 03` / `Part 04`；完成相关主线后，可进入 `Part 05` 的综合项目，把机制、实验和架构决策收敛为可交付方案。

[topic_discussion](./topic_discussion/intro.md) 用于组织跨 Part 的专题路线。页面顶部提供常用专题入口，下面的资产总览和专题总览分别说明主线结构与专题覆盖范围。

![llm-algo-leetcode 教程主图：Part 00–05、五条主学习路线与支撑专题](./docs/public/decorative/tutorial_roadmap_hero_v4.svg)


<details>
<summary>📚 查看完整资产总览</summary>

![教程资产关系：从知识到可复核结论](./docs/public/readme/tutorial_assets_map.svg)

这套教程不要求从 `00` 开始按顺序硬读。`00` 主要是前置补齐区，如果你已有基础，可以直接从最相关的部分开始；下面这张表会直接告诉你：每一部分学什么、包含哪些组、适合谁、当前进度如何。

| 部分 | 组别 | 内容定位 | 适合对象 | 状态 |
| ---- | ---- | ---- | ---- | ---- |
| [`第零部分：前置知识与环境准备（5 组 / 20 节，已完成，持续优化）`](./00_Prerequisites/intro.md) | [`0A Python 基础与数据表示（4 节）`](./00_Prerequisites/0A.md) / [`0B PyTorch 张量与自动求导（4 节）`](./00_Prerequisites/0B.md) / [`0C PyTorch 模型与训练生命周期（4 节）`](./00_Prerequisites/0C.md) / [`0D Transformer Block 与模型组件（4 节）`](./00_Prerequisites/0D.md) / [`0E 运行诊断与实验基础（4 节）`](./00_Prerequisites/0E.md) | 把 Python、NumPy、PyTorch、训练循环、调试工具和性能意识搭好。 | 第一次进入教程、需要补齐入门前置的人。 | ✅ 已完成，持续优化 |
| [`第一部分：硬件、数学与系统（6 组 / 34 节，已完成，持续优化）`](./01_Hardware_Math_and_Systems/intro.md) | [`1A 数值与规模（4 节）`](./01_Hardware_Math_and_Systems/1A.md) / [`1B 单设备执行、内存与访存（9 节）`](./01_Hardware_Math_and_Systems/1B.md) / [`1C 通信、并行与容错（6 节）`](./01_Hardware_Math_and_Systems/1C.md) / [`1D 执行模型、异步调度与算子映射（7 节）`](./01_Hardware_Math_and_Systems/1D.md) / [`1E 编译与后端（4 节）`](./01_Hardware_Math_and_Systems/1E.md) / [`1F 硬件适配与系统约束（4 节）`](./01_Hardware_Math_and_Systems/1F.md) | 理解硬件、规模、访存、通信、执行与系统约束。 | 想先弄清“为什么要这样写”和“为什么要这样部署”的学习者。 | ✅ 已完成，持续优化 |
| [`第二部分：PyTorch 算法实战（10 组，已完成，持续优化）`](./02_PyTorch_Algorithms/intro.md) | [`2.1 Decoder 组件与注意力`](./02_PyTorch_Algorithms/2_1.md) / [`2.2 模型架构`](./02_PyTorch_Algorithms/2_2.md) / [`2.3 监督微调与训练控制`](./02_PyTorch_Algorithms/2_3.md) / [`2.4 偏好优化与对齐`](./02_PyTorch_Algorithms/2_4.md) / [`2.5 反向传播与训练内存`](./02_PyTorch_Algorithms/2_5.md) / [`2.6 推理执行基础`](./02_PyTorch_Algorithms/2_6.md) / [`2.7 推理策略与服务调度`](./02_PyTorch_Algorithms/2_7.md) / [`2.8 量化与低比特适配`](./02_PyTorch_Algorithms/2_8.md) / [`2.9 通信与并行`](./02_PyTorch_Algorithms/2_9.md) / [`2.10 项目验证与工程收口`](./02_PyTorch_Algorithms/2_10.md) | 在 PyTorch 层把算法、模型、推理、压缩、并行与项目验证先跑通。 | 希望先用熟悉工具建立实现感的人。 | ✅ 已完成，持续优化 |
| [`第三部分：Triton 算子开发（5 组 / 15 节，已完成，持续优化）`](./03_Triton_Kernels/intro.md) | [`3.1 基础篇（5 节）`](./03_Triton_Kernels/intro.md) / [`3.2 过渡篇（2 节）`](./03_Triton_Kernels/intro.md) / [`3.3 进阶A：Attention优化（3 节）`](./03_Triton_Kernels/intro.md) / [`3.4 进阶B：推理优化（2 节）`](./03_Triton_Kernels/intro.md) / [`3.5 项目篇（3 节）`](./03_Triton_Kernels/intro.md) | 把前面学到的算子和优化思路落到 GPU kernel。 | 希望从 PyTorch 走向 Triton 的学习者。 | ✅ 已完成，持续优化 |
| [`第四部分：CUDA C++ 与系统优化（4 组 / 16 节，建设中）`](./04_CUDA_and_System_Optimization/intro.md) | [`4.1 CUDA 编程基础（4 节）`](./04_CUDA_and_System_Optimization/intro.md) / [`4.2 系统级性能优化（4 节）`](./04_CUDA_and_System_Optimization/intro.md) / [`4.3 分布式训练工程（4 节）`](./04_CUDA_and_System_Optimization/intro.md) / [`4.4 架构视野（4 节）`](./04_CUDA_and_System_Optimization/intro.md) | 进一步下探到 CUDA、系统调优和工程化架构。 | 准备做底层性能优化和工程落地的人。 | 🛠 建设中 |
| [`第五部分：综合项目与架构决策`](./05_Integrated_Projects_and_Decisions/intro.md) | 综合项目、架构选型与成本决策 | 汇总多条主线的机制、实验和工程约束。 | 已完成主线学习、准备进行系统收口的人。 | 🛠 建设中 |

</details>

<details>
<summary>🧭 查看完整专题总览</summary>

| 层级 | 入口 | 内容定位 | 适合对象 | 状态 |
| ---- | ---- | ---- | ---- | ---- |
| 主学习路线（建设中） | [`大模型架构演进（LLM Architecture Evolution）`](./topic_discussion/llm_architecture_evolution/intro.md) | 从 Decoder Block、Attention、位置、MoE 到模型结构审计。 | 想读懂模型结构选择及其质量、资源与系统影响的学习者。 | 🛠 建设中 |
| 主学习路线（建设中） | [`后训练优化（Post-Training Optimization）`](./topic_discussion/post_training_optimization/intro.md) | SFT / LoRA 基础、偏好数据、DPO、GRPO 与项目交付。 | 想从监督微调继续进入偏好优化与对齐的学习者。 | 🛠 建设中 |
| 主学习路线 | [`推理优化（Inference Optimization）`](./topic_discussion/inference_optimization/intro.md) | FlashAttention、解码、PagedAttention、cache 与 benchmark。 | 想系统理解推理加速路径的学习者。 | ✅ 持续优化 |
| 主学习路线 | [`性能优化（Performance Optimization）`](./topic_discussion/performance_optimization/intro.md) | 从性能分析、显存管理、通信代价到端到端决策。 | 想系统定位并优化训练、推理和系统性能的学习者。 | 🛠 建设中 |
| 主学习路线（建设中） | [`算子优化（Operator Optimization）`](./topic_discussion/operator_optimization/intro.md) | 从硬件直觉、图级优化、Triton、经典算子、Attention 到 CUDA 与端到端验证。 | 想从算子实现走向 kernel 和端到端性能优化的学习者。 | 🛠 建设中 |
| 基础支撑专题 | [`反向传播与训练机制（Backpropagation and Training Mechanics）`](./topic_discussion/backpropagation_training_mechanism/intro.md) | autograd、backward、checkpointing、offload 与训练节奏。 | 想补训练机制底座的学习者。 | ✅ 持续优化 |
| 基础支撑专题 | [`性能分析（Performance Analysis）`](./topic_discussion/performance_analysis/intro.md) | 性能取证、trace 阅读、回归验证和行动决策。 | 想系统补性能意识与排障方法的学习者。 | ✅ 持续优化 |
| 横切支撑专题 | [`量化与压缩（Quantization and Compression）`](./topic_discussion/quantization_compression/intro.md) | PTQ、QAT、GPTQ、AWQ、FP8 与部署决策。 | 想同时考虑精度、显存、吞吐和部署取舍的学习者。 | ✅ 持续优化 |
| 横切支撑专题 | [`通信与并行（Communication and Parallelism）`](./topic_discussion/communication_parallelism/intro.md) | NCCL、AllReduce、ZeRO、PP、TP 和并行验证。 | 想理解多卡训练和通信边界的学习者。 | ✅ 持续优化 |
| 横切支撑专题（建设中） | [`数据工程（Data Engineering）`](./topic_discussion/data_engineering/intro.md) | 数据生命周期、解析、清洗、去重、配比、合成、血缘与工程化。 | 想同时服务训练、RAG 与数据质量治理的学习者。 | 🛠 建设中 |
| 横切支撑专题（建设中） | [`部署与异构系统（Deployment and Heterogeneous Systems）`](./topic_discussion/deployment_heterogeneous/intro.md) | 模型产物、推理 backend、CPU/GPU/NPU 协同、资源编排和部署验证。 | 想理解模型如何从文件进入可交付服务的学习者。 | 🛠 建设中 |
| 领域专题（建设中） | [`Agent 基础设施（Agent Infrastructure）`](./topic_discussion/agent/intro.md) | 工具轨迹、沙盒执行、任务调度、可观测性与 Agentic RL。 | 想让模型可靠完成多步工具任务的学习者。 | 🛠 建设中 |
| 领域专题（建设中） | [`多模态与视觉推理（Multimodal and Vision Inference）`](./topic_discussion/multimodal/intro.md) | 以多模态推理为主线，覆盖视觉表示、对齐、视觉 token、推理成本和任务评测；训练作为共享前置。 | 想进入视觉语言模型、文档理解或图文推理的学习者。 | 🛠 建设中 |

</details>

<details>
<summary>🆕 查看更新时间线</summary>

- **2026-09-30**：统一五条主学习路线及基础、横向、领域专题的中英文名称与入口；将显存优化内容正式承接到性能优化，将 SFT / LoRA 与图级编译分别并入后训练优化和算子优化；更新 Part 05 收口定位、专题资产与首页结构图。
- **2026-09-11**：完成当时主路线与横向专题的阶段性梳理，补充 Task0–6 学习入口与项目分工；统一项目节的类型、环境和证据边界，并完成源文件到 `docs/` 的镜像同步与 VitePress 构建验证。Part 02 的 60–76 项目进入持续优化和 GPU 数据采集准备阶段。
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

推荐从 [`Part 02`](./02_PyTorch_Algorithms/intro.md) 开始：先通过 PyTorch Notebook 建立算法实现感，再根据知识缺口回补 `Part 00` / `Part 01`，最后进入 `Part 03` / `Part 04` 的 GPU 底层优化，并由 `Part 05` 完成项目与决策收口。也可以直接选择一条主学习路线，再按问题进入基础、横向或领域专题。运行 Notebook 前，请先查看 [使用指南](./docs/guide.md) 和对应小节中的环境说明。

| 想完成的任务 | 默认运行入口 | 开始前查看 |
|:---|:---|:---|
| 阅读或运行 CPU 机制练习 | 基础环境 + CPU PyTorch | [使用指南](./docs/guide.md) |
| 运行单卡训练、量化或 GPU 实验 | CUDA PyTorch + 对应能力 profile | [模型、执行栈与环境资产](./docs/gpu_environment_assets.md) |
| 运行 Serving 或多卡项目 | 独立 backend / distributed 环境，并先执行预检 | [使用指南](./docs/guide.md)与[模型、执行栈与环境资产](./docs/gpu_environment_assets.md) |

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

- [使用指南](./docs/guide.md)——选择学习入口、运行环境和实验路径
- [贡献指南](./docs/contributing.md)——参与内容、代码与测试贡献
- [维护与发布手册](./docs/maintenance.md)——源码同步、统一验证、迁移与发布

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
2. **Goal-oriented entry**: Start from Part 02 for implementation practice, or follow the architecture, post-training, inference, performance, and operator routes.
3. **From algorithms to systems**: Follow the `PyTorch → Triton → CUDA` path across model implementation, kernel optimization, memory, communication, and system performance.
4. **Evidence-based projects**: Use benchmarks, profiling, real GPU runs, and inference backends to compare throughput, latency, memory, quality, and cost.

### 👥 Suitable For

- **LLM Algorithm Learners**: Use PyTorch notebooks to understand Transformers, training, fine-tuning, inference, and model compression.
- **Performance and Systems Learners**: Study memory, profiling, communication, Triton, CUDA, and GPU optimization.
- **Project Practitioners**: Build reproducible engineering judgment through benchmarks, real GPU runs, and inference backends.


## 🌐 Tutorial Overview

This tutorial offers two ways to read: follow the `Part 00 -> Part 04` progression, or enter through one of five main routes—LLM architecture evolution, post-training optimization, inference optimization, performance optimization, and operator optimization—then backfill foundation and cross-cutting topics as needed. Their left-to-right order describes a progression of objects from model structure and capability formation to serving, diagnosis, and low-level implementation; it is not a strict prerequisite chain. We recommend starting with [`Part 02`](./02_PyTorch_Algorithms/intro.md) to build implementation intuition, then backfilling `Part 00` / `Part 01` before moving to `Part 03` / `Part 04`; after completing the relevant routes, use `Part 05` integrated projects to turn mechanisms, experiments, and architecture choices into a deliverable plan.

[topic_discussion](./topic_discussion/intro.md) organizes cross-Part topic paths. The top of the page provides common topic entries; the asset and topic overviews below describe the main structure and topic coverage.

![llm-algo-leetcode tutorial map: Parts 00–05, five main routes, and supporting topics](./docs/public/decorative/tutorial_roadmap_hero_v4.svg)

<details>
<summary>📚 View the complete asset overview</summary>

You do not need to start from `00` in strict order. `00` is the prerequisite lane; if you already have the background, jump directly to the part that matches your goal. The table below summarizes each part, its groups, its audience, and its status.

| Part | Groups | Content Positioning | Suitable For | Status |
| ---- | ---- | ---- | ---- | ---- |
| [Part 00: Prerequisites and Environment Setup (5 groups / 20 lessons)](./00_Prerequisites/intro.md) | [0A Python Basics and Data Representation (4 lessons)](./00_Prerequisites/0A.md) / [0B PyTorch Tensors and Autograd (4 lessons)](./00_Prerequisites/0B.md) / [0C PyTorch Model and Training Lifecycle (4 lessons)](./00_Prerequisites/0C.md) / [0D Transformer Block and Model Components (4 lessons)](./00_Prerequisites/0D.md) / [0E Runtime Diagnostics and Experiment Foundations (4 lessons)](./00_Prerequisites/0E.md) | Prerequisites, engineering basics, and notebook-first practice. | First-time learners who need prerequisite support. | ✅ Complete, continuously refining |
| [Part guide: Hardware, Math, and Systems (6 groups / 34 lessons)](./01_Hardware_Math_and_Systems/intro.md) | [1A Numerics and Scale (4 lessons)](./01_Hardware_Math_and_Systems/1A.md) / [1B Single-Device Execution, Memory and Access (9 lessons)](./01_Hardware_Math_and_Systems/1B.md) / [1C Communication, Parallelism and Fault Tolerance (6 lessons)](./01_Hardware_Math_and_Systems/1C.md) / [1D Execution Models, Asynchrony and Operator Mapping (7 lessons)](./01_Hardware_Math_and_Systems/1D.md) / [1E Compilers and Backends (4 lessons)](./01_Hardware_Math_and_Systems/1E.md) / [1F Hardware Adaptation and System Constraints (4 lessons)](./01_Hardware_Math_and_Systems/1F.md) | Hardware, scale, memory access, communication, execution, and system constraints. | Learners who want to understand why things are written and deployed this way. | ✅ Complete, continuously refining |
| [Part 02: PyTorch Algorithm Practice (10 groups)](./02_PyTorch_Algorithms/intro.md) | [2.1 Decoder Components and Attention](./02_PyTorch_Algorithms/2_1.md) / [2.2 Model Architecture](./02_PyTorch_Algorithms/2_2.md) / [2.3 Supervised Fine-Tuning and Training Control](./02_PyTorch_Algorithms/2_3.md) / [2.4 Preference Optimization and Alignment](./02_PyTorch_Algorithms/2_4.md) / [2.5 Backpropagation and Training Memory](./02_PyTorch_Algorithms/2_5.md) / [2.6 Inference Execution Fundamentals](./02_PyTorch_Algorithms/2_6.md) / [2.7 Inference Strategies and Serving Scheduling](./02_PyTorch_Algorithms/2_7.md) / [2.8 Quantization and Low-Bit Adaptation](./02_PyTorch_Algorithms/2_8.md) / [2.9 Communication and Parallelism](./02_PyTorch_Algorithms/2_9.md) / [2.10 Project Validation and Engineering Closure](./02_PyTorch_Algorithms/2_10.md) | PyTorch-level practice for algorithms, models, inference, compression, parallelism, and project validation. | Learners who want to build implementation intuition with familiar tools. | ✅ Complete, continuously refining |
| [Part 03: Triton Kernel Development (5 groups / 15 lessons)](./03_Triton_Kernels/intro.md) | [3.1 Foundations (5 lessons)](./03_Triton_Kernels/intro.md) / [3.2 Transition (2 lessons)](./03_Triton_Kernels/intro.md) / [3.3 Advanced A: Attention Optimization (3 lessons)](./03_Triton_Kernels/intro.md) / [3.4 Advanced B: Inference Optimization (2 lessons)](./03_Triton_Kernels/intro.md) / [3.5 Projects (3 lessons)](./03_Triton_Kernels/intro.md) | Triton kernel development. | Learners who want to move from PyTorch to Triton. | ✅ Complete, continuously refining |
| [Part 04: CUDA C++ and System Optimization (4 groups / 16 lessons)](./04_CUDA_and_System_Optimization/intro.md) | [4.1 CUDA Programming Basics (4 lessons)](./04_CUDA_and_System_Optimization/intro.md) / [4.2 System-Level Performance Optimization (4 lessons)](./04_CUDA_and_System_Optimization/intro.md) / [4.3 Distributed Training Engineering (4 lessons)](./04_CUDA_and_System_Optimization/intro.md) / [4.4 Architecture Perspective (4 lessons)](./04_CUDA_and_System_Optimization/intro.md) | CUDA C++ and system optimization. | Learners preparing for low-level performance optimization and engineering deployment. | 🛠 In progress |
| [Part 05: Integrated Projects and Architecture Decisions](./05_Integrated_Projects_and_Decisions/intro.md) | Integrated projects, architecture selection, and cost decisions | Consolidates mechanisms, experiments, and engineering constraints across the main routes. | Learners ready to close the loop with system-level decisions. | 🛠 In progress |

</details>

<details>
<summary>🧭 View the complete topic overview</summary>

| Layer | Entry | Content Positioning | Suitable For | Status |
| ---- | ---- | ---- | ---- | ---- |
| Main Study Path (In Progress) | [LLM Architecture Evolution Topic](./topic_discussion/llm_architecture_evolution/intro.md) | Decoder Block, attention, position, MoE, and model architecture audit. | Learners who want to connect structural choices to quality, resources, and system impact. | 🛠 In progress |
| Main Study Path (In Progress) | [Post-Training Optimization Topic](./topic_discussion/post_training_optimization/intro.md) | SFT / LoRA foundations, preference data, DPO, GRPO, and project delivery. | Learners who want to continue from supervised fine-tuning into alignment. | 🛠 In progress |
| Main Study Path | [Inference Optimization Topic](./topic_discussion/inference_optimization/intro.md) | FlashAttention, decoding, PagedAttention, cache, and benchmark. | Learners who want practical inference acceleration. | ✅ Ongoing |
| Main Study Path | [Performance Optimization Topic](./topic_discussion/performance_optimization/intro.md) | Performance analysis, memory management, communication costs, and end-to-end decisions. | Learners who want to diagnose and optimize training, inference, and system performance. | 🛠 In progress |
| Main Study Path (In Progress) | [Operator Optimization Topic](./topic_discussion/operator_optimization/intro.md) | Graph rewrites, fusion, Triton, CUDA, memory access, autotuning, and kernel-to-end-to-end validation. | Learners who want to move from operator implementation to kernel and end-to-end optimization. | 🛠 In progress |
| Foundation Topic | [Backpropagation and Training Mechanics Topic](./topic_discussion/backpropagation_training_mechanism/intro.md) | Autograd, backward, checkpointing, offload, and training rhythm. | Learners who want stronger training-mechanism foundations. | ✅ Ongoing |
| Foundation Topic | [Performance Analysis Topic](./topic_discussion/performance_analysis/intro.md) | Evidence collection, trace reading, regression validation, and action decisions. | Learners who want systematic performance diagnosis and debugging methods. | ✅ Ongoing |
| Cross-Cutting Topic | [Quantization and Compression Topic](./topic_discussion/quantization_compression/intro.md) | PTQ, QAT, GPTQ, AWQ, FP8, and deployment decisions. | Learners balancing accuracy, memory, throughput, and deployment cost. | ✅ Ongoing |
| Cross-Cutting Topic | [Communication and Parallelism Topic](./topic_discussion/communication_parallelism/intro.md) | NCCL, AllReduce, ZeRO, PP, TP, and validation. | Learners who want to understand multi-GPU scaling and communication cost. | ✅ Ongoing |
| Cross-Cutting Topic (In Progress) | [Data Engineering Topic](./topic_discussion/data_engineering/intro.md) | Data lifecycle, parsing, cleaning, deduplication, mixtures, synthesis, lineage, and production pipelines. | Learners building reliable data paths for training, RAG, and evaluation. | 🛠 In progress |
| Cross-Cutting Topic (In Progress) | [Deployment and Heterogeneous Systems Topic](./topic_discussion/deployment_heterogeneous/intro.md) | Model artifacts, inference backends, CPU/GPU/NPU coordination, resource orchestration, and deployment validation. | Learners who want to understand how models become deployable services. | 🛠 In progress |
| Domain Topic (In Progress) | [Agent Infrastructure Topic](./topic_discussion/agent/intro.md) | Tool trajectories, sandboxed execution, task scheduling, observability, and agentic RL. | Learners building reliable multi-step tool-using systems. | 🛠 In progress |
| Domain Topic (In Progress) | [Multimodal and Vision Inference Topic](./topic_discussion/multimodal/intro.md) | Multimodal inference as the main line: visual representations, alignment, visual tokens, inference cost, and task evaluation; training is shared prerequisite. | Learners entering vision-language models, document understanding, or multimodal reasoning. | 🛠 In progress |

</details>

<details>
<summary>🆕 View update timeline</summary>

- **2026-09-30**: Unified the Chinese and English names and entry points for the five main routes plus foundation, cross-cutting, and domain topics; moved the former memory-optimization route into Performance Optimization, merged SFT / LoRA and graph compilation into Post-Training Optimization and Operator Optimization, and refreshed Part 05 positioning, topic assets, and the homepage structure map.
- **2026-09-11**: Consolidated the main routes and cross-cutting topics available at that stage; added Task 0–6 entry points and project roles; unified project types, environments, and evidence boundaries; and completed source-to-`docs/` mirroring plus VitePress build validation. Part 02 projects 60–76 then entered continued refinement and GPU-data preparation.
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

We recommend starting with [`Part 02`](./02_PyTorch_Algorithms/intro.md) to build implementation intuition, then backfilling `Part 00` / `Part 01` as needed before moving to `Part 03` / `Part 04`. You can also enter through one of the five main routes or a foundation, cross-cutting, or domain topic.

| Goal | Default runtime entry | Read first |
|:---|:---|:---|
| Read or run CPU mechanism exercises | Base environment + CPU PyTorch | [Usage guide](./docs/guide.md) |
| Run single-GPU training, quantization, or GPU experiments | CUDA PyTorch + the required capability profile | [Model, execution stack, and environment assets](./docs/gpu_environment_assets.md) |
| Run Serving or multi-GPU projects | An isolated backend / distributed environment with preflight checks | [Usage guide](./docs/guide.md) and [runtime assets](./docs/gpu_environment_assets.md) |

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

- [Usage Guide](./docs/guide.md) — learning entry points, runtime environments, and experiment paths
- [Contribution Guide](./docs/contributing.md) — contributing content, code, and tests
- [Maintenance and Release Manual](./docs/maintenance.md) — source synchronization, verification, migration, and release

## 👨‍💻 Contributors

| Name | Role | Description |
| :---- | :---- | :---- |
| lynn_jingjing | Project initiator | An algorithm engineer |

*(Feel free to add your name here! )*

## 📄 License

Tutorial text in this repository is licensed under [CC BY 4.0](./LICENSE), and code is licensed under [Apache-2.0](./LICENSE-CODE). `.ipynb` files are mixed-content notebooks, so please follow the corresponding license by cell type.
