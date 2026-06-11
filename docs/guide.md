# 使用指南

本页只说明各章节该用什么环境学习，以及哪些章节对环境有硬性要求。

## 先看结论

| 你的目标 | 推荐环境 | 备注 |
|---|---|---|
| 先入门 Chapter 0 / 1 | 在线 Notebook 或本地基础环境 | 主要是 Python、Jupyter、NumPy |
| 系统学习 Chapter 2 | 统一 Python 环境 + 本地 CPU；部分题再切 GPU | 大多数题可 CPU-first |
| 完整学习 Chapter 3 | Linux + NVIDIA GPU + CUDA / Triton | GPU-required |
| 想统一团队环境 | CNB / Docker / 云端 GPU | 适合一致性和复现 |

---

## 环境部署决策树

如果你不知道先选什么环境，按这个顺序判断：

1. **只是先看内容**
   - 直接用在线站点
   - 先看 Chapter 0 / 1 的导学页

2. **要做 Chapter 0 / 1**
   - 优先在线 Notebook
   - 也可以用本地基础环境
   - 目标是先把 Python、Jupyter、NumPy 跑通

3. **要做 Chapter 2**
   - 主路径：本地 conda 环境
   - 第二选择：CNB 统一环境
   - 统一一套 Python 依赖
   - CPU 足够覆盖大多数题；只有少数题才切 GPU
   - 如果后续要顺手衔接更多 GPU 相关题目，也可以直接走 CNB

4. **要做 Chapter 3**
   - 优先本地 GPU 或 CNB GPU 入口
   - 需要 CUDA / Triton / GPU 可见性
   - 当前 CNB 默认交互环境不等于 GPU 环境

5. **要给团队/课程统一环境**
   - 优先 CNB / Docker
   - 目标是“大家按同一套环境跑”

如果还是拿不准，默认顺序是：

**在线阅读 -> 在线 Notebook -> 本地 conda -> CNB CPU -> 本地 GPU / CNB GPU**

---

## 四层环境分层

### 1. 轻量学习层

适用章节：
- Chapter 0
- Chapter 1

环境特点：
- Python
- Jupyter / Notebook
- NumPy / 基础科学计算
- Colab / 在线 Notebook 可直接开始

这两章对部署要求低，重点是把基础概念和前置知识学稳，不需要一开始就上复杂容器或 GPU 工具链。

### 2. 主力学习层

适用章节：
- Chapter 2

环境特点：
- 统一 Python 环境
- PyTorch
- correctness 验证脚本
- 本地 CPU 足够覆盖大多数题

- Chapter 2 是 **CPU-first**
- 已确认至少 `21_Gradient_Checkpointing` 需要 NVIDIA GPU 才能测真实 CUDA 显存峰值
- 学习时建议尽量使用同一套 Python 依赖，保证练习和验证结果一致
- 如果不想手动配本地环境，也可以直接用 CNB 作为第二选择

### 3. 高门槛实验层

适用章节：
- Chapter 3

环境特点：
- Linux
- NVIDIA GPU
- CUDA
- Triton
- 编译工具链

- Chapter 3 是 **GPU-required**
- 少数页面可能有阅读路径或 CPU fallback，但不构成完整学习路径

### 4. 统一交付层

适用场景：
- 团队协作
- 统一实验镜像
- 云端开发
- 需要减少本地环境差异时

平台选项：
- Colab
- CNB
- Docker
- 云端 GPU 托管环境

这层主要服务 Chapter 2 后段和 Chapter 3，目标是把“能跑”变成“大家都按同一套环境跑”。

当前验证状态：
- **Colab**：已验证，可一键直达
- **Docker**：推荐以 **50 系 NVIDIA GPU** 作为当前已验证基线；40 系和 30 系后续再单独补充兼容矩阵与验证结果
- **CNB**：分成两层看，`push` / `pull_request` 负责流水线校验，`vscode` 负责读者打开后的交互环境；当前已完成仓库同步，交互环境还需要实机验证
- **ModelScope / 其他在线 Notebook**：待验证

---

## 操作系统和本地部署

本项目的本地部署不是只看“能装上依赖”，还要看操作系统、包管理方式和测试脚本是否统一。

### 操作系统差异

- **Linux 22.04**：当前已验证的主基线，优先作为本地开发和 Chapter 2 / 3 验证环境
- **WSL2**：可作为过渡方案，但尚未作为正式主基线承诺
- **macOS**：适合轻量阅读和部分 Python 逻辑练习；Chapter 3 的完整 GPU / Triton 体验不做默认承诺
- **Windows 原生环境**：不作为主支持路径，若要学习建议优先走 WSL2、CNB 或在线 Notebook

### conda / venv 选择

- **conda 是推荐主路径**：本项目的 `environment.yml`、`cnb/environment.yml` 都以 conda 为核心入口，便于统一 Python、PyTorch 和 GPU 依赖
- **venv 也可以用**：如果你只需要纯 Python 虚拟环境，可以用 venv，但需要手动按 `requirements/*.txt` 安装依赖
- **不建议混用**：同一份环境里不要同时把 conda、venv 和系统 Python 混着装，否则后面验证结果不稳定

### 依赖版本策略

- `requirements/base.txt`：基础版本来源
- `requirements/dev.txt`：开发和测试版本来源
- `requirements/gpu.txt`：Chapter 3 / Triton / CUDA 扩展版本来源
- `environment.yml`：只负责 Python 版本和把上述依赖串起来
- `cnb/environment.yml`：CNB 侧的统一环境骨架

更新依赖时，优先改 `requirements/*.txt`，再同步 `environment.yml` 和 `cnb/environment.yml`，避免出现“本地一套版本、云端另一套版本”的情况。

### 测试脚本的位置

测试脚本是本地环境部署的一部分，不只是“可选附加项”。

- `project_test_scripts.md`：测试脚本索引
- `test_chapter0_1_notebooks.py`：Chapter 0 / 1 的顺序执行验证
- `test_notebook_answers.py`：Chapter 2 / 3 的答案区验证
- `check_chapter_links.py`：站内链接检查

如果环境已经装好，但测试脚本跑不起来，通常说明依赖版本、OS 差异或 GPU 兼容性还没有收口。

### CNB 里怎么验证

CNB 的验证顺序和本地类似，但要先确认当前会话拿到的是项目环境，而不是系统 Python。

#### 1. 先检查交互环境

```bash
. /opt/conda/etc/profile.d/conda.sh
conda activate llm_algo_cnb_dev

python --version
python -m pip --version
python -c "import torch; print(torch.__version__)"
python -c "import triton; print(triton.__version__)"
```

如果这些命令都能正常执行，说明 CNB 交互环境已经进入项目的统一软件栈。

#### 2. 再验证 Chapter 0 / 1

```bash
python test_chapter0_1_notebooks.py
```

这一步用于确认基础 notebook、依赖和顺序执行链路没有问题。

#### 3. 再验证 Chapter 2

```bash
python test_notebook_answers.py --all --dir 02_PyTorch_Algorithms --mode both
```

Chapter 2 是 CNB 里最应该优先跑通的主链路。

#### 4. Chapter 3 只在有 GPU 时做完整验证

```bash
python test_notebook_answers.py --all --dir 03_CUDA_and_Triton_Kernels --mode both
```

如果当前 CNB 实例没有 GPU，`torch.cuda.is_available()` 会是 `False`，这时不要把 Chapter 3 的 GPU 结果当成最终验收。

### Chapter 3 的 GPU 入口

Chapter 3 需要单独的 GPU 验证入口，不建议和默认 CNB 交互环境混用。

- 默认 CNB 交互环境：面向 Chapter 0 / 1 / 2，主要验证 Python、Notebook 和 CPU 路径
- GPU 验证入口：面向 Chapter 3，必须使用平台分配到 GPU 节点的会话
- 对应的 CNB 入口名：`vscode-gpu`
- 验证脚本：`scripts/validate_chapter3_gpu.sh`

如果当前会话没有 GPU，就只做：
- 环境检查
- 站内链接检查
- Chapter 3 的非 GPU 路径检查

不要把 `torch.cuda.is_available() == False` 的会话写成 Chapter 3 最终通过。

### Notebook 使用

- 执行单元格：`Shift + Enter` 或 `Ctrl + Enter`
- 执行所有单元格：`Run -> Run All Cells`
- 刷题流程：先看导学，再填 TODO，再跑测试
- 如果 notebook 报 `name not defined`，通常是前面的 cell 没按顺序执行

---

## Chapter 2 / 3 的实际要求

### Chapter 2

- 整体定位：CPU-first
- 大多数 notebook：可在 CPU 环境下完成学习和 correctness 验证
- 已确认 GPU 题：`21_Gradient_Checkpointing`
- 推荐策略：统一 Python 环境，GPU 用于后段实验和真实性能验证
- 已验证：当前本机 `llm_algo` 环境下，Chapter 2 答案区全量通过

### Chapter 3

- 整体定位：GPU-required
- 完整体验：Linux + NVIDIA GPU + CUDA / Triton
- 代码审计结果：本章直接面向 GPU 内核、显存和通信行为
- 推荐策略：把 Linux 作为默认参考环境，其他系统只作为补充
- 已验证：当前本机 `llm_algo` 环境下，Chapter 3 答案区全量通过

### 本地平台备注

- **Linux 22.04**：已验证，是当前最稳的本地参考环境
- **50 系 NVIDIA GPU**：已验证，是当前本机 GPU 学习与 Chapter 3 验证基线
- **40 系 NVIDIA GPU**：暂未作为已验证基线，后续补充兼容矩阵
- **WSL2**：理论上可作为过渡方案，但尚未作为主验证基线
- **macOS**：可能接近可用，但尚未完成逐项验证，不应默认承诺 Chapter 3 完整体验

### 本地 GPU 兼容矩阵（当前）

| GPU 代际 | 当前状态 | 适用范围 | 备注 |
|---|---|---|---|
| 50 系 NVIDIA GPU | 已验证 | Chapter 2 / Chapter 3 本地验证基线 | 当前本机实测基线，优先按此参考 |
| 40 系 NVIDIA GPU | 待补充验证 | Chapter 3 兼容目标 | 后续单独补版本与兼容结果，不默认承诺 |
| 30 系 NVIDIA GPU | 待补充验证 | Chapter 3 兼容目标 | 仅作为潜在兼容目标，需单独实测 |

---

## 怎么选

### 只想快速开始
- 用在线 Notebook
- 先学 Chapter 0 / 1

### 想系统学 Chapter 2
- 用本地基础环境
- 保持同一套 Python 依赖
- 需要真实显存或性能再切 GPU

### 想完整学 Chapter 3
- 用 Linux + NVIDIA GPU
- 先装好 CUDA / Triton / 编译工具链

### 想统一环境
- 优先 CNB / Docker / 云端 GPU
- 适合团队、课程和长期复现

---

## 环境文件怎么分

- `environment.yml`：本地主入口
- `requirements/base.txt`：基础依赖
- `requirements/dev.txt`：开发和测试依赖
- `requirements/gpu.txt`：Chapter 3 / Triton / GPU 扩展依赖
- `cnb/README.md`：未来云端统一环境说明
- `cnb/environment.yml`：未来云端环境骨架

---

## 相关阅读

- [维护与发布手册](./maintenance.md)
- [Chapter 2 导学](./02_PyTorch_Algorithms/intro.md)
- [Chapter 3 导学](./03_CUDA_and_Triton_Kernels/intro.md)
