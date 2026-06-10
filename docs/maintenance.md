# 维护与发布手册

本页用于记录项目维护者和贡献者的日常工作约定，目标是把仓库资产、章节汇总、使用方式、验证方式和发布流程放在同一处，避免维护信息分散。

## 仓库资产总览

### 主要入口

- `README.md`：项目总入口，给首次进入仓库的人看
- `docs/index.md`：站点首页，给浏览文档站的人看
- `docs/guide.md`：使用指南，说明如何选择环境和学习方式
- `docs/contributing.md`：贡献指南，说明如何参与开发和测试
- `docs/maintenance.md`：维护与发布手册，说明仓库如何维护

### 章节内容

- `docs/00_Prerequisites/`：Chapter 0，前置知识与环境准备
- `docs/01_Hardware_Math_and_Systems/`：Chapter 1，硬件、算力推导与系统级理论
- `docs/02_PyTorch_Algorithms/`：Chapter 2，PyTorch 算法实战
- `docs/03_CUDA_and_Triton_Kernels/`：Chapter 3，CUDA C++ 与 Triton 算子开发

### 练习与验证

- `test_notebook_answers.py`：Chapter 2 / 3 的通用 Notebook 验证脚本
- `test_chapter0_1_notebooks.py`：Chapter 0 / 1 的顺序执行验证脚本
- `check_chapter_links.py`：站内链接检查脚本

### 过程文件

- `process/`：交接、计划、总结等过程文件的统一存放目录

### 环境与骨架

- `environment.yml`：本地主入口环境
- `requirements/base.txt`
- `requirements/dev.txt`
- `requirements/gpu.txt`
- `cnb/README.md`
- `cnb/environment.yml`

#### 版本与包管理约定

- `requirements/base.txt`：基础依赖版本的主来源
- `requirements/dev.txt`：开发和测试依赖版本的主来源
- `requirements/gpu.txt`：Chapter 3 / Triton / CUDA 扩展依赖版本的主来源
- `environment.yml`：只负责 Python 版本和串联依赖，不应重复维护另一套独立版本表
- `cnb/environment.yml`：CNB 侧环境骨架，原则上复用同一套版本约定

更新依赖时，优先修改 `requirements/*.txt`，再同步 `environment.yml` 和 `cnb/environment.yml`，避免本地、CNB、Docker 出现版本漂移。

## 源文件与生成物

仓库里的章节内容遵循“源文件在根目录，网页产物在 `docs/`”的原则。

### 源文件

- 根目录下的 `00_Prerequisites/`
- 根目录下的 `01_Hardware_Math_and_Systems/`
- 根目录下的 `02_PyTorch_Algorithms/`
- 根目录下的 `03_CUDA_and_Triton_Kernels/`
- 根目录下的章节 `intro.md`
- 根目录下的章节练习 `*.ipynb`
- 根目录下的章节理论 `*.md`

### 生成物

- `docs/00_Prerequisites/`
- `docs/01_Hardware_Math_and_Systems/`
- `docs/02_PyTorch_Algorithms/`
- `docs/03_CUDA_and_Triton_Kernels/`

这些目录中的章节页面通常由 `convert_notebook.py` 从根目录源文件同步生成。维护时应优先修改根目录源文件，再重新生成 `docs/` 页面。

### 手工维护文件

以下文件不属于自动生成物，应该直接手工维护：

- `README.md`
- `docs/index.md`
- `docs/guide.md`
- `docs/contributing.md`
- `docs/maintenance.md`
- `docs/.vitepress/config.mts`

## 维护目标

- 保持章节内容、站点导航、Notebook 入口和测试脚本同步
- 控制改动范围，避免跨章节的大面积联动回归
- 将可验证的变更拆成小提交，便于 review 和回滚
- 让维护工作可以被复用，而不是依赖某个人的记忆
- 把操作系统、包版本、测试脚本和云端环境纳入同一套维护口径

## 本地 GPU 验证基线

- 当前已验证的本地 GPU 基线是 **Linux 22.04 + 50 系 NVIDIA 显卡 + `llm_algo` conda 环境**
- Chapter 2 在该环境下的答案区已验证通过
- Chapter 3 在该环境下的答案区已验证通过
- 40 系显卡暂未作为已验证基线，后续应单独补充兼容矩阵与版本验证

## 过程文件约定

仓库根目录下的 `process/` 目录专门存放维护过程中的临时记录和交接材料，例如：

- `SESSION_HANDOFF.md`
- `PLAN_*.md`
- `SUMMARY_*.md`

这类文件只用于维护协作，不进入站点导航，不作为学习内容展示，不参与章节侧边栏配置。

当一轮维护结束后，这些文件可以按需要：

- 保留在 `process/` 中作为历史记录
- 或者统一清理删除

不要把它们散落在仓库根目录，避免和正式文档入口混在一起。

## 当前维护对象

- `README.md`：项目首页和总入口
- `docs/index.md`：站点首页
- `docs/.vitepress/config.mts`：侧边栏和导航
- `docs/guide.md`：环境与平台说明
- `docs/contributing.md`：贡献和测试说明
- `docs/maintenance.md`：维护与发布说明
- `check_chapter_links.py`：站内链接检查
- `test_notebook_answers.py`：通用 Notebook 答案验证
- `test_chapter0_1_notebooks.py`：Chapter 0 / 1 顺序执行验证

## 常规维护流程

### 1. 先定边界

确认这次修改属于哪一类：

- 文档收口
- 章节内容修复
- Notebook 练习补齐
- 站点导航调整
- 环境文件更新
- 测试脚本调整

如果一项改动会同时触发多类变更，优先拆分成多个 commit。

### 2. 先改入口，再改内容

如果涉及章节结构变化，先同步：

- `README.md`
- `docs/index.md`
- `docs/.vitepress/config.mts`
- 章节 `intro.md`

然后再迁移具体文件和链接。

### 3. 先做本地验证

建议按改动类型选择测试：

- 文档链接：`python check_chapter_links.py`
- 站点构建：`cd docs && npm run docs:build`
- Chapter 0 / 1 练习：`python test_chapter0_1_notebooks.py`
- Chapter 2 / 3 题目答案：`python test_notebook_answers.py --all --dir 02_PyTorch_Algorithms --mode both`
- Chapter 3 内核题：`python test_notebook_answers.py --all --dir 03_CUDA_and_Triton_Kernels --mode both`

### 4. 再做提交

推荐按功能拆 commit：

- `docs(...)`：导学、首页、站点导航、链接收口
- `feat(...)`：新增练习、章节内容、脚本能力
- `test(...)`：新增或调整验证脚本
- `chore(...)`：环境、配置、辅助文档

## 章节维护原则

- Chapter 0：优先保证入口清晰、练习闭环完整
- Chapter 1：优先保证理论文档准确，练习资产与导学一致
- Chapter 2：优先保证题目链接、站内导学和 `.md` 页面一致
- Chapter 3：优先保证 GPU 环境说明、Triton/CUDA 路径和站内链接一致
- Chapter 2 / 3 的软件环境分层属于教程主体的一部分，环境文件、依赖拆分和章节内容要同步维护，不要把环境说明散落到单题里
- Chapter 2 / 3 的 GPU 等级标注应尽量基于 notebook 代码审计结果；新增题目时要明确它是 CPU-first、GPU-recommended 还是 GPU-required

## 分组原则

章节里的“组”不是临时容器，而是一个可持续扩展的子方向。拆组时优先看这三点：

- 是否有统一的学习目标
- 是否沿着同一条依赖链展开
- 后续能否继续补充，不需要频繁改组名或重排目录

### Chapter 2 的子方向说明

- **2.1 基础算子**：Transformer 最小构件
- **2.2 模型架构**：把算子组装成完整 block / router / model
- **2.3 微调与训练技术**：SFT、LoRA、调度器与训练流程
- **2.4 对齐技术**：RLHF、DPO 与偏好优化
- **2.5 反向传播与显存优化**：Attention backward、重计算与显存治理
- **2.6 核心推理优化**：FlashAttention、Decoding、PagedAttention
- **2.7 高级推理优化**：Speculative Decoding、RadixAttention、推理量化
- **2.8 分布式与扩展**：Checkpointing、QLoRA、ZeRO、并行扩展

### Chapter 3 的子方向说明

- **3.1 Triton 基础**：Triton 编程模型与基础 kernel
- **3.2 Triton 进阶**：融合算子、Softmax、RoPE、FlashAttention
- **3.3 Triton 项目**：调试、内存模型和综合项目
- **3.4 CUDA 内核与显存优化**：CUDA C++、Stream、Shared Memory 和 kernel 优化
- **3.5 CUDA 系统扩展**：通信原语、ZeRO 和技术选型

## 占位规范

当某个子方向的结构已经确定，但内容尚未补齐时，可以先保留入口页或占位页。占位页需要明确写出：

- 当前已有内容
- 后续准备补什么
- 现在为什么暂不展开

占位页不应伪装成“已完成”，也不应指向不存在的 notebook。

## 科学性要求

为了保证教程后续可维护、可验证、可扩展，新增内容应尽量满足：

- 知识依赖顺序清晰
- 学习单元边界明确
- 题目、概念页、项目页职责分离
- 每个组都能解释其存在理由
- 与现有验证脚本、站点导航和 notebook 入口保持一致

如果某个章节只能靠“先加几道题再说”来维持结构，说明这个分组还不够稳定，应优先回到“子方向”层面重新切分。

## 每章汇总

### Chapter 0

- 前置知识与环境准备
- 重点是 Python、NumPy、PyTorch 基础、调试和工具链
- 当前已形成练习闭环，适合新手从这里开始

### Chapter 1

- 硬件、算力推导与系统级理论
- 重点是数据类型、参数量、FLOPs、显存、通信拓扑和系统边界
- 当前理论页和练习页已对齐

### Chapter 2

- PyTorch 算法实战
- 重点是基础算子、模型组装、训练、对齐、推理优化和分布式扩展
- 站内导学已收口到 `.md` 页面

### Chapter 3

- CUDA C++ 与 Triton 算子开发
- 重点是 Triton 基础、融合算子、Attention 优化、调试、性能分析和 CUDA 实践
- 站内导学已收口到 `.md` 页面

## 使用方式

### 在线学习

- 通过 `README.md` 或 `docs/index.md` 进入章节
- 打开章节页面顶部的 Colab / ModelScope 徽章
- 直接在浏览器里运行 notebook

### 本地学习

- 使用 `environment.yml` 创建环境
- 按需安装 `requirements/gpu.txt`
- 用 Jupyter Lab 或 VSCode 打开 notebook

### 脚本验证

- 文档链接：`python check_chapter_links.py`
- Chapter 0 / 1：`python test_chapter0_1_notebooks.py`
- Chapter 2 / 3：`python test_notebook_answers.py --all --dir 02_PyTorch_Algorithms --mode both`
- Chapter 3：`python test_notebook_answers.py --all --dir 03_CUDA_and_Triton_Kernels --mode both`

### CNB 验证

CNB 的验证顺序应当和本地环境分开记录，先确认交互环境，再跑章节脚本。

1. 激活项目环境：
   - `conda activate llm_algo_cnb_dev`
2. 检查基础软件栈：
   - `python --version`
   - `python -m pip --version`
   - `python -c "import torch; print(torch.__version__)"`
   - `python -c "import triton; print(triton.__version__)"`
3. 验证 Chapter 0 / 1：
   - `python test_chapter0_1_notebooks.py`
4. 验证 Chapter 2：
   - `python test_notebook_answers.py --all --dir 02_PyTorch_Algorithms --mode both`
5. 仅在 CNB 实例具备 GPU 时验证 Chapter 3：
   - `python test_notebook_answers.py --all --dir 03_CUDA_and_Triton_Kernels --mode both`

如果当前 CNB 实例没有 `nvidia-smi` 或 `torch.cuda.is_available()` 返回 `False`，不要把 Chapter 3 的 GPU 结果写成最终验收。

### Chapter 3 GPU 入口

Chapter 3 的 GPU 验证应单独维护，不要塞进默认 CNB CPU 交互环境里。

- CPU 交互环境：验证 Chapter 0 / 1 / 2
- GPU 验证环境：验证 Chapter 3
- GPU 验证脚本：`scripts/validate_chapter3_gpu.sh`

新增 GPU 入口时，要明确标注：

- 目标节点是否真的具备 GPU
- 是否允许 `torch.cuda.is_available() == True` 作为通过条件
- Chapter 3 的结论是否可以复用到本机 50 系基线之外

### 站点验证

- `cd docs && npm run docs:build`
- 本地预览后检查章节页、侧边栏和外链是否正确

## 发布前检查清单

- [ ] `npm run docs:build` 通过
- [ ] 站内链接检查通过
- [ ] 对应章节的 notebook 测试通过
- [ ] README 和站点首页描述一致
- [ ] 导学页没有指向不存在的文件
- [ ] 新增内容没有破坏旧入口

## 维护记录建议

如果一次改动会跨多个章节，建议附带一份简短记录，说明：

- 改了什么
- 为什么改
- 影响范围
- 如何验证

这类记录可以放在 `PLAN_*.md`、`SUMMARY_*.md` 或单独的维护纪要中。
