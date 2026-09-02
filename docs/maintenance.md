# 维护与发布手册

只看三件事：怎么同步、怎么验证、脚本各管哪一层。正文模板规则见 [template_guidelines.md](./template_guidelines.md)。

## 源文件优先

- 先改源文件，再同步 `docs/` 镜像，不能直接手改 `docs/` 作为最终提交。
- 正文类改动统一从源 notebook 或源 markdown 出发，完成后再运行对应同步脚本。
- 如果镜像页和源文件出现不一致，以源文件为准，后续同步会覆盖镜像。

## 脚本分层

| 层 | 脚本 | 作用 |
|---|---|---|
| `verify` | `verify.py` | 统一验证入口 |
| `convert` | `tools/convert_notebook.py` | 正文镜像主链路 |
| `sync` | `tools/sync_docs_index.py`、`tools/sync_docs_navigation.py` | 首页 / 导学页 / 组页同步 |
| `check` | `tools/check_source_docs_mirror.py`、`tools/check_chapter_links.py` | 镜像和链接检查 |
| `test` | `tools/test_chapter0_1_notebooks.py`、`tools/test_notebook_answers.py` | Notebook 校验 |
| `audit` | `tools/audit_chapter0_1_notebooks.py` | Part 0 / Part 1 执行验证 + 结构审计 |
| `migration` | `tools/md_to_notebook.py` | markdown -> notebook 迁移辅助 |

`tools/convert_chapter0_1.py` 只保留 legacy 兼容。

## Part 0-4 维护分工

- `Part 00` 和 `Part 01` 一起作为前置知识层，重点是基础语言、张量、系统视角和性能边界。
- `Part 02` 是主干实现层，重点是 PyTorch 里的训练、推理、并行、量化和项目收口。
- `Part 02` 的项目建设按“核心项目 + 扩展项目 + 延伸方向”组织：`2.9` 是项目收口层，核心项目优先覆盖训练落地、推理选型和训练分析，扩展项目优先覆盖 profiling 闭环、并行基准和量化部署；`36-42` 则作为更细的延伸方向，继续补推理服务、cache、量化家族和通信 profiling。项目页的 TODO 仍保持 notebook-first 的统一结构，但职责从“补算法”转为“组织实验、输出对比和沉淀结论”。
- `Part 03` 是 Triton / kernel 过渡层，重点是把框架级实现继续下沉到高性能算子。
- `Part 04` 是 CUDA / 系统优化层，重点是继续向硬件、通信、调度和架构收口。
- 维护时可以按下面的验证分段理解：
  - `verify.py part0_1`：检查 `Part 00 / Part 01`
  - `verify.py part2`：检查 `Part 02`
  - `verify.py part3`：检查 `Part 03`
  - `verify.py part4`：检查 `Part 04`
- 横向专题主要横切 `Part 00 / Part 01 / Part 02`，后续若继续下探性能和实现，可以逐步接到 `Part 03 / Part 04`。

## 依赖 profile 维护规则

- `requirements/base.txt` 只放跨平台通用包，不放 `torch`、GPU 驱动或 vLLM。
- `requirements/torch-cpu.txt` 与 `requirements/torch-cu128.txt` 是互斥的 PyTorch 平台 profile；默认根依赖 `requirements.txt` 不自动选择 GPU。Conda environment 文件也不直接合并这两个 profile，避免 PyTorch 专用 index 覆盖通用包的下载源。
- `fine-tuning.txt`、`qlora.txt`、`reinforcement-learning.txt`、`distributed.txt`、`inference-vllm.txt`、`inference-sglang.txt` 和 `profiling.txt` 是能力层 profile，依赖 base，但不负责判断主机是否有 GPU。
- GPU 环境说明必须同时区分驱动、PyTorch CUDA wheel 和上层能力包；不能把 `nvidia-smi` 的 CUDA 字段写成 PyTorch 的 CUDA 版本。
- Colab、ModelScope 和云端预装环境优先复用当前 Kernel 的 PyTorch。Notebook 自动化可以补普通 Python 包，但不能静默替换 CUDA PyTorch、驱动、vLLM 或 SGLang。
- 新增 backend（如 TensorRT-LLM、Unsloth、LLaMA-Factory）前，先建立独立 profile 或官方版本矩阵，并说明与已有 profile 的兼容边界。SGLang 使用 `requirements/inference-sglang.txt`，与 vLLM profile 分开维护。
- 后训练项目使用 `requirements/reinforcement-learning.txt`；分布式项目使用 `requirements/distributed.txt`，二者都必须与一个明确的 CPU 或 CUDA PyTorch profile 组合，不能把 GPU、RL 和分布式依赖默认塞进 `requirements.txt`。

## 教程信息架构口径

后续写导学页、组导航页、专题页和 `Part 01` 正文导读时，统一使用下面四类概念，不再混写：

- `纵向主线`
  - 指教程按能力递进展开的默认学习顺序
  - 固定理解为：`Part 00 -> Part 01 -> Part 02 -> Part 03 -> Part 04`
  - 作用是回答“整个教程先学什么，后学什么”
- `学习路线`
  - 指面向任务目标的跨 Part 阅读路径
  - 当前固定为四条主路线：`推理优化路线`、`显存优化路线`、`算子优化路线`、`后训练优化路线`
  - 作用是回答“为了做成什么任务，应该重点串哪些页”
- `横向专题`
  - 指跨多个 Part 反复出现的方法轴或技术轴
  - 当前固定包括：`量化与压缩`、`性能分析`、`通信与并行`
  - 作用是回答“同一类技术问题分散在不同 Part 时，应该怎么按主题重组来看”
- `基础支撑专题`
  - 指被多条学习路线共同依赖的底层知识底座
  - 当前包括：`大模型架构`、`反向传播与训练机制`、`监督微调与训练工程`、`编译与图优化`
  - 作用是回答“哪些基础认知会在多条路线里被反复依赖”

补充说明：`基础支撑专题`、`横向专题`和`学习路线`是不同维度。一个专题可以作为某条路线的前置或支撑，但不能因此把它改写成该路线；一个项目也只能保留一个主叙事入口，其他路线通过关联链接复用机制和证据。

边界约束：

- 不把技术领域直接写成 `学习路线`
- 不把 `基础支撑专题` 混写成 `横向专题`
- 不把组导航页或专题页写成目录复述
- `学习路线` 强调目标和顺序，`横向专题` 强调主题和重组，`基础支撑专题` 强调底座和共性依赖

## 受众与写作口径

教程主要面向已经具备 Python、NumPy、PyTorch 基础操作和基本模型运行能力，希望理解大模型训练、推理与系统优化，并能完成可复现实验的学习者。默认不要求学习者一开始掌握 CUDA、Triton、vLLM、SGLang 或分布式系统；这些内容应在需要时逐层引入。

受众分为三层，但页面写作以主受众为准：

| 受众层 | 基础情况 | 页面应提供的内容 |
|---|---|---|
| 主受众 | 会 Python / PyTorch，能运行 Notebook，但不熟悉 LLM Infra | 机制解释、可运行代码、参数含义、指标、实验设计和结论边界 |
| 次受众 | 有训练或推理经验，希望补齐性能、显存和部署知识 | 代价模型、证据链、backend 约束和项目决策 |
| 扩展受众 | 熟悉 CUDA、Triton、vLLM 或分布式系统 | kernel、backend、多卡和系统级扩展入口 |

统一写作口吻为“工程教学口吻”：先说明问题，再解释机制，用代码或表格验证，最后说明结果边界和下一步。不要把页面写成零基础宣传，也不要默认读者已经是 CUDA 或 Infra 专家。

- `Notebook`：面向动手运行代码的学习者。任务、参数、TODO、docstring 和预期现象必须具体，并说明 CPU 与 GPU 各自能验证什么。
- `intro`：面向选择学习路径的学习者。说明专题解决的问题、进入条件、Task 顺序和与其他专题的边界，不展开过多公式或 API。
- `casebook`：面向需要判断方案的学习者。按现象、机制、证据和决策组织，不重复 Notebook 的代码教学。
- `walkthrough`：面向希望连续理解路线的学习者。解释因果关系和下一步为什么出现，避免口号式和营销式表达。

每个机制说明尽量遵循：

```text
问题现象 → 机制解释 → 代码或表格验证 → 证据边界 → 下一步
```

以下表达默认不使用：`这一页你会带走……`、`只要记住……`、`非常简单……`、`彻底掌握……`。应改为可检验的描述，例如“本节比较两种策略的作用对象，并说明 CPU 示例和真实 GPU 实验分别能够确认什么”。

量化与压缩专题的默认入口是“模型太大、显存不足或质量需要保持”，性能分析专题的默认入口是“运行变慢、指标异常或优化效果无法证明”。两者都面向同一主受众，但不要求学习者按完整路线顺读；应允许根据问题切入，再通过正文链接补齐前置机制。

## Part 01 导读写法口径

`Part 01` 的 `本节导读` 第二段优先按下面顺序写：

1. 这一节在 `纵向主线` 里属于哪一类基础页。
2. 它优先服务哪条 `学习路线`。
3. 学完这里后面更顺地进入哪些具体小节、项目页或判断任务。
4. 如果这里没学明白，后面通常会卡在哪些实现、判断或项目验证上。
5. 它同时归属于哪个 `横向专题` 或 `基础支撑专题`。

推荐模板：

```text
这一节在整个教程的纵向主线里属于 Part01 的基础页，主要为「某条学习路线」提供前置支撑。学完这里，后面可以更顺地进入「A / B / C 小节或项目页」；如果这里没学明白，通常会卡在「哪些判断、哪些实现或哪些项目验证」上。按专题归类，它同时属于「某个横向专题」或「某个基础支撑专题」。
```

补充约束：

- 优先写具体后续页，不只写“服务训练”或“服务推理”这种泛表述。
- 如果一页同时服务多条路线，只写主服务路线，再补一句次要关联，不要把三条路线全部堆上去。
- 如果一页更像底层机制页、很难强挂主路线，可以写“当前主要作为 Part01 基础支撑页”，再补具体后续去向。

## 专题名称与目录路径规则

专题名称和目录路径分开维护：名称面向学习者，目录路径面向仓库和脚本。页面标题、README、导航和正文引用必须使用统一的中文名称与英文标准名；已有目录路径先作为稳定 slug 保留，不因改名直接迁移目录。

| 中文标准名 | 英文标准名 | 当前目录 slug |
|---|---|---|
| 大模型架构 | Model Architecture | `model_architecture` |
| 反向传播与训练机制 | Backpropagation and Training Mechanics | `backpropagation_training_mechanism` |
| 监督微调与训练工程 | Supervised Fine-Tuning and Training Engineering | `fine_tuning_training` |
| 编译与图优化 | Compiler and Graph Optimization | `compiler_graph_optimization` |
| 量化与压缩 | Quantization and Compression | `quantization` |
| 性能分析 | Performance Analysis | `profiling` |
| 通信与并行 | Communication and Parallelism | `communication_parallel` |
| 推理优化 | Inference Optimization | `inference_optimization` |
| 显存优化 | Memory Optimization | `memory_performance_tuning` |
| 算子优化 | Operator Optimization | `operator_optimization` |
| 后训练优化 | Post-Training Optimization | `post_training_alignment` |

命名执行规则：

- 中文是页面主名称，英文作为括号副标题；不要在同一页面交替使用多个英文译名。
- 首次出现使用“中文名称（English Standard Name）”，后续默认使用中文简称。
- `Profiling` 作为工具或技术术语保留；专题名称统一写作“性能分析（Performance Analysis）”。
- “后训练与对齐”统一改为“后训练优化（Post-Training Optimization）”；DPO、GRPO 是其中的方法，不作为专题总名。
- “算子与编译优化”不作为路线名称；路线统一为“算子优化”，编译与图优化作为相邻的系统桥接专题。
- 官方工具和算法名称保留原文，例如 `vLLM`、`SGLang`、`LoRA`、`GPTQ`、`AWQ`、`FlashAttention`。
- 新专题目录优先使用英文标准名对应的简短 slug；已有目录不直接重命名。

如果以后确需迁移目录，必须单独建立迁移任务：先统计源文件、镜像、导航和脚本中的引用，完成批量改链和验证后，再执行目录移动；本规则不允许通过手改 `docs/` 镜像绕过源文件迁移。

## 专题类型与主服务目标标签规则

每个横向专题 `intro.md` 的标题下方固定标注两个标签：`专题类型` 和 `主服务目标`。标签用于说明页面在整体知识架构中的职责，不表示 Infra 层级、Practice 级别或学习难度。

专题类型只使用以下三种标准值：

- `主学习路线`：有连续 Task 主线，并承担一组项目或系统能力的主要学习入口。
- `横切支撑`：跨多条路线复用方法、工具或资源决策，不承担单一路线的完整学习闭环。
- `基础支撑`：解释机制、模型结构或训练基础，作为多个主路线的前置桥接。

主服务目标使用简短的“对象 + 决策”表达，说明专题主要帮助学习者判断什么；不要写成口号，也不要把所有关联路线堆在标签中。当前标准如下：

| 专题 | 专题类型 | 主服务目标 |
|---|---|---|
| 推理优化 | 主学习路线 | 请求性能与 Serving 决策 |
| 显存优化 | 主学习路线 | 显存预算与资源取舍 |
| 算子优化 | 主学习路线 | Kernel 性能与端到端收益 |
| 后训练优化 | 主学习路线 | 偏好对齐与训练交付 |
| 量化与压缩 | 横切支撑 | 精度、显存与部署取舍 |
| 性能分析 | 横切支撑 | 瓶颈定位与证据归因 |
| 通信与并行 | 横切支撑 | 多卡切分与通信取舍 |
| 大模型架构 | 基础支撑 | 结构理解与资源映射 |
| 反向传播与训练机制 | 基础支撑 | 计算图与训练状态理解 |
| 监督微调与训练工程 | 基础支撑 | SFT 训练与工程复现 |
| 编译与图优化 | 基础支撑 | 图变换与执行计划理解 |

标签应放在标题之后、页面导语或专题定位之前，格式保持一致：

```text
> 专题类型：主学习路线　主服务目标：显存预算与资源取舍
```

如果专题职责发生变化，先修改本表和对应 `intro.md`，再同步 README 与 `docs/` 镜像。

## Notebook 与专题的引用规则

引用规范属于维护规则，不在每个 Notebook 或横向专题中重复解释。具体页面只保留实际需要的跳转链接和简短的内容说明。

- 路线、专题、Notebook、项目、Part 导学、资产表和使用指南：首次出现且学习者可能需要继续阅读或运行时，使用可点击链接，并说明跳转目的。
- 前置阅读：只列当前页面的直接依赖；每个入口说明它补充的机制或技能，不把所有相关内容都列入前置。
- 相关阅读：列出完成当前页面后的下一步，区分扩展阅读、项目实践和结果验证，不把相关阅读写成硬性前置。
- 概念、指标和配置字段：默认使用普通文字；只有在对应专门页面且需要继续阅读时才加链接。
- 同一页面在同一文档中再次出现：可只写节次或简称，避免重复链接。
- 节次来源统一写为 `Part 01 · 13`、`Part 02 · 73`；不要使用 `P1` 表示 Part，以免和 `Practice-P1` 混淆。
- 一个项目只保留一个主叙事入口；其他专题只能通过关联链接复用机制、指标或证据，不重复定义项目目标和最终结论。
- 链接文字应包含节次或资源名称，避免只写“点击这里”；相对路径必须从当前源文件位置计算，并在提交前执行链接检查。

73 节可作为项目页示例：导读说明项目分工，前置阅读链接直接依赖，相关阅读按后续实验顺序链接 76 → 75 → 74；正文重复引用可使用普通节次。该规则同样适用于 60–71、73–76、79–81 等项目节。

## 日常流程

1. 先改 source。
2. 首页改动后跑 `python tools/sync_docs_index.py`。
3. 导学页 / 组页改动后跑 `python tools/sync_docs_navigation.py`。
4. 正文改动后跑 `python tools/convert_notebook.py`。
5. 最后跑 `cd docs && npm run docs:build`。

## 单个项目页的环境预检规则

项目 Notebook 必须允许学习者只运行当前一节，不应默认要求先创建整套训练、推理和服务环境。每个项目页的环境入口按以下顺序工作：定位仓库 → 检查 Python / PyTorch → 检查 CUDA 与显存能力 → 检查本节必需包 → 检查模型或 backend 能力 → 检查结果目录 → 才加载模型或启动服务。

统一使用 `tools.project_runtime` 中的 `bootstrap_project_root` 和 `environment_preflight`。预检输出固定为 JSON 可读的三种状态：

- `ok`：当前配置可以进入实验；
- `warning`：CPU 或降级路径可以继续，但可选能力不可用；
- `blocked`：当前请求的真实 GPU / backend 实验不能继续，并给出下一步动作。

预检只检查，不静默重装环境。普通 Python 包可以由当前内核的 `sys.executable` 显式安装；PyTorch、vLLM、SGLang、TensorRT-LLM 等 CUDA 相关组件必须由学习者选择版本并重启运行时，避免把云端 CUDA 环境替换成 CPU wheel。BF16 必须区分“可以分配 BF16 张量”和“确认原生 BF16 加速”；T4 等设备即使 `torch.cuda.is_bf16_supported()` 返回真，也不能据此宣称有硬件加速。

本地或云端可先运行：

```bash
python tools/environment_preflight.py --gpu --packages transformers --output benchmarks/results/preflight.json
```

推理 backend 项目再把 `vllm` 或 `sglang` 列为必需包；CPU-first 项目不要求 CUDA。预检通过后才执行模型下载和实验，结果 JSON 必须同时保存 `runtime`、配置、状态和失败原因。Colab / ModelScope 的推荐流程是：打开 GPU → clone 源仓库 → 运行预检 → 按预检提示安装缺少的包 → 重启内核 → 再运行 Notebook；不要从 `docs/` 镜像反向修改代码。

需要额外环境的项目页，在真实实验开关附近放一条短提示，不在每节重复完整安装命令：

```text
运行提示：先查看《使用指南》的“项目环境预检与安装”部分，再打开本节的真实实验开关。CPU-first 路径不要求 GPU；真实 GPU 或 backend 路径必须先通过预检。
```

源 Notebook 中链接到 `../docs/guide.md#项目环境预检与安装`；转换后的 `docs/` 页面使用对应的 `../../docs/guide.md#项目环境预检与安装`。项目页的提示只说明入口和运行边界，具体依赖、安装顺序和云端重启规则统一维护在《使用指南》中。

## 图片资产规则

图片资产后续统一按“先分级、再入正文”的原则处理，不再边写正文边临时插图。

- `Part02` 的正文图先看 [part02_visual_assets_audit.md](./part02_visual_assets_audit.md)。
- `topic_discussion` 的专题图先看 [topic_discussion_visual_assets_audit.md](./topic_discussion_visual_assets_audit.md)。

固定规则：

- 未经过审核的图片，不进入正文主叙事位置。
- 未审核图如果必须先保留，只能放在 `visual_assets` 页或附录型页面，不直接承担正文主解释职责。
- 图片审核至少要回答三件事：
  - 这张图是不是核心教学图、路线收束图，还是结构占位图
  - 这张图应该说明什么
  - 这张图不应该说明什么
- 图片进入正文前，至少完成：
  - 职责分级
  - 可读性初审
  - 是否需要减字 / 中文化 / 重画的判断

## Part02 图解格式收口

`Part02` 当前更大的问题不是某一张图本身，而是正文里长期混用了三种表达：

- 正式 `SVG`
- notebook 内的 `ASCII / text block`
- 尚未稳定模板化的 `Mermaid`

后续统一按下面的职责边界执行：

- `SVG`
  - 这是 `Part02` 正文正式主图的唯一默认格式
  - 只要一张图承担核心机制解释，就应进入 `SVG` 体系并先审计
- `ASCII / text block`
  - 只保留为局部辅助结构
  - 适合维度流向、短流程、图前骨架提示
  - 不再承担正文唯一主图职责
- `Mermaid`
  - 当前先冻结为“非默认正文主图格式”
  - 在没有统一模板、职责边界和维护结论前，不继续向 `Part02` 正文扩散

执行顺序固定为：

1. 先清职责边界，不先扩写新图。
2. 先盘点 `ASCII / text block`，看哪些只是辅助，哪些已经和 `SVG` 重复。
3. 先复核高价值 `SVG` 的可读性和信息密度。
4. 页面职责稳定后，再决定是否中文化或重画。

当前结论：

- `Part02` 这轮优先级是“统一图解体系”，不是“先把所有图翻成中文”。
- 没审过的图，不入正文；没定职责的格式，也不继续扩散进正文。

## 常用命令

```bash
python verify.py part0_1 --no-build
python verify.py part0_1_audit
python verify.py part2 --no-build
python verify.py part3 --no-build
python verify.py part4 --no-build
python verify.py all --no-build
python tools/sync_docs_index.py
python tools/sync_docs_navigation.py
python tools/convert_notebook.py
cd docs && npm run docs:build
```

## 测试脚本索引

| 层 | 脚本 | 作用 |
|---|---|---|
| `verify` | `verify.py` | 统一验证入口 |
| `convert` | `tools/convert_notebook.py` | 正文镜像主链路 |
| `sync` | `tools/sync_docs_index.py`、`tools/sync_docs_navigation.py` | 首页 / 导学页 / 组页同步 |
| `check` | `tools/check_source_docs_mirror.py`、`tools/check_chapter_links.py` | 镜像和链接检查 |
| `check` | `tools/check_docs_links.py`、`tools/check_math_formula_symbols.py`、`tools/check_part01_code_blocks.py` | docs 链接、公式与代码块检查 |
| `test` | `tools/test_chapter0_1_notebooks.py`、`tools/test_notebook_answers.py` | Notebook 校验 |
| `migration` | `tools/md_to_notebook.py` | markdown -> notebook 迁移辅助 |

## 验证模型对照

`Part 0 / Part 1` 和 `Part 2 / Part 3` 的 notebook 结构不同，不能用同一套脚本假设去验证。

| 范围 | 主脚本 | 默认验证逻辑 | 适用结构 |
|---|---|---|---|
| `Part 0 / Part 1` | `tools/test_chapter0_1_notebooks.py` | 顺序执行每个非空 `code cell`，只要执行过程中不抛异常就算通过 | 讲解型 / 逐代码块验证型 notebook |
| `Part 0 / Part 1` | `tools/audit_chapter0_1_notebooks.py` | 分开看 `codecell_run` 和 `structure_only`：前者检查代码块执行，后者检查 `本节导读 / 前置阅读 / 相关阅读`、链接数量、`cell id` | 验证收尾、结构审计、warning 归档 |
| `Part 2 / Part 3` | `tools/test_notebook_answers.py` | 按 `import -> 题目区 -> STOP HERE -> 参考代码与解析` 抽取代码，再分别验证题目区 / 答案区 | 练习型 / 题目区答案区双结构 notebook |

结论上要区分两类“通过”：

- `Part 0 / Part 1` 的“代码通过”主要表示：代码块能顺序运行且不报错。
- `Part 2 / Part 3` 的“代码通过”主要表示：题目区或答案区在既定抽取规则下能被正确提取并验证。
- 因此，`Part 0 / Part 1` 不应用 `Part 2` 的题目区 / 答案区脚本强套；如果需要收尾审计，应优先用 `tools/audit_chapter0_1_notebooks.py`。

## Part 0 / Part 1 固定口径

`Part 0 / Part 1` 后续统一按下面这套入口理解，不再临时拼脚本：

- 主入口：`python verify.py part0_1 --no-build`
  - 这是 `Part 0 / Part 1` 的标准验证命令
  - 职责是：镜像转换、source/docs 链接检查、逐 `code cell` 执行验证
- 补充审计：`python verify.py part0_1_audit`
  - 这是收尾审计命令，不替代主入口
  - 职责是把执行验证和结构检查拆开归档

`part0_1_audit` 下的两个 profile 固定解释为：

- `codecell_run`
  - 顺序执行 notebook 中每个非空 `code cell`
  - 目标是确认讲解型 notebook 的代码块在当前环境下能否连续运行且不报错
  - 适合定位“哪一页 / 哪一个 cell 执行失败”
- `structure_only`
  - 不执行代码
  - 只检查 `本节导读 / 前置阅读 / 相关阅读`、链接数量、`cell id`、基础 notebook 结构
  - 适合做页头收尾、warning 分类和结构回归检查

推荐理解方式：

- 日常验证只跑 `verify.py part0_1`
- 需要验证收尾、warning 归档、结构追责时，再补跑 `verify.py part0_1_audit`
- 不再直接用 `Part 2` 的 `tools/test_notebook_answers.py` 套 `Part 0 / Part 1`

## 推荐用法

```bash
python verify.py part0_1 --no-build
python verify.py part2 --no-build
python verify.py part3 --no-build
python verify.py all --no-build
python tools/check_math_formula_symbols.py
python tools/check_part01_code_blocks.py
python tools/audit_chapter0_1_notebooks.py --profile all
```

无 GPU 时，`verify.py` 会跳过 Part 2 / 3 的 GPU-only 答案验证，但仍保留转换、镜像和链接检查。单独排查时直接用底层脚本。
`tools/md_to_notebook.py` 仅用于历史迁移，不进入日常主流程。

## 说明

- `Part 0 / Part 1` 用 `tools/test_chapter0_1_notebooks.py`
- `Part 0 / Part 1` 如需把代码执行和结构/warning 分开看，用 `tools/audit_chapter0_1_notebooks.py`
- `Part 2 / Part 3` 用 `tools/test_notebook_answers.py`
- 先改源，再同步 `docs/`
- 导学页、组页、正文页分开同步
- `tools/convert_chapter0_1.py` 只保留兼容用途
- `tools/md_to_notebook.py` 只保留迁移辅助用途
