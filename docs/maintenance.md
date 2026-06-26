# 维护与发布手册

本页用于记录项目维护者和贡献者的日常工作约定，目标是把仓库资产、部分汇总、使用方式、验证方式和发布流程放在同一处，避免维护信息分散。

## 仓库资产总览

### 主要入口

- `README.md`：项目总入口，给首次进入仓库的人看
- `docs/index.md`：站点首页，给浏览文档站的人看
- `docs/guide.md`：使用指南，说明如何选择环境和学习方式
- `docs/contributing.md`：贡献指南，说明如何参与开发和测试
- `docs/maintenance.md`：维护与发布手册，说明仓库如何维护
- `docs/template_guidelines.md`：题目模板与网页化转换规范，给新增或改写 Notebook 题目时参考

### 部分内容

- `docs/00_Prerequisites/`：Part 0，前置知识与环境准备
- `docs/01_Hardware_Math_and_Systems/`：Part 1，硬件、算力推导与系统级理论
- `docs/02_PyTorch_Algorithms/`：Part 2，PyTorch 算法实战
- `docs/03_Triton_Kernels/`：Part 3，Triton 算子开发
- `docs/04_CUDA_and_System_Optimization/`：Part 4，CUDA C++ 与系统优化

### 练习与验证

- `test_notebook_answers.py`：Part 2 / 3 / 4 的通用 Notebook 验证脚本
- `test_chapter0_1_notebooks.py`：Part 0 / 1 的顺序执行验证脚本
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
- `requirements/gpu.txt`：Part 3 / Part 4 / Triton / CUDA 扩展依赖版本的主来源
- `environment.yml`：只负责 Python 版本和串联依赖，不应重复维护另一套独立版本表
- `cnb/environment.yml`：CNB 侧环境骨架，原则上复用同一套版本约定

更新依赖时，优先修改 `requirements/*.txt`，再同步 `environment.yml` 和 `cnb/environment.yml`，避免本地、CNB、Docker 出现版本漂移。

## 源文件与生成物

仓库里的部分内容遵循“源文件在根目录，网页产物在 `docs/`”的原则。

### 源文件

- 根目录下的 `00_Prerequisites/`
- 根目录下的 `01_Hardware_Math_and_Systems/`
- 根目录下的 `02_PyTorch_Algorithms/`
- 根目录下的 `03_Triton_Kernels/`
- 根目录下的部分 `intro.md`
- 根目录下的部分练习 `*.ipynb`
- 根目录下的部分理论 `*.md`

### 生成物

- `docs/00_Prerequisites/`
- `docs/01_Hardware_Math_and_Systems/`
- `docs/02_PyTorch_Algorithms/`
- `docs/03_Triton_Kernels/`

这些目录中的部分页面通常由 `convert_notebook.py` 从根目录源文件同步生成。维护时应优先修改根目录源文件，再重新生成 `docs/` 页面。

`convert_notebook.py` 现在支持两种运行方式：

- 全量模式：不带参数运行，重建 Part 2 / 3 / 4 的整章 `docs/` 镜像
- 局部模式：通过 `--dir` 或 `--file` 只同步指定目录或文件，适合只修某几个 notebook 时使用

局部模式示例：

```bash
python convert_notebook.py --dir 03_Triton_Kernels
python convert_notebook.py --dir 04_CUDA_and_System_Optimization
python convert_notebook.py --file 03_Triton_Kernels/05_Triton_Autotune_and_Profiling.ipynb
python convert_notebook.py --dry-run --dir 04_CUDA_and_System_Optimization
```

### 分组页维护

group md 不是每个 part 的硬性正文资产，而是“需要分组导航时才维护”的入口层文件。

- 如果一个 part 有稳定的分组结构，应在 source 里维护 group md，让脚本同步到 `docs/`
- 如果一个 part 只是单导学或没有稳定分组，不必强行补 group md
- group md 的职责是导航和组织，不承载正文推导

当前原则：

- source 里保留真正的内容源文件
- `docs/` 里保留脚本生成的镜像页
- group md 只在确实需要“组内导航”时出现，避免把结构文件和正文文件混在一起

### 手工维护文件

以下文件不属于自动生成物，应该直接手工维护：

- `README.md`
- `docs/index.md`
- `docs/guide.md`
- `docs/contributing.md`
- `docs/maintenance.md`
- `docs/template_guidelines.md`
- `docs/.vitepress/config.mts`

## 维护目标

- 保持部分内容、站点导航、Notebook 入口和测试脚本同步
- 控制改动范围，避免跨部分的大面积联动回归
- 将可验证的变更拆成小提交，便于 review 和回滚
- 让维护工作可以复用，不依赖个人记忆

## 本地 GPU 验证基线

- 当前已验证的本地 GPU 基线是 **Linux 22.04 + 50 系 NVIDIA 显卡 + `llm_algo` conda 环境**
- Part 2 在该环境下的答案区已验证通过
- Part 3 在该环境下的答案区已验证通过
- 40 系显卡暂未作为已验证基线，后续应单独补充兼容矩阵与版本验证

## 过程文件约定

仓库根目录下的 `process/` 目录专门存放维护过程中的临时记录和交接材料，例如：

- `SESSION_HANDOFF.md`
- `PLAN_*.md`
- `SUMMARY_*.md`

这类文件只用于维护协作，不进入站点导航，不作为学习内容展示，不参与部分侧边栏配置。

当一轮维护结束后，这些文件可以按需要：

- 保留在 `process/` 中作为历史记录
- 或者统一清理删除

不要把它们散落在仓库根目录，避免和正式文档入口混在一起。

## 当前维护对象

- `README.md`、`docs/index.md`、`docs/guide.md`、`docs/.vitepress/config.mts`：入口与导航
- `00_Prerequisites/intro.md`、`01_Hardware_Math_and_Systems/intro.md`、`02_PyTorch_Algorithms/intro.md`、`03_Triton_Kernels/intro.md`、`04_CUDA_and_System_Optimization/intro.md`：各部分导学
- `docs/` 下对应页面：站点镜像
- `convert_chapter0_1.py`、`convert_notebook.py`：source -> docs 转换
- `test_chapter0_1_notebooks.py`、`test_notebook_answers.py`：Notebook 验证
- `check_chapter_links.py`、`check_source_docs_mirror.py`：链接和镜像一致性检查

## 文档职责图

为了避免把环境说明、维护规则和 CNB 入口写混，当前分工只保留一句话版本：

- `docs/guide.md`：怎么学、怎么选环境、怎么验证
- `cnb/README.md`：CNB 入口和交互环境说明
- `docs/maintenance.md`：维护、发布和验证规则
- `docs/template_guidelines.md`：Notebook 题目模板和网页化转换规范
- `README.md` / `docs/index.md`：项目门面和站点导航
- `process/*.md`：临时记录、交接、阶段总结

## 风格约定

为了统一教程正文和站点门面的视觉风格，当前采用下面的约定：

- `README.md` 和 `docs/index.md` 可以保留较多品牌型 emoji，用于首页和导览层级
- 各部分 `intro.md` 可以保留少量章节标记 emoji，例如 `🎯`，用于章节封面
- 教程正文 notebook 应保持克制：
  - 保留测试状态标记 `⏭️` / `✅` / `❌`
  - 保留结构标记 `🛑 **STOP HERE** 🛑`
  - 尽量删除装饰性 emoji，例如 `🚀` / `✨` / `💡`
- `🚀 云端运行环境` 这类模板标题默认视为入口提示，不作为全仓风格迁移目标
- Part 3 的 07-11 小节应遵守更严格的正文风格，避免把测试状态和装饰性 emoji 混用

如果后续需要做全仓风格收口，优先只调整正文 notebook；首页、导语页和模板入口单独评估，不要默认合并进同一轮改动。

## 常规维护流程

### 1. 先定边界

确认这次修改属于哪一类：

- 文档收口
- 部分内容修复
- Notebook 练习补齐
- 站点导航调整
- 环境文件更新
- 测试脚本调整

如果一项改动会同时触发多类变更，优先拆分成多个 commit。

### 2. 先改入口，再改内容

如果涉及部分结构变化，先同步：

- `README.md`
- `docs/index.md`
- `docs/.vitepress/config.mts`
- 部分 `intro.md`

然后再迁移具体文件和链接。

### 3. 先做本地验证

建议按改动类型选择测试，优先用统一入口：

- 统一入口：`python verify.py chapter0_1` / `python verify.py chapter2` / `python verify.py chapter3` / `python verify.py all`
- Notebook 结构规范化：`python tools/normalize_notebooks.py`，用于补齐缺失的 cell id，消除 `nbformat MissingIDFieldWarning`
- 文档链接：`python check_chapter_links.py`
- 站点构建：`cd docs && npm run docs:build`
- Part 0 / 1 练习：`python test_chapter0_1_notebooks.py`
- Part 2 / 3 / 4 题目答案：`python test_notebook_answers.py --all --dir 02_PyTorch_Algorithms --mode both`
- Part 3 内核题：`python test_notebook_answers.py --all --dir 03_Triton_Kernels --mode both`
- Part 4 内核题：`python test_notebook_answers.py --all --dir 04_CUDA_and_System_Optimization --mode both`
- Part 3 / Part 4 入口页：检查对应 `intro.md` 与 `docs/` 镜像页的链接可用性

### 4. 再做提交

推荐按功能拆 commit：

- `docs(...)`：导学、首页、站点导航、链接收口
- `feat(...)`：新增练习、部分内容、脚本能力
- `test(...)`：新增或调整验证脚本
- `chore(...)`：环境、配置、辅助文档

## 部分维护原则

- Part 0：优先保证入口清晰、练习闭环完整
- Part 1：优先保证理论文档准确，练习资产与导学一致
- Part 2：主定位是算法验证层，优先保证题目链接、站内导学、参考答案和 `.md` 页面一致
- Part 3：主定位是 Triton 实现层，优先保证 GPU 环境说明、导学页跳转和站内链接一致
- Part 4：主定位是 CUDA / 系统优化层，优先保证 kernel / module 链路闭合、导学页跳转和站内链接一致
- Part 2 / 3 / 4 的软件环境分层属于教程主体的一部分，环境文件、依赖拆分和部分内容要同步维护
- Part 2 / 3 / 4 的 GPU 等级标注应尽量基于 notebook 代码审计结果；新增题目时要明确它是 CPU-first、GPU-recommended 还是 GPU-required
- Part 2 / 3 / 4 的测试脚本若在当前环境没有 GPU，应输出 `skip` 而不是 `pass`；结构检查只说明骨架可读，不代表运行级验证完成
- Part 2 / 3 / 4 后续维护要统一到同一套“页面契约”：角色、GPU 策略、测试语义、题目区边界都要先写清楚；但历史页面不强制一次性重写成同一种哨兵写法，尽量保持现有内容少动
- 新增或重做的页面应优先采用统一契约，测试脚本只认 `pass` / `skip` / `expected_fail` / `fail` 这些结果语义，不猜页面作者意图
- Part 2 后续维护以补桥、收紧少量提示、补少量边界测试为主，避免大范围重写；Part 3 当前优先推进 Triton 主线的收口和测试一致性，Part 4 则优先推进 CUDA / 系统骨架搭建

## Part 2 / 3 / 4 分工约定

- Part 2 以算法验证为主：重点检查实现是否正确、参考答案是否一致、边界 case 是否覆盖、测试是否足够强
- Part 3 以 Triton 实现为主：重点检查 GPU 路径是否完整、kernel / module 链路是否闭合、导学页和镜像页的相对路径是否正确
- Part 4 以 CUDA / 系统优化为主：重点检查 GPU 路径、通信与系统页面的链接边界
- Part 0 / 1 可以提及 CUDA / Triton 作为后续预告，但不承担 Part 3 / Part 4 的工程执行责任
- Part 2 中允许存在 GPU 特例页，例如单独的 GPU-required 小节；此类页面应单独标注并按特例验证，不应推导出整章默认规则
- 无 GPU 时，Part 2 / 3 / 4 的 GPU 依赖页应视为 `skip`；题目区的 `NotImplementedError` 仍然表示练习态正确，不应被结构检查覆盖为“运行通过”
- 长期统一的目标是“契约统一”，不是“旧页面写法统一”；后续维护优先收敛规则、角色和测试语义，旧页面只做必要修补

## Part 1 桥接约定

- Part 1 是 Part 2、Part 3 和 Part 4 的共同前置，不只是 Part 3 的前置
- 如果一个概念同时服务 Part 2 和 Part 3 / Part 4，优先在 Part 1 补强，而不是在后续章节重复堆叠
- Part 1 的职责是建立硬件、系统、内存、调度和编程模型的认知桥梁，不是提前承担 Part 3 / Part 4 的代码实现责任
- Part 1 的内容组织应优先形成一条可执行的前导导读链，明确告诉读者先看什么、解决什么问题、再进入哪一部分
- 如果某个概念只服务少数 Part 3 / Part 4 页面，则保留为局部桥接或支撑小节，不必上移到 Part 1

### Part 3 / Part 4 入口页补充约束

- Part 3 / Part 4 的 `intro.md` 都属于高风险入口页，既要检查源文件，也要检查 `docs/` 镜像页
- 入口页中的 Task 链接、前置页、后续页和环境说明必须在镜像环境里保持可用
- 任何涉及 Part 3 / Part 4 导学页结构或路径的改动，除了常规 notebook 验证外，还应补做一次链接可用性检查

### 小节链接入口规范

- 新增或重写的小节，优先采用统一的链接块顺序：`导语` -> `前置` -> `相关阅读`
- `导语` 用一两句话说明本节解决什么问题
- `前置` 只放必须先懂的内部章节、导学页或同部分页面，控制在 2 到 4 个；判断标准是“跳过它会不会卡住”
- `相关阅读` 只放可选补充、同主题扩展或外部 URL；判断标准是“跳过它也能继续，但看了更完整”
- 正文中尽量不要散插大量链接；如果必须引用，优先先把入口收敛到这三块
- Part 1 的组页可以继续保留现有的 `前置关系 / 桥接 / 扩展` 结构；如果后续重写入口页，也应尽量迁移到同一套口径
- Part 3 / Part 4 的 notebook 导入页、桥接页和总结页应优先使用这套口径，保证源文件和 `docs/` 镜像页一致可读

## 分组原则

部分里的“组”不是临时容器，而是一个可持续扩展的子方向。拆组时优先看这三点：

- 是否有统一的学习目标
- 是否沿着同一条依赖链展开
- 后续能否继续补充，不需要频繁改组名或重排目录

### Part 2 的子方向说明

- **2.1 基础算子**：Transformer 最小构件
- **2.2 模型架构**：把算子组装成完整 block / router / model
- **2.3 微调与训练技术**：SFT、LoRA、调度器与训练流程
- **2.4 对齐技术**：RLHF、DPO 与偏好优化
- **2.5 反向传播与显存优化**：Attention backward、重计算与显存治理
- **2.6 核心推理优化**：FlashAttention、Decoding、PagedAttention
- **2.7 高级推理优化**：Speculative Decoding、RadixAttention、推理量化
- **2.8 分布式与扩展**：Checkpointing、QLoRA、ZeRO、并行扩展

### Part 3 / Part 4 的子方向说明

- **3.1 Triton 基础**：Triton 编程模型与基础 kernel
- **3.2 Triton 进阶**：融合算子、Softmax、RoPE、PagedAttention、Quantization、Multi-LoRA
- **3.3 Triton 项目**：调试、内存模型和综合项目
- **4.1~4.4**：CUDA C++、系统优化、分布式工程和架构视野

## Notebook 写作约定

这个仓库里的 notebook 一般分成三个区域：题目区、答案区、测试区。维护时应把三者视为一个整体，而不是三段互不相关的文本。

### 题目区

- 只保留完成任务所需的最小骨架
- 必须包含目标、输入输出、接口名和 `TODO`
- 可以提示关注点，例如 `stride`、`mask`、`boundary shape`、`reference behavior`
- 不要在题目区直接写出完整答案、关键公式推导或完整控制流
- 不要使用 `... 省略 ...`、`假设这里调用了 ...` 这类占位句来代替真正的教学内容
- Part 2 的题目区更偏算法练习骨架，允许给出少量推导提示，但不能提前给出完整解法链
- Part 3 的题目区更偏工程实现骨架，允许给出接口、约束和验证点，但不能提前给出完整执行路径
- 如果需要占位，应优先占位“目标、约束、验证点”，而不是占位“省略实现过程”
- 修改题目区前先锁定页面角色：`exercise`、`explanation` 或 `capstone`
- 如果页面角色是 `exercise`，题目区不得包含完整实现、参考答案逻辑或主 benchmark 路径
- 如果页面角色不是 `exercise`，不要再用题目页的练习结构伪装它
- 先审题目区是否仍然是“题目”，再决定是否动手改代码

### 答案区

- 必须与题目区保持同一接口、同一输入输出语义、同一约束假设
- 答案区应该是标准参考实现，而不是另一套独立设计
- 如果为了可运行性使用简化版实现，要明确说明它是 stand-in，并保持行为可解释

### 测试区

- 优先做行为验证，而不是字符串验证
- 优先检查输出、shape、dtype、数值容差、边界形状和 GPU guard
- 对 capstone 或集成页，测试应验证真实模块行为，而不是 helper 名字是否出现
- 如果确实需要结构性检查，只能作为次要兜底，不应作为主验证方式

### 提示写法

- 提示目标、约束和验证点
- 不要把最终公式或最终代码路径直接透给读者
- 当结果依赖简化实现或环境假设时，使用有范围限制的表述
- Part 2 可以稍微偏向“推导提示”，Part 3 应更偏向“接口与工程边界提示”
- 对工程页，尽量使用 `stride`、`mask`、`block`、`program_id`、`GPU guard` 这类可执行提示
- 对算法页，尽量使用 `reference behavior`、`boundary case`、`numerical tolerance` 这类可验证提示

### 维护时的检查顺序

1. 先锁定页面角色：`exercise`、`explanation` 或 `capstone`
2. 再看题目区是否仍然是题目，而不是半个答案区
3. 再看答案区是否与题目区对齐
4. 最后看测试区是否真的验证了行为
5. 如果页面有过强或绝对的说法，再补一轮叙事审查

### 后续优化点

- Part 2 还可以继续收紧“推导提示”的粒度，避免个别页提示过强
- Part 3 还可以继续统一题目区的工程措辞，把“接口、约束、验证点”三者固定下来
- 两个部分都要持续减少“形式正确但信息量过强”的占位写法
- 题目区修改时优先做“角色检查”，再做“最小骨架检查”，最后才是测试检查
- Part 3 的新规则应先在样板小节验证通过，再推广到其它小节，避免一次性改动过多页面
- 后续新增 Triton 内容时，优先落在 `3.1~3.3` 这三组；只有明确属于 CUDA/system 的新页，才进入 `3.4~3.5`

## 占位规范

当某个子方向的结构已经确定，但内容尚未补齐时，可以先保留入口页或占位页。占位页需要明确写出：

- 当前已有内容
- 后续准备补什么
- 现在为什么暂不展开

占位页不应伪装成“已完成”，也不应指向不存在的 notebook。
