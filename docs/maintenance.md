# 维护与发布手册

本手册只回答三个操作问题：修改哪个源文件、怎样生成 `docs/` 镜像、怎样验证改动。页面结构与写作规则见[模板总纲](./template_guidelines.md)，环境和结果字段分别见[环境资产表](./gpu_environment_assets.md)与[Part 02 结果资产](./part02_result_assets.md)。

## 维护信息所有权

| 信息 | 唯一维护入口 | 本手册是否展开 |
|:---|:---|:---:|
| 页面类型与共同规则 | [模板总纲](./template_guidelines.md) | 否 |
| Part 00–01 基础页 | [基础页编写规范](./authoring/foundation_notebooks.md) | 否 |
| 专题 intro、walkthrough、casebook、Quickstart | [专题页面编写规范](./authoring/topic_pages.md) | 否 |
| Part 02 机制页 / 项目页 | [机制页规范](./authoring/part02_mechanism_notebooks.md) / [项目页规范](./authoring/part02_project_notebooks.md) | 否 |
| 图示与 SVG | [图示与 SVG 规范](./authoring/visual_guidelines.md) | 否 |
| 模型、依赖 Profile、驱动和硬件事实 | [GPU 环境资产表](./gpu_environment_assets.md) | 否 |
| 结果字段、证据等级和项目资产 | [Part 02 结果资产](./part02_result_assets.md) | 否 |
| 文档、自动化、视觉和引用资产归属 | [资产总登记](./asset_registry.md) | 否 |
| 转换、同步、检查、构建和交接 | 本手册 | 是 |

规则冲突时，先确定信息属于哪一行，再修改对应唯一维护源。不要在 Notebook、专题页或维护页增加第二份完整规则。

## 源文件与派生文件

- 源 Notebook、源 Markdown 和根目录 `README.md` 是正文事实来源。
- `docs/` 下的正文和导航是派生镜像，不能作为最终修改入口。
- `docs/public/` 中的正式 SVG、图片等静态资产直接维护，不由 Notebook 转换生成。
- 镜像与源文件不一致时，以源文件为准；重新同步会覆盖镜像。
- 兼容页、旧路径和生成链的归属见[文档交付资产表](./documentation_assets.md)。

## 标准维护流程

```text
确定信息所有权
→ 修改源文件或正式静态资产
→ 运行定向检查
→ 同步 docs 镜像
→ 运行对应 Part 的统一验证
→ 检查 diff、链接和实际渲染
→ 记录已验证项与未执行证据
```

### 1. 修改前

1. 使用 `git status --short` 确认工作区已有改动，避免覆盖他人或此前实验结果。
2. 从[模板总纲](./template_guidelines.md)确定页面类型，从[资产总登记](./asset_registry.md)确定事实归属。
3. 涉及重命名、调号或目录迁移时，先搜索源文件、镜像、导航、脚本和结果资产中的全部引用。

### 2. 修改与定向检查

- Notebook 修改后先检查 JSON、cell ID 和目标代码路径。
- SVG 修改后先检查 XML、节点边界、箭头和阅读尺寸。
- 结果 schema 或项目身份变化时，先运行对应 helper/schema 测试，再更新消费者。
- 真实 GPU、backend 或多卡实验未执行时，保留为待采证据；不能用 CPU smoke 或模拟结果替代。

### 3. 同步镜像

| 修改对象 | 同步入口 |
|:---|:---|
| 根目录 `README.md` | `python tools/sync_docs_index.py` |
| Part 导读、组页、专题 Markdown | `python tools/sync_docs_navigation.py` |
| 单个 Notebook 或正文文件 | `python tools/convert_notebook.py --file <source>` |
| 一个源目录 | `python tools/convert_notebook.py --dir <source-dir>` |

默认优先使用定向参数。全量同步只在导航、生成器或大量源文件同时变化时运行。`tools/convert_chapter0_1.py` 仅保留历史兼容，不进入日常流程。

### 4. 验证与交接

1. 运行对应 Part 的统一验证入口。
2. 运行 `git diff --check`，确认没有空白错误。
3. 检查源文件与 `docs/` 镜像、本地链接和站点构建。
4. 图片必须查看实际渲染结果；自动检查不能替代连线和阅读尺寸检查。
5. 交接时记录：修改范围、已执行命令、未执行的 GPU/外部环境证据、存量问题和下一步。

## 同步与验证层级

日常维护先使用 `verify.py`。底层脚本只用于定向检查、故障定位或 CI 规则开发。

| 范围 | 主入口 | 补充入口 | 说明 |
|:---|:---|:---|:---|
| Part 00–01 | `python verify.py part0_1 --no-build` | `python verify.py part0_1_audit` | 讲解型 Notebook：顺序执行代码并审计导读、链接和 cell ID |
| Part 02 | `python verify.py part2 --no-build` | 对应 helper/schema 测试 | 机制页和项目页：题目、测试、答案、实验契约 |
| Part 03 | `python verify.py part3 --no-build` | Kernel 专项测试 | Triton 与 Kernel 正确性、编译和性能入口 |
| Part 04 | `python verify.py part4 --no-build` | 系统专项测试 | CUDA、通信、调度与系统实验入口 |
| 全仓 | `python verify.py all --no-build` | `cd docs && npm run docs:build` | 合并前或生成链变化后使用 |

Part 00–01 不使用 Part 02 的题目—测试—答案抽取规则；Part 02/03 也不能只用“所有代码 Cell 顺序执行”替代专项验证。

## 常用命令

```bash
# 统一验证
python verify.py part0_1 --no-build
python verify.py part0_1_audit
python verify.py part2 --no-build
python verify.py part3 --no-build
python verify.py part4 --no-build
python verify.py all --no-build

# 定向同步
python tools/sync_docs_index.py
python tools/sync_docs_navigation.py
python tools/convert_notebook.py --file <source-file>
python tools/convert_notebook.py --dir <source-dir>

# 通用检查
python tools/check_source_docs_mirror.py
python tools/check_chapter_links.py --scope source
python tools/check_chapter_links.py --scope docs
python tools/check_svg_arrows.py <svg...>
python tools/check_svg_layout.py
git diff --check

# 站点构建
cd docs && npm run docs:build
```

## 测试脚本索引

| 层 | 稳定入口 | 作用 |
|:---|:---|:---|
| 统一验证 | `verify.py` | 按 Part 组织转换、检查和测试 |
| 转换 | `tools/convert_notebook.py` | 从源 Notebook / Markdown 生成正文镜像 |
| 导航同步 | `tools/sync_docs_index.py`、`tools/sync_docs_navigation.py` | 同步首页、导读、组页和专题导航 |
| 镜像与链接 | `tools/check_source_docs_mirror.py`、`tools/check_chapter_links.py`、`tools/check_docs_links.py` | 检查源/镜像与本地链接 |
| Notebook 格式 | `tools/validate_notebook_format.py` | 检查 JSON、schema、cell ID 与 metadata |
| Part 00–01 | `tools/test_chapter0_1_notebooks.py`、`tools/audit_chapter0_1_notebooks.py` | 顺序执行代码、结构审计与 warning 归档 |
| Part 02/03 | `tools/test_notebook_answers.py` | 验证题目区与答案区 |
| SVG | `tools/check_svg_arrows.py`、`tools/check_svg_layout.py` | 检查箭头结构、画布和页面尺度 |
| 历史迁移 | `tools/md_to_notebook.py` | Markdown → Notebook 迁移辅助，不进入日常流程 |

完整稳定工具资产及其消费者见[自动化与验证资产表](./automation_assets.md)。本表不枚举一次性修复脚本。

## 环境与实验维护

- `requirements/base.txt` 不放 PyTorch、GPU 驱动或 Serving backend。
- CPU / CUDA PyTorch、能力 Profile 和 backend 组合按[GPU 环境资产表](./gpu_environment_assets.md)维护。
- Notebook 可以检查环境，但不能静默替换 PyTorch、驱动、vLLM、SGLang 或其他 CUDA 执行栈。
- 学习者安装流程和项目预检见[使用指南](./guide.md#项目环境预检与安装)；Notebook 只保留当前页需要修改的配置和短入口。
- 运行结果必须保存 runtime snapshot、配置、状态和失败原因；已完成标签进入结果 JSON 和专题实验资产表，不回写为源 Notebook 默认配置。

## 模型评测与跨专题质量门槛

本节保留为历史链接入口，不在维护手册重复定义评测字段。凡候选方案会改变模型输出，项目决策必须同时检查质量、失败状态和性能，不能只凭吞吐或显存变化采用方案。

- 公共结果外层、证据等级和字段复用：[Part 02 结果资产](./part02_result_assets.md)
- 项目页中的质量门槛、决策和交接位置：[Part 02 项目页编写规范](./authoring/part02_project_notebooks.md)
- 评测数据可信度和领域指标：对应横向专题及其实验资产表

## 迁移、重命名与兼容页

1. 先建立引用清单，再修改源文件名、H1、导航、专题链接、项目注册表和镜像名。
2. 旧入口仍被引用时保留明确的兼容或 `Reserved` 页面，并指向新入口。
3. 迁移完成后运行镜像、链接和站点构建检查，确认无引用后再删除旧入口。
4. 不通过手改 `docs/` 镜像绕过源文件迁移。

稳定路径和兼容页登记见[文档交付资产表](./documentation_assets.md)；项目语义 ID 和产物关系见 `benchmarks/project_registry.json` 与 `benchmarks/project_asset_relations.json`。

## 故障处理

| 现象 | 处理 |
|:---|:---|
| 源文件与网页不一致 | 重新定向转换，并运行 `check_source_docs_mirror.py` |
| Notebook 在当前 shell 能运行、Jupyter 不能运行 | 核对 kernelspec、`sys.executable` 和环境 Profile |
| Notebook JSON 合法但测试失败 | 使用对应 Part 的测试模型，不混用 Part 00–01 与 Part 02 规则 |
| 链接检查失败 | 从源文件修复相对路径，再同步镜像 |
| SVG 自动检查通过但页面不可读 | 在实际页面宽度下检查字号、留白、连线端点和箭头 |
| GPU 或 backend 能力不可用 | 保存 preflight/failure，停止形成目标机制或性能结论 |
| 全量同步产生大量无关差异 | 停止后续编辑，核对源文件与生成器范围，不覆盖已有工作区改动 |

## 提交前最小检查

- 改动发生在正确的唯一维护源。
- `docs/` 镜像来自同步脚本，没有反向手改。
- 对应 Part 验证通过，或明确记录未运行原因。
- 本地链接、SVG 渲染和 `git diff --check` 已检查。
- GPU、backend、外部数据或多卡证据没有被模拟结果替代。
- 提交范围不包含无关工作区修改、临时脚本、缓存或本地凭据。
