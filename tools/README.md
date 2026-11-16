# 工具目录

正式工具涵盖文档转换、检查、环境准备、实验运行与回归测试。现有命令和 Python 导入暂时保留原路径。

## 先按用途选择入口

| 用途 | 现有入口 | 使用方式 |
|:---|:---|:---|
| 教程验证 | 根目录 `verify.py`、`test_notebook_answers.py`、`test_framework_conformance.py` | 按对应使用说明选择范围与模式；Notebook 正确性测试和性能测量分开运行 |
| 文档生成 | `convert_notebook.py`、`convert_chapter0_1.py`、`sync_docs_navigation.py` | 从源文件生成镜像；会写入生成目录，运行后检查差异 |
| 源文件检查 | `check_chapter_links.py`、`check_math_formula_symbols.py`、`audit_chapter0_1_notebooks.py` | 按参数检查链接、公式或 Notebook 结构；名称中含 `audit` 不代表临时脚本 |
| 环境准备 | `install_environment_profile.py`、`environment_preflight.py`、`register_notebook_runtime.py` | 区分安装、预检和路径注册；安装与注册会改变本地环境 |
| 实验运行 | `benchmark_inference_backend.py` 等实验入口 | 先确认环境、负载、标签和输出位置，再启动实际测量 |
| 公共模块 | `project_runtime.py`、`model_runtime.py`、各类 `*_result_schema.py` | 供 Notebook 与工具导入；按调用关系维护，不作为临时文件清理 |
| 工具回归测试 | [`tests/`](./tests/README.md) 与根目录尚未迁移的测试 | 按测试说明运行；是否属于回归测试不能只看文件名前缀 |

下表中的路径除 `verify.py` 外均相对于 `tools/`。完整参数以各命令的 `--help` 和所属专题 Quickstart 为准。

## 临时脚本

一次性排查、批量迁移草稿放在 [`_scratch/`](./_scratch/README.md)，不再放到 `tools/` 根目录。该目录默认忽略本地内容，仅保留说明和忽略规则。正式测试、公共运行时、实验结果不放在这里。

## 长期分类方案

下表是分批迁移的目标，不是当前可用的新命令路径。`tests/` 已承载基础页、机制实现、入口兼容与文档回归测试，其余正式分类尚未整体迁移。按工具职责分类，不按教程小节编号建目录。

| 目标目录 | 存放内容 | 现有例子 |
|:---|:---|:---|
| `docs/` | 文档转换、镜像与导航同步 | `convert_notebook.py`、`sync_docs_navigation.py` |
| `checks/` | 面向源文件、链接、公式和图片的检查命令 | `validate_notebook_format.py`、`check_svg_arrows.py` |
| `environment/` | 环境安装与预检命令 | `install_environment_profile.py`、`environment_preflight.py` |
| `runtime/` | 被 Notebook 或实验入口导入的公共模块 | `project_runtime.py`、`model_runtime.py` |
| `schemas/` | 结果结构与字段校验模块 | `inference_result_schema.py`、`quantization_result_schema.py` |
| `benchmarks/` | 实验运行、数据准备与测量聚合入口 | `benchmark_inference_backend.py`、`aggregate_distributed_evidence.py` |
| `migrations/` | 旧格式转换与限定版本的批量迁移 | `md_to_notebook.py`、历史结果归一化工具；先补写入保护，再决定迁移 |
| [`tests/`](./tests/README.md) | 验证工具和机制实现的长期回归测试 | 基础页、入口兼容与性能文档测试；其他测试分批处理 |
| `_scratch/` | 不被正式代码依赖的一次性本地脚本 | 临时排查、迁移草稿 |

这里的 `tools/benchmarks/` 是实验命令的目标位置，根目录 `benchmarks/` 存放实验配置、负载与结果。移动命令不改变实验资产的保存位置。

## 批量改写与历史转换工具

下表区分当前保护与剩余使用限制。这些工具不属于日常文档构建步骤；使用前先确认输入版本、目标文件与已有输出。

| 工具 | 当前作用与写入范围 | 当前使用限制与后续处理 |
|:---|:---|:---|
| `standardize_part02_project_failures.py` | 对固定的 13 个历史目标预览旧函数体替换规则，默认只读，可用 `--file` 缩小范围 | 写入已冻结，`--write` 明确报错。旧标题与文本规则的预览不表示当前题目需要替换；实际修复按当前单元标签逐页进行 |
| `md_to_notebook.py` | 将 Part00／01 的 Markdown 转成 Notebook；可用 `--dry-run`，已有 Notebook 默认跳过，`--overwrite` 可覆盖 | 属于早期格式迁移，已排除 intro 和数字加单个大写字母命名的组页（含 `1F.md`）。转换前先预览明确文件或目录，确认选择范围；预览与已有文件保护由 `tests/test_md_to_notebook.py` 覆盖 |
| `normalize_notebooks.py` | 规范 Notebook JSON 与单元 ID；不带参数时默认处理 Part00–04，未加 `--check` 会原地保存 | 属于可复用格式工具。先对明确文件运行 `--check`；后续改为预览默认、显式写入，并检查格式化是否造成无关差异 |
| `normalize_inference_results.py` | 将历史推理结果转换为共享结构，默认写出同目录的 `*_normalized.json` | 支持 `--dry-run`；拒绝将输入及其硬链接用作输出，也拒绝符号链接输出。已有输出字节相同则复用，不同则报错，不提供强制覆盖 |
| `normalize_training_memory_results.py` | 调用训练显存结果校验模块，默认生成 `*_v1.json`，可指定输出目录 | 支持 `--dry-run`，使用相同的输入隔离和冲突保护；多输入按文件顺序处理，后一个失败不会撤销此前已生成的旁路结果 |

两类结果转换共用 `result_migration_io.py` 保存派生报告，避免各自维护不同的覆盖规则。预览会检查已有输出，但不创建文件或目录；实际保存以独占创建模式保护新输出。字节相同的判断包含 JSON 排版，因此内容等价但排版不同的已有文件也需另选路径。迁移不会补造缺失的实测字段，也不改变原来的数值转换规则。

整理工具时采用以下约束：

- 找不到静态引用的脚本先核对 CLI、CI、Notebook 和文档用法，不据此直接删除。
- 正式迁移工具保留可复现的实现与测试；本地迁移草稿进入 `_scratch/`，不让正式代码依赖它。
- 新增批量改写命令先提供只读预览，列出目标与变化；通过显式写入选项执行，已有不同内容时停止或另选输出。
- 结果格式转换保留原始测量，派生文件注明来源；转换字段不等于重新完成实验。

## 迁移顺序与检查

先迁移调用少的回归测试，再整理独立检查与转换命令，最后处理公共运行时和结果校验模块。`test_notebook_answers.py`、`test_framework_conformance.py` 等虽然以 `test_` 命名，实际承担 Notebook 验证命令，应按用途归类，而不是只按文件名前缀搬动。

每批迁移都要同步检查 Notebook、文档、`verify.py`、CI 与脚本间的引用；重新核对根据 `__file__` 推导的项目根目录，以及直接执行脚本时的导入路径。已公开的命令和 `tools.*` 导入若需要兼容，应在旧位置保留转发入口，而不保留两份实现。通过原入口、新入口与相关回归测试后，再迁移下一批。

基础页回归测试的实现位于 `tools/tests/test_foundation_performance_evidence.py`，旧命令 `python tools/test_foundation_performance_evidence.py` 仍可运行。旧文件只负责启动，不导出测试类，避免扫描 `tools/` 时重复收集同一组测试；运行全部已迁入测试见[回归测试说明](./tests/README.md)。

章节链接、SVG 箭头、SVG 布局与数学公式兼容的回归测试也位于 `tools/tests/`，原脚本路径保留转发入口。这四个旧入口通过 `load_tests` 支持显式的 `python -m unittest tools.test_<name>`；目录扫描只收集新位置，不导出第二份测试类。数学兼容 CI 直接调用 `tools.tests.test_math_formula_compatibility`，并运行入口回归测试核对实际用例数。

入口的自动化检查位于 `tools/tests/test_test_entrypoints.py`，按实际用例数核对新旧脚本、检查类测试的模块入口、参数选择、跨目录运行、重复收集和真实断言失败。运行时模块或实验入口后续迁移时，还需增加相应导入、结果文件与调用方验证，不能用这批测试代替。

| 验证项 | 通过条件 |
|:---|:---|
| 行为与结果 | 原有断言通过；计算、字段、默认值和输出语义没有因移动改变 |
| 命令入口 | 新旧命令均可运行，参数与成功、失败退出码保持一致 |
| 导入与路径 | 实际使用的导入方式有效；项目根目录、相对资源和输出位置仍正确 |
| 测试发现 | 标准扫描能找到全部用例，兼容入口不重复收集测试 |
| 调用方 | Notebook、文档、CI、`verify.py` 和其他工具没有悬空引用；受影响调用方完成验证 |
| 副作用 | 未新增安装、下载、GPU 启动、源文件覆盖或输出位置变化 |
