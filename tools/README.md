# 工具目录

正式工具涵盖文档转换、检查、环境准备、实验运行与回归测试。现有命令和 Python 导入暂时保留原路径。

## 临时脚本

一次性排查、批量迁移草稿放在 [`_scratch/`](./_scratch/README.md)，不再放到 `tools/` 根目录。该目录默认忽略本地内容，仅保留说明和忽略规则。正式测试、公共运行时、实验结果不放在这里。

## 长期分类方案

下表是分批迁移的目标。`tests/` 已迁入首个基础页回归测试，其余正式分类尚未整体迁移。按工具职责分类，不按教程小节编号建目录。

| 目标目录 | 存放内容 | 现有例子 |
|:---|:---|:---|
| `docs/` | 文档转换、镜像与导航同步 | `convert_notebook.py`、`sync_docs_navigation.py` |
| `checks/` | 面向源文件、链接、公式和图片的检查命令 | `validate_notebook_format.py`、`check_svg_arrows.py` |
| `environment/` | 环境安装与预检命令 | `install_environment_profile.py`、`environment_preflight.py` |
| `runtime/` | 被 Notebook 或实验入口导入的公共模块 | `project_runtime.py`、`model_runtime.py` |
| `schemas/` | 结果结构与字段校验模块 | `inference_result_schema.py`、`quantization_result_schema.py` |
| `benchmarks/` | 实验运行、数据准备、归一化与聚合入口 | `benchmark_inference_backend.py`、`aggregate_distributed_evidence.py` |
| [`tests/`](./tests/README.md) | 验证工具和机制实现的长期回归测试 | 已迁入 `test_foundation_performance_evidence.py`；其他测试分批处理 |
| `_scratch/` | 不被正式代码依赖的一次性本地脚本 | 临时排查、迁移草稿 |

## 迁移顺序与检查

先迁移调用少的回归测试，再整理独立检查与转换命令，最后处理公共运行时和结果校验模块。`test_notebook_answers.py`、`test_framework_conformance.py` 等虽然以 `test_` 命名，实际承担 Notebook 验证命令，应按用途归类，而不是只按文件名前缀搬动。

每批迁移都要同步检查 Notebook、文档、`verify.py`、CI 与脚本间的引用；重新核对根据 `__file__` 推导的项目根目录，以及直接执行脚本时的导入路径。已公开的命令和 `tools.*` 导入若需要兼容，应在旧位置保留转发入口，而不保留两份实现。通过原入口、新入口与相关回归测试后，再迁移下一批。

基础页回归测试的实现位于 `tools/tests/test_foundation_performance_evidence.py`，旧命令 `python tools/test_foundation_performance_evidence.py` 仍可运行。旧文件只负责启动，不导出测试类，避免扫描 `tools/` 时重复收集同一组测试；运行全部已迁入测试见[回归测试说明](./tests/README.md)。

首批入口的自动化检查位于 `tools/tests/test_test_entrypoints.py`，覆盖新旧命令、参数选择、跨目录运行、重复收集和真实断言失败。运行时模块或实验入口后续迁移时，还需增加相应导入、结果文件与调用方验证，不能用这批测试代替。

| 验证项 | 通过条件 |
|:---|:---|
| 行为与结果 | 原有断言通过；计算、字段、默认值和输出语义没有因移动改变 |
| 命令入口 | 新旧命令均可运行，参数与成功、失败退出码保持一致 |
| 导入与路径 | 实际使用的导入方式有效；项目根目录、相对资源和输出位置仍正确 |
| 测试发现 | 标准扫描能找到全部用例，兼容入口不重复收集测试 |
| 调用方 | Notebook、文档、CI、`verify.py` 和其他工具没有悬空引用；受影响调用方完成验证 |
| 副作用 | 未新增安装、下载、GPU 启动、源文件覆盖或输出位置变化 |
