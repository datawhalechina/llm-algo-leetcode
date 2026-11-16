# 源文件与文档检查

这里存放检查命令的实际实现。旧的 `tools/<脚本名>.py` 路径保留兼容入口，导入时指向同一个模块，不保留第二份实现。

| 命令 | 检查内容 | 依赖与写入行为 |
| --- | --- | --- |
| `python -m tools.checks.check_source_docs_mirror` | Part00–05的Markdown与示例配置是否有对应镜像 | Python标准库；只读，不比较文件内容 |
| `python -m tools.checks.check_mermaid_blocks` | README、docs/index及docs中Mermaid代码块的语法 | 本地Node与docs已安装的Mermaid依赖；只读，不安装依赖或下载资源 |

也可以直接运行 `python tools/checks/<脚本名>.py`。项目根目录从实现位置解析，不随当前工作目录改变；相对输入参数仍按各命令原有规则解释。

迁移回归测试：

```bash
python -S -m unittest tools.tests.test_checks_migration tools.tests.test_document_tooling
```
