# 环境工具

这里存放环境工具的实际实现。旧的`tools/<脚本名>.py`命令与导入转发到同一模块。

| 命令 | 行为与注意事项 |
| --- | --- |
| `python -m tools.environment.install_environment_profile --profile <ID> --env-name <名称> --dry-run` | 预览Conda安装计划；外部锁定环境按Profile使用`--target`。移除预览参数后可安装依赖及注册Kernel，已有环境仍按原规则要求显式`--reuse` |
| `python -m tools.environment.environment_preflight --list-profiles` | 列出已有Profile，不安装依赖；常规预检可探测已安装包和GPU，`--output`仍按原规则检查输出目录并可能创建父目录 |
| `python -m tools.environment.register_notebook_runtime --check` | 检查当前环境的仓库路径注册和Notebook子目录导入；不带`--check`会写入当前环境的`.pth`文件 |

推荐在仓库根目录执行上述模块命令。安装和预检仍通过现有运行时寻找仓库，可用`LLM_ALGO_PROJECT_ROOT`指定；注册脚本始终使用它所在的仓库。迁移不改变环境名、注册文件名或实验保存位置，也不创建新的环境Profile。

仅运行迁移回归测试：

```bash
python -S -m unittest tools.tests.test_environment_migration
```

安装器测试模拟安装、下载和Conda命令，注册测试只写临时目录，不修改实际Python环境：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tools/test_install_environment_profile.py -q
```
