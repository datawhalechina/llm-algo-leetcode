# 文档生成与同步

这里存放文档生成工具的实际实现。旧的`tools/<脚本名>.py`命令及导入继续转发到同一个模块。

| 命令 | 处理范围与写入行为 |
| --- | --- |
| `python -m tools.docs.convert_notebook --dir <课程目录> --dry-run` | 预览课程目录中的Notebook、Markdown和示例配置；移除`--dry-run`才写入镜像。不指定范围时，保留原有的整树清理重建行为 |
| `python -m tools.docs.convert_chapter0_1` | 按原有专用规则重建Part00／01镜像；会先清理这两个生成目录，没有预览参数 |
| `python -m tools.docs.sync_docs_navigation --part <Part目录>` | 同步指定Part的导航页；`--topic <专题>`同步单个专题，只有显式`--prune`才清理该专题的旧Markdown |
| `python -m tools.docs.sync_docs_index` | 将根README转换为`docs/index.md`，调整站点链接 |

上述模块命令在仓库根目录运行。两个转换器的相对输入和输出仍按当前工作目录解析；导航与首页同步使用工具解析出的仓库根目录。迁移没有改变这些规则，也不修改源Notebook或执行其中的代码。

使用命令行同步镜像，不手工修改生成的Markdown。`docs/public/`中的源SVG另按图片资产维护。

迁移测试仅使用临时目录，不重建工作区文档：

```bash
python -S -m unittest tools.tests.test_docs_migration tools.tests.test_document_tooling tools.test_sync_docs_navigation
```
