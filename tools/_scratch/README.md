# 临时脚本

这里存放一次性排查和迁移草稿。除本说明与 `.gitignore` 外，内容默认不提交；忽略规则不是保密措施，不要在脚本中写入凭据。

- 从仓库根目录运行，例如 `python tools/_scratch/probe_issue.py`；需要导入 `tools.*` 时使用 `python -m tools._scratch.probe_issue`。
- 写操作默认先提供只读预览，并明确目标文件，确认后再执行。
- CI、Notebook 和正式工具不得依赖此目录。需要长期复用的逻辑，应补测试并迁入正式分类。
- JSON、执行轨迹、图片和模型产物仍放在对应结果或缓存目录，不混入脚本目录。
