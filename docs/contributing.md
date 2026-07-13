# 贡献指南

欢迎参与“大模型算法实战教程 / LLM Algorithm Practice Lab”。

## 先看这三份

- `docs/maintenance.md`：同步和验证规则
- `docs/template_guidelines.md`：Notebook 模板
- `docs/guide.md`：学习路径和环境选择

## 贡献方式

- 提 Issue：反馈错误、补充建议、提出新题
- 提 PR：修正文档、补练习、改实现
- 参与讨论：在 GitHub Discussions 交流

## 常用验证

```bash
python verify.py part0_1 --no-build
python verify.py part2 --no-build
python verify.py part3 --no-build
python verify.py all --no-build
```

单个 notebook：

```bash
python tools/test_notebook_answers.py path/to/your.ipynb --mode both
```

## 提交前检查

- 不透题
- 能运行
- 已同步
- 可构建

## 相关链接

- [GitHub 仓库](https://github.com/datawhalechina/llm-algo-leetcode)
- [GitHub Issues](https://github.com/datawhalechina/llm-algo-leetcode/issues)
- [GitHub Discussions](https://github.com/datawhalechina/llm-algo-leetcode/discussions)
