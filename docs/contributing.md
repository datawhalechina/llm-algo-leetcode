# 贡献指南

欢迎参与“大模型算法实战教程 / LLM Algorithm Practice Lab”。本页只保留最常用的贡献入口和提交前检查。

## 贡献方式

- 提 Issue：反馈错误、补充建议、提出新题
- 提 PR：修正文档、补充练习、优化实现
- 参与讨论：在 GitHub Discussions 里交流思路

## 提交前先看

- `docs/maintenance.md`：维护、同步、验证规则
- `docs/template_guidelines.md`：Notebook 题目模板
- `docs/guide.md`：学习路径与环境选择

## 常用验证

```bash
python verify.py chapter0_1 --no-build
python verify.py chapter2 --no-build
python verify.py chapter3 --no-build
python verify.py all --no-build
```

如果需要定点检查单个 notebook：

```bash
python test_notebook_answers.py path/to/your.ipynb --mode both
```

## 提交前检查

- 题目区没有透题
- 答案区可以运行
- 文档镜像已同步
- 站点构建通过

## 相关链接

- [GitHub 仓库](https://github.com/datawhalechina/llm-algo-leetcode)
- [GitHub Issues](https://github.com/datawhalechina/llm-algo-leetcode/issues)
- [GitHub Discussions](https://github.com/datawhalechina/llm-algo-leetcode/discussions)

