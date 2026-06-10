# 自动化测试脚本索引

测试约定、维护流程和发布规则统一见 [维护与发布手册](./docs/maintenance.md)。

## 核心脚本

| 脚本 | 作用 | 说明 |
| --- | --- | --- |
| `test_notebook_answers.py` | Chapter 2 / 3 答案验证 | 含防透题检查 |
| `test_chapter0_1_notebooks.py` | Chapter 0 / 1 顺序执行 | 直接跑练习 notebook |
| `check_chapter_links.py` | 站内链接检查 | 检查 Chapter 0 / 1 路由 |

## 去向

- 环境分层与平台选择见 [使用指南](./docs/guide.md)
- 站点入口见 [README](./README.md) 和 [docs 首页](./docs/index.md)
