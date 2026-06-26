# 自动化测试脚本索引

测试约定、维护流程和发布规则统一见 [维护与发布手册](./docs/maintenance.md)。

## 核心脚本

| 脚本 | 作用 | 说明 |
| --- | --- | --- |
| `verify.py` | 统一验证入口 | 推荐日常维护优先使用；与底层脚本是“二选一且有优先级”的关系 |
| `test_notebook_answers.py` | Part 2 / 3 答案验证 | 含防透题检查 |
| `test_chapter0_1_notebooks.py` | Part 0 / 1 顺序执行 | 直接跑练习 notebook |
| `check_chapter_links.py` | 站内链接检查 | 检查 Part 0 / 1 路由 |
| `check_source_docs_mirror.py` | 部分正文镜像检查 | 检查 source / docs 一致性 |

## 推荐用法

```bash
python verify.py chapter0_1 --no-build
python verify.py chapter2 --no-build
python verify.py chapter3 --no-build
python verify.py all --no-build
```

无 GPU 环境下，`verify.py` 会自动跳过 Part 2 / 3 的 GPU-only notebook 答案验证，但仍会保留转换、镜像和链接检查。
如果只想排查单个 notebook 或单个底层脚本，可以直接用旧脚本；日常维护优先用 `verify.py`。

## 去向

- 环境分层与平台选择见 [使用指南](./docs/guide.md)
- 站点入口见 [README](./README.md) 和 [docs 首页](./docs/index.md)
- 第零部分 / 第一部分 notebook 只作为练习资产通过链接进入；第二部分 / 第三部分 notebook 内容可在网页正文中以转换后的 Markdown 形式看到
