# 自动化测试脚本索引

规则见 [维护与发布手册](./docs/maintenance.md)。

## 脚本分层

| 层 | 脚本 | 作用 |
|---|---|---|
| `verify` | `verify.py` | 统一验证入口 |
| `convert` | `tools/convert_notebook.py` | 正文镜像主链路 |
| `sync` | `tools/sync_docs_index.py`、`tools/sync_docs_navigation.py` | 首页 / 导学页 / 组页同步 |
| `check` | `tools/check_source_docs_mirror.py`、`tools/check_chapter_links.py` | 镜像和链接检查 |
| `test` | `tools/test_chapter0_1_notebooks.py`、`tools/test_notebook_answers.py` | Notebook 校验 |
| `migration` | `tools/md_to_notebook.py` | markdown -> notebook 迁移辅助 |

## 推荐用法

```bash
python verify.py part0_1 --no-build
python verify.py part2 --no-build
python verify.py part3 --no-build
python verify.py all --no-build
```

无 GPU 时，`verify.py` 会跳过 Part 2 / 3 的 GPU-only 答案验证，但仍保留转换、镜像和链接检查。单独排查时直接用底层脚本。
`tools/md_to_notebook.py` 仅用于历史迁移，不进入日常主流程。

## 去向

- 环境选择见 [使用指南](./docs/guide.md)
- 维护规则见 [维护与发布手册](./docs/maintenance.md)
- 站点入口见 [README](./README.md) 和 [docs 首页](./docs/index.md)
