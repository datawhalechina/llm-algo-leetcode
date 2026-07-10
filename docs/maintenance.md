# 维护与发布手册

只看三件事：怎么同步、怎么验证、脚本各管哪一层。正文模板规则见 [template_guidelines.md](./template_guidelines.md)。

## 脚本分层

| 层 | 脚本 | 作用 |
|---|---|---|
| `verify` | `verify.py` | 统一验证入口 |
| `convert` | `convert_notebook.py` | 正文镜像主链路 |
| `sync` | `tools/sync_docs_index.py`、`tools/sync_docs_navigation.py` | 首页 / 导学页 / 组页同步 |
| `check` | `check_source_docs_mirror.py`、`check_chapter_links.py` | 镜像和链接检查 |
| `test` | `test_chapter0_1_notebooks.py`、`test_notebook_answers.py` | Notebook 校验 |
| `migration` | `tools/md_to_notebook.py` | markdown -> notebook 迁移辅助 |

`convert_chapter0_1.py` 只保留 legacy 兼容。

## 日常流程

1. 先改 source。
2. 首页改动后跑 `python tools/sync_docs_index.py`。
3. 导学页 / 组页改动后跑 `python tools/sync_docs_navigation.py`。
4. 正文改动后跑 `python convert_notebook.py`。
5. 最后跑 `cd docs && npm run docs:build`。

## 常用命令

```bash
python verify.py part0_1 --no-build
python verify.py part2 --no-build
python verify.py part3 --no-build
python verify.py part4 --no-build
python verify.py all --no-build
python tools/sync_docs_index.py
python tools/sync_docs_navigation.py
python convert_notebook.py
cd docs && npm run docs:build
```

## 说明

- `Part 0 / Part 1` 用 `test_chapter0_1_notebooks.py`
- `Part 2 / Part 3` 用 `test_notebook_answers.py`
- 先改源，再同步 `docs/`
- 导学页、组页、正文页分开同步
- `convert_chapter0_1.py` 只保留兼容用途
- `tools/md_to_notebook.py` 只保留迁移辅助用途
