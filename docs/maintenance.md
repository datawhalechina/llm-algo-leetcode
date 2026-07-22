# 维护与发布手册

只看三件事：怎么同步、怎么验证、脚本各管哪一层。正文模板规则见 [template_guidelines.md](./template_guidelines.md)。

## 脚本分层

| 层 | 脚本 | 作用 |
|---|---|---|
| `verify` | `verify.py` | 统一验证入口 |
| `convert` | `tools/convert_notebook.py` | 正文镜像主链路 |
| `sync` | `tools/sync_docs_index.py`、`tools/sync_docs_navigation.py` | 首页 / 导学页 / 组页同步 |
| `check` | `tools/check_source_docs_mirror.py`、`tools/check_chapter_links.py` | 镜像和链接检查 |
| `test` | `tools/test_chapter0_1_notebooks.py`、`tools/test_notebook_answers.py` | Notebook 校验 |
| `migration` | `tools/md_to_notebook.py` | markdown -> notebook 迁移辅助 |

`tools/convert_chapter0_1.py` 只保留 legacy 兼容。

## Part 0-4 维护分工

- `Part 00` 和 `Part 01` 一起作为前置知识层，重点是基础语言、张量、系统视角和性能边界。
- `Part 02` 是主干实现层，重点是 PyTorch 里的训练、推理、并行、量化和项目收口。
- `Part 03` 是 Triton / kernel 过渡层，重点是把框架级实现继续下沉到高性能算子。
- `Part 04` 是 CUDA / 系统优化层，重点是继续向硬件、通信、调度和架构收口。
- 维护时可以按下面的验证分段理解：
  - `verify.py part0_1`：检查 `Part 00 / Part 01`
  - `verify.py part2`：检查 `Part 02`
  - `verify.py part3`：检查 `Part 03`
  - `verify.py part4`：检查 `Part 04`
- 横向专题主要横切 `Part 00 / Part 01 / Part 02`，后续若继续下探性能和实现，可以逐步接到 `Part 03 / Part 04`。

## 日常流程

1. 先改 source。
2. 首页改动后跑 `python tools/sync_docs_index.py`。
3. 导学页 / 组页改动后跑 `python tools/sync_docs_navigation.py`。
4. 正文改动后跑 `python tools/convert_notebook.py`。
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
python tools/convert_notebook.py
cd docs && npm run docs:build
```

## 测试脚本索引

| 层 | 脚本 | 作用 |
|---|---|---|
| `verify` | `verify.py` | 统一验证入口 |
| `convert` | `tools/convert_notebook.py` | 正文镜像主链路 |
| `sync` | `tools/sync_docs_index.py`、`tools/sync_docs_navigation.py` | 首页 / 导学页 / 组页同步 |
| `check` | `tools/check_source_docs_mirror.py`、`tools/check_chapter_links.py` | 镜像和链接检查 |
| `check` | `tools/check_docs_links.py`、`tools/check_math_formula_symbols.py`、`tools/check_part01_code_blocks.py` | docs 链接、公式与代码块检查 |
| `test` | `tools/test_chapter0_1_notebooks.py`、`tools/test_notebook_answers.py` | Notebook 校验 |
| `migration` | `tools/md_to_notebook.py` | markdown -> notebook 迁移辅助 |

## 推荐用法

```bash
python verify.py part0_1 --no-build
python verify.py part2 --no-build
python verify.py part3 --no-build
python verify.py all --no-build
python tools/check_math_formula_symbols.py
python tools/check_part01_code_blocks.py
```

无 GPU 时，`verify.py` 会跳过 Part 2 / 3 的 GPU-only 答案验证，但仍保留转换、镜像和链接检查。单独排查时直接用底层脚本。
`tools/md_to_notebook.py` 仅用于历史迁移，不进入日常主流程。

## 说明

- `Part 0 / Part 1` 用 `tools/test_chapter0_1_notebooks.py`
- `Part 2 / Part 3` 用 `tools/test_notebook_answers.py`
- 先改源，再同步 `docs/`
- 导学页、组页、正文页分开同步
- `tools/convert_chapter0_1.py` 只保留兼容用途
- `tools/md_to_notebook.py` 只保留迁移辅助用途
