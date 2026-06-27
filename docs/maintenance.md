# 维护与发布手册

本页记录仓库维护、同步和验证的统一规则，尽量只保留“权威入口”和“常用流程”。

## 职责分工

- `README.md`：首页文案源头
- `docs/index.md`：由 `README.md` 自动同步生成
- `docs/guide.md`：学习方式与环境选择
- `docs/contributing.md`：贡献入口与快速验证
- `docs/template_guidelines.md`：Notebook 题目模板与网页化转换规范
- `docs/.vitepress/config.mts`：站点导航与侧边栏

## 仓库结构

- `00_Prerequisites/`、`01_Hardware_Math_and_Systems/`、`02_PyTorch_Algorithms/`、`03_Triton_Kernels/`、`04_CUDA_and_System_Optimization/`：各部分源文件
- `docs/00_Prerequisites/`、`docs/01_Hardware_Math_and_Systems/`、`docs/02_PyTorch_Algorithms/`、`docs/03_Triton_Kernels/`：站点镜像
- `process/`：交接、计划、总结等过程文件
- `cnb/`、`environment.yml`、`requirements/*.txt`：环境与依赖

## 同步与生成

- 修改源文件后，优先重新生成 `docs/` 镜像
- `README.md` 更新后，运行 `python tools/sync_docs_index.py`
- `convert_notebook.py` 支持全量或局部同步

常用命令：

```bash
python tools/sync_docs_index.py
python convert_notebook.py --dir 03_Triton_Kernels
python convert_notebook.py --file 03_Triton_Kernels/05_Triton_Autotune_and_Profiling.ipynb
```

## 验证入口

- `python verify.py chapter0_1`
- `python verify.py chapter2`
- `python verify.py chapter3`
- `python verify.py chapter4`
- `python verify.py all`
- `python check_chapter_links.py`
- `python check_source_docs_mirror.py`
- `cd docs && npm run docs:build`

Notebook 元数据若有 `MissingIDFieldWarning`，先跑：

```bash
python tools/normalize_notebooks.py
```

## 环境约定

- `requirements/base.txt`：基础依赖
- `requirements/dev.txt`：开发和测试依赖
- `requirements/gpu.txt`：GPU 扩展依赖
- `environment.yml`：Python 版本与依赖串联
- `cnb/environment.yml`：CNB 环境骨架

## 维护原则

- 先改源文件，再同步 `docs/`
- 入口页和正文页分开维护
- 只在需要时维护 group md
- 变更尽量拆成小 commit
- `SESSION_HANDOFF.md` 这类过程文件不进站点导航

## 风格约定

- `README.md` / `docs/index.md` 可保留首页型 emoji
- `intro.md` 可保留少量章节标记 emoji
- 正文 notebook 只保留状态符号 `⏭️ / ✅ / ❌` 和 `🛑 STOP HERE`

## 开源协议

本项目采用 [CC BY-NC-SA 4.0](../LICENSE) 进行许可。
