# 使用指南

本页只回答两件事：先在哪看，Notebook 该在哪跑。

## 先记三条

- `Colab` 是阅读和运行入口，不是单独的环境层。
- `CPU-first` / `GPU-required` 是执行能力，不是入口名称。
- `CNB` / `Docker` / 云端 GPU 是统一交付方式。

## 环境怎么选

| 场景 | 建议 |
|---|---|
| 只是先看内容 | 在线站点 |
| Part 0 / Part 1 | 在线 Notebook 或本地基础环境 |
| Part 2 | 本地 CPU-first / Colab CPU / CNB CPU |
| Part 3 | 本地 NVIDIA GPU / Colab GPU / CNB GPU |
| Part 4 | 本地 NVIDIA GPU / Colab GPU / CNB GPU |
| 团队统一交付 | CNB / Docker / 云端 GPU |

如果只想记一句话：

**先在线阅读，再按 Part 0-2 / Part 3-4 选择 CPU-first 或 GPU-required 环境；CNB 和 Docker 负责统一交付。**

## 常用验证

```bash
python verify.py chapter0_1 --no-build
python verify.py chapter2 --no-build
python verify.py chapter3 --no-build
python verify.py chapter4 --no-build
python verify.py all --no-build
```

如果要定点排查：

```bash
python test_chapter0_1_notebooks.py
python test_notebook_answers.py path/to/your.ipynb --mode both
```

## 最小规则

- Part 0 / Part 1：优先在线 Notebook 或本地基础环境。
- Part 2：默认 CPU-first，少数题再切 GPU。
- Part 3 / Part 4：完整验收需要 GPU；没有 GPU 时先阅读。
- CNB 的目标是统一交付，不是新增一套内容分层。

## 版本约定

- `requirements/base.txt`：基础依赖
- `requirements/dev.txt`：开发与测试依赖
- `requirements/gpu.txt`：GPU 扩展依赖
- `environment.yml`：Python 版本和依赖串联
- `cnb/environment.yml`：CNB 环境骨架

