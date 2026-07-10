# 使用指南

只回答两件事：先在哪看，Notebook 该在哪跑。

## 环境选择

| 场景 | 建议 |
|---|---|
| 先看内容 | 在线站点 |
| Part 0 / Part 1 | 在线 Notebook 或本地基础环境 |
| Part 2 | 本地 CPU-first / Colab CPU / CNB CPU |
| Part 3 / Part 4 | 本地 NVIDIA GPU / Colab GPU / CNB GPU |
| 团队统一交付 | CNB / Docker / 云端 GPU |

## 记住三条

- `Colab` 是阅读和运行入口。
- `CPU-first` / `GPU-required` 是执行能力，不是入口名称。
- `CNB` / `Docker` / 云端 GPU 是统一交付方式。

## 常用命令

```bash
python verify.py part0_1 --no-build
python verify.py part2 --no-build
python verify.py part3 --no-build
python verify.py part4 --no-build
python verify.py all --no-build
```

定点排查时用：

```bash
python test_chapter0_1_notebooks.py
python test_notebook_answers.py path/to/your.ipynb --mode both
```

## 最小规则

- Part 0 / Part 1：优先在线 Notebook 或本地基础环境。
- Part 2：默认 CPU-first，少数题再切 GPU。
- Part 3 / Part 4：完整验收需要 GPU；没有 GPU 时先阅读。
- CNB 的目标是统一交付，不是新增一套内容分层。
