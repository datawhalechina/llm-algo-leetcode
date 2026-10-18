# 算子优化实验 Quickstart

本专题复用 Chapter 3 的 `llm_algo_cnb_gpu` 环境。环境定义位于 `cnb/environment-gpu.yml`，GPU 依赖由 `requirements/gpu.txt` 引入；不要再创建同用途的专题私有环境。

## 创建或更新环境

首次创建：

```bash
conda env create -f cnb/environment-gpu.yml
conda activate llm_algo_cnb_gpu
```

环境已存在时，按定义补齐依赖：

```bash
conda activate llm_algo_cnb_gpu
python -m pip install -r requirements/base.txt -r requirements/dev.txt -r requirements/gpu.txt
```

如果镜像站出现文件缺失或长期限速，只对当前命令切换到官方索引，不修改仓库依赖定义：

```bash
PIP_INDEX_URL=https://pypi.org/simple \
python -m pip install -r requirements/base.txt -r requirements/dev.txt -r requirements/gpu.txt
```

## 注册 Notebook 运行时

在仓库根目录执行：

```bash
conda activate llm_algo_cnb_gpu
python tools/register_notebook_runtime.py
python -m ipykernel install --user \
  --name llm_algo_cnb_gpu \
  --display-name "Python (llm_algo_cnb_gpu)"
```

注册后重启 Notebook Kernel，并确认 `sys.executable` 指向该环境。

## 实验前预检

```bash
conda activate llm_algo_cnb_gpu
python -m pip check
bash tools/validate_cnb_gpu.sh
```

`validate_cnb_gpu.sh` 会检查 PyTorch、Triton、CUDA 可见性和设备执行。没有通过预检时，只能运行参考语义与静态检查，不能登记 GPU 性能结论。

## 推荐验证顺序

1. 先运行 PyTorch 参考实现和边界输入，固定数值容差。
2. 再运行 Triton 候选，确认实际命中目标计算内核而非回退路径。
3. 完成预热和多次微基准，保存原始样本而不是只留均值。
4. 使用性能剖析证据解释计算、访存、占用率或启动限制。
5. 最后把候选放回模型工作负载，检查端到端收益与质量守护条件。

具体实验、结果路径和当前证据状态以[实验资产表](./experiment_assets.md)为准。
