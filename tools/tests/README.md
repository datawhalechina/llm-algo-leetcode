# 回归测试

这里存放长期维护的工具与机制实现回归测试。面向学习者的 Notebook 验证命令暂时保留原入口，不因文件名以 `test_` 开头就一并迁入。

从仓库根目录运行回归测试；训练语义测试需要 CPU PyTorch，缺少时明确跳过：

```bash
python -m unittest discover -s tools/tests -p 'test_*.py'
```

| 测试 | 核对内容 | 依赖 |
|:---|:---|:---|
| `test_foundation_performance_evidence.py` | 基础页的资源计算、时间线依赖、比较条件与编译计划检查 | Python 标准库与仓库源 Notebook；不执行 GPU 或 Profiler 单元 |
| `test_foundation_hardware_constraints.py` | 设备声明、阶段独立需求、隔离指标、成本分母与功能测试前提 | Python 标准库；读取源 Notebook 中的实际定义，不探测或配置设备 |
| `test_foundation_training_semantics.py` | ReLU 零点梯度、非持久缓冲区、快照与续跑、早停、跳过更新、评价聚合和 Attention 边界 | CPU PyTorch；读取源 Notebook 定义，不安装依赖或执行 GPU 实验 |
| `test_test_entrypoints.py` | 新旧脚本命令、测试参数、跨目录运行、重复收集和断言失败退出码 | Python 标准库；通过子进程运行基础页回归测试，不修改源文件 |
