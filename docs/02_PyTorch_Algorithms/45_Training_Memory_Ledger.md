# 45. Training Memory Ledger | 训练显存账本

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/45_Training_Memory_Ledger.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


**状态：** 内容已迁移

本页原有的显存裁剪规划内容已经迁移到[性能优化专题的训练显存账本](../topic_discussion/performance_optimization/03_training_memory_ledger.md)，并由 [Part 02 · 75 显存预算压缩项目](./75_Memory_Budget_Compression_Project.md) 承接实测和决策证据。

迁移后的主线是：识别峰值来源 → 设定显存预算 → 排序 checkpoint、offload、累积和分片等候选动作 → 比较显存收益、时间代价和质量影响 → 输出 accept / tune / reject 决策。

请转到性能优化专题阅读完整机制；本占位页保留用于兼容历史链接。