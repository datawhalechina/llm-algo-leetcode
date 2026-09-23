# 04 执行模型与 Backend 约束

## 页面目标

这一页负责解释 backend 约束如何反过来塑造图优化结果，以及为什么执行模型必须进到编译讨论里。

本页的输出是 backend 约束清单：明确 layout、launch、kernel 支持和 runtime 行为如何限制图级与执行级选择。

## 问题起点

很多图级判断在 backend 级别会遇到现实约束：

- block / warp / program shape
- layout 选择
- stream / launch 组织
- kernel 粒度和资源占用

## 关键观察点

- CUDA / Triton execution model
- layout-sensitive kernel structure
- launch granularity
- backend-specific constraints

## 进入下一页

进入 [05 Backend 成本模型与最优解分化](./05_backend_cost_models_and_divergent_optima.md)，解释同一张图为什么会在不同 backend 上出现不同最优解。

## 可视化入口

> 正文暂不嵌入未审核图示；相关图册与占位说明见 [视觉资产页](./07_visual_assets.md)。

## 对应 Part

- `08 Programming Models CUDA Triton`
- `15 CUDA Execution Model`
- `18 Triton Block Model`
- `29 CUDA Stream Advanced Scheduling`

## 本节要点

backend 约束不是图优化之后才补看的细节，而是会提前决定哪些图级选择真正可用。
