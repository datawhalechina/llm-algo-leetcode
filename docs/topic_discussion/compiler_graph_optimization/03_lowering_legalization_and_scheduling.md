# 03 Lowering、Legalization 与 Scheduling

## 页面目标

这一页回答的是：高层图如何经过 legalization、lowering 和 schedule 变成可执行 kernel，以及哪些转换会把图级收益损失掉。

本页的输出是执行候选：明确语义等价之外的 tile、layout、kernel 组织和调度约束。

这一页负责解释为什么 lowering 不是简单翻译，以及 legal / executable / efficient 这三件事为什么不能混成一个判断。

## 问题起点

一个高层图即使语义上合理，也还要回答：

- 能不能合法地下沉到目标表示？
- 下沉之后是否仍然适合调度？
- schedule 是否把图级收益保住了？

## 关键观察点

- legalization
- schedule search / manual schedule
- layout transformation
- codegen 之前的执行形态

## 进入下一页

进入 [04 执行模型与 Backend 约束](./04_execution_model_and_backend_constraints.md)，把 lowering 结果放回具体硬件和 runtime 执行模型。

## 可视化入口

> 正文暂不嵌入未审核图示；相关图册与占位说明见 [视觉资产页](./07_visual_assets.md)。

## 对应 Part

- `32 TVM MLIR Deep Practice`
- `33 TCO and Cost Model`
- `29 CUDA Stream Advanced Scheduling`

## 本节要点

lowering 的关键不是“能翻译”，而是“翻译之后是否仍然划算”。
