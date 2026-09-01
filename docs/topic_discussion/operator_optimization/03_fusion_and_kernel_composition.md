# 算子融合与 Kernel 组合

Fusion 的价值来自减少中间张量写回和重复读取，但也可能增加寄存器压力、降低 occupancy 或改变并行度。节点减少不是性能结论，必须用 kernel 和端到端 benchmark 验证。

Attention、Norm、MLP 等模型组件的结构定义应回到模型架构或推理专题，本页只讨论它们如何落成 kernel 组合。
