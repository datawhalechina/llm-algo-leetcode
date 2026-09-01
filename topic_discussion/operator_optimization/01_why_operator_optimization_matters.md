# 为什么需要算子优化

算子优化关注单个数学算子如何映射到 GPU kernel，以及局部加速何时能够传递到模型和服务层。图变换、IR 和 lowering 的编译器决策见[编译与图优化](../compiler_graph_optimization/intro.md)。

核心判断：先证明语义和数值正确，再比较相同 shape、dtype 和 workload 下的 kernel 与端到端成本。
