# Kernel 语义与访存

Kernel 优化先处理边界、mask、dtype 和读写语义，再讨论 tile、layout、shared memory 和 occupancy。CPU 实验可以验证结果，不能替代 GPU 的带宽、缓存和执行时间测量。

判断时同时记录输入 shape、数据类型、基线实现、候选实现和正确性误差。
