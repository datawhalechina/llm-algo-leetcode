# CUDA 执行与硬件约束

Warp、block、shared memory、Tensor Core、stream 和异步搬运共同决定 kernel 的实际执行方式。一个实现是否更适合某张 GPU，不能只由 FLOPs 或源代码长度判断。

GPU 实验应区分 kernel microbenchmark、编译时间和端到端 workload。
