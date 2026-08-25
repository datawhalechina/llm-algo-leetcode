# 12. TensorCore and Mixed Precision | Tensor Core 与混合精度

**难度：** Medium | **环境：** CPU-first | **标签：** `硬件系统`, `Tensor Core`, `混合精度` | **目标人群：** 硬件约束学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

这一页把 Part 1 的数据格式和显存直觉，推进到混合精度、Tensor Core 和吞吐判断上。

这一页在整个教程的纵向主线里属于 `Part 01` 的训练吞吐与精度基础页，优先服务 `监督微调路线` 的训练控制和低显存微调判断。学完这里，后面再看 `13 / 60 / 65` 时，你会更容易读懂 `BF16 / AMP / Tensor Core` 为什么会同时影响训练吞吐、训练稳定性和低显存微调选择；如果这里没学明白，后面容易只记住“Tensor Core 很快”，却说不清吞吐、数值稳定性和显存取舍之间的关系。按专题归类，这一页主要属于 `监督微调路线` 的训练前置，也和 `量化与压缩专题`、`Profiling 专题` 共享部分判断视角。本节是 **CPU-first；GPU 用于扩展验证**：CPU 练习可以验证 dtype 转换、AMP 配置分支和数值误差示例的基本关系；真实 GPU 才能验证 Tensor Core 是否被使用、吞吐是否提升、峰值显存如何变化，以及不同驱动和 GPU 架构上的兼容性。显存路线在这里重点观察 dtype、带宽和训练状态大小，不把 CPU 结果当成 73 / 76 的实测结论。

**关键词：** `FP16`, `BF16`, `Tensor Core`

对应显存优化路线的 Task1（性能认知）和 Task5（量化前的精度判断），也支撑训练微调路线的训练控制。需要实际训练吞吐或峰值显存时进入 73；若比较 checkpoint / offload，则使用 76，不能用本节 CPU 结果替代。

---

## 前置阅读

**导语：** 先把数据格式和显存账本对齐，再去看 Tensor Core 和混合精度会更顺；如果你正在走 `监督微调路线`，这里会直接服务 `03 / 04 / 65`，因为后面训练吞吐为什么会上来、数值为什么更稳、低显存微调为什么偏向某种精度口径，本质上都先靠这一页打底。

- [Group 1A: Numerical Foundations and Scale Estimation | 1A: 数值基础与算力估算](./1A.md)
- [Group 1B: Single-GPU Hardware and Memory Optimization | 1B: 单卡硬件与访存优化](./1B.md)

## 相关阅读

**导语：** 把 mixed precision 放回训练控制、低显存微调和量化链路里看，判断会更稳，也更容易分清哪些收益来自 Tensor Core 吞吐，哪些收益来自更小的显存账本。

- [26. QLoRA and 4bit Quantization | QLoRA 与 4-bit 量化](../02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization.md)
- [13. End-to-End Fine-Tuning Experiment | 端到端微调实验](../02_PyTorch_Algorithms/13_End_to_End_Fine_Tuning_Experiment.md)
- [65. QLoRA Selection Project | QLoRA 选型项目](../02_PyTorch_Algorithms/65_QLoRA_Selection_Project.md)

## Q1：Tensor Core 到底是什么，为什么它比普通 CUDA Core 更适合矩阵计算？

<details>
<summary>点击展开查看解析</summary>

Tensor Core 不是“更快的标量算术单元”，而是专门为矩阵乘加设计的硬件路径。普通 CUDA Core 更像是按元素执行标量 FMA，而 Tensor Core 会把一小块矩阵乘加打包成一次 MMA（Matrix Multiply-Accumulate）完成。

这件事的重要性在于：大模型里最贵的计算几乎都来自 GEMM，也就是矩阵乘法。如果计算单元一次能处理更多乘加，且数据复用路径更短，那么同样的时钟预算就能完成更多工作。

混合精度和 Tensor Core 的关系也在这里：低精度输入可以让乘法吞吐更高，而高精度累加器保住结果稳定性。也就是说，Tensor Core 不是单独在“提速”，而是在用更合适的数据组织方式把吞吐做上去。
</details>
### Q1小验证：矩阵计算为什么更适合打包执行

把标量 FMA 和块状 MMA 的思路对比一下，先记住“打包”带来的吞吐收益。

```python
def gemm_flops(m, n, k):
    """计算 dense GEMM 的乘加 FLOPs 数量；不代表硬件实际耗时。"""
    if min(m, n, k) <= 0:
        raise ValueError('m、n、k 必须为正数')
    return 2 * m * n * k

# 一个 1024x1024 的矩阵乘法
m = n = k = 1024
flops = gemm_flops(m, n, k)
print(f'GEMM FLOPs: {flops / 1e9:.2f} GFLOPs')
print('Tensor Core 的意义不是改变 FLOPs 数量，而是提高单位时间可完成的矩阵乘加密度。')
```

## Q2：FP16、BF16、FP32 的差别在哪里，为什么混合精度不会简单等于“越低越差”？

<details>
<summary>点击展开查看解析</summary>

精度选择要同时看两个维度：表示范围和尾数精度。

- FP32 范围大、精度高，但存储和传输成本也高。
- FP16 更省空间，适合大量乘法输入，但动态范围更窄。
- BF16 保留了接近 FP32 的指数范围，更适合训练和某些数值敏感场景。

混合精度的核心做法不是把所有东西都压到低精度，而是让“适合低精度的部分”走低精度路径，让“容易数值不稳的部分”继续保留较高精度。这样既能降低带宽和显存压力，又尽量不牺牲训练/推理稳定性。

所以，混合精度不是“牺牲精度换速度”的粗暴做法，而是在计算图里分配不同的数据类型，让吞吐和稳定性同时达到可接受水平。
</details>
### Q2小验证：不同精度的显存占用差多少？

同样一个张量，只改 dtype，就能直观看到显存和带宽压力的变化。

```python
def tensor_storage_bytes(numel, dtype_bytes):
    """计算张量理论存储字节数，不包含 allocator 和临时 workspace。"""
    if numel < 0 or dtype_bytes <= 0:
        raise ValueError('numel 不能为负数，dtype_bytes 必须为正数')
    return numel * dtype_bytes

shape = (4096, 4096)
numel = shape[0] * shape[1]
for name, bytes_per_elem in [('FP32', 4), ('BF16/FP16', 2), ('FP8', 1)]:
    size_mb = tensor_storage_bytes(numel, bytes_per_elem) / 1024 / 1024
    print(f'{name:8s}: {size_mb:8.2f} MB')

assert tensor_storage_bytes(1024, 4) == 4096
assert tensor_storage_bytes(1024, 2) == 2048
print('✅ dtype 存储量计算通过；实际 GPU peak memory 仍需单独测量')
```

## Q3：精度选择为什么会同时影响内存、吞吐和量化路径？

<details>
<summary>点击展开查看解析</summary>

精度不是单纯的数值选择，它会同时改写三个成本：

1. **内存成本**：每个元素占多少字节，决定了模型参数、激活值和 KV cache 的体积。
2. **传输成本**：同样的总字节数，搬运时间会直接影响带宽瓶颈是否明显。
3. **计算路径成本**：某些硬件路径对特定低精度格式有专门加速，Tensor Core 就是典型例子。

这也是为什么量化、推理加速和吞吐比较经常绑在一起讨论。量化不只是“把数值变小”，而是在重塑模型执行时的内存、带宽和计算路径。

因此，看精度问题时，不能只问“还能不能算对”，还要问“这条路径是不是更省内存、更少搬运、也更容易跑满硬件”。
</details>
### Q3小验证：字节数如何影响模型体积

把参数量固定，看看不同 dtype 对模型大小的直接影响。

```python
params = 7_000_000_000
for name, bytes_per_elem in [('FP32', 4), ('BF16/FP16', 2), ('INT8', 1)]:
    size_gb = tensor_storage_bytes(params, bytes_per_elem) / 1e9
    print(f'{name:8s}: {size_gb:6.2f} GB')
```

## ⚠️ 常见误区

- `Tensor Core` 不是所有算子都能直接吃满，收益主要来自大块矩阵乘法。
- `BF16` 不等于比 `FP16` 更快，它更多是在数值稳定性和可用范围上更友好。
- 混合精度不是把所有地方都降精度，而是按路径分配精度。
- 量化不只是省显存，它还会影响带宽、吞吐和实现复杂度。