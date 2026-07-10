# 12. TensorCore and Mixed Precision | Tensor Core 与混合精度

**难度：** Medium | **环境：** CPU-first | **标签：** `Tensor Core`, `Mixed Precision`, `Throughput` | **目标人群：** 精度与吞吐入门者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


这一页把 Part 1 的数据格式和显存直觉，推进到混合精度、Tensor Core 和吞吐判断上。

**关键词：** `FP16`, `BF16`, `Tensor Core`
## 前置阅读

**导语：** 先把数据格式和显存账本对齐，再去看 Tensor Core 和混合精度会更顺。

- [Group 1A: Numerical Foundations and Scale Estimation | 1A: 数值基础与算力估算](./1A.md)
- [Group 1B: Single-GPU Hardware and Memory Optimization | 1B: 单卡硬件与访存优化](./1B.md)

## 相关阅读

**导语：** 把混合精度放回量化和推理链路里看，判断会更稳。

- [25. Quantization W8A16 | W8A16 量化](../02_PyTorch_Algorithms/25_Quantization_W8A16.md)
- [26. QLoRA and 4bit Quantization | QLoRA 与 4-bit 量化](../02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization.md)
- [31. Inference Performance Comparison | 推理性能对比实验](../02_PyTorch_Algorithms/31_Inference_Performance_Comparison.md)

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
shape = (4096, 4096)
numel = shape[0] * shape[1]
for name, bytes_per_elem in [('FP32', 4), ('BF16/FP16', 2), ('FP8', 1)]:
    size_mb = numel * bytes_per_elem / 1024 / 1024
    print(f'{name:8s}: {size_mb:8.2f} MB')
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
    size_gb = params * bytes_per_elem / 1e9
    print(f'{name:8s}: {size_gb:6.2f} GB')
```

## ⚠️ 常见误区

- `Tensor Core` 不是所有算子都能直接吃满，收益主要来自大块矩阵乘法。
- `BF16` 不等于比 `FP16` 更快，它更多是在数值稳定性和可用范围上更友好。
- 混合精度不是把所有地方都降精度，而是按路径分配精度。
- 量化不只是省显存，它还会影响带宽、吞吐和实现复杂度。