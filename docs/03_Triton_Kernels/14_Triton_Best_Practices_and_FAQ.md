
# 14. Triton Best Practices and FAQ | 算子开发实战避坑指南

**难度：** Hard | **标签：** `Triton`, `End-to-End Project`, `LLaMA-3`, `Integration` | **目标人群：** 核心 Infra 与算子开发

在工业界使用 Triton 开发加速算子时，写出数学公式往往只占工作量的 20%，剩下的 80% 时间都花在与显存越界 (Segmentation Fault)、数值不对 (Race Condition) 和死锁 (Deadlock) 作斗争上。

本手册总结了开发 Triton 算子时最高频的报错案例、根本原因及最佳实践。

## 本节定位

这不是一节新的实战课，而是 Part 3 的参考手册。
你可以把它当作：

- 遇到 `Segmentation Fault` 时的排障清单
- 遇到性能不达预期时的调优备忘
- 做 13 节工程集成前后的 FAQ

如果你在学习 01-13 节时遇到问题，先回到这里查常见错误模式，再回到正文继续动手。

---

## 1. 致命报错：`Segmentation Fault (core dumped)`

当你运行包含 Triton 算子的 Python 脚本时，如果整个进程直接崩溃，不报任何 Python 异常栈，只留下一句 `Segmentation Fault`，这通常是**显存越界访问**。

### 常见原因 A：忘记乘以 Stride (步长)
**现象：** 试图读取二维矩阵的某一行，但读取了错误的物理内存地址，超出了分配的显存范围。
**错误代码：**
```python
row_idx = tl.program_id(0)
# 错误：二维张量在物理内存中是一维展开的。只加上行号意味着你读取的是第 0 行的前几个元素，而不是第 i 行。
row_start_ptr = ptr + row_idx 
```
**正确解法：**
必须传入每一维度的步长（通过 `tensor.stride(0)` 获取）。
```python
row_start_ptr = ptr + row_idx * stride_row
```

### 常见原因 B：Mask (掩码) 缺失或计算错误
**现象：** 数据总量 `N` 不是 `BLOCK_SIZE` 的整数倍。最后一个 Program 块在计算偏移量时，超出了 `N` 的范围。
**错误代码：**
```python
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
# 错误：当 pid 为最后一个块且 N 不被 BLOCK_SIZE 整除时，offsets 的末尾会超出 N。
x = tl.load(ptr + offsets) 
```
**正确解法：**
必须使用掩码 `mask` 保护所有的 `load` 和 `store`。
```python
mask = offsets < N
x = tl.load(ptr + offsets, mask=mask)
```

---

## 2. 隐式错误：输出结果存在随机的“脏数据”或 NaN

有时程序不会崩溃，但每次运行输出的结果都不一样，或者突然出现极大/极小值。

### 常见原因 A：`tl.load` 越界时没有提供 `other` 默认值
**现象：** 当 `mask` 为 `False` 时，Triton 默认读取到的显存值是**未定义**的（可能是上次计算留下的垃圾数据）。如果后续你要对这些数据求和 (`tl.sum`) 或求最大值 (`tl.max`)，脏数据会毁掉整个结果。
**错误代码：**
```python
x = tl.load(ptr + offsets, mask=mask)
# 如果 x[10] 越界了，x[10] 可能是 999999.0
local_max = tl.max(x) # 结果变成了 999999.0
```
**正确解法：**
根据后续的归约操作，提供合适的 `other` 参数。
- 如果后接 `tl.max`，补负无穷：`other=-float('inf')`
- 如果后接 `tl.sum`，补零：`other=0.0`
- 如果后接 `tl.min`，补正无穷：`other=float('inf')`
- 如果后接 `tl.dot`，越界元素也应贡献 `0.0`
```python
x = tl.load(ptr + offsets, mask=mask, other=-float('inf'))
```

### 常见原因 B：多个 Program 写入同一个地址（竞态条件）
**现象：** 当你试图实现类似直方图统计、或 Scatter 操作时，多个线程块同时向同一个全局显存指针 `tl.store`。
**解法：** Triton 默认的 `tl.store` 是非原子的，后写入的会直接覆盖先写入的。如果必须跨 Program 累加，必须使用 `tl.atomic_add(ptr, val)`。

---

## 3. 性能不达预期：比 PyTorch 原生还慢？

如果你成功跑通了算子，但在 `triton.testing.perf_report` 中发现带宽吞吐极低。

### 常见原因 A：BLOCK_SIZE 不是 2 的幂
**现象：** 如果你把 `BLOCK_SIZE` 设为 100 这种奇怪的数字，Triton 编译器生成的 PTX 汇编代码效率极差，无法充分利用 GPU 的向量化加载指令（如 `LDG.E.128`）。
**最佳实践：**
- `BLOCK_SIZE` 强烈建议设为 128, 256, 512, 1024。
- 如果你的特征维度（如 Head Dim）是 64，可以通过 `@triton.jit` 的参数注解强行指定编译期常量，如 `BLOCK_D: tl.constexpr`，并在外面传入比它大的最小 2 的幂：`triton.next_power_of_2(D)`。

### 常见原因 B：没有使用 `@triton.autotune`
**现象：** `num_warps` (线程束数量) 默认可能是 4（代表 4*32=128 个线程）。对于需要巨大 SRAM 和高计算密度的算子（如 Flash Attention 的块级矩阵乘），如果 `num_warps` 设少了，寄存器会溢出到显存（Register Spilling）；如果设多了，活跃线程块变少，掩盖延迟的能力下降。
**最佳实践：**
永远不要硬编码 `num_warps`。在内核函数上方加上自动调优装饰器，让编译器在运行时通过预热 (Warmup) 去寻找最优组合。
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
    ],
    key=['seq_len', 'head_dim'],
)
```

### 常见性能诊断方法

- 先用 `triton.testing.do_bench` 观察单个 kernel 的 baseline 时间，再和 PyTorch 参考实现对比
- 如果怀疑 `num_stages` 不合适，优先看流水线是否被填满，而不是只盯着平均时延
- 如果性能突然掉得很厉害，可以先怀疑 Register Spilling，再看 tile 是否过大、`num_warps` 是否过高
- 如果要看更细的硬件层指标，再配合 Nsight Compute 观察 HBM 吞吐、寄存器使用率和访存合并情况

---

## 4. Debug 终极武器

当逻辑极其复杂（例如 PagedAttention 的四维指针偏移查表），你完全不知道算子内部到底读取了什么数据时：

### 武器 1：`TRITON_INTERPRET=1` (强制解释执行)
不要让 Triton 编译成 GPU 汇编执行。在终端运行脚本前加上环境变量：
```bash
TRITON_INTERPRET=1 python my_script.py
```
这会在 CPU 上用纯 Python 逐行解释你的算子。**一旦发生指针越界，它会立刻抛出带有具体 Python 行号的异常栈（Traceback）！** 这比 GPU 上的黑盒 `Segmentation Fault` 好用一万倍。

### 武器 2：`tl.device_print`
想知道 `offsets` 算得对不对？直接在算子内部打印。
```python
if pid == 0:  # 强烈建议只在一个 block 里打印，否则终端会瞬间被几万行日志淹没
    tl.device_print("My Offsets:", offsets)
```
注意：
- `device_print` 只能打印 `tl.tensor`，不能打印纯 Python 标量
- 大张量会输出很多行，建议只打印少量元素或在 `pid == 0` 时打印
- 它的开销很大，只适合调试，不适合留在生产代码里

---

## 综合诊断练习

下面给出一个故意带有多个错误的 Triton 片段。请你用本节的方法定位并修复：

1. `stride` 计算错误
2. `mask` / `other` 处理不完整
3. `BLOCK_SIZE` 选得不合理

```python
# 练习：这个 kernel 故意写错了
@triton.jit
def broken_kernel(x_ptr, y_ptr, stride_x, stride_y, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # TODO: 修复 stride 与 other
    x = tl.load(x_ptr + offsets, mask=mask)
    y = x + 1.0
    tl.store(y_ptr + offsets, y, mask=mask)
```

**练习要求：**
- 先用 `TRITON_INTERPRET=1` 看具体是哪一步出错
- 再用 `tl.device_print` 打印 `offsets` 或局部输入
- 最后把 `BLOCK_SIZE` 改成更合适的 2 的幂

> 目标不是写新算子，而是把本节的调试方法真正用起来。
