# 算子优化公式索引与符号约定

本页汇总时间比较、分块寻址、注意力状态与资源约束中常用的公式。先确认比较对象、单位和成立条件，再沿入口阅读推导；公式中的构造例子不代表 GPU 实测。概念辨析见[术语表](./glossary.md)，正式结果见[实验资产表](./experiment_assets.md)。每节的符号在该节内定义。

## 时间比较与准备成本

$T_b,T_c>0$ 是相同工作范围的基线、候选耗时；统计方法必须一致。$b_i,c_i$ 是第 $i$ 轮配对的两个耗时，$k$ 为有效配对数。时间可以用秒或毫秒，但同一公式内不能混用。

| 对象 | 规范表达 | 条件与解释 | 推导入口 |
|:---|:---|:---|:---|
| 加速比 | $S=T_b/T_c$ | 大于 1 表示候选更快；注明比较的是单次、均值或中位数 | [01 加速比与降幅](./01_optimization_targets_and_baselines.md#加速比与耗时降幅怎样计算) |
| 耗时相对降幅 | $d=1-T_c/T_b$ | 10 ms 降到 8 ms 时，$S=1.25$、$d=20\%$ | [01 加速比与降幅](./01_optimization_targets_and_baselines.md#加速比与耗时降幅怎样计算) |
| 配对加速比 | $s_i=b_i/c_i$，$\widehat s=\operatorname{median}_{i=1,\ldots,k}s_i$ | 先按轮计算，再汇总；与两组耗时中位数之比分开报告 | [05 多轮配对](./05_measurement_profiling_and_autotuning.md#一次最快值为什么不足以选配置) |
| 至少持平的复用次数 | $n_{\mathrm{break}}=\left\lceil\Delta C/(T_b-T_c)\right\rceil$ | $\Delta C>0$ 为新增一次性准备成本，且 $T_b>T_c$；等号只表示持平，稳态不更快时此式不适用 | [05 准备成本](./05_measurement_profiling_and_autotuning.md#准备成本需要多少次调用摊销) |

摊销模型比较的是“准备成本＋$n$ 次相同调用”；重新编译、重新捕获或每次新增的布局适配需要按实际发生频率计入，不能当作永远只付一次的成本。

## 地址、逻辑访存与任务分配

地址偏移以元素数为单位，换算成字节时再乘元素字节数。GEMM 的输出为 $M\times N$，输出块大小为 $B_M\times B_N$；$T$ 表示输出 tile 总数，$P$ 表示实际启动的 Program 数。

| 对象 | 规范表达 | 条件与解释 | 推导入口 |
|:---|:---|:---|:---|
| 二维视图地址 | $\operatorname{offset}(i,j)=i s_0+j s_1$ | $s_0,s_1$ 为相对视图首元素的步长；不能用逻辑行宽代替任意视图步长 | [03 坐标与地址](./03_kernel_semantics_and_memory.md#逻辑坐标怎样变成实际地址) |
| 输出块总数 | $T=\lceil M/B_M\rceil\lceil N/B_N\rceil$ | $M,N,B_M,B_N$ 为正整数；M/N 尾块仍需有效范围掩码 | [Part 03 · 05 GEMM](../../03_Triton_Kernels/05_Triton_GEMM_Tutorial.md) |
| 循环任务编号 | $t_{p,q}=p+qP<T$ | $0\le p<P$、$q\ge0$；按实际网格步进，合法 tile 归属余数唯一 | [03 输出块分工](./03_kernel_semantics_and_memory.md#输出块怎样分给多个程序) |
| 扁平编号到二维块 | $r=\lfloor t/N_{\mathrm{tile}}\rfloor$，$c=t\bmod N_{\mathrm{tile}}$ | $N_{\mathrm{tile}}=\lceil N/B_N\rceil$；除数是 tile 列数，不是矩阵元素列数 | [Part 03 · 06 持久化 GEMM](../../03_Triton_Kernels/06_Triton_Persistent_GEMM_and_Tile_Scheduling.md) |
| 激活后相乘的逻辑读写 | 分离 $5Eb$，融合 $3Eb$ | $E$ 个元素，每元素 $b$ 字节；中间值只有当前消费者、类型一致且省去独立写回。不是实测 HBM 流量 | [02 中间张量读写](./02_graph_compilation_and_fusion.md#两次调用之间-中间张量经历了什么) |

循环分工仍可由一次 kernel 启动完成；Program 数不是 CPU 启动次数，也不是 SM 数。每个新输出 tile 重新初始化累加器，同一个 tile 的 K 子块之间保留部分和。

## 注意力的块状态与头映射

对一个查询，$s_j$ 是含缩放和掩码含义的分数，$v_j$ 是对应值向量。旧状态为行最大值 $m$、指数和 $\ell$、未归一化的加权累积向量 $a$。下表的合并假定本块至少有一个有效键，求和只遍历有效位置；全空行与初始全空块需要单独处理，不能直接计算 $-\infty-(-\infty)$。

| 对象 | 规范表达 | 条件与解释 | 推导入口 |
|:---|:---|:---|:---|
| 更新最大值与缩放 | $m'=\max(m,\max_j s_j)$，$\alpha=e^{m-m'}$ | 旧状态和本块使用同一减最大值基准 | [04 跨块状态](./04_attention_and_inference_data_paths.md#跨块保留的最大值、指数和与输出累积量) |
| 更新指数和 | $\ell'=\alpha\ell+\sum_j e^{s_j-m'}$ | 旧指数和与新贡献一起累积 | [04 跨块状态](./04_attention_and_inference_data_paths.md#跨块保留的最大值、指数和与输出累积量) |
| 更新加权累积量 | $a'=\alpha a+\sum_j e^{s_j-m'}v_j$ | 不能只缩放分母而遗漏分子 | [04 跨块状态](./04_attention_and_inference_data_paths.md#跨块保留的最大值、指数和与输出累积量) |
| 形成输出 | $o=a'/\ell'$ | $\ell'>0$，对应整个有效键集合 | [04 跨块状态](./04_attention_and_inference_data_paths.md#跨块保留的最大值、指数和与输出累积量) |
| GQA 头映射 | $G=H_q/H_{kv}$，$h_{kv}=\lfloor h_q/G\rfloor$ | $H_q$ 为 Query 头数，$H_{kv}$ 为 KV 头数，前者为后者的整数倍；固定模型头结构，$0\le h_q<H_q$ | [04 头与有效长度](./04_attention_and_inference_data_paths.md#头共享和有效长度怎样约束-kv-读取) |
| 分页位置 | $p=\lfloor t/L\rfloor$，$u=t\bmod L$ | $t$ 为零起始逻辑词元位置，$L>0$ 为页长；物理页还需通过块表查找 | [04 分页与量化](./04_attention_and_inference_data_paths.md#分页与量化分别改变什么) |

量化缓存按实际存储计数：$B_q=B_{\mathrm{values}}+B_{\mathrm{metadata}}+B_{\mathrm{padding}}$。三项分别为量化值、缩放／所用零点、额外布局填充占用的字节；与相同逻辑 K/V 的浮点存储比较，不能只报告值的位宽比例。

## 资源上界与搬运重叠

资源公式描述给定配置的容量约束，时间公式描述理想调度。两者都需要核对假设，再与设备查询或执行测量比较。

| 对象 | 规范表达 | 条件与解释 | 推导入口 |
|:---|:---|:---|:---|
| 共享内存约束 | $B_S=\lfloor S_{\mathrm{SM}}/S_{\mathrm{block}}\rfloor$ | 字节单位一致；每块实际分配量大于零，并已考虑静态、动态、预留与分配粒度。为零时该项不施加限制 | [05 块资源](./05_measurement_profiling_and_autotuning.md#每块资源怎样限制同时驻留的任务) |
| 多资源驻留上界 | $B_{\mathrm{model}}=\min(B_T,B_W,B_S,B_H,\ldots)$ | 各项分别为线程、线程束、共享内存、硬件块数等限制；缺少精确寄存器分配时只是已建模上界 | [Part 04 · 05 资源与驻留](../../04_CUDA_and_System_Optimization/05_CUDA_Block_Resources_and_Residency.md) |
| 预测占用率换算 | $O_{\mathrm{pred}}=B_{\mathrm{pred}}\lceil t/W\rceil/W_{\mathrm{SM,max}}$ | $B_{\mathrm{pred}}$ 为合法配置的预测块数／SM，$t$ 为线程／块，$W$ 为线程束宽度，$W_{\mathrm{SM,max}}$ 为设备每 SM 线程束上限；不是实测活跃率 | [05 块资源](./05_measurement_profiling_and_autotuning.md#每块资源怎样限制同时驻留的任务) |
| 串行搬运与计算 | $T_{\mathrm{serial}}=n(h+g)$ | $n\ge1$ 个独立等长批次，每批搬运 $h$、计算 $g$，使用同一时间单位 | [06 搬运重叠](./06_cuda_execution_and_runtime_constraints.md#下一批搬运与当前批计算的重叠) |
| 两阶段理想流水 | $T_{\mathrm{pipe}}=h+g+(n-1)\max(h,g)$ | 搬运与计算各自串行、彼此可重叠；缓冲足够、依赖成立，忽略争用和额外提交成本 | [06 搬运重叠](./06_cuda_execution_and_runtime_constraints.md#下一批搬运与当前批计算的重叠) |

## 从局部改进估计整体收益

$f$ 是目标算子在原总耗时 $T_0>0$ 中的占比，$s>0$ 是该算子的加速比。以下假定其余阶段耗时不变、原阶段可以按此范围相加，没有新的重叠变化。

| 对象 | 规范表达 | 条件与解释 | 推导入口 |
|:---|:---|:---|:---|
| 无新增成本的整体加速比 | $S=1/[(1-f)+f/s]$ | $0\le f\le1$；例如 $f=0.1,s=2$ 时，整体约 $1.053$ 倍 | [01 局部与整体](./01_optimization_targets_and_baselines.md#为什么局部变快可能只带来很小的整体改善) |
| 加入净新增适配成本 | $T_{\mathrm{new}}=T_0[(1-f)+f/s]+\Delta T$ | $\Delta T$ 为同一范围中净新增且未在算子候选时间中计入的转换、调用或同步耗时，避免重复计算 | [07 布局复制](./07_integration_benchmarking_and_adoption.md#布局复制带来的额外耗时) |

理论估计用来判断量级，采用决定仍使用完整调用、组件与独立负载的实际复测。带宽上界、时间区间和回归区间的进一步计算见[性能分析公式索引](../performance_analysis/formula_reference.md)。
