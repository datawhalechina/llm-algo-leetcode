# 性能分析公式索引与符号约定

本页汇总性能分析中反复使用的测量、时间、驻留、资源参考与回归公式。先核对测量范围、单位和成立条件，再沿表中入口阅读机制解释；具体采集过程与推导保留在正文和机制练习中。术语含义见[术语表](./glossary.md)，正式结果与原始产物见[实验资产表](./experiment_assets.md)。

## 共用符号与单位

同一公式中的时间必须来自可比较的时钟和测量范围。硬件理论上界、经验参考与实际观测分别记录；基线和候选只在预先声明的实验变量上不同。

| 符号 | 含义 | 单位或范围 |
|:---|:---|:---|
| $T$ | 明确测量范围内的总耗时 | 秒；由毫秒换算时除以 1000 |
| $N$ | 测量窗口内完成的工作数量 | 样本、请求或词元，必须指定一种 |
| $T_b,T_c$ | 同一范围的基线与候选耗时 | 使用相同时间单位和统计方法 |
| $I_i=[s_i,e_i)$ | 事件 $i$ 在共同时间轴上的半开区间 | $e_i\ge s_i$ |
| $d_i$ | 事件 $i$ 的持续时间 | $d_i=e_i-s_i$ |
| $b_i,a_i,f_i$ | 存储 $i$ 的字节数、分配时刻与释放时刻 | byte；$a_i\le f_i$ |
| $A(t),R(t)$ | 同一时刻分配器的活跃分配量与保留量 | byte；对应 `allocated` 与 `reserved` |
| $F,B$ | 所选计算范围的浮点运算次数与指定存储层级读写字节数 | FLOP、byte；$B>0$ |
| $P_{\max},W_{\max}$ | 匹配精度、计算路径和存储层级的理论吞吐与带宽上界 | FLOP/s、byte/s |
| $P_{\mathrm{ref}},W_{\mathrm{ref}}$ | 由指定微基准测得的参考吞吐与参考带宽 | FLOP/s、byte/s；不是严格上界 |
| $n$ | 有效配对样本数 | 正整数；不包含失败配对 |
| $\delta,\epsilon$ | 最大允许相对延迟退化、要求的最小相对延迟降幅 | $\delta\ge0$，$0<\epsilon<1$ |

## 测量窗口、吞吐与延迟

吞吐的工作数量和时间窗口要一起定义；比较延迟时，加速比与延迟降幅使用不同分母。请求级指标还要区分首个词元、末个词元与请求完成时刻。

| 对象 | 规范表达 | 条件与解释 | 机制入口 |
|:---|:---|:---|:---|
| 窗口吞吐 | $Q=N/T$，$T>0$ | 单位为指定工作单位每秒；排队是否计入、失败与未完成数量另行记录 | [03 时间窗口汇总](./03_time_breakdown_and_trace_reading.md#同一时间窗口怎样汇总) |
| 加速比 | $S=T_b/T_c$，$T_b,T_c>0$ | $S>1$ 表示候选更快；固定比较的是均值、中位数还是配对统计量 | [03 时间窗口汇总](./03_time_breakdown_and_trace_reading.md#同一时间窗口怎样汇总) |
| 相对延迟降幅 | $d=1-T_c/T_b$ | $d>0$ 表示延迟下降；$S-1$ 与 $d$ 不是同一个比例 | [06 配对比较](./06_benchmark_design_and_regression_validation.md) |
| 首词元延迟 | $\mathrm{TTFT}=t_1-t_0$ | $t_0$ 为请求计时起点，$t_1$ 为首个输出词元到达时刻；明确是否包含网络、排队和输入处理 | [03 场景与阶段](./03_time_breakdown_and_trace_reading.md#建立可比较的时间基线) |
| 平均单输出词元间隔 | $\mathrm{TPOT}=(t_C-t_1)/(C-1)$，$C\ge2$ | $C$ 为输出词元数，$t_C$ 为末个输出词元到达时刻；流式分块不一定与词元一一对应，$C<2$ 时该式无定义 | [03 场景与阶段](./03_time_breakdown_and_trace_reading.md#建立可比较的时间基线) |
| 端到端延迟 | $T_{\mathrm{e2e}}=t_{\mathrm{end}}-t_0$ | $t_{\mathrm{end}}$ 为约定的请求完成时刻，可能晚于末个词元到达；与首词元延迟分别报告 | [01 测量问题](./01_why_profiling_matters.md) |

## 时间重叠与关键路径

事件持续时间之和会重复统计重叠区间；区间并集只描述活动覆盖时间。完成时间还取决于依赖、资源顺序和等待是否完整记录。

| 对象 | 规范表达 | 条件与解释 | 机制入口 |
|:---|:---|:---|:---|
| 活动覆盖时间 | $T_{\mathrm{busy}}=\operatorname{len}(\bigcup_i I_i)\le\sum_i d_i$ | `len` 表示区间并集的总长度；活动之间的空洞不计入，嵌套的主机与设备事件分层分析 | [03 重叠区间](./03_time_breakdown_and_trace_reading.md#同一时间窗口怎样汇总) |
| 依赖时间下界 | $T_{\mathrm{dep}}=\max_{p\in\mathcal P}\sum_{i\in p}d_i$，$T_{\mathrm{obs}}\ge T_{\mathrm{dep}}$ | $\mathcal P$ 为开始到结束的依赖路径集合，$p$ 为其中一条路径；事件图无环，时间与范围一致 | [Part 02 · PERF-TRACE](../../02_PyTorch_Algorithms/PERF-TRACE_Trace_Dependencies_and_Critical_Path.ipynb) |
| 完成时间重建 | $T_{\mathrm{obs}}=T_{\mathrm{dep}}$ | 仅当所需启动、等待、资源串行等约束均已表达且持续时间准确时成立；缺少依赖不能据此宣称重建成功 | [03 依赖与观测](./03_time_breakdown_and_trace_reading.md#同一时间窗口怎样汇总) |

## 对象驻留与显存峰值

先给独立存储确定创建与释放区间，再计算同一时刻的总量。共享存储的视图只计一次；张量账本、分配器统计和整个设备占用属于不同测量范围。

| 对象 | 规范表达 | 条件与解释 | 机制入口 |
|:---|:---|:---|:---|
| 驻留量 | $M(t)=\sum_i b_i\mathbf1(a_i\le t<f_i)$ | $\mathbf1$ 为指示函数；存储大小在各区间内固定，分配与释放事件采用一致顺序 | [04 驻留区间](./04_memory_timeline_and_residency.md#用驻留区间解释峰值) |
| 驻留峰值 | $M_{\mathrm{peak}}=\max_t M(t)$ | 峰值可持续一段时间；不是各对象在不同时刻的独立峰值之和 | [Part 02 · PERF-MEMORY-TIMELINE](../../02_PyTorch_Algorithms/PERF-MEMORY-TIMELINE_Memory_Residency_and_Peak_Overlap.ipynb) |
| 同时刻保留差值 | $D(t)=R(t)-A(t)$ | 表示分配器持有而未被活跃分配使用的空间；不等于碎片量，也不等于 $\max_tR(t)-\max_tA(t)$ | [04 分配器观测](./04_memory_timeline_and_residency.md#先定位峰值与生命周期) |

## 算术强度与资源参考

先选定算子、精度和存储层级，再比较计算与搬运。理论 Roofline 是理想上界模型；微基准测得的参考值用于定位差距，不能作为不可超越的硬件极限。

| 对象 | 规范表达 | 条件与解释 | 机制入口 |
|:---|:---|:---|:---|
| 算术强度 | $I=F/B$ | 单位 FLOP/byte；最低搬运量估计与实际测量流量分别标明 | [05 算术强度](./05_bottleneck_attribution.md#用算术强度比较计算与带宽约束) |
| 理想吞吐上界 | $P\le\min(P_{\max},I W_{\max})$ | $P$ 为所选范围的实际计算吞吐，单位 FLOP/s；各项匹配同一计算路径与存储层级 | [Part 02 · PERF-ROOFLINE](../../02_PyTorch_Algorithms/PERF-ROOFLINE_Arithmetic_Intensity_and_Resource_Ceiling.ipynb) |
| 理想耗时下界 | $T\ge\max(F/P_{\max},B/W_{\max})$ | 启动、同步、排队和不能重叠的阶段会使实际耗时更长；局部下界不直接解释端到端瓶颈 | [05 算术强度](./05_bottleneck_attribution.md#用算术强度比较计算与带宽约束) |
| 理论机器平衡点 | $I^*=P_{\max}/W_{\max}$ | 单位 FLOP/byte；在理想模型中区分计算侧与带宽侧约束 | [性能分析术语表](./glossary.md) |
| 相对经验参考 | $P_{\mathrm{obs}}=F/T_{\mathrm{kernel}}$，$\rho=P_{\mathrm{obs}}/P_{\mathrm{ref}}$ | $T_{\mathrm{kernel}}>0$ 为匹配范围的设备计时；形状与路径不同可使 $\rho>1$，不登记为硬件效率超过 100% | [05 经验参考](./05_bottleneck_attribution.md#用算术强度比较计算与带宽约束) |

## 通信与独立计算的重叠

通信已经就绪、与独立计算同时开始，且两者均需在下一消费点之前完成时，可以估计该消费点多等了多久。复杂依赖、发起延迟和资源争用仍要通过实际时间线检查。

| 对象 | 规范表达 | 条件与解释 | 机制入口 |
|:---|:---|:---|:---|
| 简化暴露等待 | $T_{\mathrm{wait}}=\max(0,T_{\mathrm{comm}}-T_{\mathrm{ind}})$ | $T_{\mathrm{comm}}$ 为通信耗时，$T_{\mathrm{ind}}$ 为独立计算窗口；三者单位一致。公式不自动适用于所有通信调度 | [05 通信重叠](./05_bottleneck_attribution.md#通信何时延长完成时间) |

## 配对回归、收益与失败分母

延迟类指标越小越好。每对样本先计算候选相对基线的比值，再按完整样本对估计区间；失败配对不进入正常延迟样本，但失败运行必须保留在各自的失败率分母中。

| 对象 | 规范表达 | 条件与解释 | 机制入口 |
|:---|:---|:---|:---|
| 配对比值 | $r_i=c_i/b_i$，$\widehat r=\operatorname{median}_{i=1,\ldots,n}(r_i)$ | $b_i,c_i>0$ 为第 $i$ 对基线、候选耗时；配对比值中位数与两组中位数之比分别报告 | [06 配对比值](./06_benchmark_design_and_regression_validation.md#配对比值怎样变成回归判断) |
| 区间估计 | $[L_r,U_r]$ 为 $\widehat r$ 的置信区间 | 记录样本数、置信水平和重采样方法；样本相关时采用匹配的分块方案，不把区间当作单次观测范围 | [Part 02 · PERF-REGRESSION](../../02_PyTorch_Algorithms/PERF-REGRESSION_Paired_Benchmark_and_Regression_Thresholds.ipynb) |
| 延迟回归通过 | $U_r\le1+\delta$ | 以质量、失败率和最小配对数通过检查为前提；通过回归检查仍可能略慢于基线 | [06 回归门槛](./06_benchmark_design_and_regression_validation.md#配对比值怎样变成回归判断) |
| 明确延迟退化 | $L_r>1+\delta$ | 延迟退化超过预先允许的范围 | [06 回归门槛](./06_benchmark_design_and_regression_validation.md#配对比值怎样变成回归判断) |
| 回归结论待定 | $L_r\le1+\delta<U_r$ | 区间跨越门槛，补充配对样本或控制噪声 | [06 回归门槛](./06_benchmark_design_and_regression_validation.md#配对比值怎样变成回归判断) |
| 达到最小延迟收益 | $U_r\le1-\epsilon$ | 在固定工作负载和置信方法下支持要求的改善；节省显存等其他目标另设收益与时间代价门槛 | [06 收益与回归的区别](./06_benchmark_design_and_regression_validation.md#结果如何进入下一步) |
| 失败率 | $p_{\mathrm{fail}}=K_{\mathrm{fail}}/K_{\mathrm{all}}$，$K_{\mathrm{all}}>0$ | $K_{\mathrm{all}}$ 为按预先定义规则计入的全部正式运行数，$K_{\mathrm{fail}}$ 为失败数；基线、候选分别统计，不删除 OOM、超时或质量失败 | [06 配对与失败记录](./06_benchmark_design_and_regression_validation.md#为什么要配对、记录顺序和失败) |

## 需要其他公式时查哪里

损失、通信算法、量化误差与成本模型各有对应的机制解释。完成测量和回归判断后，再按实际问题选择入口。

| 需要解释的对象 | 继续阅读 |
|:---|:---|
| 梯度、有效更新、训练状态与重算搬运 | [反向传播与训练机制公式索引](../backpropagation_training_mechanism/formula_reference.md) |
| 集合通信、状态切分与扩展效率 | [通信与并行](../communication_parallelism/intro.md) |
| 量化缩放、校准误差与低比特执行 | [量化与压缩](../quantization_compression/intro.md) |
| 候选资源取舍、组合收益与成本判断 | [性能优化](../performance_optimization/intro.md) |
