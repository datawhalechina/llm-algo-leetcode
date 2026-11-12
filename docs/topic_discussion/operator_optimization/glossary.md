# 算子优化专题术语表

本表解释Part03／04的执行映射、数值、编译、运行时与测量中容易混淆的概念，按问题所在环节查阅即可。代码标识、API 和工具名称保留原文，普通叙述采用中文主称；全教程统一写法见[术语表与中英文写作规范](../../terminology.md)。

Program、线程和tile分别描述什么，编译与Graph重放改变了什么，首次调用与稳态又各计入哪些成本？可先查对应类别，再从学习入口回到具体例子。

## 执行映射与访存

| 中文主称 | 英文 / 缩写 | 在本专题中的含义 | 容易混淆的概念 | 学习入口 |
|:---|:---|:---|:---|:---|
| 计算内核 | kernel | 在加速器上执行一段并行计算的程序单元 | 操作系统内核 | [03 计算内核语义与访存](./03_kernel_semantics_and_memory.md) |
| 程序实例 | Program instance | Triton 网格中的一个逻辑执行实例，可以按算法负责一个或多个数据块 | 单个线程；Program 编号不等于 SM 编号 | [03 计算内核语义与访存](./03_kernel_semantics_and_memory.md) |
| 线程 | Thread | CUDA执行实例，有自己的索引和局部状态 | Triton Program或一个张量元素 | [Part 04 · 01 线程索引](../../04_CUDA_and_System_Optimization/01_CUDA_Custom_Kernel_Intro.md) |
| 线程束与束内编号 | Warp / lane | NVIDIA GPU的32线程执行组；lane是组内编号，活跃掩码决定哪些线程参与当前操作 | tile宽度或Program编号 | [Part 04 · 02 线程地址](../../04_CUDA_and_System_Optimization/02_CUDA_Global_Memory_Coalescing.md) |
| 线程块与网格 | Block / CTA / Grid | 线程块组织可共享块内资源的线程；网格组织本次启动的线程块 | 数据tile；Triton grid统计Program实例 | [Part 04 · 01 执行层级](../../04_CUDA_and_System_Optimization/01_CUDA_Custom_Kernel_Intro.md) |
| 流式多处理器 | streaming multiprocessor / SM | GPU内执行线程块并提供寄存器、共享内存等资源的硬件单元 | 一张GPU；一个Program固定对应一个SM | [Part 04 · 05 资源与驻留](../../04_CUDA_and_System_Optimization/05_CUDA_Block_Resources_and_Residency.md) |
| 分块／数据块 | tiling / tile | 将计算或数据域分成工作块；tile 描述逻辑区域，实际资源分配由实现决定 | 张量视图；逻辑 tile 不直接等于共享内存或线程块 | [03 计算内核语义与访存](./03_kernel_semantics_and_memory.md) |
| 步长与偏移 | stride / offset | stride为维度索引加1时的存储间隔，offset为相对起点的位移；本题按元素计 | 字节地址；视图起点已改变时再次加入存储偏移 | [Part 03 · 02 地址计算](../../03_Triton_Kernels/02_Triton_Layouts_and_Memory_Access.md) |
| 批量在首维的布局 | batch-first | 本页输入按批量、序列位置、特征排列；分头后再转换为批量、头、位置、每头特征 | 每个张量永远采用相同维度顺序；只改shape而不核对特征对应 | [Part 03 · 20 组件布局](../../03_Triton_Kernels/20_Triton_Llama3_Block_Project.md) |
| 掩码 | mask | 按位置决定是否参与访问或运算的条件；读取与写回分别受保护 | 只给写回加保护就能避免越界读取 | [Part 03 · 01 尾块边界](../../03_Triton_Kernels/01_Triton_Vector_Addition.md) |
| 别名 | alias | 不同张量共享底层存储，修改重叠区域可能相互影响 | 数值相等；独立分配的相同副本 | [Part 03 · 19 算子契约](../../03_Triton_Kernels/19_Triton_PyTorch_Op_Integration.md) |
| 合并访存 | memory coalescing | 同一线程束的地址分布允许用较少内存事务服务访问 | 缓存命中率；单条指令加载多个元素的向量化 | [03 计算内核语义与访存](./03_kernel_semantics_and_memory.md) |
| 存储分组与扇区 | Bank / sector | Bank是共享内存并行访问分组；sector在地址推导中是给定大小的覆盖单位 | 把Bank冲突当全局内存事务；把sector计数当实测HBM流量 | [Part 04 · 04 共享内存布局](../../04_CUDA_and_System_Optimization/04_Bank_Conflict_Deep_Dive.md) |
| 片上复用 | on-chip reuse | 在寄存器或共享内存中复用已加载数据，减少全局显存访问 | 主机内存缓存；Tensor Core 是计算单元 | [03 计算内核语义与访存](./03_kernel_semantics_and_memory.md) |
| 寄存器与溢出 | register / spill | 寄存器保存线程局部状态；资源不足时部分值可能转存本地内存 | 源码局部变量必定驻留寄存器；本地内存就是片上共享内存 | [Part 04 · 05 多资源约束](../../04_CUDA_and_System_Optimization/05_CUDA_Block_Resources_and_Residency.md) |
| 可驻留线程块数 | resident blocks per SM | 给定内核和启动配置下，每个 SM 可同时驻留的最大块数；模型上界与 CUDA 查询分别记录 | 运行期间实际活跃的块数；启动网格大小 | [Part 04 · 05 资源与驻留](../../04_CUDA_and_System_Optimization/05_CUDA_Block_Resources_and_Residency.md) |
| 显式启用更大共享内存额度 | shared-memory opt-in | 在设备与函数允许范围内，显式请求高于默认限制的每块共享内存额度 | 任意预留都能成功；本页资源实验已自动提高默认限制 | [Part 04 · 05 默认资源限制](../../04_CUDA_and_System_Optimization/05_CUDA_Block_Resources_and_Residency.md) |
| 理论占用率 | theoretical occupancy | 预测可驻留线程束数相对设备每 SM 最大线程束数的比例，受具体函数和启动资源限制 | 运行期采集值；吞吐或 GPU 总利用率 | [05 资源与驻留](./05_measurement_profiling_and_autotuning.md#每块资源怎样限制同时驻留的任务) |
| 实测占用率 | achieved occupancy | 在指定采集窗口内观测的活跃线程束比例；注明工具、范围和聚合方式 | CUDA 容量查询；更高比例不保证更快 | [05 资源与驻留](./05_measurement_profiling_and_autotuning.md#每块资源怎样限制同时驻留的任务) |
| 矩阵计算单元与接口 | Tensor Core / MMA / WMMA | Tensor Core是硬件单元；MMA是矩阵乘加，WMMA是CUDA的Warp级矩阵操作接口 | 接口可调用就已证明特定机器指令或速度 | [Part 04 · 07 矩阵乘加](../../04_CUDA_and_System_Optimization/07_Tensor_Core_MMA_Programming.md) |

## 数值与计算状态

| 中文主称 | 英文 / 缩写 | 在本专题中的含义 | 容易混淆的概念 | 学习入口 |
|:---|:---|:---|:---|:---|
| 归约 | reduction | 沿指定维度将多个值合成较少结果，如一行求和或最大值；本教程主称归约 | 逐元素运算；规约是同义译法，不另设机制 | [Part 03 · 04 行内归约](../../03_Triton_Kernels/04_Triton_Fused_RMSNorm.md) |
| 广播 | broadcasting | 按兼容维度将共享值用于多个位置，不要求先物理复制 | 集合通信广播；两者不是同一种操作 | [Part 03 · 02 坐标广播](../../03_Triton_Kernels/02_Triton_Layouts_and_Memory_Access.md) |
| 累加器 | accumulator | 保存跨轮次部分和的中间状态，直到当前结果完成 | 每轮重置的临时乘积；输入数据类型 | [Part 03 · 05 K维累加](../../03_Triton_Kernels/05_Triton_GEMM_Tutorial.md) |
| 累加精度 | accumulation precision | 归约或矩阵乘加中保存中间累积结果的数据类型及其数值影响 | 输入与输出的数据类型 | [04 矩阵计算与推理状态](./04_attention_and_inference_data_paths.md) |
| 浮点格式与舍入 | FP16 / BF16 / FP32 / FP64; rounding | 格式决定可表示范围与精度；舍入在结果或类型转换时发生，中间舍入位置可能改变输出 | 位宽相同就精度和范围相同；融合后应逐位一致 | [Part 03 · 03 中间精度](../../03_Triton_Kernels/03_Triton_Fused_SwiGLU.md) |
| 绝对与相对容差 | atol / rtol | 逐元素允许误差不超过atol+rtol×参考值绝对值；运行前确定 | 只比较最大相对误差；看到结果后放宽阈值 | [Part 03 · 03 三路径数值要求](../../03_Triton_Kernels/03_Triton_Fused_SwiGLU.md) |

## 图编译与内核生成

| 中文主称 | 英文 / 缩写 | 在本专题中的含义 | 容易混淆的概念 | 学习入口 |
|:---|:---|:---|:---|:---|
| 即时执行 | eager execution | 按PyTorch操作调用执行本节参考表达式，不在外层使用图编译 | 整个表达式只有一个设备kernel；参考表达式已是最快实现 | [Part 03 · 03 三路径对照](../../03_Triton_Kernels/03_Triton_Fused_SwiGLU.md) |
| 图编译入口与后端 | torch.compile / Inductor | torch.compile是用户入口；本节选择Inductor，生成或选择执行实现 | 编译等于全部改写成手写Triton；fullgraph保证只有一个kernel | [Part 03 · 03 编译对照](../../03_Triton_Kernels/03_Triton_Fused_SwiGLU.md) |
| 图编译 | graph compilation | 获取计算依赖，进行语义保持的改写，并生成或选择具体执行路径 | CUDA Graph 捕获与重放 | [02 图编译、融合与执行候选](./02_graph_compilation_and_fusion.md) |
| 图中断 | graph break | 无法继续捕获当前计算时，编译图在该处中断，形成分段执行 | 整个函数都未编译，或必然发生数值错误 | [02 图编译、融合与执行候选](./02_graph_compilation_and_fusion.md) |
| 守卫 | guard | 在复用编译版本前检查形状、步长等适用条件 | 掩码；守卫失败与图中断不是同一事件 | [02 图编译、融合与执行候选](./02_graph_compilation_and_fusion.md) |
| 重编译 | recompilation | 当前输入无法复用适用编译版本时，可能重新捕获并生成版本；下层缓存仍可能命中 | 一次守卫失效必然全链重新生成；图中断 | [Part01·30 版本复用](../../01_Hardware_Math_and_Systems/30_Dynamic_Shape_Handling.md) |
| 编译特化 | specialization | 根据形状、类型、编译期常量等条件生成具体实现；形状特化是其中一种 | 一个版本覆盖所有输入；运行时参数必然触发新编译 | [02 图编译、融合与执行候选](./02_graph_compilation_and_fusion.md) |
| 即时编译 | just-in-time compilation / JIT | 需要特定实现时生成或读取适用代码；准备过程可能命中缓存 | 第一次Python调用必定是无缓存冷编译 | [Part 03 · 18 编译准备](../../03_Triton_Kernels/18_Triton_Compilation_Artifacts_and_Execution.md) |
| 中间表示与目标代码 | IR / PTX / SASS / cubin | IR表达编译阶段信息；PTX是虚拟指令表示，SASS是硬件机器指令表示，cubin是目标二进制 | PTX与最终机器指令相同；产物存在就已执行 | [Part 03 · 18 产物层次](../../03_Triton_Kernels/18_Triton_Compilation_Artifacts_and_Execution.md) |
| 特化产物 | compiled specialization | 对应源码、签名和配置生成的中间表示、目标代码及相关元数据 | Python 函数名；产物存在不证明实际启动 | [Part 03 · 18 编译产物核对](../../03_Triton_Kernels/18_Triton_Compilation_Artifacts_and_Execution.md) |
| 下沉 | lowering | 将较高层计算表示转换成更具体的操作或目标实现 | 仅减少图节点，或已经证明加速 | [02 图编译、融合与执行候选](./02_graph_compilation_and_fusion.md) |
| 算子融合 | operator fusion | 在语义允许时合并操作，争取减少中间写回与启动，同时改变资源占用 | 图节点变少就必然更快 | [02 图编译、融合与执行候选](./02_graph_compilation_and_fusion.md) |
| 持久化任务分配 | persistent scheduling | 本专题用有限 Program 循环处理多个输出块，检查覆盖和每块状态 | 减少 CPU 启动次数；Program 少不保证更快 | [Part 03 · 06 持久化 GEMM](../../03_Triton_Kernels/06_Triton_Persistent_GEMM_and_Tile_Scheduling.md) |
| 自动调优 | autotuning | 在固定语义和输入分布下搜索分块、线程束或流水阶段配置 | 不受约束地尝试任意代码 | [05 测量归因与自动调优](./05_measurement_profiling_and_autotuning.md) |
| 回退路径 | fallback path | 目标后端或计算内核不满足条件时实际采用的替代执行路径 | 正常命中目标路径 | [02 图编译、融合与执行候选](./02_graph_compilation_and_fusion.md) |

## 注意力与量化数据

| 中文主称 | 英文 / 缩写 | 在本专题中的含义 | 容易混淆的概念 | 学习入口 |
|:---|:---|:---|:---|:---|
| 在线 Softmax 状态 | online softmax state | 沿 K/V 块保留逐行最大值、指数和及输出累积量，并随最大值变化重新缩放旧贡献 | 在线训练；只计算当前块的 Softmax | [04 Attention 与推理状态](./04_attention_and_inference_data_paths.md) |
| 分组查询注意力 | grouped-query attention / GQA | 多个 Query 头共享一个 KV 头，每个 Query 头仍分别计算输出 | 把 Query 头输出也合并为一个头 | [04 Attention 与推理状态](./04_attention_and_inference_data_paths.md) |
| 变长掩码 | variable-length mask | 按请求真实有效长度排除补齐位置，在 Softmax 归一化前约束参与位置 | 归一化后只屏蔽输出 | [04 Attention 与推理状态](./04_attention_and_inference_data_paths.md) |
| 分页 KV 寻址 | paged KV addressing | 经块表把逻辑 Token 位置映射到物理页及页内偏移 | KV 量化；分页本身保持存储数值 | [04 Attention 与推理状态](./04_attention_and_inference_data_paths.md) |
| KV 缓存量化 | KV cache quantization | 用低比特值及缩放信息表示上下文 K/V，读取时恢复参与注意力计算 | 模型权重量化；两者处理不同存活周期的数据 | [04 Attention 与推理状态](./04_attention_and_inference_data_paths.md) |

## 运行时、通信与状态驻留

| 中文主称 | 英文 / 缩写 | 在本专题中的含义 | 容易混淆的概念 | 学习入口 |
|:---|:---|:---|:---|:---|
| 执行流与事件 | Stream / Event | Stream 组织操作提交顺序，Event 可用于标记完成并建立跨流等待 | 创建多个流就已经实现搬运与计算重叠 | [06 传输、提交与显存生命周期](./06_cuda_execution_and_runtime_constraints.md) |
| CUDA Graph | CUDA Graph | 捕获满足条件的一组 GPU 操作并重放，主要减少重复主机提交开销 | 图编译、算子融合或内核内部复用 | [06 传输、提交与显存生命周期](./06_cuda_execution_and_runtime_constraints.md) |
| 锁页内存 | pinned memory | 不能被操作系统换出的主机页，为异步主机—设备传输提供源缓冲条件 | GPU显存；设置non_blocking就保证搬运计算重叠 | [Part 04 · 10 双缓冲传输](../../04_CUDA_and_System_Optimization/10_PyTorch_CUDA_Streams_and_Transfer.md) |
| 已分配与预留显存 | allocated / reserved memory | 前者统计活跃张量持有的分配，后者还包含缓存分配器管理的空闲块 | 设备总占用；两数差值就是泄漏 | [Part 04 · 12 显存观察](../../04_CUDA_and_System_Optimization/12_Memory_Pool_and_VRAM_Management.md) |
| 进程编号与组规模 | rank / world size | rank是组内进程编号，world size是参与进程数 | GPU编号；一个rank必然对应一张独立GPU | [Part 04 · 13 参与者](../../04_CUDA_and_System_Optimization/13_Distributed_Communication_Primitives.md) |
| Gloo通信后端 | Gloo | 提供集合通信实现，在PyTorch中可作为通信后端选择；本页用它在多个进程间通信CPU张量 | CPU设备类型；只在纸面模拟集合通信 | [Part 04 · 13 后端与张量设备](../../04_CUDA_and_System_Optimization/13_Distributed_Communication_Primitives.md) |
| NVIDIA集合通信库 | NVIDIA Collective Communications Library / NCCL | 为NVIDIA GPU提供集合通信等实现，PyTorch可选用其后端通信CUDA张量 | 所有后端都使用固定Ring算法；CPU上的Gloo结果就是GPU性能 | [Part01·20 NCCL与AllReduce](../../01_Hardware_Math_and_Systems/20_NCCL_and_AllReduce_Basics.md) |
| 进程启动方式 | spawn / fork | spawn启动新解释器并导入工作函数；fork继承父进程状态，需考虑已有线程与设备状态 | 通信后端；Jupyter中的内存函数一定能由新解释器导入 | [Part 04 · 13 运行环境](../../04_CUDA_and_System_Optimization/13_Distributed_Communication_Primitives.md) |
| 全归约与全收集 | All-Reduce / All-Gather | 前者合并各rank对应值并分发结果，后者保留各rank输入及来源顺序 | 本地维度归约；集合语义已经规定Ring算法 | [Part 04 · 13 集合语义](../../04_CUDA_and_System_Optimization/13_Distributed_Communication_Primitives.md) |
| 状态分片与卸载 | ZeRO / offload | ZeRO决定参数、梯度与优化器状态的分片；卸载另决定驻留位置及相应搬运 | 分片自动等于CPU卸载；配置合法等于真实训练通过 | [Part 04 · 16 状态配置](../../04_CUDA_and_System_Optimization/16_DeepSpeed_Zero_Config.md) |
| 非易失存储访问协议 | Non-Volatile Memory Express / NVMe | 定义主机与非易失存储的通信方式；本页NVMe卸载把状态放到采用该协议的持久存储上 | 一种计算设备或主机内存；把整条存储到GPU路径视为一次显存拷贝 | [Part 04 · 17 卸载路径](../../04_CUDA_and_System_Optimization/17_Heterogeneous_Training_CPU_Offload_NVMe_Offload.md) |

## 比较与测量

| 中文主称 | 英文 / 缩写 | 在本专题中的含义 | 容易混淆的概念 | 学习入口 |
|:---|:---|:---|:---|:---|
| 参考实现 | reference implementation | 用于计算预期输出、检查语义的可读实现；容差由比较协议预先确定 | 被优化的候选实现；性能基线 | [03 计算内核语义与访存](./03_kernel_semantics_and_memory.md) |
| 性能基线 | performance baseline | 在固定输入、后端与测量范围下实际运行的比较路径 | 可读教学参考；二者可能采用不同执行方式 | [01 优化对象与比较基线](./01_optimization_targets_and_baselines.md) |
| 启动开销 | launch overhead | 主机提交和运行时安排计算内核所产生的成本，随调用方式与环境变化 | 设备内核自身的计算时间 | [01 优化对象与比较基线](./01_optimization_targets_and_baselines.md) |
| 首次调用完成时间 | first-call completion time | 同步边界内完成一次调用的时间，可包含编译、缓存读取、分配和首次执行 | 纯编译时间；未清空缓存却称冷启动 | [Part 03 · 03 准备与稳态](../../03_Triton_Kernels/03_Triton_Fused_SwiGLU.md) |
| 稳态计时 | steady-state timing | 完成编译、配置搜索与预热后，在明确同步边界内采集重复运行样本 | 包含首次编译的冷启动时间 | [05 测量归因与自动调优](./05_measurement_profiling_and_autotuning.md) |
| 完成时间与设备时间 | completion time / device time | 主机墙钟加同步覆盖调用与完成等待；设备事件时间覆盖选定设备窗口 | 两者可直接互换；用完整调用差值断言kernel更快 | [Part 03 · 03 计时窗口](../../03_Triton_Kernels/03_Triton_Fused_SwiGLU.md) |
| 配对加速比 | paired speedup | 每轮先算基线耗时除以候选耗时，再按指定方法汇总 | 两组耗时中位数之比；候选／基线是相反方向 | [05 测量归因与自动调优](./05_measurement_profiling_and_autotuning.md) |
| 准备成本摊销 | setup-cost amortization | 后续调用节省的时间逐渐抵消新增编译、搜索或捕获成本 | 稳态更快就一定适合一次性任务 | [05 准备成本](./05_measurement_profiling_and_autotuning.md#准备成本需要多少次调用摊销) |
| 实际执行核对 | execution-path verification | 将调用输入、编译配置及产物与实际启动或设备活动记录对应 | 配置开关已打开、产物存在或仅有数值一致 | [07 集成、基准与采用判断](./07_integration_benchmarking_and_adoption.md) |
| 微基准 | microbenchmark | 在固定输入下隔离测量单个计算内核或局部组合 | 模型端到端结果 | [07 集成、基准与采用判断](./07_integration_benchmarking_and_adoption.md) |
| 端到端复测 | end-to-end validation | 将候选放入明确定义的完整组件或负载，重新检查数值、执行路径、时间与资源门槛 | 局部加速自动传递到完整模型或在线服务 | [07 集成、基准与采用判断](./07_integration_benchmarking_and_adoption.md) |
| 采用门槛 | adoption criteria | 对目标负载预先确定的数值、路径、资源和收益要求，用于决定采用、调整或拒绝候选 | 实验成功运行，或每项有效实验都必须更快 | [07 集成、基准与采用判断](./07_integration_benchmarking_and_adoption.md) |
