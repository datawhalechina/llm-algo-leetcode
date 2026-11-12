# 算子优化实验资产表

本表区分教学参考、内核运行、局部性能与综合复测，登记可读取的结果及尚缺的证据。环境创建和恢复见[算子优化 Quickstart](./quickstart.md)，结构化结果字段参考[结果资产规范](../../docs/part02_result_assets.md)。Part03 小型正确性练习和微基准直接查看命令行输出，不强制生成项目式报告。

## 环境与实验入口

| 环境 | 实验与运行条件 | 准备入口 |
|:---|:---|:---|
| CPU 参考环境 | Part01资源推理、Part02机制模拟及Part04标准库推导；也可复用主GPU环境运行 | [BASE-CPU](../../docs/guide.md#base-cpu) |
| `llm_algo_cnb_gpu`主GPU环境 | Part03 Triton、Part04原生CUDA与单GPU Runtime；源为`cnb/environment-gpu.yml`，不要求另建一套Runtime环境 | [主GPU环境](./quickstart.md#triton-environment) |
| 原生CUDA C++工具链条件 | Part04编译页在同一CUDA PyTorch环境中还需兼容的`nvcc`、主机编译器和头文件；不是独立Python环境 | [工具链检查](./quickstart.md#原生cuda工具链)及各页运行环境 |
| `INF-CUDA-MECHANISM` | Part02·20 的显式注意力与 SDPA 对照 | [推理机制 Quickstart](../inference_optimization/quickstart.md#inf-cuda-mechanism-quickstart) |
| `PERF-CUDA-MECHANISM` | PERF-ROOFLINE 和 Part05·05；硬件门槛按各页说明 | [性能机制 Quickstart](../performance_analysis/quickstart.md#perf-cuda-mechanism-quickstart) |
| `PAR-CPU` / 多GPU通信环境 | Part04·13的CPU/Gloo核对与另行设计的NCCL实测分别选择；已有主环境支持Gloo并通过13测试时可复用 | [通信与并行Quickstart](../communication_parallelism/quickstart.md) |
| 已有独立DeepSpeed环境 | Part04·16可选配置解析，不要求两张GPU，不代表完整训练验证；真实执行状态见下方16条目 | [框架核对命令](./quickstart.md#zero-config-check) |

## 机制实验

先通过对应环境预检，再检查参考输出、边界和目标路径。正确性或路径身份不明时停止收益判断；计时有效但候选变慢时保留记录，继续归因或回到基线。是否满足采用门槛由组件和项目复测决定，不以“先测到加速”作为实验有效的条件。

| 机制入口 | 固定输入与对照 | 输出或结果 | 当前证据与适用范围 |
|:---|:---|:---|:---|
| [Part01 · 03 硬件能力](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.ipynb)、[15 执行模型](../../01_Hardware_Math_and_Systems/15_CUDA_Execution_Model.ipynb)、[16 块内复用](../../01_Hardware_Math_and_Systems/16_Warp_Block_SharedMemory_Basics.ipynb)、[18 Triton分块](../../01_Hardware_Math_and_Systems/18_Triton_Block_Model.ipynb) | CPU计算器；任务覆盖、逻辑读取、示例预算与尾块 | 2026-11-11 四页代码逐单元顺序执行通过，代码保持原值 | 区分架构／型号／CC、执行配置与硬件资源；逻辑元素槽不等于线程。不是设备能力探针或硬件性能实测 |
| [Part01 · 09 图编译](../../01_Hardware_Math_and_Systems/09_AI_Compilers_and_Graph_Optimization.ipynb)、[Part01 · 19 算子融合](../../01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.ipynb) | CPU 构造输入；估算中间读写、启动次数与片上预算，检查后端条件 | 页面中的成本推导与条件断言；不调用真实 GPU 编译路径 | 支撑 Task1 的候选判断；真实固定形状对照另见本表Part03·03三路径v1，不将CPU推导登记为GPU测量 |
| [Part02 · 20 分块注意力](../../02_PyTorch_Algorithms/20_FlashAttention_Sim.ipynb) | BF16，batch=1、8头、头维64；长度512/1024/2048/4096；非因果；显式注意力与SDPA | [长度矩阵 v3](../../benchmarks/results/20_flashattention/inference-attention-sdpa_rtx5070ti_sdpa_matrix_v3.json) | 等级为 `local_mechanism_benchmark`，决策为 `accept_for_mechanism_evidence`；不证明手写Triton或因果路径收益 |
| [Part02 · PERF-ROOFLINE](../../02_PyTorch_Algorithms/PERF-ROOFLINE_Arithmetic_Intensity_and_Resource_Ceiling.ipynb) | 拷贝与两档矩阵乘法；固定设备、预热和重复测量 | [经验参考 v1](../../benchmarks/results/perf_roofline/perf-empirical-roofline_rtx5070ti_roofline_v1.json) | 文件格式为 `performance-roofline-mechanism/v2`，决策为 `ready_for_bottleneck_hypothesis`；经验参考不是硬件理论峰值 |
| [Part03 · 01 向量](../../03_Triton_Kernels/01_Triton_Vector_Addition.ipynb) | 同设备、连续一维 FP16／FP32；同输入 PyTorch 对照 | 2026-11-10 本批 GPU 答案通过 12 个数值、7 个非法输入用例；题目在实际 kernel 的局部 TODO 按预期停止 | 保留原有讲解与主图，错误减法实现被数值测试拒绝。可选微基准入口通过，当前样本慢于 PyTorch，不将运行成功或单次比值写成加速结论 |
| [Part03 · 02 布局](../../03_Triton_Kernels/02_Triton_Layouts_and_Memory_Access.ipynb) | 非空二维 FP16／FP32；16×32 矩形 tile；支持转置、切片、非零存储偏移与广播视图 | 2026-11-10 本批 GPU 答案通过 20 个数值、8 个非法输入用例；题目在局部 TODO 按预期停止 | 精确数值、独立连续输出、二次转置及输入不变核对；忽略 stride、错误输出、仅返回视图均被测试拒绝。没有新增性能测量或合并访存结论 |
| [Part03 · 03 融合](../../03_Triton_Kernels/03_Triton_Fused_SwiGLU.ipynb) | eager／Inductor编译／手写Triton；三档长度×FP16／FP32；共同预定容差对照eager和FP64数学参考，关闭CUDA Graph | 2026-11-11题目局部TODO预期未完成；答案通过16个数值与7个非法输入。[三路径v1](../../benchmarks/results/swiglu_three_path_rtx5070ti_20261111_v1.json)保留六组数值／路径检查、每组六轮平衡排列与108个计时窗口 | 单列包装器构建及每形状首次调用，非冷编译；稳态为主机墙钟加两端设备同步，包含提交／分配／完成等待。多数输入候选较慢，FP32最大输入手写路径有局部收益；不以必须加速验收，不外推纯kernel、HBM或模型收益 |
| [Part03 · 04 归一化](../../03_Triton_Kernels/04_Triton_Fused_RMSNorm.ipynb) | 最后一维 1–4096，FP16／FP32；FP32 归约与缩放；多维及非连续输入、非均匀权重和正 epsilon | 2026-11-10 新版 GPU 答案通过 30 个数值、14 个非法输入用例；题目在局部 TODO 按预期停止 | 修正展平行间隔；漏乘权重、补齐块宽作分母、忽略 epsilon 的安全错误实现均被拒绝。独立微基准运行通过，尚无版本化性能 JSON；显式 PyTorch 参考不代表成熟 RMSNorm 内核 |
| [Part03 · 05 GEMM](../../03_Triton_Kernels/05_Triton_GEMM_Tutorial.ipynb) | 连续 CUDA FP16 矩阵；M/N/K 尾块、多轮 K 累加及不足一组的末尾输出块 | 2026-11-10 本批 GPU 答案通过 6 组数值、10 组输入约束；题目在 kernel 的局部 TODO 按预期停止 | 重置累加器、重复首个 K 子块的安全错误实现被数值测试拒绝。固定 1024³ 微基准入口运行通过，不含编译和自动调优；无缓存命中、指令路径或版本化性能 JSON |
| [Part03 · 07 Softmax](../../03_Triton_Kernels/07_Triton_Fused_Softmax.ipynb) | 连续二维 FP16／FP32，行宽 1–4096，有限分数；FP32 中间归约 | 2026-11-10 本批 GPU 答案通过 18 组数值、10 组输入约束；题目在 kernel 的局部 TODO 按预期停止 | 大正负数、全负常量行、尾部及共同平移核对；补零、漏平移、错误分母均被测试拒绝。固定 1024×1024 微基准入口运行通过，无版本化性能 JSON，不外推其他行宽或模型收益 |
| [Part03 · 08 反向](../../03_Triton_Kernels/08_Triton_Softmax_Backward.ipynb) | 同形状二维 CUDA FP32 概率和上游梯度，行宽1–4096；自动微分对照 | 2026-11-10 本批 GPU 答案通过12组数值／布局和10组输入约束，题目在 kernel 的局部 TODO 按预期停止 | 普通、均匀／集中概率、常量／零梯度及非连续视图；错误点积、漏减共享项、补齐污染均被数值断言捕获。无独立性能测量或 Attention 反向结论 |
| [Part03 · 09 自动调优](../../03_Triton_Kernels/09_Triton_Autotune_and_Profiling.ipynb) | 连续一维 CUDA FP32；三个块大小／warp 候选与同输入 PyTorch 对照 | 2026-11-10 本批 GPU 通过三个候选的15组直接启动、自动选择路径12组数值与7组输入约束；题目在配置 TODO 按预期停止 | 错误相减与单 Program 网格被测试捕获。4096／65536／1048576元素各五轮交替微基准运行通过，保留选择配置和逐轮样本；样本包含快慢两种情况，尚无版本化性能 JSON，不将当前选择视为全局最优 |
| [Part03 · 10 RoPE](../../03_Triton_Kernels/10_Triton_Fused_RoPE.ipynb) | 连续三维 CUDA FP16／FP32；偶数头维 2–256；相邻配对、FP32 中间旋转与原位写回 | 2026-11-10 GPU 答案通过20组数值与16组输入约束；题目在局部 TODO 按预期停止 | 零角度、四分之一圈、位置／头映射、配对长度、非连续角度表与返回地址核对；旋转符号及角度表行号错误均被数值断言拒绝。独立五轮微基准已运行，候选克隆与参考分配计入；无版本化性能 JSON，不外推模型收益 |
| [Part03 · 11 Attention](../../03_Triton_Kernels/11_Triton_Flash_Attention.ipynb) | 连续四维 CUDA FP16／BF16，实际头维1–128；稠密前向、因果等长或非因果不等长 | 2026-11-10 GPU 答案通过26组数值／机制不变量与20组输入约束；题目在局部 TODO 按预期停止 | 序列尾块、短头维、跨块新最大值、头间隔离及未来值不影响历史输出。漏历史修正、放开未来键、漏归一化均被数值断言拒绝；共享实现通过相同用例与 Block 集成回归。未新增性能或反向证据 |
| [Part03 · 12 GQA](../../03_Triton_Kernels/12_Triton_Variable_Length_GQA.ipynb) | CUDA FP32、连续补齐 KV；头维1–64、容量1–128；MHA／GQA／MQA，按请求给定有效长度 | 2026-11-10 GPU 答案通过17组数值／机制不变量与21组输入约束；题目在局部 TODO 按预期停止 | 不同头常量、独立查询输出、补齐扰动、全负分数与非连续视图；轮转映射、补齐进入分母、漏归一化均被数值断言拒绝。连续化可能复制输入，无性能或实际访存减少证据 |
| [Part03 · 13 分页](../../03_Triton_Kernels/13_Triton_PagedAttention.ipynb) | CUDA FP16／FP32，连续只读缓存；页容量1–64的二次幂、头维1–128；允许非连续页表与长度视图 | 2026-11-10 GPU 答案通过22组数值／机制不变量与33组输入约束；题目在局部 TODO 按预期停止 | 非连续物理页、共享页、尾页 NaN 屏蔽、物理重排与跨页新最大值；逻辑页误作物理页、尾页增加权重、漏历史修正均被拒绝。独立五轮完整接口计时包含页表检查与同步，未作后端对照；不证明分配器、服务吞吐或消除碎片 |
| [Part03 · 14 权重量化](../../03_Triton_Kernels/14_Triton_Quantization.ipynb) | CUDA FP16输入、INT8权重、FP16／FP32正列尺度；FP32反量化后舍入FP16再点积；支持矩阵步长与三轴尾块 | 2026-11-11 GPU答案通过15组数值／布局和18组输入约束；题目在局部TODO停止 | 漏尺度、覆盖K累加器均被数值断言拒绝。三条准备范围不同的路径各五轮：预重建FP16 GEMM、每次重建再GEMM、安全融合接口；当前融合接口较慢，含检查／同步／分配，不能归为裸kernel差异或通用量化收益，无版本化性能JSON |
| [Part03 · 15 Multi-LoRA](../../03_Triton_Kernels/15_Triton_Multi_LoRA.ipynb) | CUDA FP32、秩1–32；Token各自路由到配套A/B，输入及权重支持步长视图 | 2026-11-11 GPU答案通过12组数值／机制不变量和23组输入约束；题目在局部TODO停止 | 两类置换、混合路由、输入／秩／输出尾部、零A/B；固定路由、错配B、覆盖输入累加器均被拒绝。只输出已含调用方缩放约定的低秩增量，不含基础投影；不同输出块重复计算h，无调度或吞吐证据 |
| [Part03 · 16 KV量化](../../03_Triton_Kernels/16_Triton_KV_Cache_Quantized_Decode.ipynb) | CUDA有限FP32；每Token／头的K、V各自INT8编码与FP32尺度，半值远离零；容量1–128、头维1–64 | 2026-11-11 GPU通过25组编码／尺度／重建界、9组解码和31组输入约束；题目在局部TODO停止 | 与同一重建缓存严格对照，原始缓存误差另作固定样例观察；截断舍入、交换V尺度、补齐参与分母均被拒绝。实际字节符合含尺度公式，D=1比值1.25、D=16为0.3125；只是表示大小，不是峰值显存或持久缓存性能，包装器每次重新量化 |
| [Part03 · 17 调试](../../03_Triton_Kernels/17_Triton_Memory_Model_and_Debug.ipynb) | CUDA FP32；二维输入／输出分别按步长寻址，连续向量逐块点积；只检查支持的独立、不重叠输出 | 2026-11-11 新版 GPU 答案通过49组地址／布局、43组尾块点积、14组非法输入及非默认流；题目在 kernel 局部TODO停止 | 错读输入行、反向写输出列、错误填一的三个安全变体均被数值断言拒绝；检查视图间隙和输入不变，不故意制造越界或竞态。无调试工具实测或性能结论 |
| [Part03 · 18 编译产物与实际执行](../../03_Triton_Kernels/18_Triton_Compilation_Artifacts_and_Execution.ipynb) | 连续、无需梯度的一维 CUDA FP32；完整块／尾块、BLOCK=128／256／512与运行时系数；源码、特化产物和对象启动关联 | 2026-11-11 新版 GPU 通过7组数值／产物导出SHA256、11组非法输入、非默认流；题目局部TODO预期停止，最大绝对误差约1.19e-7 | 观察器核对当前对象的启动次数、网格与参数，并采集真实CUDA入口；输出正确但绕过对象的替代实现被拒绝。摘要与事件不等于设备端逐字验证cubin，无性能、SASS或硬件计数结论 |
| [Part03 · 19 算子接入](../../03_Triton_Kernels/19_Triton_PyTorch_Op_Integration.ipynb) | 非空连续CUDA FP32缩放；四种系数、非均匀及零上游梯度、独立输出 | 2026-11-11新版GPU通过20组普通调用、2组完整图编译前后向和6组非法输入；题目在kernel局部TODO停止，最大梯度误差为零 | 输入不变、输出非别名与链式法则分别核对；Python标量由算子schema规范化，不按调用前的整数／浮点类型区分。无编译性能或Attention训练结论 |
| [Part04 · 01 CUDA 入门](../../04_CUDA_and_System_Optimization/01_CUDA_Custom_Kernel_Intro.ipynb) | 连续非空一维 CUDA FP32 加法；两输入同形状、同设备；当前流调用 | 2026-11-11 终端 GPU 核对：5 组长度、6 组非法输入、非默认流中的准备／加法／消费 | 新版题目在局部表达式 TODO 预期停止，答案通过；删除源码注释扫描与整设备同步。不含性能、竞态检测器或多流重叠证据 |
| [Part04 · 03 Warp归约](../../04_CUDA_and_System_Optimization/03_Warp_Level_Primitives.ipynb) | CPU单轮交换与两级归约；非默认模拟宽度 | 2026-11-11答案通过8组归约、6组逐lane状态、两Warp真实函数调用与12组非法参数；题目在局部TODO停止 | 整轮直接求和、错误来源、漏末Warp与非零补齐均被拒绝；不执行设备Shuffle，不验证真实参与掩码或块级屏障 |
| [Part04 · 04 Bank映射](../../04_CUDA_and_System_Optimization/04_Bank_Conflict_Deep_Dive.ipynb) | CPU32位字地址模型；Bank内按字地址分组lane，区分单lane、广播及多个不同字 | 2026-11-11地址测试及7组分组／广播场景、非法输入通过；题目在局部TODO停止 | 错行跨度、错模数、把Bank当字身份、误判多个字为广播、忽略广播的五种错误被拒绝；不采集硬件事务、冲突计数或GPU性能 |
| [Part04 · 02 合并访问](../../04_CUDA_and_System_Optimization/02_CUDA_Global_Memory_Coalescing.ipynb) | 连续非空一维 CUDA FP32 拷贝；固定线程配置 | 2026-11-11 终端 GPU 复核：5 组长度与输出属性；题目预期未完成，答案通过 | 连续、整块和尾部数值核对；不是连续与跨步访问的配对性能或硬件事务测量 |
| [Part04 · 06 共享内存 GEMM](../../04_CUDA_and_System_Optimization/06_CUDA_Shared_Memory_Optimization.ipynb) | 连续同设备 FP32 方阵，固定 16×16 分块，边缘填零 | 2026-11-11 GPU 答案通过 9 组数值、7 组非法输入与非默认流；题目在局部 TODO 停止 | 重复读取首段 A 的安全错误实现被数值断言拒绝；未运行竞态检测器，不以数值通过证明删除屏障仍然安全，无性能记录 |
| [Part04 · 07 WMMA](../../04_CUDA_and_System_Optimization/07_Tensor_Core_MMA_Programming.ipynb) | 连续、32 字节对齐的 FP16 输入；各维度为 16 的正整数倍，FP32 累加／输出 | 2026-11-11 CPU 分块计划及约束通过；GPU 答案通过 8 组数值、7 组非法输入与非默认流；题目在局部 TODO 停止 | K 循环内清零累加器的错误被数值断言拒绝；未采集实际指令、硬件计数或配对性能，不支持边缘分块 |
| [Part04 · 08 双阶段流水估算](../../04_CUDA_and_System_Optimization/08_Async_Data_Prefetch_and_Double_Buffering.ipynb) | CPU 等长独立批次，给定传输／计算时间、批次数与并发条件 | 2026-11-11 CPU 答案通过首尾阶段、对称性、逐批增量、缓冲条件与非法输入测试；题目在局部 TODO 停止 | 误用较快阶段、将联合条件写成任一满足均被拒绝；时间线为构造值，没有执行 CUDA 或测得重叠，缓冲数不代表硬件并发能力 |
| [Part04 · 09 线程级异步拷贝](../../04_CUDA_and_System_Optimization/09_CUDA_Async_Copy_Pipeline.ipynb) | 连续非空一维 CUDA FP32，线程独占两处共享位置；有限 FP32 缩放系数 | 2026-11-11 GPU 答案通过 24 组长度／缩放数值、10 组非法输入与非默认流；题目在局部 TODO 停止 | 固定读取缓冲区 0、遗漏缩放均被数值断言拒绝；不是块级生产者／消费者队列，不证明 TMA 路径、真实重叠或性能提升 |
| [Part04 · 10 Stream](../../04_CUDA_and_System_Optimization/10_PyTorch_CUDA_Streams_and_Transfer.ipynb) | 同设备的两条工作流、锁页 CPU 浮点矩阵批次、匹配类型与内维的 CUDA 权重 | 2026-11-11 GPU 答案通过 15 个随机批次、5 个常量批次、10 组非法输入与非默认调用方准备依赖；题目在局部 TODO 停止 | 复制首批数据与颠倒输出顺序均被断言拒绝；数值检查不保证任意竞态均被检出，重叠程度与性能仍需独立测量 |
| [Part04 · 11 Graph重放](../../04_CUDA_and_System_Optimization/11_CUDA_Graph_and_JIT_Compile.ipynb) | CPU 串行估算；GPU 固定形状下的普通提交／Graph 配对 | 2026-11-11 仅复核 CPU 题目／答案与舍入边界；GPU 沿用 2026-11-10 的[完整记录](../../benchmarks/results/18_cuda_graph_replay_rtx5070ti_20261110.json)，本轮基准代码未变且未重跑 | CPU 题目预期未完成、答案通过，舍入占比误用于判断的错误被拒绝；历史 GPU 配对中位数约 4.03×／3.51×，仅覆盖固定短链且不计输入更新，不外推完整模型或服务收益 |
| [Part04 · 05 线程块资源与驻留](../../04_CUDA_and_System_Optimization/05_CUDA_Block_Resources_and_Residency.ipynb) | 固定 FP32 缩放加偏置、256 线程／块，仅改变 0／8／16／32／48 KiB 动态共享预留 | [完整记录](../../benchmarks/results/cuda_residency_rtx5070ti_20261110.json)：五档资源查询、六轮交替耗时，另有 20 组 GPU 数值核对 | 2026-11-10 CPU 题目预期未完成、答案及独立基准通过。CUDA 预测为 6／6／5／3／2 块／SM；简化模型在 16 KiB 档高估一块。结果是局部容量与时间对照，未采集 achieved occupancy，不代表模型或服务收益 |
| [Part04 · 12 显存筛查与生命周期](../../04_CUDA_and_System_Optimization/12_Memory_Pool_and_VRAM_Management.ipynb) | CPU规则筛查；独立子进程、native分配器、单个64 MiB张量、3轮7阶段GPU状态核对 | 2026-11-11 CPU题目预期未完成、答案通过；[GPU完整记录](../../benchmarks/results/allocator_lifecycle_rtx5070ti_allocator_lifecycle_20261111_v1.json)通过数据与引用检查，默认read | 保留别名及存活时empty_cache后allocated仍64 MiB；删最后引用后allocated为0而reserved为64 MiB，收尾均0。记录本进程窗口峰值，不测碎片、OOM阈值、申请速度或模型收益 |
| <a id="zero-config-evidence"></a>[Part04 · 16 ZeRO配置](../../04_CUDA_and_System_Optimization/16_DeepSpeed_Zero_Config.ipynb) | CPU标准库配置；可选DeepSpeed解析核对36组显式字段、3类格式错误与2组跨stage观察 | 2026-11-11 CPU答案通过4种分片关系、36组合法配置、26组非法输入、JSON往返和字典独立性；框架入口已建设，本地缺DeepSpeed而skipped，尚未真实验收 | stage策略与schema接受分别判断；框架解析即使通过也不证明训练、分片或CPUAdam适配。严格命令见Quickstart，不向现有性能／Triton环境临时混装 |
| [Part04 · 17 卸载估算与搬运](../../04_CUDA_and_System_Optimization/17_Heterogeneous_Training_CPU_Offload_NVMe_Offload.ipynb) | CPU串行路径估算；可选单GPU FP32驻留／CPU往返，4／32 MiB、每轮10步、6轮AB/BA，锁页缓冲预分配 | 2026-11-11 CPU题目预期未完成、答案通过；[GPU完整记录](../../benchmarks/results/offload_path_rtx5070ti_offload_path_20261111_v1.json)全部配对及累计值通过，默认read | 往返路径中间allocated为0，窗口峰值仍4／32 MiB；往返／驻留时间比中位数3.387／20.676，记录同步、采样与调回分配成本。不测NVMe、CPU更新、真实重叠或训练容量；602／701 ms仍为CPU构造值 |
| [Part04 · 13 集合通信](../../04_CUDA_and_System_Optimization/13_Distributed_Communication_Primitives.ipynb) | 本地2／3进程CPU/Gloo；6种长度／类型／数值输入 | 2026-11-11 答案通过30组逐 rank 核对与6组启动规模检查；题目在局部TODO停止 | SUM、列表来源顺序、独立接收缓冲与输入不变；MAX替代及列表倒序的安全错误变体被拒绝。不证明Ring算法或NCCL性能 |
| [Part04 · 14 Ring发送量](../../04_CUDA_and_System_Optimization/14_Ring_AllReduce_Deep_Dive.ipynb) | CPU均匀分片整数模型；2／3／4／7／16个rank | 2026-11-11 通过20组均分与缩放、11组非法输入；题目局部TODO停止 | 单 rank 发送口径；漏阶段和误计全系统流量被拒绝。不测真实通信或延迟 |
| [Part04 · 15 通信重叠](../../04_CUDA_and_System_Optimization/15_Communication_Computation_Overlap_Advanced_Scheduling.ipynb) | CPU无额外空闲间隙的双阶段模型 | 2026-11-11 通过9组时间线与13组非法输入；题目局部TODO停止 | 无／部分／完全重叠、通信较长、缩放与差值关系；重复减重叠和误从计算扣除被拒绝。不采集多卡轨迹 |
| [Part04 · 19 TCO选型](../../04_CUDA_and_System_Optimization/19_TCO_and_Hardware_Selection.ipynb) | CPU标准库；构造容量、质量、P99、SLO能力、实际需求与同周期成本 | 2026-11-11题目局部TODO预期停止，答案通过单项／多项筛选、40／100／200小时反转、并列、需复测、零产出、交点范围与13组非法输入 | 六种错误变体被拒绝：容量等号、忽略质量／证据、能力冒充业务、小时换算与丢弃并列；没有设备报价、GPU压测或真实选型结论 |
| [Part03 · 06 持久化 GEMM](../../03_Triton_Kernels/06_Triton_Persistent_GEMM_and_Tile_Scheduling.ipynb) | 同一 FP16 GEMM kernel、32×64×32 tile；普通网格与 46／92 Program 预算；统一 Graph 承载计时 | [完整记录](../../benchmarks/results/persistent_gemm_rtx5070ti_20261110.json)：7 组数值与实际 tile 访问核对，4 组形状、6 轮、72 个窗口 | 2026-11-10 题目预期未完成、答案及独立基准通过。小形状网格相同；其余三组有限 Program 均较慢，配对比值约 0.30–0.62×。只证明当前配置，不外推所有持久化实现或库性能 |
| [Part04 · 18 实现方式选型](../../04_CUDA_and_System_Optimization/18_CUDA_vs_Triton_vs_PyTorch.ipynb) | 三个构造场景：热点占比、准备成本摊销与调用流依赖 | 内容型判断练习，附参考判断；无可执行题目／答案代码，运行级测试不适用 | 用于选择下一项核对，不登记CUDA、Triton或PyTorch的通用性能排名 |

## 项目实验

组件项目和综合项目分别运行各自负载。复用Attention代码不等于复用性能结论，综合项目也不要求读取组件项目的结果文件。

| 项目 | 直接输入与比较对象 | 输出与结果状态 | 验证范围 |
|:---|:---|:---|:---|
| [Part03 · 20 Block集成](../../03_Triton_Kernels/20_Triton_Llama3_Block_Project.ipynb) | 独立合成输入、同权重CUDA FP16组件；只替换因果Attention | 2026-11-11练习补强后通过8组CPU布局输入、4组非法形状、非连续Q/K/V适配；5组组件形状逐段核对Q/K/V、Attention、合头、完整输出，并检查输入不变、因果前缀及各形状CUDA入口。拆头／合头错轴、漏连续化、交换Q/K、漏因果条件5种错误变体均被拒绝。此前benchmark通过33／128／257三档长度、各6轮交替配对，本轮未重测性能 | 独立Step5新增跨长度固定权重、逐形状路径核对与block-attention-paired/v1保存／读取，默认plan；计划与临时构造记录读写检查通过，新协议尚无实测JSON。此前计时为完整Block稳态设备窗口，不含准备／编译／H2D；显存为两模型同驻留下各路径增量峰值。有慢于基线的历史样本，不外推完整模型生成或训练 |
| [Part05 · 05 算子优化综合项目](../../05_Integrated_Projects_and_Decisions/05_Operator_Optimization_Delivery/index.md) | 独立两层解码器，长度128/512/1024/2048，BF16、因果；SDPA自动后端与Triton对照 | 本地未发现项目结果目录，尚未登记实测；输出由[项目配置](../../05_Integrated_Projects_and_Decisions/05_Operator_Optimization_Delivery/project_config.example.yaml)指定 | 数值、实际内核、配对延迟、峰值显存与短序列回退；采用门槛按项目协议 |

## 结果版本与复现

当前状态按上表逐页的日期、代码与测量协议读取：正确性通过、性能已测和新协议待测分别登记。例如Part03·20已有组件正确性及历史终端计时，但新协议仍无实测JSON；Part04·16的CPU练习已通过，DeepSpeed解析仍为缺依赖跳过。历史结果文件名、内容与测量时来源路径不随小节改号重写。

正式阅读入口采用上表的具体结果链接。Part02·20使用Notebook已登记的长度矩阵v3；Roofline使用现有v2格式记录。保留原始决策，不补写未采集的指标，不因驱动升级自动视为完成复测。

Part03/04的终端反馈说明相应测试曾执行；代码、输入或环境改变后，应按当前页面重新运行。未完成题目按预期失败、答案通过，说明对应练习测试符合预期；CPU推导、CPU/Gloo、框架解析与GPU执行按实际路径分别记录。`skipped`表示未执行，不计为通过；内容型选型页记录运行级测试不适用。微基准使用 `--mode benchmark` 独立运行，命令成功不等于测到加速。

版本化实验读取已有标签时查看对应记录；新运行使用唯一标签并保留失败结果。数值错误或路径身份不明时停止采用判断；计时可配对但收益不足时，可记录 `tune` 或 `reject`，无需为得到 `accept` 反复扩大实验。

<details>
<summary>历史：2026-11-10 编号调整后的批次核对</summary>

当时登记Part03的20节练习答案通过、题目预期未完成；Part04登记17节练习通过，另1个纯选型Notebook不适用运行级测试。这是该批次的历史汇总，不是当前页数或后续改版的统一验收。后续补强与新增练习以逐页条目为准，不据历史汇总追加新版性能或多卡证据。

</details>

<details>
<summary>历史：单长度SDPA v1</summary>

[单长度v1原始记录](../../benchmarks/results/20_flashattention/inference-attention-sdpa_rtx5070ti_sdpa_v1.json)仅覆盖长度512，原决策为 `reject`，原因为数值误差未通过当时门槛。其格式和判定条件与后续长度矩阵不同，不合并统计，也不改写原始结论。

</details>
