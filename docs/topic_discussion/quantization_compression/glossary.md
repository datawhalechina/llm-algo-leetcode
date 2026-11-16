# 量化与压缩专题术语表

本表集中解释量化语境中容易混淆的概念。全教程的标准中文名、英文名和写作规则仍以[术语表与中英文写作规范](../../terminology.md)为准；数学符号、单位和常用约束统一查阅[量化与压缩公式索引](./formula_reference.md)。

| 中文主称 | 英文 / 缩写 | 在本专题中的含义 | 容易混淆的概念 | 学习入口 |
|:---|:---|:---|:---|:---|
| 量化对象 | quantization target | 被改变数值表示的权重、激活、训练状态或 KV Cache | “整个模型都执行整数计算” | [01 对象、表示与误差](./01_quantization_object_and_error.md) |
| 位宽 | bit width | 一个量化码或浮点格式用于表示数值的位数 | 数据类型、实际存储字节、累加精度 | [Part 01 · 21 量化理论](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.md) |
| 量化范围 | quantization range | 浮点数值映射到离散码之前采用的上下界 | 数据类型本身的理论范围 | [52 校准与误差](../../02_PyTorch_Algorithms/52_Quantization_Calibration_and_Error.md) |
| 缩放因子 | scale | 浮点值与量化码之间的映射步长 | SmoothQuant 的通道平滑系数 | [25 W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.md) |
| 零点 | zero point | 非对称量化中浮点零对应的整数码 | 数值范围中点 | [25 W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.md) |
| 量化参数 | quantization parameters, qparams | 缩放因子、零点及其粒度、轴和数据类型等共同决定量化映射的状态 | 模型超参数、训练参数 | [25 W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.md) |
| 计算数据类型 | compute dtype | 低比特权重反量化后参与前向或反向计算时采用的数据类型 | 权重存储格式、累加数据类型 | [03 低比特训练适配](./03_low_bit_training_adaptation.md) |
| 量化粒度 | granularity | 一组数值共享缩放因子和零点的范围 | 位宽、分组大小 | [25 W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.md) |
| 逐张量 / 逐通道 / 逐组 | per-tensor / per-channel / per-group | 整个张量、单个通道或固定大小元素组分别共享一套量化参数 | 位宽、张量并行组 | [25 W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.md) |
| 分组大小 | group size | 分组量化中共享量化参数的连续元素数量 | 批大小、张量并行组 | [40 GPTQ / AWQ](../../02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.md) |
| 对称量化 | symmetric quantization | 以零为中心、通常使用零点 0 的映射 | 数值分布本身完全对称 | [25 W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.md) |
| 非对称量化 | asymmetric quantization | 用缩放因子与非零零点覆盖偏移范围 | 仅使用无符号整数 | [25 W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.md) |
| 校准 | calibration | 使用独立样本估计范围、异常值或敏感度统计 | 训练、任务评测 | [52 校准与误差](../../02_PyTorch_Algorithms/52_Quantization_Calibration_and_Error.md) |
| 饱和 / 裁剪 | saturation / clipping | 超出量化范围的值被限制到边界码 | 浮点溢出 | [52 校准与误差](../../02_PyTorch_Algorithms/52_Quantization_Calibration_and_Error.md) |
| 训练后量化 | post-training quantization, PTQ | 不重新训练完整模型，使用校准或权重统计生成低精度产物 | QLoRA、QAT | [02 PTQ 与 QAT](./02_ptq_and_qat_timing.md) |
| 量化感知训练 | quantization-aware training, QAT | 训练时用伪量化误差调整浮点主权重 | 直接以整数权重训练 | [QUANT-QAT](../../02_PyTorch_Algorithms/QUANT-QAT_Quantization_Aware_Training.md) |
| 伪量化 | fake quantization | 前向模拟裁剪和舍入、反向使用近似梯度的浮点训练操作 | 真实 INT8 / INT4 计算内核 | [QUANT-QAT](../../02_PyTorch_Algorithms/QUANT-QAT_Quantization_Aware_Training.md) |
| 参考量化表示 | reference quantized representation | 用可解释、可转换的张量表示验证量化数值或训练质量闭环，但不主张已经获得紧凑存储或生产计算内核 | 可部署量化产物、真实低精度计算内核 | [QUANT-QAT](../../02_PyTorch_Algorithms/QUANT-QAT_Quantization_Aware_Training.md) |
| 统计观测器 | observer | 收集权重或激活范围并生成量化参数的训练期模块 | 性能分析中的运行时观测工具 | [QUANT-QAT](../../02_PyTorch_Algorithms/QUANT-QAT_Quantization_Aware_Training.md) |
| 直通估计器 | straight-through estimator, STE | 为不可导的舍入操作指定近似反向梯度 | 自动求导得到的精确梯度 | [QUANT-QAT](../../02_PyTorch_Algorithms/QUANT-QAT_Quantization_Aware_Training.md) |
| 仅权重量化 | weight-only quantization | 权重低比特存储，激活和累加通常仍使用浮点格式 | 纯整数端到端执行 | [04 权重量化](./04_weight_only_compression.md) |
| 激活平滑 | activation smoothing | 在保持线性层浮点等价的前提下，把激活异常值压力迁移到权重 | 普通归一化层 | [53 SmoothQuant](../../02_PyTorch_Algorithms/53_Activation_Quantization_and_SmoothQuant.md) |
| 混合位宽 | mixed-bit quantization | 不同层或组依据敏感度和后端支持使用不同位宽 | 混合精度训练 | [QUANT-MIXED-BIT](../../02_PyTorch_Algorithms/QUANT-MIXED-BIT_Mixed_Bit_Allocation.md) |
| 层敏感度 | layer sensitivity | 在固定校准协议下，将某层降到更低位宽后引入的质量变化，用于形成保护层排序 | 单层参数量、梯度大小 | [QUANT-MIXED-BIT](../../02_PyTorch_Algorithms/QUANT-MIXED-BIT_Mixed_Bit_Allocation.md) |
| 打包格式 | packing format | 量化码在张量底层的排列和存储布局，它决定可使用的张量子类与计算内核 | 位宽、量化粒度、文件格式 | [QUANT-MIXED-BIT](../../02_PyTorch_Algorithms/QUANT-MIXED-BIT_Mixed_Bit_Allocation.md) |
| KV 异轴量化 | asymmetric K/V axes | Key 和 Value 根据不同异常值分布选择不同统计轴 | 非对称量化 | [54 KV Cache 量化](../../02_PyTorch_Algorithms/54_KV_Cache_Quantization_Strategies.md) |
| 高精度残留窗口 | residual window | KV Cache 中暂不量化的最新 Token 区域 | Transformer 残差连接 | [54 KV Cache 量化](../../02_PyTorch_Algorithms/54_KV_Cache_Quantization_Strategies.md) |
| 量化产物 | quantized artifact | 量化权重、量化参数、格式和产物清单组成的可加载交付物 | 原始模型检查点、适配器 | [06 执行与基准测试](./06_deployment_and_benchmark_decision.md) |
| 产物清单 | artifact manifest | 记录模型来源、量化配置、文件、哈希和兼容条件的可追溯清单 | 单个权重文件、结果 JSON | [06 执行与基准测试](./06_deployment_and_benchmark_decision.md) |
| 合并模型 | merged model | 已将适配器增量合并回基础权重、可独立加载的模型产物 | 独立适配器、推理量化产物 | [03 低比特训练适配](./03_low_bit_training_adaptation.md) |
| 加载器 | loader | 读取量化产物并恢复模型结构与量化元数据的组件 | 执行后端、计算内核 | [06 执行与基准测试](./06_deployment_and_benchmark_decision.md) |
| 执行后端 | backend | 负责组织模型、缓存和计算内核运行的执行系统 | 加载器、模型格式 | [06 执行与基准测试](./06_deployment_and_benchmark_decision.md) |
| 计算内核 | kernel | 在目标设备上执行低精度计算或反量化的具体算子实现 | 数值格式、执行后端 | [06 执行与基准测试](./06_deployment_and_benchmark_decision.md) |
| 路径证据 | path evidence | 记录实际张量类型、打包格式、量化状态、性能剖析器事件或计算内核，证明候选确实命中声明路径 | 配置文件中写了目标数据类型 | [实验资产表](./experiment_assets.md) |
| 后端预检 | backend preflight | 在下载大模型和数据前，用最小真实算子验证设备、张量表示和目标执行路径是否可初始化 | 机制实验结果、性能结论 | [实验 Quickstart](./quickstart.md) |
| 浮点回退 | floating-point fallback | 执行后端不支持目标格式或计算内核时，转为反量化或普通浮点执行 | 正确命中低精度计算内核 | [41 FP8 运行时](../../02_PyTorch_Algorithms/41_FP8_Runtime_Low_Precision_Paths.md) |
| 工作负载 | workload | 用于形成结论的模型、输入长度、批大小、并发、生成长度和采样条件组合 | 单个样本、数据集名称 | [06 执行与基准测试](./06_deployment_and_benchmark_decision.md) |
| 基线 / 候选 | baseline / candidate | 在相同工作负载与环境下进行配对比较的参照方案和待评估方案 | 任意两次未对齐条件的运行 | [06 执行与基准测试](./06_deployment_and_benchmark_decision.md) |
| 证据等级 | evidence level | 区分机制模拟、产物生成、执行后端实测和项目决策的证据强度 | 用一个低层结果替代端到端结论 | [实验资产表](./experiment_assets.md) |

## 使用规则

- 正文首次出现术语时使用“中文主称（English）”，后续只使用中文主称或本表列出的通用缩写。
- 代码、反引号内的配置项、JSON 字段、官方类名、模型名和格式名保留原始英文；不要为了中文化改写可执行标识符。
- `PTQ`、`QAT`、`QLoRA`、`GPTQ`、`AWQ`、`SmoothQuant`、`FP8`、`INT8`、`INT4`、`KV Cache` 可直接使用；普通叙述不直接使用 artifact、backend、workload、baseline、candidate、scale、zero point、group size、observer、kernel、manifest、fallback。
- 标题可以保留中英对照；正文句子避免“比较 baseline 和 candidate 的 latency gain”一类半中半英表达。
- 新术语先判断是否属于全教程共同概念；共同概念更新全局术语表，只有量化语境中的差异才加入本页。
- 同名概念存在多种含义时，在“容易混淆的概念”中明确区分，不通过新增近义词解决。
- 术语修订必须逐处判断语义，不对 Markdown、代码注释或 Notebook JSON 做无上下文批量替换。`scale`、`zero_point`、`group_size`、Attention 的 K/V、配置键和结果字段属于受保护标识符。
- 修改后回查变量级提示、题目/答案/测试标识符、K/V 语义和相邻重复词；例如 `# scale = ???` 不得改成中文变量名，Value 不得按普通“数值”替换。
