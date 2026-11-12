# 大模型架构演进实验资产表

本表登记架构机制与项目使用的 Profile、工作负载、结果路径和证据状态。环境创建见[实验 Quickstart](./quickstart.md)，Profile 与隔离关系见[环境资产索引](../../docs/gpu_environment_assets.md)，实际依赖与硬件写入结果 JSON；项目公共外层见[Part 02 结果资产与证据契约](../../docs/part02_result_assets.md)。

## 环境角色

| 环境 ID | 直接入口 | 可形成的证据 |
|:---|:---|:---|
| `ARCH-CPU` | 00–08、44、45、ARCH-MLA、ARCH-HYBRID-MEMORY 的 Step1–4 | 形状、状态、访问、容量和配置审计机制 |
| `ARCH-CUDA-MECHANISM` | 44、45、ARCH-MLA、ARCH-HYBRID-MEMORY | 单卡访问模式、状态增长、KV 表示与混合层计划证据 |
| 项目指定 Transformers 环境 | 60、71、77、78 | 固定模型快照、分词器、输入和长度阶梯的质量—成本证据 |
| 独立推理服务后端 | 71 或后续部署扩展 | 受支持架构在 vLLM / SGLang 等后端的真实加载和执行证据 |
| 通信与并行环境 | 80 | MoE 专家容量、分发 / 合并、All-to-All 和最慢进程证据 |

`ARCH-CUDA-MECHANISM` 不安装推理服务、量化或多卡编排依赖。机制页先验证局部结构，项目页再固定真实模型快照和执行栈；不能把跨模型观察写成单变量架构因果结论。

环境预检已于 2026-10-08 在 NVIDIA GeForce RTX 5070 Ti Laptop GPU 上通过：PyTorch 2.11.0+cu128、CUDA 12.8、11.48 GiB 显存、原生 BF16 可用，`pip check`、仓库运行时导入和 Jupyter Kernel 注册均成功。44、45、ARCH-MLA 与 ARCH-HYBRID-MEMORY 已分别生成单卡机制结果；这些记录不替代真实模型质量、专用 Kernel 和生产后端验证。

## 验证顺序

| 阶段 | 固定项 | 本阶段只回答什么 | 不得替代的证据 |
|:---|:---|:---|:---|
| CPU 机制 | 形状、随机种子、候选机制 | 数据流、状态、访问和容量不变量是否成立 | 真实 GPU 性能与模型质量 |
| 单卡机制 GPU | 环境配置、数据类型、形状与长度阶梯 | 目标机制在 GPU 上的状态、峰值显存和局部时间怎样变化 | 真实模型快照的端到端质量 |
| 架构专项项目 | 模型快照、分词器、输入、长度和候选 | 单变量结构候选能否形成质量—成本证据 | 推理服务后端是否支持生产执行 |
| 后端与多卡验证 | 产物、加载器、执行后端、硬件拓扑 | 目标路径是否命中，通信与服务代价是否可接受 | 未执行路径的推测收益 |
| 架构审计 | 前述结果的稳定引用与哈希 | 声明、实例化和运行证据是否支持采用决策 | 重新复制并执行上游机制代码 |

### 单卡机制执行顺序

四个实验复用同一 `ARCH-CUDA-MECHANISM` 环境，但每一步使用新的 `RUN_LABEL` 和独立结果文件。前一步失败时先保留失败记录并修复环境或执行路径，不跳过后继续汇总为架构结论。

| 顺序 | 页面与工作负载 | 先验证它的理由 | 进入下一步的条件 |
|:---|:---|:---|:---|
| 0 | Profile 安装与预检 | 先排除 Python、PyTorch、CUDA、仓库导入和 Kernel 选择问题 | `pip check` 无冲突；Profile 返回 `status=ok`；Notebook 的 `sys.executable` 指向目标环境 |
| 1 | [Part 02 · 44](../../02_PyTorch_Algorithms/44_Local_Sliding_and_Sparse_Attention.ipynb) · `arch-window-sdpa-v1` | 使用最小的配对 SDPA 矩阵确认 CUDA 计时、峰值显存与显式掩码路径 | 稠密与窗口结果均有限；记录实际执行限制，不把显式掩码等同于专用稀疏 Kernel |
| 2 | [Part 02 · 45](../../02_PyTorch_Algorithms/45_Linear_Attention_and_Recurrent_State.ipynb) · `arch-linear-recurrence-v1` | 在单层上确认显式历史随长度增长、递推状态保持固定 | 三档长度均生成状态字节、单步时间与峰值显存记录 |
| 3 | [Part 02 · ARCH-MLA](../../02_PyTorch_Algorithms/ARCH-MLA_Multihead_Latent_Attention.ipynb) · `arch-mla-representation-v2` | 复用前一步的状态账本，再对照 MHA、GQA、潜在状态重建与吸收式代理路径 | `enable_gqa` 可用；四类路径使用同长度、同 dtype；吸收式代理还需检查与重建路径的语义一致性 |
| 4 | [Part 02 · ARCH-HYBRID-MEMORY](../../02_PyTorch_Algorithms/ARCH-HYBRID-MEMORY_Attention_SSM_Hybrid.ipynb) · `arch-hybrid-memory-growth-v1` | 最后组合注意力历史、递推状态与混合层计划，避免在单机制未验证时直接解释组合结果 | 全注意力、递推和混合候选均完成长度阶梯；只形成局部状态与时间证据 |

默认配置按单卡 8 GB 以上环境设计；是否适合当前 GPU 仍以 Profile 预检和实际峰值显存为准。首次验证使用默认形状，不先扩大序列长度、批次或注意力头数。

## 模型与工作负载角色

| 角色 | 固定方式 | 使用规则 |
|:---|:---|:---|
| 合成机制输入 | 固定形状、长度阶梯、dtype、随机种子和候选集合 | 用于局部结构不变量与状态成本，不产生模型质量结论 |
| Transformers 语义基线 | 固定模型 / 分词器版本、输入词元和生成参数 | 验证配置、模型语义与输出；切换模型后新建结果 |
| 架构专项模型快照 | 项目页登记支持目标结构的真实模型 | MLA、SSM、稀疏注意力等不得用不匹配模型快照强行比较 |
| 推理服务后端候选 | 固定同一模型、输入、硬件和输出约束 | 能力检查未通过时记录失败，不静默回退 |

## 机制实验注册表

| 实验 ID / 页面 | 固定对照 | 主要指标 | 当前状态 |
|:---|:---|:---|:---|
| `ARCH-ATTENTION-ACCESS-MECHANISM` / 44 | 稠密访问 → 窗口访问 | 可见连接、实际 SDPA 路径、显式掩码、显存、延迟 | 2026-10-08 已完成四档长度验证：`benchmarks/results/44_sparse_attention/gpu_sdpa_window_rtx5070ti_window_ladder_v2.json`，并保留两档 v1；访问约束成立，但显式掩码与额外峰值按长度快速增长，判定为 `investigate_mask_backend`；专用稀疏计算内核仍由后续项目验证 |
| `ARCH-RECURRENT-STATE-MECHANISM` / 45 | 显式历史 → 固定递推状态 | 状态字节、长度增长、单步更新时间 | 2026-10-08 已完成候选独立分配验证：`benchmarks/results/45_linear_recurrent_state_rtx5070ti_recurrent_v2.json`；固定状态为 0.0625 MiB，长度增长时状态、时间与峰值保持稳定，判定为 `accept_fixed_state_growth_mechanism`；v1 仅保留状态与计时初步记录，其峰值因候选共驻不进入结论 |
| `ARCH-MLA-REPRESENTATION-MECHANISM` / ARCH-MLA | MHA / GQA → 潜在状态重建 / 吸收式代理 | 每词元字节数、物化 K/V 字节、语义一致性、峰值显存、执行时间 | `rtx5070ti_mla_v2` 验证潜在状态压缩与重建代价；`rtx5070ti_mla_v3` 验证吸收式代理不物化 K/V、余弦约 0.99999，长度 8192 的峰值比重建低约 30.5%，但通用算子路径未加速；结果位于 `benchmarks/results/arch_mla_representation_*.json`，真实模型快照与执行后端由 71 验证 |
| `ARCH-HYBRID-MEMORY-MECHANISM` / ARCH-HYBRID-MEMORY | 全注意力 / 递推 / 混合层计划 | 状态增长、层计划、峰值显存、单步时间 | RTX 5070 Ti 已验证：四层递推状态固定为 0.25 MiB；一层注意力的混合候选在长度 8192 时状态约 16.19 MiB，对比全注意力 64 MiB；判定 `accept_hybrid_state_growth_mechanism`，真实质量与专用计算内核证据由 77 验证 |

05 的单个 Transformer 块组合顺序、06/07 的路由与负载反馈先由 CPU 机制测试验证；多层 Transformer 块稳定性进入 60，多卡词元分发进入 80。01/02 的计算内核性能归算子优化，03 的真实长上下文质量进入 78/77。

## 项目接口

| 稳定项目 ID | 输入机制资产 | 项目补充的证据 | 当前职责 |
|:---|:---|:---|:---|
| `ARCH-BLOCK-STABILITY` / 60 | 01、02、05、08 | 模型快照匹配、训练稳定性和质量—成本 | 接收局部机制结果，补充多层稳定性证据 |
| `ARCH-ATTENTION-ACCESS` / 78 | 03、44 | 长程覆盖、真实质量和执行后端能力 | 接收访问模式结果；专用稀疏计算内核尚未接入 |
| `ARCH-MLA-KV` / 71 | 04、ARCH-MLA | 真实 MLA 模型快照、KV 容量、重建与执行后端 | 接收 MLA 表示结果；尚缺真实执行后端证据 |
| `ARCH-LONG-SEQUENCE-MEMORY` / 77 | 45、ARCH-HYBRID-MEMORY | 长度阶梯上的质量、状态与成本 | 局部证据生产者；默认属于跨模型观察 |
| `ARCH-MODEL-AUDIT` / 61 | 前述项目的稳定引用与哈希 | 配置差异、证据等级和综合判断 | 聚合出口；不重新执行局部 benchmark |
| `PAR-MOE-EXPERT` / 80 | 06、07 | 专家容量、容量溢出、分发和通信 | 通信与并行主归属，架构审计引用其结果 |

60、71、77、78 以 `architecture-evaluation/v1` 写出生产者记录；61 只消费已登记的语义 ID、结果路径和 SHA-256，不读取上游 Notebook 的内部变量。受控 GPU 对照、跨模型观察和匹配质量基准分别登记为 `controlled_gpu_benchmark`、`observational_gpu_benchmark` 和 `matched_quality_benchmark`；质量或后端证据缺失时保持 `tune`，不得仅因代码成功运行而进入 `accept`。

## 产物与执行后端准入

| 检查项 | 必须记录 | 未通过时的处理 |
|:---|:---|:---|
| 模型产物 | 模型 / 分词器版本、配置来源、权重格式与许可证 | 不进入真实模型比较 |
| 加载器 | 加载器及版本、远程代码开关、加载日志 | 记录 capability failure，不更换模型后沿用原结论 |
| 执行后端 | Transformers / vLLM / SGLang 等实际后端与版本 | 不把语义基线结果写成推理服务性能 |
| 目标路径 | 注意力、稀疏、SSM、MoE 或回退路径的运行证据 | 回退后结果单独登记，不声称候选机制已执行 |
| 结果产物 | JSON 路径、schema、run label、原始日志和哈希 | 不进入 Part 02 · 61 聚合审计 |

## 公共结果字段

| 字段组 | 必须记录 |
|:---|:---|
| 身份 | `semantic_id`、`workload_id`、`run_label`、页面与候选 |
| 模型语义 | 模型 / 分词器版本、配置来源、输入与长度阶梯 |
| 环境 | Profile ID、Python、PyTorch、CUDA、GPU、dtype 与执行栈 |
| 正确性或质量 | 形状/状态不变量、数值误差、任务质量或明确的未测状态 |
| 资源与性能 | 状态字节、峰值显存、延迟/吞吐及其统计口径 |
| 路径证据 | 实际注意力 / 状态 / 执行后端路径、能力检查与回退记录 |
| 结论 | `failure`、`evidence_level`、decision 与下一步 |
