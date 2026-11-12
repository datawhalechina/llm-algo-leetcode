import { defineConfig } from 'vitepress'
import { createHash } from 'node:crypto'
import { existsSync, readFileSync } from 'node:fs'

const mermaidCachePath = new URL('./cache/mermaid-cache.json', import.meta.url)
const mermaidCache = existsSync(mermaidCachePath)
  ? JSON.parse(readFileSync(mermaidCachePath, 'utf8')) as Record<string, string>
  : {}

const escapeHtml = (value: string) =>
  value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')

const renderMermaidSvg = (content: string) => {
  const hash = createHash('sha256').update(content, 'utf8').digest('hex')
  const svg = mermaidCache[hash]
  if (svg) {
    return `<div class="mermaid-snapshot">\n${svg}\n</div>\n`
  }

  return `<pre class="mermaid-fallback"><code>${escapeHtml(content)}</code></pre>\n`
}

const isEdgeOne = process.env.EDGEONE === '1'
const baseConfig = isEdgeOne ? '/' : '/llm-algo-leetcode/'

export default defineConfig({
  lang: 'zh-CN',
  title: "大模型算法与系统教程",
  description: "面向大模型算法与系统的可运行 Notebook 教程",
  base: baseConfig,
  ignoreDeadLinks: true,
  markdown: {
    math: true,
    config(md) {
      const defaultFence = md.renderer.rules.fence
      md.renderer.rules.fence = (tokens, idx, options, env, self) => {
        const token = tokens[idx]
        if (token.info.trim() === 'mermaid') {
          return renderMermaidSvg(token.content)
        }
        return defaultFence
          ? defaultFence(tokens, idx, options, env, self)
          : self.renderToken(tokens, idx, options)
      }
    }
  },
  themeConfig: {
    logo: '/datawhale-logo.png',
    nav: [
      { text: '第零部分\n前置知识与环境准备', link: '/00_Prerequisites/intro' },
      { text: '第一部分\n硬件与系统基础', link: '/01_Hardware_Math_and_Systems/intro' },
      { text: '第二部分\nPyTorch 核心算法', link: '/02_PyTorch_Algorithms/intro' },
      { text: '第三部分\nTriton 算子开发', link: '/03_Triton_Kernels/intro' },
      { text: '第四部分\nCUDA C++ 与系统优化', link: '/04_CUDA_and_System_Optimization/intro' },
      { text: '专题讨论', link: '/topic_discussion/intro' },
      { text: '组队学习', link: '/team_study/intro' },
    ],
    sidebar: [
      {
        text: '介绍',
        items: [
          { text: '项目概览', link: '/' },
          { text: '使用指南', link: '/guide' },
          { text: '贡献指南', link: '/contributing' },
          { text: '维护与发布手册', link: '/maintenance' }
        ]
      },
      {
        text: '专题讨论',
        items: [
          { text: '专题总览', link: '/topic_discussion/intro' },
          {
            text: '主学习路线：模型架构与训练',
            collapsed: true,
            items: [
              { text: '专题导读与 Task0–6', link: '/topic_discussion/model_architecture_and_training/intro' },
              { text: '从模型结构到一次训练更新', link: '/topic_discussion/model_architecture_and_training/walkthrough' }
            ]
          },
          {
            text: '主学习路线：后训练优化',
            collapsed: true,
            items: [
              { text: '专题导读与 Task0–6', link: '/topic_discussion/post_training_optimization/intro' },
              { text: '01. SFT 数据、监督位置与长样本处理', link: '/topic_discussion/post_training_optimization/01_sft_data_and_loss' },
              { text: '02. LoRA、PEFT 与训练对象', link: '/topic_discussion/post_training_optimization/02_lora_peft_design' },
              { text: '03. 训练控制与低资源 SFT', link: '/topic_discussion/post_training_optimization/03_training_control' },
              { text: '04. SFT 基线、升级条件与训练交付', link: '/topic_discussion/post_training_optimization/04_end_to_end_experiment' },
              { text: '05. DPO、KTO 与离线反馈优化', link: '/topic_discussion/post_training_optimization/05_dpo_and_preference_optimization' },
              { text: '06. 奖励信号、PPO 与 RLHF', link: '/topic_discussion/post_training_optimization/06_rlhf_and_ppo_system_cost' },
              { text: '07. GRPO、RLOO 与组内奖励', link: '/topic_discussion/post_training_optimization/07_grpo_and_groupwise_alignment' },
              { text: '08. 轨迹生命周期与环境交互', link: '/topic_discussion/post_training_optimization/08_trajectory_lifecycle_and_environment_interaction' },
              { text: '09. 在线训练交付与执行系统', link: '/topic_discussion/post_training_optimization/09_rl_execution_system_and_efficiency' },
              { text: '10. 独立评测、能力回归与采用决策', link: '/topic_discussion/post_training_optimization/10_evaluation_and_adoption_decision' },
              { text: '问题链与图册', link: '/topic_discussion/post_training_optimization/walkthrough' },
              { text: '判断手册', link: '/topic_discussion/post_training_optimization/casebook' },
              { text: '术语表', link: '/topic_discussion/post_training_optimization/glossary' },
              { text: '公式索引', link: '/topic_discussion/post_training_optimization/formula_reference' },
              { text: '环境 Quickstart', link: '/topic_discussion/post_training_optimization/quickstart' },
              { text: '实验资产', link: '/topic_discussion/post_training_optimization/experiment_assets' }
            ]
          },
          {
            text: '主学习路线：推理优化',
            collapsed: true,
            items: [
              { text: '专题导读与 Task0–6', link: '/topic_discussion/inference_optimization/intro' },
              { text: '01. 请求链路与指标', link: '/topic_discussion/inference_optimization/01_request_path_and_metrics' },
              { text: '02. 模型前向与计算执行', link: '/topic_discussion/inference_optimization/02_prefill_and_attention_kernel' },
              { text: '03. 解码策略', link: '/topic_discussion/inference_optimization/03_decoding_strategies' },
              { text: '04. KV Cache 表示、管理与复用', link: '/topic_discussion/inference_optimization/04_kv_cache_lifecycle_and_reuse' },
              { text: '05. 服务预算、连续批处理与阶段调度', link: '/topic_discussion/inference_optimization/05_serving_scheduling_and_pd' },
              { text: '06. 量化推理与部署', link: '/topic_discussion/inference_optimization/06_quantized_inference_and_deployment' },
              { text: '07. 端到端基准与决策', link: '/topic_discussion/inference_optimization/07_benchmark_and_decision' },
              { text: '08. 分布式推理与并行扩展', link: '/topic_discussion/inference_optimization/08_distributed_inference_and_parallelism' },
              {
                text: '教学引擎实践',
                collapsed: true,
                items: [
                  { text: '阅读入口', link: '/topic_discussion/inference_optimization/teaching_engines' },
                  { text: 'nano-vLLM 请求跟踪', link: '/topic_discussion/inference_optimization/nano_vllm_request_walkthrough' },
                  { text: 'Mini-SGLang 缓存与执行', link: '/topic_discussion/inference_optimization/mini_sglang_cache_scheduling' }
                ]
              },
              { text: '问题链与图册', link: '/topic_discussion/inference_optimization/walkthrough' },
              { text: '判断手册', link: '/topic_discussion/inference_optimization/casebook' },
              { text: '术语表', link: '/topic_discussion/inference_optimization/glossary' },
              { text: '环境 Quickstart', link: '/topic_discussion/inference_optimization/quickstart' },
              { text: '实验资产', link: '/topic_discussion/inference_optimization/experiment_assets' }
            ]
          },
          {
            text: '主学习路线：性能优化',
            collapsed: true,
            items: [
              { text: '专题导读与 Task0–6', link: '/topic_discussion/performance_optimization/intro' },
              { text: '01. 明确优化目标与比较条件', link: '/topic_discussion/performance_optimization/01_performance_problem_and_benchmark' },
              { text: '02. 资源约束与候选代价', link: '/topic_discussion/performance_optimization/02_observability_and_profiling' },
              { text: '03. 训练输入、状态与资源账本', link: '/topic_discussion/performance_optimization/03_training_memory_ledger' },
              { text: '04. 推理状态容量与 KV Cache 成本', link: '/topic_discussion/performance_optimization/04_inference_memory_and_kv_cache' },
              { text: '05. 跨路径：计算、通信与 I/O', link: '/topic_discussion/performance_optimization/05_compute_communication_and_io' },
              { text: '06. 单候选执行与准入', link: '/topic_discussion/performance_optimization/06_cost_model_and_production_decision' },
              { text: '07. 策略组合与瓶颈转移', link: '/topic_discussion/performance_optimization/07_end_to_end_performance_engineering' },
              { text: '问题链与图册', link: '/topic_discussion/performance_optimization/walkthrough' },
              { text: '判断手册', link: '/topic_discussion/performance_optimization/casebook' },
              { text: '术语表', link: '/topic_discussion/performance_optimization/glossary' },
              { text: '环境 Quickstart', link: '/topic_discussion/performance_optimization/quickstart' },
              { text: '实验资产', link: '/topic_discussion/performance_optimization/experiment_assets' }
            ]
          },
          {
            text: '主学习路线：算子优化',
            collapsed: true,
            items: [
              { text: '专题导读与 Task0–6', link: '/topic_discussion/operator_optimization/intro' },
              { text: '01. 优化对象与比较基线', link: '/topic_discussion/operator_optimization/01_optimization_targets_and_baselines' },
              { text: '02. 图编译、融合与执行候选', link: '/topic_discussion/operator_optimization/02_graph_compilation_and_fusion' },
              { text: '03. 计算内核语义与访存', link: '/topic_discussion/operator_optimization/03_kernel_semantics_and_memory' },
              { text: '04. 矩阵计算与推理状态的数据路径', link: '/topic_discussion/operator_optimization/04_attention_and_inference_data_paths' },
              { text: '05. 测量归因与自动调优', link: '/topic_discussion/operator_optimization/05_measurement_profiling_and_autotuning' },
              { text: '06. 传输、提交与显存生命周期', link: '/topic_discussion/operator_optimization/06_cuda_execution_and_runtime_constraints' },
              { text: '07. 集成、基准与采用判断', link: '/topic_discussion/operator_optimization/07_integration_benchmarking_and_adoption' },
              { text: '问题链与图册', link: '/topic_discussion/operator_optimization/walkthrough' },
              { text: '判断手册', link: '/topic_discussion/operator_optimization/casebook' },
              { text: '术语表', link: '/topic_discussion/operator_optimization/glossary' },
              { text: '公式索引', link: '/topic_discussion/operator_optimization/formula_reference' },
              { text: '环境 Quickstart', link: '/topic_discussion/operator_optimization/quickstart' },
              { text: '实验资产', link: '/topic_discussion/operator_optimization/experiment_assets' }
            ]
          },
          {
            text: '横向工程专题',
            collapsed: true,
            items: [
              { text: '量化与压缩', link: '/topic_discussion/quantization_compression/intro' },
              { text: '通信与并行', link: '/topic_discussion/communication_parallelism/intro' },
              { text: '数据工程', link: '/topic_discussion/data_engineering/intro' },
              { text: '部署与异构系统', link: '/topic_discussion/deployment_heterogeneous/intro' }
            ]
          },
          {
            text: '领域专题',
            collapsed: true,
            items: [
              { text: 'Agent 基础设施', link: '/topic_discussion/agent/intro' },
              { text: '多模态与视觉推理', link: '/topic_discussion/multimodal/intro' }
            ]
          },
          {
            text: '独立专题与分析',
            collapsed: true,
            items: [
              { text: '模型架构演进', link: '/topic_discussion/llm_architecture_evolution/intro' },
              { text: '反向传播与训练机制', link: '/topic_discussion/backpropagation_training_mechanism/intro' },
              { text: '性能分析', link: '/topic_discussion/performance_analysis/intro' }
            ]
          }
        ]
      },
      { text: '组队学习', link: '/team_study/intro' },
      {
        text: '第零部分：前置知识',
        items: [
          { text: '部分导读', link: '/00_Prerequisites/intro' },
          {
            text: '0A 基础语言与数据表示',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/00_Prerequisites/0A' },
              { text: '01. Python 基础与数据表示', link: '/00_Prerequisites/01_Python_Essentials_for_LLM' },
              { text: '02. NumPy 与 Einsum', link: '/00_Prerequisites/02_NumPy_and_Einsum' },
              { text: '03. Python 面向对象与工具模式', link: '/00_Prerequisites/03_Python_OOP_and_Utility_Patterns' },
              { text: '04. Python 配置与数据入口', link: '/00_Prerequisites/04_Python_Config_and_Data_Entry' }
            ]
          },
          {
            text: '0B PyTorch 张量与自动求导',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/00_Prerequisites/0B' },
              { text: '05. PyTorch 张量基础操作', link: '/00_Prerequisites/05_PyTorch_Tensor_Fundamentals' },
              { text: '06. PyTorch 张量布局与索引', link: '/00_Prerequisites/06_PyTorch_Tensor_Layout_and_Indexing' },
              { text: '07. PyTorch 自动求导与反向传播', link: '/00_Prerequisites/07_PyTorch_Autograd_and_Backward' },
              { text: '08. PyTorch 梯度习惯与无梯度模式', link: '/00_Prerequisites/08_PyTorch_Grad_Hygiene_and_No_Grad' }
            ]
          },
          {
            text: '0C PyTorch 模型构建',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/00_Prerequisites/0C' },
              { text: '09. PyTorch nn.Module 基础', link: '/00_Prerequisites/09_PyTorch_nn_Module_Basics' },
              { text: '10. PyTorch 状态管理与持久化', link: '/00_Prerequisites/10_PyTorch_State_dict_and_Persistence' },
              { text: '11. PyTorch 优化器与损失函数', link: '/00_Prerequisites/11_PyTorch_Optimizers_and_Loss' },
              { text: '12. PyTorch 最小训练接口', link: '/00_Prerequisites/12_PyTorch_Minimal_Training_Interface' }
            ]
          },
          {
            text: '0D Transformer Block 数据流与组件基础',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/00_Prerequisites/0D' },
              { text: '13. Transformer Block 数据流与残差通路', link: '/00_Prerequisites/13_Transformer_Block_Dataflow_and_Residual_Stream' },
              { text: '14. 激活函数', link: '/00_Prerequisites/14_Activation_Functions' },
              { text: '15. 归一化技术', link: '/00_Prerequisites/15_Normalization_Techniques' },
              { text: '16. Attention 机制导论', link: '/00_Prerequisites/16_Attention_Mechanism_Intro' }
            ]
          },
          {
            text: '0E 运行诊断与实验基础',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/00_Prerequisites/0E' },
              { text: '17. PyTorch 性能分析基础', link: '/00_Prerequisites/17_PyTorch_Profiling_Basics' },
              { text: '18. 显存分析与优化', link: '/00_Prerequisites/18_Memory_Profiling_and_Optimization' },
              { text: '19. 调试与异常定位', link: '/00_Prerequisites/19_Debugging_and_Anomaly_Localization' },
              { text: '20. 性能证据与实验设计', link: '/00_Prerequisites/20_Performance_Evidence_and_Experiment_Design' }
            ]
          }
        ]
      },
      {
        text: '第一部分：硬件与系统基础',
        items: [
          { text: '部分导读', link: '/01_Hardware_Math_and_Systems/intro' },
          {
            text: '1A 数值与规模',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/01_Hardware_Math_and_Systems/1A' },
              { text: '01. 大模型的数据格式与混合精度', link: '/01_Hardware_Math_and_Systems/01_Data_Types_and_Precision' },
              { text: '02. 大模型参数量与 FLOPs', link: '/01_Hardware_Math_and_Systems/02_LLM_Params_and_FLOPs' },
              { text: '21. 量化理论与 INT4/INT8', link: '/01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8' },
              { text: '22. MoE 参数量与计算量关系', link: '/01_Hardware_Math_and_Systems/22_MoE_Parameter_and_Compute' }
            ]
          },
          {
            text: '1B 单设备执行、内存与访存',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/01_Hardware_Math_and_Systems/1B' },
              { text: '03. GPU 物理架构、内存层级与核心硬件单元', link: '/01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory' },
              { text: '04. Attention 显存优化', link: '/01_Hardware_Math_and_Systems/04_Attention_Memory_Optimization' },
              { text: '11. KV Cache 与显存增长', link: '/01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth' },
              { text: '12. TensorCore 与混合精度', link: '/01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision' },
              { text: '13. 性能分析与瓶颈定位', link: '/01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis' },
              { text: '14. FlashAttention 显存模型', link: '/01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model' },
              { text: '23. TensorCore 深度剖析', link: '/01_Hardware_Math_and_Systems/23_TensorCore_Deep_Dive' },
              { text: '24. SRAM 优化技巧', link: '/01_Hardware_Math_and_Systems/24_SRAM_Optimization_Techniques' },
              { text: '25. 稀疏计算与稀疏 Attention', link: '/01_Hardware_Math_and_Systems/25_Sparse_Computation_and_Sparse_Attention' }
            ]
          },
          {
            text: '1C 通信、并行与容错',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/01_Hardware_Math_and_Systems/1C' },
              { text: '05. 通信拓扑', link: '/01_Hardware_Math_and_Systems/05_Communication_Topologies' },
              { text: '06. 显存计算与 ZeRO', link: '/01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO' },
              { text: '20. NCCL 与 AllReduce 基础', link: '/01_Hardware_Math_and_Systems/20_NCCL_and_AllReduce_Basics' },
              { text: '26. 并行策略决策框架', link: '/01_Hardware_Math_and_Systems/26_Parallel_Strategy_Decision_Framework' },
              { text: '27. 通信调度优化', link: '/01_Hardware_Math_and_Systems/27_Communication_Scheduling_Optimization' },
              { text: '28. 容错与 Checkpointing', link: '/01_Hardware_Math_and_Systems/28_Fault_Tolerance_and_Checkpointing' }
            ]
          },
          {
            text: '1D 执行模型、异步调度与算子映射',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/01_Hardware_Math_and_Systems/1D' },
              { text: '07. CPU/GPU 异构调度', link: '/01_Hardware_Math_and_Systems/07_CPU_GPU_Heterogeneous_Scheduling' },
              { text: '08. CUDA/Triton 编程模型', link: '/01_Hardware_Math_and_Systems/08_Programming_Models_CUDA_Triton' },
              { text: '15. CUDA 执行模型', link: '/01_Hardware_Math_and_Systems/15_CUDA_Execution_Model' },
              { text: '16. Warp / Block / Shared Memory', link: '/01_Hardware_Math_and_Systems/16_Warp_Block_SharedMemory_Basics' },
              { text: '17. CUDA Stream 与异步', link: '/01_Hardware_Math_and_Systems/17_CUDA_Stream_and_Asynchrony' },
              { text: '18. Triton 块模型', link: '/01_Hardware_Math_and_Systems/18_Triton_Block_Model' },
              { text: '29. CUDA Stream 高级调度', link: '/01_Hardware_Math_and_Systems/29_CUDA_Stream_Advanced_Scheduling' }
            ]
          },
          {
            text: '1E 编译与后端',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/01_Hardware_Math_and_Systems/1E' },
              { text: '09. AI 编译器与图优化', link: '/01_Hardware_Math_and_Systems/09_AI_Compilers_and_Graph_Optimization' },
              { text: '19. 算子融合导论', link: '/01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction' },
              { text: '30. 动态 Shape 处理', link: '/01_Hardware_Math_and_Systems/30_Dynamic_Shape_Handling' },
              { text: '32. TVM / MLIR 深度实践', link: '/01_Hardware_Math_and_Systems/32_TVM_MLIR_Deep_Practice' }
            ]
          },
          {
            text: '1F 硬件适配与系统约束',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/01_Hardware_Math_and_Systems/1F' },
              { text: '10. 国产 AI 芯片概览', link: '/01_Hardware_Math_and_Systems/10_Domestic_AI_Chips_Overview' },
              { text: '31. GPU 虚拟化与 MIG', link: '/01_Hardware_Math_and_Systems/31_GPU_Virtualization_and_MIG' },
              { text: '33. TCO 与成本模型', link: '/01_Hardware_Math_and_Systems/33_TCO_and_Cost_Model' },
              { text: '34. 异构加速器与运行时约束', link: '/01_Hardware_Math_and_Systems/34_Heterogeneous_Accelerators_and_Runtime_Constraints' }
            ]
          }
        ]
      },
      {
        text: '第二部分：PyTorch 核心算法',
        items: [
          { text: '部分导读', link: '/02_PyTorch_Algorithms/intro' },
          {
            text: '2.1 Decoder 组件与注意力',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/02_PyTorch_Algorithms/2_1' },
              { text: '00. PyTorch 热身', link: '/02_PyTorch_Algorithms/00_PyTorch_Warmup' },
              { text: '01. RMSNorm 教程', link: '/02_PyTorch_Algorithms/01_RMSNorm_Tutorial' },
              { text: '02. SwiGLU 激活', link: '/02_PyTorch_Algorithms/02_SwiGLU_Activation' },
              { text: '03. 旋转位置编码教程', link: '/02_PyTorch_Algorithms/03_RoPE_Tutorial' },
              { text: '04. 多头注意力', link: '/02_PyTorch_Algorithms/04_Attention_MHA_GQA' }
            ]
          },
          {
            text: '2.2 模型架构',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/02_PyTorch_Algorithms/2_2' },
              { text: '05. LLaMA3 Block 教程', link: '/02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial' },
              { text: '06. MoE 路由器', link: '/02_PyTorch_Algorithms/06_MoE_Router' },
              { text: '07. MoE 负载均衡损失', link: '/02_PyTorch_Algorithms/07_MoE_Load_Balancing_Loss' },
              { text: '08. 架构技巧', link: '/02_PyTorch_Algorithms/08_Architecture_Tricks' },
              { text: '44. 局部、滑动窗口与稀疏注意力', link: '/02_PyTorch_Algorithms/44_Local_Sliding_and_Sparse_Attention' },
              { text: '45. 线性注意力与递推状态', link: '/02_PyTorch_Algorithms/45_Linear_Attention_and_Recurrent_State' },
              { text: 'ARCH-MLA. MLA 与紧凑 KV 表示', link: '/02_PyTorch_Algorithms/ARCH-MLA_Multihead_Latent_Attention' },
              { text: 'ARCH-HYBRID. Attention、SSM 与混合记忆', link: '/02_PyTorch_Algorithms/ARCH-HYBRID-MEMORY_Attention_SSM_Hybrid' }
            ]
          },
          {
            text: '2.3 反向传播与训练控制',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/02_PyTorch_Algorithms/2_3' },
              { text: '09. 激活与损失反向传播', link: '/02_PyTorch_Algorithms/09_Activation_and_Loss_Backward' },
              { text: '10. Attention 反向传播与自定义 Autograd', link: '/02_PyTorch_Algorithms/10_Attention_Backward_and_Custom_Autograd' },
              { text: '11. 梯度累积', link: '/02_PyTorch_Algorithms/11_Gradient_Accumulation' },
              { text: '混合精度反向传播与梯度安全', link: '/02_PyTorch_Algorithms/TRAIN-MIXED-PRECISION-BACKWARD_Mixed_Precision_Backward' },
              { text: '12. 优化器更新与学习率调度', link: '/02_PyTorch_Algorithms/12_Optimizer_Updates_and_Learning_Rate_Scheduling' },
              { text: '55. 训练更新动力学', link: '/02_PyTorch_Algorithms/55_Training_Update_Dynamics' },
              { text: '19. 激活检查点', link: '/02_PyTorch_Algorithms/19_Activation_Checkpointing' },
              { text: '42. 激活卸载', link: '/02_PyTorch_Algorithms/42_Activation_Offload' },
              { text: '43. 统一训练内存账本', link: '/02_PyTorch_Algorithms/43_Unified_Training_Memory_Ledger' },
              { text: '56. 持续预训练与能力保留', link: '/02_PyTorch_Algorithms/56_Continual_Pretraining_and_Retention' }
            ]
          },
          {
            text: '2.4 监督微调与参数适配',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/02_PyTorch_Algorithms/2_4' },
              { text: '33. 后训练准入与阶段选择', link: '/02_PyTorch_Algorithms/33_Fine_Tuning_Readiness' },
              { text: '32. SFT 数据契约与泄漏审计', link: '/02_PyTorch_Algorithms/32_Data_Engineering_for_SFT' },
              { text: '合成指令生成与验证准入', link: '/02_PyTorch_Algorithms/PT-SYNTHETIC-DATA_Instruction_Generation_and_Verifier_Admission' },
              { text: '13. SFT 监督边界与损失', link: '/02_PyTorch_Algorithms/13_SFT_Training_Loop' },
              { text: '15. SFT 有效更新与训练闭环', link: '/02_PyTorch_Algorithms/15_End_to_End_Fine_Tuning_Experiment' },
              { text: '14. 低秩适配教程', link: '/02_PyTorch_Algorithms/14_LoRA_Tutorial' },
              { text: '30. 长上下文微调机制', link: '/02_PyTorch_Algorithms/30_Long_Context_Fine_Tuning' },
              { text: '31. LoRA 变体缩放与预算', link: '/02_PyTorch_Algorithms/31_LoRA_Variants_Theory' }
            ]
          },
          {
            text: '2.5 偏好优化与对齐',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/02_PyTorch_Algorithms/2_5' },
              { text: '50. 偏好数据与对齐评测', link: '/02_PyTorch_Algorithms/50_Preference_Data_and_Evaluation' },
              { text: '奖励、价值与优势信号', link: '/02_PyTorch_Algorithms/PT-REWARD-SIGNAL_Reward_Model_Value_and_Advantage' },
              { text: '17. 直接偏好优化损失', link: '/02_PyTorch_Algorithms/17_DPO_Loss_Tutorial' },
              { text: '二元反馈对齐', link: '/02_PyTorch_Algorithms/PT-KTO_Binary_Feedback_Alignment' },
              { text: '奖励模型训练、校准与过度优化', link: '/02_PyTorch_Algorithms/PT-REWARD-MODEL_Reward_Model_Training_and_Calibration' },
              { text: '16. RLHF / PPO 资源与状态', link: '/02_PyTorch_Algorithms/16_RLHF_PPO_Memory' },
              { text: '18. 组相对策略优化与可验证奖励', link: '/02_PyTorch_Algorithms/18_GRPO_Loss_Tutorial' },
              { text: '留一基线策略优化', link: '/02_PyTorch_Algorithms/PT-RLOO_Leave_One_Out_Policy_Optimization' },
              { text: '轨迹与策略新鲜度', link: '/02_PyTorch_Algorithms/PT-ROLLOUT-LIFECYCLE_Rollout_and_Policy_Freshness' },
              { text: '环境轨迹与奖励归因', link: '/02_PyTorch_Algorithms/PT-AGENTIC-RL_Environment_Trajectory_and_Credit_Assignment' },
              { text: '采样、训练与奖励系统', link: '/02_PyTorch_Algorithms/PT-RL-SYSTEM_Rollout_Training_and_Reward_System' },
              { text: '51. 在线 DPO', link: '/02_PyTorch_Algorithms/51_Online_DPO' }
            ]
          },
          {
            text: '2.6 推理执行基础',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/02_PyTorch_Algorithms/2_6' },
              { text: '20. FlashAttention 模拟', link: '/02_PyTorch_Algorithms/20_FlashAttention_Sim' },
              { text: 'MoE 词元分组与专家执行', link: '/02_PyTorch_Algorithms/INF-MOE-EXECUTION_MoE_Inference_Dispatch_and_Expert_Execution' },
              { text: '21. 解码策略', link: '/02_PyTorch_Algorithms/21_Decoding_Strategies' },
              { text: '22. vLLM 分页注意力', link: '/02_PyTorch_Algorithms/22_vLLM_PagedAttention' }
            ]
          },
          {
            text: '2.7 推理策略与服务调度',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/02_PyTorch_Algorithms/2_7' },
              { text: '23. 投机解码', link: '/02_PyTorch_Algorithms/23_Speculative_Decoding' },
              { text: '24. Prefix Cache 匹配与复用', link: '/02_PyTorch_Algorithms/24_Prefix_Cache_Matching_and_Reuse' },
              { text: '34. SGLang 基数注意力', link: '/02_PyTorch_Algorithms/34_SGLang_RadixAttention' },
              { text: '35. Multi-Token Decoding', link: '/02_PyTorch_Algorithms/35_Multi_Token_Decoding' },
              { text: '36. Decode Scheduling', link: '/02_PyTorch_Algorithms/36_Decode_Scheduling' },
              { text: '37. KV Cache Lifecycle and Capacity Control', link: '/02_PyTorch_Algorithms/37_KV_Cache_Lifecycle_and_Capacity_Control' },
              { text: 'Continuous Batching', link: '/02_PyTorch_Algorithms/INF-CONTINUOUS-BATCHING_Continuous_Batching' },
              { text: 'Serving Overlap and State Handoff', link: '/02_PyTorch_Algorithms/INF-SERVING-OVERLAP_Serving_Overlap_and_State_Handoff' },
              { text: '38. Prefill/Decode 调度', link: '/02_PyTorch_Algorithms/38_Prefill_Decode_Scheduling' },
              { text: '39. 异构 PD 与服务分层', link: '/02_PyTorch_Algorithms/39_Hetero_PD_and_Serving_Tiers' }
            ]
          },
          {
            text: '2.8 量化与低比特适配',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/02_PyTorch_Algorithms/2_8' },
              { text: '25. W8A16 量化', link: '/02_PyTorch_Algorithms/25_Quantization_W8A16' },
              { text: '26. QLoRA 与 4-bit 量化', link: '/02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization' },
              { text: '40. GPTQ 与 AWQ 权重量化', link: '/02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization' },
              { text: '41. FP8 运行时低精度路径', link: '/02_PyTorch_Algorithms/41_FP8_Runtime_Low_Precision_Paths' },
              { text: '52. 量化校准与误差控制', link: '/02_PyTorch_Algorithms/52_Quantization_Calibration_and_Error' },
              { text: '53. 激活量化与 SmoothQuant', link: '/02_PyTorch_Algorithms/53_Activation_Quantization_and_SmoothQuant' },
              { text: '54. KV Cache 量化策略', link: '/02_PyTorch_Algorithms/54_KV_Cache_Quantization_Strategies' }
            ]
          },
          {
            text: '2.9 通信与并行',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/02_PyTorch_Algorithms/2_9' },
              { text: '27. ZeRO 优化器模拟', link: '/02_PyTorch_Algorithms/27_ZeRO_Optimizer_Sim' },
              { text: '28. Pipeline 并行微批次', link: '/02_PyTorch_Algorithms/28_Pipeline_Parallelism_MicroBatch' },
              { text: '29. Tensor 并行模拟', link: '/02_PyTorch_Algorithms/29_Tensor_Parallelism_Sim' },
              { text: '47. NCCL 通信 profiling', link: '/02_PyTorch_Algorithms/47_Communication_Profiling_with_NCCL' },
              { text: '46. MoE 专家并行', link: '/02_PyTorch_Algorithms/46_MoE_Expert_Parallel' },
              { text: '48. 通信热点与缓解策略', link: '/02_PyTorch_Algorithms/48_Communication_Hotspots_and_Mitigation' },
              { text: '49. 并行策略选型', link: '/02_PyTorch_Algorithms/49_Parallelism_Strategy_Selection' }
            ]
          },
          {
            text: '2.10 项目验证与工程收口',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/02_PyTorch_Algorithms/2_10' },
              { text: '60. Decoder Block 稳定性基准', link: '/02_PyTorch_Algorithms/60_Decoder_Block_Stability_Benchmark' },
              { text: '61. 架构验证项目', link: '/02_PyTorch_Algorithms/61_Model_Architecture_Exploration' },
              { text: '62. 指令微调项目', link: '/02_PyTorch_Algorithms/62_Instruction_Fine_Tuning_Project' },
              { text: '63. LoRA 变体对比', link: '/02_PyTorch_Algorithms/63_LoRA_Variants_Benchmark' },
              { text: '64. SFT 数据质量项目', link: '/02_PyTorch_Algorithms/64_SFT_Data_Quality_Project' },
              { text: '65. QLoRA 选型项目', link: '/02_PyTorch_Algorithms/65_QLoRA_Selection_Project' },
              { text: '66. 单卡推理测量基线', link: '/02_PyTorch_Algorithms/66_Inference_Performance_Comparison' },
              { text: '67. 量化推理与部署项目', link: '/02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment' },
              { text: '68. 推测解码基准', link: '/02_PyTorch_Algorithms/68_Speculative_Decoding_Benchmark' },
              { text: '69. 前缀缓存基准', link: '/02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark' },
              { text: '70. 推理服务调度基准', link: '/02_PyTorch_Algorithms/70_Serving_Scheduler_Benchmark' },
              { text: '71. MLA 与 KV Cache 结构基准', link: '/02_PyTorch_Algorithms/71_MLA_KV_Cache_Architecture_Benchmark' },
              { text: '72. 单卡推理部署与 Backend 对比', link: '/02_PyTorch_Algorithms/72_Multi_Model_Deployment_Comparison' },
              { text: '73. 训练性能分析', link: '/02_PyTorch_Algorithms/73_Training_Performance_Analysis' },
              { text: '74. Profiling 驱动端到端优化', link: '/02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization' },
              { text: '75. 优化器实现路径基准', link: '/02_PyTorch_Algorithms/75_Optimizer_Execution_Path_Benchmark' },
              { text: '76. Activation / Checkpoint / Offload 对比项目', link: '/02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark' },
              { text: '77. 长序列记忆架构基准', link: '/02_PyTorch_Algorithms/77_Long_Sequence_Memory_Architecture_Benchmark' },
              { text: '78. Attention 访问模式基准', link: '/02_PyTorch_Algorithms/78_Attention_Access_Pattern_Benchmark' },
              { text: '79. 分布式并行基准', link: '/02_PyTorch_Algorithms/79_Distributed_Parallel_Benchmark' },
              { text: '80. MoE 专家并行 benchmark', link: '/02_PyTorch_Algorithms/80_MoE_Expert_Parallel_Benchmark' },
              { text: '81. 分布式推理逻辑验证', link: '/02_PyTorch_Algorithms/81_Distributed_Inference_Project' },
              { text: '82. 量化产物评估项目', link: '/02_PyTorch_Algorithms/82_Quantization_Artifact_Evaluation_Project' },
              { text: '83. 量化 Serving Profile 基准', link: '/02_PyTorch_Algorithms/83_Quantized_Serving_Profile_Benchmark' },
              { text: '84. DPO 偏好优化项目', link: '/02_PyTorch_Algorithms/84_DPO_Preference_Project' },
              { text: '85. GRPO 组内对齐项目', link: '/02_PyTorch_Algorithms/85_GRPO_Groupwise_Alignment_Project' },
              { text: '86. 在线后训练生命周期与恢复', link: '/02_PyTorch_Algorithms/86_DPO_Online_Benchmark' }
            ]
          }
        ]
      },
      {
        text: '第三部分：Triton 算子开发',
        items: [
          { text: '部分导读', link: '/03_Triton_Kernels/intro' },
          {
            text: '3.1 编程模型与数据布局',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/03_Triton_Kernels/3_1' },
              { text: '01. Triton Vector Addition', link: '/03_Triton_Kernels/01_Triton_Vector_Addition' },
              { text: '02. Triton Layouts and Memory Access', link: '/03_Triton_Kernels/02_Triton_Layouts_and_Memory_Access' }
            ]
          },
          {
            text: '3.2 常用算子与调优',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/03_Triton_Kernels/3_2' },
              { text: '03. Triton Fused SwiGLU', link: '/03_Triton_Kernels/03_Triton_Fused_SwiGLU' },
              { text: '04. Triton Fused RMSNorm', link: '/03_Triton_Kernels/04_Triton_Fused_RMSNorm' },
              { text: '05. Triton GEMM Tutorial', link: '/03_Triton_Kernels/05_Triton_GEMM_Tutorial' },
              { text: '06. Persistent GEMM and Tile Scheduling', link: '/03_Triton_Kernels/06_Triton_Persistent_GEMM_and_Tile_Scheduling' },
              { text: '07. Triton Fused Softmax', link: '/03_Triton_Kernels/07_Triton_Fused_Softmax' },
              { text: '08. Triton Softmax Backward', link: '/03_Triton_Kernels/08_Triton_Softmax_Backward' },
              { text: '09. Triton Autotune and Profiling', link: '/03_Triton_Kernels/09_Triton_Autotune_and_Profiling' }
            ]
          },
          {
            text: '3.3 Attention 与序列计算',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/03_Triton_Kernels/3_3' },
              { text: '10. Triton Fused RoPE', link: '/03_Triton_Kernels/10_Triton_Fused_RoPE' },
              { text: '11. Triton Flash Attention', link: '/03_Triton_Kernels/11_Triton_Flash_Attention' },
              { text: '12. Triton Variable-Length GQA', link: '/03_Triton_Kernels/12_Triton_Variable_Length_GQA' },
              { text: '13. Triton PagedAttention', link: '/03_Triton_Kernels/13_Triton_PagedAttention' }
            ]
          },
          {
            text: '3.4 推理专用 kernel',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/03_Triton_Kernels/3_4' },
              { text: '14. Triton Quantization', link: '/03_Triton_Kernels/14_Triton_Quantization' },
              { text: '15. Triton Multi-LoRA', link: '/03_Triton_Kernels/15_Triton_Multi_LoRA' },
              { text: '16. Triton KV Cache Quantized Decode', link: '/03_Triton_Kernels/16_Triton_KV_Cache_Quantized_Decode' }
            ]
          },
          {
            text: '3.5 调试、编译与集成',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/03_Triton_Kernels/3_5' },
              { text: '17. Triton Memory Model and Debug', link: '/03_Triton_Kernels/17_Triton_Memory_Model_and_Debug' },
              { text: '18. Compilation Artifacts and Execution', link: '/03_Triton_Kernels/18_Triton_Compilation_Artifacts_and_Execution' },
              { text: '19. Triton PyTorch Op Integration', link: '/03_Triton_Kernels/19_Triton_PyTorch_Op_Integration' },
              { text: '20. Triton Llama3 Block Project', link: '/03_Triton_Kernels/20_Triton_Llama3_Block_Project' }
            ]
          }
        ]
      },
      {
        text: '第四部分：CUDA C++ 与系统优化',
        items: [
          { text: '部分导读', link: '/04_CUDA_and_System_Optimization/intro' },
          {
            text: '4.1 CUDA 执行、访存与资源',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/04_CUDA_and_System_Optimization/4_1' },
              { text: '01. CUDA Custom Kernel Intro', link: '/04_CUDA_and_System_Optimization/01_CUDA_Custom_Kernel_Intro' },
              { text: '02. CUDA Global Memory Coalescing', link: '/04_CUDA_and_System_Optimization/02_CUDA_Global_Memory_Coalescing' },
              { text: '03. Warp-Level Primitives', link: '/04_CUDA_and_System_Optimization/03_Warp_Level_Primitives' },
              { text: '04. Bank Conflict Deep Dive', link: '/04_CUDA_and_System_Optimization/04_Bank_Conflict_Deep_Dive' },
              { text: '05. Block Resources and Residency', link: '/04_CUDA_and_System_Optimization/05_CUDA_Block_Resources_and_Residency' }
            ]
          },
          {
            text: '4.2 分块计算与数据复用',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/04_CUDA_and_System_Optimization/4_2' },
              { text: '06. CUDA Shared Memory Optimization', link: '/04_CUDA_and_System_Optimization/06_CUDA_Shared_Memory_Optimization' },
              { text: '07. Tensor Core MMA Programming', link: '/04_CUDA_and_System_Optimization/07_Tensor_Core_MMA_Programming' },
              { text: '08. 异步数据预取与 Double Buffering', link: '/04_CUDA_and_System_Optimization/08_Async_Data_Prefetch_and_Double_Buffering' },
              { text: '09. CUDA Async Copy Pipeline', link: '/04_CUDA_and_System_Optimization/09_CUDA_Async_Copy_Pipeline' }
            ]
          },
          {
            text: '4.3 CUDA Runtime 与显存管理',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/04_CUDA_and_System_Optimization/4_3' },
              { text: '10. CUDA Streams and Transfer', link: '/04_CUDA_and_System_Optimization/10_PyTorch_CUDA_Streams_and_Transfer' },
              { text: '11. CUDA Graph and JIT', link: '/04_CUDA_and_System_Optimization/11_CUDA_Graph_and_JIT_Compile' },
              { text: '12. 内存池与显存管理', link: '/04_CUDA_and_System_Optimization/12_Memory_Pool_and_VRAM_Management' }
            ]
          },
          {
            text: '4.4 分布式训练系统',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/04_CUDA_and_System_Optimization/4_4' },
              { text: '13. Distributed Communication Primitives', link: '/04_CUDA_and_System_Optimization/13_Distributed_Communication_Primitives' },
              { text: '14. Ring-AllReduce Deep Dive', link: '/04_CUDA_and_System_Optimization/14_Ring_AllReduce_Deep_Dive' },
              { text: '15. 通信计算重叠高级调度', link: '/04_CUDA_and_System_Optimization/15_Communication_Computation_Overlap_Advanced_Scheduling' },
              { text: '16. DeepSpeed ZeRO Config', link: '/04_CUDA_and_System_Optimization/16_DeepSpeed_Zero_Config' },
              { text: '17. 异构训练：CPU Offload 与 NVMe Offload', link: '/04_CUDA_and_System_Optimization/17_Heterogeneous_Training_CPU_Offload_NVMe_Offload' }
            ]
          },
          {
            text: '4.5 实现方式与硬件选型',
            collapsed: true,
            items: [
              { text: '组内导读', link: '/04_CUDA_and_System_Optimization/4_5' },
              { text: '18. CUDA vs Triton vs PyTorch', link: '/04_CUDA_and_System_Optimization/18_CUDA_vs_Triton_vs_PyTorch' },
              { text: '19. TCO and Hardware Selection', link: '/04_CUDA_and_System_Optimization/19_TCO_and_Hardware_Selection' }
            ]
          }
        ]
      }
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/datawhalechina/llm-algo-leetcode' }
    ],
    editLink: {
      pattern: 'https://github.com/datawhalechina/llm-algo-leetcode/blob/main/docs/:path'
    },
    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2024-present'
    }
  }
})
