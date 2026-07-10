import { defineConfig } from 'vitepress'

const isEdgeOne = process.env.EDGEONE === '1'
const baseConfig = isEdgeOne ? '/' : '/llm-algo-leetcode/'

export default defineConfig({
  lang: 'zh-CN',
  title: "大模型算法实战教程",
  description: "面向大模型入门到进阶的算法实战教程",
  base: baseConfig,
  ignoreDeadLinks: true,
  markdown: {
    math: true
  },
  themeConfig: {
    logo: '/datawhale-logo.png',
    nav: [
      { text: '第零部分\n前置知识与环境准备', link: '/00_Prerequisites/intro' },
      { text: '第一部分\n硬件与系统基础', link: '/01_Hardware_Math_and_Systems/intro' },
      { text: '第二部分\nPyTorch 核心算法', link: '/02_PyTorch_Algorithms/intro' },
      { text: '第三部分\nTriton 算子开发', link: '/03_Triton_Kernels/intro' },
      { text: '第四部分\nCUDA C++ 与系统优化', link: '/04_CUDA_and_System_Optimization/intro' },
    ],
    search: {
      provider: 'local',
      options: {
        translations: {
          button: {
            buttonText: '搜索',
            buttonAriaLabel: '搜索'
          },
          modal: {
            noResultsText: '无法找到相关结果',
            resetButtonTitle: '清除查询条件',
            footer: {
              selectText: '选择',
              navigateText: '切换'
            }
          }
        }
      }
    },
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
        text: '第零部分：前置知识与环境准备',
        items: [
          { text: '📖 完整导学', link: '/00_Prerequisites/intro' },
          {
            text: '0A 基础语言与数据表示',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/00_Prerequisites/0A' },
              { text: '01. Python 基础与数据表示', link: '/00_Prerequisites/01_Python_Essentials_for_LLM' },
              { text: '02. NumPy 与 Einsum', link: '/00_Prerequisites/02_NumPy_and_Einsum' },
              { text: '03. Python 面向对象与工具模式', link: '/00_Prerequisites/03_Python_OOP_and_Utility_Patterns' },
              { text: '04. Python 配置与 I/O 模式', link: '/00_Prerequisites/04_Python_Config_and_IO_Patterns' }
            ]
          },
          {
            text: '0B PyTorch 张量与自动求导',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/00_Prerequisites/0B' },
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
              { text: '📖 组页', link: '/00_Prerequisites/0C' },
              { text: '09. PyTorch nn.Module 基础', link: '/00_Prerequisites/09_PyTorch_nn_Module_Basics' },
              { text: '10. PyTorch 状态管理与持久化', link: '/00_Prerequisites/10_PyTorch_State_dict_and_Persistence' },
              { text: '11. PyTorch 优化器与损失函数', link: '/00_Prerequisites/11_PyTorch_Optimizers_and_Loss' },
              { text: '12. PyTorch 最小训练接口', link: '/00_Prerequisites/12_PyTorch_Minimal_Training_Interface' }
            ]
          },
          {
            text: '0D 训练与模型直觉',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/00_Prerequisites/0D' },
              { text: '13. 简单神经网络训练循环', link: '/00_Prerequisites/13_Simple_Neural_Network_Training' },
              { text: '14. 激活函数', link: '/00_Prerequisites/14_Activation_Functions' },
              { text: '15. 归一化技术', link: '/00_Prerequisites/15_Normalization_Techniques' },
              { text: '16. Attention 机制导论', link: '/00_Prerequisites/16_Attention_Mechanism_Intro' }
            ]
          },
          {
            text: '0E 调试与性能',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/00_Prerequisites/0E' },
              { text: '17. PyTorch 性能分析基础', link: '/00_Prerequisites/17_PyTorch_Profiling_Basics' },
              { text: '18. 显存分析与优化', link: '/00_Prerequisites/18_Memory_Profiling_and_Optimization' },
              { text: '19. 调试与异常定位', link: '/00_Prerequisites/19_Debugging_and_Anomaly_Localization' },
              { text: '20. 性能剖析与显存账本', link: '/00_Prerequisites/20_Profiling_and_Memory_Ledger' }
            ]
          }
        ]
      },
      {
        text: '第一部分：硬件与系统基础',
        items: [
          { text: '📖 完整导学', link: '/01_Hardware_Math_and_Systems/intro' },
          {
            text: '1A 数值基础与算力估算',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/01_Hardware_Math_and_Systems/1A' },
              { text: '01. 大模型的数据格式与混合精度', link: '/01_Hardware_Math_and_Systems/01_Data_Types_and_Precision' },
              { text: '02. 大模型参数量与 FLOPs', link: '/01_Hardware_Math_and_Systems/02_LLM_Params_and_FLOPs' },
              { text: '21. 量化理论与 INT4/INT8', link: '/01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8' },
              { text: '22. MoE 参数量与计算量关系', link: '/01_Hardware_Math_and_Systems/22_MoE_Parameter_and_Compute' }
            ]
          },
          {
            text: '1B 单卡硬件与访存优化',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/01_Hardware_Math_and_Systems/1B' },
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
            text: '1C 多卡通信与显存共享',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/01_Hardware_Math_and_Systems/1C' },
              { text: '05. 通信拓扑', link: '/01_Hardware_Math_and_Systems/05_Communication_Topologies' },
              { text: '06. 显存计算与 ZeRO', link: '/01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO' },
              { text: '20. NCCL 与 AllReduce 基础', link: '/01_Hardware_Math_and_Systems/20_NCCL_and_AllReduce_Basics' },
              { text: '26. 并行策略决策框架', link: '/01_Hardware_Math_and_Systems/26_Parallel_Strategy_Decision_Framework' },
              { text: '27. 通信调度优化', link: '/01_Hardware_Math_and_Systems/27_Communication_Scheduling_Optimization' },
              { text: '28. 容错与 Checkpointing', link: '/01_Hardware_Math_and_Systems/28_Fault_Tolerance_and_Checkpointing' }
            ]
          },
          {
            text: '1D 异构调度与算子编程',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/01_Hardware_Math_and_Systems/1D' },
              { text: '07. CPU/GPU 异构调度', link: '/01_Hardware_Math_and_Systems/07_CPU_GPU_Heterogeneous_Scheduling' },
              { text: '08. CUDA/Triton 编程模型', link: '/01_Hardware_Math_and_Systems/08_Programming_Models_CUDA_Triton' },
              { text: '15. CUDA 执行模型', link: '/01_Hardware_Math_and_Systems/15_CUDA_Execution_Model' },
              { text: '16. Warp / Block / Shared Memory', link: '/01_Hardware_Math_and_Systems/16_Warp_Block_SharedMemory_Basics' },
              { text: '17. CUDA Stream 与异步', link: '/01_Hardware_Math_and_Systems/17_CUDA_Stream_and_Asynchrony' },
              { text: '18. Triton 块模型', link: '/01_Hardware_Math_and_Systems/18_Triton_Block_Model' },
              { text: '19. 算子融合导论', link: '/01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction' },
              { text: '29. CUDA Stream 高级调度', link: '/01_Hardware_Math_and_Systems/29_CUDA_Stream_Advanced_Scheduling' },
              { text: '30. 动态 Shape 处理', link: '/01_Hardware_Math_and_Systems/30_Dynamic_Shape_Handling' },
              { text: '31. GPU 虚拟化与 MIG', link: '/01_Hardware_Math_and_Systems/31_GPU_Virtualization_and_MIG' }
            ]
          },
          {
            text: '1E 编译优化与硬件生态',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/01_Hardware_Math_and_Systems/1E' },
              { text: '09. AI 编译器与图优化', link: '/01_Hardware_Math_and_Systems/09_AI_Compilers_and_Graph_Optimization' },
              { text: '10. 国产 AI 芯片概览', link: '/01_Hardware_Math_and_Systems/10_Domestic_AI_Chips_Overview' },
              { text: '32. TVM / MLIR 深度实践', link: '/01_Hardware_Math_and_Systems/32_TVM_MLIR_Deep_Practice' },
              { text: '33. TCO 与成本模型', link: '/01_Hardware_Math_and_Systems/33_TCO_and_Cost_Model' }
            ]
          }
        ]
      },
      {
        text: '第二部分：PyTorch 核心算法',
        items: [
          { text: '📖 完整导学', link: '/02_PyTorch_Algorithms/intro' },
          {
            text: '2.1 基础算子',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/02_PyTorch_Algorithms/2_1' },
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
              { text: '📖 组页', link: '/02_PyTorch_Algorithms/2_2' },
              { text: '05. LLaMA3 Block 教程', link: '/02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial' },
              { text: '06. MoE 路由器', link: '/02_PyTorch_Algorithms/06_MoE_Router' },
              { text: '07. MoE 负载均衡损失', link: '/02_PyTorch_Algorithms/07_MoE_Load_Balancing_Loss' },
              { text: '08. 架构技巧', link: '/02_PyTorch_Algorithms/08_Architecture_Tricks' }
            ]
          },
          {
            text: '2.3 微调与训练技术',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/02_PyTorch_Algorithms/2_3' },
              { text: '09. 监督微调训练循环', link: '/02_PyTorch_Algorithms/09_SFT_Training_Loop' },
              { text: '10. LoRA 教程', link: '/02_PyTorch_Algorithms/10_LoRA_Tutorial' },
              { text: '11. WSD 余弦学习率调度器', link: '/02_PyTorch_Algorithms/11_LR_Schedulers_WSD_Cosine' }
            ]
          },
          {
            text: '2.4 对齐技术',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/02_PyTorch_Algorithms/2_4' },
              { text: '12. 梯度累积', link: '/02_PyTorch_Algorithms/12_Gradient_Accumulation' },
              { text: '13. 直接偏好优化损失教程', link: '/02_PyTorch_Algorithms/13_DPO_Loss_Tutorial' }
            ]
          },
          {
            text: '2.5 反向传播与显存优化',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/02_PyTorch_Algorithms/2_5' },
              { text: '14. Attention 反向传播数学', link: '/02_PyTorch_Algorithms/14_Attention_Backward_Math' }
            ]
          },
          {
            text: '2.6 核心推理优化',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/02_PyTorch_Algorithms/2_6' },
              { text: '15. FlashAttention 模拟', link: '/02_PyTorch_Algorithms/15_FlashAttention_Sim' },
              { text: '16. 解码策略', link: '/02_PyTorch_Algorithms/16_Decoding_Strategies' },
              { text: '17. vLLM 分页注意力', link: '/02_PyTorch_Algorithms/17_vLLM_PagedAttention' }
            ]
          },
          {
            text: '2.7 高级推理优化',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/02_PyTorch_Algorithms/2_7' },
              { text: '18. 投机解码', link: '/02_PyTorch_Algorithms/18_Speculative_Decoding' },
              { text: '19. SGLang 基数注意力', link: '/02_PyTorch_Algorithms/19_SGLang_RadixAttention' },
              { text: '20. W8A16 量化', link: '/02_PyTorch_Algorithms/20_Quantization_W8A16' }
            ]
          },
          {
            text: '2.8 分布式与扩展',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/02_PyTorch_Algorithms/2_8' },
              { text: '21. 梯度检查点', link: '/02_PyTorch_Algorithms/21_Gradient_Checkpointing' },
              { text: '22. QLoRA 与 4-bit 量化', link: '/02_PyTorch_Algorithms/22_QLoRA_and_4bit_Quantization' },
              { text: '23. ZeRO 优化器模拟', link: '/02_PyTorch_Algorithms/23_ZeRO_Optimizer_Sim' },
              { text: '24. Tensor 并行模拟', link: '/02_PyTorch_Algorithms/24_Tensor_Parallelism_Sim' },
              { text: '25. Pipeline 并行微批次', link: '/02_PyTorch_Algorithms/25_Pipeline_Parallelism_MicroBatch' }
            ]
          }
        ]
      },
      {
        text: '第三部分：Triton 算子开发',
        items: [
          { text: '📖 完整导学', link: '/03_Triton_Kernels/intro' },
          {
            text: '3.1 基础篇',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/03_Triton_Kernels/3_1' },
              { text: '01. Triton Vector Addition', link: '/03_Triton_Kernels/01_Triton_Vector_Addition' },
              { text: '02. Triton Fused SwiGLU', link: '/03_Triton_Kernels/02_Triton_Fused_SwiGLU' },
              { text: '03. Triton Fused RMSNorm', link: '/03_Triton_Kernels/03_Triton_Fused_RMSNorm' },
              { text: '04. Triton GEMM Tutorial', link: '/03_Triton_Kernels/04_Triton_GEMM_Tutorial' },
              { text: '05. Triton Autotune and Profiling', link: '/03_Triton_Kernels/05_Triton_Autotune_and_Profiling' }
            ]
          },
          {
            text: '3.2 过渡篇',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/03_Triton_Kernels/3_2' },
              { text: '06. Triton Fused Softmax', link: '/03_Triton_Kernels/06_Triton_Fused_Softmax' },
              { text: '06.5 Triton Design Patterns', link: '/03_Triton_Kernels/06_5_Triton_Design_Patterns' },
            ]
          },
          {
            text: '3.3 进阶A：Attention优化',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/03_Triton_Kernels/3_3' },
              { text: '07. Triton Fused RoPE', link: '/03_Triton_Kernels/07_Triton_Fused_RoPE' },
              { text: '08. Triton Flash Attention', link: '/03_Triton_Kernels/08_Triton_Flash_Attention' },
              { text: '09. Triton PagedAttention', link: '/03_Triton_Kernels/09_Triton_PagedAttention' }
            ]
          },
          {
            text: '3.4 进阶B：推理优化',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/03_Triton_Kernels/3_4' },
              { text: '10. Triton Quantization', link: '/03_Triton_Kernels/10_Triton_Quantization' },
              { text: '11. Triton Multi-LoRA', link: '/03_Triton_Kernels/11_Triton_Multi_LoRA' }
            ]
          },
          {
            text: '3.5 项目篇',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/03_Triton_Kernels/3_5' },
              { text: '12. Triton Memory Model and Debug', link: '/03_Triton_Kernels/12_Triton_Memory_Model_and_Debug' },
              { text: '13. Triton Llama3 Block Project', link: '/03_Triton_Kernels/13_Triton_Llama3_Block_Project' },
              { text: '14. Triton Best Practices and FAQ', link: '/03_Triton_Kernels/14_Triton_Best_Practices_and_FAQ' }
            ]
          }
        ]
      },
      {
        text: '第四部分：CUDA C++ 与系统优化',
        items: [
          { text: '📖 完整导学', link: '/04_CUDA_and_System_Optimization/intro' },
          {
            text: '4.1 CUDA 编程基础（01-04）',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/04_CUDA_and_System_Optimization/4_1' },
              { text: '01. CUDA Custom Kernel Intro', link: '/04_CUDA_and_System_Optimization/15_CUDA_Custom_Kernel_Intro' },
              { text: '02. CUDA Shared Memory Optimization', link: '/04_CUDA_and_System_Optimization/16_CUDA_Shared_Memory_Optimization' },
              { text: '02.1 Bank Conflict Deep Dive', link: '/04_CUDA_and_System_Optimization/02_1_Bank_Conflict_Deep_Dive' },
              { text: '03. Tensor Core MMA Programming', link: '/04_CUDA_and_System_Optimization/03_Tensor_Core_MMA_Programming' },
              { text: '04. Warp-Level Primitives', link: '/04_CUDA_and_System_Optimization/04_Warp_Level_Primitives' }
            ]
          },
          {
            text: '4.2 系统级性能优化（05-08）',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/04_CUDA_and_System_Optimization/4_2' },
              { text: '05. CUDA Streams and Transfer', link: '/04_CUDA_and_System_Optimization/17_PyTorch_CUDA_Streams_and_Transfer' },
              { text: '06. CUDA Graph and JIT', link: '/04_CUDA_and_System_Optimization/18_CUDA_Graph_and_JIT_Compile' },
              { text: '07. 异步数据预取与 Double Buffering', link: '/04_CUDA_and_System_Optimization/07_Async_Data_Prefetch_and_Double_Buffering' },
              { text: '07.1 Double Buffering Deep Dive', link: '/04_CUDA_and_System_Optimization/07_1_Double_Buffering_Deep_Dive' },
              { text: '08. 内存池与显存管理', link: '/04_CUDA_and_System_Optimization/08_Memory_Pool_and_VRAM_Management' }
            ]
          },
          {
            text: '4.3 分布式训练工程（09-12）',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/04_CUDA_and_System_Optimization/4_3' },
              { text: '09. Distributed Communication Primitives', link: '/04_CUDA_and_System_Optimization/19_Distributed_Communication_Primitives' },
              { text: '09.1 Ring-AllReduce Deep Dive', link: '/04_CUDA_and_System_Optimization/09_1_Ring_AllReduce_Deep_Dive' },
              { text: '10. DeepSpeed ZeRO Config', link: '/04_CUDA_and_System_Optimization/20_DeepSpeed_Zero_Config' },
              { text: '11. 通信计算重叠高级调度', link: '/04_CUDA_and_System_Optimization/11_Communication_Computation_Overlap_Advanced_Scheduling' },
              { text: '12. 异构训练：CPU Offload 与 NVMe Offload', link: '/04_CUDA_and_System_Optimization/12_Heterogeneous_Training_CPU_Offload_NVMe_Offload' }
            ]
          },
          {
            text: '4.4 架构视野与总结（13-16）',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/04_CUDA_and_System_Optimization/4_4' },
              { text: '13. CUDA vs Triton vs PyTorch', link: '/04_CUDA_and_System_Optimization/21_CUDA_vs_Triton_vs_PyTorch' },
              { text: '14. TCO and Hardware Selection', link: '/04_CUDA_and_System_Optimization/22_TCO_and_Hardware_Selection' },
              { text: '15. 推理服务架构设计', link: '/04_CUDA_and_System_Optimization/15_Inference_Service_Architecture_Design' },
              { text: '16. 未来硬件趋势', link: '/04_CUDA_and_System_Optimization/16_Future_Hardware_Trends' }
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
