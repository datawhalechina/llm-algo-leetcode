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
            text: '0A Python 基础',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/00_Prerequisites/0A' },
              { text: '00. Python Essentials for LLM', link: '/00_Prerequisites/00_Python_Essentials_for_LLM' },
              { text: '01. NumPy and Einsum', link: '/00_Prerequisites/01_NumPy_and_Einsum' }
            ]
          },
          {
            text: '0B PyTorch 基础',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/00_Prerequisites/0B' },
              { text: '02. PyTorch Tensor Fundamentals', link: '/00_Prerequisites/02_PyTorch_Tensor_Fundamentals' },
              { text: '03. PyTorch Autograd and Backward', link: '/00_Prerequisites/03_PyTorch_Autograd_and_Backward' },
              { text: '04. PyTorch nn.Module Basics', link: '/00_Prerequisites/04_PyTorch_nn_Module_Basics' },
              { text: '05. PyTorch Optimizers and Loss', link: '/00_Prerequisites/05_PyTorch_Optimizers_and_Loss' }
            ]
          },
          {
            text: '0C 深度学习基础',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/00_Prerequisites/0C' },
              { text: '06. Simple Neural Network Training', link: '/00_Prerequisites/06_Simple_Neural_Network_Training' },
              { text: '07. Activation Functions', link: '/00_Prerequisites/07_Activation_Functions' },
              { text: '08. Normalization Techniques', link: '/00_Prerequisites/08_Normalization_Techniques' },
              { text: '09. Attention Mechanism Intro', link: '/00_Prerequisites/09_Attention_Mechanism_Intro' }
            ]
          },
          {
            text: '0D 工具与调试',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/00_Prerequisites/0D' },
              { text: '10. PyTorch Profiling Basics', link: '/00_Prerequisites/10_PyTorch_Profiling_Basics' },
              { text: '11. Memory Profiling and Optimization', link: '/00_Prerequisites/11_Memory_Profiling_and_Optimization' },
              { text: '12. Debugging Techniques', link: '/00_Prerequisites/12_Debugging_Techniques' },
              { text: '13. Jupyter and Git Basics', link: '/00_Prerequisites/13_Jupyter_and_Git_Basics' }
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
              { text: '01. Data Types and Precision', link: '/01_Hardware_Math_and_Systems/01_Data_Types_and_Precision' },
              { text: '02. LLM Params and FLOPs', link: '/01_Hardware_Math_and_Systems/02_LLM_Params_and_FLOPs' },
              { text: '21. Quantization Theory and INT4/INT8', link: '/01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8' },
              { text: '22. MoE Parameter and Compute', link: '/01_Hardware_Math_and_Systems/22_MoE_Parameter_and_Compute' }
            ]
          },
          {
            text: '1B 单卡硬件与访存优化',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/01_Hardware_Math_and_Systems/1B' },
              { text: '03. GPU Architecture and Memory', link: '/01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory' },
              { text: '04. Attention Memory Optimization', link: '/01_Hardware_Math_and_Systems/04_Attention_Memory_Optimization' },
              { text: '11. KV Cache and Memory Growth', link: '/01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth' },
              { text: '12. TensorCore and Mixed Precision', link: '/01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision' },
              { text: '13. Profiling and Bottleneck Analysis', link: '/01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis' },
              { text: '14. FlashAttention Memory Model', link: '/01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model' },
              { text: '23. TensorCore Deep Dive', link: '/01_Hardware_Math_and_Systems/23_TensorCore_Deep_Dive' },
              { text: '24. SRAM Optimization Techniques', link: '/01_Hardware_Math_and_Systems/24_SRAM_Optimization_Techniques' },
              { text: '25. Sparse Computation and Sparse Attention', link: '/01_Hardware_Math_and_Systems/25_Sparse_Computation_and_Sparse_Attention' }
            ]
          },
          {
            text: '1C 多卡通信与显存共享',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/01_Hardware_Math_and_Systems/1C' },
              { text: '05. Communication Topologies', link: '/01_Hardware_Math_and_Systems/05_Communication_Topologies' },
              { text: '06. VRAM Calculation and ZeRO', link: '/01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO' },
              { text: '20. NCCL and AllReduce Basics', link: '/01_Hardware_Math_and_Systems/20_NCCL_and_AllReduce_Basics' },
              { text: '26. Parallel Strategy Decision Framework', link: '/01_Hardware_Math_and_Systems/26_Parallel_Strategy_Decision_Framework' },
              { text: '27. Communication Scheduling Optimization', link: '/01_Hardware_Math_and_Systems/27_Communication_Scheduling_Optimization' },
              { text: '28. Fault Tolerance and Checkpointing', link: '/01_Hardware_Math_and_Systems/28_Fault_Tolerance_and_Checkpointing' }
            ]
          },
          {
            text: '1D 异构调度与算子编程',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/01_Hardware_Math_and_Systems/1D' },
              { text: '07. CPU GPU Heterogeneous Scheduling', link: '/01_Hardware_Math_and_Systems/07_CPU_GPU_Heterogeneous_Scheduling' },
              { text: '08. Programming Models CUDA Triton', link: '/01_Hardware_Math_and_Systems/08_Programming_Models_CUDA_Triton' },
              { text: '15. CUDA Execution Model', link: '/01_Hardware_Math_and_Systems/15_CUDA_Execution_Model' },
              { text: '16. Warp Block SharedMemory Basics', link: '/01_Hardware_Math_and_Systems/16_Warp_Block_SharedMemory_Basics' },
              { text: '17. CUDA Stream and Asynchrony', link: '/01_Hardware_Math_and_Systems/17_CUDA_Stream_and_Asynchrony' },
              { text: '18. Triton Block Model', link: '/01_Hardware_Math_and_Systems/18_Triton_Block_Model' },
              { text: '19. Operator Fusion Introduction', link: '/01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction' },
              { text: '29. CUDA Stream Advanced Scheduling', link: '/01_Hardware_Math_and_Systems/29_CUDA_Stream_Advanced_Scheduling' },
              { text: '30. Dynamic Shape Handling', link: '/01_Hardware_Math_and_Systems/30_Dynamic_Shape_Handling' },
              { text: '31. GPU Virtualization and MIG', link: '/01_Hardware_Math_and_Systems/31_GPU_Virtualization_and_MIG' }
            ]
          },
          {
            text: '1E 编译优化与算力生态',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/01_Hardware_Math_and_Systems/1E' },
              { text: '09. AI Compilers and Graph Optimization', link: '/01_Hardware_Math_and_Systems/09_AI_Compilers_and_Graph_Optimization' },
              { text: '10. Domestic AI Chips Overview', link: '/01_Hardware_Math_and_Systems/10_Domestic_AI_Chips_Overview' },
              { text: '32. TVM MLIR Deep Practice', link: '/01_Hardware_Math_and_Systems/32_TVM_MLIR_Deep_Practice' },
              { text: '33. TCO and Cost Model', link: '/01_Hardware_Math_and_Systems/33_TCO_and_Cost_Model' }
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
              { text: '00. PyTorch Warmup', link: '/02_PyTorch_Algorithms/00_PyTorch_Warmup' },
              { text: '01. RMSNorm Tutorial', link: '/02_PyTorch_Algorithms/01_RMSNorm_Tutorial' },
              { text: '02. SwiGLU Activation', link: '/02_PyTorch_Algorithms/02_SwiGLU_Activation' },
              { text: '03. RoPE Tutorial', link: '/02_PyTorch_Algorithms/03_RoPE_Tutorial' },
              { text: '04. Attention MHA GQA', link: '/02_PyTorch_Algorithms/04_Attention_MHA_GQA' }
            ]
          },
          {
            text: '2.2 模型架构',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/02_PyTorch_Algorithms/2_2' },
              { text: '05. LLaMA3 Block Tutorial', link: '/02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial' },
              { text: '06. MoE Router', link: '/02_PyTorch_Algorithms/06_MoE_Router' },
              { text: '07. MoE Load Balancing Loss', link: '/02_PyTorch_Algorithms/07_MoE_Load_Balancing_Loss' },
              { text: '08. Architecture Tricks', link: '/02_PyTorch_Algorithms/08_Architecture_Tricks' }
            ]
          },
          {
            text: '2.3 微调与训练技术',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/02_PyTorch_Algorithms/2_3' },
              { text: '09. SFT Training Loop', link: '/02_PyTorch_Algorithms/09_SFT_Training_Loop' },
              { text: '10. LoRA Tutorial', link: '/02_PyTorch_Algorithms/10_LoRA_Tutorial' },
              { text: '11. LR Schedulers WSD Cosine', link: '/02_PyTorch_Algorithms/11_LR_Schedulers_WSD_Cosine' }
            ]
          },
          {
            text: '2.4 对齐技术',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/02_PyTorch_Algorithms/2_4' },
              { text: '12. RLHF PPO Memory', link: '/02_PyTorch_Algorithms/12_RLHF_PPO_Memory' },
              { text: '13. DPO Loss Tutorial', link: '/02_PyTorch_Algorithms/13_DPO_Loss_Tutorial' }
            ]
          },
          {
            text: '2.5 反向传播与显存优化',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/02_PyTorch_Algorithms/2_5' },
              { text: '14. Attention Backward Math', link: '/02_PyTorch_Algorithms/14_Attention_Backward_Math' }
            ]
          },
          {
            text: '2.6 核心推理优化',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/02_PyTorch_Algorithms/2_6' },
              { text: '15. FlashAttention Sim', link: '/02_PyTorch_Algorithms/15_FlashAttention_Sim' },
              { text: '16. Decoding Strategies', link: '/02_PyTorch_Algorithms/16_Decoding_Strategies' },
              { text: '17. vLLM PagedAttention', link: '/02_PyTorch_Algorithms/17_vLLM_PagedAttention' }
            ]
          },
          {
            text: '2.7 高级推理优化',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/02_PyTorch_Algorithms/2_7' },
              { text: '18. Speculative Decoding', link: '/02_PyTorch_Algorithms/18_Speculative_Decoding' },
              { text: '19. SGLang RadixAttention', link: '/02_PyTorch_Algorithms/19_SGLang_RadixAttention' },
              { text: '20. Quantization W8A16', link: '/02_PyTorch_Algorithms/20_Quantization_W8A16' }
            ]
          },
          {
            text: '2.8 分布式与扩展',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/02_PyTorch_Algorithms/2_8' },
              { text: '21. Gradient Checkpointing', link: '/02_PyTorch_Algorithms/21_Gradient_Checkpointing' },
              { text: '22. QLoRA and 4bit Quantization', link: '/02_PyTorch_Algorithms/22_QLoRA_and_4bit_Quantization' },
              { text: '23. ZeRO Optimizer Sim', link: '/02_PyTorch_Algorithms/23_ZeRO_Optimizer_Sim' },
              { text: '24. Tensor Parallelism Sim', link: '/02_PyTorch_Algorithms/24_Tensor_Parallelism_Sim' },
              { text: '25. Pipeline Parallelism MicroBatch', link: '/02_PyTorch_Algorithms/25_Pipeline_Parallelism_MicroBatch' }
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
            text: '4.1 CUDA 编程基础',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/04_CUDA_and_System_Optimization/4_1' },
              { text: '15. CUDA Custom Kernel Intro', link: '/04_CUDA_and_System_Optimization/15_CUDA_Custom_Kernel_Intro' },
              { text: '16. CUDA Shared Memory Optimization', link: '/04_CUDA_and_System_Optimization/16_CUDA_Shared_Memory_Optimization' }
            ]
          },
          {
            text: '4.2 系统级性能优化',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/04_CUDA_and_System_Optimization/4_2' },
              { text: '17. CUDA Streams and Transfer', link: '/04_CUDA_and_System_Optimization/17_PyTorch_CUDA_Streams_and_Transfer' },
              { text: '18. CUDA Graph and JIT Compile', link: '/04_CUDA_and_System_Optimization/18_CUDA_Graph_and_JIT_Compile' }
            ]
          },
          {
            text: '4.3 分布式训练工程',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/04_CUDA_and_System_Optimization/4_3' },
              { text: '19. Distributed Communication Primitives', link: '/04_CUDA_and_System_Optimization/19_Distributed_Communication_Primitives' },
              { text: '20. DeepSpeed ZeRO & Offload', link: '/04_CUDA_and_System_Optimization/20_DeepSpeed_Zero_Config' }
            ]
          },
          {
            text: '4.4 架构视野',
            collapsed: true,
            items: [
              { text: '📖 组页', link: '/04_CUDA_and_System_Optimization/4_4' },
              { text: '21. CUDA vs Triton vs PyTorch', link: '/04_CUDA_and_System_Optimization/21_CUDA_vs_Triton_vs_PyTorch' },
              { text: '22. TCO and Hardware Selection', link: '/04_CUDA_and_System_Optimization/22_TCO_and_Hardware_Selection' }
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
