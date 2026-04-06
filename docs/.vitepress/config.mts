import { defineConfig } from 'vitepress'

const isEdgeOne = process.env.EDGEONE === '1'
const baseConfig = isEdgeOne ? '/' : '/llm-algo-leetcode/'

export default defineConfig({
  lang: 'zh-CN',
  title: "LLM-Algo-LeetCode",
  description: "大语言模型算法与系统实战库",
  base: baseConfig,
  ignoreDeadLinks: true,
  markdown: {
    math: true
  },
  themeConfig: {
    logo: '/datawhale-logo.png',
    nav: [
      { text: '开始刷题', link: '/01_Hardware_Math_and_Systems/01_Data_Types_and_Precision' },
      { text: 'GitHub 仓库', link: 'https://github.com/lynnyulinlin-debug/llm-algo-leetcode' },
    ],
    search: {
      provider: 'local',
      options: {
        translations: {
          button: {
            buttonText: '搜索文档',
            buttonAriaLabel: '搜索文档'
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
          { text: '项目概览', link: '/' }
        ]
      },
      {
        text: '第一部分：硬件与系统基础',
        items: [
          { text: '01. Data Types and Precision', link: '/01_Hardware_Math_and_Systems/01_Data_Types_and_Precision' },
          { text: '02. LLM Params and FLOPs', link: '/01_Hardware_Math_and_Systems/02_LLM_Params_and_FLOPs' },
          { text: '03. GPU Architecture and Memory', link: '/01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory' },
          { text: '04. Attention Memory Optimization', link: '/01_Hardware_Math_and_Systems/04_Attention_Memory_Optimization' },
          { text: '05. Communication Topologies', link: '/01_Hardware_Math_and_Systems/05_Communication_Topologies' },
          { text: '06. VRAM Calculation and ZeRO', link: '/01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO' },
          { text: '07. CPU GPU Heterogeneous Scheduling', link: '/01_Hardware_Math_and_Systems/07_CPU_GPU_Heterogeneous_Scheduling' },
          { text: '08. Programming Models CUDA Triton', link: '/01_Hardware_Math_and_Systems/08_Programming_Models_CUDA_Triton' },
          { text: '09. AI Compilers and Graph Optimization', link: '/01_Hardware_Math_and_Systems/09_AI_Compilers_and_Graph_Optimization' },
          { text: '10. Domestic AI Chips Overview', link: '/01_Hardware_Math_and_Systems/10_Domestic_AI_Chips_Overview' }
        ]
      },
      {
        text: '第二部分：PyTorch 核心算法',
        items: [
          { text: '00. PyTorch Warmup', link: '/02_PyTorch_Algorithms/00_PyTorch_Warmup' },
          { text: '01. RMSNorm Tutorial', link: '/02_PyTorch_Algorithms/01_RMSNorm_Tutorial' },
          { text: '02. SwiGLU Activation', link: '/02_PyTorch_Algorithms/02_SwiGLU_Activation' },
          { text: '03. RoPE Tutorial', link: '/02_PyTorch_Algorithms/03_RoPE_Tutorial' },
          { text: '04. Attention MHA GQA', link: '/02_PyTorch_Algorithms/04_Attention_MHA_GQA' },
          { text: '05. LLaMA3 Block Tutorial', link: '/02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial' },
          { text: '06. MoE Router', link: '/02_PyTorch_Algorithms/06_MoE_Router' },
          { text: '07. MoE Load Balancing Loss', link: '/02_PyTorch_Algorithms/07_MoE_Load_Balancing_Loss' },
          { text: '08. Architecture Tricks', link: '/02_PyTorch_Algorithms/08_Architecture_Tricks' },
          { text: '09. LoRA Tutorial', link: '/02_PyTorch_Algorithms/09_LoRA_Tutorial' },
          { text: '10. SFT Training Loop', link: '/02_PyTorch_Algorithms/10_SFT_Training_Loop' },
          { text: '11. LR Schedulers WSD Cosine', link: '/02_PyTorch_Algorithms/11_LR_Schedulers_WSD_Cosine' },
          { text: '12. DPO Loss Tutorial', link: '/02_PyTorch_Algorithms/12_DPO_Loss_Tutorial' },
          { text: '13. Attention 反向传播推导', link: '/02_PyTorch_Algorithms/13_Attention_Backward_Math' },
          { text: '14. FlashAttention Sim', link: '/02_PyTorch_Algorithms/14_FlashAttention_Sim' },
          { text: '15. Decoding Strategies', link: '/02_PyTorch_Algorithms/15_Decoding_Strategies' },
          { text: '16. vLLM PagedAttention', link: '/02_PyTorch_Algorithms/16_vLLM_PagedAttention' },
          { text: '17. SGLang RadixAttention', link: '/02_PyTorch_Algorithms/17_SGLang_RadixAttention' },
          { text: '18. Quantization W8A16', link: '/02_PyTorch_Algorithms/18_Quantization_W8A16' },
          { text: '19. Gradient Checkpointing', link: '/02_PyTorch_Algorithms/19_Gradient_Checkpointing' },
          { text: '20. QLoRA and 4bit Quantization', link: '/02_PyTorch_Algorithms/20_QLoRA_and_4bit_Quantization' },
          { text: '21. ZeRO 1 Optimizer Sim', link: '/02_PyTorch_Algorithms/21_ZeRO_1_Optimizer_Sim' },
          { text: '22. Tensor Parallelism Sim', link: '/02_PyTorch_Algorithms/22_Tensor_Parallelism_Sim' }
        ]
      },
      {
        text: '第三部分：CUDA 与 Triton 算子',
        items: [
          { text: '01. Triton Vector Addition', link: '/03_CUDA_and_Triton_Kernels/01_Triton_Vector_Addition' },
          { text: '02. Triton Fused SwiGLU', link: '/03_CUDA_and_Triton_Kernels/02_Triton_Fused_SwiGLU' },
          { text: '03. Triton Fused RMSNorm', link: '/03_CUDA_and_Triton_Kernels/03_Triton_Fused_RMSNorm' },
          { text: '04. Triton GEMM Tutorial', link: '/03_CUDA_and_Triton_Kernels/04_Triton_GEMM_Tutorial' },
          { text: '05. Triton Autotune and Profiling', link: '/03_CUDA_and_Triton_Kernels/05_Triton_Autotune_and_Profiling' },
          { text: '06. Triton Fused Softmax', link: '/03_CUDA_and_Triton_Kernels/06_Triton_Fused_Softmax' },
          { text: '07. Triton Fused RoPE', link: '/03_CUDA_and_Triton_Kernels/07_Triton_Fused_RoPE' },
          { text: '08. Triton Flash Attention', link: '/03_CUDA_and_Triton_Kernels/08_Triton_Flash_Attention' },
          { text: '09. Triton Fused LoRA', link: '/03_CUDA_and_Triton_Kernels/09_Triton_Fused_LoRA' },
          { text: '10. Triton KV Cache and PagedAttention', link: '/03_CUDA_and_Triton_Kernels/10_Triton_KV_Cache_and_PagedAttention' },
          { text: '11. Triton Quantization Support', link: '/03_CUDA_and_Triton_Kernels/11_Triton_Quantization_Support' },
          { text: '12. Triton Memory Model and Debug', link: '/03_CUDA_and_Triton_Kernels/12_Triton_Memory_Model_and_Debug' },
          { text: '14. Triton Llama3 Block Project', link: '/03_CUDA_and_Triton_Kernels/14_Triton_Llama3_Block_Project' },
          { text: '15. Triton Best Practices and FAQ', link: '/03_CUDA_and_Triton_Kernels/15_Triton_Best_Practices_and_FAQ' },
          { text: '16. PyTorch CUDA Streams and Transfer', link: '/03_CUDA_and_Triton_Kernels/16_PyTorch_CUDA_Streams_and_Transfer' },
          { text: '18. Distributed Communication Primitives', link: '/03_CUDA_and_Triton_Kernels/18_Distributed_Communication_Primitives' },
          { text: '19. DeepSpeed Zero Config', link: '/03_CUDA_and_Triton_Kernels/19_DeepSpeed_Zero_Config' },
          { text: '20. CUDA Custom Kernel Intro', link: '/03_CUDA_and_Triton_Kernels/20_CUDA_Custom_Kernel_Intro' },
          { text: '21. CUDA Shared Memory Optimization', link: '/03_CUDA_and_Triton_Kernels/21_CUDA_Shared_Memory_Optimization' },
          { text: '22. CUDA vs Triton vs PyTorch', link: '/03_CUDA_and_Triton_Kernels/22_CUDA_vs_Triton_vs_PyTorch' }
        ]
      }
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/lynnyulinlin-debug/llm-algo-leetcode' }
    ],
    editLink: {
      pattern: 'https://github.com/lynnyulinlin-debug/llm-algo-leetcode/blob/main/docs/:path'
    },
    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2024-present'
    }
  }
})
