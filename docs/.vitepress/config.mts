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
{ text: '00. PyTorch 核心基础热身: 张量、前反向传播与 Embedding (Warmup)', link: '/02_PyTorch_Algorithms/00_PyTorch_Warmup' },
          { text: '01. 均方根层归一化 (RMSNorm)', link: '/02_PyTorch_Algorithms/01_RMSNorm_Tutorial' },
          { text: '02. 激活函数与门控机制 (SwiGLU Activation)', link: '/02_PyTorch_Algorithms/02_SwiGLU_Activation' },
          { text: '03. 旋转位置编码 (RoPE)', link: '/02_PyTorch_Algorithms/03_RoPE_Tutorial' },
          { text: '04. 注意力机制与键值缓存 (MHA / GQA / MQA)', link: '/02_PyTorch_Algorithms/04_Attention_MHA_GQA' },
          { text: '05. 经典模型搭建: LLaMA-3 Transformer Block', link: '/02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial' },
          { text: '06. 混合专家架构: 稀疏路由与负载均衡 (MoE)', link: '/02_PyTorch_Algorithms/06_MoE_Router' },
          { text: '07. MoE 进阶：负载均衡损失函数 (Load Balancing Loss)', link: '/02_PyTorch_Algorithms/07_MoE_Load_Balancing_Loss' },
          { text: '08. 经典架构变体：Qwen 与 Gemma 的核心机制 (Architecture Tricks)', link: '/02_PyTorch_Algorithms/08_Architecture_Tricks' },
          { text: '09. 监督微调训练框架: 数据构造与 Loss Masking (SFT Training Loop)', link: '/02_PyTorch_Algorithms/09_SFT_Training_Loop' },
          { text: '10. 参数高效微调: 深入剖析 LoRA (PEFT)', link: '/02_PyTorch_Algorithms/10_LoRA_Tutorial' },
          { text: '11. 大模型训练调参难点：学习率调度器 (Warmup, Cosine, WSD)', link: '/02_PyTorch_Algorithms/11_LR_Schedulers_WSD_Cosine' },
          { text: '12. RLHF 对齐：PPO 算法的核心 Loss 与显存流转 (RLHF PPO)', link: '/02_PyTorch_Algorithms/12_RLHF_PPO_Memory' },
          { text: '13. 直接偏好优化 Loss 源码解析与实现 (DPO)', link: '/02_PyTorch_Algorithms/13_DPO_Loss_Tutorial' },
          { text: '14. 注意力机制反向传播推导与自定义 Autograd (Attention Backward)', link: '/02_PyTorch_Algorithms/14_Attention_Backward_Math' },
          { text: '15. 深入理解 FlashAttention：分块计算与 Online Softmax', link: '/02_PyTorch_Algorithms/15_FlashAttention_Sim' },
          { text: '16. 大模型解码策略：Top-K, Top-p (Nucleus) 与 Temperature', link: '/02_PyTorch_Algorithms/16_Decoding_Strategies' },
          { text: '17. 经典推理框架: 模拟 Continuous Batching 与 PagedAttention', link: '/02_PyTorch_Algorithms/17_vLLM_PagedAttention' },
          { text: '18. 投机解码：打破推理的访存瓶颈 (Speculative Decoding)', link: '/02_PyTorch_Algorithms/18_Speculative_Decoding' },
          { text: '19. SGLang 与 RadixAttention: 突破 vLLM 多轮对话瓶颈', link: '/02_PyTorch_Algorithms/19_SGLang_RadixAttention' },
          { text: '20. 模型量化基础: INT8 绝对最大值量化与反量化 (Quantization)', link: '/02_PyTorch_Algorithms/20_Quantization_W8A16' },
          { text: '21. 极致显存优化：激活值重计算 (Gradient Checkpointing)', link: '/02_PyTorch_Algorithms/21_Gradient_Checkpointing' },
          { text: '22. QLoRA 与 4-bit NormalFloat 量化核心机制 (QLoRA & 4bit)', link: '/02_PyTorch_Algorithms/22_QLoRA_and_4bit_Quantization' },
          { text: '23. 显存优化进阶：模拟 ZeRO-1 切分与 ZeRO 原理 (ZeRO Optimizer)', link: '/02_PyTorch_Algorithms/23_ZeRO_Optimizer_Sim' },
          { text: '24. 突破单卡显存上限：张量并行 (Tensor Parallelism, TP) 的矩阵切片模拟', link: '/02_PyTorch_Algorithms/24_Tensor_Parallelism_Sim' },
          { text: '25. 分布式进阶：流水线并行与微批次气泡 (Pipeline Parallelism)', link: '/02_PyTorch_Algorithms/25_Pipeline_Parallelism_MicroBatch' },
                ]
      },
      {
        text: '第三部分：CUDA 与 Triton 算子',
        items: [
          { text: '01. Triton 入门与 Hello World：向量加法 (Vector Addition)', link: '/03_CUDA_and_Triton_Kernels/01_Triton_Vector_Addition' },
          { text: '02. Triton 算子开发：融合门控激活函数 (Fused SwiGLU)', link: '/03_CUDA_and_Triton_Kernels/02_Triton_Fused_SwiGLU' },
          { text: '003. Triton 算子开发实战：Fused RMSNorm', link: '/03_CUDA_and_Triton_Kernels/03_Triton_Fused_RMSNorm' },
          { text: '04. GPU 编程的皇冠明珠：矩阵乘法 (GEMM) 与分块搜索空间 (Autotune)', link: '/03_CUDA_and_Triton_Kernels/04_Triton_GEMM_Tutorial' },
          { text: '05. Triton 性能调优与基准测试 (Autotune & Profiling)', link: '/03_CUDA_and_Triton_Kernels/05_Triton_Autotune_and_Profiling' },
          { text: '06. Triton 进阶：跨线程归约与数值稳定 (Safe Softmax)', link: '/03_CUDA_and_Triton_Kernels/06_Triton_Fused_Softmax' },
          { text: '07. Triton 进阶：融合旋转位置编码 (Fused RoPE)', link: '/03_CUDA_and_Triton_Kernels/07_Triton_Fused_RoPE' },
          { text: '08. Triton Flash Attention：编写真正的 Flash Attention 前向算子', link: '/03_CUDA_and_Triton_Kernels/08_Triton_Flash_Attention' },
          { text: '09. Triton 进阶算子：Multi-LoRA 融合推理与 Batch 内指针路由 (Punica 思想)', link: '/03_CUDA_and_Triton_Kernels/09_Triton_Fused_LoRA' },
          { text: '10. Triton 进阶：PagedAttention 的底层实现 (KV Cache 间接寻址)', link: '/03_CUDA_and_Triton_Kernels/10_Triton_KV_Cache_and_PagedAttention' },
          { text: '11. Triton 量化算子：W8A16 权重量化融合矩阵乘法 (Quantization GEMM)', link: '/03_CUDA_and_Triton_Kernels/11_Triton_Quantization_Support' },
          { text: '12. Triton 内存模型、指针计算与 Debug 避坑指南', link: '/03_CUDA_and_Triton_Kernels/12_Triton_Memory_Model_and_Debug' },
          { text: '13. 综合工程实战：使用 Triton 从头组装 LLaMA-3 Transformer Block', link: '/03_CUDA_and_Triton_Kernels/13_Triton_Llama3_Block_Project' },
          { text: '14. Triton Best Practices and FAQ', link: '/03_CUDA_and_Triton_Kernels/14_Triton_Best_Practices_and_FAQ' },
          { text: '15. 突破 PCIe 瓶颈：CPU-GPU 锁页内存与 CUDA 异步流通信', link: '/03_CUDA_and_Triton_Kernels/15_PyTorch_CUDA_Streams_and_Transfer' },
          { text: '16. 分布式进阶：多机通信原语实战 (All-Reduce, All-Gather)', link: '/03_CUDA_and_Triton_Kernels/16_Distributed_Communication_Primitives' },
          { text: '17. 分布式工程落地：解析 DeepSpeed ZeRO 配置文件与 CPU Offload', link: '/03_CUDA_and_Triton_Kernels/17_DeepSpeed_Zero_Config' },
          { text: '18. 硬核降维打击：原生 CUDA C++ 编程与 PyTorch C++ 扩展 (JIT)', link: '/03_CUDA_and_Triton_Kernels/18_CUDA_Custom_Kernel_Intro' },
          { text: '19. 榨干硬件极限：CUDA Shared Memory (共享内存) 优化与 GEMM', link: '/03_CUDA_and_Triton_Kernels/19_CUDA_Shared_Memory_Optimization' },
          { text: '20. 大模型 Infra 架构视野：PyTorch vs Triton vs CUDA C++ 的三层降维', link: '/03_CUDA_and_Triton_Kernels/20_CUDA_vs_Triton_vs_PyTorch' }
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
