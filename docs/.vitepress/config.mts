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
          { text: '01. 数据格式与混合精度', link: '/01_Hardware_Math_and_Systems/01_Data_Types_and_Precision' }
        ]
      },
      {
        text: '第二部分：PyTorch 核心算法',
        items: [
          { text: '01. RMSNorm 算子实现', link: '/02_PyTorch_Algorithms/01_RMSNorm_Tutorial' }
        ]
      },
      {
        text: '第三部分：CUDA 与 Triton 算子',
        items: [
          { text: '01. Triton 向量加法入门', link: '/03_CUDA_and_Triton_Kernels/01_Triton_Vector_Addition' }
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
