# 06. VRAM Calculation and ZeRO Practice | VRAM Calculation and ZeRO - 计算练习

**难度：** Hard | **标签：** `显存计算`, `ZeRO`, `梯度累积` | **目标人群：** 训练 / 分布式学习者

> 🚀 **云端运行环境**
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO_Practice.ipynb)

本练习配套理论文档：[06_VRAM_Calculation_and_ZeRO.md](./06_VRAM_Calculation_and_ZeRO.md)

---

## 🎯 学习目标

- 计算 DDP 训练的显存占用
- 计算 ZeRO-1 / ZeRO-2 / ZeRO-3 的显存节省
- 反推给定显存和卡数下可训练的最大模型规模
- 把显存公式转成可复用的函数

## 核心练习

### Part 1: DDP 显存计算
- `calculate_ddp_memory(num_params_b, model_dtype='fp16', optimizer='adam')`

### Part 2: ZeRO 显存计算
- `calculate_zero_memory(num_params_b, zero_stage, num_gpus, model_dtype='fp16', optimizer='adam')`
- `max_trainable_params(gpu_memory_gb, num_gpus, zero_stage, overhead_ratio=0.2, model_dtype='fp16', optimizer='adam')`

### Part 3: 场景应用
- 对比 DDP 与 ZeRO 的可训练模型规模

## 典型结论

- DDP 下，FP16 + Adam 常用近似是 `16Φ`
- ZeRO-1 切分优化器状态
- ZeRO-2 切分优化器状态和梯度
- ZeRO-3 切分参数、梯度和优化器状态
