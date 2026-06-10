# 02. LLM Params and FLOPs Practice | LLM Params and FLOPs - 计算练习

**难度：** Medium | **标签：** `参数量`, `FLOPs`, `训练时间` | **目标人群：** 算法 / Infra 学习者

> 🚀 **云端运行环境**
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/02_LLM_Params_and_FLOPs_Practice.ipynb)

本练习配套理论文档：[02_LLM_Params_and_FLOPs.md](./02_LLM_Params_and_FLOPs.md)

---

## 🎯 学习目标

- 拆解 Transformer 的参数组成
- 计算训练与推理 FLOPs
- 估算给定硬件配置下的训练时间
- 把理论公式转成可复用的计算函数

## 核心练习

### Part 1: Transformer 参数量计算
- `calculate_transformer_params(vocab_size, hidden_dim, num_layers, intermediate_size=None, tie_embeddings=False)`

### Part 2: 训练与推理 FLOPs 计算
- `calculate_training_flops(num_params_b, num_tokens, flops_per_param_token=6)`
- `estimate_training_time(num_params_b, num_tokens, gpu_tflops, num_gpus, efficiency=0.35)`

### Part 3: 场景应用
- 对比不同 GPU 组合下的训练时间

## 典型结论

- 训练 FLOPs 通常可近似为 `6 × 参数量 × token 数`
- 训练时间估算需要额外乘以实际利用率
