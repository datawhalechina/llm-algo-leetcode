# 17 DeepSpeed Zero Config

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/03_CUDA_and_Triton_Kernels/17_DeepSpeed_Zero_Config.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 17. 分布式工程落地：解析 DeepSpeed ZeRO 配置文件与 CPU Offload

**难度：** Medium | **标签：** `Distributed Training`, `DeepSpeed`, `JSON Config` | **目标人群：** 核心 Infra 与算子开发

在实际工程中，我们很少会去手写底层的 `dist.all_gather`，而是将 PyTorch 原生模型丢给 **Microsoft DeepSpeed** 或是 **HuggingFace Accelerate**，通过极简的 JSON 配置文件，一键开启 ZeRO-1 / 2 / 3 加速。
面试中极常考察：“你能解释一下 ZeRO 配置文件中的 `stage`, `overlap_comm`, `cpu_offload` 分别是干什么的吗？”
本节我们将以一份真实的 DeepSpeed 配置文件为题眼，解析其各个核心字段的工程意义。


### Step 1: ZeRO 配置文件核心参数深度解析

> **`stage` (优化阶段):**
> - **Stage 1 (ZeRO-1)**：只切分优化器状态 (Optimizer States)。通常用于显存微小压力时。
> - **Stage 2 (ZeRO-2)**：切分优化器状态 + 梯度 (Gradients)。训练 7B~13B 模型的标配，既极大地省了显存，又没有引入太可怕的通信开销。
> - **Stage 3 (ZeRO-3)**：连模型参数 (Parameters) 也切了！单卡里只有 $\frac{1}{N}$ 的参数，用的时候再 `All-Gather` 临时拼装。用于训练 70B 这种庞然大物。

> **`offload_optimizer` / `offload_param` (CPU Offload):**
> 把原本放在 GPU 显存里的优化器状态或模型参数，扔到 CPU 的主存 (RAM) 里！
> - **优点：** 彻底打破 GPU 显存限制，拿廉价的内存条当显存用。
> - **缺点：** 需要频繁地通过 PCIe 总线在 CPU 和 GPU 之间搬运数据，极大降低训练速度 (通常慢 2-5 倍)。需要配合 `pin_memory=True` 缓解。

> **`overlap_comm` (通信计算重叠):**
> 我们在第 21 节学过，计算和传输应该是并行的！开启此项，DeepSpeed 会在执行矩阵乘法 (Compute) 时，提前异步在后台拉取下一个需要的块的数据 (Communication)。


### Step 2: ZeRO 配置文件与显存切分映射
DeepSpeed 最核心的能力是接管 PyTorch 的底层通信逻辑。在 ZeRO-1 中切分 Optimizer State；ZeRO-2 进一步切分 Gradients；ZeRO-3 甚至把模型 Weights 彻底切碎，只有在经过该层时才动态 Gather 回来。还能通过 `offload_optimizer` 把沉重的 Adam 状态踢到 CPU 内存去计算。


### Step 3: JSON Config 解析框架
这里不需要写底层 C++。你需要深刻理解 `ds_config.json` 中的字段意义，如 `zero_optimization.stage`, `overlap_comm`, `reduce_scatter` 等。在这个实验中，读取配置字典，分析开启某些开关后对显存和带宽造成的双向影响。


###  Step 4: 动手实战

**要求**：请补全下方的 `build_deepspeed_config` 函数，返回一个字典。该字典需要配置为：开启 ZeRO-2、开启优化器 CPU Offload 且启用计算通信重叠。


```python
import json

def build_deepspeed_config():
    """
    构建并返回一个标准的 DeepSpeed ZeRO 配置字典。
    """
    ds_config = {
        # 我们使用 AdamW 优化器，学习率为 3e-4
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 3e-4,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.1
            }
        },
        
        # 设置混合精度训练 (FP16 或 BF16 都可以极大地降低显存占用)
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        
        # ==========================================
        # TODO: 核心配置区
        # ==========================================
        "zero_optimization": {
            # 1. 开启 ZeRO-2 阶段 (切分优化器和梯度)
            # "stage": ???
            "stage": 2,
            
            # 2. 开启通信与计算的并行重叠 (Overlap Communication)
            # "overlap_comm": ???
            "overlap_comm": True,
            
            # (进阶配置，默认通常为 True，为了降低内存峰值而分配连续的梯度缓冲区)
            "contiguous_gradients": True,
            
            # 3. 开启优化器状态的 CPU Offload，并使用锁页内存加速 (pin_memory)
            # "offload_optimizer": { "device": ???, "pin_memory": ??? }
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            }
        },
        
        # 设置单卡 Batch Size 和梯度累加步数
        "train_batch_size": 32,
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 8,
    }
    
    return ds_config

```

```python
# 测试并验证配置
def test_deepspeed_config():
    config = build_deepspeed_config()
    
    # 将字典转换为美观的 JSON 字符串进行打印
    config_json = json.dumps(config, indent=4)
    print("生成的 DeepSpeed 配置如下：\n")
    print(config_json)
    
    # 验证核心参数的正确性
    try:
        z_opt = config.get("zero_optimization", {})
        assert z_opt.get("stage") == 2, "没有正确配置 ZeRO-2 (stage=2)"
        assert z_opt.get("overlap_comm") is True, "没有开启 overlap_comm"
        
        offload = z_opt.get("offload_optimizer", {})
        assert offload.get("device") == "cpu", "没有将设备配置为 cpu"
        assert offload.get("pin_memory") is True, "由于需要高频传输，务必开启 pin_memory"
        
        print("\n✅ DeepSpeed 配置通过验证！")
        print("💡 面试中被问到如何不加卡也能训 13B 模型时，脱口而出 'ZeRO-2 + CPU Offload' 是一个经验成熟的炼丹师的标配回答。")
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")

test_deepspeed_config()

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

::: details 💡 点击查看官方解析与参考代码


### 💡 核心实现原理解析

在 DeepSpeed 中，配置文件是控制所有底层优化魔法的总开关。本题考察的是你在实际算力吃紧的场景下，如何配置出兼顾显存和速度的最优解。

1.  ** (ZeRO-2)**: 这是最经典且性价比极高的切分策略。它在各个 GPU 之间切分**优化器状态 (Optimizer States, 比如 Adam 的一阶和二阶动量，占显存极高)** 和**梯度 (Gradients)**。与不切分的 DDP 相比，显存占用大大降低（尤其是优化器状态）；与极限切分模型参数的 ZeRO-3 相比，它又避免了在前向传播时频繁触发巨量通信 ( weight)，因此训练吞吐量往往更高。
2.  ** (通信/计算重叠)**: 这对应了我们在第 15 节中手动写过的异步 Stream 技巧。DeepSpeed 在做反向传播（算矩阵乘法算出梯度）的同时，会悄悄在后台用另一个 CUDA Stream 把算好的梯度发给别的卡 ()，完美掩盖了传输的延迟。
3.  ** 到 CPU**: 这是对抗 Out-Of-Memory (OOM) 的最后一道杀手锏。即使是 ZeRO-2，切分后的优化器状态依然可能吃光你的 24G 显存。配置了  后，Adam 优化器的状态将被转移到主板的内存 (RAM) 中。参数更新也由 CPU 计算。
    *   **代价是**：每个 step 结束时，必须把 GPU 上的梯度沿着 PCIe 传给 CPU，CPU 算完后，再把更新后的参数沿着 PCIe 传回 GPU。这个过程极慢。
    *   **解法是**：配置 。我们在第 15 节学过，使用锁页内存可以避免 CPU 内存的换页开销，让 DMA (直接内存访问) 满速运行，从而尽量减轻 offload 带来的性能衰减。


```python
def build_deepspeed_config():
    ds_config = {
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 3e-4,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.1
            }
        },
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "zero_optimization": {
            # 1. 开启 ZeRO-2 阶段
            "stage": 2,
            
            # 2. 开启通信与计算的并行重叠
            "overlap_comm": True,
            
            "contiguous_gradients": True,
            
            # 3. 开启优化器状态的 CPU Offload，并开启 pin_memory
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            }
        },
        "train_batch_size": 32,
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 8,
    }
    
    return ds_config
```

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
