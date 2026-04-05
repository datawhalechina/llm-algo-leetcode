# 16 Distributed Communication Primitives

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/03_CUDA_and_Triton_Kernels/16_Distributed_Communication_Primitives.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 16. 分布式进阶：多机通信原语实战 (All-Reduce, All-Gather)

**难度：** Hard | **标签：** `Distributed Training`, `NCCL`, `Communication Primitives` | **目标人群：** 核心 Infra 与算子开发
在前几节的 ZeRO-1 和 TP (张量并行) 中，我们只是通过**数组切片**逻辑上模拟了分布式计算。
但在真实的工业界集群中（如 8 张 A100 甚至千卡集群），GPU 之间必须通过 NCCL (Nvidia Collective Communication Library) 进行真实的物理数据交换。
本节我们将深入 `torch.distributed`，实战最核心的两大通信原语：`All-Reduce` 和 `All-Gather`。这也是面试极其高频的考点（如何计算通信量？Ring-AllReduce 怎么跑的？）。


### Step 1: 集合通信原语的本质

> **All-Reduce (全归约)：**
> 假设每张 GPU 上都有一个相同形状的梯度张量。你想把所有 GPU 的梯度加起来，然后再把总和发还给每张 GPU（在 DDP 数据并行中更新权重必备）。
> - **底层逻辑：** 通常通过 Ring-AllReduce 算法，将数据分为 N 份（N为GPU数），在环形拓扑上传输。
> - **通信量：** 大约是 $2 \times \frac{N-1}{N} \times 	ext{Size}$，它不受 GPU 数量激增的影响，极其高效。

> **All-Gather (全收集)：**
> 假设每张 GPU 算出了模型的一部分输出（如 TP 列切分），你需要把所有 GPU 的这些片段拼装成一个完整的大张量，分发给所有人。
> - **底层逻辑：** 每张卡把自己的那块数据广播给其他所有人。
> - **ZeRO-3 中的应用：** 每张卡只有自己负责的 $\frac{1}{N}$ 权重，在前向传播时，必须通过 `All-Gather` 临时把完整权重拼出来才能算矩阵乘法。


### Step 2: torch.distributed 代码框架
利用 `torch.distributed.init_process_group(backend='nccl')` 初始化通信后端。获取 `dist.get_rank()` (当前 GPU 编号) 和 `dist.get_world_size()` (总 GPU 数) 后，执行 `dist.all_reduce(tensor)` 或 `dist.all_gather(tensor_list, local_tensor)` 进行原语调用。


###  Step 3: 动手实战

**要求**：请补全下方 `simulate_distributed_primitives`，使用 PyTorch 的多进程包 `torch.multiprocessing` 模拟 2 张卡的真实通信环境，并在其中实现 `all_reduce` 和 `all_gather` 的调用。


```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run_worker(rank, world_size):
    """
    在子进程中执行的代码。代表单张 GPU 的视角。
    """
    # 1. 初始化进程组 (Backend 推荐 nccl，但如果本地无多卡或只是 CPU 测试，则使用 gloo)
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    # 配置临时环境变量，让进程能互相找到
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    # 设置设备
    device = torch.device(f'cuda:{rank}') if torch.cuda.is_available() else torch.device('cpu')
    
    try:
        # ==========================================
        # TODO 1: 模拟 All-Reduce (求和)
        # 假设 rank 0 上是 [1.0, 2.0]，rank 1 上是 [3.0, 4.0]
        # 要求所有卡最后都得到 [4.0, 6.0]
        # ==========================================
        tensor_to_reduce = torch.tensor([float(rank * 2 + 1), float(rank * 2 + 2)], device=device)
        
        # 调用 dist.all_reduce() 进行原位 (In-place) 操作，op 默认为 SUM
        # dist.???(tensor_to_reduce, op=dist.ReduceOp.SUM)
        dist.all_reduce(tensor_to_reduce, op=dist.ReduceOp.SUM)
        
        # ==========================================
        # TODO 2: 模拟 All-Gather (收集拼装)
        # 假设每张卡产生一个独立的标量特征 [rank * 10]
        # 要求所有卡最后都得到 [0.0, 10.0]
        # ==========================================
        local_tensor = torch.tensor([float(rank * 10)], device=device)
        
        # 准备一个空列表，用于接收所有卡发来的张量
        gathered_list = [torch.zeros_like(local_tensor) for _ in range(world_size)]
        
        # 调用 dist.all_gather()
        # dist.???(gathered_list, local_tensor)
        dist.all_gather(gathered_list, local_tensor)
        
        # 验证结果 (只在 rank 0 打印以防刷屏)
        if rank == 0:
            print(f"✅ Rank 0 All-Reduce 后得到: {tensor_to_reduce.tolist()} (期望: [4.0, 6.0])")
            print(f"✅ Rank 0 All-Gather 后得到: {[t.item() for t in gathered_list]} (期望: [0.0, 10.0])")
            assert tensor_to_reduce.tolist() == [4.0, 6.0], "All-Reduce 结果不正确！"
            assert [t.item() for t in gathered_list] == [0.0, 10.0], "All-Gather 结果不正确！"
            
    finally:
        # 清理销毁进程组
        dist.destroy_process_group()

def simulate_distributed_primitives(num_gpus=2):
    # 如果可用 GPU 数不够，回退到 CPU (gloo) 测试
    if torch.cuda.device_count() < num_gpus:
        print(f"⚠️ 当前机器可用 GPU 数量少于 {num_gpus}，将使用 CPU (gloo 后端) 模拟多进程通信。")
        
    # 使用 mp.spawn 启动多个进程
    # 注意: 这个函数会阻塞，直到所有子进程运行完毕
    mp.spawn(run_worker,
             args=(num_gpus,),
             nprocs=num_gpus,
             join=True)

```

```python
# 运行分布式模拟测试
def test_distributed():
    try:
        print("🚀 启动多进程分布式通信模拟 (模拟 2 个节点/显卡)...")
        # 运行模拟
        simulate_distributed_primitives(num_gpus=2)
        print("\n🔥 太棒了！你跨过了分布式计算门槛，成功调用了工业界底层最核心的 NCCL / gloo 通信原语。")
        
    except Exception as e:
        print(f"❌ 运行失败: {e}")

test_distributed()

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

在多卡/多机分布式训练中，不同进程（每个 GPU 对应一个进程）拥有各自独立的内存空间。我们需要通过底层库（如 NCCL 或 gloo）来实现跨进程/跨设备的数据同步。

1.  **进程组初始化 ()**: 这是构建分布式拓扑的基石。必须指定当前进程在集群中的身份  (0, 1, 2...) 和整个集群的总进程数 。还需要指定通信后端（NVIDIA GPU 必须选  才能获得最高性能，CPU 通信回退到 ）。
2.  **All-Reduce ()**:
    *   **原理**: 它将所有  上指定的张量进行某种归约操作（默认是 ，还可以是 , ,  等），最后每个进程上的张量都会被**就地 (In-place)** 修改为归约后的结果。
    *   **应用**: 分布式数据并行 (DDP) 在反向传播后，需要将所有设备上算出的不同梯度求平均，这就是 All-Reduce 的典型用法。
3.  **All-Gather ()**:
    *   **原理**: 每个进程贡献一个本地张量 ，收集所有人贡献的张量，最后将结果填充到一个列表  中。每个进程都得到完整的收集结果。
    *   **应用**: 张量并行 (TP) 中，如果某一层计算将特征按列切分，那么输出特征在各个 GPU 上只是切片。进入下一层需要完整特征时，就需要调用 All-Gather 将这些特征片段重新拼接。


```python
def run_worker(rank, world_size):
    # 根据可用 GPU 数量决定 backend。单卡机器无法模拟 2 卡 NCCL，必须降级到 gloo 和 CPU
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    backend = "nccl" if use_cuda else "gloo"
    
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    device = torch.device(f"cuda:{rank}") if use_cuda else torch.device("cpu")
    
    try:
        # TODO 1: 模拟 All-Reduce (求和)
        tensor_to_reduce = torch.tensor([float(rank * 2 + 1), float(rank * 2 + 2)], device=device)
        dist.all_reduce(tensor_to_reduce, op=dist.ReduceOp.SUM)
        
        # TODO 2: 模拟 All-Gather (收集拼装)
        local_tensor = torch.tensor([float(rank * 10)], device=device)
        gathered_list = [torch.zeros_like(local_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_list, local_tensor)
        
        if rank == 0:
            print(f"✅ Rank 0 All-Reduce 后得到: {tensor_to_reduce.tolist()} (期望: [4.0, 6.0])")
            print(f"✅ Rank 0 All-Gather 后得到: {[t.item() for t in gathered_list]} (期望: [0.0, 10.0])")
            assert tensor_to_reduce.tolist() == [4.0, 6.0], "All-Reduce 结果不正确！"
            assert [t.item() for t in gathered_list] == [0.0, 10.0], "All-Gather 结果不正确！"
            
    finally:
        dist.destroy_process_group()
```

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
