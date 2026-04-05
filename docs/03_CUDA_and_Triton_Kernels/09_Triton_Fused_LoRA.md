# 09 Triton Fused LoRA

> 🚀 **云端运行环境**
> 
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/03_CUDA_and_Triton_Kernels/09_Triton_Fused_LoRA.ipynb)  
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

# 09. Triton 进阶算子：Multi-LoRA 融合推理与 Batch 内指针路由 (Punica 思想)

**难度：** Hard | **标签：** `Triton`, `LoRA`, `Punica`, `Serving` | **目标人群：** 核心 Infra 与算子开发

在 `02_PyTorch_Algorithms/09_LoRA_Tutorial.ipynb` 中，我们在 PyTorch 层面实现了 LoRA ($h = xW + xAB$) 的逻辑。
然而，在工业级的大模型推理服务 (Serving) 中，面对并发的多个用户请求，如果每个用户的 prompt 挂载了**不同的** LoRA 权重（如用户 A 请求写代码的 LoRA，用户 B 请求翻译的 LoRA），如果将他们拆分并循环执行 PyTorch 的 `linear()`，会严重降低 GPU 的吞吐量（无法利用 Batch 计算）。
本节我们将手撕 **Multi-LoRA (如 S-LoRA / Punica 论文核心思路)** 的底层 Triton 融合算子：**通过传入 `lora_indices`，让每一个 Token 在 SRAM 内自动去内存池中拉取它专属的 LoRA 权重，完成批量计算！**


> **相关阅读**:
> 本节使用 Triton 实现了高阶的 Token 级动态路由与融合推理。
> 如果你对该算子的基础 PyTorch 层面的权重分解与训练不熟悉，建议先复习 PyTorch 篇：
>  [`../02_PyTorch_Algorithms/09_LoRA_Tutorial.ipynb`](../02_PyTorch_Algorithms/09_LoRA_Tutorial.ipynb)

### Step 1: Multi-LoRA 内存池与 Batch 路由

> **LoRA 的内存池 (Weight Pool)：**
> 我们不再为每一个单独的 LoRA 权重分配离散的张量。而是将机器上加载的所有 LoRA 矩阵 A (形状 `r \t\times in_features`) 拼接成一个巨大的连续张量 `lora_a_pool`，形状为 `(num_loras, r, in_features)`。

> **Token 级别的细粒度路由 (Token-wise Routing)：**
> 假设输入特征 $X$ 的形状为 `(batch_size, in_features)`。
> 我们额外传入一个长度为 `batch_size` 的一维整型数组 `lora_indices`。它记录了：当前 Batch 中，第 $i$ 个 Token 到底需要使用 `lora_a_pool` 里的第几个 LoRA 模型。

> **SRAM 内部的极致并行：**
> - Triton `pid` 处理矩阵 $X$ 的某一行（某个 Token）。
> - 我们直接在内核里根据 `lora_idx = tl.load(lora_indices + pid)`，算出指向内存池中特定 LoRA A 和 B 的偏移量！
> - 一次性读取 $X_i$ 和专属的 $A_{idx}, B_{idx}$，在极速的 SRAM 中完成 $X_i \times A \times B$，最后写回 $H_i$。
> 这样，**原本必须串行计算的不同模型请求，被完美地合并在了一个 Triton Kernel 调用中（Batch Inference）！**


### Step 2: 内存池与 Batch 指针路由
在推理服务器中，往往存在一个大底座模型对应几百个微调的 LoRA 权重。为了避免切换开销，我们将所有的 LoRA 权重放进统一的巨大的显存池中。每个发来的 Token 请求都会带有一个 `lora_id`。内核利用这个 `lora_id` 充当偏移指针，直接在同一次前向计算中去抓取不同的权重完成点积。


### Step 3: 指针路由代码框架
传入包含所有权重的张量 `lora_pool` 和整数数组 `lora_indices`。在内核中，先读取 `lora_idx = tl.load(lora_indices_ptr + pid_batch)`，将该索引乘上权重的 stride，动态确定当前线程块该加载哪一份 LoRA A 和 B，随后做标准的低秩乘加运算。


###  Step 4: 动手实战

**要求**：请补全下方 `fused_multi_lora_kernel`。我们需要根据传入的 `lora_indices`，正确地在三维张量 `lora_a_pool` 和 `lora_b_pool` 中计算偏移量，并执行点积。为了简化，我们假设秩 `r` 很小（如 8 或 16），且输入只进行列并行分块（行完全塞入 SRAM）。


```python
import torch
import triton
import triton.language as tl

@triton.jit
def fused_multi_lora_kernel(
    x_ptr, out_ptr, 
    lora_a_pool_ptr, lora_b_pool_ptr, 
    lora_indices_ptr,
    M, IN_DIM, OUT_DIM, R: tl.constexpr,
    stride_x_m, stride_x_in,
    stride_out_m, stride_out_dim,
    stride_a_pool, stride_a_r, stride_a_in,
    stride_b_pool, stride_b_out, stride_b_r,
    BLOCK_IN: tl.constexpr, BLOCK_OUT: tl.constexpr
):
    # 1. 确定当前处理的是哪个 Token (行 M) 和哪个输出维度块 (列 OUT_DIM)
    pid_m = tl.program_id(0) # 对应 M 维度 (batch_size)
    pid_n = tl.program_id(1) # 对应 OUT_DIM 维度的分块
    
    # 获取输出列的偏移
    offs_n = pid_n * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    
    # 2. 读取当前 Token 专属的 LoRA Index (极其关键的路由逻辑)
    lora_idx = tl.load(lora_indices_ptr + pid_m)
    
    # 3. 计算内存池中该 LoRA A 和 B 的基地址偏移
    # a_pool_base_ptr = ???
    # b_pool_base_ptr = ???
    a_pool_base_ptr = lora_a_pool_ptr + lora_idx * stride_a_pool
    b_pool_base_ptr = lora_b_pool_ptr + lora_idx * stride_b_pool
    
    # 4. 初始化累加器 (当前 Token 加上该分块的所有输出)
    acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)
    
    # 5. 为了完成 x @ A @ B，我们这里采用极致简化版本 (针对 r 极小的情况)：
    # 先在 SRAM 中累加 x @ A，得到一个长度为 R 的中间激活 h_r
    h_r = tl.zeros((R,), dtype=tl.float32)
    
    # 5.1 循环遍历 IN_DIM 计算 x @ A (形状: (1, IN_DIM) @ (IN_DIM, R) = (1, R))
    num_k_blocks = tl.cdiv(IN_DIM, BLOCK_IN)
    for k in range(num_k_blocks):
        offs_in = k * BLOCK_IN + tl.arange(0, BLOCK_IN)
        
        # 加载 x
        x_ptrs = x_ptr + pid_m * stride_x_m + offs_in * stride_x_in
        x_val = tl.load(x_ptrs, mask=offs_in < IN_DIM, other=0.0)
        
        # 加载 A 的这一小块 (注意 A 的形状是 R x IN_DIM)
        # 偏移：r 的维度我们用一个连续的 arange 覆盖整个 R (因为 R 极小)，列用 offs_in
        offs_r = tl.arange(0, R)
        a_ptrs = a_pool_base_ptr + offs_r[:, None] * stride_a_r + offs_in[None, :] * stride_a_in
        a_val = tl.load(a_ptrs, mask=offs_in[None, :] < IN_DIM, other=0.0)
        
        # 累加 x @ A.T (也就是 x 和 A 的每一行点积)
        # tl.sum 对列求和，得到长为 R 的向量
        h_r += tl.sum(x_val[None, :] * a_val, axis=1)
        
    # 5.2 已经拿到了中间激活 h_r (长为 R)。现在计算 h_r @ B
    # 循环遍历 R (通常不需要循环因为 R 小于 BLOCK 限制，但我们按照规范展开)
    # B 的形状是 (OUT_DIM, R)，我们需要提取对应的 OUT_DIM 列分块
    offs_r = tl.arange(0, R)
    b_ptrs = b_pool_base_ptr + offs_n[:, None] * stride_b_out + offs_r[None, :] * stride_b_r
    b_val = tl.load(b_ptrs, mask=offs_n[:, None] < OUT_DIM, other=0.0)
    
    # 累加 h_r @ B.T
    acc += tl.sum(h_r[None, :] * b_val, axis=1)
    
    # 6. 写回显存 (仅包含 LoRA 的旁路增量，真实使用时需要加回原输出矩阵)
    out_ptrs = out_ptr + pid_m * stride_out_m + offs_n * stride_out_dim
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=offs_n < OUT_DIM)

def triton_multi_lora_forward(x: torch.Tensor, lora_a_pool: torch.Tensor, lora_b_pool: torch.Tensor, lora_indices: torch.Tensor):
    M, IN_DIM = x.shape
    num_loras, R, _ = lora_a_pool.shape
    _, OUT_DIM, _ = lora_b_pool.shape
    
    out = torch.empty((M, OUT_DIM), device=x.device, dtype=x.dtype)
    
    BLOCK_IN = 64
    BLOCK_OUT = 64
    
    grid = (M, triton.cdiv(OUT_DIM, BLOCK_OUT))
    
    fused_multi_lora_kernel[grid](
        x, out, 
        lora_a_pool, lora_b_pool, 
        lora_indices,
        M, IN_DIM, OUT_DIM, R,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        lora_a_pool.stride(0), lora_a_pool.stride(1), lora_a_pool.stride(2),
        lora_b_pool.stride(0), lora_b_pool.stride(1), lora_b_pool.stride(2),
        BLOCK_IN=BLOCK_IN, BLOCK_OUT=BLOCK_OUT
    )
    return out

```

```python
# 测试并验证 Multi-LoRA 路由的正确性
def test_multi_lora():
    if not torch.cuda.is_available():
        print("⏭️ 忽略测试：无 GPU。")
        return
        
    try:
        torch.manual_seed(42)
        batch_size = 4
        in_dim = 128
        out_dim = 256
        num_loras = 3 # 内存池中有 3 个不同的 LoRA
        r = 16
        
        x = torch.randn(batch_size, in_dim, device='cuda')
        
        # 构造内存池
        # A_pool: (3, 16, 128)
        # B_pool: (3, 256, 16)
        lora_a_pool = torch.randn(num_loras, r, in_dim, device='cuda')
        lora_b_pool = torch.randn(num_loras, out_dim, r, device='cuda')
        
        # 构造复杂的请求路由 (Token 0用LoRA_2, Token 1用LoRA_0...)
        lora_indices = torch.tensor([2, 0, 1, 2], device='cuda', dtype=torch.int32)
        
        # 1. 纯 PyTorch 参考计算 (必须使用极其缓慢的 for 循环拼接)
        out_ref = torch.zeros(batch_size, out_dim, device='cuda')
        for i in range(batch_size):
            idx = lora_indices[i].item()
            # 提取专属权重
            A = lora_a_pool[idx] # (r, in_dim)
            B = lora_b_pool[idx] # (out_dim, r)
            
            # x[i]: (1, in_dim)
            # x_i @ A.T @ B.T
            h_r = x[i].unsqueeze(0) @ A.T # (1, r)
            y_i = h_r @ B.T               # (1, out_dim)
            out_ref[i] = y_i.squeeze(0)
            
        # 2. Triton 单算子极致并行计算
        out_tri = triton_multi_lora_forward(x, lora_a_pool, lora_b_pool, lora_indices)
        
        # 3. 验证结果
        diff = torch.max(torch.abs(out_ref - out_tri))
        print(f"最大误差: {diff.item():.6e}")
        assert diff < 1e-3, "Triton Multi-LoRA 路由或计算结果不正确！"
        
        print("✅ 完美！你成功实现了类似 S-LoRA / Punica 框架底层的 Token 级别路由融合推理！")
        print("💡 能够通过一个 kernel 处理并发的多模型请求，这是目前工业界降低大模型服务成本 (Serving Cost) 最核心的架构优化。")
        
    except NotImplementedError:
        print("请先完成 TODO 代码！")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

test_multi_lora()

```

::: details 💡 点击查看官方解析与参考代码

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

### 💡 参考解答：Triton 融合 Multi-LoRA 计算

在这个实现中，最核心的逻辑是：
```python
a_pool_base_ptr = lora_a_pool_ptr + lora_idx * stride_a_pool
b_pool_base_ptr = lora_b_pool_ptr + lora_idx * stride_b_pool
```
我们将原本的 batch inference 分解到 Triton 的每一个 `program_id(0)`（代表 batch 中的一个 token），通过 `tl.load` 获取对应 token 需要使用的 LoRA 权重索引 `lora_idx`。
然后，直接利用步长 `stride` 将内存指针跳转到 `lora_a_pool` 和 `lora_b_pool` 中对应的起始位置。这样每个计算单元就能无缝地拉取到自己需要的特有 LoRA 权重，而不需要在 PyTorch 中使用低效的 for 循环去逐个推理，实现了极高的并行度。

```python
import torch
import triton
import triton.language as tl

@triton.jit
def fused_multi_lora_kernel(
    x_ptr, out_ptr, 
    lora_a_pool_ptr, lora_b_pool_ptr, 
    lora_indices_ptr,
    M, IN_DIM, OUT_DIM, R: tl.constexpr,
    stride_x_m, stride_x_in,
    stride_out_m, stride_out_dim,
    stride_a_pool, stride_a_r, stride_a_in,
    stride_b_pool, stride_b_out, stride_b_r,
    BLOCK_IN: tl.constexpr, BLOCK_OUT: tl.constexpr
):
    # 1. 确定当前处理的是哪个 Token (行 M) 和哪个输出维度块 (列 OUT_DIM)
    pid_m = tl.program_id(0) # 对应 M 维度 (batch_size)
    pid_n = tl.program_id(1) # 对应 OUT_DIM 维度的分块
    
    # 获取输出列的偏移
    offs_n = pid_n * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    
    # 2. 读取当前 Token 专属的 LoRA Index (极其关键的路由逻辑)
    lora_idx = tl.load(lora_indices_ptr + pid_m)
    
    # 3. 计算内存池中该 LoRA A 和 B 的基地址偏移
    a_pool_base_ptr = lora_a_pool_ptr + lora_idx * stride_a_pool
    b_pool_base_ptr = lora_b_pool_ptr + lora_idx * stride_b_pool
    
    # 4. 初始化累加器 (当前 Token 加上该分块的所有输出)
    acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)
    
    # 5. 为了完成 x @ A @ B，我们这里采用极致简化版本 (针对 r 极小的情况)：
    # 先在 SRAM 中累加 x @ A，得到一个长度为 R 的中间激活 h_r
    h_r = tl.zeros((R,), dtype=tl.float32)
    
    # 5.1 循环遍历 IN_DIM 计算 x @ A (形状: (1, IN_DIM) @ (IN_DIM, R) = (1, R))
    num_k_blocks = tl.cdiv(IN_DIM, BLOCK_IN)
    for k in range(num_k_blocks):
        offs_in = k * BLOCK_IN + tl.arange(0, BLOCK_IN)
        
        # 加载 x
        x_ptrs = x_ptr + pid_m * stride_x_m + offs_in * stride_x_in
        x_val = tl.load(x_ptrs, mask=offs_in < IN_DIM, other=0.0)
        
        # 加载 A 的这一小块 (注意 A 的形状是 R x IN_DIM)
        # 偏移：r 的维度我们用一个连续的 arange 覆盖整个 R (因为 R 极小)，列用 offs_in
        offs_r = tl.arange(0, R)
        a_ptrs = a_pool_base_ptr + offs_r[:, None] * stride_a_r + offs_in[None, :] * stride_a_in
        a_val = tl.load(a_ptrs, mask=offs_in[None, :] < IN_DIM, other=0.0)
        
        # 累加 x @ A.T (也就是 x 和 A 的每一行点积)
        # tl.sum 对列求和，得到长为 R 的向量
        h_r += tl.sum(x_val[None, :] * a_val, axis=1)
        
    # 5.2 已经拿到了中间激活 h_r (长为 R)。现在计算 h_r @ B
    # 循环遍历 R (通常不需要循环因为 R 小于 BLOCK 限制，但我们按照规范展开)
    # B 的形状是 (OUT_DIM, R)，我们需要提取对应的 OUT_DIM 列分块
    offs_r = tl.arange(0, R)
    b_ptrs = b_pool_base_ptr + offs_n[:, None] * stride_b_out + offs_r[None, :] * stride_b_r
    b_val = tl.load(b_ptrs, mask=offs_n[:, None] < OUT_DIM, other=0.0)
    
    # 累加 h_r @ B.T
    acc += tl.sum(h_r[None, :] * b_val, axis=1)
    
    # 6. 写回显存 (仅包含 LoRA 的旁路增量，真实使用时需要加回原输出矩阵)
    out_ptrs = out_ptr + pid_m * stride_out_m + offs_n * stride_out_dim
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=offs_n < OUT_DIM)

def triton_multi_lora_forward(x: torch.Tensor, lora_a_pool: torch.Tensor, lora_b_pool: torch.Tensor, lora_indices: torch.Tensor):
    M, IN_DIM = x.shape
    num_loras, R, _ = lora_a_pool.shape
    _, OUT_DIM, _ = lora_b_pool.shape
    
    out = torch.empty((M, OUT_DIM), device=x.device, dtype=x.dtype)
    
    BLOCK_IN = 64
    BLOCK_OUT = 64
    
    grid = (M, triton.cdiv(OUT_DIM, BLOCK_OUT))
    
    fused_multi_lora_kernel[grid](
        x, out, 
        lora_a_pool, lora_b_pool, 
        lora_indices,
        M, IN_DIM, OUT_DIM, R,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        lora_a_pool.stride(0), lora_a_pool.stride(1), lora_a_pool.stride(2),
        lora_b_pool.stride(0), lora_b_pool.stride(1), lora_b_pool.stride(2),
        BLOCK_IN=BLOCK_IN, BLOCK_OUT=BLOCK_OUT
    )
    return out
```

:::

---

> 💡 **有更好的解法或性能优化？**
> 欢迎在下方评论区交流你的思路，或者直接点击页面底部的「在 GitHub 上编辑此页」提交 PR，将你的优质代码合并到官方题解中！
