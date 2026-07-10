# Top 10 大模型算法系统核心基础知识

## 1. Transformer Decoder骨架（Pre-Norm + RMSNorm + SwiGLU FFN）

不只是读论文，要看现代大模型（Llama/Qwen）的实际代码实现。重点理解：

· Pre-Norm结构（先归一化再Attention/FFN）对训练稳定性的影响，以及推理时归一化层的计算量占比。
· SwiGLU相比ReLU多了一个线性层，直接拉高了FFN部分的参数量和FLOPs。
  系统视角：FFN通常占模型总参数量的2/3，它的计算密度（算术强度）决定了推理时是算力瓶颈还是带宽瓶颈。

## 2. 旋转位置编码（RoPE）及其外推机制

RoPE是当前所有主流大模型的位置编码方案。重点看：

· 它如何通过旋转矩阵将位置信息"注入"Q和K，以及高频/低频维度对长文本外推的影响。
· YaRN等外推方法的原理。
  系统视角：长上下文（128K+）下，RoPE的向量计算本身是轻量的，但它通过影响注意力分布，直接决定了KV Cache的大小和有效利用率。

## 3. 多头注意力（MHA）与分组查询注意力（GQA）

GQA是Llama 3/Qwen 2.5选择的核心优化。重点对比：

· MHA（每头独立KV）→ MQA（共享KV）→ GQA（分组共享KV）的演进。
· 手算GQA如何将KV Cache从 num_heads * head_dim 缩减到 num_kv_heads * head_dim。
  系统视角：Decode阶段是访存密集型的，GQA直接减少了从HBM读取KV Cache的数据量，这是vLLM等框架做Prefill/Decode分离优化的理论基础。

## 4. FlashAttention的IO感知（Tiling + 重计算）

不只是"快"，要理解它为什么快：

· 分块（Tiling）把Q/K/V切成小块，利用SRAM做高速缓存。
· 在线Softmax + 重计算，避免存储S/P中间矩阵，减少HBM访问。
  系统视角：FlashAttention是算术强度（Arithmetic Intensity） 的教科书案例——它不减少FLOPs，而是通过减少HBM读写次数来提速，这是算法与硬件协同设计的典范。

## 5. PagedAttention与KV Cache管理

这是vLLM的"灵魂"。重点：

· 将KV Cache分页存储，用block_table做逻辑块到物理块的映射（借鉴OS的分页管理）。
· 理解显存碎片和共享前缀（如P-MoE） 带来的优化空间。
  系统视角：PagedAttention是推理引擎调度器（Scheduler） 的核心数据结构，它决定了Continuous Batching能否高效运行。

## 6. Continuous Batching（连续批处理）

对比传统的Static Batching（等所有请求完成才释放）：

· Continuous Batching允许逐Token动态增删请求，让GPU永不空闲。
· 调度器在每步（Step）决策：把哪些Waiting请求加入Running队列，把哪些Running请求抢占（Preempt）换出。
  系统视角：这是推理框架从"批处理"走向"流式处理"的关键转变，直接决定了高并发下的吞吐量。

## 7. 训练后量化（GPTQ vs AWQ）

这是你之前深入问过的方向：

· GPTQ：基于海森矩阵（输入协方差）逐层量化，在4-bit下精度稳定。
· AWQ：基于激活值分布，识别并保护1%的关键权重。
  系统视角：量化是"计算量不变，但访存量减半/减75%"的技术。它的收益完全体现在HBM带宽瓶颈的缓解上——当你用vLLM部署AWQ模型时，吞吐量的提升比精度损失更直观。

## 8. MoE（混合专家）的路由与负载均衡

MoE是DeepSeek等模型的核心架构。重点：

· 路由网络（Router）如何将Token分配给专家（Expert）。
· 负载均衡损失（Load Balance Loss）——防止Token全涌向同一个专家导致GPU闲置。
  系统视角：MoE在推理时只激活部分专家，总参数量大但活跃参数量小。但其专家并行（Expert Parallelism）引入的All-to-All通信开销巨大，需要NVLink等高速互联支撑。

## 9. 分布式通信原语（All-Reduce / All-Gather）与并行策略

不深究NCCL源码，但要理解：

· All-Reduce：数据并行（DP）中同步梯度的核心操作。
· All-Gather：张量并行（TP）中聚合注意力头输出的核心操作。
· 通信/计算比：判断哪种并行策略更适合当前硬件（TP通信量大，适合NVLink；DP通信量小，适合跨机）。
  系统视角：大模型从单卡走向多卡，通信调度就成了新的"瓶颈调度"问题。

## 10. DPO（直接偏好优化）——对齐的"轻量级"方案

对比RLHF的奖励模型+PPO训练，DPO用闭式解一步到位优化偏好对数据。重点：

· DPO的损失函数如何将"偏好排序"转化为可微分的训练目标。
· 它不需要单独的奖励模型，显存占用比RLHF低得多。
  系统视角：DPO虽然轻量，但它引入了隐式奖励，训练时对梯度稳定性要求极高，实践中Loss飞涨是常见问题——这是算法稳定性与工程调参的博弈。

## 经典应用（带着系统视角看）

RAG（检索增强生成）：作为P0级应用场景，它的系统痛点不是检索算法，而是长上下文的KV Cache显存爆炸。学RAG时，重点看上下文压缩和Rerank策略如何减少送入模型的数据量。

Agent（工具调用）：作为P1级场景，Agent反复调用工具意味着大量的短序列生成请求，这对推理调度器的低延迟调度（Low-latency Scheduling） 是个极大考验——Continuous Batching在这里的价值就体现在"快速响应新请求"上。

一句话串起来

这10个点，学透GQA、FlashAttention、PagedAttention，你就能看懂推理引擎（vLLM）的优化基本盘；学透量化（GPTQ/AWQ）和MoE，你就能看懂部署降本的核心思路；再补上分布式通信和DPO，你就有了理解"训练-对齐-部署"全链路的雏形。

RAG和Agent不用当作独立领域去学，而是作为"需求源"——反问自己：这个应用场景下，上述10个知识点中哪些在解决它的性能瓶颈？这个思维方式，就是你想要的"重算法系统，轻应用"的本质。