# 用 nano-vLLM 与 Mini-SGLang 跟踪推理请求

模型的一次前向不等于完整推理服务：引擎还要接收请求、保存生成状态、选择下一轮工作，并把结果交还调用方。先沿 nano-vLLM 看清这一过程，再用 Mini-SGLang 对照缓存复用、阶段调度与低精度执行，可以把分散的机制放回同一个请求中理解。

## 从哪里开始

[nano-vLLM 请求跟踪](./nano_vllm_request_walkthrough.md)从离线生成入口开始，逐步观察请求、模型调用和缓存块。[Mini-SGLang 缓存与执行对照](./mini_sglang_cache_scheduling.md)从共享前缀开始，再比较分块预填充、执行重叠和条件满足时的量化路径。两者是随学习阶段逐步深入的源码实践，不必在入门时通读整套引擎。

| 阶段 | 主要观察问题 | 阅读入口 |
| --- | --- | --- |
| <a id="task0"></a>Task0 请求入门 | 输入怎样进入引擎、逐步生成并结束？离线调用与在线服务有什么区别？ | [nano-vLLM 请求入口](./nano_vllm_request_walkthrough.md#request-entry) |
| <a id="task3"></a>Task3 缓存管理与复用 | 状态存在哪里，哪些前缀可以复用，结束后哪些状态仍需保留？ | [nano-vLLM 块管理](./nano_vllm_request_walkthrough.md#cache-blocks) → [Mini-SGLang 前缀索引](./mini_sglang_cache_scheduling.md#prefix-cache) |
| <a id="task4"></a>Task4 轮次调度 | 下一轮选谁、处理多少输入，哪些准备可以与计算重叠？ | [nano-vLLM 请求选择](./nano_vllm_request_walkthrough.md#iteration-scheduling) → [Mini-SGLang 阶段调度](./mini_sglang_cache_scheduling.md#phase-scheduling) |
| <a id="task5"></a>Task5 低精度对照 | 哪类数据变小了，转换与计算实际走哪条路径？ | [Mini-SGLang 低精度执行对照](./mini_sglang_cache_scheduling.md#quantized-execution)，先确认所选版本和分支的能力 |

## 怎样继续深入

<a id="task1"></a>
<a id="task2"></a>

模型前向与生成循环已有专门的机制讲解，源码阅读时分别对照 [nano-vLLM 模型输入](./nano_vllm_request_walkthrough.md#model-forward)和[序列更新与停止](./nano_vllm_request_walkthrough.md#generation-loop)。先回答当前阶段的问题，再沿同一个请求进入下一部分。

<a id="task6"></a>

教学引擎中的观察可以提出性能假设，例如共享前缀减少了多少重复输入处理。采用方案时，返回[端到端基准与决策](./07_benchmark_and_decision.md)，在固定请求、质量和资源条件下复测；两套引擎的默认吞吐差异不能单独解释某个机制的收益。

安装入口见 [Quickstart](./quickstart.md)，固定版本与运行准备状态见[教学引擎资产](./experiment_assets.md#teaching-engine)。
