# nano-vLLM 请求跟踪实践

输入两条提示词之后，引擎为什么不能只调用一次模型就返回？自回归生成每次只得到下一步结果，还要把生成进度与历史状态保留下来。沿着一条请求追踪，先看输入怎样进入、结果怎样返回，再逐步观察模型执行、缓存和调度。

Task0 从请求入口开始；模型前向与生成循环分别在 Task1、2 简短对照，块管理和轮次选择在 Task3、4 继续阅读。安装见 [Quickstart](./quickstart.md)，固定代码版本、模型与运行准备见[教学引擎资产](./experiment_assets.md#teaching-engine)；下面的观察顺序不表示已完成本机实验。

<a id="request-entry"></a>

## 从两条输入跟到正常结束

先读[请求链路与指标](./01_request_path_and_metrics.md)，区分请求、序列和批次：请求是一次生成任务，序列保存它当前的输入与输出词元，批次则是某一轮一起执行的序列集合。以 [nano-vLLM 示例](https://github.com/GeeeekExplorer/nano-vllm/blob/main/example.py)为起点，观察提示词如何经过对话模板变成模型输入，采样参数如何传入生成调用，返回结果怎样对应原输入。

阅读时先只保留一条短输入，再观察两条输入。记录输入词元、生成上限、返回词元和停止条件，确认“传入两条提示词”不等于“所有请求始终占用同一个固定批次”。首轮读到能够指出输入、生成循环和结束位置即可；具体排序与补位在轮次调度部分继续。离线调用的整批耗时不能直接当作首 token 延迟（TTFT）或后续 token 平均耗时（TPOT）：这两个指标需要对应的输出事件，请求测量由 [Part 02 · 66 单卡推理测量基线](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.md)展开。

| 调用方式 | 谁接收请求并交还结果 | 能从哪些事件开始观察 |
| --- | --- | --- |
| 示例中的离线生成 | 当前 Python 程序调用引擎，生成调用返回后读取结果 | 调用开始、内部生成步与整批返回 |
| 在线推理服务 | 长驻服务接收客户端请求，可能逐段发送生成结果 | 客户端提交、首 token 到达、后续输出与请求完成 |

<a id="model-forward"></a>

## 看一次前向接收了什么

结合[模型前向与计算执行](./02_prefill_and_attention_kernel.md)，阅读[模型执行器](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/model_runner.py)：被选中的序列怎样整理为输入词元、位置和缓存信息，再送入模型？预填充处理尚未计算的输入，解码则利用历史 KV 推进生成，输入规模和历史读取量不同。

只对照两阶段传入模型的信息与输出用途。需要解释 CUDA Graph 时，再读 [Part 04 · 11 CUDA Graph 与 JIT 编译](../../04_CUDA_and_System_Optimization/11_CUDA_Graph_and_JIT_Compile.md)，区分准备、捕获与重放；首次初始化不能混入稳态前向耗时。

<a id="generation-loop"></a>

## 看序列怎样更新与停止

沿[生成循环](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/llm_engine.py)检查：一次模型调用后怎样选择下一个词元、把它追加到序列，遇到结束词元（EOS）或长度上限后怎样完成请求。已输出词元与已写入 KV 的位置需要分别跟踪；刚选出的词元是否已经参与模型计算，要根据它所在的生成步判断。

这条普通生成循环是[自回归生成与投机验证](./03_decoding_strategies.md)的参照。投机方法一次提出多个候选，还需要接受、修正和回退，不能仅把普通循环的每步输出数增大就视为完成投机解码。

<a id="cache-blocks"></a>

## 看缓存块怎样申请、共享与释放

理解[KV 状态、分页与复用](./04_kv_cache_lifecycle_and_reuse.md)后，再读[块管理器](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/block_manager.py)。块表把序列中的逻辑位置映射到物理存储；重点跟踪新请求的分配、序列增长、已有块的复用，以及请求结束时引用怎样变化。

不要把“请求结束”“引用归零”和“状态被覆盖”画成同一个事件。将这些状态变化与 [Mini-SGLang 前缀索引](./mini_sglang_cache_scheduling.md#prefix-cache)并排观察：前缀索引回答已有内容能匹配到哪里，分配器回答状态存在哪里、哪些空间可以再次使用。两者并非互相替代的缓存方案。

<a id="iteration-scheduling"></a>

## 看下一轮怎样选择请求

沿[调度器](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/scheduler.py)找出等待与运行集合，观察请求进入运行集合、生成一步和完成退出的时点。使用[服务调度正文](./05_serving_scheduling_and_pd.md)中的“三个解码请求加一个新长输入”情景，记录每轮成员、输入推进量与容量条件，而不是先比较总吞吐。

再进入 [Mini-SGLang 阶段调度对照](./mini_sglang_cache_scheduling.md#phase-scheduling)，区分选择谁、每轮处理多少输入、哪些准备可与 GPU 计算重叠。源码说明实现怎样安排工作，性能结论仍需要同一后端、同一负载下的受控比较。
