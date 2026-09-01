# 06. Benchmark and Decision | 端到端对比与选型

## 页面目标

本节负责把前面的机制判断收束到 `66` 的 benchmark report 里。

完整的命令、运行开关、结果文件和 JSON schema 检查见[66–70 推理项目验证清单](../../docs/verification/inference_projects.md)；本节只保留统一口径、项目分级和 `accept / tune / reject` 的判断规则。

## 本节在路线中的位置

本节对应 **Task6：综合 benchmark 与项目决策**。它不再新增推理机制，而是把 Task0–5 的 workload、指标、瓶颈诊断和候选策略放回同一套实验口径，形成可复查的采用建议。

`66` 是核心综合项目；`67`、`69` 是主题项目，分别验证量化部署和 Prefix Cache；`68`、`70` 是可按目标选择的扩展项目。它们可以提供局部证据，但最终综合判断仍应回到同一个 workload。

## 问题起点

如果前面的 `01-05` 负责解释“慢在哪里、为什么会慢、有哪些候选动作”，那么 `06` 负责回答最后一个问题：**这次优化值不值得留下来**。

如果缺少统一 benchmark，前面的技巧就无法转化为可复查的工程判断。

## 你要先确认什么

- workload 是否固定。
- baseline 和 candidate 是否只改一个变量。
- TTFT、TPOT、throughput 和 peak memory 是否一起报。
- 是否记录模型、backend、dtype、prompt tokens、generated tokens、batch、concurrency 和 cache policy。
- 是否保留质量、acceptance rate、cache hit rate 或公平性等策略特有约束。

## 项目闭环

```text
workload config
      │
      ▼
prefill/decode metrics
      │
      ▼
bottleneck diagnosis
      │
      ▼
baseline vs candidate comparison
      │
      ▼
accept / tune / reject
```

## 66 的实验步骤

完整启动和保存逻辑见 [66 Inference Performance Comparison](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb)。执行时按以下顺序保持口径一致：

1. 探测 GPU、驱动、PyTorch、vLLM 和可用 dtype；不直接复制另一台机器的版本。
2. 固定模型、backend、workload、prompt tokens、generated tokens、`batch`、`concurrency`、cache policy 和 warm-up。
3. 先启动真实 backend，确认 `/v1/models` 和一次最小请求可用。
4. 运行 baseline，再只改变一个变量运行 candidate；每组保存 JSON 和服务启动参数。
5. 同时查看 TTFT、TPOT、E2E latency、请求/输出吞吐、P99 和峰值显存。
6. 根据目标场景判断：交互服务优先延迟，离线批处理优先吞吐，最后输出 `accept / tune / reject`。

当前 66 的本机实测表明，并发从 1 增加到 4 后输出吞吐提高，但 TTFT 和 E2E 延迟明显上升；这是服务目标权衡的证据，不是“并发越高越好”。

## 为什么 `66` 是项目收口

`66` 的价值不在于再讲一遍机制，而在于把前面的判断塞回同一个 workload。只有在同一个模型、backend、batch、prompt tokens、generated tokens、dtype 和 cache policy 下，下面这些结论才有意义：

- FlashAttention 值不值得保留；
- speculative decoding 是真的更快，还是 acceptance 太低；
- KV cache 管理是否真的让并发收益上来；
- 量化到底是帮了忙，还是只是把代价换了个位置。

## 判定原则

- `accept`：候选方案在当前 workload 和约束下值得采用。
- `tune`：方向有效，但仍需要调参、调整请求分布或补充证据。
- `reject`：当前约束下收益不足、代价过高或质量不达标。

不要把“某个指标最好”直接等同于 `accept`。在线交互通常更关注 TTFT / P99，离线批处理可能更关注 throughput / cost；决策必须回扣 Task0 定义的服务目标。

## 报告应该怎么写

一个合格的推理优化报告，至少要同时说明：

- 你改的是 prefill、decode、cache 还是量化；
- 这次改动对应的是哪一种瓶颈诊断；
- 指标变化是否和目标场景一致；
- 候选方案有没有引入新的副作用；
- 是否满足服务目标、质量约束和资源预算；
- 下一步是 `accept`、`tune` 还是 `reject`。

> 正文暂不嵌入未审核图示；相关图册与占位说明见 [视觉资产页](./07_visual_assets.md)。

## 证据边界

CPU 可以执行指标聚合、schema 检查和 `accept / tune / reject` 逻辑；真实的推理性能结论需要固定 workload、可复现的 backend 运行和完整报告。单次 smoke test 只能证明服务可用，不能代表稳定 benchmark。

## 报告清单

- workload 是否固定：模型、backend、batch、prompt tokens、generated tokens、dtype、cache policy。
- 是否拆分 prefill 和 decode。
- 是否同时报告 TTFT、TPOT、throughput 和 peak memory。
- candidate 是否只改一个变量。
- 主要瓶颈是否能解释下一步动作。

## 66–70 的统一结果格式与 Practice 分级

五个项目允许保留策略特有指标，但公共结果字段不再各写一套。Notebook 使用
`tools/inference_result_schema.py` 生成 `inference-benchmark/v1` 结果，至少包含：

```text
config:
  model / backend / dtype / prompt_tokens / generated_tokens
  batch / concurrency / cache_policy

metrics:
  ttft_ms / tpot_ms / e2e_latency_ms
  throughput_tokens_per_s / p99_ms / peak_memory_mb

quality:      质量、acceptance rate 或误差约束
decision:     accept / tune / reject / not_evaluated
```

`68` 的 acceptance rate、`69` 的 cache hit rate、`70` 的公平性和队列开销等，放在
`strategy_metrics`，不改变公共字段。旧版 benchmark 的原始字段仍保留在报告中，便于兼容已有结果。

| 项目 | 默认实践级别 | 真实 backend 入口 | 不能直接宣称的内容 |
|---|---|---|---|
| `66` | Practice-P2 | 自动解析模型、选择 dtype/端口、启动 vLLM 并保存 JSON | smoke test 不足以代表稳定线上结论 |
| `67` | Practice-P1，真实服务为可选 P2 | 可做 backend smoke test；量化格式参数需按引擎补充 | 服务能启动不等于量化收益成立 |
| `68` | Practice-P1；P2 为扩展 | 可验证 baseline endpoint；speculative 还需要 draft/verify 能力 | 普通 vLLM baseline 不等于 speculative 实验 |
| `69` | Practice-P1，真实服务为可选 P2 | 可自动启动服务，策略开关需按 vLLM/SGLang 版本确认 | 服务启动成功不等于 prefix cache 已命中 |
| `70` | Practice-P1，真实服务为可选 P2 | 可自动启动服务并跑并发 workload | 单一并发值不能代表调度器整体收益 |

真实入口统一使用 `tools/inference_project_runtime.py`：`auto / modelscope /
huggingface / local` 选择模型来源，dtype 和空闲端口由 helper 处理，服务在 `finally`
中关闭。Colab / ModelScope 仍需先从仓库根目录运行 Notebook；无 GPU 时保持开关关闭，
完成 CPU-first 题目区。

## 最小报告模板

```text
workload:
  model / backend / dtype / batch / concurrency
  prompt_tokens / generated_tokens / cache_policy

diagnosis:
  prefill-bound / decode-bound / memory-bound / serving-bound

metrics:
  TTFT / TPOT / e2e_latency / throughput / P99 / peak_memory
  quality / acceptance_rate / cache_hit_rate / fairness (按策略选择)

decision:
  accept / tune / reject
  reason / next_action
```

## 文献与工程入口

- [66 Inference Performance Comparison](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb)
- [67 Quantized Inference and Deployment](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb)
- [68 Speculative Decoding Benchmark](../../02_PyTorch_Algorithms/68_Speculative_Decoding_Benchmark.ipynb)
- [69 Prefix Caching Benchmark](../../02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark.ipynb)
- [70 Serving Scheduler Benchmark](../../02_PyTorch_Algorithms/70_Serving_Scheduler_Benchmark.ipynb)
- 性能分析：当报告还无法证明慢点在哪里时先回去补 profiling。
- 推理优化 `01-05`：当报告还不能解释“为什么该切换/保留”时，回到对应问题页。

## 经典阅读入口

- [66 Inference Performance Comparison](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb)
- [01 Request Path and Metrics](./01_request_path_and_metrics.md)
- [05 Quantized Inference and Deployment](./05_quantized_inference_and_deployment.md)

## 项目结论

`06` 不是新增机制页，而是把前面的判断变成最终结论。合格的收口报告不只回答“哪个方案更快”，还要回答“在什么 workload、什么服务目标和什么资源约束下，哪个方案值得采用”。
