# 实验入口与结果汇总

本目录存放实验命令的实现，根目录`benchmarks/`仍存工作负载、配置和结果。旧的`tools/<脚本名>.py`命令与导入继续转发到同一模块。

| 命令 | 作用与副作用 |
| --- | --- |
| `python -m tools.benchmarks.aggregate_distributed_evidence --project <编号> --input <结果.json> --output <汇总.json>` | 读取已有结果并写汇总；保留失败和smoke证据，汇总不会自动成为完整模型或服务验收结论 |
| `python -m tools.benchmarks.benchmark_inference_backend --model <名称> --workload <负载.jsonl> --output <结果.json>` | 请求已单独启动的兼容服务，记录延迟、吞吐、请求结果及所支持的质量字段；本命令不启动服务 |

推荐从仓库根目录执行。相对输入和输出仍按当前工作目录解释；指定输出时会创建父目录，并保留原有覆盖行为。正式运行先选择新的结果文件名，不要覆盖已登记测量。

迁移不修改指标计算、请求协议、候选决策或结果结构。请求客户端的真实运行会访问配置的服务地址；迁移测试只使用模拟响应，汇总测试只写临时目录：

```bash
python -S -m unittest tools.tests.test_benchmarks_migration tools.test_benchmark_inference_backend
```
