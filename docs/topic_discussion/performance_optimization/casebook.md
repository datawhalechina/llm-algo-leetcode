# 性能优化判断手册

本手册按性能现象选择测量对象和下一步动作，不替代 Task0–6 的连续学习。

统一判断链：

```text
现象 → 阶段 → 对象 → 证据 → 策略代价 → 复测与决策
```

| 现象 | 优先检查 | 可能方向 | 不能直接推出的结论 |
|:---|:---|:---|:---|
| step time 变长 | forward、backward、optimizer、数据加载 | profiler、数据管道、算子或调度 | 不能只凭一个慢 kernel 定位原因 |
| 峰值显存升高 | 参数、Activation、缓存、workspace、临时张量 | checkpoint、offload、量化、分页和并发控制 | 显存下降不等于端到端更快 |
| GPU 利用率低 | CPU 等待、I/O、同步、通信和 kernel 空洞 | overlap、数据管道、batch 和调度 | 利用率低不一定是算力不足 |
| 吞吐提高但延迟变差 | batch、队列、P99、TTFT / TPOT | 并发策略、服务调度和容量控制 | 平均吞吐不能代表交互质量 |
| 多卡扩展不理想 | collective、拓扑、通信占比和负载均衡 | TP、PP、DP、EP、通信重叠 | GPU 数量增加不保证线性加速 |

最终结论必须包含：固定条件、baseline、candidate、指标变化、质量门槛、证据等级和适用范围。
