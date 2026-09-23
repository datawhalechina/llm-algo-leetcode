# 06 Benchmark 与项目验证

## 页面目标

这一页负责把编译和 backend 结论收束到 benchmark 与项目验证：到底哪种执行方案值得保留。

本页的输出是可交付决策：在质量、延迟、吞吐、显存、编译时间和部署复杂度之间明确 `keep / tune / switch backend / reject`。

## 决策框架

1. 先确认 graph / lowering / backend 假设是否一致。
2. 再确认 benchmark 是否覆盖真实 workload。
3. 最后判断是：
   - `keep`
   - `tune`
   - `switch backend`
   - `reject`

## 可视化入口

> 正文暂不嵌入未审核图示；相关图册与占位说明见 [视觉资产页](./07_visual_assets.md)。

## 对应 Part

- `2.2`
- `2.6`
- `2.7`
- `2.10`

## 项目结论

编译专题的终点不是“生成了更复杂的执行链”，而是“项目结果是否真的更好”。

## 回到项目

将结论回填到 `66 推理性能比较 -> 67 量化推理与部署 -> 74 Profiling 驱动优化`。局部 kernel 变快但端到端结果不变时，应保留为待调优或回退，而不是直接宣布图优化成功。
