# 02 可观测性与 Profiling 工具链

## 本节定位

建立从轻量计时到 trace 和系统级工具的证据链，回答“时间、显存和等待消耗在哪里”。

## 内容安排

1. 计时边界：warmup、同步、重复次数和测量区间。
2. 时间证据：operator、kernel、launch、CPU 等待和阶段拆分。
3. 显存证据：allocation、reserved、峰值、驻留和生命周期。
4. 系统证据：CPU-GPU overlap、通信等待、I/O 和调度空洞。
5. 证据解释：从热点观察升级为可证伪的瓶颈假设。

## 主要产出

一份带测量条件、热点、证据等级和下一步验证动作的 profiling 记录。
