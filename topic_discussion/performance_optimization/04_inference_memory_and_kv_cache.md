# 04 推理显存与 KV Cache

## 本节定位

连接请求生命周期、KV Cache 容量、前缀复用和推理并发，解释推理侧显存如何影响 TTFT、TPOT 和服务容量。

## 内容安排

1. KV Cache 账本：层数、KV heads、head dimension、序列长度、并发和 dtype。
2. 生命周期：创建、增长、复用、驱逐和释放。
3. 运行时组织：PagedAttention、Prefix Cache、RadixAttention 和 Chunked Prefill。
4. 架构与表示：GQA、MQA、MLA 和 KV Cache 量化。
5. 验证方法：命中率、峰值显存、TTFT、TPOT、并发和质量。

## 主要产出

一份说明 KV Cache 增长规律、复用条件和容量决策的证据表。
