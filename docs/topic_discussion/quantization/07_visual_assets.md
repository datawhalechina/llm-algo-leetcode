# 07. Visual Assets | 图册收口

## 页面目标

本页收口量化专题的关键图，方便后续把量化对象、训练时机、后训练压缩、执行栈低精度和部署决策串起来。

## 图册职责

`07` 不是装饰页，而是把 `01-06` 的量化关系压成一组能快速回答下面问题的图：

- 量化压的是哪一种对象？
- 当前应该先看 PTQ / QAT，还是看 GPTQ / AWQ / FP8 / cache quant？
- 这条量化路线最终服务的是精度、显存、带宽，还是部署成本？

## 建议图册

- quantization object / error 总图
- PTQ / QAT timing 图
- low-bit training adaptation 图
- weight-only compression 图
- FP8 / KV cache quant 图
- deployment / keep-tune-switch 决策图

## 当前已落地图

### 01 Quantization Object / Error

> 图示占位：Quantization objects and error routes 尚未生成。

### 02 PTQ / QAT Timing

> 图示占位：PTQ and QAT timing 尚未生成。

### 04 Weight-Only Compression

> 图示占位：Weight-only compression 尚未生成。

### 05 FP8 / KV Cache Quantization

> 图示占位：FP8 and KV cache quantization 尚未生成。

## 建议顺序

1. 量化对象与误差：对应 `01`
2. PTQ / QAT 介入时机：对应 `02`
3. 低比特训练适配：对应 `03`
4. 权重量化与后训练压缩：对应 `04`
5. FP8 与 KV cache quant：对应 `05`
6. 部署与 benchmark 决策：对应 `06`

## 图的风格约束

- 一张图只回答一个量化问题，不把所有低比特路线塞在一起。
- 正式资产优先用 `SVG`。
- 图标题尽量直接说明“压的是哪类对象、换来什么、代价在哪里”。

## 相关跳转

- 回到 [量化与压缩入口](./intro.md)
- 回到 [量化与压缩正文](./casebook.md)
- 回到 [量化与压缩深入阅读](./walkthrough.md)
