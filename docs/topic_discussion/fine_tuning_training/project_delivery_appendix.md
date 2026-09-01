# 项目交付附录：Checkpoint / Tracking / Artifact / Report

## 页面目标

这一页回答的是：当 `SFT / LoRA` 已经能跑、训练工程也已经搭起来之后，怎样把实验真正收成可恢复、可复盘、可交付的项目结果。

它是 `监督微调与训练工程` 的项目交付附录页，不替代 `05 Project Delivery` 主线，只负责把“训练完成”推进到“结果可交付”。

## 这页负责什么

- `checkpoint save/load strategy`：关注从哪一步恢复、保存哪些状态、如何避免恢复口径不一致。
- `artifact checklist`：关注 `adapter / tokenizer / config / report / samples` 是否齐全。
- `logging / experiment tracking`：关注训练过程是否留下足够证据支撑最终结论。
- `report template`：关注 baseline、candidate、资源指标和采用建议怎么写清楚。

60–65 的公共报告协议见[训练微调项目验证清单](../../verification/fine_tuning_projects.md)，用于统一结果外层结构；本附录继续负责 checkpoint、artifact、tracking 和交付证据的细节。

## 这页不展开什么

- `LoRA / QLoRA` 的机制差异
- `scheduler / accumulation / AMP` 的训练控制原理
- `DDP / FSDP / ZeRO / DeepSpeed` 的并行机制细节
- 完整训练平台、模型仓库和在线部署系统

这些分别放在 `监督微调主线`、`训练工程附录`、`通信与并行` 和项目页里更合适。

## 最小交付闭环

一个最小的 SFT 项目交付，至少要能回答下面四件事：

1. 这次训练是基于什么数据和什么训练口径完成的
2. 如果中断，能从哪里恢复、恢复后会不会改写结论
3. 最终有哪些 artifact 可以交给别人复现或继续使用
4. 为什么这个方案值得 `adopt / tune / reject`

如果这四件事答不清，训练结果通常还停留在“实验现象”，没有进入“项目交付”。

## checkpoint save/load 要留什么

至少要确认下面这些状态有没有保存：

- `model / adapter weights`
- `tokenizer / special tokens / chat template`
- `optimizer / scheduler state`
- `global step / epoch / accumulation progress`
- `random seed / config snapshot`

一个实用判断是：恢复后至少要能回答“这是不是同一条训练线的继续”，而不是“又重新跑了一次看起来差不多的实验”。

## artifact checklist

| 类别 | 最少要留什么 | 为什么必须留 |
|:---|:---|:---|
| 模型产物 | `adapter` 或 merge 后权重 | 没有最终权重就无法复用 |
| 文本侧产物 | `tokenizer / vocab / special tokens / template` | tokenizer 漏掉时最容易复现失败 |
| 配置产物 | `train config / LoRA config / eval config` | 只留权重不足以解释训练口径 |
| 结果产物 | `train/val metrics`、样例输出 | 需要能回看训练证据 |
| 结论产物 | `report / adopt decision` | 项目需要明确结论而不只是日志 |

## logging / experiment tracking

不要求一开始就接完整平台，但至少要有一条稳定的记录路径。

最少建议记录：

- `train loss / val loss`
- `learning rate`
- `tokens per step` 或基础吞吐
- `max memory` 或显存占用
- 周期性生成样例
- checkpoint 保存点

如果已经接了 `W&B / TensorBoard / MLflow`，重点不是“工具更高级”，而是：

- 同一条训练线是否能回溯
- baseline 和 candidate 是否能并排对照
- 中断恢复后曲线是否还能接得上

## 一个简化报告模板

项目报告至少建议有这几块：

1. `Goal`
   - 这次要比较什么，为什么比较
2. `Setup`
   - 数据、模型、LoRA 配置、训练步数、batch、scheduler
3. `Evidence`
   - train / val 曲线、样例输出、吞吐、显存
4. `Artifacts`
   - 保存了哪些 adapter、tokenizer、config、report
5. `Decision`
   - `adopt / tune / reject`

这里最常见的问题不是证据不够多，而是证据和结论没对齐。

## adopt / tune / reject 怎么落

- `adopt`
  - 效果达到目标，训练成本和交付完整度都可接受
- `tune`
  - 主方向成立，但数据、LoRA 配置、训练步数或评测口径还需要继续修
- `reject`
  - 方案没有证明比 baseline 更值得采用，或者 artifact / 证据链不完整

不要把“loss 下降了”直接当成 `adopt`，也不要把“有一两个坏样例”直接当成 `reject`。

## 和主线怎么连接

| 当你在主线里卡住什么 | 回到这页看什么 |
|:---|:---|
| `04` 实验跑完，但无法复盘 | `logging / tracking / report template` |
| `05` 要做项目页结论，但 artifact 不全 | `artifact checklist` |
| 训练中断后想继续 | `checkpoint save/load strategy` |
| baseline 和 LoRA 对照不清 | `evidence + decision` 模板 |

## 相关专题

- [训练工程附录](./training_engineering_appendix.md)：当你还在搭 `Trainer / Accelerate / DeepSpeed / Lightning` 这层工程闭环时先看那里。
- [性能分析](../profiling/intro.md)：当你需要补吞吐、显存和训练热点证据时先看这里。
- [显存优化](../memory_performance_tuning/intro.md)：当项目交付需要解释 OOM、checkpointing 或资源账本时先看这里。

## 本节要点

项目交付附录的重点不是再跑一次训练，而是把 checkpoint、artifact、tracking 和最终结论收成一套别人能接手的证据链。
