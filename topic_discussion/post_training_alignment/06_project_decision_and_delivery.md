# 06 项目决策与交付

## 页面目标

这一页把 `02-05` 的方法、数据和评测重新收束成项目决策问题：  

本页的输出是可交付判断：明确采用、调优或回退，并说明方法、数据、质量、系统成本和后续验证之间的证据关系。
什么时候值得 adopt，什么时候只是继续 tune，什么时候应该 reject。

## 问题起点

项目页真正需要的不是“我学了哪种方法”，而是：

- 当前数据形态适合哪条路线？
- 训练和系统代价是否值得？
- 评测是否真的支持结果更优？
- 最终项目交付该给出什么结论？

## 决策框架

可以按下面四步判断：

1. **先看前置是否满足**  
   SFT / LoRA 基线是否稳定，偏好数据是否可用。
2. **再看方法是否匹配**  
   是 pairwise preference，还是 group-wise candidate comparison。
3. **再看评测是否一致**  
   指标是否真的反映对齐收益。
4. **最后给出结论**  
   `adopt / tune / reject`，而不是只报 loss。

## 离线到在线的收口链

如果把 `Part 02` 的项目页接回这个专题，最自然的顺序是：

`84 DPO Preference Project -> 85 GRPO Groupwise Alignment Project -> 86 DPO Online Benchmark`

- `84` 负责验证离线 pairwise preference 是否已经形成可信的 DPO 收益。
- `85` 负责验证 group-wise 候选比较是否真的比 baseline 更稳。
- `86` 负责把离线结论推进到在线更新场景，判断收益、代价和安全阈值能否同时成立。

也就是说，这一页不是替代项目页，而是把三类项目页放回同一套决策坐标里。

## 项目页对应

| 项目页 | 你要确认什么 |
|:---|:---|
| `84 DPO Preference Project` | preference pair、reference 口径、评测结果是否支持采用 DPO |
| `85 GRPO Groupwise Alignment Project` | 候选组构造、group-wise 结果和评测是否支持采用 GRPO |
| `86 DPO Online Benchmark` | 在线收益、更新时延、稳定性和安全阈值是否支持进入正式闭环 |

## 决策口径对照

| 决策 | 什么时候给出 | 下一步优先做什么 |
|:---|:---|:---|
| `accept` | baseline 合法，candidate 在核心指标上形成稳定增益 | 进入更长训练、更多数据或在线验证 |
| `tune` | 已有局部收益，但数据、波动或代价边界还不够稳 | 先回数据模板、奖励设计、beta、更新频率等关键旋钮 |
| `reject` | 比较口径不合法，或 candidate 没有形成可信增益 | 先回前置审计，而不是继续堆训练 |

## 可视化入口

> 正文暂不嵌入未审核图示；相关图册与占位说明见 [视觉资产页](./07_visual_assets.md)。

## 常见失败模式

- 方法选对了，但数据质量不足，结果不稳。
- 指标提升了，但无法解释是否真是偏好收益。
- 离线结论看起来不错，但一到在线更新就被时延、波动或安全阈值打回。
- 项目页有很多实验表，却没有明确 adopt / tune / reject 结论。

## 对应 Part 02

- `84 DPO Preference Project`
- `85 GRPO Groupwise Alignment Project`
- `86 DPO Online Benchmark`
- 前置回跳：`15 / 16 / 50`
- 专题入口回跳：[后训练优化入口](./intro.md)

## 文献锚点

- DPO / GRPO 项目化实践资料。
- 对齐 benchmark 报告与项目采用标准说明。

## 项目结论

后训练专题真正的收口，不是“我知道几个方法”，而是“我能把 `84 / 85 / 86` 这类项目页放回同一套方法、数据、评测和交付坐标里，给出可执行的 adopt / tune / reject 判断”。

## 回到项目

将结论回填到 `84 DPO 偏好项目 -> 85 GRPO 组内对齐项目 -> 86 在线 DPO benchmark`。离线质量提升但系统成本或在线链路不可接受时，应保留为 `tune`，而不是直接 `accept`。
