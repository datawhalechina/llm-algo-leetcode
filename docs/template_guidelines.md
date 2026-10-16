# 大模型算法实战教程页面模板总纲

本文件是教程页面编写规范的入口。先判断页面类型，再进入对应专项规范；环境事实、结果 schema、网页同步与发布流程分别由各自的单一维护源负责。

## 一、选择页面类型

| 页面或任务 | 核心产出 | 使用的规范 |
| --- | --- | --- |
| 专题 intro、walkthrough、casebook、Quickstart | 学习路线、页面分工、环境入口与证据出口 | [专题页面编写规范](./authoring/topic_pages.md) |
| Part 00–01 讲解型 Notebook | 概念关系、数量级和最小代码验证 | [基础页编写规范](./authoring/foundation_notebooks.md) |
| Part 02 非项目机制页 | 局部机制实现、测试与受控真实环境证据 | [机制页编写规范](./authoring/part02_mechanism_notebooks.md) |
| Part 02 项目页 | 固定工作负载、真实产物、项目决策与交接 | [项目页编写规范](./authoring/part02_project_notebooks.md) |
| Notebook 正文图与 SVG | 对象关系、状态变化、时间线或证据路径 | [图示与 SVG 规范](./authoring/visual_guidelines.md) |
| 源码转网页、镜像、构建和发布 | 可复现的文档生成与验证 | [维护手册](./maintenance.md) |

完整入口也可从[教程页面编写规范索引](./authoring/README.md)进入。

## 二、规则优先级与单一维护源

发生冲突时按以下顺序处理：

1. 当前页面类型的专项规范优先于本总纲的共同规则。
2. 结果字段、证据等级和 schema 以 [Part 02 结果资产与证据契约](./part02_result_assets.md)为准。
3. Profile、依赖版本、驱动和硬件事实以 [GPU 环境资产表](./gpu_environment_assets.md)为准。
4. 转换、镜像、构建和发布命令以[维护手册](./maintenance.md)为准。
5. 明确标为历史兼容的说明不能覆盖当前模板。

同一事实只维护一次：安装命令属于 Quickstart，精确版本属于环境资产，实验结果标签属于专题实验资产表和结果 JSON，Notebook 只声明本次运行配置与学习者需要观察的证据。

## 三、所有 Notebook 的共同规则

- Notebook 是教学内容的单一源码；生成的网页 Markdown 不反向维护。
- 文件名、H1 和页面主题保持一致；编号表示目录顺序，稳定身份由语义 ID、注册表或结果资产记录。
- 每个 cell 必须有唯一 id；代码输出必须符合合法 Notebook schema。
- 中文承担解释；API、代码标识符、模型 ID、协议名和 JSON 字段保留原文。
- 正文直接说明对象、变化、原因、代价和判断动作，不使用建设状态、模板治理和抽象口号代替教学内容。
- 文字提出问题和解释因果，表格表达精确映射，图片呈现关系；三者不逐项复述。
- CPU 示例只能证明公式、状态、控制流或数量级；GPU、Kernel、backend、通信和服务结论必须来自固定工作负载下的真实证据。
- 前置阅读只保留理解当前内容不可缺少的入口；相关阅读是最后一个非空 Markdown cell。

## 四、Notebook 结构分流

Part 00–01 使用按问题推进的讲解型结构；Part 02 的机制页和项目页使用 Step、TODO、测试、答案与可选真实实验结构。具体顺序只在对应专项规范维护，不在总纲复制。

<a id="part00-part01-template"></a>
## 五、Part 00–01 基础页

Q 递进、代码角色、主图、阅读链接、metadata 和验收清单见[基础页编写规范](./authoring/foundation_notebooks.md)。

<a id="二项目节分类与-step1-模板"></a>
## 六、Part 02 项目页

Project、Benchmark、Analysis、Decision 与 Qualification 的分类，Step 1–5.6、运行状态机、结果登记和交接规则见[项目页编写规范](./authoring/part02_project_notebooks.md)。

<a id="五part-02-非项目机制页模板"></a>
## 七、Part 02 非项目机制页

导读、Step 1–5、TODO、测试、答案、解析、可选 GPU 实验和 metadata 规则见[机制页编写规范](./authoring/part02_mechanism_notebooks.md)。

## 八、图示与 SVG

图示职责、布局、连线、配色和自动验证规则见[图示与 SVG 规范](./authoring/visual_guidelines.md)。字段清单优先使用表格；只有关系、状态变化、分支或时间顺序确实更易由图表达时才增加图片。

## 九、验证入口

按页面类型完成专项验收后，使用[维护手册](./maintenance.md#同步与验证层级)中的统一入口完成转换、镜像、链接和站点验证。验证失败时修复对应单一维护源，不在页面正文增加临时例外。
