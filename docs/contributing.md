# 贡献指南

欢迎参与“大模型算法与系统教程”。

本指南是贡献流程的入口，不重复页面模板和维护命令。先判断改动类型，再进入对应文档。

## 贡献入口

| 要做的事 | 下一步 |
|:---|:---|
| 阅读或运行现有 Notebook | 查看[使用指南](./guide.md)，选择环境并完成项目预检 |
| 编写或修改教学页面 | 参考同类源 Notebook／Markdown 的结构，并按下方要求检查 |
| 同步镜像、运行仓库检查或发布 | 按[维护与发布手册](./maintenance.md)执行 |
| 修改环境、结果或自动化工具 | 分别查看[环境索引](./gpu_environment_assets.md)、[公共结果说明](./part02_result_assets.md)和[仓库工具说明](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/tools/README.md) |

## 贡献方式

- 提 Issue：反馈错误、补充建议、提出新题
- 提 PR：修正文档、补练习、改实现
- 参与讨论：在 GitHub Discussions 交流

## 基本流程

```text
确定改动类型
→ 参考同类页面和对应技术说明
→ 修改源文件
→ 按维护手册同步并验证
→ 检查提交范围
```

统一验证命令和单页排查脚本只在[维护手册的常用命令](./maintenance.md#常用命令)中维护。

## 内容、代码与图片

- 本节导读从具体问题引入；核心术语在首次参与解释时说明用途，必要时配短例子。
- 机制说明、题目、测试与解析使用相同定义；测试检查数值、状态与边界，解析说明答案为何成立。
- 图片解释数据流、执行顺序或约束，表格表达需要精确对照的信息，不用二者重复目录。
- CPU 演示、GPU 测量和真实后端结果分别记录；未运行的实验保留为待验证。
- 公式同时检查 GitHub 与网页显示，具体写法见[公式兼容说明](./math_compatibility.md)。

## 外部来源与许可

引用论文、官方文档或开源实现时，链接到直接支撑当前内容的来源，并说明具体看什么；涉及接口和模型行为时记录适用版本。引入外部代码或图片时保留来源与许可说明，记录修改，不默认沿用本仓库许可。

本仓库文字与代码的许可分别见 [LICENSE](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/LICENSE) 和 [LICENSE-CODE](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/LICENSE-CODE)。模型权重、数据集与外部素材仍遵循其自身使用条件；下载缓存和私有数据不随教程提交。

## 提交前检查

- 修改了正确的源文件，没有反向维护 `docs/` 正文镜像。
- 页面结构与同类内容一致；环境、结果字段和运行配置来自各自的维护入口。
- 已按维护手册同步镜像、运行适用验证并检查实际渲染。
- 提交范围不包含无关改动、缓存、意外覆盖的结果或本地凭据。

## 相关链接

- [GitHub 仓库](https://github.com/datawhalechina/llm-algo-leetcode)
- [GitHub Issues](https://github.com/datawhalechina/llm-algo-leetcode/issues)
- [GitHub Discussions](https://github.com/datawhalechina/llm-algo-leetcode/discussions)
