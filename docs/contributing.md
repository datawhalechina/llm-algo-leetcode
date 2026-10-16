# 贡献指南

欢迎参与“大模型算法实战教程 / LLM Algorithm Practice Lab”。

本指南是贡献流程的入口，不重复页面模板和维护命令。先判断改动类型，再进入对应文档。

## 贡献入口

| 要做的事 | 下一步 |
|:---|:---|
| 阅读或运行现有 Notebook | 查看[使用指南](./guide.md)，选择环境并完成项目预检 |
| 编写或修改教学页面 | 从[模板总纲](./template_guidelines.md)选择页面类型，再进入对应专项规范 |
| 同步镜像、运行仓库检查或发布 | 按[维护与发布手册](./maintenance.md)执行 |
| 修改环境、结果或自动化资产 | 先在[资产总登记](./asset_registry.md)确认唯一维护源 |

## 贡献方式

- 提 Issue：反馈错误、补充建议、提出新题
- 提 PR：修正文档、补练习、改实现
- 参与讨论：在 GitHub Discussions 交流

## 基本流程

```text
确定改动类型
→ 阅读对应模板或资产说明
→ 修改源文件
→ 按维护手册同步并验证
→ 检查提交范围
```

统一验证命令和单页排查脚本只在[维护手册的常用命令](./maintenance.md#常用命令)中维护。

## 提交前检查

- 修改了正确的源文件，没有反向维护 `docs/` 正文镜像。
- 页面符合对应专项规范；环境、结果和资产字段来自各自的唯一维护源。
- 已按维护手册同步镜像、运行适用验证并检查实际渲染。
- 提交范围不包含无关改动、缓存、意外覆盖的结果或本地凭据。

## 相关链接

- [GitHub 仓库](https://github.com/datawhalechina/llm-algo-leetcode)
- [GitHub Issues](https://github.com/datawhalechina/llm-algo-leetcode/issues)
- [GitHub Discussions](https://github.com/datawhalechina/llm-algo-leetcode/discussions)
