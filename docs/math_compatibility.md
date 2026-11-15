# 数学公式的双端兼容与验收

公式同时面向 GitHub Markdown 与 VitePress。行内使用 `$...$`，独立公式使用 `$$...$$`；行内公式起始符前保留空格，尤其是中文逗号、顿号、冒号和句号之后。不能只搜索某个宏就认为整页已经完成公式审核。

1. GitHub 已出现 `operatorname` 禁止宏报错；自定义函数名按含义选用 `\mathrm{...}` 等兼容写法，保留参数、索引与数学定义。不要为统一外观改写算法目标。
2. `\,`、`\!` 等排版微调及 `\%`、`\#`、`\{`、`\}` 可能在数学渲染前被 Markdown 处理。优先删除非必要间距；乘法需要显式表达时用 `\cdot`，集合括号可用 `\lbrace`／`\rbrace`，普通百分比放在正文。改写计数表达式时保留原有统计含义。
3. 星号上标优先使用 `^{\ast}`。重音命令后接下划线等易与 Markdown 强调规则冲突的写法，需要逐项复核；不能通过去掉下标、符号或公式来消除错误。
4. 静态回归命令：`python tools/check_math_formula_symbols.py --github-compatible --static-only --file <源文件>`。支持 Markdown 与 Notebook 的 Markdown 单元，忽略代码块及行内代码；去掉 `--static-only` 后另外执行本地 MathJax 解析，需已有的站点依赖。
5. 验收分别记录公式提取、字符保真、本地解析与 GitHub 最终显示。静态检查无风险、API 产生数学元素或站点构建成功，都不能替代 GitHub 最终渲染；在线接口不可用时应明确记录未验证。Notebook 修复只改 Markdown，代码、输出和 metadata 需保持不变；docs 镜像通过脚本同步。
