# LLM-Algo-LeetCode 题目与网页化模板指南

为了保证整个项目在源码（Jupyter Notebook）和前端网页（VitePress）上的展示一致性、防剧透体验以及低维护成本，所有题目请严格遵循以下结构与模板规范。

## 一、 源码规范：Jupyter Notebook (`.ipynb`) 结构

每一个题目的 Notebook 必须采用 **"SSOT (Source of Truth)"** 原则：所有的题目、测试用例、解析和答案都写在同一个 `.ipynb` 文件中。

### 标准文件结构
1. **【题目区】(Markdown + Code)**
   - 介绍核心概念。
   - 给出填空代码（使用 `TODO` 占位符）。
2. **【测试区】(Code)**
   - 提供 `test_xxx()` 函数，用以验证用户填入的代码正确性以及输出性能（如有）。
3. **【防剧透缓冲带】(Markdown)**
   - **必须**放置在测试代码块之后，答案块之前。用于在 Colab 或本地执行时防止用户不小心往下滚看到答案。
4. **【解析与答案区】(Markdown + Code)**
   - 提供详细的思路和无 Bug 的满分代码。

### 尾部三段式模板代码

在 Notebook 的最末尾，请依次添加以下三个 Cell：

**Cell 1: 防剧透缓冲带 (Markdown)**
```markdown
---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---
```

**Cell 2: 答案文字解析 (Markdown)**
*(注意：标题中必须包含 `解析`、`答案` 或 `参考代码` 关键词，以便转换脚本触发折叠！)*
```markdown
## 官方解析与参考代码

**解析：**
1. **TODO 1 (命名/目标)：** [解释第一步的思路...]
2. **TODO 2 (防溢出/加速等)：** [解释工程实现中的细节，如强转 float32，避免 nan 等...]
3. **复杂度与带宽：** [可选：分析 FLOPs 和访存带宽瓶颈...]
```

**Cell 3: 完整参考代码 (Code)**
```python
# 建议命名带有 Solution 后缀，避免和上面的 TODO 模板类名冲突
class RMSNormSolution(nn.Module):
    def __init__(self, ...):
        ...
```

---

## 二、 网页化转换规范：转换脚本逻辑 (`convert_notebook.py`)

我们在根目录下使用 `convert_notebook.py` 将 `.ipynb` 自动转换为 `docs/` 下的 `.md` 文件，实现“源码写一次，网页自动生成”的工作流。

### 转换脚本当期支持的特性：
1. **自动生成云端运行徽章 (Hero Badges)**
   - 脚本会在每一页的最上方自动植入 "Open In Colab" 和 "Open In ModelScope" (魔搭) 徽章。
   - 自动映射 GitHub 源码路径，确保一键跳转。
2. **智能识别并折叠答案 (Smart Folding)**
   - 脚本扫描 Markdown Cell 的文本，一旦匹配到 `答案`、`解析` 或 `solution` 关键词，会自动注入 VitePress 的 `::: details 💡 点击查看官方解析与参考代码` 容器。
   - 将这之后的解析文字和答案代码块全部包裹并默认折叠。
3. **评论区与 PR 引导 (Community CTA)**
   - 在每篇文章的底部，脚本会自动追加引导提交 PR 的 Markdown 块。
   - VitePress 底部自动挂载配置好的 GitHub Giscus 讨论区组件。

## 三、 维护者的日常工作流

1. **出题/修改代码**：只在根目录下的相应目录（如 `02_PyTorch_Algorithms/`）打开或新建 `.ipynb`，按照上文的“尾部三段式”模板编写题目和答案。
2. **生成网页**：在根目录运行 `python3 convert_notebook.py`。
3. **预览校验**：进入 `docs/` 运行 `npm run docs:preview` 查看网页排版是否完美。
4. **提交代码**：`git add` 修改过的 `.ipynb` 和 `docs/` 下的 `.md`，然后 `git commit` & `git push`。
