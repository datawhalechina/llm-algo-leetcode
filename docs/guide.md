# 使用指南

本页面详细介绍 LLM-Algo-LeetCode 的四种学习方式，帮助你选择最适合自己的学习路径。

## 🎓 四种学习方式对比

| 学习方式 | 环境 | GPU | 测试方式 | 适合人群 |
|---------|------|-----|---------|---------|
| **🌐 网页 + 云端 GPU**<br>（零配置，推荐） | 免费云端<br>(Colab/魔搭) | ✅ 免费GPU | Notebook 内置测试 | 零基础 / 无本地环境 |
| **💻 本地 + Jupyter Lab** | 本地虚拟环境 | 可选 | Notebook 内置测试 | 喜欢交互式编程 |
| **🔧 本地 + VSCode + 脚本** | 本地虚拟环境 | 可选 | `test_notebook_answers.py` | 专业开发者 / 批量测试 |
| **☁️ 其他 GPU 平台** | Kaggle/Paperspace 等 | ✅ 云端GPU | Notebook 内置测试 | 需要更强算力 |

> **💡 快速选择：**
> - 🆕 **零基础 / 无本地环境** → 使用方式 1（网页 + 云端 GPU），零配置开始学习
> - 🏠 **有本地环境 / 喜欢 Jupyter** → 使用方式 2（本地 + Jupyter Lab）
> - 👨‍💻 **专业开发者 / 需要批量测试** → 使用方式 3（本地 + VSCode + 脚本）
> - 🚀 **需要更强算力** → 使用方式 4（其他 GPU 平台）

---

## 方式 1：网页 + 云端 GPU（零配置，推荐新手）

### 工作流程

1. **沉浸式阅读与学习**：在本网站查看题目描述、排版精美的数学公式和前置知识
2. **一键跳转云端 IDE**：点击题目顶部的 **"Open In Colab"** 或 **"Open In ModelScope (魔搭)"** 徽章
3. **在云端填空并测试**：浏览器会自动在免费的云端 GPU 环境中拉起对应的 Jupyter Notebook。你可以在 `TODO` 处敲入代码，并运行下方的 `test_xxx()` 测试用例验证正确性
4. **防剧透与查看答案**：如果卡住，可以向下滚动穿过红色的 `🛑 STOP HERE 🛑` 缓冲带查看参考答案
5. **参与讨论**：网页最底端接入了 GitHub Discussions 评论区，欢迎交流不同解法

### 优点

- ✅ **零环境配置**（最大优势）
- ✅ **免费 GPU**（Colab/魔搭提供）
- ✅ 随时随地学习（只需浏览器）
- ✅ 适合快速试错

### 缺点

- ❌ 需要网络连接
- ❌ 云端环境有时间限制
- ❌ 不适合版本控制

---

## 方式 2：本地 + Jupyter Lab

### 工作流程

```bash
# 1. 克隆仓库
git clone https://github.com/datawhalechina/llm-algo-leetcode.git
cd llm-algo-leetcode

# 2. 创建虚拟环境
conda create -n llm_algo python=3.10 -y
conda activate llm_algo

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动 Jupyter Lab
jupyter lab
```

在浏览器中打开 notebook，逐个 cell 运行，运行内置的 `test_xxx()` 函数验证。

### 优点

- ✅ 交互式编程体验
- ✅ 可视化输出（图表、表格）
- ✅ 可使用本地 GPU（如果有）
- ✅ 支持 Markdown 渲染
- ✅ 完全离线工作

### 缺点

- ❌ 需要配置本地环境
- ❌ 不适合批量测试

---

## 方式 3：本地 + VSCode + 脚本测试

### 工作流程

```bash
# 1-3. 同方式 2（克隆仓库、创建环境、安装依赖）

# 4. 用 VSCode 编辑 .ipynb 文件
code 02_PyTorch_Algorithms/00_PyTorch_Warmup.ipynb

# 5. 运行测试脚本
python test_notebook_answers.py 02_PyTorch_Algorithms/00_PyTorch_Warmup.ipynb --mode both
```

### 优点

- ✅ 完全离线工作
- ✅ 可使用本地 GPU（如果有）
- ✅ 版本控制友好（Git）
- ✅ 可批量测试
- ✅ 适合专业开发者

### 缺点

- ❌ 需要配置本地环境
- ❌ 学习曲线稍高

### 测试脚本说明

详细的测试脚本使用方法，请查看 [贡献指南](./contributing.md)。

---

## 方式 4：其他 GPU 平台

### 支持的平台

- **Kaggle Notebooks**（免费 GPU）
- **Paperspace Gradient**（免费/付费）
- **AWS SageMaker Studio Lab**（免费）
- **阿里云 PAI-DSW**（付费）

### 工作流程

1. 在平台上创建 Notebook
2. 从 GitHub 克隆仓库
3. 打开 .ipynb 文件编辑
4. 运行测试

### 优点

- ✅ 更强大的 GPU（部分平台）
- ✅ 更长的运行时间
- ✅ 企业级环境

### 缺点

- ❌ 部分平台需要付费
- ❌ 配置相对复杂

---

## 🔧 硬件要求说明

| 你的环境 | 可学内容 | 限制说明 |
| ---- | ---- | ---- |
| **无 GPU (CPU only)** | • Chapter 0: 全部<br>• Chapter 1: 全部<br>• Chapter 2: 00-20 题<br>• Chapter 3: 无法学习 | 可完成约 70% 的内容，覆盖算法工程师核心考点<br>⚠️ Chapter 2 的 21-25 题（显存优化、分布式）无法测试实际效果 |
| **有 NVIDIA GPU** | • Chapter 0-3: 全部 | 100% 完整学习体验，额外覆盖：<br>• 显存优化实战（Ch2: 21-25）<br>• GPU 算子开发（Ch3: 全部） |

### GPU 需求详细说明

- **Chapter 0（前置知识）**：全部可在 CPU 运行，Profiling 工具在 GPU 上效果更好但非必需
- **Chapter 1（理论）**：纯理论讨论，无需 GPU
- **Chapter 2（算法实战）**：大部分可在 CPU 运行，但以下题目强烈建议使用 GPU：
  - 21 题（Gradient Checkpointing）：需要测量真实 CUDA 显存优化效果
  - 22-25 题（量化、分布式训练）：性能测试需要 GPU 环境
- **Chapter 3（CUDA/Triton）**：必须使用 NVIDIA GPU（Compute Capability 7.0+）

### 推荐配置

- **学习算法原理**：CPU 即可完成 70% 的内容
- **完整学习体验**：建议使用 NVIDIA GPU（GTX 1060+ 或云端 GPU）
- **GPU 环境配置**：查看 [Chapter 3 环境配置指南](./03_CUDA_and_Triton_Kernels/intro.md)

---

## 📝 Jupyter Notebook 使用技巧

### 基础操作

**执行单元格 (Cell)**
- **方式一：** 点击单元格，然后点击顶部工具栏的 ▶️ "Run" 按钮
- **方式二：** 选中单元格后按 `Shift + Enter`（执行并跳到下一个单元格）
- **方式三：** 按 `Ctrl + Enter`（执行但停留在当前单元格）

**执行所有单元格**
- 点击顶部菜单栏：`Run` → `Run All Cells`
- 或点击工具栏的 ⏩ "Run All" 按钮

### 推荐的刷题流程

1. 先执行第一个 cell（导入库）
2. 按顺序阅读每个 Part 的说明
3. 在 TODO 处填写你的代码
4. 执行测试 cell 验证答案

### 常见问题

**Q: 为什么会出现 `name 'xxx' is not defined` 错误？**
- **原因：** 没有按顺序执行前面的 cell，导致变量或函数未定义
- **解决：** 点击 `Run` → `Run All Cells` 重新执行整个 notebook

**Q: 如何重置 notebook 状态？**
- 点击 `Kernel` → `Restart Kernel and Clear All Outputs`
- 然后重新按顺序执行所有 cell

**Q: 修改代码后测试仍然失败？**
- 确保修改后重新执行了该 cell（单元格左侧会显示执行序号）
- 如果修改了函数定义，需要重新执行定义该函数的 cell

### 快捷键速查

| 操作 | 快捷键 |
|------|--------|
| 执行当前 cell 并跳到下一个 | `Shift + Enter` |
| 执行当前 cell 不跳转 | `Ctrl + Enter` |
| 在上方插入新 cell | `A` (命令模式) |
| 在下方插入新 cell | `B` (命令模式) |
| 删除当前 cell | `D + D` (命令模式) |
| 切换到命令模式 | `Esc` |
| 切换到编辑模式 | `Enter` |
| 保存 notebook | `Ctrl + S` |

---

## 🎯 选择建议总结

| 你的情况 | 推荐方式 | 原因 |
|---------|---------|------|
| 零基础 + 无本地环境 | **方式 1** | 零配置，免费 GPU |
| 有 Python 基础 + 无 GPU | **方式 2** | 本地开发体验更好 |
| 有本地 NVIDIA GPU | **方式 2** | 充分利用本地资源 |
| 需要批量测试/CI | **方式 3** | 脚本化测试 |
| 企业/研究用途 | **方式 4** | 更强算力 |

开始你的学习之旅吧！🚀
