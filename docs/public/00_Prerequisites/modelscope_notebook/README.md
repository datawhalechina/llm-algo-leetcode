# ModelScope Notebook 教程配图来源

## official-binding-prompt.png

- 内容：ModelScope 模型详情页中的阿里云账号绑定提示。
- 来源文档：[ModelScope《Notebook 实例创建与管理》](https://modelscope.cn/docs/notebook/instance-management)。
- 原始图片：[官方文档资源文件 image10.png](https://resouces.modelscope.cn/document/docdata/2026-11-10_17-29-CN/dist/notebook/instance-management/_resources/image10.png)。
- 获取日期：2026-11-16。
- 处理方式：原图保存，未裁剪或修改；图内原有遮挡由来源文档提供。
- 使用位置：ModelScope Notebook GPU 环境搭建指南的“注册登录”部分，作为官方旧版界面的绑定提示示例。

## 其余操作配图

下表图片均于 2026-11-16 获取。官方图片按原图保存，未添加或修改标注；其中的界面版本、文件名、内核与资源类型仅作示意，教程图注已注明相关差异。

| 文件 | 对应操作 | 来源 |
| --- | --- | --- |
| `current-instance-list.png` | 图 2：免费 GPU 规格、镜像与启动入口 | [我的 Notebook](https://modelscope.cn/my/mynotebook)现场截图，仅截取资源表格区域，不包含账号姓名；图中剩余时长为该账号当时的余额 |
| `current-webide-jupyter-entry.png` | 图 3：VS Code 风格 IDE 右上角的 Jupyter Notebook 入口 | [代码工作区](https://www.modelscope.cn/code/workspace)现场截图，来自用户已经运行的 CPU 实例；仅截取右上角菜单及编辑区，不含地址栏和文件列表，未修改界面内容 |
| `official-view-instance.png` | 图 3a：旧版查看实例、停止实例入口 | [新版编辑器](https://modelscope.cn/docs/notebook/new-editor)；[原图](https://resouces.modelscope.cn/document/docdata/2026-11-10_17-29-CN/dist/notebook/new-editor/_resources/view-instance.png) |
| `current-jupyterlab-launcher.png` | 图 3b：登录后的 JupyterLab 启动器 | 从[代码工作区](https://www.modelscope.cn/code/workspace)的“Jupyter Notebook”入口进入，用户完成阿里云登录后现场截图；仅截取启动器中的 Notebook、Console 与 Other 区域，不含地址栏、实例标识和私人文件路径 |
| `official-jupyter-layout.png` | 图 4：JupyterLab 界面区域 | [实例管理](https://modelscope.cn/docs/notebook/instance-management)；[原图 image12.png](https://resouces.modelscope.cn/document/docdata/2026-11-10_17-29-CN/dist/notebook/instance-management/_resources/image12.png) |
| `official-open-terminal.png` | 图 5：Terminal 入口 | [实例管理](https://modelscope.cn/docs/notebook/instance-management)；[原图 image14.png](https://resouces.modelscope.cn/document/docdata/2026-11-10_17-29-CN/dist/notebook/instance-management/_resources/image14.png) |
| `official-upload.png` | 图 6：上传文件 | [数据存储](https://modelscope.cn/docs/notebook/data-storage)；[原图](https://resouces.modelscope.cn/document/docdata/2026-11-10_17-29-CN/dist/notebook/data-storage/_resources/1659693464416-107a7f18-1a68-4f82-8b7d-614e2a2da9db.png) |
| `official-new-notebook.png` | 备用配图：新建 Notebook；当前正文直接打开课程文件，未使用此图 | [实例管理](https://modelscope.cn/docs/notebook/instance-management)；[原图 image13.png](https://resouces.modelscope.cn/document/docdata/2026-11-10_17-29-CN/dist/notebook/instance-management/_resources/image13.png) |
| `current-clean-gpu-check.png` | 留存证据：切换官方 GPU 镜像后的 torch 与 CUDA 实测结果；当前正文仅链接验证记录 | 2026-11-16 在免费 GPU 实例的 JupyterLab 现场截取验证 Notebook 的输出区域，不含地址栏、账号、实例标识或私人文件列表；未改写或合成输出 |
| `official-download.png` | 图 7：下载 Notebook 文件 | [数据存储](https://modelscope.cn/docs/notebook/data-storage)；[原图](https://resouces.modelscope.cn/document/docdata/2026-11-10_17-29-CN/dist/notebook/data-storage/_resources/1659693570932-ea5b12ca-a3d7-4614-aac1-223a99613112.png) |

实例列表截图采集时 CPU 与 GPU 均已停止。补拍 IDE 菜单和启动器时，只打开了用户已经运行的 CPU 实例；该次截图采集没有启动 GPU，也没有运行训练。已确认点击“Jupyter Notebook”、完成阿里云登录后能够进入 JupyterLab，并打开 Launcher 查看 Python Notebook 与 Terminal 入口。图片本身不作为本教程训练、备份或恢复成功的证明。

同日先完成了现有镜像 `torch 2.13.0+cu130` 的可用性检查，随后按用户要求补做干净镜像验证：切换到官方 `ubuntu22.04-cuda12.1.0-py312-torch2.3.1-tf2.16.1-1.40.1`，启动后确认运行实例标识变化，保留 NFS 工作目录中的原文件。默认内核实际为 Python `3.11.11`、`torch 2.3.1+cu121`，直接导入并通过 CUDA 矩阵乘法及一次前向、反向、SGD 参数更新；空临时目录中的 `python -I -S` 独立进程仅加入系统包路径，同样通过 CUDA 检查。全程未安装或升级依赖。

`current-clean-gpu-check.png` 来自第二次验证的真实输出，其他图片仍来自表中各自注明的采集环境。第二次验证 Notebook 为云端工作目录中的 `torch_clean_image_check_20261116.ipynb`，已下载留存；不含账号或实例标识的结果保存在本目录的 `clean-gpu-verification-20261116.json`。文件核对确认原项目、数据、压缩包和 `.venv` 无差异，编辑器配置、缓存、旧日志和本次临时文档的变化另行记录。检查结束后已确认 GPU、CPU 均停止。此检查不代表完整课程训练、备份和恢复流程全部通过。

当前正文的实战入口改为仓库现有的 `00_Prerequisites/12_PyTorch_Minimal_Training_Interface.ipynb`。2026-11-16 在本地 Python `3.12.10`、PyTorch `2.14.1` 的 CPU 环境中，按顺序执行其全部 14 个代码单元，7 组验证均通过，未修改 Notebook 源文件。这是本地代码执行结果，不是该课程 Notebook 已在魔搭默认内核中完整运行的记录。

图片权利归原权利人所有。本目录的来源说明不表示图片采用本仓库的代码许可证。
