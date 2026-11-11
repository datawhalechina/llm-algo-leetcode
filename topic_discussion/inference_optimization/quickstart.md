# 推理优化实验 Quickstart

以下命令在仓库根目录执行。本页负责创建、恢复和验证环境；模型准备、逐实验服务参数和结果状态见[实验资产表](./experiment_assets.md)，跨专题隔离关系见[环境资产索引](../../docs/gpu_environment_assets.md)。

## 选择环境

| Profile | 环境名 | 适用实验与最低条件 |
| --- | --- | --- |
| [BASE-CPU](../../docs/guide.md#base-cpu) | `llm_algo_cpu` | 请求、路由、缓存、预算和调度的 CPU 题目；安装复用使用指南 |
| [INF-CUDA-MECHANISM](#inf-cuda-mechanism-quickstart) | `llm_inference_mechanism_cu128` | 单卡计算、MoE 专家层、生成预算与执行重叠；需要匹配 CUDA 的 GPU |
| [INF-VLLM-011-CU128](#inf-vllm-quickstart) | `llm_vllm_cu128` | 真实服务机制、Part02 传统服务基线及 Part05 服务项目 |
| [INF-VLLM-SPEC025](#inf-vllm-spec025-quickstart) | `llm_vllm_spec025_cu129` | Part02·68 投机候选选型；先确认配对资产和 CUDA 12.9 驱动门禁 |
| [INF-VLLM-MTP](#inf-vllm-mtp-quickstart) | `llm_vllm_mtp_cu129` | Part02·35 的独立原生 MTP 对照；不覆盖选型或历史服务环境 |
| SGLang／通信与并行环境 | 由对应专题登记 | 可选跨后端或多卡分支，见[资产表](./experiment_assets.md#环境与实验入口)；本页不提供混装命令 |

选择一个直接匹配实验的环境即可。服务端与客户端可以使用同一服务环境；是否由 Notebook 自动管理服务或需要外部终端，按[具体实验入口](./experiment_assets.md#外部服务准备)区分。

教学引擎从 [nano-vLLM 请求跟踪](./nano_vllm_request_walkthrough.md#request-entry)开始，到缓存阶段加入 [Mini-SGLang 对照](./mini_sglang_cache_scheduling.md#prefix-cache)。两套独立环境尚未登记和验收，安装入口在实验前准备阶段补齐，准备状态见[资产表](./experiment_assets.md#teaching-engine)。Mini-SGLang 的低精度分支先确认代码身份与依赖，不直接混入上述机制或正式服务环境。

## 首次创建

安装器按登记依赖完成安装、仓库运行时注册、依赖检查、Profile 预检与 Kernel 注册。第一次运行使用新环境名；只看计划时，在对应命令末尾添加 `--dry-run`。运行时注册写入环境的导入路径，Notebook 重启后才使用新状态。

如需增加 pip 下载超时和重试，可在命令前添加 `PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=8`；下载源可用 `PIP_INDEX_URL` 临时指定，下次省略即可恢复默认行为。这些设置不保证下载速度，也不覆盖 requirements 中固定的 PyTorch 或后端 wheel 来源。

<a id="inf-cuda-mechanism-quickstart"></a>

### INF-CUDA-MECHANISM：单卡推理机制

使用 Conda 创建独立 PyTorch／Transformers 环境，提供局部计算和运行时实验所需依赖。MoE 专家层无需下载模型；其他模型与输入准备按资产表对应条目进行。该环境与 vLLM、SGLang 服务栈隔离。

```bash
python tools/install_environment_profile.py --profile INF-CUDA-MECHANISM --env-name llm_inference_mechanism_cu128
```

<a id="inf-vllm-quickstart"></a>

### INF-VLLM-011-CU128：服务基线

使用 Conda 创建服务环境，由该 Profile 的依赖提供匹配的 vLLM 与 PyTorch 栈。Part05 服务项目也使用本环境；Notebook 与服务进程无需另建两套环境。SGLang 对照另行隔离。

```bash
PIP_INDEX_URL=https://pypi.org/simple python tools/install_environment_profile.py --profile INF-VLLM-011-CU128 --env-name llm_vllm_cu128
```

<a id="inf-vllm-spec025-quickstart"></a>

### INF-VLLM-SPEC025：投机候选选型

使用 Conda 创建 Part02·68 的独立选型环境，安装仓库登记的 CUDA 12.9 后端 wheel 与约束。沿用 CUDA 12.9 Update 1 的 Linux R575.57 或更高驱动门禁；先通过 target-only 基线，再按模型资产、显存和方法支持条件开放候选。具体依赖身份以 Profile 与约束文件为准。

```bash
PIP_INDEX_URL=https://pypi.org/simple python tools/install_environment_profile.py --profile INF-VLLM-SPEC025 --env-name llm_vllm_spec025_cu129
```

<a id="inf-vllm-mtp-quickstart"></a>

### INF-VLLM-MTP：原生 MTP 对照

使用 Conda 创建 Part02·35 的独立环境，安装登记的 CUDA 12.9 专用 wheel，沿用上面的驱动门禁。它与 SPEC025 的用途和依赖组合不同；PyPI 默认构建或其他 CUDA 环境不能代替本 Profile。模型配置、历史失败与复测顺序见[原生 MTP 条目](./experiment_assets.md#mtp-experiment)。

```bash
PIP_INDEX_URL=https://pypi.org/simple python tools/install_environment_profile.py --profile INF-VLLM-MTP --env-name llm_vllm_mtp_cu129
```

## 中断续装与重新进入

环境尚未创建时，重新运行对应创建命令。已创建但依赖下载中断时，先激活选择表中的环境，按日志定位对应 requirements；保持原有安装顺序、约束与缓存。

| Profile | 依赖续装顺序 |
| --- | --- |
| INF-CUDA-MECHANISM | 先 `python -m pip install -r requirements/torch-cu128.txt`，再 `python -m pip install -r requirements/inference-mechanism.txt` |
| INF-VLLM-011-CU128 | `python -m pip install -r requirements/inference-vllm.txt` |
| INF-VLLM-SPEC025 | `python -m pip install -r requirements/inference-vllm-spec025.txt` |
| INF-VLLM-MTP | `python -m pip install -r requirements/inference-vllm-mtp.txt` |

依赖续装后，给原安装器命令添加 `--reuse`，补齐运行时、预检和 Kernel。下面以机制环境为例；服务分支同时替换 Profile 与环境名。只对确认属于同一 Profile 的环境使用 `--reuse`，身份不明或已有冲突时使用新环境名。

```bash
python tools/install_environment_profile.py --profile INF-CUDA-MECHANISM --env-name llm_inference_mechanism_cu128 --reuse
```

安装已完成时，直接激活并检查，不必重装。下面仍以机制环境为例，其他环境使用选择表中的对应名称和 Profile：

```bash
conda activate llm_inference_mechanism_cu128
python tools/register_notebook_runtime.py --check
python -m pip check
python tools/environment_preflight.py --profile INF-CUDA-MECHANISM
python -m jupyter lab
```

## 手动创建（可选）

<details>
<summary>安装器不可用时先查看等价安装计划</summary>

本专题的四个 Profile 都由安装器按注册表和约束生成顺序。尚未单独验收另一套手动安装链，因此先检查计划并修复安装器入口，避免临时拼接不同后端的依赖。命令按“基础环境 → requirements → 运行时注册 → 依赖检查 → Profile 预检 → Kernel”核对：

```bash
python tools/install_environment_profile.py --profile INF-CUDA-MECHANISM --env-name llm_inference_mechanism_cu128 --dry-run
```

</details>

<a id="常见异常"></a>

## 验证与常见问题

完成门槛是：解释器属于所选环境，运行时 `registered=true`，`pip check` 无冲突，Profile 预检 `ready=true`。Notebook 选择对应 Kernel 后再运行实验；环境通过不代表实验成功。真实服务还要按资产表确认健康检查与配置。

```bash
python -c "import sys; print(sys.executable)"
python tools/register_notebook_runtime.py --check
python -m pip check
python tools/environment_preflight.py --profile INF-CUDA-MECHANISM
```

| 现象 | 处理方式 |
| --- | --- |
| Conda 提示覆盖 PYTHONPATH 等变量 | 属于 Profile 隔离设置；检查预检是否通过，不恢复混入的外部 site-packages |
| 下载中断或速度过低 | 停止当前安装，保留缓存，按同 Profile 续装；只运行一个安装进程 |
| `pip check` 冲突或预检不通过 | 保留失败输出，核对 Profile、约束、解释器与隔离状态；不逐包覆盖另一后端 |
| Notebook 无法导入 `tools` | 在仓库根目录运行 `python tools/register_notebook_runtime.py`，重启 Kernel |
| CUDA wheel 或版本不匹配 | 以已登记的 Profile 和约束为准，停止真实实验；不把另一 CUDA 构建当作替代 |
| 模型或数据访问受限 | 按资产条目检查缓存、认证与网络；令牌不写入 Notebook，也不清空已有缓存 |
| 后端启动但没有成对结果 | 健康检查只证明可用性；固定输入与配置、完成基线和候选后再判断性能 |
