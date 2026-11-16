# 回归测试

这里存放长期维护的工具与机制实现回归测试。面向学习者的 Notebook 验证命令暂时保留原入口，不因文件名以 `test_` 开头就一并迁入。

从仓库根目录运行已迁入的 `unittest` 用例；部分结构测试需要 PyYAML，训练语义测试需要 CPU PyTorch，缺少 PyTorch 时明确跳过：

```bash
python -m unittest discover -s tools/tests -p 'test_*.py'
```

下表列出常用检查；目录中的迁移测试还会核对新旧入口、模块导入和调用方行为。

| 测试 | 核对内容 | 依赖 |
|:---|:---|:---|
| `test_foundation_performance_evidence.py` | 基础页的资源计算、时间线依赖、比较条件与编译计划检查 | Python 标准库与仓库源 Notebook；不执行 GPU 或 Profiler 单元 |
| `test_foundation_hardware_constraints.py` | 设备声明、阶段独立需求、隔离指标、成本分母与功能测试前提 | Python 标准库；读取源 Notebook 中的实际定义，不探测或配置设备 |
| `test_foundation_training_semantics.py` | ReLU 零点梯度、非持久缓冲区、快照与续跑、早停、跳过更新、评价聚合和 Attention 边界 | CPU PyTorch；读取源 Notebook 定义，不安装依赖或执行 GPU 实验 |
| `test_gemm_epilogue_semantics.py` | GEMM分步参考的列广播、两处FP16舍入、激活顺序与CPU输入拒绝 | CPU PyTorch；读取Part03·05实际参考和包装器，CUDA内核另行验证 |
| `test_test_entrypoints.py` | 新旧脚本、检查类测试的模块入口、测试参数、跨目录运行、重复收集和断言失败退出码 | Python 标准库；子进程核对基础页与链接／SVG／公式测试的实际用例数，不修改源文件 |
| `test_chapter_link_checks.py` | 只检查 Notebook 正文链接、先生成跨 Part 镜像再检查 docs 的调用顺序 | Python 标准库；临时 Notebook 夹具与模拟调用，不执行转换或构建 |
| `test_check_svg_arrows.py` | SVG 路径端点、节点边界、连线标注与无效路径 | Python 标准库；内存 XML 夹具，不改图片 |
| `test_svg_layout.py` | SVG 画布尺寸、宽高比与布局角色 | Python 标准库；临时 SVG，不读取整库布局基线 |
| `test_math_formula_compatibility.py` | GitHub 公式风险、Notebook 正文定位与静态命令退出码 | Python 标准库；临时夹具，不启动 Node、网络或 GPU |
| `test_md_to_notebook.py` | Markdown 迁移排除 intro／组页、单文件与目录预览、已有 Notebook 保护 | `nbformat`；仅使用临时目录夹具，不转换教程文件 |
| `test_legacy_question_migration.py` | 旧题目修复器默认只读、写入拒绝与 CLI 失败退出码 | Python 标准库；仅使用临时夹具 |
| `test_result_migration_safety.py` | 两类历史结果转换的原始输入隔离、同名输出冲突、重复运行、预览与数值保真 | Python 标准库；仅转换临时 JSON，不改仓库结果 |
| `test_performance_analysis_docs.py` | 性能优化正文、方法归属、旧性能分析跳转、镜像及环境入口 | Python 标准库与仓库文档工具；不导入测量审查脚本或执行 Notebook 代码 |
| `test_training_baseline_contracts.py` | Part02·73的题目骨架、历史样本重算和close前校验 | Python 标准库与73源Notebook、结果JSON；仅执行CPU统计与登记测试 |
| `test_performance_result_readers.py` | 机制结果身份、样本统计、历史协议与Notebook读取接入 | `tools/performance_result_review.py`及对应机制页、结果JSON；不执行GPU测量 |
| `test_topic_route_identity.py` | 全库主学习路线、基础、横向与领域专题的身份分类 | Python 标准库与各专题intro；随共享专题导航变更维护 |
| `test_project_contracts.py` | 共享注册表、适配器成功／失败／阻断结果 | Python 标准库；不启动外部命令或模型 |
| `test_part02_project_structure.py` | 项目六层配置、运行阶段、决策范围及74历史清单来源哈希 | Python 标准库；只读源Notebook与已登记结果 |
| `test_quantization_project_contracts.py` | 量化产物声明、加载判断与Serving候选选择 | Python 标准库；内存夹具，不加载权重或启动服务 |
| `test_project_contract_entrypoints.py` | 契约测试的旧入口聚合、独立入口、发现去重与失败退出码 | Python 标准库；子进程与只读项目检查 |
| `test_part05_integrated_projects.py` | 五个综合项目的六层结构、安全默认配置、运行阶段及最终报告契约 | PyYAML；静态编译源单元，不执行Notebook |
| `test_project_runtime.py` | 已有实验结果与登记文件的覆盖保护 | `pytest`；仅临时目录夹具，需使用下方独立命令 |

仅检查性能文档整合时运行 `python tools/tests/test_performance_analysis_docs.py`。共享契约、Part02结构与量化判定可分别使用对应的 `python -m unittest tools.tests.<模块名>`。

原聚合入口 `python tools/test_project_contracts.py` 和 `python -m unittest tools.test_project_contracts` 均运行上述三个契约模块；目录发现由新位置收集，旧文件不重复导出测试类。Part05综合项目结构使用独立模块，已接入CPU检查流程。

`unittest` 不会收集 `test_project_runtime.py` 中的函数式 `pytest` 用例，需另外运行：

```bash
python -m pytest tools/tests/test_project_runtime.py -q
```

若本地无关的pytest插件阻止测试启动，可仅对本次命令设置 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，无需修改全局环境或安装该插件的依赖。

项目运行时的业务测试仍保留在 `tools/`，其目录迁移另行处理。检查这批已审核运行时时可运行：

```bash
python -m unittest \
  tools.test_inference_delivery_runtime \
  tools.test_minimind_post_training_runtime \
  tools.test_performance_delivery_runtime \
  tools.test_performance_delivery_benchmark \
  tools.test_part05_quantization_delivery_runtime \
  tools.test_part05_operator_delivery_runtime
```

这些业务测试需要PyYAML及仓库中的配置／负载文件，使用临时数据或模拟后端；通过不代表完成真实训练、量化、Serving或GPU性能验证。
