# 编译与图优化专题

## 专题概览
本专题用于沉淀图优化、融合、lowering、调度和代码生成视角，回答“怎么把高层结构系统化变成更高效的执行”。

## 职责边界

这个专题只负责图级、编译级和后端执行链路里的优化视角，不负责推理策略本身，也不负责多卡通信主线。

- `Graph Optimization` 关注图结构、依赖关系和成本向量。
- `Fusion` 关注算子合并、中间张量消除和布局约束。
- `Lowering / Scheduling` 关注从高层表示到可执行形式的逐层收敛。
- `Codegen / Backend` 关注不同后端上的约束差异和落地方式。
- `Compiler Vision` 关注“为什么同一张图在不同 backend 上会得到不同最优解”。

## 对应来源

| 来源 | 适合纳入的内容 |
|:---|:---|
| `Part 1E` | AI 编译器、图优化、芯片选型和成本决策 |
| `Part 1D` | 执行模型、CUDA/Triton 编程模型、fusion 与调度衔接 |
| `Part 2.2` | 模型结构里的实现视角，理解结构如何影响执行 |
| `Part 2.6 / 2.7A` | 推理链路里和图优化、调度、cache 相关的后端视角 |

## 章节跳转

| 章节 | 你会看到什么 | 跳转 |
|:---|:---|:---|
| `1E-09` | AI 编译器、图优化和 backend 约束的主入口 | [09 AI Compilers and Graph Optimization](../../01_Hardware_Math_and_Systems/09_AI_Compilers_and_Graph_Optimization.ipynb) |
| `1E-19` | 算子融合为什么能减少中间结果开销 | [19 Operator Fusion Introduction](../../01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.ipynb) |
| `1E-32` | TVM / MLIR 的 lowering、schedule 和 codegen 链路 | [32 TVM MLIR Deep Practice](../../01_Hardware_Math_and_Systems/32_TVM_MLIR_Deep_Practice.ipynb) |
| `1E-33` | TCO 和成本模型，理解“为什么要优化” | [33 TCO and Cost Model](../../01_Hardware_Math_and_Systems/33_TCO_and_Cost_Model.ipynb) |
| `1D-08` | CUDA / Triton 编程模型，理解 kernel 组织方式 | [08 Programming Models CUDA Triton](../../01_Hardware_Math_and_Systems/08_Programming_Models_CUDA_Triton.ipynb) |
| `1D-15` | CUDA 执行模型，理解 block / warp / device 的执行层级 | [15 CUDA Execution Model](../../01_Hardware_Math_and_Systems/15_CUDA_Execution_Model.ipynb) |
| `1D-18` | Triton block model，理解程序块到执行块的映射 | [18 Triton Block Model](../../01_Hardware_Math_and_Systems/18_Triton_Block_Model.ipynb) |
| `1D-29` | Stream 高级调度，理解调度和执行之间的关系 | [29 CUDA Stream Advanced Scheduling](../../01_Hardware_Math_and_Systems/29_CUDA_Stream_Advanced_Scheduling.ipynb) |

## 推荐入口

- 先看 `1E-09`，把图优化、fusion 和 backend 约束先立住。
- 再看 `1E-19 -> 1E-32 -> 1E-33`，把融合、lowering 和成本模型补齐。
- 最后看 `1D-08 -> 1D-15 -> 1D-18 -> 1D-29`，把编程模型、执行模型和调度衔接起来。

## 入口摘要

- 第一入口：`Part 1E` + `1E-09 -> 1E-19`，先把图优化、融合和成本向量立住。
- 第二入口：`1E-32 -> 1E-33 -> 1D-08 -> 1D-18`，把 lowering、codegen 和执行模型串起来。
- 验证入口：`2.2 -> 2.6 -> 2.7A -> 2.9`，把后端视角放回模型结构、推理链路和项目结果里验证。

## 正文页

- [编译与图优化正文](./casebook.md)：按“图优化 / 融合 / lowering / 调度 / codegen”展开正文，适合做更细的后端视角案例。
- [编译与图优化深入阅读](./walkthrough.md)：按完整后端链路展开，适合想看连续推演的人。

## 相关专题

- [Profiling 专题](../profiling/intro.md)：当你需要先看哪里贵、哪里慢、哪里不稳定时先看这里。
- [推理优化专题](../inference_optimization/intro.md)：当 backend 差异直接影响推理路径和 cache 行为时先看这里。
- [通信与并行专题](../communication_parallel/intro.md)：当执行模型和通信调度、并行切分一起分析时先看这里。

## Part 1 / Part 2 入口顺序

### Part 1 入口

- 先看 `Part 1E`，把 AI 编译器、图优化、芯片选型和成本决策先立住。
- 再看 `1E-09 -> 1E-19 -> 1E-32 -> 1E-33`，把图优化、融合、lowering 和成本模型串起来。
- 然后看 `1D-08 -> 1D-15 -> 1D-18 -> 1D-29`，把编程模型、执行模型和调度接起来。

### Part 2 入口

- 先看 `2.2`，从模型结构层理解执行路径为什么会变。
- 再看 `2.6 -> 2.7A`，把推理链路里和图优化、调度、cache 相关的后端视角补齐。
- 如果想把后端视角回到项目验证里，再看 `2.9` 的性能结果和工程闭环。

## 读法建议

- 如果你关心“图优化先改什么”，先看 `09 -> 19`。
- 如果你关心“lowering 为什么不是翻译”，先看 `32`。
- 如果你关心“同一张图为什么在不同 backend 上结果不同”，再看 `09 -> 32 -> 33`。
- 如果你想把编译视角和 kernel 视角接起来，先看 `08 -> 15 -> 18 -> 29`。

## 建设方式

- 入口页只负责告诉读者从哪进、怎么选路径、怎么回到 Part。
- 具体的图级判断、执行级推演和 backend 差异都放到正文页展开。
- 后续新增内容优先沿着 `09 / 19 / 32 / 33 / 08 / 15 / 18 / 29` 回收。

## 专题状态
当前为专题占位页，后续将逐步补充跨 Part 索引、图优化案例和编译视角拆解。
