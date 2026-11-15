# X01 · LLM 直出决策模型打 2D 格斗（电子斗蛐蛐）

> 项目类型：开放实践　运行环境：macOS MPS / CUDA 24GB（训练）/ llama.cpp（GGUF 推理）

## 项目问题

给 LLM 一个 ~300 token 的游戏状态文本和 8 个固定动作，让它**不经推理链、单次前向直接输出每个动作的概率**——不写作文，只做选择。本项目回答一个可检验的问题：

> 4B 指令模型在这种"直出决策"负载下，零样本提示（SemIf）与 QLoRA 微调（kev）的胜率差距有多大？微调后的模型能否迁移到不同尺寸的场地？

覆盖：2D 格斗环境、四类大脑（随机 / 规则专家 / SemIf 零训练 / kev 微调）、评估与录像、数据采集与训练复现。
不覆盖：多模态视觉输入、长思维链决策、强化学习训练。

## 实验设计

| 项目 | 设定 |
|:---|:---|
| 基线与候选 | 基线：random、规则专家、SemIf（零样本提示）；候选：kev（QLoRA NF4 微调 Qwen3.5-4B，step2000） |
| 工作负载与数据 | 决策负载：~300 token 状态 → 8 动作概率，5Hz；训练数据由第 7 节规则专家自采集（36k 样本 jsonl），seed 可复现 |
| 运行环境 | 推理：llama.cpp Metal（GGUF Q4）或 transformers（bf16，MPS/CUDA/CPU 自适应）；训练：24GB 显存（A10/4090D），QLoRA NF4 |
| 质量/正确性门槛 | kev 评估胜率：vs random ≥ 90%、vs 规则 dvd ≥ 90%（演示默认各 3 局）；实测值：30 局 100% / 96.7% |
| 资源预算 | 训练约 6.6h（24GB 卡，bs8+ckpt）；推理 GGUF Q4 约 3GB 内存；笔记本 CPU 亦可跑通演示 |

## 运行入口

```
X01_llm_direct_decision/
├── index.md                    # 本文件
├── project_config.example.yaml # 唯一配置入口
└── Project.ipynb               # 预检 → 环境/大脑 → 评估 → 录像 → 采集 → 训练 → 结论
```

运行顺序（详见 `Project.ipynb` 第 0 节预检）：

```bash
# 1. 依赖（核心路径只要这几个）
pip install pyyaml numpy pygame imageio imageio-ffmpeg
# 可选：SemIf 零训练大脑
pip install llama-cpp-python
# 可选：kev 微调大脑 / 训练
pip install torch transformers peft accelerate

# 2. 复制并修改配置
cp project_config.example.yaml project_config.yaml
# 编辑 project_config.yaml：填模型路径（不填的章节自动跳过，不影响 Run All）

# 3. JupyterLab 打开 Project.ipynb，Run All
```

模型权重**不在仓库内**，按需下载：

```bash
# kev / 训练用 bf16 基座（8.8GB，二选一）
modelscope download --model Qwen/Qwen3.5-4B --local_dir /path/qwen35-4b-bf16
# 或：HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen3.5-4B --local-dir /path/qwen35-4b-bf16

# SemIf 用 GGUF（2.6GB）
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
  SandLogicTechnologies/Qwen3.5-4B-GGUF Qwen3.5-4B_Q4_k_m.gguf --local-dir /path/models

# kev 成品 adapter（81MB）：见项目开源包，或用第 8 节自行训练
```

## 结果与判断

原始结果与账本登记在 `benchmarks/results/X01_llm_direct_decision/`（配置 `outputs.directory`），目录结构：

```
<outputs.directory>/
  replays/   录像 jsonl + 导出的 mp4 + 关键帧 png
  models/    训练产出的 adapter 与 ledger.json
  dataset/   数据采集产物
```

关键实测结果（判断：**accept**，适用边界见下）：

| 指标 | 结果 |
|:---|:---|
| kev 胜率 vs random | 100%（30/30，平均 19.7s 一局） |
| kev 胜率 vs 规则 dvd | 96.7%（29 胜 1 负，11 次真击杀）；基线 SemIf 为 0 胜 10 负 |
| 场地迁移（零样本） | 480×270 / 600×600 无损；1920×1080 双臂同退化至 40%（瓶颈为长距追击，非 prompt 失配） |
| 训练资源（E1 账本） | 4B QLoRA：bs8 相对 bs4 提速 ×2.95 的"免费午餐"；g8 bs8 关 checkpoint 24GB OOM |
| 9B 阶梯（E2） | bf16 / bf16+ckpt 24GB 均 OOM；NF4 可训（13.97GB） |
| 推理优化（E3/E4） | vLLM prefix cache 仅并发场景有效（>页 1.61x，单流 0.97x）；GGUF Q4≈Q5，bf16 慢 5× |

边界：kev 仅在 960×540 分布内训练，超大场地长距追击退化；演示章节默认 3 局仅为冒烟，正式验收需 30 局。失败与未完成实验（如温度校准、AWQ/GPTQ 混合架构线）保留状态，未写成实测值。

## 注意事项

- Notebook 全程 headless（SDL_VIDEODRIVER=dummy），想看实时对战请用 `--live` 模式；
- 第 5/6/7/8 节是**可选章节**：对应依赖或路径缺失时打印"跳过"并继续，不会中断 Run All；
- 训练节默认注释掉（需要 GPU 与数据），取消注释即可；
- 完整实验矩阵、踩坑记录与技术分享材料见项目原始仓库的 `DESIGN.md` / `HANDOFF.md` / `docs/`。
