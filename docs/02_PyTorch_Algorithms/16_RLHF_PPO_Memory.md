# 16. RLHF PPO Memory | RLHF / PPO 资源与状态

**难度：** Hard | **环境：** [CPU 机制练习](#runtime-environment) | **标签：** `后训练对齐`, `RLHF`, `PPO` | **目标人群：** 训练机制学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/16_RLHF_PPO_Memory.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

完成 SFT 以后，模型已经能模仿数据里的回答，但还需要学习在多个可行答案中倾向于哪一种。RLHF/PPO 让当前策略先生成 response 或多步轨迹，再依据奖励信号调整生成倾向；Actor、Reference、Reward Model 和 Critic 因此同时参与同一轮训练。

这里先把 PPO 的更新信号拆成 `reward`、`value`、`advantage`、`ratio` 和 `clip`，再把它们放回一次 rollout 的状态账本。这样可以同时看清：奖励怎样转成更新方向，为什么当前策略、参考策略和反馈模型会带来额外驻留状态，以及 DPO / GRPO 为什么会选择更轻的反馈路径。

**关键词：** `PPO`, `reward`, `value`, `advantage`, `KL`, `ratio`, `clip`

---
<a id="runtime-environment"></a>

## 运行环境

- **基础练习：** 默认使用 [`BASE-CPU`](../guide.md#base-cpu)，运行 [Step 4 的题目与测试](#mechanism-practice)。

---
## 前置阅读

**导语：** 先补齐训练闭环和优化器基础，再看 RLHF / PPO 如何在 SFT 之后继续做偏好对齐会更顺。
- [P0: 11. PyTorch Optimizers and Loss | PyTorch 优化器与损失函数](../00_Prerequisites/11_PyTorch_Optimizers_and_Loss.md)
- [P0: 12. PyTorch Data Interface and Training Lifecycle | 数据接口与训练生命周期](../00_Prerequisites/12_PyTorch_Minimal_Training_Interface.md)
- [15. End-to-End Fine-Tuning Experiment | 端到端微调实验](../02_PyTorch_Algorithms/15_End_to_End_Fine_Tuning_Experiment.md)

---
## Step 1：PPO 一轮采样与更新需要哪些模型角色

PPO 先由策略模型生成回答或动作轨迹，奖励模型或验证器给出任务反馈，价值模型提供回报基线，参考模型限制策略长期偏移，最后只有策略模型接收策略梯度。这里说的是四类模型角色，不是四个奖励模型；具体系统可以分阶段加载、共享基座或卸载部分角色。采样时保存的行为策略对数概率是一份记录，也不是第五个完整模型。PPO 通常保留上述四类角色；GRPO 去掉显式价值模型并用组内基线；DPO 则以离线偏好对和冻结参考模型替代在线采样、奖励模型与价值模型。

| 角色或记录 | 产生什么 | 是否更新 | 主要驻留状态 |
| --- | --- | --- | --- |
| 策略模型（Actor） | 回答、当前策略对数概率 | 是 | 参数、激活、梯度与优化器状态 |
| 参考模型（Reference） | 固定参考对数概率 | 否 | 冻结参数与前向临时状态 |
| 奖励模型或验证器 | 回答质量或任务终态奖励 | 否 | 评分模型状态或验证器执行状态 |
| 价值模型（Critic） | 状态价值估计 | 通常是 | 参数、激活、梯度与优化器状态 |
| 行为策略记录 | 采样时的旧策略对数概率 | 否；不是独立模型 | 回答范围内的逐 Token 记录 |
| 有效回答范围 | 真正由模型生成的部分 | 不适用 | 回答掩码与终止状态 |

![PPO 训练与资源路径总览](../public/02_PyTorch_Algorithms/16_rlhf_ppo_overview_cn.svg)

## Step 2：奖励如何变成受约束的策略更新

任务奖励与参考模型的 KL 代价先形成受约束奖励，回报与价值估计再形成优势。当前策略与采样时的行为策略之比描述本轮更新幅度，裁剪目标限制单次更新过度偏离。裁剪约束短期步幅，参考模型约束长期漂移，两者解决的问题不同。

$$r_t^{\text{train}} = r_t^{\text{task}} - \beta_{KL}(\log \pi_{actor}-\log \pi_{ref}), \qquad A_t \approx R_t - V(s_t)$$

$$
\rho_t = \exp(\log \pi_{new}-\log \pi_{old}), \qquad
L_t^{clip}=\min\!\left(\rho_t A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\right).
$$

Critic 用固定回报目标拟合状态价值：

$$
\mathcal L_V=\frac{1}{2}\,\mathbb E_t\left[(V_\phi(s_t)-R_t)^2\right].
$$

把策略、价值、熵与参考策略约束放回同一轮更新，可写成：

$$
\mathcal L_{PPO}=-\mathbb E_t[L_t^{clip}]
+c_V\mathcal L_V-c_H\mathbb E_t[\mathcal H(\pi_\theta(\cdot\mid s_t))]
+\beta_{KL}\mathbb E_t[D_{KL}(\pi_\theta\Vert\pi_{ref})].
$$

具体实现可以把 KL 写入奖励，也可以作为独立损失项，但同一次实验必须固定口径，不能重复计入。

| 环节 | 输入 | 输出 | 主要检查 |
| --- | --- | --- | --- |
| 奖励约束 | 任务奖励、策略与参考模型的差异 | 受约束奖励 | KL 方向与奖励方向一致 |
| 回报与价值 | 奖励序列、价值估计 | 优势 | 延迟奖励仍归属于正确状态 |
| 策略比率 | 当前策略、行为策略 | 相对更新幅度 | 行为策略在本轮保持固定 |
| 裁剪目标 | 策略比率、优势 | 保守更新目标 | 正负优势都不能放大越界更新 |

## Step 3：哪些状态决定 PPO 的峰值与吞吐

PPO 的资源压力取决于哪些模型和轨迹状态同时驻留，而不是简单地把四个模型参数量相加。采样阶段保存回答与行为策略记录；评分阶段读取参考、奖励和价值模型；更新阶段还要保留策略模型的激活、梯度和优化器状态。分阶段加载或卸载可以降低峰值，但会增加模型搬运与等待时间。

| 阶段 | 必须驻留或保留的对象 | 主要代价 | 可释放条件 |
| --- | --- | --- | --- |
| 采样 | 策略模型、KV Cache、回答、行为策略记录 | 生成时间与序列状态 | 回答与采样依据持久化后 |
| 评分 | 回答、参考模型、奖励模型、价值模型 | 多次冻结前向与模型驻留 | 奖励、参考概率与价值估计完成后 |
| 优势估计 | 奖励、价值和有效回答范围 | 状态对齐与中间结果 | 优势固定后 |
| 策略更新 | 激活、梯度、优化器状态、行为策略记录、优势 | 训练峰值显存 | 参数更新完成后 |

![PPO 单步状态生命周期](../public/02_PyTorch_Algorithms/16_ppo_memory_ledger_cn.svg)

<a id="mechanism-practice"></a>

## Step 4：实现 PPO Actor Loss 的裁剪机制

题目区实现 token 级裁剪目标，并比较 `token_mean` 与 `sequence_mean` 两种聚合口径。前者让每个有效 token 权重相同，后者先对每条 response 求均值，使长短 response 在 batch 中权重相同。函数同时返回近似 KL 与裁剪比例，帮助判断更新是否过激。

| TODO / 函数 | 学习者完成的机制 | 必须满足的约束 | 测试证据 |
| --- | --- | --- | --- |
| TODO 1 / `compute_actor_loss`：策略比率 | 从固定旧策略与当前策略的 log-prob 差得到 ratio | 形状一致，梯度只回到当前策略 | ratio 与梯度隔离 |
| TODO 2 / `compute_actor_loss`：裁剪目标 | 比较原始与裁剪 surrogate | 保留 advantage 正负方向 | 上下界与保守目标 |
| TODO 3 / `compute_actor_loss`：聚合口径 | 实现 token mean 与 sequence mean | 仅聚合有效 response token | 变长 response 权重差异 |
| TODO 4 / `compute_actor_loss`：更新诊断 | 计算近似 KL 与裁剪比例 | 诊断值不进入梯度目标 | 数值与有效 mask |


```python
import torch
import torch.nn.functional as F

```


```python
def compute_actor_loss(
    log_probs_new, log_probs_old, advantages, response_mask=None, clip_range=0.2, reduction="token_mean"
):
    """
    计算 PPO 的 Actor Clip Loss。

    Args:
        log_probs_new: 当前 Actor 模型在采样的 Token 上的对数概率, shape [batch_size, seq_len]
        log_probs_old: 采样时(旧) Actor 模型在对应 Token 上的对数概率, shape [batch_size, seq_len]
        advantages: 优势函数 (Reward - Critic Value), shape [batch_size, seq_len]
        response_mask: 有效生成 token 的布尔掩码；None 表示全部有效
        clip_range: 截断范围 epsilon，默认 0.2
        reduction: token_mean 或 sequence_mean

    Returns:
        actor_loss: 标量
        metrics: 近似 KL、裁剪比例与有效 token 数
    """
    if log_probs_new.shape != log_probs_old.shape or log_probs_new.shape != advantages.shape:
        raise ValueError("log_probs_new / log_probs_old / advantages 的形状必须一致")
    if log_probs_new.ndim != 2:
        raise ValueError("log_probs 必须是 [batch, sequence] 二维张量")
    if clip_range < 0:
        raise ValueError("clip_range 必须非负")
    if response_mask is None:
        response_mask = torch.ones_like(log_probs_new, dtype=torch.bool)
    if response_mask.shape != log_probs_new.shape or not torch.all(response_mask.bool().any(dim=-1)):
        raise ValueError("response_mask 必须同形且至少包含一个有效位置")
    if reduction not in {"token_mean", "sequence_mean"}:
        raise ValueError("reduction 必须是 token_mean 或 sequence_mean")

    # ==========================================
    # TODO 1：冻结旧策略记录，从当前与旧策略的对数概率差构造 token 级 ratio。
    # ratio = ???
    # ==========================================

    # ==========================================
    # TODO 2：冻结 advantage，构造原始、裁剪和保守目标。
    # surrogate = ???
    # clipped_ratio = ???
    # clipped_surrogate = ???
    # conservative = ???
    # ==========================================

    # ==========================================
    # TODO 3：按 reduction 口径聚合有效 response token。
    mask = response_mask.bool()
    if reduction == "token_mean":
        # loss = ???
        pass
    else:
        # counts = ???
        # per_sequence = ???
        # loss = ???
        pass
    # ==========================================

    # ==========================================
    # TODO 4：在有效位置统计近似 KL、裁剪比例和 token 数。
    # log_ratio = ???
    # approx_kl = ???
    # clip_fraction = ???
    # valid_tokens = ???
    metrics = {
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
        "valid_tokens": valid_tokens,
    }
    # ==========================================

    return loss, metrics

```

## 测试

```python
# 测试目标：分别验证 ratio、clipping、变长回复聚合、诊断值、梯度隔离与输入契约。
# 每个断言对应 Step 4 的一个 TODO，避免将多个失败混为同一错误。

def _build_ppo_loss_case():
    """构造同时覆盖正负优势、掩码和梯度隔离的 PPO 样例。"""
    new = torch.tensor([[-2.0, -1.5], [-1.0, -0.5]], requires_grad=True)
    old = torch.tensor([[-2.1, -1.4], [-1.0, -0.6]], requires_grad=True)
    advantage = torch.tensor([[1.0, 0.5], [-1.0, -0.5]], requires_grad=True)
    mask = torch.tensor([[True, False], [True, True]])
    return new, old, advantage, mask

def _assert_ppo_ratio_and_loss(new, old, advantage, mask):
    """验证策略比率、裁剪目标和有效 token 聚合。"""
    loss, metrics = compute_actor_loss(new, old, advantage, mask, clip_range=0.2)
    ratio = torch.exp(new - old.detach())
    targets = torch.minimum(ratio * advantage.detach(), torch.clamp(ratio, 0.8, 1.2) * advantage.detach())
    expected = -targets[mask].mean()
    assert torch.allclose(loss, expected), "PPO Actor Loss 计算错误"
    assert metrics["valid_tokens"] == int(mask.sum()), "有效 token 计数错误"

def _assert_ppo_clipping():
    """验证正负优势分别命中 PPO 裁剪边界。"""
    high = compute_actor_loss(torch.tensor([[2.0]]).log(), torch.zeros(1, 1), torch.ones(1, 1))[0]
    low = compute_actor_loss(torch.tensor([[0.5]]).log(), torch.zeros(1, 1), -torch.ones(1, 1))[0]
    assert torch.allclose(high, torch.tensor(-1.2)), "正优势上界截断错误"
    assert torch.allclose(low, torch.tensor(0.8)), "负优势下界截断错误"

def _assert_ppo_reduction_and_metrics():
    """验证两种聚合口径及其诊断指标。"""
    new = torch.zeros(2, 2)
    old = torch.zeros(2, 2)
    advantage = torch.tensor([[1.0, 1.0], [3.0, 0.0]])
    mask = torch.tensor([[True, True], [True, False]])
    token_loss, metrics = compute_actor_loss(new, old, advantage, mask, reduction="token_mean")
    sequence_loss, _ = compute_actor_loss(new, old, advantage, mask, reduction="sequence_mean")
    assert torch.allclose(token_loss, torch.tensor(-5.0 / 3.0))
    assert torch.allclose(sequence_loss, torch.tensor(-2.0))
    assert metrics["approx_kl"] == 0.0 and metrics["clip_fraction"] == 0.0

def _assert_ppo_gradient(new, old, advantage, mask):
    """验证梯度只流向当前策略记录。"""
    compute_actor_loss(new, old, advantage, mask)[0].backward()
    assert new.grad is not None and torch.isfinite(new.grad).all(), "ratio 没有形成有效梯度"
    assert old.grad is None and advantage.grad is None, "旧策略与 advantage 不应接收梯度"

def _assert_ppo_input_contract():
    """验证不一致的 token 形状会被拒绝。"""
    try:
        compute_actor_loss(torch.zeros(1, 2), torch.zeros(1, 1), torch.zeros(1, 2))
    except ValueError:
        return
    raise AssertionError("形状不一致时应主动报错")

def test_ppo_actor_loss():
    """汇总验证 PPO Actor Loss 的六项机制责任。"""
    try:
        new, old, advantage, mask = _build_ppo_loss_case()
        _assert_ppo_ratio_and_loss(new, old, advantage, mask)
        _assert_ppo_clipping()
        _assert_ppo_reduction_and_metrics()
        _assert_ppo_gradient(new, old, advantage, mask)
        _assert_ppo_input_contract()
        print("✅ PPO ratio、clipping、变长回复聚合、诊断值、梯度隔离和输入契约测试通过。")
    except (AttributeError, NameError, TypeError, ValueError) as e:
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e

test_ppo_actor_loss()

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

## 参考代码与解析

### 代码

```python
def compute_actor_loss(
    log_probs_new, log_probs_old, advantages, response_mask=None, clip_range=0.2, reduction="token_mean"
):
    """
    计算 PPO 的 Actor Clip Loss。

    Args:
        log_probs_new: 当前 Actor 模型在采样的 Token 上的对数概率, shape [batch_size, seq_len]
        log_probs_old: 采样时(旧) Actor 模型在对应 Token 上的对数概率, shape [batch_size, seq_len]
        advantages: 优势函数 (Reward - Critic Value), shape [batch_size, seq_len]
        response_mask: 有效生成 token 的布尔掩码；None 表示全部有效
        clip_range: 截断范围 epsilon，默认 0.2
        reduction: token_mean 或 sequence_mean

    Returns:
        actor_loss: 标量
        metrics: 近似 KL、裁剪比例与有效 token 数
    """
    if log_probs_new.shape != log_probs_old.shape or log_probs_new.shape != advantages.shape:
        raise ValueError("log_probs_new / log_probs_old / advantages 的形状必须一致")
    if log_probs_new.ndim != 2:
        raise ValueError("log_probs 必须是 [batch, sequence] 二维张量")
    if clip_range < 0:
        raise ValueError("clip_range 必须非负")
    if response_mask is None:
        response_mask = torch.ones_like(log_probs_new, dtype=torch.bool)
    if response_mask.shape != log_probs_new.shape or not torch.all(response_mask.bool().any(dim=-1)):
        raise ValueError("response_mask 必须同形且至少包含一个有效位置")
    if reduction not in {"token_mean", "sequence_mean"}:
        raise ValueError("reduction 必须是 token_mean 或 sequence_mean")

    # ==========================================
    # TODO 1：当前 / 旧策略在同一 token 位置上的概率比率。
    ratio = torch.exp(log_probs_new - log_probs_old.detach())
    # ==========================================

    # ==========================================
    # TODO 2：原始与裁剪目标共同保留 advantage 指示的更新方向。
    surrogate = ratio * advantages.detach()
    clipped_surrogate = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages.detach()
    conservative = torch.minimum(surrogate, clipped_surrogate)
    # ==========================================

    # ==========================================
    # TODO 3：选择 token 等权或 response 等权的聚合口径。
    mask = response_mask.bool()
    if reduction == "token_mean":
        loss = -conservative[mask].mean()
    else:
        counts = mask.sum(dim=-1)
        per_sequence = (conservative * mask).sum(dim=-1) / counts
        loss = -per_sequence.mean()
    # ==========================================

    # ==========================================
    # TODO 4：诊断当前更新相对旧策略的偏移与裁剪覆盖率。
    log_ratio = log_probs_new.detach() - log_probs_old.detach()
    approx_kl = ((ratio.detach() - 1.0) - log_ratio)[mask].mean()
    clip_fraction = ((ratio.detach() - 1.0).abs() > clip_range)[mask].float().mean()
    metrics = {
        "approx_kl": float(approx_kl),
        "clip_fraction": float(clip_fraction),
        "valid_tokens": int(mask.sum()),
    }
    # ==========================================

    return loss, metrics

```

### 解析

#### TODO 1：计算策略比率

- 使用 `torch.exp(log_probs_new - log_probs_old.detach())` 计算逐 token ratio。
- `detach()` 固定采样策略，测试确认梯度只回到当前策略。

#### TODO 2：构造裁剪目标

- 分别计算 `ratio * advantage` 与 `clipped_ratio * advantage`。
- 使用 `torch.minimum` 保留更保守的逐 token 目标，测试覆盖正负 advantage 的两个裁剪方向。

#### TODO 3：固定聚合口径

- `token_mean` 直接对 mask 内目标取均值；`sequence_mean` 先按每条 response 的有效 token 数归一化。
- 变长回复测试确认两种口径不会被误写成同一个结果。

#### TODO 4：记录更新诊断

- 用 `(ratio - 1) - log_ratio` 的有效位置均值估计 KL，用越过 `clip_range` 的位置比例计算裁剪率。
- 诊断从分离后的张量生成，测试同时核对有效 token 数与零偏移结果。

## 相关阅读

- [17. DPO Loss Tutorial | 直接偏好优化损失教程](../02_PyTorch_Algorithms/17_DPO_Loss_Tutorial.md)
- [18. GRPO Loss Tutorial | 群体相对策略优化损失教程](../02_PyTorch_Algorithms/18_GRPO_Loss_Tutorial.md)
- [PPO 原论文](https://arxiv.org/abs/1707.06347)
- [TRL PPOTrainer 文档](https://huggingface.co/docs/trl/ppo_trainer)
