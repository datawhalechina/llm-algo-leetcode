# PT-REWARD-SIGNAL. Reward Model Value and Advantage | 奖励、价值与优势信号

**难度：** Hard | **环境：** [CPU 主线，GPU 可选](#runtime-environment) | **标签：** `后训练`, `奖励信号`, `优势估计` | **目标人群：** 奖励优化机制学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/PT-REWARD-SIGNAL_Reward_Model_Value_and_Advantage.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

后训练中的任务反馈、奖励、回报、价值与优势都可能表现为数值，但它们处在不同阶段。任务反馈与采样对数概率比约束先组成动作位置奖励，奖励再反向累计为回报；PPO 用逐词元价值基线形成优势，GRPO 用同提示候选组形成回答级优势。

输入包括任务反馈、当前策略与参考策略在已采样词元上的对数概率，以及动作掩码；输出包括动作位置奖励、折扣回报、PPO 逐词元优势和 GRPO 组内优势。

**关键词：** 任务奖励、采样对数概率比约束、回报（Return）、价值（Value）、优势（Advantage）

---
<a id="runtime-environment"></a>

## 运行环境

- **基础练习：** 默认使用 [`BASE-CPU`](../guide.md#base-cpu)，运行 [Step 4 的题目与测试](#mechanism-practice)。
- **框架核对：** 如需核对 TRL 的已采样 Token 对数概率和 GRPO 组内缩放约定，使用 [`PT-CPU`](../topic_discussion/post_training_optimization/quickstart.md#pt-cpu-quickstart)；完成 Step 4 后运行 [框架一致性核对](#framework-check)。
- **GPU 实测：** 如需验证真实模型上的策略偏移与奖励信号矩阵，阅读 [Step 5](#gpu-experiment)；实验环境与固定工作负载见其中 5.1。

---
## 前置阅读

**导语：** 先区分语言模型损失、策略对数概率与任务反馈；使用学习型奖励时，再确认奖励模型已经通过留出校准。

- [Part 02 · 13 SFT 训练循环](./13_SFT_Training_Loop.md)：理解 Token 级预测目标与有效监督位置。
- [Part 02 · 50 偏好数据与对齐评测](./50_Preference_Data_and_Evaluation.md)：理解任务反馈与独立评测的边界。
- 学习型奖励分支：[Part 02 · PT-REWARD-MODEL 奖励模型训练与校准](./PT-REWARD-MODEL_Reward_Model_Training_and_Calibration.md)。

---
## Step 1：任务反馈和参考策略约束如何组成 Token 奖励

策略模型先生成回答或轨迹，环境、规则验证器或奖励模型对结果给出任务反馈。冻结参考策略不评价任务质量，而是为同一批已采样 Token 提供对照对数概率，使当前策略的偏移可以被约束。

当前策略与参考策略在已采样 Token 上的对数概率差用于构造惩罚项。它是逐样本的对数概率比估计，不等同于对整个词表分布计算出的精确 KL 散度。任务反馈与该约束项先组成奖励序列，再据此计算回报和优势。令 $s_t$、$a_t$ 分别表示位置 $t$ 的状态与已采样动作，$m_t\in\{0,1\}$ 表示动作掩码，$T$ 是终局位置，$r^{task}$ 是任务反馈，$\pi_\theta$ 与 $\pi_{ref}$ 分别是当前策略和冻结参考策略，$\beta_{KL}\geq0$ 是约束系数。本节采用的 Token 奖励是：

$$
r_t=m_t\left[\mathbf 1(t=T)r^{task}-\beta_{KL}\left(\log\pi_\theta(a_t\mid s_t)-\log\pi_{ref}(a_t\mid s_t)\right)\right].
$$

这里的对数概率差只在已经采样的动作上计算，是 KL 约束的样本估计；它不能替代对完整词表分布的精确 KL。

| 信号 | 来源 | 回答的问题 |
|:---|:---|:---|
| 任务反馈 | 环境、规则验证器或奖励模型 | 这次结果完成得怎样？ |
| 参考策略约束 | 当前策略与参考策略在已采样 Token 上的对数概率差 | 为偏离基线付出多少代价？ |
| 回报（Return） | 从当前位置开始的折扣累计奖励 | 这个状态之后实际获得了多少反馈？ |
| 价值（Value） | 价值模型对未来回报的估计 | 在当前状态下通常能获得多少反馈？ |
| 优势（Advantage） | 回报相对某种基线的差值 | 这次动作或回答比基线好多少？ |

![奖励信号从轨迹到策略更新的转换链](../public/02_PyTorch_Algorithms/pt_reward_signal_overview.svg)
## Step 2：终局奖励如何反向传播到每个有效 Token

任务正确性通常要到回答结束后才能判断，而策略更新作用于每个生成 Token。对于只有终局反馈的轨迹，可以把任务奖励写入最后一个有效 Token，并在各有效位置加入参考策略约束，再从后向前累计折扣回报。若任务提供过程奖励，则应写入对应动作位置，不能全部挪到终点。

`gamma` 控制远期反馈向前传播的衰减速度：`gamma=1` 保留完整终局反馈，较小的 `gamma` 会减弱早期 Token 接收到的远期信号。Padding、提示词和环境观察不属于策略动作时必须由掩码排除，否则奖励会写入错误位置。

令 $r_k$ 是位置 $k$ 的 Token 奖励，$\gamma\in[0,1]$ 是折扣因子。从位置 $t$ 到终局位置 $T$ 的折扣回报为：

$$
G_t=\sum_{k=t}^{T}\gamma^{k-t}r_k
=r_t+\gamma G_{t+1},\qquad G_{T+1}=0.
$$

该递推解释了为什么实现必须从后向前扫描，也解释了 $\gamma$、终止位置和动作掩码会共同改变早期 Token 接收到的训练信号。

| 时间线对象 | 典型粒度 | 常见风险 |
|:---|:---|:---|
| 任务反馈 | 回答 / 轨迹级，或少量过程位置 | 稀疏、延迟，无法直接区分所有中间动作 |
| 参考策略约束 | 有效动作 Token 级 | 系数过大抑制学习，过小导致策略漂移 |
| 折扣回报 | Token 级累计目标 | 折扣、长度和掩码会改变不同位置的权重 |
| 相对优势 | Token 级或回答级 | 基线偏差、同分或方差过大使更新不稳定 |

![终局任务奖励、采样对数概率比约束与反向累计回报](../public/02_PyTorch_Algorithms/pt_reward_signal_timeline.svg)
## Step 3：PPO 与 GRPO 分别用什么基线计算优势

PPO 训练价值模型估计状态价值，并用回报相对价值估计的差构造逐位置优势。GRPO 通常不维护独立价值模型，而是对同一提示的多个候选回答做组内奖励归一化，先得到回答级相对优势，再把该信号应用到对应回答的有效动作 Token。

奖励模型、价值和优势不能互换：奖励模型评价结果，价值预测未来累计反馈，优势衡量相对基线的改进。PPO 的状态价值基线与 GRPO 的组内比较单位也不能混用。若奖励代理存在漏洞，两种方法都可能提高代理分数却降低真实任务质量，因此必须同时记录奖励分量、独立评测和失败样本。

令 $V_\phi(s_t)$ 表示参数为 $\phi$ 的价值模型对状态 $s_t$ 的回报估计。本节练习使用最直接的回报基线：

$$
A_t^{PPO}=G_t-V_\phi(s_t).
$$

完整 PPO 常用广义优势估计（GAE）控制偏差—方差权衡，其中 $\delta_t$ 是时序差分残差，$\lambda\in[0,1]$ 控制多步残差的衰减：

$$
\delta_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t),\qquad
A_t^{GAE}=\sum_{l=0}^{T-t}(\gamma\lambda)^l\delta_{t+l}.
$$

GRPO 不使用状态价值模型，而是在同一提示的候选组 $g$ 内形成回答级优势。这里 $R_i$ 是第 $i$ 个回答的奖励，$\mu_g$ 与 $\sigma_g$ 是组内均值和标准差，$\varepsilon>0$ 防止近零方差导致除零：

$$
A_i^{GRPO}=\frac{R_i-\mu_g}{\max(\sigma_g,\varepsilon)}.
$$

GAE 用于说明 PPO 的完整机制，但 Step 4 只实现 $G_t-V(s_t)$，以便把回报递推、价值基线和组内基线分别核对。组内标准差还存在统计口径差异：总体标准差直接除以候选数，样本标准差使用贝塞尔修正；两者不改变中心化方向，但会改变有限候选组的缩放幅度，因此实验与复现必须记录所用口径。

| 方法 | 相对基线 | 优点 | 主要失败模式 |
|:---|:---|:---|:---|
| PPO | 价值模型的状态价值估计 | 可利用状态信息，支持逐位置优势估计 | 价值偏差、额外模型状态与训练成本 |
| GRPO | 同一提示下候选回答的组内奖励统计 | 省去独立价值模型，组内比较直接 | 候选过少、奖励同分或排序错误 |
<a id="mechanism-practice"></a>

## Step 4：实现 Token 奖励、回报与两类优势

题目区先在动作掩码范围内组合终局任务反馈与采样 Token 对数概率比，再从后向前计算折扣回报。最后分别实现 PPO 的逐 Token 价值基线和 GRPO 的组内回答基线，避免把两种优势估计混成同一个公式。

| TODO / 函数 | 学习者完成的机制 | 必须满足的约束 | 测试证据 |
|:---|:---|:---|:---|
| TODO 1：组合 Token 奖励<br>`build_token_rewards` | 在有效动作上组合任务奖励与采样对数概率比约束 | 对数概率与动作掩码同形；任务奖励只加入有效终止位置 | 约束符号、掩码、终止位置、输入契约 |
| TODO 2：反向累计回报<br>`discounted_returns` | 从后向前累计未来奖励 | `gamma` 位于 `[0, 1]` | 递推方向与折扣值 |
| TODO 3：计算 PPO 优势<br>`compute_advantages` | 用逐 Token 价值估计构造相对更新信号 | 回报与价值估计同形 | 基线相减与可选标准化 |
| TODO 4：计算 GRPO 组内优势<br>`group_relative_advantages` | 用同提示候选组的奖励统计形成回答级优势 | 奖励为非空一维张量；标准差修正量必须合法；近零方差时保持有限 | 平移不变性、两种标准差口径与退化组 |

```python
import torch
```


```python
# 题目区：沿着奖励 → 回报 → PPO / GRPO 优势的顺序补全训练信号。
def build_token_rewards(task_reward, actor_log_probs, reference_log_probs, beta_kl, terminal_index=-1, action_mask=None):
    """构造 token 级奖励序列。

    task_reward: response / trajectory 完成后的标量反馈。
    actor_log_probs/reference_log_probs: 同一批已采样 token 的对数概率。
    beta_kl: 采样 Token 对数概率比约束强度；更大时策略更贴近参考策略。
    terminal_index: 接收终局任务奖励的 token 位置。
    action_mask: 仅生成动作位置为 True；提示与 padding 位置不参与更新。
    """
    if actor_log_probs.ndim != 1 or actor_log_probs.shape != reference_log_probs.shape:
        raise ValueError("actor/reference log probabilities must be same-shape 1D tensors")
    if beta_kl < 0:
        raise ValueError("beta_kl must be non-negative")

    # ==========================================
    # TODO 1：仅在有效动作上组合采样对数概率比约束与终局任务奖励。
    # sampled_log_ratio = ???  # 当前策略与参考策略在已采样 Token 上的对数概率差。
    # token_rewards = ???  # 提示与 padding 位置必须保持为零。
    # ==========================================
    raise NotImplementedError("TODO 1：构造 Token 奖励")
    return token_rewards


def discounted_returns(token_rewards, gamma=1.0):
    """从序列末端向前计算每个位置的折扣累计奖励。"""
    if token_rewards.ndim != 1:
        raise ValueError("token_rewards must be a 1D tensor")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")

    # ==========================================
    # TODO 2：按 G_t = r_t + gamma * G_{t+1} 反向递推。
    # running_return = ???  # 从序列末端开始的累计回报。
    # returns = ???  # 与奖励序列同形状的逐位置回报。
    # ==========================================
    raise NotImplementedError("TODO 2：计算折扣回报")
    return returns


def compute_advantages(returns, values, normalize=False, eps=1e-8):
    """计算 return 相对 value 基线的优势，可选做零均值标准化。"""
    if returns.shape != values.shape:
        raise ValueError("returns and values must have the same shape")

    # ==========================================
    # TODO 3：计算回报相对价值基线的优势。
    # advantages = ???  # 逐位置回报减价值估计。
    # std = ???  # 仅在启用标准化时计算总体标准差。
    # ==========================================
    raise NotImplementedError("TODO 3：计算优势")
    return advantages


def group_relative_advantages(group_rewards, eps=1e-8, correction=0):
    """按指定标准差修正量计算同提示候选组的回答级相对优势。"""
    # ==========================================
    # TODO 4：以组均值为基线，并在方差有效时按组标准差缩放。
    # centered = ???  # 每个回答奖励减去组均值。
    # scale = ???  # correction=0 为总体标准差，correction=1 为样本标准差。
    # ==========================================
    raise NotImplementedError("TODO 4：计算 GRPO 组内优势")
    return advantages
```

## 测试

测试聚焦四类机制：终局奖励与采样对数概率比约束的位置、回报的反向递推、PPO 对价值基线的响应，以及 GRPO 组内优势的平移不变性。

```python
def test_token_reward_placement():
    """终局奖励只能进入指定位置，采样对数概率比约束的符号必须正确。"""
    actor = torch.tensor([-0.2, -0.3, -0.4])
    reference = torch.tensor([-0.3, -0.3, -0.2])
    rewards = build_token_rewards(
        1.0, actor, reference, beta_kl=0.5,
        action_mask=torch.tensor([False, True, True]),
    )
    expected = torch.tensor([0.0, 0.0, 1.10])
    assert torch.allclose(rewards, expected)


def test_discounted_return_direction():
    """回报必须从序列末端向前递推。"""
    rewards = torch.tensor([0.0, 0.0, 1.0])
    expected = torch.tensor([0.25, 0.5, 1.0])
    assert torch.allclose(discounted_returns(rewards, gamma=0.5), expected)


def test_advantage_baseline_and_normalization():
    """Advantage 应响应 value baseline，并在启用时完成标准化。"""
    returns = torch.tensor([1.0, 2.0, 4.0])
    values = torch.tensor([0.5, 2.5, 2.0])
    raw = compute_advantages(returns, values)
    assert torch.allclose(raw, torch.tensor([0.5, -0.5, 2.0]))
    normalized = compute_advantages(returns, values, normalize=True)
    assert torch.isclose(normalized.mean(), torch.tensor(0.0), atol=1e-6)


def test_group_relative_advantage():
    """GRPO 优势对奖励平移不敏感，并保持组内零均值。"""
    rewards = torch.tensor([0.0, 1.0, 2.0])
    first = group_relative_advantages(rewards)
    shifted = group_relative_advantages(rewards + 7.0)
    assert torch.allclose(first, shifted)
    assert torch.isclose(first.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.all(group_relative_advantages(torch.ones(3)) == 0)
    sample_scaled = group_relative_advantages(rewards, eps=1e-4, correction=1)
    expected = (rewards - rewards.mean()) / (rewards.std(correction=1) + 1e-4)
    assert torch.allclose(sample_scaled, expected)


def test_signal_input_contracts():
    """非法折扣系数和终止位置必须被输入契约拒绝。"""
    try:
        discounted_returns(torch.ones(2), gamma=1.2)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid gamma must raise ValueError")
    try:
        build_token_rewards(1.0, torch.ones(2), torch.ones(2), 0.1, terminal_index=3)
    except IndexError:
        pass
    else:
        raise AssertionError("invalid terminal_index must raise IndexError")


test_token_reward_placement()
test_discounted_return_direction()
test_advantage_baseline_and_normalization()
test_group_relative_advantage()
test_signal_input_contracts()
print("All reward-signal tests passed.")
```

---

🛑 **STOP HERE** 🛑

> 请先完成题目区并运行测试，再查看参考实现。

---

## 参考代码与解析

```python
def build_token_rewards(task_reward, actor_log_probs, reference_log_probs, beta_kl, terminal_index=-1, action_mask=None):
    """在有效动作上组合采样对数概率比约束与终局任务反馈。"""
    if actor_log_probs.ndim != 1 or actor_log_probs.shape != reference_log_probs.shape:
        raise ValueError("actor/reference log probabilities must be same-shape 1D tensors")
    if beta_kl < 0:
        raise ValueError("beta_kl must be non-negative")
    if action_mask is None:
        action_mask = torch.ones_like(actor_log_probs, dtype=torch.bool)
    if action_mask.shape != actor_log_probs.shape:
        raise ValueError("action_mask 必须与对数概率同形")
    action_mask = action_mask.to(dtype=torch.bool, device=actor_log_probs.device)
    if not -actor_log_probs.numel() <= terminal_index < actor_log_probs.numel():
        raise IndexError("terminal_index 超出 Token 序列范围")
    if not bool(action_mask[terminal_index]):
        raise ValueError("terminal_index 必须指向有效动作 Token")
    sampled_log_ratio = actor_log_probs - reference_log_probs
    token_rewards = torch.where(
        action_mask, -beta_kl * sampled_log_ratio, torch.zeros_like(sampled_log_ratio)
    )
    token_rewards = token_rewards.clone()
    token_rewards[terminal_index] += task_reward
    return token_rewards


def discounted_returns(token_rewards, gamma=1.0):
    """从序列末端向前计算折扣累计奖励。"""
    if token_rewards.ndim != 1:
        raise ValueError("token_rewards must be a 1D tensor")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    returns = torch.empty_like(token_rewards)
    running_return = torch.zeros((), dtype=token_rewards.dtype, device=token_rewards.device)
    for index in range(token_rewards.numel() - 1, -1, -1):
        running_return = token_rewards[index] + gamma * running_return
        returns[index] = running_return
    return returns


def compute_advantages(returns, values, normalize=False, eps=1e-8):
    """计算 return 相对 value 的优势，并可选做标准化。"""
    if returns.shape != values.shape:
        raise ValueError("returns and values must have the same shape")
    advantages = returns - values
    if normalize:
        std = advantages.std(unbiased=False)
        if std > eps:
            advantages = (advantages - advantages.mean()) / (std + eps)
        else:
            advantages = advantages - advantages.mean()
    return advantages


def group_relative_advantages(group_rewards, eps=1e-8, correction=0):
    """按指定标准差修正量计算同提示候选组的回答级相对优势。"""
    if group_rewards.ndim != 1 or group_rewards.numel() == 0:
        raise ValueError("group_rewards 必须是非空一维张量")
    if correction not in (0, 1) or group_rewards.numel() <= correction:
        raise ValueError("标准差修正量必须为 0 或 1，且小于候选数量")
    centered = group_rewards - group_rewards.mean()
    scale = group_rewards.std(correction=correction)
    return centered / (scale + eps) if scale > eps else torch.zeros_like(centered)
```

### 解析

#### TODO 1：构造 Token 奖励

- **机制：** 在动作位置写入当前策略与参考策略的采样对数概率比约束，并把任务奖励加到有效终止位置。
- **不变量：** 提示位置奖励保持为零；该差值是已采样 Token 上的估计，不是整个词表分布的精确 KL。
- **测试证据：** 测试核对约束符号、动作掩码、终局奖励位置和输入形状错误。

#### TODO 2：反向累计回报

- **机制：** 按 `G_t = r_t + gamma × G_{t+1}` 从序列末端反向累计折扣回报。
- **不变量：** 终局反馈只能沿时间反向传播，且折扣系数必须位于合法区间。
- **测试证据：** 测试用可手算序列核对每个位置的回报，并拒绝非法 `gamma`。

#### TODO 3：计算相对优势

- **机制：** PPO 路径用回报减去逐 Token 价值基线，并可对优势做零中心和尺度归一。
- **不变量：** 回报与价值必须同形；近零方差时不能用噪声尺度放大优势。
- **测试证据：** 测试核对原始优势、标准化均值、零方差分支和形状错误。

#### TODO 4：计算 GRPO 组内优势

- **机制：** GRPO 路径在同一提示的候选组内用奖励减组均值，再按选定的标准差口径缩放。
- **不变量：** 整组奖励平移不改变优势；总体与样本标准差必须显式区分；组内奖励完全相同时返回零优势。
- **测试证据：** 测试覆盖平移不变性、两种标准差修正量、零方差组和输入契约。

<a id="framework-check"></a>

## 框架一致性核对（可选）

下面使用 TRL 的公开张量工具核对两个最容易产生实现偏差的环节：从模型 Logits 读取已采样 Token 对数概率，以及按 TRL 口径计算 GRPO 组内优势。该单元不启动 Trainer，也不把当前 TRL 已不提供的 PPO Trainer 计入覆盖范围。

| 核对项 | 本节口径 | TRL 路径 | 通过条件 |
|:---|:---|:---|:---|
| 已采样 Token 对数概率 | `log_softmax` 后按 Token 索引读取，并由动作掩码排除无效位置 | `selective_log_softmax` | 有效位置数值一致，无效位置保持为零 |
| GRPO 组内中心化 | 每个回答奖励减同组均值 | `nanstd` 所在的组内缩放口径 | 中心化结果与组内零均值一致 |
| GRPO 标准差缩放 | `correction=1`、`eps=1e-4` | TRL 样本标准差与稳定项 | 非退化组优势一致，零方差组保持为零 |

```python
from importlib.metadata import version

try:
    from trl.trainer.utils import nanstd, selective_log_softmax
except ImportError:
    framework_check = {
        "status": "skipped",
        "reason": "请使用 PT-CPU 环境运行本单元",
    }
else:
    def check_reward_signal_framework_conformance():
        """核对本节采样对数概率与 GRPO 组内缩放是否匹配 TRL。"""
        logits = torch.tensor([
            [[2.0, 1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, 2.0]],
            [[1.0, 3.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 3.0]],
        ])
        sampled_ids = torch.tensor([[0, 1, 2], [1, 0, 2]])
        action_mask = torch.tensor([[True, True, False], [True, False, True]])
        framework_logps = selective_log_softmax(
            logits, sampled_ids, row_mask=action_mask
        )
        local_logps = torch.gather(
            logits.log_softmax(dim=-1), -1, sampled_ids.unsqueeze(-1)
        ).squeeze(-1)
        local_logps = torch.where(action_mask, local_logps, torch.zeros_like(local_logps))

        rewards = torch.tensor([0.0, 1.0, 2.0])
        framework_scale = nanstd(rewards)
        framework_advantages = (rewards - rewards.mean()) / (framework_scale + 1e-4)
        local_advantages = group_relative_advantages(
            rewards, eps=1e-4, correction=1
        )
        constant = torch.ones(3)
        constant_advantages = group_relative_advantages(
            constant, eps=1e-4, correction=1
        )
        checks = {
            "sampled_log_probs_match": bool(torch.allclose(framework_logps, local_logps)),
            "masked_positions_zero": bool(torch.all(framework_logps[~action_mask] == 0)),
            "sample_std_match": bool(torch.isclose(framework_scale, rewards.std(correction=1))),
            "group_advantages_match": bool(torch.allclose(framework_advantages, local_advantages)),
            "zero_variance_group_safe": bool(torch.all(constant_advantages == 0)),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise AssertionError(f"框架一致性核对失败：{failed}")
        return {
            "status": "passed",
            "versions": {"trl": version("trl"), "torch": torch.__version__},
            "framework_path": "trl.trainer.utils.selective_log_softmax + nanstd",
            "checks": checks,
        }

    framework_check = check_reward_signal_framework_conformance()

framework_check

```

## Step 5：可选 GPU 实验——真实模型策略偏移与奖励信号矩阵

CPU 题目确认动作掩码、回报递推以及 PPO 与 GRPO 的两类优势；GPU 实验用同一真实模型快照准备冻结参考策略和有限 LoRA 行为策略，再扫描约束强度与折扣系数，观察真实回复 Token 上的采样对数概率比怎样改变奖励和回报。该实验使用零价值基线隔离信号构造，不验证 Critic 或 GRPO 组采样。

| 学习问题 | CPU 已确认 | GPU 仍需确认 |
|:---|:---|:---|
| 采样对数概率比约束是否作用于真实策略偏移 | 人工对数概率上的符号、动作掩码与终止位置 | 真实模型回复 Token 的当前 / 参考策略对数概率差 |
| `beta_kl` 与 `gamma` 分别改变什么 | 奖励强度与时间传播公式 | 固定策略、样本和任务奖励下的配对信号矩阵 |
| 单卡结果能说明什么 | PPO 与 GRPO 信号公式及输入契约正确 | 真实模型上的奖励—回报路径可执行且资源可记录；不验证 Critic 更新、GRPO 组采样、PPO 训练循环或在线质量 |
### 5.1 环境与固定工作负载

本实验使用 [`PT-RL-CUDA`](../gpu_environment_assets.md#pt-rl-cuda)。首次运行先完成[后训练优化实验 Quickstart](../topic_discussion/post_training_optimization/quickstart.md#pt-rl-cuda-quickstart)；已有环境通过 Profile 预检后，从同一解释器启动 JupyterLab。页面、工作负载和结果状态见[后训练优化实验资产表](../topic_discussion/post_training_optimization/experiment_assets.md)。普通复测只修改 `RUN_MODE` 与 `RUN_LABEL`；模型、固定回复、任务奖励、行为策略准备方式或候选矩阵变化时递增 `WORKLOAD_VERSION`。

| 配置项 | 默认值 | 作用 |
|:---|:---|:---|
| Profile | `PT-RL-CUDA` | 单卡偏好与策略信号机制环境；参考容量为 12 GB 级单卡 |
| 模型 | `Qwen/Qwen2.5-0.5B-Instruct@main` | 5.2 自动解析并记录不可变 commit |
| 固定样本 | 3 组提示、回复与任务奖励 | 隔离数据变化，只比较信号构造 |
| 行为策略 | LoRA，2 次固定监督更新 | 从同一参考快照形成非零、可复查的策略偏移 |
| 候选矩阵 | `beta_kl = 0 / 0.05 / 0.20`，`gamma = 0.90 / 1.00` | 分开观察偏离惩罚和终局反馈传播 |

```python
# 学习者入口：普通复测通常只修改这两项。
RUN_MODE = "dry_run"  # 默认值 dry_run；改为 real_gpu 才执行真实模型实验。
RUN_LABEL = "default"  # 默认值 default；正式复测改为唯一标签，避免覆盖结果。

# 模型快照：main 会在 5.2 自动解析并记录为不可变 commit。
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "main"
LOCAL_FILES_ONLY = False  # 默认值 False；相同快照已缓存且需离线运行时改为 True。

# 固定工作负载：改变以下参数后递增 WORKLOAD_VERSION。
ENVIRONMENT_PROFILE = "PT-RL-CUDA"
WORKLOAD_ID = "pt-reward-signal-matrix"
WORKLOAD_VERSION = "v1"
BETA_KL_CANDIDATES = [0.0, 0.05, 0.20]  # 当前 / 参考策略偏移的三档惩罚强度。
GAMMA_CANDIDATES = [0.90, 1.00]  # 终局反馈向早期 Token 传播的两档折扣。
MAX_LENGTH = 128  # 提示与固定回复合并后的最大 Token 数。
LORA_RANK = 4  # 行为策略适配器秩；只用于形成固定非零策略偏移。
LORA_ALPHA = 8  # LoRA 缩放分子，实际缩放为 alpha / rank。
PREPARE_STEPS = 2  # 行为策略执行的固定监督更新次数。
LEARNING_RATE = 1e-4  # 只更新 LoRA 参数。
SEED = 42
OUTPUT_RELATIVE_PATH = f"benchmarks/results/pt_reward_signal/{WORKLOAD_ID}_{RUN_LABEL}.json"
```

### 5.2 执行配对实验并保存 JSON

5.2 从同一不可变模型快照取得参考策略 Token 对数概率，再用固定 LoRA 更新准备行为策略。候选矩阵只改变 `beta_kl` 与 `gamma`，不重新训练模型。

```python
import gc
import platform
import time

import torch
import torch.nn.functional as F
from tools.notebook_runtime import failure_record, hash_ids, package_version, resolve_model_commit, save_json
from tools.post_training_runtime import build_response_batch, build_token_rewards, compute_advantages, discounted_returns, move_response_batch
from tools.project_runtime import resolve_project_root
from tools.training_mechanism_runtime import cuda_hardware_snapshot

FIXED_TRAJECTORIES = [
    {"prompt": "Answer briefly: Why keep a reference policy?", "response": "It makes policy drift measurable during constrained updates.", "task_reward": 1.0},
    {"prompt": "Answer briefly: What does an advantage represent?", "response": "It measures return relative to a baseline expectation.", "task_reward": 0.6},
    {"prompt": "Answer briefly: Can reward replace evaluation?", "response": "No, reward is a training signal and still needs independent evaluation.", "task_reward": 0.9},
]

def response_token_logps(model, batch):
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    targets = batch["input_ids"][:, 1:]
    mask = batch["response_mask"][:, 1:]
    token_logps = F.log_softmax(outputs.logits[:, :-1].float(), dim=-1)
    selected = token_logps.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return selected, mask

REPO_ROOT = resolve_project_root()
OUTPUT_PATH = REPO_ROOT / OUTPUT_RELATIVE_PATH
result = {
    "schema_version": "post-training-mechanism-gpu/v1",
    "semantic_id": "PT-REWARD-SIGNAL",
    "environment_profile": ENVIRONMENT_PROFILE,
    "run_mode": RUN_MODE, "run_label": RUN_LABEL,
    "framework": {"python": platform.python_version(), "torch": torch.__version__, "transformers": package_version("transformers"), "peft": package_version("peft")},
    "hardware": {"cuda_available": torch.cuda.is_available(), "cuda": torch.version.cuda},
    "model": {"id": MODEL_ID, "requested_revision": MODEL_REVISION, "resolved_commit": None},
    "dataset": {"source": "embedded_fixed_trajectories", "samples": len(FIXED_TRAJECTORIES), "sample_hash": hash_ids([str(item) for item in FIXED_TRAJECTORIES])},
    "workload": {"id": WORKLOAD_ID, "version": WORKLOAD_VERSION, "beta_kl_candidates": BETA_KL_CANDIDATES, "gamma_candidates": GAMMA_CANDIDATES, "max_length": MAX_LENGTH, "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA, "prepare_steps": PREPARE_STEPS, "learning_rate": LEARNING_RATE, "seed": SEED},
    "baseline": {"name": "frozen_reference"}, "actor_preparation": None,
    "candidates": [], "failure": None, "evidence_level": "dry_run", "decision": "not_measured",
}

execution_stage = "environment_preflight"
model = optimizer = None
if RUN_MODE == "real_gpu":
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("RUN_MODE=real_gpu，但 CUDA 不可用")
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = torch.device("cuda")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()
        execution_stage = "revision_and_tokenization"
        model_commit, revision_source = resolve_model_commit(MODEL_ID, MODEL_REVISION, local_files_only=LOCAL_FILES_ONLY)
        result["model"].update({"resolved_commit": model_commit, "resolution_source": revision_source})
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=model_commit, local_files_only=LOCAL_FILES_ONLY)
        batch = build_response_batch(tokenizer, [item["prompt"] for item in FIXED_TRAJECTORIES], [item["response"] for item in FIXED_TRAJECTORIES], max_length=MAX_LENGTH)
        batch = move_response_batch(batch, device)

        execution_stage = "reference_and_actor_preparation"
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=model_commit, local_files_only=LOCAL_FILES_ONLY, dtype=dtype).to(device)
        base.config.use_cache = False
        model = get_peft_model(base, LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.0, target_modules=["q_proj", "v_proj"], bias="none", task_type=TaskType.CAUSAL_LM))
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE)
        model.eval()
        with torch.no_grad(), model.disable_adapter(), torch.autocast("cuda", dtype=dtype):
            reference_logps, response_mask = response_token_logps(model, batch)
        losses, prepare_times_ms = [], []
        model.train()
        for _ in range(PREPARE_STEPS):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); started = time.perf_counter()
            with torch.autocast("cuda", dtype=dtype):
                actor_logps, response_mask = response_token_logps(model, batch)
                loss = -(actor_logps * response_mask).sum() / response_mask.sum()
            loss.backward(); optimizer.step(); torch.cuda.synchronize()
            losses.append(float(loss.detach())); prepare_times_ms.append((time.perf_counter() - started) * 1000)
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            actor_logps, response_mask = response_token_logps(model, batch)
        result["actor_preparation"] = {"losses": losses, "times_ms": prepare_times_ms, "mean_absolute_logp_shift": float(((actor_logps - reference_logps).abs() * response_mask).sum() / response_mask.sum())}

        execution_stage = "signal_candidate_matrix"
        terminal_checks = []
        for beta_kl in BETA_KL_CANDIDATES:
            for gamma in GAMMA_CANDIDATES:
                starts, terminal_returns, advantage_stds = [], [], []
                for index, item in enumerate(FIXED_TRAJECTORIES):
                    valid = response_mask[index]
                    actor_row = actor_logps[index][valid].detach().cpu()
                    reference_row = reference_logps[index][valid].detach().cpu()
                    rewards = build_token_rewards(item["task_reward"], actor_row, reference_row, beta_kl)
                    returns = discounted_returns(rewards, gamma=gamma)
                    advantages = compute_advantages(returns, torch.zeros_like(returns), normalize=True)
                    starts.append(float(returns[0])); terminal_returns.append(float(returns[-1])); advantage_stds.append(float(advantages.std(unbiased=False)))
                    terminal_checks.append(abs(float(rewards[-1] + beta_kl * (actor_row[-1] - reference_row[-1]) - item["task_reward"])) < 1e-5)
                result["candidates"].append({"beta_kl": beta_kl, "gamma": gamma, "mean_start_return": sum(starts) / len(starts), "mean_terminal_return": sum(terminal_returns) / len(terminal_returns), "mean_normalized_advantage_std": sum(advantage_stds) / len(advantage_stds)})
        matrix_complete = len(result["candidates"]) == len(BETA_KL_CANDIDATES) * len(GAMMA_CANDIDATES)
        passed = result["actor_preparation"]["mean_absolute_logp_shift"] > 0 and matrix_complete and all(terminal_checks)
        result.update({"hardware": cuda_hardware_snapshot(), "mechanism_checks": {"candidate_matrix_complete": matrix_complete, "terminal_reward_placement_passed": all(terminal_checks), "peak_allocated_bytes": int(torch.cuda.max_memory_allocated())}, "evidence_level": "real_model_reward_signal_benchmark", "decision": "accept_mechanism" if passed else "tune"})
    except Exception as error:
        result.update({"failure": failure_record(execution_stage, error), "evidence_level": "failed_real_gpu_attempt", "decision": "reject_until_fixed"})
    finally:
        del optimizer, model
        gc.collect(); torch.cuda.empty_cache()
elif RUN_MODE != "dry_run":
    raise ValueError("RUN_MODE 只能是 dry_run 或 real_gpu")

save_json(OUTPUT_PATH, result)
print(OUTPUT_PATH)
print(result["decision"], result["failure"])
```

### 5.3 读取结果

本节只读取当前 `RUN_LABEL` 对应的 JSON，不回退到其他运行。先检查失败记录，再比较策略偏移、终局奖励放置检查和六组信号候选。

```python
import json

if not OUTPUT_PATH.exists():
    print(f"尚未找到结果文件：{OUTPUT_PATH}")
else:
    record = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    print(json.dumps({key: record.get(key) for key in ("run_label", "model", "actor_preparation", "mechanism_checks", "failure", "evidence_level", "decision")}, ensure_ascii=False, indent=2))
    for candidate in record.get("candidates", []):
        print({key: candidate.get(key) for key in ("beta_kl", "gamma", "mean_start_return", "mean_terminal_return", "mean_normalized_advantage_std")})
```

### 5.4 解释证据与决策

先确认行为策略确实相对参考策略产生非零偏移、任务奖励仍只落在终止位置，并且六组候选使用同一模型、样本和行为策略。零价值基线只用于隔离奖励与回报机制，不代表已训练 Critic。RTX 5070 Ti 实测完成了全部候选，并验证了采样对数概率比约束强度与折扣系数分别改变约束幅度和时间传播。

| 观察项 | 读取字段 | 当前结果 | 解释与动作 |
|:---|:---|:---|:---|
| 运行身份 | `failure`、model commit、固定样本、行为策略准备 | `rtx5070ti_reward_signal_v3`；无失败；固定 commit | 模型、样本或策略准备变化时整体重跑 |
| 机制与路径 | 策略偏移、终止位置检查、候选矩阵完整性 | 动作位置平均绝对对数概率偏移 0.0902；终止位置检查通过；6/6 候选完整 | 参考策略冻结，任务奖励只写入终止位置 |
| 正确性或质量 | `mean_start_return`、`mean_terminal_return`、优势标准差 | `beta_kl` 增大时起始与终止回报下降；`gamma=0.9` 时终局反馈向前折扣 | 两个候选轴分别控制约束强度与传播范围 |
| 资源与性能 | 行为策略准备时间、峰值显存 | 两步准备约 76.59 / 41.95 ms；峰值约 2.16 GB | 只解释当前信号实验资源，不代表 PPO / GRPO 系统吞吐 |
| 证据与决策 | `evidence_level`、`decision` | `real_model_reward_signal_benchmark`；`accept_mechanism` | 接受奖励信号机制链；Critic 与在线系统另行验证 |

## 相关阅读

- [Part 02 · 16 PPO 资源与状态](./16_RLHF_PPO_Memory.md)
- [Part 02 · 18 GRPO 损失](./18_GRPO_Loss_Tutorial.md)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
