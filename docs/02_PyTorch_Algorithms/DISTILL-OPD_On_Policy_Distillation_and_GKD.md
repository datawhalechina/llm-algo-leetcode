# DISTILL-OPD. On Policy Distillation and GKD | 在策略蒸馏与 GKD

**难度：** Hard | **环境：** [CPU 主线](#runtime-environment) | **标签：** `蒸馏`、`GKD`、`在策略采样`、`教师学生` | **目标人群：** 理解概率分布、准备学习蒸馏目标的读者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/DISTILL-OPD_On_Policy_Distillation_and_GKD.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

学生只看固定正确答案时，可能一直没学过“自己答错之后怎样继续”。在策略蒸馏让当前学生先生成回答，再把它访问过的前缀送给冻结教师，获得下一词元的分布反馈。这样，训练会遇到学生实际容易产生的回答，而不仅是预先准备好的文本。

广义知识蒸馏（Generalized Knowledge Distillation，GKD）把轨迹来源与分布比较方式分开选择。接下来依次看回答由谁生成、两个模型怎样比较同一位置的分布、哪些回答位置参与学生更新。

**关键词：** `on-policy distillation`、`GKD`、`generalized JSD`、`completion mask`、`teacher feedback`

---

<a id="runtime-environment"></a>

## 运行环境

- **基础练习：** 默认使用 [`BASE-CPU`](../guide.md#base-cpu)，运行 [Step 4 的题目与测试](#mechanism-practice)。
- **框架核对：** 如需用 PyTorch 分布 API 独立核对广义 JSD，使用 [`PT-CPU`](../topic_discussion/post_training_optimization/quickstart.md#pt-cpu-quickstart)，运行[框架一致性核对](#framework-check)。

---

## 前置阅读

先理解提示和回答怎样对齐到预测位置，再看学生生成的轨迹为什么需要记录模型版本。

- [SFT 监督边界与损失](./13_SFT_Training_Loop.md)：核对下一词元预测、回答位置与损失掩码。
- [轨迹与策略新鲜度](./PT-ROLLOUT-LIFECYCLE_Rollout_and_Policy_Freshness.md)：理解生成轨迹的策略版本与训练时模型的关系。

---

## Step 1：训练批次中的轨迹由谁生成

轨迹是一条提示后接出的回答序列，前缀是生成到某一位置之前已经出现的词元。固定轨迹来自数据集或预先保存的回答；在策略轨迹由当前学生采样。冻结教师在选定轨迹的每个前缀上给出下一词元分布，两种来源因而都能提供蒸馏位置。

设输入为 $x$，完整回答为 $y$，固定轨迹分布为 $\mu_{\mathrm{off}}(y\mid x)$，当前学生策略为 $\pi_S(y\mid x)$，在策略比例为 $\lambda\in[0,1]$。蒸馏批次的轨迹分布写成

$$
\mu_{\lambda}(y\mid x)
=(1-\lambda)\mu_{\mathrm{off}}(y\mid x)
+\lambda\pi_S(y\mid x).
$$

$\lambda=0$ 表示全部使用固定轨迹， $\lambda=1$ 表示全部由当前学生生成。这个比例描述整批回答的来源，不决定教师与学生分布怎样比较。小批次实际数量要取整数，例如 3 条回答取一半时只能选 1 条或 2 条，并记录采用的取整规则。

| 轨迹来源 | 生成者 | 主要作用 | 必须记录 |
|:---|:---|:---|:---|
| 固定轨迹 | 数据集或预先生成的回答 | 保持任务覆盖与训练稳定性 | 数据版本、样本身份、生成来源 |
| 在策略轨迹 | 当前学生策略 $\pi_S$ | 暴露学生当前会访问的前缀与错误 | 学生版本、采样参数、轨迹身份 |

![离线样本与学生在策略轨迹进入同一蒸馏批次](../public/02_PyTorch_Algorithms/distill_opd_sampling_paths.svg)

## Step 2：比较同一位置的教师与学生分布

同一轨迹前缀分别送入冻结教师与可训练学生。Logits 是模型尚未归一化的词表得分，经 softmax 得到下一词元概率 $p_T$ 和 $p_S$；二者每一维应指向同一个词元。KL 散度比较两个概率分布，方向改变时对概率差异的惩罚也会改变。

对 $\beta\in(0,1)$，这里采用未另加归一化因子的加权 Jensen–Shannon 散度（JSD）。先形成混合分布，再将两个分布分别与它比较。两个端点另外定义为前向 KL 和反向 KL；内部加权式在端点附近趋向零，端点 KL 是单独选择的目标。 $\lambda$ 选择回答来源， $\beta$ 选择该位置的分布比较方式，两者各控制一件事。

$$
m_{\beta}=\beta p_T+(1-\beta)p_S,
$$

$$
D_{\mathrm{GJS}}^{\beta}(p_T,p_S)
=\beta D_{\mathrm{KL}}(p_T\Vert m_{\beta})
+(1-\beta)D_{\mathrm{KL}}(p_S\Vert m_{\beta}),\quad 0<\beta<1.
$$

| 参数取值 | 对应计算 | 解释 |
|:---|:---|:---|
| $\beta=0$ | $D_{\mathrm{KL}}(p_T\Vert p_S)$ | 以教师分布为参照惩罚学生差异 |
| $0<\beta<1$ | 上述加权 JSD | 教师、学生分别与混合分布比较 |
| $\beta=1$ | $D_{\mathrm{KL}}(p_S\Vert p_T)$ | 按学生分布衡量与教师的差异 |
| $\lambda\in[0,1]$ | 固定轨迹与学生轨迹的来源比例 | 选择在哪些前缀上比较上述分布 |

## Step 3：用回答位置形成学生更新

完成区指回答所占的序列位置，Padding 是补齐不同长度序列用的填充。取值为 0 或 1 的掩码 $M_{b,t}$ 在批次 $b$、位置 $t$ 标记是否计算蒸馏目标：提示、填充以及指定排除的工具返回取零，保留的回答位置取一。除数使用有效位置数，例如只保留 3 个回答词元时就除以 3，而不是整段输入长度。

对齐后的教师、学生分布分别记作 $p^T_{b,t}$、 $p^S_{b,t}$，目标如下。有效位置数为零时应停止本批计算。教师分布停止梯度，学生分布及其在混合分布中的部分参与求导；已经采样出的离散词元作为固定输入。数值计算还需要输入有限：掩码按位置选择损失，NaN 与零相乘仍会得到 NaN。

$$
\mathcal L_{\mathrm{distill}}
=\frac{\sum_{b,t}M_{b,t}
D_{\mathrm{GJS}}^{\beta}(p^T_{b,t},p^S_{b,t})}
{\sum_{b,t}M_{b,t}}.
$$

| 对象 | 在训练中的作用 | 核对内容 |
|:---|:---|:---|
| 轨迹 | 给出两个模型共同处理的前缀 | 来源、生成学生版本与采样参数 |
| 教师分布 | 提供每个前缀下的词表概率 | 教师冻结、词元映射与位置对齐 |
| 蒸馏掩码 | 选择参与平均的回答位置 | 提示、填充、排除区及有效数量 |
| 学生参数 | 接收所选位置的损失梯度 | 学生有梯度，教师梯度为空 |

![两个词表分布先比较，完成区掩码再选择参与平均的位置](../public/02_PyTorch_Algorithms/distill_opd_teacher_feedback.svg)

<a id="mechanism-practice"></a>
## Step 4：实现轨迹混合与掩码蒸馏目标

题目区使用小型 Logits 张量复现 GKD 的四个核心动作：按 $\lambda$ 标记在策略样本来源、按 $\beta$ 计算广义 JSD、从序列边界构造完成区掩码，再按有效 Token 归约损失。输入 Logits 已对齐到预测同一目标 Token 的位置；真实因果模型需先对齐前一位置 Logits 与下一 Token。提示长度按无左填充的位置计数，本题使用右侧 Padding。测试覆盖比例与随机种子、散度端点、提示与 Padding 排除、工具结果排除、空掩码和教师冻结。

| TODO / 函数 | 学习者完成的机制 | 必须满足的约束 | 测试证据 |
|:---|:---|:---|:---|
| TODO 1：选择轨迹来源<br>`build_on_policy_mask` | 按 $\lambda$ 和固定随机种子分配来源标记（采样回答另行完成） | 比例位于零到一；数量正确；同一随机种子可复现 | 检查两个边界、中间比例、种子复现和非法输入 |
| TODO 2：计算逐 Token 广义 JSD<br>`generalized_jsd_from_logits` | 将教师和学生 Logits 转成分布并与混合分布比较 | $\beta\in[0,1]$；两个端点分别使用前向和反向 KL；教师分布停止梯度 | 相同分布接近零，两个端点与对应 KL 一致，非法 $\beta$ 报错 |
| TODO 3：构造完成区掩码<br>`build_completion_mask` | 从 Attention Mask、提示长度和排除区得到蒸馏位置 | 提示、Padding 和工具结果均不得进入掩码 | 精确核对两条变长序列的有效位置，非法形状与边界报错 |
| TODO 4：归约有效蒸馏位置<br>`masked_distillation_loss` | 对有限 Logits 计算逐位置散度，再按掩码求平均 | 掩码形状匹配；有效位置数大于零；梯度只进入学生 | 修改被屏蔽位置的有限 Logits 不改变损失，空掩码报错，学生有梯度而教师无梯度 |


```python
import torch

```


```python
def build_on_policy_mask(
    batch_size: int, on_policy_fraction: float, seed: int = 0
) -> torch.Tensor:
    """按给定比例标记批次中由当前学生生成的轨迹。"""
    # ==========================================
    # TODO 1：检查输入，用固定随机种子抽取准确数量的在策略样本。
    # on_policy_count = ???
    # ==========================================
    raise NotImplementedError("TODO 1：构造在策略轨迹掩码")


def generalized_jsd_from_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """返回每个 Token 位置的广义 Jensen-Shannon 散度。"""
    # ==========================================
    # TODO 2：计算教师、学生和混合分布，再组合两项 KL。
    # teacher_probs = ???
    # student_probs = ???
    # mixture_probs = ???
    # ==========================================
    raise NotImplementedError("TODO 2：计算广义 JSD")


def build_completion_mask(
    attention_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
    excluded_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """构造只保留回答 Token 且排除工具结果的布尔掩码。"""
    # ==========================================
    # TODO 3：组合 Attention Mask、提示长度和可选排除区。
    # positions = ???
    # completion_mask = ???
    # ==========================================
    raise NotImplementedError("TODO 3：构造完成区掩码")


def masked_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    loss_mask: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """只在掩码选中的 Token 位置归约蒸馏损失。"""
    # ==========================================
    # TODO 4：检查形状与有效位置数，并按有效 Token 数归约。
    # per_token_loss = ???
    # ==========================================
    raise NotImplementedError("TODO 4：归约掩码蒸馏损失")

```

## 测试

运行下方 CPU 机制测试，检查轨迹来源比例、广义 JSD 的数值边界、完成区掩码和教师—学生梯度隔离。


```python
def _assert_raises(error_type, fn) -> None:
    """检查调用是否抛出指定异常。"""
    try:
        fn()
    except error_type:
        return
    raise AssertionError(f"预期抛出 {error_type.__name__}")


def test_on_policy_mask():
    """核对来源数量与种子复现。"""
    assert build_on_policy_mask(4, 0.0, seed=7).tolist() == [False] * 4
    assert build_on_policy_mask(4, 1.0, seed=7).tolist() == [True] * 4
    first = build_on_policy_mask(8, 0.5, seed=7)
    repeated = build_on_policy_mask(8, 0.5, seed=7)
    changed = build_on_policy_mask(8, 0.5, seed=11)
    assert first.sum().item() == 4 and torch.equal(first, repeated)
    assert not torch.equal(first, changed)
    _assert_raises(ValueError, lambda: build_on_policy_mask(4, 1.1))
    _assert_raises(ValueError, lambda: build_on_policy_mask(0, 0.5))


def test_generalized_jsd_boundaries():
    """核对内部散度与显式 KL 端点。"""
    logits = torch.tensor([[[1.0, 0.0, -1.0]]])
    same = generalized_jsd_from_logits(logits, logits, beta=0.4)
    assert same.shape == (1, 1)
    assert torch.allclose(same, torch.zeros_like(same), atol=1e-7)

    shifted = generalized_jsd_from_logits(
        torch.tensor([[[4.0, -2.0, -2.0]]]),
        torch.tensor([[[-2.0, 4.0, -2.0]]]),
        beta=0.4,
    )
    assert shifted.item() > 0.1

    student_logits = torch.tensor([[[2.0, 0.0, -1.0]]])
    teacher_logits = torch.tensor([[[0.0, 2.0, -1.0]]])
    student_probs = torch.softmax(student_logits, dim=-1)
    teacher_probs = torch.softmax(teacher_logits, dim=-1)
    forward_kl = torch.sum(teacher_probs * (torch.log(teacher_probs) - torch.log(student_probs)), dim=-1)
    reverse_kl = torch.sum(student_probs * (torch.log(student_probs) - torch.log(teacher_probs)), dim=-1)
    assert torch.allclose(generalized_jsd_from_logits(student_logits, teacher_logits, 0.0), forward_kl)
    assert torch.allclose(generalized_jsd_from_logits(student_logits, teacher_logits, 1.0), reverse_kl)
    _assert_raises(ValueError, lambda: generalized_jsd_from_logits(logits, logits, -0.1))


def test_completion_mask():
    """逐位置核对回答与排除区。"""
    attention_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    prompt_lengths = torch.tensor([2, 1])
    excluded_mask = torch.tensor([[0, 0, 0, 1, 0], [0, 0, 1, 0, 0]])
    actual = build_completion_mask(attention_mask, prompt_lengths, excluded_mask)
    expected = torch.tensor(
        [[False, False, True, False, False], [False, True, False, True, True]]
    )
    assert torch.equal(actual, expected)
    _assert_raises(ValueError, lambda: build_completion_mask(torch.tensor([[0, 1, 1]]), torch.tensor([1])))
    _assert_raises(
        ValueError,
        lambda: build_completion_mask(attention_mask, torch.tensor([6, 1])),
    )
    _assert_raises(
        ValueError,
        lambda: build_completion_mask(attention_mask, prompt_lengths, torch.ones(2, 4)),
    )


def test_mask_and_gradient_isolation():
    """检查有限输入的掩码与梯度。"""
    student = torch.tensor(
        [[[2.0, 0.0], [0.0, 2.0], [1.0, -1.0]]], requires_grad=True
    )
    teacher = torch.tensor(
        [[[0.0, 2.0], [2.0, 0.0], [-1.0, 1.0]]], requires_grad=True
    )
    mask = torch.tensor([[0, 1, 1]], dtype=torch.bool)
    loss = masked_distillation_loss(student, teacher, mask, beta=0.5)

    changed_teacher = teacher.detach().clone()
    changed_teacher[:, 0, :] = torch.tensor([100.0, -100.0])
    changed = masked_distillation_loss(student, changed_teacher, mask, beta=0.5)
    assert torch.allclose(loss, changed, atol=1e-7)

    _assert_raises(ValueError, lambda: masked_distillation_loss(student, teacher, torch.tensor([[0.0, 0.5, 1.0]]), 0.5))
    loss.backward()
    assert student.grad is not None and student.grad[:, 1:, :].abs().sum() > 0
    assert teacher.grad is None
    _assert_raises(
        ValueError,
        lambda: masked_distillation_loss(student.detach(), teacher.detach(), torch.zeros_like(mask), 0.5),
    )


test_on_policy_mask()
test_generalized_jsd_boundaries()
test_completion_mask()
test_mask_and_gradient_isolation()
print("✅ 在策略蒸馏轨迹、广义 JSD、掩码和梯度隔离测试通过。")

```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---

## 参考代码与解析

### 代码


```python
def build_on_policy_mask(
    batch_size: int, on_policy_fraction: float, seed: int = 0
) -> torch.Tensor:
    """按给定比例标记批次中由当前学生生成的轨迹。"""
    # TODO 1：按固定随机种子抽取指定数量，避免来源与批次位置绑定。
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正整数")
    if not 0.0 <= on_policy_fraction <= 1.0:
        raise ValueError("on_policy_fraction 必须位于 [0, 1]")
    on_policy_count = int(batch_size * on_policy_fraction + 0.5)
    mask = torch.zeros(batch_size, dtype=torch.bool)
    if on_policy_count:
        generator = torch.Generator().manual_seed(seed)
        selected = torch.randperm(batch_size, generator=generator)[:on_policy_count]
        mask[selected] = True
    return mask


def generalized_jsd_from_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """返回每个 Token 位置的广义 Jensen-Shannon 散度。"""
    # TODO 2：教师停止梯度后，与学生共同构造混合分布。
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("教师和学生 Logits 形状必须一致")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta 必须位于 [0, 1]")

    teacher_probs = torch.softmax(teacher_logits.detach(), dim=-1)
    student_probs = torch.softmax(student_logits, dim=-1)
    tiny = torch.finfo(student_probs.dtype).tiny
    safe_teacher = teacher_probs.clamp_min(tiny)
    safe_student = student_probs.clamp_min(tiny)
    if beta == 0.0:
        return torch.sum(teacher_probs * (torch.log(safe_teacher) - torch.log(safe_student)), dim=-1)
    if beta == 1.0:
        return torch.sum(student_probs * (torch.log(safe_student) - torch.log(safe_teacher)), dim=-1)

    mixture_probs = beta * teacher_probs + (1.0 - beta) * student_probs
    safe_mixture = mixture_probs.clamp_min(tiny)
    teacher_kl = torch.sum(
        teacher_probs * (torch.log(safe_teacher) - torch.log(safe_mixture)), dim=-1
    )
    student_kl = torch.sum(
        student_probs * (torch.log(safe_student) - torch.log(safe_mixture)), dim=-1
    )
    return beta * teacher_kl + (1.0 - beta) * student_kl


def build_completion_mask(
    attention_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
    excluded_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """构造只保留回答 Token 且排除工具结果的布尔掩码。"""
    # TODO 3：只保留有效回答 Token，并排除工具结果等非监督区域。
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask 必须为二维张量")
    batch_size, sequence_length = attention_mask.shape
    if prompt_lengths.shape != (batch_size,):
        raise ValueError("prompt_lengths 形状必须等于批大小")
    if torch.any(prompt_lengths < 0) or torch.any(prompt_lengths > sequence_length):
        raise ValueError("prompt_lengths 必须位于序列边界内")
    if torch.any((attention_mask != 0) & (attention_mask != 1)):
        raise ValueError("attention_mask 必须为二值掩码")
    if torch.any(attention_mask[:, 1:] > attention_mask[:, :-1]):
        raise ValueError("本题只支持无左填充的右侧 Padding")
    positions = torch.arange(sequence_length, device=attention_mask.device).unsqueeze(0)
    completion_mask = attention_mask.bool() & (positions >= prompt_lengths.unsqueeze(1))
    if excluded_mask is not None:
        if excluded_mask.shape != attention_mask.shape:
            raise ValueError("excluded_mask 形状必须与 attention_mask 一致")
        completion_mask &= ~excluded_mask.to(device=attention_mask.device, dtype=torch.bool)
    return completion_mask


def masked_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    loss_mask: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """只在掩码选中的 Token 位置归约蒸馏损失。"""
    # TODO 4：只按有效蒸馏 Token 数归约，空掩码立即报错。
    per_token_loss = generalized_jsd_from_logits(student_logits, teacher_logits, beta)
    if loss_mask.shape != per_token_loss.shape:
        raise ValueError("loss_mask 形状必须与逐 Token 损失一致")
    if torch.any((loss_mask != 0) & (loss_mask != 1)):
        raise ValueError("loss_mask 必须为二值掩码")
    float_mask = loss_mask.to(device=per_token_loss.device, dtype=per_token_loss.dtype)
    valid_tokens = float_mask.sum()
    if valid_tokens.item() == 0:
        raise ValueError("蒸馏掩码没有有效 Token")
    return (per_token_loss * float_mask).sum() / valid_tokens

```

### 解析

#### TODO 1：分配轨迹来源标记

- 用 $\lambda$ 乘批大小并按“加 0.5 后取整”得到学生来源数量，再从固定种子的随机排列中选择对应位置，输出布尔掩码。
- 掩码保存来源分配，回答采样由后续生成完成。测试核对比例两端、中间数量及相同种子的复现。

#### TODO 2：计算同一位置的分布差异

- 教师 Logits 先 detach，再与学生分别转概率；内部取值形成混合分布并组合两项 KL，学生在混合分布中的部分也保留梯度。
- 两个端点使用独立 KL 分支。测试用显式前向／反向 KL 核对，并检查相同分布、偏移分布与越界参数。

#### TODO 3：找出回答的有效位置

- 对每条右填充序列，先保留提示结束之后且 Attention Mask 为真的位置，再排除工具返回等指定区域。
- 两条变长序列的逐位置断言检查有效位置；左填充、提示越界和排除掩码形状错误分别报错。

#### TODO 4：按有效词元数求平均

- 先核对掩码形状、二值性和非空，再将有限输入得到的逐位置散度乘掩码，除以有效词元数。
- 测试改变屏蔽区的有限得分后损失保持一致，并检查空掩码报错、学生得到梯度、教师梯度为空。

<a id="framework-check"></a>

## 框架一致性核对（可选）

使用相同教师／学生 Logits、 $\beta=0.35$ 和回答掩码，将教学公式与 PyTorch 的 `Categorical`、`kl_divergence` 独立计算路径比较。反向检查执行教学损失，确认学生有有限梯度、教师无梯度。另有 TRL 类签名与 beta 配置的接口探测；训练器实际损失由其版本实现决定，数值比较对象是下表中的 PyTorch 分布路径。

| 核对对象 | Step 4 教学实现 | 成熟框架 CPU 路径 | 共同输入与通过条件 |
|:---|:---|:---|:---|
| 逐词元加权 JSD | `generalized_jsd_from_logits` 组合两项 KL | PyTorch `Categorical`／`kl_divergence` 独立计算 | 相同 Logits 与 beta，逐位置差异在容差内 |
| 回答区平均损失 | `masked_distillation_loss` 按有效数量归约 | 独立分布结果按同一布尔掩码求平均 | 同一掩码，标量损失在容差内 |


```python
def check_gkd_distribution_conformance():
    """核对教学版目标、PyTorch 分布计算和 TRL 稳定接口。"""
    import inspect
    from importlib.metadata import version
    from torch.distributions import Categorical, kl_divergence
    try:
        from trl import DistillationConfig, DistillationTrainer
    except ImportError:
        return {"status": "skipped", "reason": "请使用 PT-CPU 环境运行本单元"}

    student_logits = torch.tensor(
        [[[1.2, 0.1, -0.7], [0.0, 1.0, -1.0], [0.4, -0.2, 0.7]]],
        requires_grad=True,
    )
    teacher_logits = torch.tensor(
        [[[0.2, 1.1, -0.4], [0.8, 0.1, -0.5], [-0.1, 0.9, 0.2]]],
        requires_grad=True,
    )
    beta = 0.35
    loss_mask = torch.tensor([[0, 1, 1]], dtype=torch.bool)

    local_per_token = generalized_jsd_from_logits(student_logits, teacher_logits, beta)
    teacher_probs = torch.softmax(teacher_logits.detach(), dim=-1)
    student_probs = torch.softmax(student_logits, dim=-1)
    mixture_probs = beta * teacher_probs + (1.0 - beta) * student_probs
    framework_per_token = (
        beta * kl_divergence(Categorical(probs=teacher_probs), Categorical(probs=mixture_probs))
        + (1.0 - beta)
        * kl_divergence(Categorical(probs=student_probs), Categorical(probs=mixture_probs))
    )

    local_loss = masked_distillation_loss(student_logits, teacher_logits, loss_mask, beta)
    framework_loss = framework_per_token[loss_mask].mean()
    per_token_match = torch.allclose(local_per_token, framework_per_token, atol=1e-6)
    loss_match = torch.allclose(local_loss, framework_loss, atol=1e-6)

    local_loss.backward()
    gradient_boundary = (
        student_logits.grad is not None
        and torch.isfinite(student_logits.grad).all().item()
        and teacher_logits.grad is None
    )

    trainer_parameters = inspect.signature(DistillationTrainer).parameters
    stable_on_policy_interface = "model" in trainer_parameters and "teacher_model" in trainer_parameters
    stable_config_has_no_lambda = "lmbda" not in DistillationConfig.__dataclass_fields__
    endpoint_configs = [
        DistillationConfig(output_dir="/tmp/trl-distillation-interface", beta=value, use_cpu=True)
        for value in (0.0, 1.0)
    ]
    endpoint_beta_supported = [config.beta for config in endpoint_configs] == [0.0, 1.0]
    try:
        DistillationConfig(output_dir="/tmp/trl-distillation-interface", beta=1.1, use_cpu=True)
    except ValueError:
        invalid_beta_rejected = True
    else:
        invalid_beta_rejected = False

    assert (
        per_token_match
        and loss_match
        and gradient_boundary
        and stable_on_policy_interface
        and stable_config_has_no_lambda
        and endpoint_beta_supported
        and invalid_beta_rejected
    )
    return {
        "status": "passed",
        "check_kind": "framework_conformance",
        "versions": {"torch": torch.__version__, "trl": version("trl")},
        "checks": {
            "per_token_gjsd_match": bool(per_token_match),
            "masked_loss_match": bool(loss_match),
            "student_only_gradient": bool(gradient_boundary),
            "stable_on_policy_interface": bool(stable_on_policy_interface),
            "stable_config_has_no_lambda": bool(stable_config_has_no_lambda),
            "beta_endpoints_supported": bool(endpoint_beta_supported),
            "invalid_beta_rejected": bool(invalid_beta_rejected),
        },
    }


framework_check = check_gkd_distribution_conformance()
framework_check

```

## 相关阅读

从轨迹来源继续追查训练目标，或进入真实学生模型比较时，可以阅读以下资料。

- [On-Policy Distillation of Language Models](https://arxiv.org/abs/2306.13649)：看学生自生成回答与固定数据训练的差别，以及不同散度的比较。
- [TRL DistillationTrainer](https://huggingface.co/docs/trl/distillation_trainer)：查看训练器的回答生成与教师反馈流程，配置以所用版本为准。
- [88. 蒸馏学生交付项目](./88_Distilled_Student_Delivery_Project.md)：比较学生训练前后质量、模型文件大小与推理延迟。
