# 17. DPO Loss Tutorial | 直接偏好优化：从偏好数据到 DPO 的推导

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/17_DPO_Loss_Tutorial.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


模型能写出两种回答，并不意味着它更愿意写出人们想要的那一种。怎样利用“这条回答比那条好”的反馈，改变模型的选择？这就是本章的问题。

DPO（Direct Preference Optimization，直接偏好优化）给出了一个很简洁的损失。理解它，还需要回答几个问题：为什么要有参考模型？为什么比较两个对数概率比？奖励模型去了哪里？训练损失下降，是否真的意味着回答变好了？

本章讲解**原始的离线 DPO**：从回答概率和偏好数据出发，逐步推导目标函数，并解释它怎样利用偏好反馈更新模型。

阅读所需的基础是条件概率、对数、求导，以及模型参数与梯度的基本概念。Step 2 会在有限回答集合上完整推导最优策略；遇到 KL 散度时，我们也会先解释它在这里做什么。

Step 1 建立回答概率与偏好模型，Step 2 推导 KL 正则化下的最优策略，Step 3 得到并解释 DPO 损失。Step 4 只留一道代码填空：从逐位置的词表预测中，取出回答 token 的对数概率并求和，再用给定的 DPO 损失检验结果。

### Step 1：建立回答概率与偏好模型

**从“模仿好回答”到“学习两条回答的比较”**

假设用户问：“做梯度累积时，什么时候更新模型参数？”我们得到两条候选回答：

- **回答 A**：多个微批次分别反向传播，累积梯度，然后统一更新一次参数。
- **回答 B**：每个微批次反向传播后都更新参数，这样就是梯度累积。

在这个问题下，标注者选择 A。我们把一条记录写成

$$
(x,y_w,y_l),
$$

其中 $x$ 是输入，$y_w$ 是被选中的回答（chosen），$y_l$ 是另一条回答（rejected）。下标 $w,l$ 来自 winner、loser，只表示**这一次比较的结果**，不表示一条回答在所有场景都好，另一条在所有场景都坏。

监督微调（SFT）可以只拿 A 当示范，增加模型生成 A 的概率。偏好学习还希望利用另一半信息：**面对同一个问题，为什么 A 应当排在 B 前面？** 这个“相对于谁更好”的信息，不能仅靠把 chosen 当作普通示范就完整表达。

实践中，DPO 常从一个已经完成 SFT 的模型开始：复制它作为固定参考模型，再继续更新当前模型。SFT 负责先建立任务能力和输出习惯，DPO 根据偏好对调整选择。参考模型不必在数学上专指 SFT 模型；这里采用这一常见设置，是为了让偏好调整有一个明确的起点。

先记住一项数据要求：一对回答必须对应**同一个输入和同一套要求**。如果 A 回答的是一个问题，B 回答的是另一个问题，后面的成对比较就失去了原来的含义。

**一整条回答，怎样得到一个概率？**

**先逐 token 相乘，再取对数**

给定输入 $x$，自回归语言模型逐个预测回答 token。若回答为 $y=(y_1,\ldots,y_T)$，则

$$
\pi_\theta(y\mid x)
=\prod_{t=1}^{T}\pi_\theta(y_t\mid x,y_{<t}),
$$

所以整条回答的对数概率是

$$
\ell_\theta(x,y)
:=\log\pi_\theta(y\mid x)
=\sum_{t=1}^{T}\log\pi_\theta(y_t\mid x,y_{<t}). \tag{1}
$$

这里的 $\theta$ 表示当前语言模型的参数，$\pi_\theta(y\mid x)$ 是它生成整条回答的概率。$\ell_\theta(x,y)$ 是这个概率取对数后的简写，即**整条回答的对数概率**。在强化学习的说法里，语言模型也叫“策略”；本章中这两个词指向同一对象。

对于固定参考模型，同样记 $\ell_{\mathrm{ref}}(x,y):=\log\pi_{\mathrm{ref}}(y\mid x)$。下标 $\theta$ 和 $\mathrm{ref}$ 区分的是“由哪个模型来计算回答概率”。

**例 1：一条只有两个 token 的回答。** 假设模型对第一个正确 token 给出概率 $0.4$；读入这个 token 后，对第二个正确 token 给出概率 $0.5$。整条回答的概率是 $0.4\times0.5=0.2$，对数概率是 $\log0.4+\log0.5=\log0.2\approx-1.6094$。

代码中直接加对数，而不先乘很多小概率：这样既对应式 (1)，也能避免长序列概率乘积过小。

这里的 $T$ 指实际评分的回答 token 数。若把 EOS（结束符）作为完整回答的一部分，就应把它也计入。下面的算例将 EOS 计入回答。

**哪些 token 要计分？**

看一道具体的算术题：prompt 是 `1 + 1 =`，chosen 回答是 `2`，rejected 回答是 `3`。这里按空格分词，`<bos>` 表示序列起点，`<eos>` 表示回答结束，各占一个 token。

评分时，分别组成两条独立序列：

```text
chosen：  <bos> 1 + 1 = 2 <eos>
rejected：<bos> 1 + 1 = 3 <eos>
```

先看 chosen。我们要计算的是：**给定题目 `1 + 1 =`，模型接着生成 `2`，然后结束回答的概率。** 因此，需要取出下面两个概率：

| 模型已经读到的内容 | 接下来要预测的 token | 取出的条件概率 | 假设数值 |
|---|---|---|---:|
| `<bos> 1 + 1 =` | `2` | $\pi_\theta(2\mid\texttt{<bos> 1 + 1 =})$ | $0.6$ |
| `<bos> 1 + 1 = 2` | `<eos>` | $\pi_\theta(\texttt{<eos>}\mid\texttt{<bos> 1 + 1 = 2})$ | $0.8$ |

每一行中，模型都会给出整个词表的概率分布。第一行只取其中 `2` 的概率，第二行只取其中 `<eos>` 的概率。这些待评分的 token 来自数据中已有的回答；评分时按顺序读入它们，就能计算这条回答的概率。

于是，chosen 回答的概率是 $0.6\times0.8=0.48$，对数概率为

$$
\ell_\theta(x,y_w)
=\log0.6+\log0.8
=\log0.48\approx-0.734.
$$

这里的条件 $x$ 已经包含题目，所以只累加回答 `2` 和 `<eos>` 的对数概率。题目 token 为预测提供上下文；批处理补齐用的 padding 只负责对齐序列长度。

**这两个概率在代码中怎样取？** 对 chosen 这条输入，从 0 开始编号：

```text
位置：  0     1  2  3  4  5    6
输入： <bos>  1  +  1  =  2  <eos>
```

语言模型在每个位置的输出预测下一个 token，因此：

- **取 `2` 的概率**：使用位置 4 的输出。此时模型读到 `=`，预测位置 5 的 token `2`。
- **取 `<eos>` 的概率**：使用位置 5 的输出。此时模型读到 `2`，预测位置 6 的 token `<eos>`。

若 chosen 在 batch 的第 0 行，`logits[0, 4, :]` 就是第一项预测的词表 logits，做 softmax 后取 `2` 对应的概率。第二项同理，使用 `logits[0, 5, :]`，取 `<eos>` 对应的概率。

代码用 `logits[:, :-1]` 和 `input_ids[:, 1:]` 配对“当前位置的预测”与“下一个位置的 token”。`response_mask` 在回答所在的位置 5、6 标记为真；配对时使用 `response_mask[:, 1:]`，就能选出这两项预测，最后把取出的对数概率相加。

对 rejected 序列也独立评分：读到 `<bos> 1 + 1 =` 时取 `3` 的概率，读到 `<bos> 1 + 1 = 3` 时取 `<eos>` 的概率，再将两者的对数相加。这样得到的两个数，分别是模型生成 chosen、rejected 的对数概率。

**怎样把“更喜欢 A”写成一个概率模型？**

前面计算的是**语言模型生成一条回答的概率**。现在换一个问题：给标注者看同一道题的两条回答，**标注者选择其中一条的概率是多少？** 偏好数据记录的正是这种选择。

**Bradley–Terry：用得分解释成对选择**

Bradley–Terry 模型是一种**描述两两比较结果的概率模型**。可以先把它想成一场两人比赛：每个选手有一个正的“实力”数值，某人获胜的概率等于他的实力占两人总实力的比例。

用于回答比较时，两个“选手”就是同一输入下的两条回答。回到前面的梯度累积问题：

- **回答 A**：多个微批次分别反向传播，累积梯度，然后统一更新一次参数。
- **回答 B**：每个微批次反向传播后都更新参数，这样就是梯度累积。

我们用实数 $r(x,y)$ 表示回答 $y$ 在问题 $x$ 下的好坏，称为**奖励分数**。再用指数把它变成正的比较权重：

$$
s_A=e^{r(x,y_a)},\qquad s_B=e^{r(x,y_b)}.
$$

指数保证权重始终为正，并且分数越高，权重越大。Bradley–Terry 模型的假设是：**选择 A 的概率等于 A 的权重除以两条回答的权重之和。** 因而

$$
\begin{aligned}
P_r(y_a\succ y_b\mid x)
&=\frac{s_A}{s_A+s_B}\\
&=\frac{e^{r(x,y_a)}}{e^{r(x,y_a)}+e^{r(x,y_b)}}\\
&=\frac{1}{1+e^{-(r(x,y_a)-r(x,y_b))}}\\
&=\sigma\bigl(r(x,y_a)-r(x,y_b)\bigr).
\end{aligned} \tag{2}
$$

符号 $y_a\succ y_b$ 表示“标注者在这次比较中选择 A”，$\sigma(u)=1/(1+e^{-u})$ 称为 sigmoid 函数。它把任意实数映射到 $0$ 与 $1$ 之间，恰好可以表示概率。

**算一次具体的比较。** 假设上面两条回答的奖励分数分别是 $r(x,y_a)=2$、$r(x,y_b)=0$，则

$$
P_r(y_a\succ y_b\mid x)
=\frac{e^2}{e^2+e^0}
=\sigma(2)\approx0.881.
$$

也就是说，在这组假设分数下，模型为“标注者选择 A”分配约 $88.1\%$ 的概率。这里的 2 分和 0 分用于演示公式；实际的偏好数据只记录选择了哪条回答，奖励分数需要通过这些比较来学习。

从式 (2) 可以直接读出三个性质：

- **分数相同**：两条回答的权重相同，各有 $1/2$ 的选择概率。
- **A 的分数更高**：选择 A 的概率大于 $1/2$；分差越大，这个概率越接近 1。
- **两条回答同时加上相同分数**：分差保持不变，选择概率也保持不变。例如，$(2,0)$ 和 $(102,100)$ 给出相同的比较结果。

因此，成对偏好数据提供的是**同一输入下两条回答的相对好坏**。Bradley–Terry 用奖励差把这种相对关系写成概率；后面推导 DPO 时，我们正是要把这个奖励差表示出来。

这是一项关于偏好如何产生的建模假设。它用一套标量分数解释各条回答的比较结果；当数据中存在稳定的循环偏好，例如多数人选择 A 胜过 B、B 胜过 C、C 又胜过 A 时，这套分数对所有比较的拟合就会受到限制。

**一条偏好记录怎样形成损失？**

对于一条 $(x,y_w,y_l)$ 记录，标注者已经选择了 $y_w$。按照 Bradley–Terry 模型，这个已发生事件的概率为

$$
P_r(y_w\succ y_l\mid x)
=\sigma\bigl(r(x,y_w)-r(x,y_l)\bigr).
$$

训练的目标是让观测到的选择获得较高概率。为此，将它的负对数作为损失：

$$
L_r(x,y_w,y_l)
=-\log\sigma\bigl(r(x,y_w)-r(x,y_l)\bigr). \tag{3}
$$

例如，在上面的假设分数下，标注者选择 A 对应的损失是 $-\log0.881\approx0.127$；两条回答分数相同时，损失是 $-\log0.5\approx0.693$。增加被选中回答相对另一条回答的分差，就会提高该选择的预测概率，从而降低这条记录的损失。

把许多记录的损失取平均，就能训练一个显式奖励模型。典型的 PPO-based RLHF 还会用这个奖励模型提供反馈，再进行策略优化。

DPO 则希望**直接用语言模型表示式 (3) 中的奖励差，从偏好数据更新语言模型**。为此，我们先看看奖励与最优策略之间究竟有什么关系。

### Step 2：推导 KL 正则化下的最优策略

**给定奖励，什么样的策略才最优？**

**为什么引入参考模型？**

单纯最大化期望奖励，会使模型倾向于把概率集中到高奖励回答上。我们希望在提高奖励的同时保留模型原有的表达与任务能力，因此用一个固定模型作为参考，衡量回答分布的变化程度。

设固定参考策略为 $\pi_{\mathrm{ref}}$，考虑最大化下列目标：

$$
\begin{aligned}
J(\pi)
&=\mathbb E_{y\sim\pi(\cdot\mid x)}[r(x,y)]\\
&\quad-\beta D_{\mathrm{KL}}\bigl(\pi(\cdot\mid x)\Vert
\pi_{\mathrm{ref}}(\cdot\mid x)\bigr),\quad \beta>0.
\end{aligned} \tag{4}
$$

这里先固定一个输入 $x$，把符号减到最少。最后对数据中的输入求平均即可。

KL 散度在这里是

$$
D_{\mathrm{KL}}(\pi\Vert\pi_{\mathrm{ref}})
=\sum_y\pi(y\mid x)
\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}.
$$

KL 散度非负，在两个分布相同时为零。式 (4) 用它衡量当前回答分布相对参考分布的偏离：提高奖励会增加目标值，增大偏离则要付出 $\beta D_{\mathrm{KL}}$ 的代价。这里的期望由当前策略 $\pi$ 加权，比较的对象是两份模型给出的回答概率分布。

为了完整地算一次，我们先假设回答只有有限种，参考概率均为正，奖励有限。真实语言模型的回答空间更大，推导需要相应的支持与可归一化条件；我们暂时不让这些技术细节挡住主线。

**构造一个分布，把目标写成 KL**

我们希望找到一个分布，使它同时保留参考概率，并偏向高奖励回答。自然尝试

$$
\begin{aligned}
q_r(y\mid x)
&=\frac{\pi_{\mathrm{ref}}(y\mid x)e^{r(x,y)/\beta}}{Z(x)},\\
Z(x)&=\sum_y\pi_{\mathrm{ref}}(y\mid x)e^{r(x,y)/\beta}.
\end{aligned} \tag{5}
$$

分子中的指数使高奖励回答获得更大权重；$Z(x)$ 把所有权重加起来，使概率和等于 1。接下来把这个分布代入目标，验证它何时取得最大值。

对式 (5) 取对数，有

$$
\log\frac{\pi(y\mid x)}{q_r(y\mid x)}
=\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
-\frac{r(x,y)}{\beta}+\log Z(x).
$$

两边用 $\pi(y\mid x)$ 加权求和，再乘 $\beta$，得到

$$
\beta D_{\mathrm{KL}}(\pi\Vert q_r)
=\beta D_{\mathrm{KL}}(\pi\Vert\pi_{\mathrm{ref}})
-\mathbb E_\pi[r]+\beta\log Z(x).
$$

把项移回去，原目标恰好变成

$$
J(\pi)=\beta\log Z(x)-\beta D_{\mathrm{KL}}(\pi\Vert q_r). \tag{6}
$$

对固定奖励和参考模型，$Z(x)$ 不依赖我们正在优化的 $\pi$。KL 非负，因此当 $\pi=q_r$ 时目标达到最大值：

$$
\boxed{\displaystyle
\pi_r^*(y\mid x)
=\frac{\pi_{\mathrm{ref}}(y\mid x)e^{r(x,y)/\beta}}{Z(x)}.
} \tag{7}
$$

这里用到的 KL 非负性也能直接验证。对正数 $u$，有 $-\log u\geq1-u$，令 $u=q_r(y\mid x)/\pi(y\mid x)$，用 $\pi$ 加权求和，便得 $D_{\mathrm{KL}}(\pi\Vert q_r)\geq\sum_y(\pi(y\mid x)-q_r(y\mid x))=0$；零概率项用极限理解。

这就是所需的最优策略。我们证明的是**在允许任意回答分布的空间中**，式 (4) 的解长什么样；有限容量神经网络和有限步训练未必能到达这个解。

**用三个回答看懂 $\beta$**

设参考分布和奖励分别为

$$
\pi_{\mathrm{ref}}=(0.5,\,0.3,\,0.2),\qquad r=(0,\,1,\,2).
$$

代入式 (7)，三个回答的最优概率为

$$
\pi_r^*
=\frac{(0.5,\;0.3e^{1/\beta},\;0.2e^{2/\beta})}
{0.5+0.3e^{1/\beta}+0.2e^{2/\beta}}.
$$

取不同的 $\beta$，得到下表。概率保留三位小数，KL 保留四位小数。

| $\beta$ | 回答 1 的概率 | 回答 2 的概率 | 回答 3 的概率 | $D_{\mathrm{KL}}(\pi_r^*\Vert\pi_{\mathrm{ref}})$ |
|---:|---:|---:|---:|---:|
| 0.5 | 0.037 | 0.163 | 0.801 | 0.9155 |
| 1.0 | 0.179 | 0.292 | 0.529 | 0.3228 |
| 2.0 | 0.325 | 0.322 | 0.353 | 0.0835 |

这里奖励和参考分布保持固定。$\beta$ 较小时，奖励差经过 $1/\beta$ 放大，最优策略更集中在高奖励回答上；$\beta$ 较大时，奖励的影响减弱，最优策略更接近参考分布。

### Step 3：从最优策略得到 DPO 损失

**DPO 的关键：把奖励改写成策略的函数**

**反过来看式 (7)**

对最优策略的表达式取对数并整理：

$$
r(x,y)
=\beta\log\frac{\pi_r^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
+\beta\log Z(x). \tag{8}
$$

乍看并没有更容易：未知奖励变成了未知最优策略，还多了一个难算的 $Z(x)$。但我们真正需要的不是每条回答的奖励绝对值，而是**同一个输入下，两条回答的奖励差**。

将式 (8) 对 $y_w$ 和 $y_l$ 分别写一次，相减：

$$
\begin{aligned}
r(x,y_w)-r(x,y_l)
&=\beta\log\frac{\pi_r^*(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}\\
&\quad-\beta\log\frac{\pi_r^*(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}.
\end{aligned} \tag{9}
$$

**两项 $\beta\log Z(x)$ 消掉了。** 这也再次说明，为什么必须比较同一个输入下的回答：如果输入不同，归一化项一般就不能这样消去。

**从最优策略的关系，得到可训练的模型**

现在用参数化语言模型 $\pi_\theta$ 表示待学习的策略，定义

$$
\begin{aligned}
\Delta_\theta
&=\underbrace{\ell_\theta(x,y_w)-\ell_{\mathrm{ref}}(x,y_w)}_{\text{chosen 相对参考的变化}}\\
&\quad-\underbrace{\bigl(\ell_\theta(x,y_l)-\ell_{\mathrm{ref}}(x,y_l)\bigr)}_{\text{rejected 相对参考的变化}}.
\end{aligned} \tag{10}
$$

把式 (9) 代入 Step 1 的偏好模型，并拟合观察到的选择，得到

$$
\widehat P_\theta(y_w\succ y_l\mid x)
=\sigma(\beta\Delta_\theta),
$$

$$
\boxed{\displaystyle
L_{\mathrm{DPO}}(\theta)
=-\frac1N\sum_{i=1}^N
\log\sigma\bigl(\beta\Delta_\theta(x_i,y_{w,i},y_{l,i})\bigr).
} \tag{11}
$$

式 (11) 所需的量都能由 Policy、固定 Reference 和已有偏好对算出。偏好标签在训练前准备；每次更新时，模型对这些已有回答评分，再通过损失的梯度更新 Policy。这就构成了直接学习偏好的离线训练过程。

这个对应关系还可以从策略出发验证。对任意在参考支持内为正的归一化策略 $\pi_\theta$，定义 $r_\theta=\beta\log(\pi_\theta/\pi_{\mathrm{ref}})$，它在式 (5) 中的归一化常数就是

$$
Z_\theta(x)=\sum_y\pi_{\mathrm{ref}}(y\mid x)
\exp\!\left(\frac{r_\theta(x,y)}{\beta}\right)
=\sum_y\pi_\theta(y\mid x)=1.
$$

所以这个策略确实对应一组奖励，并且是那组奖励下式 (4) 的最优解。偏好数据的任务，则是从这些可由策略表示的奖励中，找出能解释观察选择的一组。

由此，策略参数成为偏好似然的参数。训练通过调整 $\theta$，使式 (11) 更好地解释已有标签。最终拟合效果取决于数据、模型表达能力和优化过程；式 (7) 描述的理论最优分布，为这项参数化提供了依据。

**“隐式奖励”是什么意思？**

常把

$$
\widehat r_\theta(x,y)
=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
$$

称为隐式奖励。这里的“隐式”是说：它由策略与参考策略共同定义，不需要另一个独立网络输出这个分数。

隐式奖励的数值由 Policy、Reference 和 $\beta$ 共同确定。对同一个输入，给所有奖励加同一个常数，成对偏好概率保持不变。因此，偏好数据识别的是同一输入下的奖励差。训练日志中的 chosen reward 可以用来观察相对参考模型的变化；实际回答质量则由独立评测衡量。

传统流程先拟合奖励再优化策略；DPO 用策略来参数化偏好模型，直接根据偏好标签更新策略。两种方式都以可靠的反馈数据为基础。

**三个算例：损失究竟在比较什么？**

**例 2：Policy 与 Reference 相同**

此时每条回答的 $\ell_\theta-\ell_{\mathrm{ref}}$ 都为零，故

$$
\Delta_\theta=0,\qquad
\widehat P_\theta=\tfrac12,\qquad
L=\log2\approx0.6931.
$$

即使 Reference 原本认为 chosen 比 rejected 更可能，结论也一样。DPO 的偏好概率来自**相对参考策略的变化**，不是原始生成概率之间的比较。

**例 3：先看四个对数概率**

假设 chosen 的 Policy、Reference 得分分别为 $-2,-3$，rejected 的两个得分分别为 $-5,-4$。chosen 相对参考上升了 1，rejected 相对参考下降了 1，因此

$$
\Delta=(-2+3)-(-5+4)=2.
$$

取 $\beta=0.1$，偏好 logit 为 $\beta\Delta=0.2$，于是

$$
\widehat P_\theta=\sigma(0.2)\approx0.5498,\qquad
L=-\log\sigma(0.2)\approx0.5981.
$$

请注意，这里的 $-2,-3$ 等都是**整条回答**的对数概率，不是单个位置的词表 logits。

**例 4：chosen 概率下降，损失也可能下降**

考虑同一个输入的完整回答分布：

| 回答 | Reference，也就是初始 Policy | 更新后的 Policy |
|---|---:|---:|
| chosen | 0.4 | 0.3 |
| rejected | 0.2 | 0.1 |
| 其他回答合计 | 0.4 | 0.6 |

chosen 的绝对概率从 $0.4$ 降到 $0.3$，但 rejected 降得更多。相对偏好间隔变为

$$
\Delta
=\log\frac{0.3}{0.4}-\log\frac{0.1}{0.2}
=\log1.5>0.
$$

初始间隔是零；对任意正 $\beta$，更新后的 DPO 损失更小。这是一个归一化、概率均为正的合法例子。

在这个例子中，损失随 chosen 相对 rejected 的概率比改善而下降，同时 chosen 的绝对概率也在下降。式 (10) 描述的正是这种成对的相对变化。其他回答的概率与实际生成结果，可以作为另外的观察量一起检查。

**梯度、参考模型与 $\beta$**

把一条记录的式 (11) 记为 $L(\Delta)=-\log\sigma(\beta\Delta)$。直接求导：

$$
\frac{\partial L}{\partial\Delta}
=-\beta\sigma(-\beta\Delta). \tag{12}
$$

设四个回答得分中的 Policy 两项为 $a=\ell_\theta(x,y_w)$、$b=\ell_\theta(x,y_l)$，Reference 固定。把 $a,b$ **暂时当作独立变量**，就有

$$
\frac{\partial L}{\partial a}=-\beta\sigma(-\beta\Delta),
\qquad
\frac{\partial L}{\partial b}=+\beta\sigma(-\beta\Delta).
$$

这两个偏导给出了损失对 chosen、rejected 得分的敏感性。若间隔已很大且为正，权重接近零；若排序方向很差，权重更大。对模型参数求梯度时，再通过链式法则将这些权重传递到两条回答的计算路径。

实际的 $a,b$ 由**同一套模型参数**产生，一次参数更新会同时影响多条回答。因此，更新效果应结合两项得分及其间隔来观察。前面的例 4 展示了其中一种情况：两条回答的绝对概率都下降，成对的相对间隔仍然增大。

**为什么 Reference 必须固定？**

式 (10) 中的 Reference 是基线，不是第二个待优化的偏好模型。标准离线 DPO 训练过程中不随每一步 Policy 一起更新它。否则，目标中“相对哪个分布”的含义也在变化。

代码中需要分清：

- `requires_grad_(False)` 冻结参考参数；
- `torch.no_grad()` 让参考前向不构造反向图；
- `eval()` 控制 dropout 等层的行为，**它本身不冻结参数**。

Reference 保留前向计算能力，持续给出固定基线下的回答得分。这些得分也可以预先计算并缓存，复用时保持模型版本、模板和评分 token 范围一致。

**$\beta$ 怎样影响目标与训练梯度？**

Step 2 讨论的是**固定奖励下的最优分布**。实际 DPO 训练还受到损失梯度影响。由式 (12)，当初始 Policy 等于 Reference 时，$\Delta=0$，于是

$$
\frac{\partial L}{\partial\Delta}=-\frac\beta2.
$$

在这一初始点，对间隔的偏导大小是 $\beta/2$，随 $\beta$ 增大。训练继续进行后，权重还取决于当前间隔，$\beta$ 也会改变 sigmoid 的饱和速度。Step 2 描述固定奖励下的最优分布，这里描述参数优化中的梯度；二者分别说明了 $\beta$ 在目标和训练过程中的作用。

式 (11) 通过偏好对计算损失。训练后整个回答分布相对 Reference 的实际 KL，属于另一项需要评估的分布性质。实践中，可以结合训练稳定性、留出偏好数据和实际生成质量选择 $\beta$。

**出处**

- Rafailov 等，*Direct Preference Optimization: Your Language Model is Secretly a Reward Model*：[原论文](https://arxiv.org/abs/2305.18290)，第 3—4 节与附录 A。
- [作者实现中的 trainers.py](https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py)：可对照序列对数概率和 DPO 损失的实现。

### Step 4：实现回答评分，并接入 DPO 损失

这道题只补全 `sequence_log_probs` 中的两行：**从每个位置的词表分布中取出目标 token 的对数概率，再把回答部分相加。** DPO 损失函数已给出，用来检验这份评分是否正确。

| TODO / 函数 | 学习者完成的机制 | 必须满足的约束 | 测试证据 |
| --- | --- | --- | --- |
| TODO 1 / `sequence_log_probs` | 选出已有回答中每个 token 的对数概率，并按回答掩码求和 | 预测与下一个 token 对齐；计入回答及 EOS；prompt 与 padding 不计分；每条回答返回一个总和 | 长度不同的回答、无关位置扰动、DPO 算例和梯度路径检查 |

**输入与输出**

设 batch 中有 $B$ 条偏好记录，共 $N=2B$ 条回答。前 $B$ 行放 chosen，后 $B$ 行按相同顺序放对应的 rejected；每一对回答有相同的 prompt。每一行都是一条独立的“prompt + 回答”序列。

`sequence_log_probs` 一次处理一个模型的输出：

| 参数或返回值 | 类型与形状 | 含义 |
| --- | --- | --- |
| `logits` | 浮点 `torch.Tensor`，`[N, L, V]` | 模型在各输入位置对下一个 token 给出的词表得分；$L$ 是补齐后的序列长度，$V$ 是词表大小 |
| `input_ids` | `torch.long`，`[N, L]` | 实际输入的 token ID，每个 ID 均在 $[0,V)$ 内 |
| `response_mask` | `torch.bool`，`[N, L]` | 按输入位置标记回答 token，含 EOS；prompt、padding 为假，首位置为假 |
| 返回值 | 浮点 `torch.Tensor`，`[N]` | 每一行回答的整条对数概率，即式 (1) 的 $\ell(x,y)$ |

三个输入位于同一设备，每行至少有一个计分 token。Policy 和 Reference 分别调用这个函数，就得到 $\ell_\theta$ 与 $\ell_{\mathrm{ref}}$。

**先把位置对齐**

沿用 Step 1 的例子，读到 `=` 的位置输出用于给下一个 token `2` 计分，读到 `2` 的位置输出用于给 `<eos>` 计分。骨架已完成这个错位配对：

- `log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)`：每个预测位置的词表对数概率，形状 `[N, L-1, V]`；
- `next_ids = input_ids[:, 1:]`：每个预测位置对应的目标 token，形状 `[N, L-1]`；
- `scored = response_mask[:, 1:]`：这些目标 token 中，哪些属于回答，形状 `[N, L-1]`。

**要补的两行对应什么公式？**

第一行取出目标 token 的对数概率：

$$
u_{i,j}=\texttt{log\_probs}[i,j,\texttt{next\_ids}[i,j]].
$$

第二行只沿回答位置求和：

$$
\ell_i=\sum_{j=0}^{L-2}\mathbf 1[\texttt{scored}[i,j]]\,u_{i,j}.
$$

其中 $i$ 表示第几条序列，$j$ 表示错位配对后的第几个预测位置。这里是对回答 token 求和；不同回答的长度可以不同。

可以用 `gather` 沿最后一维取目标值：先用 `unsqueeze(-1)` 将索引变成 `[N, L-1, 1]`，取值后用 `squeeze(-1)` 恢复 `[N, L-1]`。再用 `masked_fill` 将未计分位置置零，并沿序列维求和。

**给定的 DPO 损失如何使用这份评分？**

`dpo_loss` 接收 Policy、Reference 的两组 `[2B]` 评分和正的浮点数 `beta`，返回形状均为 `[B]` 的 `(losses, margins)`。例如，$B=2$ 时：

```text
输入评分顺序：[chosen_0, chosen_1, rejected_0, rejected_1]
组成偏好对：  (第 0 行, 第 2 行)、(第 1 行, 第 3 行)
输出顺序：    [偏好对 0 的结果, 偏好对 1 的结果]
```

`margins` 是式 (10) 中尚未乘 $\beta$ 的间隔，`losses` 是式 (11) 中每条记录的损失。训练时对 `losses` 取 batch 平均；回答内部的 token 对数概率仍按式 (1) 求和。


```python
import torch
import torch.nn.functional as F
```


```python
def sequence_log_probs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """[N, L, V] 预测 + [N, L] token 与回答掩码 -> [N] 回答对数概率之和。"""
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    next_ids = input_ids[:, 1:]
    scored = response_mask[:, 1:]

    # TODO 1：按上面两条公式，补全目标取值与掩码求和。
    token_log_probs = ...  # [N, L-1]，每个目标 token 的对数概率
    sequence_scores = ...  # [N]，只对回答 token 求和
    if token_log_probs is Ellipsis or sequence_scores is Ellipsis:
        raise NotImplementedError("请补全目标 token 取值与回答求和两行")
    return sequence_scores


def dpo_loss(
    policy_scores: torch.Tensor,
    reference_scores: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """输入评分 [2B]（先 chosen、后 rejected），返回 losses [B]、margins [B]。"""
    batch_size = policy_scores.shape[0] // 2
    changes = policy_scores - reference_scores.detach()
    margins = changes[:batch_size] - changes[batch_size:]
    losses = -F.logsigmoid(beta * margins)
    return losses, margins
```

**运行测试**

下面的评分样例把词表固定为 `<bos>, 1, +, =, 2, 3, <eos>, <pad>`，共 8 个 token。两条输入分别为：

```text
chosen：  <bos> 1 + 1 = 2   <eos> <pad>   计分 2 个 token
rejected：<bos> 1 + 1 = 3   3     <eos>   计分 3 个 token
```

当所有 logits 为零时，每个 token 的概率均为 $1/8$，两条回答的总对数概率应为 $-2\log8$、$-3\log8$。这能区分“整条回答求和”和“每 token 取平均”。随后检查 prompt、padding 和最后一个输出位置的扰动，以及评分进入 DPO 后的梯度。

两行尚未补全时，测试会抛出 `NotImplementedError`。补全后，所有断言都应通过。


```python
def test_dpo_scoring():
    # 词表：<bos>=0, 1=1, +=2, ==3, 2=4, 3=5, <eos>=6, <pad>=7。
    input_ids = torch.tensor([
        [0, 1, 2, 1, 3, 4, 6, 7],
        [0, 1, 2, 1, 3, 5, 5, 6],
    ])
    response_mask = torch.tensor([
        [False, False, False, False, False, True, True, False],
        [False, False, False, False, False, True, True, True],
    ])
    logits = torch.zeros(2, 8, 8, dtype=torch.float64)
    expected_scores = -logits.new_tensor([2.0, 3.0]) * logits.new_tensor(8.0).log()
    scores = sequence_log_probs(logits, input_ids, response_mask)
    assert scores.shape == (2,)
    torch.testing.assert_close(scores, expected_scores)

    # 把 chosen 的第一个目标 token 概率改为 2/9，其他位置保持 1/8。
    changed = logits.clone()
    changed[0, 4, 4] = logits.new_tensor(2.0).log()
    expected_changed = expected_scores.clone()
    expected_changed[0] = logits.new_tensor(2.0 / 9.0).log() - logits.new_tensor(8.0).log()
    torch.testing.assert_close(sequence_log_probs(changed, input_ids, response_mask), expected_changed)

    # 这些输出只预测 prompt、padding，或位于最后一个输入位置。
    changed[:, :4, 0] = 100.0
    changed[0, 6, 0] = 100.0
    changed[:, -1, 0] = 100.0
    torch.testing.assert_close(sequence_log_probs(changed, input_ids, response_mask), expected_changed)

    # Step 3 的两个算例：第一对 Policy = Reference，第二对间隔为 2。
    policy_scores = logits.new_tensor([-3.0, -2.0, -4.0, -5.0]).requires_grad_()
    reference_scores = logits.new_tensor([-3.0, -3.0, -4.0, -4.0]).requires_grad_()
    losses, margins = dpo_loss(policy_scores, reference_scores, beta=0.1)
    assert losses.shape == margins.shape == (2,)
    torch.testing.assert_close(margins, logits.new_tensor([0.0, 2.0]))
    torch.testing.assert_close(losses, logits.new_tensor([0.69314718056, 0.59813886938]))
    losses.mean().backward()
    assert reference_scores.grad is None

    # 从词表 logits 一直反传：chosen 的目标得分应上升，rejected 的应下降。
    policy_logits = logits.clone().requires_grad_()
    reference_logits = logits.clone().requires_grad_()
    policy_scores = sequence_log_probs(policy_logits, input_ids, response_mask)
    reference_scores = sequence_log_probs(reference_logits, input_ids, response_mask)
    losses, _ = dpo_loss(policy_scores, reference_scores, beta=0.1)
    losses.mean().backward()
    assert reference_logits.grad is None
    assert policy_logits.grad[0, 4, 4] < 0
    assert policy_logits.grad[1, 4, 5] > 0
    scored_predictions = response_mask[:, 1:]
    unscored_grad = policy_logits.grad[:, :-1, :][~scored_predictions]
    torch.testing.assert_close(unscored_grad, torch.zeros_like(unscored_grad))
    torch.testing.assert_close(policy_logits.grad[:, -1, :], torch.zeros_like(policy_logits.grad[:, -1, :]))
    print("回答总分、token 对齐、掩码、DPO 损失与梯度检查通过。")


test_dpo_scoring()
```

---

🛑 **STOP HERE** 🛑

请先补全两行并运行测试，下面是参考答案。

<br><br><br><br><br><br><br><br><br><br>
<br><br><br><br><br><br><br><br><br><br>

---

**参考代码与解析**

`gather` 的最后一维是词表维。`next_ids.unsqueeze(-1)` 为每个预测位置指定一个目标 token，取值后移去大小为 1 的末维，就得到每个位置的目标对数概率。

`scored` 按目标 token 所在的位置标记回答；将其余位置置零后，沿序列维求和，便得到式 (1) 的整条回答对数概率。EOS 在掩码内，因此参与求和。

接着，`dpo_loss` 把前、后半个 batch 配成 chosen / rejected，计算式 (10) 的相对间隔，再计算式 (11) 的负对数似然。Reference 的得分通过 `detach()` 固定，梯度只通过 Policy 的回答评分回到模型。


```python
def sequence_log_probs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """[N, L, V] 预测 + [N, L] token 与回答掩码 -> [N] 回答对数概率之和。"""
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    next_ids = input_ids[:, 1:]
    scored = response_mask[:, 1:]

    token_log_probs = log_probs.gather(dim=-1, index=next_ids.unsqueeze(-1)).squeeze(-1)
    sequence_scores = token_log_probs.masked_fill(~scored, 0.0).sum(dim=-1)
    return sequence_scores


def dpo_loss(
    policy_scores: torch.Tensor,
    reference_scores: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """输入评分 [2B]（先 chosen、后 rejected），返回 losses [B]、margins [B]。"""
    batch_size = policy_scores.shape[0] // 2
    changes = policy_scores - reference_scores.detach()
    margins = changes[:batch_size] - changes[batch_size:]
    losses = -F.logsigmoid(beta * margins)
    return losses, margins
```
