# 后训练优化公式索引与符号约定

本页汇总监督、偏好、奖励、轨迹和教师—学生训练中使用的公式，说明计算对象、有效位置和归约分母。各机制页解释推导与验证；梯度、累计窗口和有效更新见[反向传播与训练机制公式索引](../backpropagation_training_mechanism/formula_reference.md)，蒸馏分布目标见[在策略蒸馏与 GKD 机制](../../02_PyTorch_Algorithms/DISTILL-OPD_On_Policy_Distillation_and_GKD.md)。

## 使用方法

阅读公式时依次确认四件事：计算对象是 Token、回答、偏好对还是候选组；哪些位置由掩码排除；先在哪个维度求和；最后使用什么分母。本页负责后训练目标及其归约粒度；反向传播专题负责这些标量目标怎样形成梯度并提交更新。

## 共用符号

| 符号 | 含义 | 常见粒度 |
|:---|:---|:---|
| $x$ / $q$ | 输入序列或提示 | 样本 |
| $y$ | 回答；$y^+$、$y^-$ 分别表示优选与非优选回答 | 回答 |
| $b,t$ | 批次索引与序列位置 | Token |
| $m_{b,t}$ | 有效目标或动作掩码，取值为 0 或 1 | Token |
| $\pi_\theta$ | 当前可训练策略 | Token 条件分布 |
| $\pi_{\mathrm{ref}}$ | 冻结参考策略 | Token 条件分布 |
| $\pi_{\mathrm{old}}$ | 生成当前轨迹或更新前保存的旧策略 | Token 条件分布 |
| $r_t,G_t,V_t,A_t$ | 即时奖励、折扣回报、状态价值与优势；部分机制页把回报记为 $R_t$ | 时间步 / Token |
| $T,\gamma,\lambda$ | 奖励序列的最后一个有效位置、折扣因子与 GAE 衰减系数；后两者取值为 $[0,1]$ | 有效动作序列 |
| $\sigma(z),\mathbf1(E)$ | Sigmoid 函数 $1/(1+e^{-z})$；条件 $E$ 成立时为 1、否则为 0 | 标量 |
| $c$ | 二元反馈标签；1 表示 desirable，0 表示 undesirable | 回答 |
| $g,i,K$ | 提示组、组内回答索引与每组回答数 | 回答组 |
| $S_g$ | 提示组 $g$ 中可评分回答的索引集合 | 回答组 |
| $z_{KL}$ | KTO 从独立 KL 样本估计并停止梯度的非负基线 | 批次 |
| $\beta$ | 参考约束或偏好目标的系数，含义由所属目标确定 | 目标函数 |
| $\epsilon,\epsilon_l,\epsilon_h$ | 对称、下侧和上侧裁剪范围；$\varepsilon>0$ 单独表示数值稳定项 | 目标函数 |

## 监督微调与参数适配

自回归损失先选定要预测的位置，再按有效监督词元归约。下表的 $x_{1:T}$ 是长度为 $T$ 的序列，$x_{\le0}$ 表示起始上下文；$b$ 是样本索引。LoRA 中 $W_0$ 是冻结权重，$W'$ 是适配后的权重，$d_{in},d_{out}$ 是输入、输出维度，$r$ 是低秩维度，$\alpha$ 是缩放参数。长度统计中 $L_i$ 为第 $i$ 条样本的长度，$N>0$ 为样本数，$C$ 为窗口上限，$\#$ 表示计数。

| 机制 | 规范表达 | 粒度与归约 | 正文与代码观察量 |
|:---|:---|:---|:---|
| 自回归分解 | $p_\theta(x_{1:T})=\prod_{t=0}^{T-1}p_\theta(x_{t+1}\mid x_{\le t})$ | 位置 $t$ 的输出预测位置 $t+1$ | [13 SFT 监督边界与损失](../../02_PyTorch_Algorithms/13_SFT_Training_Loop.md)：`shift_logits`、`shift_labels` |
| SFT 有效目标损失 | $L_{\mathrm{SFT}}=-\dfrac{\sum_{b,t}m_{b,t}\log p_\theta(x_{b,t+1}\mid x_{b,\le t})}{\sum_{b,t}m_{b,t}}$ | 有效词元总数必须大于 0；先汇总分子和分母，再统一归约 | [13 SFT 监督边界与损失](../../02_PyTorch_Algorithms/13_SFT_Training_Loop.md)：`loss_sum`、`valid_token_count` |
| LoRA 权重增量 | $W'=W_0+sBA$，其中 $A\in\mathbb{R}^{r\times d_{in}}$、$B\in\mathbb{R}^{d_{out}\times r}$ | 线性模块；标准 LoRA 常用 $s=\alpha/r$ | [14 LoRA](../../02_PyTorch_Algorithms/14_LoRA_Tutorial.md)：可训练参数、旁路输出、合并误差 |
| LoRA 参数量 | $N_{\mathrm{LoRA}}=r(d_{in}+d_{out})$ | 按目标模块逐项累计 | [31 LoRA 变体](../../02_PyTorch_Algorithms/31_LoRA_Variants_Theory.md)：`trainable_params`、`parameter_budget` |
| 长度溢出率 | $r_{\mathrm{overflow}}=\#\{L_i>C\}/N$ | 样本级统计，$C$ 为目标窗口 | [30 长上下文微调](../../02_PyTorch_Algorithms/30_Long_Context_Fine_Tuning.md)：长度分布、监督保留量 |

## 偏好、奖励与策略更新

奖励先确定反馈方向，再通过回报和基线形成优势。以下奖励序列包含终局位置 $T$，真实终止后的价值 $V_{T+1}=0$；因时间上限截断而未终止时，是否使用后续价值由采集协议决定。$r_\theta(x,y)$ 是奖励模型分数，$\Delta r_i$ 是成对分数差，$z_i\in\{0,1\}$ 是偏好标签，$M>0$ 是校准对数；校准温度另记为 $\tau>0$。

策略对数概率均在同一回答或动作及其上下文上计算。KTO 中 $s_\theta,s_{\mathrm{ref}}$ 是当前回答的序列对数概率，$d_j^{\mathrm{KL}}$ 是第 $j$ 条独立 KL 样本的序列偏移，$B_{\mathrm{KL}}>0$ 为其数量。GRPO 中 $\mu_g,\sigma_g$ 是组内奖励均值、标准差；$s_{i,t}$ 是 PPO 形式的裁剪增益，$\widehat D_{\mathrm{KL},i,t}$ 是参考约束估计量。$\operatorname{clip}$ 将输入裁剪到给定区间，$\mathbb E$ 表示按声明的有效位置或样本分布取均值。

| 机制 | 规范表达 | 粒度与归约 | 正文与代码观察量 |
|:---|:---|:---|:---|
| 成对奖励目标 | $L_{\mathrm{RM}}=-\log\sigma(r_\theta(x,y^+)-r_\theta(x,y^-))$ | 同一提示内的回答对 | [奖励模型](../../02_PyTorch_Algorithms/PT-REWARD-MODEL_Reward_Model_Training_and_Calibration.md)：分数、边际、成对损失 |
| 奖励概率校准 | $\hat p_i(\tau)=\sigma(\Delta r_i/\tau)$，$\operatorname{Brier}(\tau)=M^{-1}\sum_i(\hat p_i(\tau)-z_i)^2$ | 独立留出偏好对；正温度只调整置信度，不改变分数排序 | [奖励模型](../../02_PyTorch_Algorithms/PT-REWARD-MODEL_Reward_Model_Training_and_Calibration.md)：温度、Brier 误差、隐藏质量 |
| 受约束 Token 奖励 | $r_t=m_t[\mathbf1(t=T)r^{task}-\beta_{KL}(\log\pi_\theta-\log\pi_{ref})]$ | 有效动作 Token；终局任务反馈只写入终止位置 | [奖励信号](../../02_PyTorch_Algorithms/PT-REWARD-SIGNAL_Reward_Model_Value_and_Advantage.md)：任务反馈、策略偏移、动作掩码 |
| 折扣回报 | $G_t=\sum_{k=t}^{T}\gamma^{k-t}r_k=r_t+\gamma G_{t+1}$，$G_{T+1}=0$ | 包含终局奖励；填充、提示及非动作位置不进入有效序列 | [奖励信号](../../02_PyTorch_Algorithms/PT-REWARD-SIGNAL_Reward_Model_Value_and_Advantage.md)：Token 奖励、回报 |
| GAE | $\delta_t=r_t+\gamma V_{t+1}-V_t$，$A_t=\sum_{l=0}^{T-t}(\gamma\lambda)^l\delta_{t+l}$ | $\delta_t$ 为时序差分残差；在真实终止点停止累计，不跨越轨迹边界 | [奖励信号](../../02_PyTorch_Algorithms/PT-REWARD-SIGNAL_Reward_Model_Value_and_Advantage.md) / [16 PPO](../../02_PyTorch_Algorithms/16_RLHF_PPO_Memory.md)：TD 残差、优势、价值目标 |
| DPO | $z=\beta[(\log\pi_\theta(y^+\mid x)-\log\pi_{\mathrm{ref}}(y^+\mid x))-(\log\pi_\theta(y^-\mid x)-\log\pi_{\mathrm{ref}}(y^-\mid x))]$<br>$L_{\mathrm{DPO}}=-\log\sigma(z)$ | 先在回答 Token 范围聚合对数概率，再对偏好对求损失 | [17 DPO](../../02_PyTorch_Algorithms/17_DPO_Loss_Tutorial.md)：回答 log-prob、隐式奖励、偏好边际 |
| KTO 二元反馈目标 | $d_\theta=s_\theta-s_{\mathrm{ref}}$<br>$z_{\mathrm{KL}}=\max(0,B_{\mathrm{KL}}^{-1}\sum_j d_j^{\mathrm{KL}})$<br>$\ell^+=1-\sigma(\beta(d_\theta-z_{\mathrm{KL}})),\quad \ell^-=1-\sigma(\beta(z_{\mathrm{KL}}-d_\theta))$ | 当前回答与独立 KL 样本分开；$z_{\mathrm{KL}}$ 停止梯度，正负样本权重和数量分别记录 | [PT-KTO](../../02_PyTorch_Algorithms/PT-KTO_Binary_Feedback_Alignment.md)：回答聚合、独立 KL 基线、二元目标与奖励诊断 |
| PPO 策略比率与裁剪目标 | $\rho_t=\exp(\log\pi_\theta-\log\pi_{old})$，$L_t^{clip}=\min(\rho_tA_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t)$ | 最大化该增益；最小化损失时使用 $-\mathbb E_t[L_t^{clip}]$；旧概率与优势停止梯度 | [16 PPO](../../02_PyTorch_Algorithms/16_RLHF_PPO_Memory.md)：策略比率、裁剪目标、裁剪比例 |
| PPO 价值目标 | $L_V=\frac12\mathbb E_t[(V_\phi(s_t)-G_t)^2]$ | $\phi$ 为价值模型参数，$s_t$ 为状态；回报目标在价值模型更新中固定 | [16 PPO](../../02_PyTorch_Algorithms/16_RLHF_PPO_Memory.md)：策略、价值与状态账本 |
| GRPO 组内优势 | $A_i=(r_i-\mu_g)/\max(\sigma_g,\varepsilon)$ | 同一提示的候选组；必须记录使用总体标准差还是样本标准差；零方差组不应制造方向 | [奖励信号](../../02_PyTorch_Algorithms/PT-REWARD-SIGNAL_Reward_Model_Value_and_Advantage.md) / [18 GRPO](../../02_PyTorch_Algorithms/18_GRPO_Loss_Tutorial.md)：组均值、组标准差、候选优势 |
| GRPO 受约束目标 | $L=\dfrac{1}{K}\sum_{i=1}^{K}\dfrac{\sum_t m_{i,t}[-s_{i,t}+\beta\widehat D_{\mathrm{KL},i,t}]}{\sum_t m_{i,t}}$ | $K$ 个候选分别按有效回答 Token 求均值，再在组内等权；$s_{i,t}$ 是裁剪后的策略增益 | [18 GRPO](../../02_PyTorch_Algorithms/18_GRPO_Loss_Tutorial.md)：裁剪目标、Reference KL、候选等权 |
| RLOO 奖励塑形 | $R_i=r_i^{ext}-\beta\sum_t m_{i,t}(\log\pi_{old}-\log\pi_{ref})$ | KL 在回答 Token 上求和后进入奖励；不可评分回答保持缺失 | [PT-RLOO](../../02_PyTorch_Algorithms/PT-RLOO_Leave_One_Out_Policy_Optimization.md)：外部奖励、Reference KL 与有效回答范围 |
| RLOO 留一优势 | $b_{g,-i}=(\lvert S_g\rvert-1)^{-1}\sum_{j\in S_g,j\ne i}R_{g,j}$，$A_{g,i}=R_{g,i}-b_{g,-i}$ | $R_{g,i}$ 为塑形后回答奖励；要求 $\lvert S_g\rvert\ge2$。不足时不计算该式，练习将优势置零并记录无有效比较 | [PT-RLOO](../../02_PyTorch_Algorithms/PT-RLOO_Leave_One_Out_Policy_Optimization.md)：组边界、缺失奖励与留一优势 |
| RLOO 序列裁剪目标 | $\rho_i=\exp[\log\pi_\theta(y_i\mid x_i)-\log\pi_{\mathrm{old}}(y_i\mid x_i)]$<br>$L_i=-\min[\rho_iA_i,\operatorname{clip}(\rho_i,1-\epsilon_l,1+\epsilon_h)A_i]$ | 先聚合回答 Token 得到序列得分；旧策略与优势停止梯度 | [PT-RLOO](../../02_PyTorch_Algorithms/PT-RLOO_Leave_One_Out_Policy_Optimization.md)：序列比率、非对称裁剪与梯度隔离 |

## 评测、在线治理与系统账本

先按样本数计算覆盖和胜率，再将阶段吞吐换算到同一工作单位。$n_{expected},n_{evaluated}$ 是计划、完成评测数，$n_{candidate},n_{baseline}$ 是两侧获胜数；各分母必须大于 0，无有效判决时记为缺失。$j_i^{(1)},j_i^{(2)}$ 是同一对回答的两次判分，$\widetilde j_i^{(swap)}$ 是交换位置后还原到原方向的判分，$N$ 是参与一致性比较的样本数。

在线候选中，下标 $c,b$ 表示候选、基线，$W,R,S,C$ 分别表示胜率、重复性、安全和成本指标，具体定义随固定评测协议记录。版本差 $\Delta v$ 仅用于可排序的更新计数；动作 $a_t$、状态 $s_t$ 与掩码 $m_t$ 用于同一冻结轨迹的概率比较。

系统阶段 $j$ 的原始处理率为 $q_j$，每个公共工作单位在该阶段展开 $m_j>0$ 个对象，$\lambda$ 是公共单位的到达率。边 $e$ 上单对象大小为 $z_e$ 字节，倍率为 $m_e$，到达率为 $\lambda_e$，链路带宽为 $B_e>0$ 字节/秒，重叠比例为 $o_e\in[0,1]$；$f$ 是共享互连域，$b_e$ 为字节率，$u_e,x_e,X_f$ 为无量纲占用估计，不是端到端秒数。

| 机制 | 规范表达 | 粒度与归约 | 正文与代码观察量 |
|:---|:---|:---|:---|
| 偏好评测覆盖与胜率 | $C_{eval}=n_{evaluated}/n_{expected}$，$R_{decision}=(n_{candidate}+n_{baseline})/n_{evaluated}$，$W_{non\text{-}tie}=n_{candidate}/(n_{candidate}+n_{baseline})$ | 偏好对；平局属于已完成评测，但不进入非平局胜率分母 | [50 偏好数据与评测](../../02_PyTorch_Algorithms/50_Preference_Data_and_Evaluation.md)：计划数、完成数、胜负平与明确判决数 |
| 判分一致性 | $C_{repeat}=N^{-1}\sum_i\mathbf1(j_i^{(1)}=j_i^{(2)})$，$C_{swap}=N^{-1}\sum_i\mathbf1(j_i^{(1)}=\widetilde j_i^{(swap)})$，$C_{judge}=\min(C_{repeat},C_{swap})$ | 同一批偏好对；交换回答位置后的判分先还原到原方向 | [50 偏好数据与评测](../../02_PyTorch_Algorithms/50_Preference_Data_and_Evaluation.md)：重复一致性、交换一致性与保守一致性 |
| 在线候选变化量 | $\Delta W=W_c-W_b$，$\Delta R=R_c-R_b$，$\Delta S=S_c-S_b$，$\rho_C=(C_c-C_b)/C_b$ | 同一冻结评测协议下的候选与基线；$C_b>0$ | [51 在线 DPO](../../02_PyTorch_Algorithms/51_Online_DPO.md)：质量、重复性、安全与成本门槛 |
| 策略版本与行为偏离 | $\Delta v=v_{\mathrm{current}}-v_{\mathrm{behavior}}$，$D_{\mathrm{beh}}=\dfrac{\sum_t m_t\lvert\log\pi_{\mathrm{current}}(a_t\mid s_t)-\log\pi_{\mathrm{behavior}}(a_t\mid s_t)\rvert}{\sum_t m_t}$ | 单条轨迹的有效动作位置；版本差和行为偏离必须同时检查 | [轨迹生命周期](../../02_PyTorch_Algorithms/PT-ROLLOUT-LIFECYCLE_Rollout_and_Policy_Freshness.md)：版本状态、动作位置平均偏离与准入原因 |
| 流水线容量与队列余量 | $c_j=q_j/m_j$，$C_{pipe}=\min_j c_j$，$s_j=c_j-\lambda$ | 所有阶段先换算到同一工作单位；$s_j<0$ 表示队列增长 | [RL 系统](../../02_PyTorch_Algorithms/PT-RL-SYSTEM_Rollout_Training_and_Reward_System.md)：阶段等价处理率、瓶颈与队列余量 |
| 跨池传输暴露 | $b_e=z_em_e\lambda_e$，$u_e=b_e/B_e$，$x_e=u_e(1-o_e)$，$X_f=\sum_{e\in f}x_e$ | 边级载荷先按共享互连域聚合；不同互连域不直接串行相加 | [RL 系统](../../02_PyTorch_Algorithms/PT-RL-SYSTEM_Rollout_Training_and_Reward_System.md)：字节率、带宽占用、暴露占用与关键互连域 |

## Agent 环境轨迹与奖励归因

这里将 $T$ 统一为最后一个动作位置：动作 $a_t$ 在状态 $s_t$ 下执行，返回观察 $o_{t+1}$ 和奖励 $r_t$，最终状态为 $s_{T+1}$。任务、过程、安全与成本分量分别记为 $r_t^{task},r_t^{process},r_t^{safety},r_t^{cost}$，后两项按非负惩罚记录；回报沿有效动作序列累计。

| 机制 | 规范表达 | 粒度与归约 | 正文与代码观察量 |
|:---|:---|:---|:---|
| 多步环境轨迹 | $\tau=(s_0,a_0,o_1,r_0,\ldots,s_T,a_T,o_{T+1},r_T,s_{T+1})$ | 成功、失败、超时和截断分别记录；观察与动作保留不同来源 | [Agentic RL 机制页](../../02_PyTorch_Algorithms/PT-AGENTIC-RL_Environment_Trajectory_and_Credit_Assignment.md)：环境版本、动作、观察、奖励与终止原因 |
| 奖励分量与轨迹回报 | $r_t=r_t^{task}+r_t^{process}-r_t^{safety}-r_t^{cost}$，$G_t=\sum_{k=t}^{T}\gamma^{k-t}r_k$ | 奖励只归因到有效模型动作；工具观察保留为状态证据，不冒充策略动作 | [Agentic RL 机制页](../../02_PyTorch_Algorithms/PT-AGENTIC-RL_Environment_Trajectory_and_Credit_Assignment.md)：分量账本、动作掩码与折扣回报 |

## 教师反馈与独立学生训练

蒸馏先确定谁生成回答，再确定教师怎样监督这些位置。$\mu_{off}$ 是冻结轨迹分布，$\pi_S$ 是当前学生策略，$\lambda_{on}\in[0,1]$ 是学生在线轨迹比例；它与 GAE 的 $\lambda$ 不同。$d_t$ 是同一上下文位置的教师—学生分布差异，$m_t$ 标记有效回答词元。教师参数冻结；采样得到的离散轨迹按常量使用，梯度通过学生分布计算。

| 机制 | 规范表达 | 粒度与归约 | 正文与代码观察量 |
|:---|:---|:---|:---|
| 轨迹来源 | $\mu_{\lambda_{on}}=(1-\lambda_{on})\mu_{off}+\lambda_{on}\pi_S$ | 固定轨迹与当前学生轨迹混合；教师反馈来源与轨迹生成者分别记录 | [DISTILL-OPD](../../02_PyTorch_Algorithms/DISTILL-OPD_On_Policy_Distillation_and_GKD.md)：轨迹来源、学生版本、冻结教师 |
| 回答位置归约 | $L_{KD}=\dfrac{\sum_t m_t d_t}{\sum_t m_t}$ | 有效位置数大于 0；跨样本采用词元等权还是样本等权需单独声明 | [DISTILL-OPD](../../02_PyTorch_Algorithms/DISTILL-OPD_On_Policy_Distillation_and_GKD.md)：回答掩码、差异分子与分母 |
| KL 方向与 GKD 分布目标 | 前向、反向 KL 与 GJS 的定义见[在策略蒸馏与 GKD 机制](../../02_PyTorch_Algorithms/DISTILL-OPD_On_Policy_Distillation_and_GKD.md) | 分布目标、轨迹混合比例和训练步数分别控制；名称相同不足以证明配置一致 | [Part 02 · 88 蒸馏学生交付](../../02_PyTorch_Algorithms/88_Distilled_Student_Delivery_Project.md)：共同学生起点、路线与独立评测 |

## 容易混淆的计算单位与归约方式

| 容易混淆的对象 | 区别 |
|:---|:---|
| Attention mask 与目标 / 动作掩码 | 前者控制模型可见位置，后者决定哪些位置进入训练目标。 |
| Token 平均与样本平均 | 回答长度不同时，两种分母会产生不同的损失和梯度权重。 |
| 参考策略与旧策略 | 参考策略约束长期偏离；旧策略记录当前轨迹来自哪个行为分布。 |
| 采样 Token 上的对数概率差与完整 KL | 前者是已采样动作上的估计，不能直接称为完整词表分布的精确 KL。 |
| 候选级优势与 Token 级策略比率 | GRPO 可把一个候选优势应用到该回答的有效 Token，但二者粒度不同。 |
