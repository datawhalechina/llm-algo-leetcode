# 量化与压缩公式索引

本页统一量化专题反复使用的符号、单位和约束。机制页负责解释公式为何成立、怎样进入代码和如何验证；本页用于跨页对照，不替代机制推导。

## 公共符号

| 符号 | 含义 | 单位或范围 |
|:---|:---|:---|
| $x, \hat{x}$ | 浮点原值与反量化近似值 | 与原张量一致 |
| $q$ | 量化码 | $[q_{\min},q_{\max}]$ 内的整数 |
| $s,z$ | 缩放因子与零点 | $s>0$， $z$ 为整数码 |
| $b,g$ | 位宽与分组大小 | bit，元素数 |
| $N$ | 被量化元素数 | 元素数 |
| $\mathcal C$ | 校准范围候选集合 | — |

## 表示、校准与误差

对称量化以零为中心：

以下量化式假定缩放因子为正、非对称输入范围具有非零宽度；全零张量或常量张量需要单独约定正的最小缩放值或常量编码方式。

$$
s=\frac{\max_i|x_i|}{q_{\max}},\qquad
q_i=\mathrm{clip}\left(\mathrm{round}\frac{x_i}{s},q_{\min},q_{\max}\right),\qquad
{\hat{x}}_i=sq_i.
$$

非对称量化用零点覆盖偏移区间：

$$
s=\frac{x_{\max}-x_{\min}}{q_{\max}-q_{\min}},\quad
z=\mathrm{clip}\left(\mathrm{round}\left(q_{\min}-\frac{x_{\min}}s\right),q_{\min},q_{\max}\right),
$$

$$
q_i=\mathrm{clip}\left(\mathrm{round}\frac{x_i}{s}+z,q_{\min},q_{\max}\right),\qquad
{\hat{x}}_i=s(q_i-z).
$$

校准需要同时观察重建误差和饱和比例：

$$
\mathrm{MSE}(c)=\frac1N\sum_{i=1}^N(x_i-{\hat{x}}_i(c))^2,\qquad \rho_{\mathrm{sat}}(c)=\frac1N\sum_{i=1}^N\mathbf 1(|x_i|>c),
$$

$$
c^{\ast}=\arg\min_{c\in\mathcal C}\mathrm{MSE}(c)
\quad\text{s.t.}\quad \rho_{\mathrm{sat}}(c)\le\tau.
$$

## 训练期低比特适配

伪量化在前向中执行量化—反量化，浮点主权重仍由优化器更新：

$$
\widetilde{x}=s\left(\mathrm{clip}\left(\mathrm{round}\frac{x}{s}+z,q_{\min},q_{\max}\right)-z\right),
\qquad
\frac{\partial\widetilde{x}}{\partial x}\approx\mathbf 1(x_{\min}\le x\le x_{\max}).
$$

QLoRA 冻结低比特底座，只训练低秩增量：

$$
y=x\widehat W^{\mathsf T}+\frac\alpha r(xA^{\mathsf T})B^{\mathsf T}.
$$

其中 $\widehat W$ 是反量化后的底座近似， $A,B$ 是秩为 $r$ 的适配器参数， $\alpha$ 是缩放系数。

## 权重、激活与位宽分配

令 $\Delta w$ 为量化前后的权重扰动， $H$ 为校准输入估计的局部曲率或二阶代理， $\Delta\mathcal L$ 为对应的损失变化近似。GPTQ 类方法用曲率加权误差近似扰动影响：

$$
\Delta\mathcal L\approx\frac12\Delta w^{\mathsf T}H\Delta w.
$$

令 $x_j$ 和 $W_j$ 分别表示输入通道 $j$ 的激活与权重， $s_j>0$ 为该通道的平滑系数， $\alpha\in[0,1]$ 控制动态范围从激活侧迁移到权重侧的程度。SmoothQuant 用逐通道正缩放保持线性层等价：

$$
xW=(x\mathrm{diag}(s)^{-1})(\mathrm{diag}(s)W),
\qquad s_j=\frac{\max|x_j|^\alpha}{\max|W_j|^{1-\alpha}}.
$$

令 $l$ 为层编号， $\mathcal B$ 为可用位宽集合， $E_l(b_l)$ 为第 $l$ 层采用位宽 $b_l$ 时的误差代理， $N_l$ 为该层元素数， $B_{\mathrm{bit}}$ 为总 bit 预算。混合位宽是在预算下分配离散位宽：

$$
\min_{b_l\in\mathcal B}\sum_lE_l(b_l)
\quad\text{s.t.}\quad \sum_lN_lb_l\le B_{\mathrm{bit}}.
$$

## 运行期状态与容量

对 FP8 或其他有限范围格式，令 $f_{\max}$ 为可表示上限， $s$ 为映射缩放因子， $x_{\mathrm{low}}$ 为缩放后转换到低精度格式的张量。缩放需让绝对最大值落入格式范围：

$$
s=\frac{\max_i|x_i|}{f_{\max}},\qquad x_{\mathrm{low}}=\mathrm{cast}(x/s).
$$

忽略量化元数据时，单层 KV Cache 主体容量可近似为

$$
M_{\mathrm{KV}}\approx2BLH_{\mathrm{kv}}D_h\frac b8.
$$

其中 $B,L,H_{\mathrm{kv}},D_h$ 分别是批大小、缓存长度、KV Head 数和 Head 维度；系数 2 对应 Key 与 Value。全模型容量需对所有缓存层求和；各层形状和位宽相同时，再乘以层数。

## 教师—学生蒸馏扩展

蒸馏不改变量化表示，而是用教师反馈训练独立学生模型。令固定轨迹分布为 $\mu_{\mathrm{off}}(y\mid x)$，当前学生策略为 $\pi_S(y\mid x)$， $\lambda\in[0,1]$ 是学生在策略轨迹比例，则训练轨迹来自

$$
\mu_{\lambda}(y\mid x)
=(1-\lambda)\mu_{\mathrm{off}}(y\mid x)
+\lambda\pi_S(y\mid x).
$$

令同一 Token 位置的教师和学生分布为 $p_T,p_S$。当 $\beta\in(0,1)$ 时，混合分布和广义 Jensen–Shannon 散度为

$$
m_{\beta}=\beta p_T+(1-\beta)p_S,
$$

$$
D_{\mathrm{GJS}}^{\beta}(p_T,p_S)
=\beta D_{\mathrm{KL}}(p_T\Vert m_{\beta})
+(1-\beta)D_{\mathrm{KL}}(p_S\Vert m_{\beta}).
$$

端点单独定义： $\beta=0$ 使用前向 $D_{\mathrm{KL}}(p_T\Vert p_S)$， $\beta=1$ 使用反向 $D_{\mathrm{KL}}(p_S\Vert p_T)$。其中 $\lambda$ 控制轨迹来源， $\beta$ 控制教师—学生分布的比较方式；二者不能互换。具体的完成区掩码、有效 Token 归约和梯度边界见 [DISTILL-OPD 在策略蒸馏与 GKD](../../02_PyTorch_Algorithms/DISTILL-OPD_On_Policy_Distillation_and_GKD.md)。

## 使用边界

- 公式给出数值与容量关系，不证明目标后端具有对应低精度计算内核。
- 校准误差不能替代任务质量；机制页应同时记录校准集、验证集和量化参数。
- 元数据、打包对齐、临时反量化缓冲区和后端回退会使真实容量偏离理想估算。
