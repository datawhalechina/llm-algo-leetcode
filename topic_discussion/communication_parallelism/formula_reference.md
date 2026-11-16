# 通信与并行公式索引

本页统一通信与并行专题反复使用的符号、单位和理想化成本模型。机制页负责解释切分语义、执行顺序和验证方法；本页用于跨页核对。

## 公共符号

| 符号 | 含义 | 单位 |
|:---|:---|:---|
| $P$ | 参与某进程组的进程数 | 个 |
| $n$ | 单次集合通信的有效载荷 | byte |
| $\alpha,\beta$ | 单次启动延迟、每 byte 传输时间 | s，s/byte |
| $T_{\mathrm{comp}},T_{\mathrm{comm}}$ | 计算时间、通信时间 | s |
| $T_{\mathrm{step}}$ | 全局训练或推理步骤时间 | s |
| $\Phi$ | 参数量 | 个 |
| $b_p,b_g,b_o$ | 每参数的参数、梯度、优化器状态字节数 | byte/参数 |

## 集合通信成本

在理想 ring、均匀链路且忽略竞争的近似下：

下面的 $n$ 采用完整逻辑张量的字节数：All-Reduce 是每个进程的输入大小，All-Gather 是拼接后的完整输出大小，Reduce-Scatter 是切分前的完整输入大小。均匀分片时，All-Gather 的每进程输入为 $n/P$ 字节；三种公式均估算一次集合通信的时间。

$$
T_{\mathrm{AR}}\approx2(P-1)\alpha+2\frac{P-1}{P}n\beta,
$$

$$
T_{\mathrm{AG}}\approx(P-1)\alpha+\frac{P-1}{P}n\beta,
\qquad
T_{\mathrm{RS}}\approx(P-1)\alpha+\frac{P-1}{P}n\beta.
$$

这些式子用于解释消息大小与进程数趋势，不替代真实拓扑、算法选择、协议和链路竞争测量。

## 状态分片与 ZeRO

忽略临时聚合缓冲区时，数据并行和 ZeRO 各阶段的每进程常驻训练状态可近似为

$$
M_{\mathrm{DP}}\approx\Phi(b_p+b_g+b_o),
$$

$$
M_{\mathrm{Z1}}\approx\Phi\left(b_p+b_g+\frac{b_o}{P}\right),\quad
M_{\mathrm{Z2}}\approx\Phi\left(b_p+\frac{b_g+b_o}{P}\right),\quad
M_{\mathrm{Z3}}\approx\frac{\Phi(b_p+b_g+b_o)}{P}.
$$

真实峰值还要加入参数 All-Gather、梯度 Reduce-Scatter、激活和通信桶。

## 模型内部切分

对于 $S$ 个流水线 stage 和 $m$ 个 micro-batch，理想均衡的 GPipe 气泡比例为

$$
\rho_{\mathrm{bubble}}\approx\frac{S-1}{m+S-1}.
$$

若第 $s$ 个 stage 的单个 micro-batch 时间为 $t_s$，稳态吞吐由最慢 stage 约束：

$$
T_{\mathrm{clock}}\ge\max_s t_s.
$$

张量并行可从线性层的两种切分理解。令 $X$ 为输入， $W$ 为线性层权重， $P$ 为张量并行进程数。列切分沿输出维写成 $W=[W_1,\ldots,W_P]$：

$$
XW=[XW_1,\ldots,XW_P],
$$

行切分 $X=[X_1,\ldots,X_P]$、 $W=[W_1^{\mathsf T},\ldots,W_P^{\mathsf T}]^{\mathsf T}$ 时

$$
XW=\sum_{p=1}^PX_pW_p.
$$

上下文并行中，令 $p$ 为 K/V 分片编号， $m_p$ 为局部最大分数， $\ell_p$ 为局部指数和， $o_p$ 为按局部分母归一化后的加权值； $m,\ell,o$ 是合并后的全局状态。局部在线 Softmax 状态可按下式合并：

$$
m=\max_p m_p,\qquad
\ell=\sum_pe^{m_p-m}\ell_p,\qquad
o=\frac{\sum_pe^{m_p-m}\ell_po_p}{\ell}.
$$

## 动态路由与热点

MoE 中 $T$ 个 token、 $E$ 个专家、Top- $k$ 路由和容量系数 $c$ 对应的单专家容量常写为

$$
C=\left\lceil c\frac{kT}{E}\right\rceil.
$$

令 $n_e$ 为专家 $e$ 接收的路由数， $kT/E$ 为均匀路由时的平均专家负载。负载不均可用最大专家负载与平均负载之比描述：

$$
\rho_{\mathrm{imbalance}}=\frac{\max_e n_e}{kT/E}.
$$

## 重叠、热点与扩展

令 $T_{\mathrm{overlap}}$ 为被计算区间覆盖的通信时长， $T_r$ 为进程 $r$ 完成该同步步骤所需的时间。若通信只有一部分能被无依赖计算覆盖，则

$$
T_{\mathrm{exposed}}=\max(0,T_{\mathrm{comm}}-T_{\mathrm{overlap}}),
$$

$$
T_{\mathrm{step}}\ge\max_r T_r.
$$

前式区分“通信发生”与“通信进入关键路径”，后式说明同步步骤由最慢进程决定。

令 $P_{\mathrm{DP}},P_{\mathrm{TP}},P_{\mathrm{PP}},P_{\mathrm{CP}},P_{\mathrm{EP}}$ 分别为数据、张量、流水线、上下文和专家并行度。组合并行必须满足进程总数约束：

$$
P=P_{\mathrm{DP}}P_{\mathrm{TP}}P_{\mathrm{PP}}P_{\mathrm{CP}}P_{\mathrm{EP}},
$$

实际策略还需同时满足设备网格、模型可整除性、容量、拓扑和恢复条件。

## 使用边界

- $\alpha$– $\beta$ 模型是解释工具；真实结论应记录集合通信算法、链路、拓扑域和竞争。
- 理想分片公式不包含激活、临时聚合、通信桶、碎片和运行时工作区。
- 平均时间不能替代最大进程时间；总通信时长不能替代暴露通信时间。
