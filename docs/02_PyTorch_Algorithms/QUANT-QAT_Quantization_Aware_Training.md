# QUANT-QAT. Quantization Aware Training | 量化感知训练

**难度：** Hard | **环境：** [CPU 主线，GPU 可选](#runtime-environment) | **标签：** `量化`、`QAT`、`近似梯度` | **目标人群：** 理解量化误差与反向传播的读者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/QUANT-QAT_Quantization_Aware_Training.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


## 本节导读

直接把训练好的权重降到低位宽，预测损失可能超过要求。若给模型少量继续训练的机会，并在前向中加入量化造成的离散误差，权重能否学会补偿？

量化感知训练（QAT）保留可更新的浮点主权重，用伪量化模拟前向误差，再以近似梯度更新。沿范围统计、伪量化与权重更新，最后检查转换后的结果，判断付出的训练成本换回了多少质量。

<a id="runtime-environment"></a>

## 运行环境

- **基础练习与框架核对：** 使用 [BASE-CPU](../guide.md#base-cpu)，完成题目后核对 PyTorch 的伪量化行为。
- **可选 GPU 实验：** 使用 [Q-CUDA-MECHANISM](../topic_discussion/quantization_compression/quickstart.md#q-cuda-mechanism-quickstart)，比较一个 Transformer Block 的训练与转换结果。

## 前置阅读

先建立量化参数、校准集与独立评测集的口径，再进入训练期误差适配。

- [52 量化校准与误差控制](./52_Quantization_Calibration_and_Error.md)
- [25 W8A16 量化表示](./25_Quantization_W8A16.md)
- [11 梯度累积与训练控制](./11_Gradient_Accumulation.md)
## Step 1：训练怎样看到量化误差

伪量化在浮点张量中模拟缩放、裁剪、舍入与恢复，使模型前向看到离散化后的近似值。优化器仍更新浮点主权重，统计观测器则记录数据范围并据此生成量化参数；量化参数包括缩放因子、零点和整数码范围。

令 $x$ 为浮点值， $s>0$ 为每个整数码间隔对应的浮点步长， $z$ 为零点， $[q_{\min},q_{\max}]$ 为整数码范围， $\widetilde{x}$ 为恢复值。下式给出一般仿射伪量化；对称量化令零点为零，训练结束后将权重转换为目标表示，再验证实际输出。

$$
\widetilde{x}=s\left(\mathrm{clip}\left(\mathrm{round}(x/s)+z,q_{\min},q_{\max}\right)-z\right).
$$

| 对象 | 训练中的作用 | 转换时核对什么 |
| --- | --- | --- |
| 浮点主权重 | 接收优化器更新 | 转换后的权重与训练结果对应 |
| 范围统计 | 估计量化参数 | 冻结参数是否正确保存 |
| 伪量化 | 模拟低精度误差 | 实际数值表示和计算规则是否匹配 |
| 任务损失 | 驱动权重补偿误差 | 转换后在留出样本上的质量 |

![浮点主权重经近似梯度更新再转换](../public/02_PyTorch_Algorithms/quant_qat_flow.svg)

## Step 2：舍入后为什么还可以更新权重

舍入函数的真实导数几乎处处为零，直接沿它求导难以更新权重。直通估计器（STE）在前向保留离散值，反向用一条近似连续路径代替舍入导数。这里先裁剪到观测范围：可表示区间内传递梯度，超出范围的饱和值梯度被裁剪截断。

若 $[x_{\min},x_{\max}]$ 是可表示范围，近似梯度写作下式，其中指示函数在条件成立时为 1，否则为 0。例如范围为 −1 到 1，输入 0.5 继续传递梯度，输入 2 的梯度为零。这个梯度用于训练补偿，转换后的整数运算仍按目标表示执行。

$$
\frac{\partial\widetilde{x}}{\partial x}\approx
\mathbf{1}(x_{\min}\le x\le x_{\max}).
$$

| 环节 | 前向 | 反向 |
| --- | --- | --- |
| 范围统计 | 从样本更新范围 | 统计更新不由优化器求导 |
| 裁剪 | 限制到可表示区间 | 区间内传递，饱和区截断 |
| 舍入与恢复 | 映射到整数码再恢复浮点值 | 用近似路径绕过离散导数 |

![伪量化前向与近似梯度反向](../public/02_PyTorch_Algorithms/quant_qat_ste.svg)

## Step 3：训练恢复的质量怎样保留到转换后

观测器持续变化时，前向采用的范围也在变化；冻结观测器后，验证路径复用已有范围。指数移动平均把历史范围和当前绝对最大值按比例混合，降低批次波动。首次观察先建立范围，后续训练继续更新，评估时保持不变。

转换时核对位宽、分组或逐张量粒度、缩放参数以及计算精度是否与训练假设一致。先比较转换前后的数值和质量，再用目标后端检查实际加载与执行成本。

| 检查对象 | 训练记录 | 转换后记录 |
| --- | --- | --- |
| 范围 | 更新与冻结时机 | 已保存的量化参数 |
| 精度配置 | 位宽、粒度与饱和情况 | 数值表示和算子配置 |
| 质量 | 验证集上的预算比较 | 选中候选的留出测试 |
| 成本 | 总训练时间与峰值 | 文件存储、实际内核和服务表现 |

<a id="mechanism-practice"></a>

## Step 4：实现范围更新、STE 与伪量化线性层

本题提供浮点线性层与范围状态，骨架负责裁剪、离散值计算和模块结构。请补全范围的首次／平滑更新、近似梯度组合，以及激活和权重分别伪量化后的前向。测试检查评估冻结、区间内与饱和梯度、输出形状和浮点主权重更新；CPU 采用对称 INT8 模拟，GPU 扩展比较权重 INT4 参考转换。

| TODO / 函数 | 学习者完成的机制 | 必须满足的约束 | 测试证据 |
| --- | --- | --- | --- |
| TODO 1：更新统计观测器范围<br>`FakeQuantizer.update_observer` | 建立并平滑更新绝对最大值 | 首次赋值，后续按历史比例更新 | 训练建立范围，评估冻结 |
| TODO 2：构造直通估计器<br>`FakeQuantizer.fake_quant` | 前向取离散恢复值，反向沿裁剪路径 | 区间内传梯度，饱和值截断 | 区间内梯度 1、饱和区梯度 0 |
| TODO 3：执行双路伪量化<br>`QATLinearSim.forward` | 激活与权重各自量化后执行线性层 | 分别维护范围，优化器更新浮点主权重 | 输出形状、评估范围和参数更新 |


```python
import json
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
```


```python
# 题目区：补全统计观测器、STE 伪量化和双路径线性层三个 QAT 机制。
class FakeQuantizer:
    """维护训练期范围，并用伪量化在浮点张量中模拟 INT8 离散误差。"""

    def __init__(self, qmax: int = 127, momentum: float = 0.9, eps: float = 1e-8):
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1).")
        self.qmax = qmax
        self.momentum = momentum
        self.eps = eps
        self.observed_absmax = None

    def update_observer(self, tensor: torch.Tensor):
        """使用当前 absmax 更新指数移动平均范围。"""
        current = tensor.detach().float().abs().max().clamp_min(self.eps)
        # TODO 1（训练范围）：首次观察直接赋值；后续使用 momentum 更新历史范围。
        # ==========================================
        # TODO 1：更新统计观测器范围
        # ==========================================
        # self.observed_absmax = ???  # 首次建立范围，后续按历史比例平滑更新。
        raise NotImplementedError("请完成 TODO 1：self.observed_absmax")
        return self.observed_absmax

    def fake_quant(self, tensor: torch.Tensor, *, update_observer: bool):
        """前向模拟裁剪与舍入，反向在未饱和区使用 STE。"""
        if update_observer:
            self.update_observer(tensor)
        if self.observed_absmax is None:
            raise RuntimeError("observer has no range; run a training observation first")

        absmax = self.observed_absmax.to(tensor.device, tensor.dtype).clamp_min(self.eps)
        scale = self.qmax / absmax
        clipped = torch.clamp(tensor, -absmax, absmax)
        dequantized = torch.round(clipped * scale) / scale

        # TODO 2（STE）：前向取 dequantized，反向沿 clipped 传递近似梯度。
        # ==========================================
        # TODO 2：构造直通估计器
        # ==========================================
        # fake_quantized = ???  # 前向取离散值，梯度沿裁剪路径传播。
        raise NotImplementedError("请完成 TODO 2：fake_quantized")
        return fake_quantized


class QATLinearSim(nn.Module):
    """保留浮点主权重，并在训练前向中模拟激活与权重量化。"""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.uniform_(self.weight, -0.5, 0.5)
        self.activation_quantizer = FakeQuantizer()
        self.weight_quantizer = FakeQuantizer()

    def forward(self, inputs: torch.Tensor):
        """训练时更新统计观测器；评估时冻结范围并复用既有量化参数。"""
        # TODO 3（QAT 前向）：分别伪量化激活和权重，再执行线性层。
        # ==========================================
        # TODO 3：执行双路伪量化
        # ==========================================
        # quantized_inputs = ???  # 激活量化结果；由训练状态决定是否更新范围。
        # quantized_weight = ???  # 权重量化结果；保留浮点主权重的梯度。
        # outputs = ???  # 使用两路量化结果计算线性输出。
        raise NotImplementedError("请完成 TODO 3：双路伪量化前向")
        return outputs
```

## 测试

核对范围建立与评估冻结、两类梯度，以及线性层前向和主权重更新。


```python
# 机制测试：统计观测器生命周期、STE 梯度、QAT 前向与训练更新。
def test_observer_updates_then_freezes():
    """训练观察应更新范围，冻结路径不应被新异常值改写。"""
    quantizer = FakeQuantizer(momentum=0.5)
    quantizer.fake_quant(torch.tensor([-1.0, 2.0]), update_observer=True)
    trained_range = quantizer.observed_absmax.clone()
    quantizer.fake_quant(torch.tensor([100.0]), update_observer=False)
    assert torch.equal(quantizer.observed_absmax, trained_range)


def test_ste_passes_in_range_gradient_and_clips_saturation():
    """STE 在可表示区传递梯度，超出统计观测器范围的值由 clamp 截断。"""
    quantizer = FakeQuantizer()
    quantizer.update_observer(torch.tensor([-1.0, 1.0]))
    inputs = torch.tensor([-2.0, 0.5], requires_grad=True)
    quantizer.fake_quant(inputs, update_observer=False).sum().backward()
    assert inputs.grad[0].item() == 0.0
    assert inputs.grad[1].item() == 1.0


def test_qat_linear_training_and_eval_contract():
    """训练前向建立范围，评估前向复用范围并保持输出形状。"""
    torch.manual_seed(0)
    layer = QATLinearSim(4, 3)
    inputs = torch.randn(5, 4)
    train_output = layer(inputs)
    activation_range = layer.activation_quantizer.observed_absmax.clone()
    layer.eval()
    eval_output = layer(inputs * 10)
    assert train_output.shape == (5, 3) and eval_output.shape == (5, 3)
    assert torch.equal(layer.activation_quantizer.observed_absmax, activation_range)


def test_qat_step_updates_float_master_weight():
    """伪量化前向后，优化器仍应更新浮点主权重。"""
    torch.manual_seed(1)
    layer = QATLinearSim(3, 2)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    inputs, target = torch.randn(4, 3), torch.randn(4, 2)
    before = layer.weight.detach().clone()
    loss = F.mse_loss(layer(inputs), target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert layer.weight.grad is not None
    assert not torch.equal(layer.weight.detach(), before)


def run_qat_tests():
    """汇总 QAT 的四项机制测试。"""
    for test in (
        test_observer_updates_then_freezes,
        test_ste_passes_in_range_gradient_and_clips_saturation,
        test_qat_linear_training_and_eval_contract,
        test_qat_step_updates_float_master_weight,
    ):
        test()
    print("✅ QAT 机制测试通过：observer、STE、双路径前向和主权重更新均已验证。")


run_qat_tests()
```

## 参考代码与解析

### 代码

```python
# 题目区：补全统计观测器、STE 伪量化和双路径线性层三个 QAT 机制。
class FakeQuantizer:
    """维护训练期范围，并用伪量化在浮点张量中模拟 INT8 离散误差。"""

    def __init__(self, qmax: int = 127, momentum: float = 0.9, eps: float = 1e-8):
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1).")
        self.qmax = qmax
        self.momentum = momentum
        self.eps = eps
        self.observed_absmax = None

    def update_observer(self, tensor: torch.Tensor):
        """使用当前 absmax 更新指数移动平均范围。"""
        current = tensor.detach().float().abs().max().clamp_min(self.eps)
        # TODO 1（训练范围）：首次观察直接赋值；后续使用 momentum 更新历史范围。
        self.observed_absmax = (
            current if self.observed_absmax is None
            else self.momentum * self.observed_absmax + (1.0 - self.momentum) * current
        )
        return self.observed_absmax

    def fake_quant(self, tensor: torch.Tensor, *, update_observer: bool):
        """前向模拟裁剪与舍入，反向在未饱和区使用 STE。"""
        if update_observer:
            self.update_observer(tensor)
        if self.observed_absmax is None:
            raise RuntimeError("observer has no range; run a training observation first")

        absmax = self.observed_absmax.to(tensor.device, tensor.dtype).clamp_min(self.eps)
        scale = self.qmax / absmax
        clipped = torch.clamp(tensor, -absmax, absmax)
        dequantized = torch.round(clipped * scale) / scale

        # TODO 2（STE）：前向取 dequantized，反向沿 clipped 传递近似梯度。
        fake_quantized = clipped + (dequantized - clipped).detach()
        return fake_quantized


class QATLinearSim(nn.Module):
    """保留浮点主权重，并在训练前向中模拟激活与权重量化。"""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.uniform_(self.weight, -0.5, 0.5)
        self.activation_quantizer = FakeQuantizer()
        self.weight_quantizer = FakeQuantizer()

    def forward(self, inputs: torch.Tensor):
        """训练时更新统计观测器；评估时冻结范围并复用既有量化参数。"""
        # TODO 3（QAT 前向）：分别伪量化激活和权重，再执行线性层。
        quantized_inputs = self.activation_quantizer.fake_quant(
            inputs, update_observer=self.training
        )
        quantized_weight = self.weight_quantizer.fake_quant(
            self.weight, update_observer=self.training
        )
        outputs = F.linear(quantized_inputs, quantized_weight)
        return outputs
```

### 解析

#### TODO 1：更新范围

- 当前绝对最大值先脱离梯度，再参与统计；第一次直接赋值，后续用历史比例进行指数移动平均。
- 答案区分首次与后续更新，评估分支复用已有范围。测试用新异常值检查评估时范围保持不变；连续平滑更新还可另加数值样例。

#### TODO 2：构造近似梯度

- 前向等于离散恢复值，反向通过裁剪后的浮点路径；停止梯度的残差只改变前向数值。
- 答案组合这两条路径。测试在范围 −1 到 1 内外取点，核对 0.5 的梯度为 1、−2 的梯度为 0。

#### TODO 3：分别处理激活与权重

- 两类张量使用各自的范围状态，训练时更新、评估时复用；浮点主权重仍是优化器参数。
- 答案把两路伪量化结果送入线性运算。测试核对形状、评估冻结和一次优化器更新后的权重变化。

<a id="framework-check"></a>

## 框架一致性核对（可选）

使用 PyTorch 逐张量伪量化算子核对本节前向值与采样点上的梯度，再核对一次对称范围观测。共同输入包含区间内与明显饱和的点，判断对应样例的数值和状态是否一致。

| 核对对象 | Step 4 教学实现 | 成熟框架 CPU 路径 | 共同输入与通过条件 |
| --- | --- | --- | --- |
| 伪量化前向 | `FakeQuantizer.fake_quant` | PyTorch 逐张量伪量化 | 同输入、步长和码范围，恢复值一致 |
| STE 反向 | 裁剪路径的近似梯度 | PyTorch 自动微分 | 同损失，区间内和饱和样例梯度一致 |
| 初始范围 | `update_observer` 的首次统计 | `MinMaxObserver` | 相同张量与对称码范围，缩放步长一致 |


```python
from torch.ao.quantization.observer import MinMaxObserver


def check_qat_framework_conformance():
    """核对本节伪量化前向、STE 梯度与 PyTorch 参考算子。"""
    scale = 1.0 / 127.0
    local_input = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
    framework_input = local_input.detach().clone().requires_grad_(True)
    quantizer = FakeQuantizer()
    quantizer.observed_absmax = torch.tensor(1.0)
    local_output = quantizer.fake_quant(local_input, update_observer=False)
    framework_output = torch.fake_quantize_per_tensor_affine(
        framework_input, scale, 0, -127, 127
    )
    local_output.sum().backward()
    framework_output.sum().backward()

    observed = torch.tensor([-0.8, 0.0, 1.0], dtype=torch.float32)
    local_observer = FakeQuantizer()
    local_absmax = local_observer.update_observer(observed)
    framework_observer = MinMaxObserver(
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        quant_min=-127,
        quant_max=127,
    )
    framework_observer(observed)
    framework_scale, framework_zero_point = framework_observer.calculate_qparams()
    checks = {
        "forward_match": bool(torch.allclose(local_output, framework_output, atol=1e-7, rtol=1e-6)),
        "gradient_match": bool(torch.equal(local_input.grad, framework_input.grad)),
        "saturation_gradient_zero": local_input.grad[[0, -1]].abs().sum().item() == 0.0,
        "interior_gradient_one": bool(torch.equal(local_input.grad[1:4], torch.ones(3))),
        "observer_scale_match": bool(torch.isclose(local_absmax / 127.0, framework_scale[0], atol=1e-7, rtol=1e-6)),
        "observer_zero_point_is_zero": int(framework_zero_point.item()) == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"框架一致性核对失败：{failed}")
    return {
        "status": "passed",
        "versions": {"torch": torch.__version__},
        "checks": checks,
        "local_output": local_output.detach().tolist(),
        "local_gradient": local_input.grad.tolist(),
    }


framework_check = check_qat_framework_conformance()
framework_check
```

<a id="gpu-experiment"></a>

## Step 5：可选 GPU 实验——比较 PTQ 与不同 QAT 预算

CPU 题目已经验证统计观测器、伪量化与直通估计器；GPU 实验进一步在真实 Transformer Block 上比较训练后量化（PTQ）与三个 QAT 训练预算。只更新所选 Block 的参数，其他参数冻结；每档训练后转换，再用验证集选预算，并重建所选预算在测试集评估。

| 学习问题 | CPU 已确认 | GPU 要确认 |
|---|---|---|
| QAT 训练预算能否回收 PTQ 质量损失 | 统计观测器生命周期、伪量化前向和近似梯度 | 真实 Block 的 `prepare / train / convert` 路径、质量恢复率、训练时间和峰值显存 |

### 5.1 环境与固定工作负载

本实验使用 `Q-CUDA-MECHANISM`。首次运行先完成[量化机制 GPU 环境 Quickstart](../topic_discussion/quantization_compression/quickstart.md#q-cuda-mechanism-quickstart)，已有环境执行 Profile 预检后再启动 JupyterLab。训练集更新目标 Block，验证集选择预算，测试集只形成最终质量证据；验证与测试不更新参数。

| 配置层 | 本节固定口径 | 修改规则 |
|---|---|---|
| 环境 | `Q-CUDA-MECHANISM`；Transformers + Datasets + torchao；单张 NVIDIA GPU | 参考 INT4 表示用于验证转换后质量；原生 INT4 内核需要目标后端证据 |
| 模型与数据 | Qwen2.5-0.5B 的中间 Block；固定 WikiText-2 train/validation/test | 模型或数据改变后新建运行记录 |
| 工作负载 | 最大长度 128 Token；QAT 预算 5 / 20 / 50 个优化器步 | 目标 Block、长度或预算改变后递增工作负载版本 |
| 配对变量 | BF16 → INT4 PTQ → 三档 QAT 预算转换结果 | PTQ 与 QAT 共享 group size 和目标 Block |
| 测量指标 | 转换后 NLL、质量回收率、训练时间、总时长除以步数的平均耗时与峰值显存 | 验证集选预算，测试集只评一次；参考表示不作为压缩字节或 Kernel 证据 |
| 结果资产 | JSON 与所选 QAT Block 状态文件 | 正式复测使用唯一 `RUN_LABEL` |


```python
# 5.1 只声明模型、数据划分、QAT 预算和结果路径；默认不下载模型或数据。
import os
from pathlib import Path

# 学习者入口：普通复测通常只修改这两项。
RUN_MODE = "dry_run"  # 默认值 dry_run：只登记计划；改为 real_gpu 才执行准备、训练和转换矩阵。
RUN_LABEL = "default2"  # 默认值 default2；正式复测改为唯一标签，避免覆盖旧结果。

# 模型来源：main 会在 5.2 自动解析为本次运行使用的不可变 commit。
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # 单卡机制基线；更换模型后不得合并旧结果。
MODEL_REVISION = "main"  # 可填写分支、标签或 commit hash。
DATA_SOURCE = "huggingface"  # huggingface 下载固定快照；local_jsonl 读取离线副本。
DATASET_ID = "Salesforce/wikitext"  # 用于因果语言模型训练与质量比较的数据集。
DATASET_CONFIG = "wikitext-2-raw-v1"  # 保留原始文本的 WikiText-2 配置。
DATASET_REVISION = "f776294184f13b8ff2337b3841cf9269a6216d1e"  # 固定数据 commit。
TRAIN_SPLIT = "train"  # 更新 QAT Block 参数。
VALIDATION_SPLIT = "validation"  # 选择训练预算，不用于最终质量结论。
TEST_SPLIT = "test"  # 只在候选选择后形成最终质量证据。
TRAIN_SAMPLES = 128  # QAT 训练文本数；增大会延长每个候选训练时间。
VALIDATION_SAMPLES = 64  # 预算选择样本数；不得参与参数更新。
TEST_SAMPLES = 64  # 独立测试样本数；不得参与候选选择。
DATA_SEED = 42  # 固定抽样与训练 DataLoader generator。
MAX_LENGTH = 128  # token 截断上限；增大会提高训练显存与时间。
TRAIN_BATCH_SIZE = 1  # 每步 micro-batch；所有 QAT 预算必须一致。
EVAL_BATCH_SIZE = 1  # 验证与测试 batch；PTQ/QAT 必须一致。

# 固定实验身份：协议改变时递增版本，不把版本编码进稳定 ID。
ENVIRONMENT_PROFILE = "Q-CUDA-MECHANISM"  # 运行前必须通过同名 Profile 预检。
WORKLOAD_ID = "quant-qat-closed-loop"  # BF16、PTQ 与多预算 QAT 的闭环矩阵。
WORKLOAD_VERSION = "v3"  # v3 使用 unpacked INT4 参考路径，隔离 QAT 质量闭环与生产 Kernel 兼容性。

# QAT 候选矩阵：改变下列内容时递增 WORKLOAD_VERSION。
TARGET_BLOCK_INDEX = 12  # 训练中间 Block，兼顾代表性和单卡成本。
GROUP_SIZE = 64  # INT4 仅权重量化分组大小；PTQ 与 QAT 必须一致。
REFERENCE_WEIGHT_DTYPE = "int4"  # 参考量化位宽；当前实验固定为 INT4。
REFERENCE_PACKING_FORMAT = "unpacked_to_int8"  # 参考执行表示；用于质量闭环，不代表生产压缩 Kernel。
QAT_STEP_BUDGETS = (5, 20, 50)  # 不少于三个训练预算，用于绘制质量恢复—成本曲线。
LEARNING_RATE = 1e-5  # 目标 Block 学习率；所有预算保持一致。
MAX_TEST_NLL_RATIO = 1.10  # convert 后测试 NLL 相对 BF16 的最大允许比例。
ARTIFACT_RELATIVE_PATH = f"benchmarks/artifacts/quant_qat/{RUN_LABEL}"  # 仓库相对产物目录；保存所选转换后 Block。
HF_HUB_CACHE = os.getenv("HF_HUB_CACHE")  # 可选模型缓存环境变量；未设置时使用默认目录。
MODEL_CACHE_DIR = Path(HF_HUB_CACHE) if HF_HUB_CACHE else None  # 模型缓存目录。
HF_DATASETS_CACHE = os.getenv("HF_DATASETS_CACHE")  # 可选数据缓存环境变量；未设置时使用默认目录。
DATA_CACHE_DIR = Path(HF_DATASETS_CACHE) if HF_DATASETS_CACHE else None  # 数据缓存目录。
LOCAL_DATA_FILES = None  # local_jsonl 模式传入 train/validation/test JSONL 路径。
LOCAL_FILES_ONLY = False  # 模型已按相同 revision 预下载时可设为 True。
OUTPUT_RELATIVE_PATH = f"benchmarks/results/quant_qat_closed_loop_{RUN_LABEL}.json"  # 仓库相对结果路径；由 5.2 锚定。
```

### 5.2 执行并保存 JSON

执行单元让每个候选从同一模型检查点独立开始。PTQ 直接转换目标 Block；QAT 候选先 prepare，再按固定步数训练，最后 convert。验证集用于选择预算，测试集不参与训练和选择；每个预算记录验证 NLL、训练时间、峰值及失败；所选预算按相同种子重新训练并转换，再测试和保存 Block 状态。


```python
# 5.2 执行 BF16 / PTQ / 多预算 QAT 闭环并保存 JSON。
from tools.notebook_runtime import hash_ids, package_version, resolve_model_commit, stable_sample
from tools.project_runtime import resolve_project_root

import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import time

import torch


REPO_ROOT = resolve_project_root()
ARTIFACT_ROOT = REPO_ROOT / ARTIFACT_RELATIVE_PATH
OUTPUT_PATH = REPO_ROOT / OUTPUT_RELATIVE_PATH


result = {
    "schema_version": "quantization-mechanism-gpu/v5",
    "chapter": "QUANT-QAT",
    "run_mode": RUN_MODE,
    "run_label": RUN_LABEL,
    "environment_profile": ENVIRONMENT_PROFILE,
    "framework": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "datasets": package_version("datasets"),
        "torchao": package_version("torchao"),
        "mslk": package_version("mslk"),
    },
    "hardware": {
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        "compute_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
    },
    "dataset": {
        "source": DATA_SOURCE,
        "dataset_id": DATASET_ID,
        "config": DATASET_CONFIG,
        "revision": DATASET_REVISION,
        "train_split": TRAIN_SPLIT,
        "validation_split": VALIDATION_SPLIT,
        "test_split": TEST_SPLIT,
        "train_samples": TRAIN_SAMPLES,
        "validation_samples": VALIDATION_SAMPLES,
        "test_samples": TEST_SAMPLES,
        "selection_seed": DATA_SEED,
        "max_length": MAX_LENGTH,
    },
    "workload": {
        "id": WORKLOAD_ID,
        "version": WORKLOAD_VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "resolved_model_commit": None,
        "revision_resolution_source": None,
        "target_block_index": TARGET_BLOCK_INDEX,
        "group_size": GROUP_SIZE,
        "reference_weight_dtype": REFERENCE_WEIGHT_DTYPE,
        "reference_packing_format": REFERENCE_PACKING_FORMAT,
        "qat_step_budgets": list(QAT_STEP_BUDGETS),
        "learning_rate": LEARNING_RATE,
    },
    "baseline": {"name": "bf16"},
    "ptq": None,
    "qat_candidates": [],
    "selected_candidate": None,
    "artifact_manifest": None,
    "backend_preflight": None,
    "failure": None,
    "evidence_level": "not_executed",
    "decision": None,
}

if RUN_MODE == "real_gpu":
    current_stage = "environment_preflight"
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("RUN_MODE=real_gpu 但 CUDA 不可用")

        from datasets import load_dataset
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from torchao.quantization import IntxWeightOnlyConfig, quantize_
        from torchao.quantization.granularity import PerGroup
        from torchao.quantization.qat import QATConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer

        def make_reference_int4_config():
            """构造可训练、可转换但不声称生产 Kernel 的参考 INT4 配置。"""
            if REFERENCE_WEIGHT_DTYPE != "int4" or REFERENCE_PACKING_FORMAT != "unpacked_to_int8":
                raise ValueError("当前 QAT 教学协议只支持 int4 + unpacked_to_int8 参考路径")
            return IntxWeightOnlyConfig(
                weight_dtype=torch.int4,
                granularity=PerGroup(GROUP_SIZE),
            )

        current_stage = "reference_qat_preflight"
        preflight = nn.Sequential(
            nn.Linear(256, 256, bias=False, device="cuda", dtype=torch.bfloat16)
        )
        preflight_config = make_reference_int4_config()
        quantize_(preflight, QATConfig(preflight_config, step="prepare"))
        preflight_input = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
        preflight(preflight_input).sum().backward()
        quantize_(preflight, QATConfig(preflight_config, step="convert"))
        preflight_output = preflight(preflight_input)
        torch.cuda.synchronize()
        result["backend_preflight"] = {
            "status": "ok",
            "representation": REFERENCE_PACKING_FORMAT,
            "weight_type": type(preflight[0].weight).__name__,
            "output_shape": list(preflight_output.shape),
            "kernel_evidence": False,
        }
        del preflight, preflight_input, preflight_output
        torch.cuda.empty_cache()

        current_stage = "resolve_model_revision"
        resolved_commit, resolution_source = resolve_model_commit(
            MODEL_ID, MODEL_REVISION, MODEL_CACHE_DIR, LOCAL_FILES_ONLY
        )
        result["workload"]["resolved_model_commit"] = resolved_commit
        result["workload"]["revision_resolution_source"] = resolution_source

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=resolved_commit,
            cache_dir=MODEL_CACHE_DIR,
            local_files_only=LOCAL_FILES_ONLY,
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        if DATA_SOURCE == "huggingface":
            raw = load_dataset(
                DATASET_ID,
                DATASET_CONFIG,
                revision=DATASET_REVISION,
                cache_dir=str(DATA_CACHE_DIR) if DATA_CACHE_DIR else None,
            )
        elif DATA_SOURCE == "local_jsonl":
            if not LOCAL_DATA_FILES:
                raise ValueError("DATA_SOURCE=local_jsonl 时必须配置 LOCAL_DATA_FILES")
            raw = load_dataset("json", data_files=LOCAL_DATA_FILES)
        else:
            raise ValueError(f"未知 DATA_SOURCE：{DATA_SOURCE}")

        train_rows, train_ids = stable_sample(raw[TRAIN_SPLIT], TRAIN_SAMPLES, DATA_SEED)
        validation_rows, validation_ids = stable_sample(raw[VALIDATION_SPLIT], VALIDATION_SAMPLES, DATA_SEED + 1)
        test_rows, test_ids = stable_sample(raw[TEST_SPLIT], TEST_SAMPLES, DATA_SEED + 2)
        result["dataset"].update({
            "train_ids_hash": hash_ids(train_ids),
            "validation_ids_hash": hash_ids(validation_ids),
            "test_ids_hash": hash_ids(test_ids),
            "filter": "non_empty_text",
        })

        class TokenizedTextDataset(Dataset):
            """把固定文本转换为因果语言模型输入。"""

            def __init__(self, texts):
                self.texts = list(texts)

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, index):
                return tokenizer(
                    self.texts[index],
                    truncation=True,
                    max_length=MAX_LENGTH,
                    add_special_tokens=True,
                )["input_ids"]

        def collate_token_ids(batch):
            """右侧填充并构造标签；padding token 不参与损失。"""
            width = max(len(ids) for ids in batch)
            input_ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
            for row, ids in enumerate(batch):
                values = torch.tensor(ids, dtype=torch.long)
                input_ids[row, : len(ids)] = values
                attention_mask[row, : len(ids)] = 1
            labels = input_ids.masked_fill(attention_mask == 0, -100)
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        def make_train_loader():
            """为每个候选重建相同随机种子的训练顺序。"""
            train_generator = torch.Generator().manual_seed(DATA_SEED)
            return DataLoader(
                TokenizedTextDataset(train_rows["text"]),
                batch_size=TRAIN_BATCH_SIZE,
                shuffle=True,
                generator=train_generator,
                collate_fn=collate_token_ids,
            )
        validation_loader = DataLoader(
            TokenizedTextDataset(validation_rows["text"]),
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_token_ids,
        )
        test_loader = DataLoader(
            TokenizedTextDataset(test_rows["text"]),
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_token_ids,
        )

        base_config = make_reference_int4_config()

        def load_model():
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                revision=resolved_commit,
                cache_dir=MODEL_CACHE_DIR,
                local_files_only=LOCAL_FILES_ONLY,
                dtype=torch.bfloat16,
            ).to("cuda")
            if TARGET_BLOCK_INDEX >= len(model.model.layers):
                raise ValueError(f"TARGET_BLOCK_INDEX 超出模型层数 {len(model.model.layers)}")
            return model

        def evaluate(model, loader):
            """在冻结状态下计算 token 加权 NLL 和峰值显存。"""
            model.eval()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            total_loss = 0.0
            total_tokens = 0
            started = time.perf_counter()
            with torch.inference_mode():
                for batch in loader:
                    batch = {key: value.to("cuda") for key, value in batch.items()}
                    outputs = model(**batch)
                    valid_tokens = int((batch["labels"][:, 1:] != -100).sum().item())
                    total_loss += float(outputs.loss.item()) * valid_tokens
                    total_tokens += valid_tokens
            torch.cuda.synchronize()
            elapsed_s = time.perf_counter() - started
            mean_nll = total_loss / max(1, total_tokens)
            return {
                "mean_nll": mean_nll,
                "perplexity": math.exp(min(mean_nll, 20.0)),
                "evaluated_tokens": total_tokens,
                "elapsed_s": elapsed_s,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            }

        def release(model):
            del model
            gc.collect()
            torch.cuda.empty_cache()

        bf16_model = load_model()
        bf16_validation = evaluate(bf16_model, validation_loader)
        bf16_test = evaluate(bf16_model, test_loader)
        release(bf16_model)
        result["baseline"].update({"validation": bf16_validation, "test": bf16_test})

        current_stage = "ptq_reference_baseline"
        ptq_model = load_model()
        target_block = ptq_model.model.layers[TARGET_BLOCK_INDEX]
        quantize_(target_block, base_config)
        ptq_validation = evaluate(ptq_model, validation_loader)
        ptq_test = evaluate(ptq_model, test_loader)
        result["ptq"] = {
            "name": "int4_ptq",
            "validation": ptq_validation,
            "test": ptq_test,
            "path_evidence": {
                "target_block_type": type(target_block).__name__,
                "linear_types": sorted({type(module).__name__ for module in target_block.modules()}),
                "reference_weight_type": "IntxUnpackedToInt8Tensor",
                "kernel_evidence": False,
            },
        }
        release(ptq_model)

        current_stage = "qat_budget_matrix"
        qat_candidates = []
        for budget in QAT_STEP_BUDGETS:
            torch.manual_seed(DATA_SEED)
            model = load_model()
            for parameter in model.parameters():
                parameter.requires_grad = False
            target_block = model.model.layers[TARGET_BLOCK_INDEX]
            quantize_(target_block, QATConfig(base_config, step="prepare"))
            for parameter in target_block.parameters():
                parameter.requires_grad = True
            optimizer = torch.optim.AdamW(
                [parameter for parameter in target_block.parameters() if parameter.requires_grad],
                lr=LEARNING_RATE,
            )

            model.train()
            train_loader = make_train_loader()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            losses = []
            iterator = iter(train_loader)
            for _step in range(budget):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    batch = next(iterator)
                batch = {key: value.to("cuda") for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                outputs = model(**batch)
                outputs.loss.backward()
                optimizer.step()
                losses.append(float(outputs.loss.detach().item()))
            torch.cuda.synchronize()
            training_seconds = time.perf_counter() - started
            training_peak = int(torch.cuda.max_memory_allocated())

            quantize_(target_block, QATConfig(base_config, step="convert"))
            validation_metrics = evaluate(model, validation_loader)
            candidate = {
                "name": f"qat_{budget}_steps",
                "optimizer_steps": budget,
                "first_train_loss": losses[0],
                "last_train_loss": losses[-1],
                "training_seconds": training_seconds,
                "median_step_ms": 1000.0 * training_seconds / budget,
                "training_peak_allocated_bytes": training_peak,
                "validation": validation_metrics,
                "path_evidence": {
                    "prepare_and_convert": True,
                    "target_block_type": type(target_block).__name__,
                    "linear_types": sorted({type(module).__name__ for module in target_block.modules()}),
                    "reference_weight_type": "IntxUnpackedToInt8Tensor",
                    "kernel_evidence": False,
                },
            }
            qat_candidates.append(candidate)
            release(model)

        selected = min(qat_candidates, key=lambda row: row["validation"]["mean_nll"])
        selected_budget = selected["optimizer_steps"]

        current_stage = "selected_candidate_test"
        # 只重建验证集选中的预算并在独立测试数据划分上形成最终证据。
        torch.manual_seed(DATA_SEED)
        selected_model = load_model()
        for parameter in selected_model.parameters():
            parameter.requires_grad = False
        selected_block = selected_model.model.layers[TARGET_BLOCK_INDEX]
        quantize_(selected_block, QATConfig(base_config, step="prepare"))
        for parameter in selected_block.parameters():
            parameter.requires_grad = True
        optimizer = torch.optim.AdamW(
            [parameter for parameter in selected_block.parameters() if parameter.requires_grad],
            lr=LEARNING_RATE,
        )
        selected_model.train()
        train_loader = make_train_loader()
        iterator = iter(train_loader)
        for _step in range(selected_budget):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            batch = {key: value.to("cuda") for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = selected_model(**batch)
            outputs.loss.backward()
            optimizer.step()
        quantize_(selected_block, QATConfig(base_config, step="convert"))
        selected_test = evaluate(selected_model, test_loader)
        ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
        artifact_path = ARTIFACT_ROOT / f"block_{TARGET_BLOCK_INDEX}_{selected_budget}_steps.pt"
        torch.save(selected_block.state_dict(), artifact_path)
        artifact_manifest = {
            "path": str(artifact_path.relative_to(REPO_ROOT)),
            "bytes": artifact_path.stat().st_size,
            "base_model_id": MODEL_ID,
            "base_model_commit": resolved_commit,
            "target_block_index": TARGET_BLOCK_INDEX,
            "optimizer_steps": selected_budget,
            "group_size": GROUP_SIZE,
            "reference_packing_format": REFERENCE_PACKING_FORMAT,
            "storage_compression_evidence": False,
            "kernel_evidence": False,
            "scope": "tutorial_reference_block_state_not_deployable_model_artifact",
        }
        release(selected_model)

        ptq_gap = ptq_test["mean_nll"] - bf16_test["mean_nll"]
        qat_gap = selected_test["mean_nll"] - bf16_test["mean_nll"]
        recovery_ratio = (ptq_gap - qat_gap) / max(abs(ptq_gap), 1e-12)
        selected["test"] = selected_test
        selected["recovery_ratio_vs_ptq"] = recovery_ratio
        selected["test_nll_ratio_vs_bf16"] = selected_test["mean_nll"] / max(
            bf16_test["mean_nll"], 1e-12
        )

        result["qat_candidates"] = qat_candidates
        result["selected_candidate"] = selected["name"]
        result["selected_test"] = selected_test
        result["recovery_ratio_vs_ptq"] = recovery_ratio
        result["artifact_manifest"] = artifact_manifest
        result["evidence_level"] = "matched_reference_ptq_qat_quality_matrix"
        result["decision"] = (
            "accept_qat_candidate"
            if recovery_ratio > 0 and selected["test_nll_ratio_vs_bf16"] <= MAX_TEST_NLL_RATIO
            else "tune_qat_budget"
        )
    except Exception as error:
        result["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "stage": locals().get("current_stage", "initialization"),
            "traceback_tail": traceback.format_exc().splitlines()[-12:],
        }
        result["evidence_level"] = "failed_execution"
        result["decision"] = "reject_until_environment_fixed"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
display_path = OUTPUT_PATH.relative_to(REPO_ROOT) if OUTPUT_PATH.is_relative_to(REPO_ROOT) else OUTPUT_PATH
print({
    "result_path": str(display_path),
    **{key: result.get(key) for key in ("failure", "selected_candidate", "recovery_ratio_vs_ptq", "evidence_level", "decision")},
})
```

### 5.3 读取结果

当前单元读取执行阶段建立的 `OUTPUT_PATH`，展示预算曲线、选中候选和测试结果；重启内核后先按已有标签恢复该路径。兼容字段 `median_step_ms` 由总训练秒数除以步数换算，表示平均耗时，逐步计时样本尚未保存。


```python
# 5.3 只读取 5.2 保存的 JSON，并按运行身份、预算曲线、最终候选分层展示。
import json

if OUTPUT_PATH.exists():
    saved = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    workload = saved.get("workload", {})
    print("运行身份与决策")
    print({
        "run_label": saved.get("run_label"),
        "model_id": workload.get("model_id"),
        "resolved_model_commit": workload.get("resolved_model_commit"),
        "failure": saved.get("failure"),
        "evidence_level": saved.get("evidence_level"),
        "decision": saved.get("decision"),
    })
    print("\nQAT 预算曲线")
    for row in saved.get("qat_candidates", []):
        print({
            "name": row.get("name"),
            "optimizer_steps": row.get("optimizer_steps"),
            "validation_nll": row.get("validation", {}).get("mean_nll"),
            "training_seconds": row.get("training_seconds"),
            "training_peak_allocated_bytes": row.get("training_peak_allocated_bytes"),
        })
    print("\n最终候选与产物")
    print({
        "selected_candidate": saved.get("selected_candidate"),
        "selected_test": saved.get("selected_test"),
        "recovery_ratio_vs_ptq": saved.get("recovery_ratio_vs_ptq"),
        "artifact_manifest": saved.get("artifact_manifest"),
    })
else:
    print(f"等待 QAT 闭环结果：{OUTPUT_PATH}")
```

### 5.4 解释局部质量恢复与训练成本

`default2` 在 Qwen2.5-0.5B 的索引 12 Block 上比较 5、20、50 步预算，验证集选中 50 步。重建该预算后的测试 NLL 为 3.378714，直接 PTQ 为 3.387613，BF16 为 3.379010。相对 PTQ 与 BF16 间约 0.008604 的损失差，候选减少约 0.008899，恢复率约 103.4%：表示本次样本上略低于 BF16 的 NLL，重复样本与随机种子复测才能判断这一微小差异是否稳定。

本次改变的是局部 Block，NLL 通过完整模型前向计算；其余参数冻结。训练、验证和测试使用登记的不同划分，摘要用于追溯抽样，数据交集仍需样本或内容核对。保存的约 14.9 MiB 文件是参考 Block 状态，后续部署需要生成目标格式并重新验证。

| 观察项 | 当前结果 | 解释与下一动作 |
| --- | --- | --- |
| 训练对象 | 索引 12 的 Block，5／20／50 步 | 局部更新，成本含完整模型前向与相关反向 |
| 预算选择 | 验证 NLL 选择 50 步 | 保留同一选择规则，测试不参与选预算 |
| 留出质量 | 测试 NLL 3.378714，恢复率约 103.4% | 差值较小，扩大样本并重复种子 |
| 时间与资源 | 记录每档试训总时长和峰值 | 单步字段为平均值，未含选中预算重建训练；独立峰值测量前清理前档引用 |
| 产物与决策 | 历史 `accept_qat_candidate`；参考状态文件 | 当前支持局部质量恢复；压缩文件和目标 INT4 执行另行验证 |

## 相关阅读

转换后的质量确认后，再检查保存格式与目标执行路径。

- [TorchAO QAT 工作流](https://docs.pytorch.org/ao/stable/workflows/qat.html)：查看准备、训练与转换阶段。
- [量化产物评估](./82_Quantization_Artifact_Evaluation_Project.md)：将参考训练结果进一步整理为可加载产物并检查。
