# 60. LoRA Fine Tuning Project | LoRA 微调项目

**难度：** Hard | **环境：** CPU-first | **标签：** `训练微调`, `LoRA`, `项目评估` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/60_LoRA_Fine_Tuning_Project.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

本节承接第 13 节的端到端 SFT 小项目，把“训练链路能否跑通”推进为“LoRA 方案是否值得交付”。你需要先固定数据、训练步数和评测口径，分别比较 baseline 与 LoRA，再记录可训练参数量、显存、训练耗时、验证结果和生成样例。

**实验分层：** Step 1-5 是 CPU-first 的项目判断练习，验证数据审计、loss mask、参数账本、报告字段和决策逻辑；它们不证明模型训练效果或 GPU 显存收益。Step 6 是真实 GPU smoke test，只验证模型、数据、LoRA 和 artifact 链路；Step 7 才是同口径 baseline / LoRA 对比。只有固定数据、验证集和重复运行后，才能讨论是否值得交付。

**本节机制边界：** 本节承接第 10、12、13 节，把冻结基座、低秩 A/B、target modules、rank/alpha/dropout、optimizer state、effective batch、loss mask 和 adapter 交付串成项目闭环。QLoRA/NF4、LoRA 变体、学习率调度、显存 profiling 和量化部署分别由 26/65、63、11/13、73/74 和 67 负责，本节只记录它们对项目决策的接口。

| 目标 | 重点检查 | 证据等级 |
|:---|:---|:---|
| 参数高效 | trainable params、参数占比、target modules、rank | CPU 账本可验证；真实参数量需 GPU/模型加载确认 |
| 训练正确 | 数据审计、labels、padding mask、train/validation split | CPU 测试可验证口径；效果需 GPU 训练和验证集 |
| 资源收益 | optimizer state、peak memory、step time、tokens/s | 只能由同口径 GPU benchmark 支持 |
| 可交付性 | adapter、tokenizer、merge 和生成 sanity check | 真实模型运行后才能确认 |


**关键词：** `LoRA`, `training`, `project`, `profiling`, `report`

---
## 前置阅读

**导语：** 先把 LoRA 机制、有效 batch 口径和端到端训练闭环理顺，再进入这个项目；本节默认你已经知道训练循环怎么跑，重点转向 LoRA 方案是否值得采用。
- [10. LoRA Tutorial | LoRA 教程](./10_LoRA_Tutorial.md)
- [12. Gradient Accumulation | 梯度累积](./12_Gradient_Accumulation.md)
- [13. End-to-End Fine-Tuning Experiment | 端到端微调实验](./13_End_to_End_Fine_Tuning_Experiment.md)
- [11. LR Schedulers WSD Cosine | WSD 余弦学习率调度器](./11_LR_Schedulers_WSD_Cosine.md)

## 相关阅读

**导语：** 做完基础 LoRA 项目后，最自然的下一步是继续比较 LoRA 变体，或回看训练成本是否真的划算。
- [63. LoRA Variants Benchmark | LoRA 变体对比项目](./63_LoRA_Variants_Benchmark.md)
- [73. Training Performance Analysis | 训练性能分析](./73_Training_Performance_Analysis.md)

---
### Step 1（CPU 项目设计）: 定义 LoRA 微调目标
先回答一个问题：在尽量少训练参数的前提下，LoRA 能否完成目标任务，并保留可接受的 train / val loss 表现？

| 实验组 | 环境 | 比较内容 | 只允许改变什么 | 主要输出 |
|:---|:---|:---|:---|:---|
| CPU 机制 | CPU 或 GPU | 数据、mask、参数账本和决策函数 | 输入样例或候选配置 | 逻辑测试结果 |
| GPU smoke | 单卡 GPU | 真实模型与 LoRA 链路 | 不做 baseline 对照 | 模型加载、反向传播和 adapter 是否成功 |
| GPU matched | 单卡 GPU | full-parameter baseline vs LoRA | 只改变是否启用 LoRA | 显存、速度、验证损失和生成质量 |

CPU 结果用于检查口径，GPU 结果用于支持资源和效果结论；三组实验不能混用。

- 固定底座模型、数据集、batch size、seq len、优化器、学习率和训练 step 数。
- 明确 baseline 是全参数微调、冻结底座不训练，还是已有的普通微调配置。
- 训练前先做数据审计：样本数、空 response、重复样本、超长样本和长度分布。
- 抽样核对 `input_ids / attention_mask / labels`：response 是否进入 loss，padding 是否被 `-100` 屏蔽。
- 记录 LoRA 配置：target modules、rank、alpha、dropout、learning rate、micro batch、accum steps 和 scheduler。
- 统一记录核心指标：可训练参数量、参数占比、step time、peak memory、train loss、val loss。
- 这节先建立 LoRA 项目交付模板，再把数据、loss、参数、显存、速度、效果和 artifact 收成一份项目汇总。

### Step 2（CPU 项目设计）: 固定 baseline 口径并建立账本

LoRA 的收益必须和稳定 baseline 对比，不能只看 LoRA 自己能不能跑。

- 先在同一批样本和同一套训练配置下跑通 baseline。
- 记录 baseline 的可训练参数量、train/val loss、平均 step time 和 peak memory。
- 确认 baseline loss 能正常下降，再进入 LoRA 对比。
- 如果 baseline 本身不稳定，后面的 LoRA 结果就没有可解释性。

### Step 3（CPU 项目设计）: 比较 LoRA 配置与资源账本

把 LoRA adapter 插到 attention projection 或 MLP linear layer 上，只训练低秩旁路。

- 冻结底座权重，只让 LoRA 的 `A / B` 矩阵参与训练。
- 先计算单层 LoRA 参数量，再估算多层插入后的总可训练参数量。
- 用同样的 batch、输入长度、训练步数和评估方式比较 LoRA 与 baseline。
- 重点看三个问题：参数量省了多少，显存 / 速度是否改善，train/val loss 是否仍然正常。

### Step 4（CPU 项目设计）: 按约束输出项目结论

最后把 LoRA 和 baseline 放到同一张表里，说明这次微调方案是否值得采用。

- 输出 baseline vs LoRA 对比表，至少包含 trainable params、param ratio、step time、peak memory、train loss、val loss。
- 写清楚 LoRA 节省的是训练参数和优化器状态，不等于底座模型权重不存在。
- 记录本次 target modules、rank、alpha、dropout、学习率、effective batch 和 scheduler，方便后续复现实验。
- 保存 adapter，并记录 tokenizer、special tokens、merge 检查和最小生成样例检查。
- 如果效果不足，下一轮优先调整 rank、插层范围、学习率或 gradient accumulation。
- 最终产物应回答：数据和 loss 是否可信，LoRA 少训练了多少参数，换来了多少显存 / 速度收益，val loss 损失是否还能接受，adapter 是否可以交付。

### Step 5（CPU 代码练习）: 最小代码模板

上面的 Step 1-4 是完整 LoRA 微调项目流程。下面的代码实现其中最小、可复用的六块：数据审计、loss mask 核对、项目配置、LoRA 参数账本、结果汇总和交付检查。
#### 图解：09-13 如何收束到 LoRA 项目报告

`60` 不重复实现训练循环，而是把前面几节已经跑通的机制收成一份可复现的项目报告。

```text
09 SFT data       input_ids / attention_mask / labels
      │
10 LoRA           target modules / rank / alpha / dropout
      │
11 Scheduler      lr schedule counted by optimizer update
      │
12 Accumulation   micro batch -> effective batch
      │
13 E2E report     initial/final train loss + val loss
      │
      ▼
60 LoRA project   data audit + loss mask + parameter ledger + artifacts + decision
```

项目页最小产物：

| 模块 | 必须记录 | 用途 |
|:---|:---|:---|
| 数据 | 样本数、空 response、重复样本、超长样本 | 证明训练输入可信 |
| Loss | supervised tokens、padding supervised tokens | 证明 loss 口径正确 |
| 配置 | target modules、rank、alpha、dropout、lr、effective batch | 保证可复现 |
| 账本 | trainable params、param ratio | 证明 LoRA 是否省参数 |
| 训练结果 | train/val loss、step time、peak memory | 判断效果和成本 |
| 交付 | adapter、tokenizer、merge check、sanity generation | 判断是否能交付 |
| 决策 | accept / tune / reject | 输出项目结论 |

60 作为 60–65 的统一模板，最终报告外层使用 `fine-tuning-project/v1`：`config / baseline / candidates / quality / resources / artifacts / decision / environment`。后续 62–65 只替换项目特有指标，不改变这组公共区域。

#### 图解：微调项目 v2 的交付链路

```text
training data ──► data audit ──► loss mask check ──► baseline run
                                                        │
                                                        ▼
LoRA config ──► adapter training ──► metric comparison ──► artifact check ──► final decision
```


```python
import math

```


```python

# TODO: 完成 LoRA 项目的 5 个核心判断：数据审计、loss 核对、项目汇总、交付检查和最终决策
# 目标：从 09-13 的训练闭环收束到 baseline vs LoRA 项目交付报告

def audit_sft_examples(examples, max_total_chars):
    """审计 SFT 样本，输出训练前最小数据可信度摘要。"""
    # ==========================================
    # TODO 1: 审计 SFT 样本
    # 提示：检查样本数、空 response、重复 prompt/response 和超长样本。
    # ==========================================
    # total_samples = ???
    # empty_response_count = ???
    # duplicate_count = ???
    # over_length_count = ???
    # avg_total_chars = ???
    return {
        'total_samples': total_samples,
        'empty_response_count': empty_response_count,
        'duplicate_count': duplicate_count,
        'over_length_count': over_length_count,
        'avg_total_chars': round(avg_total_chars, 2),
    }

def loss_mask_report(attention_mask, labels, ignore_index=-100):
    """汇总真正参与监督损失的 token 口径。"""
    # ==========================================
    # TODO 2: 核对 loss mask
    # 提示：labels != -100 的 token 会参与 loss；attention_mask == 0 的 padding 不应参与 loss。
    # ==========================================
    # total_tokens = ???
    # non_padding_tokens = ???
    # supervised_tokens = ???
    # padding_supervised_tokens = ???
    # supervised_ratio = ???
    return {
        'total_tokens': total_tokens,
        'non_padding_tokens': non_padding_tokens,
        'supervised_tokens': supervised_tokens,
        'padding_supervised_tokens': padding_supervised_tokens,
        'supervised_ratio': round(supervised_ratio, 4),
    }

# 给定实现：汇总 LoRA 项目配置
def build_lora_project_config(
    base_model,
    target_modules,
    rank,
    alpha,
    dropout,
    learning_rate,
    micro_batch_size,
    accum_steps,
    scheduler,
):
    """打包一次 LoRA 训练的最小复现实验配置。"""
    effective_batch_size = micro_batch_size * accum_steps
    return {
        'base_model': base_model,
        'target_modules': target_modules,
        'rank': rank,
        'alpha': alpha,
        'dropout': dropout,
        'learning_rate': learning_rate,
        'micro_batch_size': micro_batch_size,
        'accum_steps': accum_steps,
        'effective_batch_size': effective_batch_size,
        'scheduler': scheduler,
    }

# 给定实现：计算单层 LoRA 的可训练参数量
def lora_trainable_params(in_dim, out_dim, rank):
    """估算单层 LoRA 需要训练的参数量。"""
    trainable_params = rank * (in_dim + out_dim)
    return trainable_params

# 给定实现：计算完整线性层的参数量
def full_linear_params(in_dim, out_dim):
    """计算对应完整线性层的参数量。"""
    total_params = in_dim * out_dim
    return total_params

# 给定实现：计算 LoRA 参数占比
def lora_param_ratio(in_dim, out_dim, rank):
    """计算 LoRA 可训练参数占完整层参数的比例。"""
    trainable = lora_trainable_params(in_dim, out_dim, rank)
    total = full_linear_params(in_dim, out_dim)
    ratio = trainable / total
    return ratio

def summarize_lora_project(baseline_metrics, lora_metrics):
    """把 baseline 与 LoRA 指标收束成项目对比摘要。"""
    # ==========================================
    # TODO 3: 汇总 baseline 和 LoRA 的项目指标
    # 提示：这里重点只补 5 个项目判断量，资源类 delta = baseline - lora；loss delta = lora - baseline。
    # ==========================================
    # param_reduction = 1.0 - ??? / ???
    # memory_delta = ??? - ???
    # time_delta = ??? - ???
    # train_loss_delta = ??? - ???
    # val_loss_delta = ??? - ???
    return {
        'param_reduction': round(param_reduction, 4),
        'peak_mem_delta_mb': round(memory_delta, 2),
        'step_time_delta_ms': round(time_delta, 2),
        'final_train_loss_delta': round(train_loss_delta, 4),
        'final_val_loss_delta': round(val_loss_delta, 4),
    }

# 给定实现：记录 adapter 交付物
def build_adapter_artifact_record(adapter_path, tokenizer_path, merge_checked, sanity_generation_checked):
    """记录 adapter 交付所需的最小产物信息。"""
    return {
        'adapter_path': adapter_path,
        'tokenizer_path': tokenizer_path,
        'merge_checked': merge_checked,
        'sanity_generation_checked': sanity_generation_checked,
    }

# 给定实现：组装 60 项目的统一结果报告
def build_lora_project_report(config, baseline, candidates, quality, resources, artifacts, decision, environment=None):
    """组装 fine-tuning-project/v1 的公共报告外壳。"""
    return {
        'schema_version': 'fine-tuning-project/v1',
        'project': '60_lora_fine_tuning',
        'stage': 'project_decision',
        'config': config,
        'baseline': baseline,
        'candidates': candidates,
        'quality': quality,
        'resources': resources,
        'artifacts': artifacts,
        'decision': decision,
        'environment': environment or {},
    }

def check_lora_project_readiness(data_audit, mask_report, artifact_record):
    """检查数据、loss 口径和交付产物是否达到上线前闸门。"""
    # ==========================================
    # TODO 4: 检查项目是否可以交付
    # 提示：这里只补关键闸门条件，把 issue 名称按下面给定字符串挂上去即可。
    # ==========================================
    issues = []
    # 这里只做训练前闸门判断，不在这里做最终 accept / reject。
    # if data_audit['empty_response_count'] > ???:
    #     issues.append('empty_response')
    # if data_audit['duplicate_count'] > ???:
    #     issues.append('duplicate_examples')
    # if mask_report['padding_supervised_tokens'] > ???:
    #     issues.append('padding_supervised')
    # if mask_report['supervised_tokens'] == ???:
    #     issues.append('no_supervised_tokens')
    # if not artifact_record['merge_checked']:
    #     issues.append('merge_not_checked')
    # if not artifact_record['sanity_generation_checked']:
    #     issues.append('sanity_generation_not_checked')
    return {'ready': len(issues) == 0, 'issues': issues}

def recommend_lora_decision(summary, readiness, min_param_reduction=0.5, max_val_loss_delta=0.03, min_peak_mem_delta_mb=128.0, min_step_time_delta_ms=-3.0):
    """根据项目摘要输出 accept / tune / reject 结论。"""
    # ==========================================
    # TODO 5: 根据项目汇总和交付检查给出采用建议
    # 规则：
    # - 数据、loss 或 artifact 未准备好：tune
    # - 参数节省达标、val loss 损失可接受，且显存收益足够或速度没有明显恶化：accept
    # - 参数节省达标、val loss 可接受，但显存收益偏弱且速度变慢：tune
    # - 参数节省达标但 val loss 损失偏大：tune
    # - 参数节省不达标：reject
    # ==========================================
    # 资源判断不是只看省了多少参数，还要看显存收益和速度代价是否值得。
    # memory_gain_ok = summary['peak_mem_delta_mb'] >= ???
    # speed_not_too_bad = summary['step_time_delta_ms'] >= ???
    # if ???:
    #     decision = ???
    #     reason = ???
    # elif ???:
    #     decision = ???
    #     reason = ???
    # elif ??? and not (memory_gain_ok or speed_not_too_bad):
    #     decision = ???
    #     reason = ???
    # elif ???:
    #     decision = ???
    #     reason = ???
    # else:
    #     decision = ???
    #     reason = ???
    return {'decision': decision, 'reason': reason}

```


```python
# 测试你的实现
def test_lora_project_template():
    try:
        examples = [
            {'prompt': '问：什么是 LoRA？', 'response': '答：LoRA 是低秩适配方法。'},
            {'prompt': '问：如何检查 loss？', 'response': '答：检查 labels 中参与监督的 token。'},
            {'prompt': '问：什么是 LoRA？', 'response': '答：LoRA 是低秩适配方法。'},
            {'prompt': '问：空回答？', 'response': ''},
        ]
        audit = audit_sft_examples(examples, max_total_chars=30)
        assert audit['total_samples'] == 4, "样本数统计不正确！"
        assert audit['empty_response_count'] == 1, "空 response 统计不正确！"
        assert audit['duplicate_count'] == 1, "重复样本统计不正确！"
        assert audit['over_length_count'] == 1, "超长样本统计不正确！"

        mask = [[1, 1, 1, 0], [1, 1, 0, 0]]
        labels = [[-100, 7, 8, -100], [-100, 9, -100, 3]]
        report = loss_mask_report(mask, labels)
        assert report['total_tokens'] == 8, "total_tokens 统计不正确！"
        assert report['non_padding_tokens'] == 5, "non_padding_tokens 统计不正确！"
        assert report['supervised_tokens'] == 4, "supervised_tokens 统计不正确！"
        assert report['padding_supervised_tokens'] == 1, "padding_supervised_tokens 统计不正确！"
        assert report['supervised_ratio'] == 0.8, "supervised_ratio 计算不正确！"

        config = build_lora_project_config(
            base_model='tiny-llama',
            target_modules=['q_proj', 'v_proj'],
            rank=8,
            alpha=16,
            dropout=0.05,
            learning_rate=2e-4,
            micro_batch_size=2,
            accum_steps=4,
            scheduler='wsd-cosine',
        )
        assert config['effective_batch_size'] == 8, "effective_batch_size 计算不正确！"
        assert config['target_modules'] == ['q_proj', 'v_proj'], "target_modules 应保留原始配置！"

        trainable = lora_trainable_params(8, 8, 2)
        total = full_linear_params(8, 8)
        ratio = lora_param_ratio(8, 8, 2)

        assert trainable == 32, "LoRA 可训练参数量计算不正确！"
        assert total == 64, "完整线性层参数量计算不正确！"
        assert abs(ratio - 0.5) < 1e-12, "LoRA 参数占比计算不正确！"

        baseline = {
            'trainable_params': 1000,
            'step_time_ms': 20.0,
            'peak_mem_mb': 1024.0,
            'final_train_loss': 0.40,
            'final_val_loss': 0.50,
        }
        lora = {
            'trainable_params': 100,
            'step_time_ms': 22.0,
            'peak_mem_mb': 768.0,
            'final_train_loss': 0.42,
            'final_val_loss': 0.52,
        }
        summary = summarize_lora_project(baseline, lora)
        assert summary['param_reduction'] == 0.9, "param_reduction 计算不正确！"
        assert summary['peak_mem_delta_mb'] == 256.0, "peak_mem_delta_mb 计算不正确！"
        assert summary['step_time_delta_ms'] == -2.0, "step_time_delta_ms 计算不正确！"
        assert summary['final_train_loss_delta'] == 0.02, "final_train_loss_delta 计算不正确！"
        assert summary['final_val_loss_delta'] == 0.02, "final_val_loss_delta 计算不正确！"

        artifact = build_adapter_artifact_record(
            adapter_path='outputs/lora-adapter',
            tokenizer_path='outputs/tokenizer',
            merge_checked=True,
            sanity_generation_checked=True,
        )
        project_report = build_lora_project_report(
            config={'model': 'tiny-llama', 'dtype': 'bf16', 'seed': 42},
            baseline=baseline,
            candidates=[{'name': 'lora', **lora}],
            quality={'train_loss': 0.42, 'val_loss': 0.52, 'task_metrics': {}},
            resources={'trainable_params': 100, 'peak_memory_mb': 768.0, 'step_time_ms': 22.0},
            artifacts={'adapter': artifact},
            decision={'decision': 'accept', 'reason': 'test'},
        )
        for section in ('config', 'baseline', 'candidates', 'quality', 'resources', 'artifacts', 'decision', 'environment'):
            assert section in project_report, f'报告缺少 {section} 区域！'
        clean_audit = {'total_samples': 2, 'empty_response_count': 0, 'duplicate_count': 0, 'over_length_count': 0, 'avg_total_chars': 12.0}
        clean_report = {'total_tokens': 8, 'non_padding_tokens': 5, 'supervised_tokens': 3, 'padding_supervised_tokens': 0, 'supervised_ratio': 0.6}
        readiness = check_lora_project_readiness(clean_audit, clean_report, artifact)
        assert readiness['ready'] is True, "干净项目应允许交付！"

        dirty_readiness = check_lora_project_readiness(audit, report, artifact)
        assert dirty_readiness['ready'] is False, "存在数据或 mask 问题时不能交付！"
        assert 'empty_response' in dirty_readiness['issues'], "应报告空 response 问题！"
        assert 'padding_supervised' in dirty_readiness['issues'], "应报告 padding 参与 loss 问题！"

        decision = recommend_lora_decision(summary, readiness, min_param_reduction=0.5, max_val_loss_delta=0.03, min_peak_mem_delta_mb=128.0, min_step_time_delta_ms=-3.0)
        assert decision['decision'] == 'accept', "LoRA 决策应为 accept！"

        assert recommend_lora_decision(summary, dirty_readiness)['decision'] == 'tune', "交付检查未通过时应建议 tune！"

        worse_summary = dict(summary)
        worse_summary['final_val_loss_delta'] = 0.08
        assert recommend_lora_decision(worse_summary, readiness)['decision'] == 'tune', "val loss 损失过大时应建议 tune！"

        weak_summary = dict(summary)
        weak_summary['param_reduction'] = 0.2
        assert recommend_lora_decision(weak_summary, readiness)['decision'] == 'reject', "参数节省不足时应建议 reject！"

        tradeoff_summary = dict(summary)
        tradeoff_summary['peak_mem_delta_mb'] = 32.0
        tradeoff_summary['step_time_delta_ms'] = -6.0
        assert recommend_lora_decision(tradeoff_summary, readiness)['decision'] == 'tune', "显存收益偏弱且速度恶化时应建议 tune！"

        print("✅ LoRA 项目数据审计、loss 核对、账本、交付检查和决策代码通过基础校验。")

    except NotImplementedError:
        print("请先完成 TODO 代码！")
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError, RuntimeError) as e:
        if isinstance(e, AttributeError):
            print("代码未完成，无法找到必要的属性")
        elif isinstance(e, NameError):
            print("代码可能未完成，导致了变量未定义")
        elif isinstance(e, TypeError):
            print("代码可能未完成，导致了操作错误")
        elif isinstance(e, ValueError):
            print("代码可能未完成，导致了数值错误")
        elif isinstance(e, AssertionError):
            print(f"❌ 测试失败: {e}")
        elif isinstance(e, RuntimeError):
            print("代码可能未完成，导致了运行时错误")
        else:
            print("代码可能未完成，导致了断言失败")
        raise NotImplementedError("请先完成 TODO 代码！") from e
    except Exception as e:
        print(f"❌ 发生未知异常: {e}")
        raise


test_lora_project_template()

```

🛑 **STOP HERE** 🛑

## 参考代码与解析

### 代码


```python

# TODO 1: 审计 SFT 样本
def audit_sft_examples(examples, max_total_chars):
    seen = set()
    total_chars = 0
    empty_response_count = 0
    duplicate_count = 0
    over_length_count = 0

    for example in examples:
        prompt = example.get('prompt', '')
        response = example.get('response', '')
        pair = (prompt, response)
        total = len(prompt) + len(response)

        total_chars += total
        if not response.strip():
            empty_response_count += 1
        if pair in seen:
            duplicate_count += 1
        else:
            seen.add(pair)
        if total > max_total_chars:
            over_length_count += 1

    total_samples = len(examples)
    avg_total_chars = total_chars / total_samples if total_samples else 0.0
    return {
        'total_samples': total_samples,
        'empty_response_count': empty_response_count,
        'duplicate_count': duplicate_count,
        'over_length_count': over_length_count,
        'avg_total_chars': round(avg_total_chars, 2),
    }

# TODO 2: 核对 loss mask
def loss_mask_report(attention_mask, labels, ignore_index=-100):
    mask_flat = [value for row in attention_mask for value in row]
    labels_flat = [value for row in labels for value in row]

    if len(mask_flat) != len(labels_flat):
        raise ValueError('attention_mask and labels must have the same number of tokens')

    total_tokens = len(labels_flat)
    non_padding_tokens = sum(1 for mask in mask_flat if mask == 1)
    supervised_tokens = sum(1 for label in labels_flat if label != ignore_index)
    padding_supervised_tokens = sum(
        1 for mask, label in zip(mask_flat, labels_flat)
        if mask == 0 and label != ignore_index
    )
    supervised_ratio = supervised_tokens / non_padding_tokens if non_padding_tokens else 0.0
    return {
        'total_tokens': total_tokens,
        'non_padding_tokens': non_padding_tokens,
        'supervised_tokens': supervised_tokens,
        'padding_supervised_tokens': padding_supervised_tokens,
        'supervised_ratio': round(supervised_ratio, 4),
    }

# 给定实现：汇总 LoRA 项目配置
def build_lora_project_config(
    base_model,
    target_modules,
    rank,
    alpha,
    dropout,
    learning_rate,
    micro_batch_size,
    accum_steps,
    scheduler,
):
    effective_batch_size = micro_batch_size * accum_steps
    return {
        'base_model': base_model,
        'target_modules': target_modules,
        'rank': rank,
        'alpha': alpha,
        'dropout': dropout,
        'learning_rate': learning_rate,
        'micro_batch_size': micro_batch_size,
        'accum_steps': accum_steps,
        'effective_batch_size': effective_batch_size,
        'scheduler': scheduler,
    }

# 给定实现：计算单层 LoRA 的可训练参数量
def lora_trainable_params(in_dim, out_dim, rank):
    """Estimate trainable LoRA parameters for a single linear layer."""
    trainable_params = rank * (in_dim + out_dim)
    return trainable_params

# 给定实现：计算完整线性层的参数量
def full_linear_params(in_dim, out_dim):
    total_params = in_dim * out_dim
    return total_params

# 给定实现：计算 LoRA 参数占比
def lora_param_ratio(in_dim, out_dim, rank):
    trainable = lora_trainable_params(in_dim, out_dim, rank)
    total = full_linear_params(in_dim, out_dim)
    ratio = trainable / total
    return ratio

# TODO 3: 汇总 baseline 和 LoRA 项目指标
def summarize_lora_project(baseline_metrics, lora_metrics):
    param_reduction = 1.0 - lora_metrics['trainable_params'] / baseline_metrics['trainable_params']
    memory_delta = baseline_metrics['peak_mem_mb'] - lora_metrics['peak_mem_mb']
    time_delta = baseline_metrics['step_time_ms'] - lora_metrics['step_time_ms']
    train_loss_delta = lora_metrics['final_train_loss'] - baseline_metrics['final_train_loss']
    val_loss_delta = lora_metrics['final_val_loss'] - baseline_metrics['final_val_loss']
    return {
        'param_reduction': round(param_reduction, 4),
        'peak_mem_delta_mb': round(memory_delta, 2),
        'step_time_delta_ms': round(time_delta, 2),
        'final_train_loss_delta': round(train_loss_delta, 4),
        'final_val_loss_delta': round(val_loss_delta, 4),
    }

# 给定实现：记录 adapter 交付物
def build_adapter_artifact_record(adapter_path, tokenizer_path, merge_checked, sanity_generation_checked):
    return {
        'adapter_path': adapter_path,
        'tokenizer_path': tokenizer_path,
        'merge_checked': merge_checked,
        'sanity_generation_checked': sanity_generation_checked,
    }

# 给定实现：组装 60 项目的统一结果报告
def build_lora_project_report(config, baseline, candidates, quality, resources, artifacts, decision, environment=None):
    return {
        'schema_version': 'fine-tuning-project/v1',
        'project': '60_lora_fine_tuning',
        'stage': 'project_decision',
        'config': config,
        'baseline': baseline,
        'candidates': candidates,
        'quality': quality,
        'resources': resources,
        'artifacts': artifacts,
        'decision': decision,
        'environment': environment or {},
    }

# TODO 4: 检查项目是否可以交付
def check_lora_project_readiness(data_audit, mask_report, artifact_record):
    issues = []
    if data_audit['empty_response_count'] > 0:
        issues.append('empty_response')
    if data_audit['duplicate_count'] > 0:
        issues.append('duplicate_examples')
    if mask_report['padding_supervised_tokens'] > 0:
        issues.append('padding_supervised')
    if mask_report['supervised_tokens'] == 0:
        issues.append('no_supervised_tokens')
    if not artifact_record['merge_checked']:
        issues.append('merge_not_checked')
    if not artifact_record['sanity_generation_checked']:
        issues.append('sanity_generation_not_checked')
    return {'ready': len(issues) == 0, 'issues': issues}

# TODO 5: 根据项目汇总和交付检查给出采用建议
def recommend_lora_decision(summary, readiness, min_param_reduction=0.5, max_val_loss_delta=0.03, min_peak_mem_delta_mb=128.0, min_step_time_delta_ms=-3.0):
    memory_gain_ok = summary['peak_mem_delta_mb'] >= min_peak_mem_delta_mb
    speed_not_too_bad = summary['step_time_delta_ms'] >= min_step_time_delta_ms
    if not readiness['ready']:
        decision = 'tune'
        reason = '数据、loss mask 或 adapter 交付检查未通过，先修复项目可信度问题。'
    elif summary['param_reduction'] < min_param_reduction:
        decision = 'reject'
        reason = '参数节省不足，LoRA 没有带来足够训练成本收益。'
    elif summary['final_val_loss_delta'] > max_val_loss_delta:
        decision = 'tune'
        reason = '参数节省达标，但验证集 loss 损失偏大，优先调 rank、target modules 或学习率。'
    elif not (memory_gain_ok or speed_not_too_bad):
        decision = 'tune'
        reason = '参数节省和验证损失可接受，但显存收益偏弱且速度恶化，优先继续调 rank、插层范围或 batch 配置。'
    else:
        decision = 'accept'
        reason = '参数节省达标，验证集损失可接受，交付检查通过，可以保留当前 LoRA 配置。'
    return {'decision': decision, 'reason': reason}

examples = [
    {'prompt': '问：什么是 LoRA？', 'response': '答：LoRA 是低秩适配方法。'},
    {'prompt': '问：如何检查 loss？', 'response': '答：检查 labels 中参与监督的 token。'},
]
audit = audit_sft_examples(examples, max_total_chars=64)
print(audit)

mask_report = loss_mask_report(
    attention_mask=[[1, 1, 1, 0]],
    labels=[[-100, 7, 8, -100]],
)
print(mask_report)

config = build_lora_project_config(
    base_model='tiny-llama',
    target_modules=['q_proj', 'v_proj'],
    rank=8,
    alpha=16,
    dropout=0.05,
    learning_rate=2e-4,
    micro_batch_size=2,
    accum_steps=4,
    scheduler='wsd-cosine',
)
print(config)

for hidden_size, rank in [(4096, 8), (4096, 16), (8192, 16)]:
    trainable = lora_trainable_params(hidden_size, hidden_size, rank)
    total = full_linear_params(hidden_size, hidden_size)
    ratio = lora_param_ratio(hidden_size, hidden_size, rank)
    print(f"hidden={hidden_size}, rank={rank} -> trainable={trainable:,}, full={total:,}, ratio={ratio:.4%}")

baseline = {'trainable_params': 1000, 'step_time_ms': 20.0, 'peak_mem_mb': 1024.0, 'final_train_loss': 0.40, 'final_val_loss': 0.50}
lora = {'trainable_params': 100, 'step_time_ms': 22.0, 'peak_mem_mb': 768.0, 'final_train_loss': 0.42, 'final_val_loss': 0.52}
summary = summarize_lora_project(baseline, lora)
artifact = build_adapter_artifact_record('outputs/lora-adapter', 'outputs/tokenizer', True, True)
readiness = check_lora_project_readiness(audit, mask_report, artifact)
project_report = build_lora_project_report(
    config={'model': 'tiny-llama', 'dtype': 'bf16', 'seed': 42},
    baseline=baseline,
    candidates=[{'name': 'lora', **lora}],
    quality={'train_loss': lora['final_train_loss'], 'val_loss': lora['final_val_loss'], 'task_metrics': {}},
    resources={'trainable_params': lora['trainable_params'], 'peak_memory_mb': lora['peak_mem_mb'], 'step_time_ms': lora['step_time_ms']},
    artifacts={'adapter': artifact},
    decision=recommend_lora_decision(summary, readiness),
)
print(summary)
print(readiness)
print(project_report)
print(recommend_lora_decision(summary, readiness))

```

### 解析

这一版题目区保留 `5` 个核心 TODO：数据审计、loss 核对、项目汇总、交付检查和最终决策；其余配置打包、参数公式和 artifact 字段整理改成给定实现，把练习重点收回到项目判断本身。


**1. TODO 1: 审计 SFT 样本**
- **实现方式**：遍历 `prompt / response` 样本，统计总样本数、空 response、重复样本、超长样本和平均长度。
- **关键点**：微调前先确认数据可信。空 response 会让样本没有有效监督，重复样本会放大小数据过拟合风险，超长样本会改变截断和显存口径。
- **项目意义**：这一步把第 09 节的数据正确性从单条样本扩展到项目级数据集检查。

**2. TODO 2: 核对 loss mask**
- **实现方式**：把 `attention_mask` 和 `labels` 展平后对齐检查，统计非 padding token、参与监督的 token，以及 padding 中错误参与 loss 的 token。
- **关键点**：`labels != -100` 的 token 会参与 loss；`attention_mask == 0` 的 padding token 不应该参与 loss。
- **项目意义**：这是 SFT 项目最关键的正确性检查之一。loss 下降不代表训练对了，必须确认监督 token 的位置正确。

**给定实现 A：汇总 LoRA 项目配置**
- **实现方式**：把 base model、target modules、rank、alpha、dropout、学习率、micro batch、accum steps 和 scheduler 放进同一个配置对象。
- **关键点**：`effective_batch_size = micro_batch_size * accum_steps`，这要和第 12 节的梯度累积口径一致。
- **项目意义**：这部分更偏复现实验的脚手架，因此直接给出实现，不占用核心 TODO 配额。

**给定实现 B：计算单层 LoRA 的可训练参数量**
- **实现方式**：LoRA 为一个线性层增加两个低秩矩阵，`A` 的参数量是 `rank * in_dim`，`B` 的参数量是 `rank * out_dim`，合起来是 `rank * (in_dim + out_dim)`。
- **关键点**：这里统计的是 LoRA adapter 的可训练参数，不包括冻结的底座权重。
- **项目意义**：这是 LoRA 微调项目的第一张账本，但公式本身偏机械，因此也改为给定实现。

**给定实现 C：计算完整线性层的参数量**
- **实现方式**：完整线性层的 weight 参数量是 `in_dim * out_dim`。本节为了突出主线，不额外统计 bias。
- **关键点**：全参线性层是 baseline，用来衡量 LoRA 的参数节省比例。
- **技术细节**：如果真实模型中包含 bias 或多个投影层，需要把这些层逐项累加。

**给定实现 D：计算 LoRA 参数占比**
- **实现方式**：先分别计算 LoRA 参数量和完整线性层参数量，再用 `trainable / total` 得到参数占比。
- **关键点**：参数占比越小，说明同一层上需要训练和保存的 adapter 越少。
- **项目意义**：这个比例可以和 step time、peak memory、train/val loss 一起放进项目报告，但不需要读者再为基础公式分散注意力。

**3. TODO 3: 汇总 baseline 和 LoRA 项目指标**
- **实现方式**：资源类指标使用 `baseline - LoRA`，正数表示 LoRA 更省或更快；loss 指标使用 `LoRA - baseline`，正数表示 LoRA 效果更差。
- **关键点**：train loss 和 val loss 要分开看。train loss 接近不代表泛化可接受，最终决策更应该看 val loss delta。
- **工程判断**：如果参数和显存明显下降，但 val loss 损失很小，LoRA 方案通常值得保留；如果 val loss 明显变差，需要继续调整 rank、插层位置或学习率。

**给定实现 E：记录 adapter 交付物**
- **实现方式**：记录 adapter 路径、tokenizer 路径、merge 检查和最小生成样例检查。
- **关键点**：LoRA 微调的交付物不是一行 loss，而是一组可加载、可复现、能做 sanity check 的 artifact。
- **项目意义**：这一步把训练实验推进到交付边界，但字段整理本身不应挤占核心 TODO。

**4. TODO 4: 检查项目是否可以交付**
- **实现方式**：把数据审计、loss mask 报告和 artifact 记录合并检查，返回 `ready` 和问题列表。
- **关键点**：只要存在空 response、padding 参与 loss、无监督 token、merge 未检查或生成样例未检查，就不应该直接把项目判为 accept。
- **项目意义**：这一步让项目报告不只比较指标，也能说明指标是否可信。

**5. TODO 5: 输出采用建议**
- **accept**：交付检查通过，参数节省达标，val loss 损失在阈值内。
- **tune**：交付检查未通过，或参数节省达标但 val loss 损失偏大，或显存收益偏弱且速度恶化。
- **reject**：交付检查通过，但参数节省不足，LoRA 没有带来足够训练成本收益。
- **项目意义**：决策不再只看 LoRA 参数比例，而是同时看数据可信度、loss 口径、artifact 交付、资源收益和效果损失。

## Step 6（可选）：真实模型 LoRA 验证

这一步对应 66 节的真实 backend 分支，但验证对象不同：66 验证推理服务，60 验证真实模型、tokenizer、LoRA adapter、训练 step 和 artifact 保存链路。默认关闭，不影响 CPU-first 练习。

真实运行只完成小规模 smoke test，不等于完整微调效果结论；它主要检查 GPU、模型下载、tokenizer、LoRA 注入、反向传播和 adapter 保存是否连通。要形成正式结论，还需要固定数据集、训练步数、验证集和同口径 baseline。Colab / ModelScope 运行前请先阅读[训练微调项目验证清单](../docs/verification/fine_tuning_projects.md)。

| 实验层级 | 运行位置 | 固定内容 | 可以得出的结论 | 不能得出的结论 |
|:---|:---|:---|:---|:---|
| CPU 题目区 | CPU 或 GPU | 人工构造的小样本、函数输入 | 数据审计、mask、参数账本和决策逻辑正确 | LoRA 训练有效、GPU 显存收益 |
| GPU smoke | 有 CUDA 的单卡 | 模型、tokenizer、LoRA 配置 | 真实模型链路和 adapter 能否保存 | baseline 对比、泛化能力、稳定收益 |
| GPU matched | 目标 GPU | 同一数据切分、dtype、batch、seq_len、steps | baseline 与 LoRA 的资源和 validation loss 差异 | 更大模型、更多数据或其他 GPU 的结论 |


数据可以从 `inline`、Hugging Face、ModelScope 或本地 JSON/JSONL 读取。远程数据集支持 `instruction / input / output` 或 `prompt / response` 字段；真实项目建议使用固定版本、固定抽样数量，并把数据集 ID、来源和审计结果写入报告。


```python
# 只需要修改这一格；默认只运行 CPU 代码。
RUN_MODE = 'cpu'  # cpu / dry_run / real_gpu；dry_run 只检查环境，不训练。
RUN_REAL_TRAINING = False  # 兼容旧入口；real_gpu 模式下再显式打开对应实验。
REAL_MODEL_SOURCE = 'huggingface'  # 模型来源：auto / modelscope / huggingface / local。
REAL_MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'  # 基座模型；小模型便于 Colab 和约 12 GB 显存设备运行。
# 缓存、结果和本地数据路径均由 Notebook 根据仓库根目录自动推导，不需要手填路径。
REAL_DTYPE = 'auto'  # auto 优先 BF16（硬件支持时），否则回退 FP16；也可写 bfloat16 / float16。
REAL_MAX_SEQ_LEN = 256  # 每条样本最大 token 长度；影响截断、显存和 step time。
REAL_STEPS = 3  # Step 6 smoke test 的更新步数；Step 7 使用 MATCHED_STEPS。
REAL_LR = 2e-4
AUTO_INSTALL_REAL_DEPS = True  # 自动安装 transformers / peft / datasets 等当前内核依赖。
REAL_DATA_SOURCE = 'huggingface'  # 数据来源：inline 仅适合 smoke test；正式比较用 huggingface / modelscope / local。
REAL_DATASET_ID = 'tatsu-lab/alpaca'
REAL_DATA_FILE = None  # None 时自动搜索 benchmarks/data/ 和 data/
REAL_MAX_SAMPLES = 32  # 最多读取样本数；baseline 和 LoRA 必须使用同一批数据。
REAL_SEED = 2024  # 模型初始化、训练随机性的种子；不同实验可改变它。
SPLIT_SEED = 42  # 训练/验证集划分种子；跨实验固定，避免验证集随 REAL_SEED 改变。
RUN_REAL_MATCHED = False  # Step 7：需要正式采集时显式改为 True。
MATCHED_BATCH_SIZE = 1  # 每次送入 GPU 的 micro-batch，不是有效 batch 总大小。
MATCHED_VAL_RATIO = 0.2  # 固定留作验证的数据比例。
MATCHED_STEPS = 20  # baseline 和 LoRA 必须使用相同更新步数。

```


```python
# 可选：打开真实模型验证后，自动为当前 Notebook 内核补齐依赖
# 必须先运行上一格配置，再运行这一格。
if (RUN_REAL_TRAINING or RUN_REAL_MATCHED) and AUTO_INSTALL_REAL_DEPS:
    import subprocess
    import sys
    packages = ['transformers', 'peft', 'accelerate', 'datasets', 'httpx[socks]']
    if REAL_MODEL_SOURCE == 'modelscope' or REAL_DATA_SOURCE == 'modelscope':
        packages.append('modelscope')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-U', *packages])
    print('真实模型和数据集依赖安装完成，请继续运行后续单元。')
elif not (RUN_REAL_TRAINING or RUN_REAL_MATCHED):
    print('跳过依赖安装：真实模型验证未开启。')

```


```python
if RUN_REAL_TRAINING:
    import random
    def _real_audit(records, max_total_chars):
        pairs = [(item.get('prompt', ''), item.get('response', '')) for item in records]
        return {'total_samples': len(records), 'empty_response_count': sum(not response.strip() for _, response in pairs), 'duplicate_count': len(pairs) - len(set(pairs)), 'over_length_count': sum(len(prompt) + len(response) > max_total_chars for prompt, response in pairs), 'avg_total_chars': round(sum(len(prompt) + len(response) for prompt, response in pairs) / len(records), 2) if records else 0.0}
    def _real_report(**sections):
        return {'schema_version': 'fine-tuning-project/v1', 'project': '60_lora_fine_tuning', 'stage': 'project_decision', **sections}
    import json
    import os
    import sys
    import time
    from pathlib import Path

    import torch
    random.seed(REAL_SEED)
    torch.manual_seed(REAL_SEED)
    torch.cuda.manual_seed_all(REAL_SEED)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError('真实 LoRA 验证需要 peft：请先安装 transformers peft accelerate。') from exc

    project_root = next((path for path in [Path.cwd(), *Path.cwd().parents] if (path / 'tools').is_dir()), None)
    if project_root is None:
        raise RuntimeError('未找到项目根目录，请从仓库根目录启动 Notebook。')
    os.chdir(project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from tools.model_runtime import resolve_model

    if not torch.cuda.is_available():
        raise RuntimeError('RUN_REAL_TRAINING=True 需要可用 CUDA GPU。')
    device = torch.device('cuda')
    if REAL_DTYPE == 'auto':
        try:
            bf16_supported = torch.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:
            bf16_supported = torch.cuda.get_device_capability(0)[0] >= 8
        dtype = torch.bfloat16 if bf16_supported else torch.float16
    else:
        dtype = getattr(torch, REAL_DTYPE)
    model_path = resolve_model(REAL_MODEL_ID, REAL_MODEL_SOURCE)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM',
    ))
    model.to(device).train()
    model.print_trainable_parameters()

    if REAL_DATA_SOURCE == 'inline':
        examples = [
            {'prompt': '用一句话解释 LoRA。', 'response': 'LoRA 是一种低秩参数高效微调方法。'},
            {'prompt': '用一句话解释梯度累积。', 'response': '梯度累积通过多次小批量反向传播模拟更大的 batch。'},
            {'prompt': '用一句话解释验证集。', 'response': '验证集用于检查模型对未参与训练样本的泛化表现。'},
            {'prompt': '用一句话解释 adapter。', 'response': 'adapter 是挂载在基座模型上的可训练增量参数。'},
        ]
    elif REAL_DATA_SOURCE == 'local':
        search_roots = [project_root / 'benchmarks' / 'data', project_root / 'data']
        candidates = [path for root in search_roots if root.exists() for path in root.glob('*') if path.suffix.lower() in {'.json', '.jsonl'}]
        data_path = Path(REAL_DATA_FILE) if REAL_DATA_FILE else (candidates[0] if candidates else None)
        if data_path is None:
            raise FileNotFoundError('未在 benchmarks/data 或 data 中找到 JSON/JSONL 数据文件')
        if not data_path.exists():
            raise FileNotFoundError(f'本地数据文件不存在：{data_path}')
        if data_path.suffix.lower() == '.jsonl':
            records = [json.loads(line) for line in data_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        else:
            records = json.loads(data_path.read_text(encoding='utf-8'))
    elif REAL_DATA_SOURCE == 'huggingface':
        from datasets import load_dataset
        records = load_dataset(REAL_DATASET_ID, split='train')
    elif REAL_DATA_SOURCE == 'modelscope':
        from modelscope.msdatasets import MsDataset
        records = MsDataset.load(REAL_DATASET_ID, split='train')
        if hasattr(records, 'to_hf_dataset'):
            records = records.to_hf_dataset()
    else:
        raise ValueError('REAL_DATA_SOURCE 必须是 inline / huggingface / modelscope / local')
    if REAL_DATA_SOURCE != 'inline':
        examples = []
        for record in list(records)[:REAL_MAX_SAMPLES]:
            if 'prompt' in record and 'response' in record:
                prompt, response = record['prompt'], record['response']
            else:
                prompt = str(record.get('instruction', '')) + str(record.get('input', ''))
                response = record.get('output', record.get('response', ''))
            if prompt and response:
                examples.append({'prompt': str(prompt), 'response': str(response)})
        if not examples:
            raise ValueError('数据集中没有识别到 prompt/response 或 instruction/input/output 字段')
    examples = examples[:REAL_MAX_SAMPLES]
    data_audit = _real_audit(examples, max_total_chars=REAL_MAX_SEQ_LEN * 4)
    texts = [item['prompt'] + '\n' + item['response'] for item in examples]
    batch = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=REAL_MAX_SEQ_LEN)
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = input_ids.masked_fill(attention_mask == 0, -100)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=REAL_LR)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    losses = []
    for _ in range(REAL_STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    output_dir = project_root / 'benchmarks' / 'results' / '60_real_lora'
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_dir / 'adapter'
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(output_dir / 'tokenizer')
    report_builder = globals().get('build_lora_project_report', _real_report)
    report = report_builder(
        config={'model': REAL_MODEL_ID, 'model_path': model_path, 'dtype': str(dtype), 'batch_size': len(examples), 'seq_len': REAL_MAX_SEQ_LEN, 'steps': REAL_STEPS, 'seed': REAL_SEED},
        baseline={'status': 'not_run', 'reason': 'real smoke test does not run a matched full-parameter baseline'},
        candidates=[{'name': 'real_lora_smoke', 'status': 'ok', 'losses': losses}],
        quality={'train_loss': losses[-1], 'val_loss': None, 'task_metrics': {}, 'data_audit': data_audit, 'quality_status': 'smoke_only'},
        resources={'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad), 'peak_memory_mb': round(torch.cuda.max_memory_allocated() / 2**20, 2), 'step_time_ms': round(elapsed / REAL_STEPS * 1000, 2), 'tokens_per_s': round(input_ids.numel() * REAL_STEPS / elapsed, 2)},
        artifacts={'adapter': str(adapter_dir), 'tokenizer': str(output_dir / 'tokenizer'), 'report': str(output_dir / '60_real_lora.json')},
        decision={'decision': 'tune', 'reason': '真实 LoRA smoke test 已完成，但尚未与同口径 baseline 和验证集比较。', 'next_action': 'run_matched_baseline_and_validation'},
        environment={'python': sys.version, 'torch': torch.__version__, 'torch_cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(0)},
    )
    (output_dir / '60_real_lora.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
else:
    print('跳过真实模型验证：保持 CPU-first 模式。')

```


```python
# dry_run 只做环境预检：不下载模型、不读取数据、不创建 optimizer。
if globals().get('RUN_MODE', 'cpu') == 'dry_run':
    import importlib.util
    import sys
    from pathlib import Path

    project_root = next((path for path in [Path.cwd(), *Path.cwd().parents] if (path / 'tools').is_dir()), None)
    if project_root is None:
        raise RuntimeError('未找到项目根目录，请先 clone 仓库或从仓库启动 Notebook。')
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        import torch
        from tools.fine_tuning_project_runtime import preflight_runtime
        preflight = preflight_runtime(torch, run_mode='dry_run')
    except ImportError as exc:
        preflight = {'run_mode': 'dry_run', 'ready': False, 'reasons': [f'缺少运行依赖：{exc}'], 'next_action': 'install_dependencies'}
    dependencies = {name: importlib.util.find_spec(name) is not None for name in ('transformers', 'peft', 'datasets')}
    preflight['dependencies'] = dependencies
    print('dry_run 预检结果：')
    print(preflight)
    print('提示：预检通过后，将 RUN_MODE 改为 real_gpu，并显式打开对应 GPU 实验开关。')

```

## Step 7（GPU 项目实验，可选）：自动采集 baseline vs LoRA

Step 6 只验证真实 LoRA 链路；Step 7 才用于正式采集。它会自动固定 train/validation 划分，按 batch 分批，并依次运行 full-parameter baseline 与 LoRA。这里比较的是同一模型、同一数据切分、同一 dtype、同一训练步数下的资源与 validation loss，不是完整任务能力评测。

正式采集前必须确认：模型快照、数据集版本和样本数、`SPLIT_SEED`、`MATCHED_BATCH_SIZE`、`REAL_MAX_SEQ_LEN`、`MATCHED_STEPS`、dtype、GPU 和 PyTorch/CUDA 版本均已写入报告。更换其中任一项，都应视为新的实验条件。

默认关闭。打开 `RUN_REAL_MATCHED = True` 后，不需要复制训练代码或填写路径；结果会保存为 `benchmarks/results/60_real_lora/60_real_lora_matched.json`。

### 已有三组真实数据：先看表，再决定是否继续实验

下面三组结果来自同一份真实数据和同一套训练配置，只改变模型初始化用的 `REAL_SEED`；`SPLIT_SEED=42` 在三组中固定。每一组内部都使用同一数据切分、同一 batch、同一序列长度和同一步数，因此可以比较 baseline 与 LoRA；不同 seed 之间用于观察训练波动，不应当作三次独立数据集实验。

**共同实验条件**

| 条件 | 取值 | 说明 |
|---|---|---|
| 基座模型 | `Qwen/Qwen2.5-0.5B-Instruct` | 真实 Hugging Face 模型；三组使用同一缓存快照 |
| 数据 | `tatsu-lab/alpaca`，32 条 | `prompt / response` 规范化；空回答和重复样本均为 0 |
| 数据切分 | `val_ratio=0.2` | baseline 与 LoRA 在每组内共享切分 |
| dtype | `torch.bfloat16` | RTX 5070 Ti Laptop GPU，BF16 可用 |
| micro-batch | `1` | 不是有效 batch；本实验未使用梯度累积 |
| 最大序列长度 | `256` | 影响截断、显存和吞吐 |
| 更新步数 | `20` | baseline 与 LoRA 完全一致 |
| 评测 | validation loss | 当前还没有 task-level 生成指标，因此结论仍是 `tune` |
| 环境 | Python 3.10.20，PyTorch 2.11.0+cu128，CUDA 12.8 | NVIDIA GeForce RTX 5070 Ti Laptop GPU，约 12 GB 显存 |

**三组 matched 结果**

| REAL_SEED | SPLIT_SEED | baseline val loss | LoRA val loss | baseline 峰值显存 MB | LoRA 峰值显存 MB | baseline step ms | LoRA step ms | baseline token/s | LoRA token/s | 数据审计 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 42 | 4.3103 | 2.0299 | 4774.92 | 1741.94 | 126.91 | 39.14 | 558.28 | 1810.38 | 32 条；超长 3 |
| 123 | 42 | 4.3103 | 2.0263 | 4786.98 | 1741.94 | 106.09 | 41.87 | 667.82 | 1692.22 | 32 条；超长 3 |
| 2024 | 42 | 4.3103 | 2.0307 | 4786.98 | 1741.94 | 107.62 | 38.62 | 658.31 | 1834.62 | 32 条；超长 3 |
| **均值** | **42** | **4.3103** | **2.0289** | **4782.96** | **1741.94** | **113.54** | **39.88** | **628.14** | **1779.07** | **每组超长 3** |

这三组结果支持一个**暂定资源结论**：LoRA 的峰值显存约降低 63.6%，step time 约降低 64.9%，token 吞吐约提高 2.83 倍；LoRA validation loss 约为 2.029，低于 matched baseline 的 4.310。但由于样本只有 32 条、每组有 3 条超出字符审计阈值，且还没有生成质量指标，暂不把它写成最终 `accept`。本轮数据采集先冻结；后续只在需要补生成质量、实际截断统计或压力实验时继续。

`over_length_count=3` 表示字符长度代理指标超过 `REAL_MAX_SEQ_LEN * 4`，不等于一定发生 token 截断；后续应把实际 tokenizer 截断数也记录下来。

```python
# 先画已有结果和后续实验计划；本单元只读 JSON，不启动模型、不下载数据。
import json
from pathlib import Path

import pandas as pd
from IPython.display import display

RESULT_DIR = Path('benchmarks/results/60_real_lora')
RESULT_FILES = sorted(RESULT_DIR.glob('60_real_lora_matched_seed*.json'))

rows = []
for path in RESULT_FILES:
    report = json.loads(path.read_text(encoding='utf-8'))
    cfg = report['config']
    audit = report['quality']['data_audit']
    baseline = report['baseline']
    # 兼容旧报告：早期版本把 LoRA 放在 candidates[name='lora'] 中。
    lora = report.get('lora')
    if lora is None:
        lora = next(item for item in report.get('candidates', []) if item.get('name') == 'lora')
    rows.append({
        'seed': cfg['seed'],
        'split_seed': cfg.get('split_seed', 'legacy'),
        'samples': audit['total_samples'],
        'val_ratio': cfg['val_ratio'],
        'baseline_val_loss': round(baseline['val_loss'], 4),
        'lora_val_loss': round(lora['val_loss'], 4),
        'baseline_peak_MB': round(baseline['peak_memory_mb'], 2),
        'lora_peak_MB': round(lora['peak_memory_mb'], 2),
        'baseline_step_ms': round(baseline['step_time_ms'], 2),
        'lora_step_ms': round(lora['step_time_ms'], 2),
        'baseline_tok/s': round(baseline['tokens_per_s'], 2),
        'lora_tok/s': round(lora['tokens_per_s'], 2),
        'over_length': audit['over_length_count'],
        'file': path.name,
    })

results_df = pd.DataFrame(rows).sort_values('seed') if rows else pd.DataFrame()
display(results_df)

# 后续实验矩阵：先画计划表，完成每组实验后再把 status 改为 measured。
EXPERIMENT_PLAN = pd.DataFrame([
    {'id': 'S1', 'variable': 'seed', 'value': 42, 'fixed': '真实数据 / batch=1 / seq=256 / steps=20', 'status': 'measured'},
    {'id': 'S2', 'variable': 'seed', 'value': 123, 'fixed': '真实数据 / batch=1 / seq=256 / steps=20', 'status': 'measured'},
    {'id': 'S3', 'variable': 'seed', 'value': 2024, 'fixed': '真实数据 / batch=1 / seq=256 / steps=20', 'status': 'measured'},
    {'id': 'B1', 'variable': 'matched_steps', 'value': 40, 'fixed': '固定 split seed / batch=1 / seq=256', 'status': 'planned'},
    {'id': 'B2', 'variable': 'seq_len', 'value': 512, 'fixed': '固定数据 / seed / batch=1 / steps=20', 'status': 'planned'},
    {'id': 'B3', 'variable': 'batch_size', 'value': 2, 'fixed': '固定数据 / seed / seq=256 / steps=20', 'status': 'planned'},
    {'id': 'B4', 'variable': 'task_metric', 'value': 'generation_eval', 'fixed': '固定 split / prompt / max_new_tokens', 'status': 'planned'},
])
display(EXPERIMENT_PLAN)
print('说明：一次只改变 variable；不要同时改变 seed、数据切分、seq_len、batch 或 steps。')

```


```python
if RUN_REAL_MATCHED:
    import random
    def _matched_audit(records, max_total_chars):
        pairs = [(item.get('prompt', ''), item.get('response', '')) for item in records]
        return {'total_samples': len(records), 'empty_response_count': sum(not response.strip() for _, response in pairs), 'duplicate_count': len(pairs) - len(set(pairs)), 'over_length_count': sum(len(prompt) + len(response) > max_total_chars for prompt, response in pairs), 'avg_total_chars': round(sum(len(prompt) + len(response) for prompt, response in pairs) / len(records), 2) if records else 0.0}
    def _matched_report(**sections):
        return {'schema_version': 'fine-tuning-project/v1', 'project': '60_lora_fine_tuning', 'stage': 'project_decision', **sections}
    import gc
    import json
    import os
    import sys
    import time
    from pathlib import Path

    import torch
    random.seed(REAL_SEED)
    torch.manual_seed(REAL_SEED)
    torch.cuda.manual_seed_all(REAL_SEED)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    project_root = next((path for path in [Path.cwd(), *Path.cwd().parents] if (path / 'tools').is_dir()), None)
    if project_root is None:
        raise RuntimeError('未找到项目根目录，请从仓库根目录启动 Notebook。')
    os.chdir(project_root)
    if not torch.cuda.is_available():
        raise RuntimeError('RUN_REAL_MATCHED=True 需要可用 CUDA GPU。')
    from tools.model_runtime import resolve_model

    model_path = resolve_model(REAL_MODEL_ID, REAL_MODEL_SOURCE)
    try:
        bf16_supported = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        bf16_supported = torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if REAL_DTYPE == 'auto' and bf16_supported else torch.float16
    if REAL_DTYPE != 'auto':
        dtype = getattr(torch, REAL_DTYPE)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 优先复用 Step 6 已经加载的数据；否则按同一配置自动加载。
    if 'examples' not in globals() or REAL_DATA_SOURCE != 'inline':
        if REAL_DATA_SOURCE == 'inline':
            records = [
                {'prompt': '用一句话解释 LoRA。', 'response': 'LoRA 是一种低秩参数高效微调方法。'},
                {'prompt': '用一句话解释梯度累积。', 'response': '梯度累积通过多次小批量反向传播模拟更大的 batch。'},
                {'prompt': '用一句话解释验证集。', 'response': '验证集用于检查模型对未参与训练样本的泛化表现。'},
                {'prompt': '用一句话解释 adapter。', 'response': 'adapter 是挂载在基座模型上的可训练增量参数。'},
            ]
        elif REAL_DATA_SOURCE == 'huggingface':
            from datasets import load_dataset
            records = load_dataset(REAL_DATASET_ID, split='train')
        elif REAL_DATA_SOURCE == 'modelscope':
            from modelscope.msdatasets import MsDataset
            records = MsDataset.load(REAL_DATASET_ID, split='train')
            if hasattr(records, 'to_hf_dataset'):
                records = records.to_hf_dataset()
        elif REAL_DATA_SOURCE == 'local':
            search_roots = [project_root / 'benchmarks' / 'data', project_root / 'data']
            data_candidates = [path for root in search_roots if root.exists() for path in root.glob('*') if path.suffix.lower() in {'.json', '.jsonl'}]
            data_path = Path(REAL_DATA_FILE) if REAL_DATA_FILE else (data_candidates[0] if data_candidates else None)
            if data_path is None:
                raise FileNotFoundError('未在 benchmarks/data 或 data 中找到 JSON/JSONL 数据文件')
            records = [json.loads(line) for line in data_path.read_text(encoding='utf-8').splitlines() if line.strip()] if data_path.suffix.lower() == '.jsonl' else json.loads(data_path.read_text(encoding='utf-8'))
        else:
            raise ValueError('REAL_DATA_SOURCE 必须是 inline / huggingface / modelscope / local')
        examples = []
        for record in list(records)[:REAL_MAX_SAMPLES]:
            prompt = record.get('prompt', record.get('instruction', ''))
            if 'prompt' not in record:
                prompt = str(prompt) + str(record.get('input', ''))
            response = record.get('response', record.get('output', ''))
            if prompt and response:
                examples.append({'prompt': str(prompt), 'response': str(response)})
        if not examples:
            raise ValueError('数据集中没有识别到 prompt/response 或 instruction/input/output 字段')
    examples = examples[:REAL_MAX_SAMPLES]
    # 数据划分使用独立且固定的种子；不要用 REAL_SEED，否则质量变化会混入切分变化。
    random.Random(SPLIT_SEED).shuffle(examples)
    split = max(1, int(len(examples) * (1 - MATCHED_VAL_RATIO)))
    train_examples, val_examples = examples[:split], examples[split:] or examples[-1:]

    def encode_records(records):
        texts = [item['prompt'] + '\n' + item['response'] for item in records]
        batch = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=REAL_MAX_SEQ_LEN)
        batch['labels'] = batch['input_ids'].masked_fill(batch['attention_mask'] == 0, -100)
        return batch

    train_batch = encode_records(train_examples)
    val_batch = encode_records(val_examples)
    device = torch.device('cuda')

    def batches(encoded):
        size = encoded['input_ids'].shape[0]
        for start in range(0, size, MATCHED_BATCH_SIZE):
            yield {key: value[start:start + MATCHED_BATCH_SIZE].to(device) for key, value in encoded.items()}

    def run_candidate(name):
        random.seed(REAL_SEED)
        torch.manual_seed(REAL_SEED)
        torch.cuda.manual_seed_all(REAL_SEED)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
        model.config.use_cache = False
        if name == 'lora':
            model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
        model.to(device).train()
        trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=REAL_LR)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        losses = []
        processed_tokens = 0
        started = time.perf_counter()
        # 训练数据保留在 CPU；每个 step 只把当前 micro-batch 搬到 GPU，避免把整个数据集计入显存峰值。
        train_batch_iterator = batches(train_batch)
        for step in range(MATCHED_STEPS):
            batch = next(train_batch_iterator, None)
            if batch is None:
                train_batch_iterator = batches(train_batch)
                batch = next(train_batch_iterator)
            processed_tokens += int(batch['attention_mask'].sum().item())
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().item()))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        model.eval()
        with torch.no_grad():
            val_losses = [float(model(**batch).loss.item()) for batch in batches(val_batch)]
        peak_memory = round(torch.cuda.max_memory_allocated() / 2**20, 2)
        result = {'name': name, 'status': 'ok', 'train_losses': losses, 'train_loss': losses[-1], 'val_loss': sum(val_losses) / len(val_losses), 'trainable_params': trainable_params, 'peak_memory_mb': peak_memory, 'step_time_ms': round(elapsed / MATCHED_STEPS * 1000, 2), 'tokens_per_s': round(processed_tokens / elapsed, 2)}
        if name == 'lora':
            output_dir = project_root / 'benchmarks' / 'results' / '60_real_lora'
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir / 'matched_adapter')
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()
        return result

    candidates = [run_candidate('baseline'), run_candidate('lora')]
    baseline = candidates[0]
    lora = candidates[1]
    report_builder = globals().get('build_lora_project_report', _matched_report)
    report = report_builder(
        config={'model': REAL_MODEL_ID, 'model_path': model_path, 'dtype': str(dtype), 'batch_size': MATCHED_BATCH_SIZE, 'seq_len': REAL_MAX_SEQ_LEN, 'steps': MATCHED_STEPS, 'val_ratio': MATCHED_VAL_RATIO, 'seed': REAL_SEED, 'split_seed': SPLIT_SEED},
        baseline=baseline, candidates=candidates,
        quality={'train_loss': lora['train_loss'], 'val_loss': lora['val_loss'], 'baseline_val_loss': baseline['val_loss'], 'data_audit': _matched_audit(examples, REAL_MAX_SEQ_LEN * 4)},
        resources={'trainable_params': lora['trainable_params'], 'peak_memory_mb': lora['peak_memory_mb'], 'step_time_ms': lora['step_time_ms'], 'tokens_per_s': lora['tokens_per_s']},
        artifacts={'adapter': str(project_root / 'benchmarks' / 'results' / '60_real_lora' / 'matched_adapter'), 'report': str(project_root / 'benchmarks' / 'results' / '60_real_lora' / '60_real_lora_matched.json')},
        decision={'decision': 'tune', 'reason': 'matched baseline 与 LoRA 已完成，仍需增加重复运行和任务指标后再决定是否采用。', 'next_action': 'repeat_with_fixed_validation_and_task_metric'},
        environment={'python': sys.version, 'torch': torch.__version__, 'torch_cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(0)},
    )
    report_dir = project_root / 'benchmarks' / 'results' / '60_real_lora'
    report_path = report_dir / '60_real_lora_matched.json'
    seed_report_path = report_dir / f'60_real_lora_matched_seed{REAL_SEED}.json'
    report_text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    report_path.write_text(report_text, encoding='utf-8')
    seed_report_path.write_text(report_text, encoding='utf-8')
    print(f'报告已保存：{report_path}')
    print(f'按 seed 保存：{seed_report_path}')
    print(json.dumps(report, ensure_ascii=False, indent=2))
else:
    print('跳过 matched baseline：保持 CPU-first / smoke-test 模式。')

```
