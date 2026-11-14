# Profiling 線性投影：從逐樣本呼叫到批次執行

**難度：** 中高｜**環境：** Python 3.10+、PyTorch 2.x、CPU 即可｜**預估時間：** 45–60 分鐘

本實戰從一個小型模型的 nn.Linear 前向開始，使用 PyTorch Profiler 找出逐樣本執行造成的重複 operator 呼叫，再把相同計算改成批次線性投影。重點不是追逐一個孤立的耗時占比，而是讓「瓶頸判斷 → 優化候選 → profiler 證據 → 未加 profiler 的計時」互相對照。

文字依 CC BY 4.0 授權；可執行程式碼依 Apache-2.0 授權。

## 1. 本次要回答的問題

同一個線性層收到 64 個向量時，逐個呼叫 Linear 與一次處理整個 batch，數學結果相同，但執行路徑不同。本節會判斷小型 workload 的時間花在哪裡，並檢查批次化是否減少重複的矩陣乘法 operator 呼叫。

| 證據層 | 本節觀察 |
|:--|:--|
| workload | batch、輸入 / 輸出特徵數、dtype、CPU 執行緒數 |
| profiler | aten::linear、aten::addmm、aten::stack、aten::cat 的呼叫次數與 CPU 時間 |
| 正確性 | 逐樣本與批次輸出的最大誤差 |
| benchmark | 固定 warm-up 與重複次數後的 latency median |

CPU profiler 收集的是 CPU operator 活動；它不能推導 GPU kernel launch、GPU 帶寬或多卡通信結論。

## 2. 固定 workload 與模型

SmallProjection 是一個真實的 PyTorch 線性層。輸入是一批獨立的向量；逐樣本路徑刻意用 Python 迴圈呼叫相同 nn.Linear，批次路徑則讓線性層直接接受二維 tensor。兩邊沿用同一組權重、bias 與輸入。


~~~python
import platform
import statistics
import time

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile

torch.manual_seed(7)
torch.set_num_threads(1)
print(f"Platform: {platform.platform()}")
print(f"CPU: {platform.processor() or platform.machine()} | capability: {torch.backends.cpu.get_cpu_capability()}")
print(f"PyTorch: {torch.__version__} | device: CPU | threads: {torch.get_num_threads()}")
print("Profiler scope: CPU operators only")
~~~


## 題目區：完成批次線性投影

請實作 ProfilingSolution.batched_projection，讓 model 一次接收整個二維 batch，回傳 shape 為 [batch, out_features] 的 tensor。不要改變模型權重、bias 或輸入資料。


~~~python
class ProfilingSolution:
    @staticmethod
    def batched_projection(model, x):
        # TODO: 使用單次 nn.Linear 呼叫處理整個 batch。
        raise NotImplementedError("TODO: implement batched_projection")
~~~


## 測試區

第一個測試檢查批次化前後輸出一致；第二個測試以 CPU profiler 比較每個 forward 的 aten::addmm 次數。Profiler 只用於查證執行路徑，稍後的效能數字會在關閉 profiler 時量測。


~~~python
def per_sample_reference(model, x):
    return torch.stack([model(row) for row in x], dim=0)


def count_addmm_calls(fn, model, x, profiled_steps=1):
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(profiled_steps):
            fn(model, x)
    events = {event.key: event for event in prof.key_averages()}
    return events.get("aten::addmm").count if "aten::addmm" in events else 0


class TestProjection:
    @staticmethod
    def test_batched_matches_per_sample():
        model = nn.Linear(32, 48, bias=True).eval()
        x = torch.randn(8, 32)
        with torch.inference_mode():
            expected = per_sample_reference(model, x)
            actual = ProfilingSolution.batched_projection(model, x)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    @staticmethod
    def test_batch_reduces_addmm_calls():
        model = nn.Linear(32, 48, bias=True).eval()
        x = torch.randn(8, 32)
        loop_calls = count_addmm_calls(
            lambda m, t: per_sample_reference(m, t), model, x
        )
        batch_calls = count_addmm_calls(
            ProfilingSolution.batched_projection, model, x
        )
        assert batch_calls < loop_calls, (
            f"批次路徑應減少 addmm 呼叫：loop={loop_calls}, batch={batch_calls}"
        )


TestProjection.test_batched_matches_per_sample()
TestProjection.test_batch_reduces_addmm_calls()
print("輸出一致性與 profiler 呼叫次數測試通過")
~~~


### 防劇透緩衝帶

---

先完成 TODO 並跑測試。觀察 profiler 的 operator 次數時，請區分「每次 forward 呼叫數」與「整段 profile 的總次數」；測試會先用小 batch 驗證路徑，正式量測則固定另一組 batch 大小。

## 解析 / 答案 / 參考程式碼

將二維 tensor 直接交給線性層即可。nn.Linear 對最後一維做相同的仿射變換，因此一次處理多列不會改變數學運算，只改變 operator 的發起方式。


~~~python
class ProfilingSolution:
    @staticmethod
    def batched_projection(model, x):
        return model(x)
~~~


## 3. 用 profiler 檢視前向執行路徑

record_shapes=True 讓 operator 帶上輸入形狀；profile_memory=True 記錄 profiler 可觀察到的 tensor 記憶體事件。這兩個選項會增加 profiling 開銷，所以這裡只用呼叫次數與熱點做歸因，不把 profiler 內的時間當成正式 benchmark。


~~~python
batch_size, in_features, out_features = 64, 256, 256
model = nn.Linear(in_features, out_features, bias=True).eval()
x = torch.randn(batch_size, in_features, dtype=torch.float32)

def loop_forward(model, tensor):
    return per_sample_reference(model, tensor)

def batch_forward(model, tensor):
    return ProfilingSolution.batched_projection(model, tensor)

with torch.inference_mode():
    loop_output = loop_forward(model, x)
    batch_output = batch_forward(model, x)
torch.testing.assert_close(batch_output, loop_output, rtol=1e-5, atol=1e-5)
max_abs_error = (batch_output - loop_output).abs().max().item()
print(f"Input: [{batch_size}, {in_features}] | weight: [{out_features}, {in_features}]")
print(f"dtype: {x.dtype} | device: {x.device} | max abs error: {max_abs_error:.3g}")
~~~



~~~python
def profile_forward(fn, model, tensor, steps=3):
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with torch.inference_mode():
            for _ in range(steps):
                fn(model, tensor)
    return prof

loop_profile = profile_forward(loop_forward, model, x)
batch_profile = profile_forward(batch_forward, model, x)

def event_map(prof):
    return {event.key: event for event in prof.key_averages()}

loop_events = event_map(loop_profile)
batch_events = event_map(batch_profile)
print("Profiler steps per path: 3")
print("Operator counts per forward:")
for name in ("aten::linear", "aten::addmm", "aten::stack", "aten::cat"):
    loop_count = loop_events[name].count / 3 if name in loop_events else 0
    batch_count = batch_events[name].count / 3 if name in batch_events else 0
    print(f"  {name:14s} loop={loop_count:5.1f} | batch={batch_count:5.1f}")
print("\nLoop path — top CPU events:")
print(loop_profile.key_averages().table(sort_by="self_cpu_time_total", row_limit=8))
print("Batched path — top CPU events:")
print(batch_profile.key_averages().table(sort_by="self_cpu_time_total", row_limit=8))
~~~


## 4. 做一輪不含 profiler 的 benchmark

兩條路徑使用相同 tensor、權重、bias、單執行緒設定與輸出檢查。先 warm-up，再量 31 次並取 median；CPU 呼叫同步完成，因此直接使用 perf_counter。若把本例改成 GPU，必須另外處理 asynchronous execution 與 device synchronization。


~~~python
def median_latency_ms(fn, model, tensor, warmup=5, repeats=31):
    with torch.inference_mode():
        for _ in range(warmup):
            fn(model, tensor)
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn(model, tensor)
            samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)

loop_ms = median_latency_ms(loop_forward, model, x)
batch_ms = median_latency_ms(batch_forward, model, x)
print(f"Workload: batch={batch_size}, in={in_features}, out={out_features}, FP32")
print("Warm-up: 5 | measured repeats: 31 | statistic: median")
print(f"Per-sample loop: {loop_ms:.3f} ms")
print(f"Single batched Linear: {batch_ms:.3f} ms")
print(f"Speedup: {loop_ms / batch_ms:.2f}x")
~~~


## 5. 依證據歸因，不把結論擴大

本次 workload 的逐樣本路徑每個 forward 會多次呼叫 Linear / addmm，還需要組合輸出；批次路徑把同一個線性層表達成一次較大的矩陣運算。若 profiler 顯示 aten::addmm 從每 batch 一次變成 batch size 次，並且未加 profiler 的延遲同步下降，證據支持「小 GEMM 的重複 operator dispatch 與 Python 迴圈成本限制這個 workload」。

這不代表大型矩陣乘法都受 dispatch 限制，也不代表 CPU profiler 量到了硬體頻寬。batch、矩陣尺寸、執行緒數或 backend 改變時，計算本身可能成為主成本；下一步應固定 workload，再做一項可推翻本結論的對照。

## 6. 練習延伸

1. 固定特徵數，將 batch 改成 1、8、64、256，記錄每個 shape 的 profiler 呼叫數與未加 profiler 的 median。
2. 提高 in_features / out_features，觀察批次化收益何時開始受矩陣計算成本主導。
3. 增加第二個 Linear + ReLU 區塊，分別用 profiler 與 benchmark 檢查算子數、輸出正確性與 latency。
4. 若在 GPU 上重做，使用 CUDA activity、正確同步並記錄硬體 / CUDA / PyTorch 版本；不要用 CPU trace 推論 kernel launch 時間。

## 延伸閱讀

- [PyTorch Profiler API](https://docs.pytorch.org/docs/stable/profiler.html) — CPU / CUDA activity、輸入形狀、記憶體追蹤與 operator 聚合。
- [瓶頸歸因：計算、帶寬、容量、通信、I/O 與排隊](../performance_analysis/05_bottleneck_attribution.md) — 從 profiler 現象提出可證偽的判斷。
- [可重複 Baseline 與測量契約](../performance_analysis/02_reproducible_baseline.md) — 固定 workload、環境、warm-up 與統計口徑。
- [算術強度與資源上限](../../02_PyTorch_Algorithms/PERF-ROOFLINE_Arithmetic_Intensity_and_Resource_Ceiling.md) — 用算術強度輔助判斷計算或頻寬限制。

本文文字採 CC BY 4.0；程式碼採 Apache-2.0。