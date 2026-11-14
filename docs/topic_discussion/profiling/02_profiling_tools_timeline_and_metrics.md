# 性能剖析方法與工具入門

**難度：** 中等｜**預估時間：** 60–90 分鐘｜**範圍：** Part 0E、Part 1B

本節用同一個小型 MLP 示範 CPU 與 GPU 採樣，練習從 operator summary 和時間線找出耗時位置，再選擇更合適的深度工具。文字以 CC BY 4.0 授權，程式碼以 Apache-2.0 授權。

## 1. 先選工具，再問一個可驗證的問題

| 工具 | 主要觀察面 | 適合回答 | 不能單獨回答 |
|:---|:---|:---|:---|
| **py-spy** | Python 呼叫堆疊的取樣火焰圖 | Python 時間主要花在哪些函式？ | CUDA kernel 內的 occupancy 或 DRAM throughput |
| **torch.profiler** | PyTorch operators、CPU/CUDA events、shape、記憶體與 trace | 哪些 operator 或 kernel 花時間？CPU 是否在發出許多小工作？ | 硬體 counter 的精確 SM occupancy 與記憶體頻寬 |
| **Nsight Systems（nsys）** | 程序、CPU 執行緒、CUDA API、GPU stream、kernel 與 NVTX 時間線 | kernel 何時被排入？是否有長空檔、同步或拷貝？ | 單一 kernel 為何受到暫存器或記憶體瓶頸限制 |
| **Nsight Compute（ncu）** | 單一 CUDA kernel 的硬體計數器、occupancy 與 roofline | kernel 是算力、訪存、資源配置或延遲受限？ | 整個程式的端到端流程與其他程序的互動 |
| **Triton Proton** | Triton scope / kernel profile 與階層化結果 | Triton kernel / scope 的時間與吞吐如何分布？ | 一般 PyTorch 模型所有系統層級互動 |

**快速工作流：** 先以 py-spy 或 torch.profiler 定位，再用 nsys 看時間線；若某個 kernel 確實主導，再用 ncu 看硬體 counter。Triton 工作負載可用 Proton 追蹤 kernel 與 scope。

Timeline 的橫軸是時間；CPU row 上的 CUDA launch API 代表 host 正在送出工作，GPU row 上的 kernel 區段代表裝置執行。GPU 呼叫通常是非同步的：Python 呼叫返回的時間、kernel 執行時間與同步後端到端時間是三種不同量。先做 warm-up、只 profile 代表性步數，最後在 profiler 關閉時量測整體 latency。


```python
import copy
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function
```


## 2. 固定工作負載與環境快照

範例固定為兩層 MLP：輸入 shape 為 batch × 256，隱層寬度 1024，輸出維度 256。CPU 和 GPU 共用相同權重與輸入 shape。執行 GPU cell 需要 NVIDIA GPU、相容的 driver 與 CUDA-enabled PyTorch；CPU-only 環境仍可完整執行 CPU 採樣。PyTorch wheel 已附帶對應 CUDA runtime，這個 PyTorch 範例不需要另裝 CUDA Toolkit。


```python
print("Python platform:", platform.platform())
print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))
if torch.cuda.is_available() and shutil.which("nvidia-smi"):
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    )
    if driver.returncode == 0:
        print("NVIDIA driver:", driver.stdout.strip())

```



```python
torch.manual_seed(7)
torch.set_num_threads(1)

cpu_model = nn.Sequential(
    nn.Linear(256, 1024),
    nn.GELU(),
    nn.Linear(1024, 256),
).eval()
cpu_input = torch.randn(128, 256)
print(f"Workload ready: batch={cpu_input.shape[0]}, input=256, hidden=1024, output=256")

```


## 題目區：正確設定要採集的 activity

完成 **ProfilerConfigSolution.activities(include_cuda)**：CPU activity 永遠要採集；只有呼叫者明確要求且 PyTorch 提供 CUDA activity 時，才加入 CUDA。這個小設定函式讓同一份教學能在 CPU-only 與 CUDA 環境執行。


```python
class ProfilerConfigSolution:
    @staticmethod
    def activities(include_cuda: bool) -> list[ProfilerActivity]:
        # TODO: 永遠包含 CPU；include_cuda 為 True 時加入 CUDA activity。
        raise NotImplementedError("TODO: implement activities")
```


## 測試區

測試不依賴這台機器有沒有 GPU：它會以明確的 include_cuda 參數驗證 CPU 一定存在，以及 CUDA 只在要求時加入。真實 GPU 採樣在後面的可選 cell 執行。


```python
def test_cpu_activity_is_always_enabled():
    assert ProfilerConfigSolution.activities(False) == [ProfilerActivity.CPU]


def test_cuda_activity_is_opt_in():
    activities = ProfilerConfigSolution.activities(True)
    assert activities[0] == ProfilerActivity.CPU
    assert activities == [ProfilerActivity.CPU, ProfilerActivity.CUDA]


test_cpu_activity_is_always_enabled()
test_cuda_activity_is_opt_in()
print("CPU 必選與 CUDA 選用測試通過")
```


### 防劇透緩衝帶

先完成 TODO 並執行測試。想一想：如果在 CPU-only 環境無條件加入 CUDA activity，profile 會如何失敗？若 CUDA kernel 非同步執行，為什麼 profile 結束前要同步裝置？

## 解析 / 答案 / 參考程式碼

CPU activity 是基本採集範圍；CUDA activity 是額外選項。把兩者明確列入 profile，比只看 wall clock 更容易將 host operator 與 device 工作分開。


```python
class ProfilerConfigSolution:
    @staticmethod
    def activities(include_cuda: bool) -> list[ProfilerActivity]:
        activities = [ProfilerActivity.CPU]
        if include_cuda:
            activities.append(ProfilerActivity.CUDA)
        return activities
```


## 4. 真實採集：GPU trace 與 operator summary

本次先採 GPU，再採 CPU。Windows 11 + PyTorch 2.13.0 的這次實驗中，若先開 CPU-only Kineto session，後續 CUDA session 沒有 kernel events；因此 GPU cell 會檢查是否真的取得 device timing。以下在可用時同時採集 CPU 與 CUDA events。每次 active step 結束後呼叫 step()，在 profile 結束前同步裝置，避免非同步工作尚未完成就讀取結果。GPU cell 使用 skip-answer-test 標記，通用答案測試不會要求所有讀者都具備 CUDA GPU。


```python
gpu_capture_available = torch.cuda.is_available()
if not gpu_capture_available:
    print("跳過 GPU 採集：目前環境沒有 CUDA-enabled PyTorch 或可用 GPU。")
else:
    device = torch.device("cuda")
    gpu_model = copy.deepcopy(cpu_model).to(device).eval()
    gpu_input = cpu_input.to(device)

    with torch.inference_mode():
        for _ in range(5):
            gpu_model(gpu_input)
    torch.cuda.synchronize()

    gpu_schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
    with profile(
        activities=ProfilerConfigSolution.activities(include_cuda=True),
        schedule=gpu_schedule,
        record_shapes=True,
        profile_memory=True,
    ) as gpu_profile:
        with torch.inference_mode():
            for step in range(5):
                with record_function(f"gpu_mlp_step_{step}"):
                    gpu_model(gpu_input)
                gpu_profile.step()
        torch.cuda.synchronize()

    print("GPU operator summary over the 3 active steps:")
    try:
        print(gpu_profile.key_averages().table(sort_by="self_cuda_time_total", row_limit=12))
    except (KeyError, TypeError):
        print(gpu_profile.key_averages().table(sort_by="self_device_time_total", row_limit=12))

    def event_device_time_us(event):
        value = getattr(event, "self_device_time_total", None)
        if value is None:
            value = getattr(event, "self_cuda_time_total", 0.0)
        return float(value or 0.0)

    device_events = [
        event for event in gpu_profile.key_averages()
        if event_device_time_us(event) > 0
    ]
    assert device_events, "No CUDA device events were captured; check ProfilerActivity.CUDA and run this GPU cell before a CPU-only profiler session."
    device_events.sort(key=event_device_time_us, reverse=True)
    print("Top events with device time:")
    for event in device_events[:8]:
        print(
            f"{event.key:38s} calls={event.count:3d} "
            f"self_device={event_device_time_us(event) / 1000:.4f} ms "
            f"self_cpu={event.self_cpu_time_total / 1000:.4f} ms"
        )

    gpu_trace_dir = tempfile.TemporaryDirectory(prefix="dw-profiler-gpu-")
    gpu_trace_path = Path(gpu_trace_dir.name) / "gpu_trace.json"
    gpu_profile.export_chrome_trace(str(gpu_trace_path))
    print(f"GPU Chrome trace exported as {gpu_trace_path.name} ({gpu_trace_path.stat().st_size:,} bytes)")
    print("Open the trace with gpu_trace_path while this kernel session is active.")
```


### 本次 GPU 報告的結論

本次環境快照是 Windows 11、PyTorch 2.13.0+cu132、CUDA runtime 13.2、RTX 4060（compute capability 8.9）、driver 610.88。三個 active steps 共出現 6 次 aten::addmm（每個 MLP step 兩個 Linear）；operator row 的 Self CUDA 為 96.889 µs，約佔報表 device-time 總量 69.74%。aten::gelu 3 次共 5.664 µs（4.08%）。底層事件出現 ampere_sgemm kernel，與 Linear 的矩陣乘法路徑一致。

**結論：** 這個小型 workload 的 GPU 工作主要落在兩個 GEMM，而不是 GELU。CPU operator 自身累計 1.662 ms、CUDA device event 累計 138.934 µs；這兩種活動的計時範圍不同，只能提示這種小工作負載應檢查 host launch / dispatch 是否重要，不能把兩個總數直接當成 latency 比值。本次資料沒有 occupancy 或 DRAM counter，還不能判斷 GEMM 是 compute-bound 還是 memory-bound。不同硬體和執行環境的時間會變動。

### 如何讀 GPU timeline

用 Perfetto 或 Chrome tracing 開啟輸出的 Chrome trace；在目前 Notebook kernel 中可用 gpu_trace_path 變數找到檔案。先找 gpu_mlp_step_* 範圍，再沿時間軸對齊 CPU thread、CUDA API 與 GPU stream：若 CPU launch 間隔很長而 kernel 很短，先查 host-side dispatch、同步或 Python overhead；若 kernel 連續且單一 kernel 佔總裝置時間大，轉用 ncu 深入檢查該 kernel。torch.profiler 能關聯 operator 與 device time，但不提供完整的硬體 counter。

## 3. 真實採集：CPU operator profile

**torch.profiler.schedule** 用 wait 忽略初始步、warmup 穩定執行，再保留三個 active steps。**record_shapes** 將輸入 shape 加入 operator events，**profile_memory** 記錄 profiler 可見的 tensor 記憶體事件。Trace 會留在 kernel session 暫存目錄；cpu_trace_path 變數可用來開啟報告。重設 kernel 後暫存檔會移除，教材不保存本機路徑或大型 trace。


```python
with torch.inference_mode():
    for _ in range(3):
        cpu_model(cpu_input)

cpu_schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
with profile(
    activities=ProfilerConfigSolution.activities(include_cuda=False),
    schedule=cpu_schedule,
    record_shapes=True,
    profile_memory=True,
) as cpu_profile:
    with torch.inference_mode():
        for step in range(5):
            with record_function(f"cpu_mlp_step_{step}"):
                cpu_model(cpu_input)
            cpu_profile.step()

print("CPU operator summary over the 3 active steps:")
print(cpu_profile.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

cpu_trace_dir = tempfile.TemporaryDirectory(prefix="dw-profiler-cpu-")
cpu_trace_path = Path(cpu_trace_dir.name) / "cpu_trace.json"
cpu_profile.export_chrome_trace(str(cpu_trace_path))
print(f"CPU Chrome trace exported as {cpu_trace_path.name} ({cpu_trace_path.stat().st_size:,} bytes)")
print("Open the trace with cpu_trace_path while this kernel session is active.")
```


### 本次 CPU 報告的結論

同一個三步 MLP 的 CPU-only profile 中，aten::addmm 共 6 次；CPU total 為 10.653 ms，占 profiler CPU total 83.25%。aten::gelu 3 次、1.226 ms，占 9.58%。這份固定 shape 的 CPU 報告顯示線性層的矩陣乘法是主要耗時位置；若要優化 CPU 路徑，先看 BLAS / shape / batching，而不是先改 GELU。這些是 profiler 採集期間的 CPU operator 數字，不等於未開 profiler 的端到端 latency。

### 如何從 CPU 報告下結論

先看 **Self CPU** 與 **CPU total**，再看 **Calls** 和 input shape。aten::linear、aten::addmm 的時間集中，表示線性運算路徑值得先檢查；若小 operator 呼叫很多、單次時間很短，應檢查 Python loop、資料搬運或重複發出的 operator。這份報告記錄的是 PyTorch operator，不等同於 Python 每一行的時間；要看 Python 函式堆疊，可用 py-spy 取樣並查看火焰圖。

Profiler 本身有開銷，因此這個表用來定位，不用來報告最終 latency。要比較延遲，關閉 profiler、warm-up 後再用同一個輸入重複量測。

## 5. 關鍵指標：從數字到瓶頸假設

| 指標 | 先在哪裡看 | 如何解讀 | 下一步 |
|:---|:---|:---|:---|
| Kernel duration / self device time | torch.profiler table、nsys GPU row、ncu kernel report | 對比 kernel 時間總和、端到端 latency 與是否有 overlap；裝置時間總和不一定等於 wall time | 在 nsys 找主導區段，對重點 kernel 用 ncu |
| Operator launch overhead | nsys 的 CUDA API row、CPU launch 與 GPU kernel row | 很多短 kernel、launch 到執行的空檔大，表示 host / launch 成本可能佔比高 | 減少小 operator、融合工作或降低同步；改後重新 profile |
| SM occupancy | ncu Occupancy / Launch Statistics | 活躍 warps 相對可駐留上限的比例；低值可能由 registers、shared memory、block 配置或 grid 太小造成 | 查看 register/thread、shared memory、CTA 數與 stall；occupancy 高不代表 kernel 一定快 |
| DRAM throughput | ncu Memory Workload Analysis | 將實際 DRAM bytes / kernel time 與 GPU 可持續頻寬比較；高頻寬利用率且有 memory stalls 才支持 bandwidth-bound 假設 | 優先減少 bytes 或改善合併訪存；同時檢查 cache hit 與 arithmetic intensity |
| Python stack samples | py-spy flame graph | 樣本集中在哪些 Python 函式，顯示 CPU 上執行時間分布，不是精確逐行計時 | 檢查熱路徑、迴圈與呼叫次數；C/CUDA 算子再用 PyTorch / NVIDIA 工具交叉確認 |

判斷要組合證據。單獨的低 occupancy 不是要增加 occupancy 的證明；只有當主要 kernel 確實耗時且資源或 stall 指標能支持某個假設時，才值得改 kernel。頻寬比例高也要配合 DRAM bytes、cache、算術強度與時間占比，才能判斷是否受記憶體限制。

一個實用的推理順序是：先找最長的時間區段 → 判斷它位於 Python/CPU、CUDA API 還是 GPU kernel → 看呼叫次數與重疊 → 對疑似主導 kernel 採硬體 counter → 一次只提出一個瓶頸假設並重新量測。

## 6. 進一步工具的最小上手命令

下列命令是 profiling 起點，不是跨機器固定腳本。先確認工具支援的平台、driver 和權限；Windows 版 nsys CLI 需以系統管理員身分執行。ncu 會重新執行或序列化部分 kernel 來收集 counter，因此不要把 ncu 結果當一般 latency benchmark。

### py-spy：CPU Python 火焰圖

~~~bash
py-spy record -o profile.svg -- python workload.py
py-spy top --pid <PID>
~~~

### Nsight Systems：整體流程和 CUDA timeline

~~~bash
nsys profile -t cuda,nvtx -o run_profile python workload.py
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum run_profile.nsys-rep
~~~

### Nsight Compute：聚焦單一 kernel

~~~bash
ncu --set full -o kernel_report python workload.py
ncu --query-metrics --query-metrics-mode suffix --metrics sm__throughput
~~~

不同 GPU、driver 和 ncu 版本可用的 metric 名稱會變動。先用 ncu --query-metrics 查目前裝置支援的項目，再選 Occupancy、Memory Workload Analysis 或 throughput section；避免一開始無篩選地 profile 長時間程式。

### Triton Proton：Triton scope / kernel

~~~python
import triton.profiler as proton

proton.start("profile_name", hook="triton")
# 在這裡執行 Triton kernel 或代表性 workload
proton.finalize()
~~~

~~~bash
proton-viewer -m time/s profile_name.hatchet
~~~

Triton Proton 的結果也可輸出 Chrome trace；使用方式與支援的 profiler backend 依 Triton 版本和 GPU 平台而異。

## 7. 本次結果的邊界

本節的 CPU / GPU trace 是真實採集，但 workload 只是一個固定 shape 的小 MLP。它能示範如何把 PyTorch operator 與 CUDA activity 對上，不能代表所有模型、batch、driver 或 GPU 的延遲。SM occupancy 與 DRAM throughput 必須用 ncu 對目標 kernel 讀 counter；torch.profiler 的 trace 本身沒有提供這兩項硬體值。本次沒有安裝 nsys / ncu，也沒有執行 Triton Proton；它們的命令與閱讀重點作為後續深入工具鏈。

## 8. 延伸練習

1. 將 batch 改為 1、16、128，分別比較 CPU operator 呼叫數、GPU kernel 數與未開 profiler 的 median latency。
2. 把 MLP 換成逐樣本 Python loop；觀察 py-spy、torch.profiler CPU row 與 GPU timeline 的 launch pattern 是否一致。
3. 針對一個明確耗時的 CUDA kernel 執行 ncu，記錄 kernel duration、occupancy、DRAM throughput 與 register / shared-memory 用量，再提出一個可驗證的改動。
4. 比較 eager 與 compile 後的 trace：工作是否融合、kernel 數是否下降、端到端 latency 是否也下降？只在相同輸入和 warm-up 條件下比較。

## 延伸閱讀

- [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html)：CPU/CUDA activities、schedule、key averages 與 trace 匯出。
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)：系統時間線、CLI 收集與報表統計。
- [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) 與 [Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)：counter、metrics、occupancy 與 memory analysis。
- [Triton Proton profiler 範例](https://github.com/triton-lang/triton/tree/main/third_party/proton/tutorials)：Triton profiler 與 trace 使用方式。
- [py-spy](https://github.com/benfred/py-spy)：Python sampling profiler 與 flame graph。
- [Part 0E：除錯與效能](../../00_Prerequisites/0E.md) · [Part 1B：記憶體最佳化](../../01_Hardware_Math_and_Systems/1B.md)。