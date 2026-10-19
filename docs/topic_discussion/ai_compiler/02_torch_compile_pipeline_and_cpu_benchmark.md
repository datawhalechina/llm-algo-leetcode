# TorchInductor Compile Path | torch.compile 編譯鏈路與 CPU 實測

**難度：** 中高｜**環境：** PyTorch 2.5+、CPU 即可；TorchInductor 需要可用的 C++ 編譯器｜**預估時間：** 60–90 分鐘

本節從 eager PyTorch 函式出發，沿著 **Python → FX graph → Inductor IR → fusion 與排程 → C++ codegen → CPU 執行** 逐段找證據。最後以固定的逐元素 workload 比較 eager 與編譯後的穩態延遲，並把首次編譯時間另外列出。

文字與圖示採 **CC BY 4.0**；程式碼採 **Apache-2.0**。

## 1. 本節要回答的問題

torch.compile(fn) 看起來只多包一層函式，實際上會嘗試把 Python 中可捕捉的 PyTorch 運算變成圖，再交給 backend 做 lowering、fusion、排程與 codegen。只看最後耗時，無法知道編譯器是否真的改寫了執行路徑；本節會同時看 FX graph、生成 kernel 數量與計時結果。

完成後你可以：

1. 說明常見編譯路徑中 TorchDynamo、AOTAutograd、TorchInductor 各自處理什麼。
2. 從輸出確認 TorchDynamo 捕捉到的 FX graph。
3. 在 CPU 上執行 TorchInductor，確認逐元素運算是否融合成較少的生成 kernel。
4. 用相同輸入、固定執行緒、warm-up 與多次計時比較 eager 和 compiled 的穩態延遲。

本實驗是 CPU 上的合成逐元素 workload，用來展示 fusion 如何減少 Python dispatch 與中間張量往返。結果只描述這個形狀、這台 CPU 與這個 PyTorch 版本，不代表所有模型都會加速，也不外推到 GPU。

## 2. 編譯鏈路：每一站留下什麼

| 階段 | 主要工作 | 本節可觀察的證據 |
| --- | --- | --- |
| Python / eager | Python 逐行呼叫 PyTorch operators；每個 eager tensor op 通常各自執行並建立結果張量 | 基準函式與其五個逐元素運算 |
| TorchDynamo | 執行時解讀 Python bytecode，捕捉可編譯區域並產生 FX graph；guards 描述快取可重用條件 | 自訂 backend 印出的 FX graph；fullgraph=True 讓 graph break 直接報錯 |
| AOTAutograd（訓練路徑） | 為需要梯度的圖捕捉、分解 forward/backward，讓後端能編譯相關計算 | 本節是 inference workload，因此不把 AOTAutograd 的訓練收益算進 benchmark |
| TorchInductor | 將圖 lowering 到自己的 IR，做 fusion 與 scheduling，選擇符合目標裝置的實作 | metrics.generated_kernel_count；可選的 TORCH_COMPILE_DEBUG=1 dump |
| CPU codegen / runtime | CPU 端產生 C++ 並由本機 C++ compiler 建置；執行編譯結果並依 guards 使用快取或重新編譯 | 首次呼叫耗時、穩態耗時、生成 kernel 數 |

這是一條概念路徑，不是每次呼叫都經過完全相同的階段。Dynamo 遇到不支援的 Python 行為時可能切出多個 graph；訓練、inference、動態 shape 與 backend 選項也會改變後續工作。torch.compile 預設 backend 是 inductor。

## 3. 為何這個 workload 適合看 fusion

基準函式依序執行 multiply → add → ReLU → multiply → add。Eager 執行時，這些 operator 會各自讀寫 tensor；中間值會被 materialize。圖沒有分支、副作用或 Python 資料依賴，因此有機會把相鄰逐元素運算合併到一個 CPU loop / kernel。

融合能減少中間結果與呼叫成本，但不保證總是更快。小 tensor 可能主要受啟動與 Python overhead 影響；大 tensor 可能受記憶體頻寬或計算吞吐限制；圖含有不支援的操作時，也可能出現 graph break 或 fallback。實測會把這些結論限制在固定 workload 內。

## 4. 環境與執行方式

- 建議 PyTorch 2.5 或更新版本。CPU-only wheel 即可，不需要 CUDA。
- TorchInductor CPU 需要 C++ compiler。Windows 可使用 Visual Studio 2022 的 Desktop development with C++ 工具；啟動 Jupyter 的命令列也要先載入 vcvars64.bat 環境。Linux 需安裝可用的 C++ toolchain。
- 若在中文 Windows 遇到 PyTorch 套件內部的 UnicodeDecodeError，從支援 UTF-8 的 Python session 啟動 Notebook，例如設定 PYTHONUTF8=1 後再啟動 Jupyter，或以 python -X utf8 執行驗證腳本。
- 想查看 lowering 與生成程式碼，可在啟動 Python 之前設定 TORCH_COMPILE_DEBUG=1。PyTorch 會在日誌中列出 debug trace 目錄，其中可找到 fx_graph_transformed.py、ir_pre_fusion.txt、ir_post_fusion.txt 與 output_code.py。請以實際版本輸出的內容為準。

Linux shell 範例：

    python -m pip install "torch>=2.5"
    TORCH_COMPILE_DEBUG=1 python your_notebook_runner.py

Windows 請在已載入 C++ compiler 環境的命令列啟動 Jupyter；TORCH_COMPILE_DEBUG 與 PYTHONUTF8 都必須在 kernel 啟動前設定，不能只在已載入 PyTorch 的 cell 內補設。


```python
import os
import platform
import statistics
import time

# TorchInductor 在啟動時讀取這個設定，用於輸出 fusion 與 codegen 的診斷資料。
os.environ.setdefault("TORCH_COMPILE_DEBUG", "1")

import torch

print(f"Python platform: {platform.platform()}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print('本實驗固定使用 CPU。')
```


## 題目區：實作可編譯的逐元素函式

請補完 CompilerChainSolution：

- pointwise(x) 依序做乘法、加法、ReLU、乘法、加法，並回傳 tensor。
- compile(fn, backend=...) 使用 torch.compile，啟用 fullgraph=True，並允許測試傳入自訂 backend。

先執行下面測試；題目區應因 NotImplementedError 失敗。完成後再向下閱讀答案區。


```python
class CompilerChainSolution:
    @staticmethod
    def pointwise(x):
        # TODO: 依序實作 multiply → add → ReLU → multiply → add。
        raise NotImplementedError("TODO: implement pointwise")

    @staticmethod
    def compile(fn, backend="inductor"):
        # TODO: 回傳 torch.compile 包裝後的函式，要求 fullgraph。
        raise NotImplementedError("TODO: implement compile")
```


## 測試區

測試使用小型 CPU tensor，檢查公式與 Dynamo graph capture。自訂測試 backend 只回傳 FX graph 的 forward，用來驗證前端捕捉，不會取代後面真正的 TorchInductor codegen。


```python
def reference_pointwise(x):
    return torch.relu(x * 1.125 + 0.25) * 0.75 + 0.125


def test_pointwise_matches_reference():
    x = torch.linspace(-2.0, 2.0, steps=17)
    actual = CompilerChainSolution.pointwise(x)
    torch.testing.assert_close(actual, reference_pointwise(x))


def test_compile_wrapper_captures_fx_graph():
    captured = []

    def capture_backend(graph_module, example_inputs):
        captured.append(graph_module)
        return graph_module.forward

    x = torch.linspace(-2.0, 2.0, steps=17)
    compiled = CompilerChainSolution.compile(
        CompilerChainSolution.pointwise,
        backend=capture_backend,
    )
    actual = compiled(x)
    torch.testing.assert_close(actual, reference_pointwise(x))
    assert len(captured) == 1, f"預期一個 graph，實際 {len(captured)} 個"
    graph_text = str(captured[0].graph)
    for op_name in ("mul", "add", "relu"):
        assert op_name in graph_text, f"FX graph 缺少 {op_name}:\n{graph_text}"


test_pointwise_matches_reference()
test_compile_wrapper_captures_fx_graph()
print("pointwise semantics 與 FX graph capture 通過")
```


### 防劇透緩衝帶

---

先自己完成兩個 TODO，並檢查測試所用的 capture_backend 為何只能證明 Dynamo 前端捕捉成功，不能證明 Inductor 已產生 CPU code。完成後再往下看解析。

## 解析 / 答案 / 參考程式碼

先把 eager 函式寫成純 tensor 運算，再用 torch.compile 包裝。將 backend 做成參數，讓相同函式可用自訂 backend 檢視 FX graph，也可用 inductor 走到底層 CPU codegen。fullgraph=True 會要求整個函式留在單一 graph；不支援的程式若造成 graph break，Dynamo 會明確報錯，方便縮小問題範圍。


```python
class CompilerChainSolution:
    @staticmethod
    def pointwise(x):
        y = x * 1.125
        y = y + 0.25
        y = torch.relu(y)
        y = y * 0.75
        return y + 0.125

    @staticmethod
    def compile(fn, backend="inductor"):
        return torch.compile(fn, backend=backend, fullgraph=True)
```


## 5. 先看 TorchDynamo 捕捉的 FX graph

這個自訂 backend 將圖模組記錄下來後回傳 graph_module.forward。它能證明 Python 運算被抓成 FX graph，並顯示 Dynamo 傳給 backend 的 operator；它沒有做 Inductor lowering，也不會產生最佳化後的 C++ kernel。


```python
captured = []

def capture_backend(graph_module, example_inputs):
    captured.append(graph_module)
    return graph_module.forward

capture_fn = CompilerChainSolution.compile(
    CompilerChainSolution.pointwise,
    backend=capture_backend,
)
probe = torch.linspace(-2.0, 2.0, steps=9)
_ = capture_fn(probe)
print(captured[0].code)
```


觀察 graph 中的五個運算節點。這是前端交給 backend 的表示；此時尚不能從節點數推斷最後會生成幾個 CPU kernel。編譯器可以在 lowering 和 scheduling 階段融合多個節點，也可能因 layout、資料依賴或 backend 限制保留多個 kernel。

## 6. 用 TorchInductor 在 CPU 上編譯並量測

固定 workload 為一個 8192 × 256 的 float32 tensor，使用單執行緒。首次呼叫會包含 trace、lowering、codegen 與 C++ compiler 建置時間，另外列出；穩態延遲先 warm-up，再各量 31 次並取 median。比較時兩邊使用相同輸入與 inference mode。

generated_kernel_count 是 PyTorch Inductor 的診斷計數器，位於內部 metrics 模組，非穩定 public API。若未來版本移除此欄位，應以 TORCH_COMPILE_DEBUG=1 產生的 output_code.py 和 pre/post-fusion IR 檔案檢查 codegen；計時與輸出正確性仍可獨立執行。


```python
from torch._inductor import metrics

# 固定執行緒與輸入，減少本次小型實驗中的環境變因。
torch.manual_seed(7)
torch.set_num_threads(1)
x = torch.randn((8192, 256), dtype=torch.float32, device="cpu")

# 首次呼叫包含編譯時間，另外列出，不混進穩態 median。
metrics.reset()
compiled_pointwise = CompilerChainSolution.compile(CompilerChainSolution.pointwise)
compile_start = time.perf_counter()
compiled_result = compiled_pointwise(x)
first_call_s = time.perf_counter() - compile_start
generated_kernels = getattr(metrics, "generated_kernel_count", None)

eager_result = CompilerChainSolution.pointwise(x)
torch.testing.assert_close(compiled_result, eager_result, rtol=1e-4, atol=1e-5)
max_abs_error = (compiled_result - eager_result).abs().max().item()


def median_latency_ms(fn, tensor, warmup=5, repeats=31):
    for _ in range(warmup):
        fn(tensor)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(tensor)
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


with torch.inference_mode():
    eager_ms = median_latency_ms(CompilerChainSolution.pointwise, x)
    compiled_ms = median_latency_ms(compiled_pointwise, x)

print(f"PyTorch: {torch.__version__}")
print(f"Device: CPU | threads: {torch.get_num_threads()} | dtype: {x.dtype} | shape: {tuple(x.shape)}")
print(f"Eager median: {eager_ms:.3f} ms")
print(f"Inductor median: {compiled_ms:.3f} ms")
print(f"Steady-state ratio (eager / compiled): {eager_ms / compiled_ms:.2f}x")
print(f"First compiled call, including compilation: {first_call_s:.3f} s")
print(f"Max absolute output error: {max_abs_error:.3g}")
if generated_kernels is None:
    print("Generated kernel count: unavailable in this PyTorch version")
else:
    print(f"Generated kernel count for this graph: {generated_kernels}")
```


## 7. 讀懂結果與編譯器實際做了什麼

先確認輸出誤差在浮點容差內，再比較兩個 median。若本 workload 得到一個生成 kernel，代表這五個逐元素 operator 在當前 PyTorch / CPU backend 上被排進同一份生成程式；配合 TORCH_COMPILE_DEBUG=1 的 ir_post_fusion.txt 與 output_code.py，可以進一步確認 fusion 與 C++ codegen 的具體形狀。

若本機生成多個 kernel，先看 debug trace：graph break 會將函式拆成多張圖；Inductor 也可能基於 layout、運算支援或成本模型分開排程。不要把「五個 FX 節點必定變成一個 kernel」當成規則。

速度比是穩態比較，沒有把首次編譯時間攤入重複推論。如果服務只呼叫一次，編譯成本可能抵銷穩態收益；若輸入 shape / dtype 改變，guards 可能失效並觸發重編譯。部署前應用目標 batch、shape 與併發方式做完整的端到端 profiling。

## 8. 練習延伸

1. 把第二次乘法換成 sigmoid，檢查生成 kernel 數與穩態延遲是否變化。
2. 在函式中加一段依賴 tensor 值的 Python 分支，再以 fullgraph=True 觀察 graph capture 的限制；不要把 graph break 誤認為 optimizer pass。
3. 將 shape 縮到 8 × 8，比較 compile 冷啟動與穩態推論，說明為何小型 workload 不一定值得編譯。
4. 若使用 training loop，分開測試 forward 與 backward，觀察 AOTAutograd 對訓練圖的作用；不要把 inference benchmark 當成 training benchmark。

## 延伸閱讀

- [Part 1E：編譯器與後端](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/1E.md) — 認識編譯器、backend 與硬體映射。
- [Part 4：CUDA Graph 與 JIT 編譯](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/04_CUDA_and_System_Optimization/18_CUDA_Graph_and_JIT_Compile.md) — 從系統層理解啟動成本與圖執行。
- [算子優化：編譯與圖優化問題鏈](https://github.com/datawhalechina/llm-algo-leetcode/tree/main/topic_discussion/operator_optimization/graph_compiler) — 補足 graph、fusion、backend 約束與成本判斷。
- [PyTorch torch.compile API](https://docs.pytorch.org/docs/stable/generated/torch.compile.html) — 查看目前 API 與 backend 參數。
- [PyTorch TorchInductor CPU 除錯與 profiling](https://docs.pytorch.org/tutorials/intermediate/inductor_debug_cpu.html) — 查看 TORCH_COMPILE_DEBUG、融合 IR 與生成程式碼。
- [Windows CPU/XPU 使用 torch.compile](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html) — 查閱 Windows CPU compiler 的安裝前置條件。