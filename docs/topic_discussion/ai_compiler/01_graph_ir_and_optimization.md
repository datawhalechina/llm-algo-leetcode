# AI Compiler: Graph IR and Optimization | 計算圖 IR 與圖優化

**難度：** 中高｜**執行環境：** Python 3，CPU 即可｜**預估時間：** 45–60 分鐘

這一節把模型運算表示成計算圖，並用純 Python 的小型 IR 實作常量折疊、公共子表達式消除、死碼消除與算子融合。例子會比較改寫前後的圖節點數和中間向量數；這些是結構指標，不是硬體延遲或實際記憶體流量。

文字與圖示採 CC BY 4.0；程式碼採 Apache-2.0。

## 學習目標與 Part 1E 的連結

讀完並完成練習後，你可以：

1. 解釋 IR 如何把運算、資料依賴和輸出表示成有向無環圖（DAG）。
2. 說明常量折疊、公共子表達式消除、死碼消除、融合，以及 layout 與記憶體規劃各自能解決的問題。
3. 用測試確認圖改寫保留輸出語義，再用結構代理指標描述改寫效果。

本節是橫向專題的動手補充，接續 [Part 1E：編譯最佳化與硬體生態](../../01_Hardware_Math_and_Systems/1E.md)。Part 1E 建立編譯器和硬體的概念；這裡把其中一條圖改寫路徑縮小成可執行例子。

典型路徑可以先想成：

~~~text
模型程式 / 前端擷取
    → 計算圖 IR
    → 圖與算子改寫
    → layout、排程與記憶體規劃
    → backend lowering
    → 目標裝置上的執行
~~~

例如，PyTorch FX 會以 Graph、Node 表示擷取到的程式，讓工具能遍歷、改寫並產生程式碼；torch.compile 則涵蓋圖擷取到後端程式碼產生的執行路徑。這個 Notebook 的 tuple IR 只模擬其中的圖與改寫層。

## 計算圖 IR 是什麼？

一個計算圖把「做什麼」和「值如何相依」表示出來。節點描述輸入、常量或運算；邊描述某個運算要讀取哪些值；輸出節點定義圖的可觀察結果。當圖只包含純函數運算時，無環資料依賴讓編譯器可以安全地排序與重寫節點。

真實 IR 還可能攜帶 dtype、shape、stride、裝置、別名、控制依賴與副作用等資訊。若這些資訊被省略，某些看起來等價的改寫就未必合法。例如，兩個隨機數節點不能只因為運算名稱相同就合併；寫入狀態的運算也不能像純算術節點一樣被刪除。

| IR 的資訊 | 編譯器用它回答的問題 |
|:--|:--|
| 運算和輸入值 | 哪些節點有相同的計算？能否重排？ |
| shape、dtype、stride | 哪個 kernel 或指令合法？是否要轉換 layout？ |
| 使用者與輸出 | 節點是否可刪？結果能否安全共用？ |
| 副作用與控制依賴 | 節點能否搬移、合併或省略？ |
| backend 能力 | 改寫後的運算能否在目標裝置執行？ |

## 常見圖最佳化

| 改寫 | 原理與適用場景 | 主要限制或成本 |
|:--|:--|:--|
| 常量折疊 | 對已知常量先計算，例如將 2 + 2 換成 4，減少執行時工作。 | dtype、溢位、捨入與例外行為必須符合原語義。 |
| 公共子表達式消除（CSE） | 將輸入相同的純運算共用一次結果，適合重複的 shape 計算或算術子圖。 | 只可合併語義和副作用都相同的節點；共用結果可能延長生命週期、增加暫存空間。 |
| 死碼消除（DCE） | 從圖輸出及副作用根節點反向追蹤，只保留可觀察結果依賴的節點。 | 真實編譯器要保留 I/O、狀態寫入、同步等副作用。 |
| 算子融合 | 將相鄰運算放進同一執行單元，通常可減少啟動次數與中間張量的寫回/讀取。 | 暫存器壓力、occupancy、layout、不規則 shape 或 backend 限制可能使融合退化或不合法。 |
| layout 變換與記憶體規劃 | 依消費者所需的 stride/排列配置資料；按 live range 重用暫存區，降低搬移或峰值配置。 | 轉置/複製可能增加成本；共享值的生命週期會限制重用；多個消費者可能偏好不同 layout。 |

常見的合法性檢查包括：改寫前後輸出相等、dtype 與 shape 相容、副作用順序不變、以及目標 backend 支援候選運算。融合的收益假說可以是「少一次 kernel 啟動和兩個中間張量」；是否真的變快，仍須在固定 workload 與目標硬體上量測。


```python
from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from typing import Any, Mapping


@dataclass(frozen=True)
class Node:
    name: str
    op: str
    inputs: tuple[str, ...] = ()
    value: Any = None


@dataclass(frozen=True)
class Graph:
    nodes: tuple[Node, ...]
    outputs: tuple[str, ...]

    def __post_init__(self):
        names = [node.name for node in self.nodes]
        if len(names) != len(set(names)):
            raise ValueError("節點名稱必須唯一")
        known = set()
        for node in self.nodes:
            missing = set(node.inputs) - known
            if missing:
                raise ValueError(f"{node.name} 引用了尚未定義的輸入：{sorted(missing)}")
            known.add(node.name)
        if set(self.outputs) - known:
            raise ValueError("圖輸出必須指向已定義的節點")


def _is_vector(value):
    return isinstance(value, (tuple, list))


def _map_unary(fn, value):
    if _is_vector(value):
        return tuple(fn(item) for item in value)
    return fn(value)


def _map_binary(fn, left, right):
    if _is_vector(left) and _is_vector(right):
        if len(left) != len(right):
            raise ValueError("向量長度必須一致")
        return tuple(fn(a, b) for a, b in zip(left, right))
    if _is_vector(left):
        return tuple(fn(a, right) for a in left)
    if _is_vector(right):
        return tuple(fn(left, b) for b in right)
    return fn(left, right)


def _map_ternary(fn, first, second, third):
    vectors = [value for value in (first, second, third) if _is_vector(value)]
    if vectors and any(len(value) != len(vectors[0]) for value in vectors):
        raise ValueError("向量長度必須一致")
    if not vectors:
        return fn(first, second, third)
    size = len(vectors[0])
    values = (first, second, third)
    return tuple(
        fn(*(value[index] if _is_vector(value) else value for value in values))
        for index in range(size)
    )


def run_graph(graph: Graph, inputs: Mapping[str, Any]) -> dict[str, Any]:
    values = {}
    for node in graph.nodes:
        if node.op == "input":
            if node.name not in inputs:
                raise KeyError(f"缺少圖輸入：{node.name}")
            values[node.name] = inputs[node.name]
        elif node.op == "const":
            values[node.name] = node.value
        elif node.op == "add":
            values[node.name] = _map_binary(lambda a, b: a + b, *(values[name] for name in node.inputs))
        elif node.op == "mul":
            values[node.name] = _map_binary(lambda a, b: a * b, *(values[name] for name in node.inputs))
        elif node.op == "relu":
            values[node.name] = _map_unary(lambda value: max(0, value), values[node.inputs[0]])
        elif node.op == "fused_relu_affine":
            x, scale, bias = (values[name] for name in node.inputs)
            values[node.name] = _map_ternary(
                lambda value, factor, offset: max(0, value * factor + offset),
                x, scale, bias,
            )
        else:
            raise ValueError(f"不支援的運算：{node.op}")
    return {name: values[name] for name in graph.outputs}


def graph_text(graph: Graph) -> str:
    lines = []
    for node in graph.nodes:
        if node.op == "input":
            expression = "input"
        elif node.op == "const":
            expression = f"const({node.value})"
        else:
            expression = f"{node.op}({', '.join(node.inputs)})"
        lines.append(f"{node.name} = {expression}")
    lines.append(f"outputs = {graph.outputs}")
    return "\n".join(lines)


def graph_cost_report(graph: Graph, inputs: Mapping[str, Any]) -> dict[str, int]:
    values = run_graph(graph, inputs)
    del values  # 只計算結構代理值，不以 Python 執行時間推論硬體效能。
    env = {}
    for node in graph.nodes:
        if node.op == "input":
            env[node.name] = inputs[node.name]
        elif node.op == "const":
            env[node.name] = node.value
        elif node.op in {"add", "mul"}:
            env[node.name] = _map_binary(
                (lambda a, b: a + b) if node.op == "add" else (lambda a, b: a * b),
                *(env[name] for name in node.inputs),
            )
        elif node.op == "relu":
            env[node.name] = _map_unary(lambda value: max(0, value), env[node.inputs[0]])
        elif node.op == "fused_relu_affine":
            x, scale, bias = (env[name] for name in node.inputs)
            env[node.name] = _map_ternary(
                lambda value, factor, offset: max(0, value * factor + offset),
                x, scale, bias,
            )
    output_names = set(graph.outputs)
    vector_ops = [
        node for node in graph.nodes
        if node.op not in {"input", "const"} and _is_vector(env[node.name])
    ]
    intermediates = sum(
        len(env[node.name])
        for node in vector_ops
        if node.name not in output_names
    )
    return {
        "operation_nodes": sum(node.op not in {"input", "const"} for node in graph.nodes),
        "vector_operation_nodes": len(vector_ops),
        "intermediate_vector_elements_proxy": intermediates,
    }


def find_node(graph: Graph, name: str) -> Node:
    return next(node for node in graph.nodes if node.name == name)
```


## 練習：補上四個圖改寫

下面的測試會檢查改寫後的節點結構與輸出值。請保持 Graph 不可變：每個方法回傳一個新的 Graph。融合只接受這個小 IR 中的純量 affine 參數，且只在中間結果沒有其他使用者時進行。


```python
class GraphOptimizationSolution:
    @staticmethod
    def constant_fold(graph: Graph) -> Graph:
        # TODO: 對 add / mul 的純量常量輸入預先求值，並保留原節點名稱。
        raise NotImplementedError("完成常量折疊")

    @staticmethod
    def eliminate_common_subexpressions(graph: Graph) -> Graph:
        # TODO: 對純運算建立鍵值表，並更新下游輸入和 graph.outputs。
        raise NotImplementedError("完成公共子表達式消除")

    @staticmethod
    def eliminate_dead_code(graph: Graph) -> Graph:
        # TODO: 從 graph.outputs 反向追蹤依賴，只保留可達節點。
        raise NotImplementedError("完成死碼消除")

    @staticmethod
    def fuse_relu_affine(graph: Graph) -> Graph:
        # TODO: 將 relu(add(mul(x, scale), bias)) 融成一個節點。
        raise NotImplementedError("完成 ReLU-affine 融合")
```



```python
def test_constant_fold():
    graph = Graph((
        Node("x", "input"),
        Node("two_a", "const", value=2),
        Node("two_b", "const", value=2),
        Node("scale", "mul", ("two_a", "two_b")),
        Node("scaled", "mul", ("x", "scale")),
    ), ("scaled",))
    folded = GraphOptimizationSolution.constant_fold(graph)
    assert find_node(folded, "scale").op == "const"
    assert find_node(folded, "scale").value == 4
    assert run_graph(graph, {"x": (1, 2)}) == run_graph(folded, {"x": (1, 2)})
    print("test_constant_fold: pass")


def test_cse():
    graph = Graph((
        Node("x", "input"),
        Node("two", "const", value=2),
        Node("left", "mul", ("x", "two")),
        Node("right", "mul", ("x", "two")),
        Node("out", "add", ("left", "right")),
    ), ("out",))
    optimized = GraphOptimizationSolution.eliminate_common_subexpressions(graph)
    assert sum(node.op == "mul" for node in optimized.nodes) == 1
    assert run_graph(graph, {"x": (2, 3)}) == run_graph(optimized, {"x": (2, 3)})
    print("test_cse: pass")


def test_dead_code_elimination():
    graph = Graph((
        Node("x", "input"),
        Node("one", "const", value=1),
        Node("live", "add", ("x", "one")),
        Node("unused", "mul", ("x", "one")),
    ), ("live",))
    optimized = GraphOptimizationSolution.eliminate_dead_code(graph)
    assert "unused" not in {node.name for node in optimized.nodes}
    assert run_graph(graph, {"x": (4, 5)}) == run_graph(optimized, {"x": (4, 5)})
    print("test_dead_code_elimination: pass")


def test_fusion():
    graph = Graph((
        Node("x", "input"),
        Node("scale", "const", value=2),
        Node("bias", "const", value=1),
        Node("scaled", "mul", ("x", "scale")),
        Node("shifted", "add", ("scaled", "bias")),
        Node("active", "relu", ("shifted",)),
    ), ("active",))
    fused = GraphOptimizationSolution.fuse_relu_affine(graph)
    assert any(node.op == "fused_relu_affine" for node in fused.nodes)
    assert run_graph(graph, {"x": (-2, -1, 0, 1, 2)}) == run_graph(
        fused, {"x": (-2, -1, 0, 1, 2)}
    )
    before = graph_cost_report(graph, {"x": (-2, -1, 0, 1, 2)})
    after = graph_cost_report(fused, {"x": (-2, -1, 0, 1, 2)})
    assert after["vector_operation_nodes"] < before["vector_operation_nodes"]
    assert after["intermediate_vector_elements_proxy"] < before["intermediate_vector_elements_proxy"]
    print("test_fusion: pass")


def test_graph_optimizations():
    test_constant_fold()
    test_cse()
    test_dead_code_elimination()
    test_fusion()


test_graph_optimizations()
```


## 防劇透緩衝帶

先停在這裡完成 TODO，執行測試，再往下看解析。若測試指出失敗，先檢查圖輸出是否更新、節點輸入是否仍指向被移除的名稱，以及融合是否尊重中間節點的使用者數。

## 解析 / 答案：改寫順序與融合條件


```python
class GraphOptimizationSolution:
    @staticmethod
    def constant_fold(graph: Graph) -> Graph:
        values = {}
        nodes = []
        for node in graph.nodes:
            inputs = node.inputs
            if node.op == "const":
                values[node.name] = node.value
                nodes.append(node)
                continue
            if node.op in {"add", "mul"} and len(inputs) == 2 and all(name in values for name in inputs):
                left, right = (values[name] for name in inputs)
                if isinstance(left, Number) and isinstance(right, Number):
                    value = left + right if node.op == "add" else left * right
                    folded = Node(node.name, "const", value=value)
                    values[node.name] = value
                    nodes.append(folded)
                    continue
            nodes.append(Node(node.name, node.op, inputs, node.value))
        return Graph(tuple(nodes), graph.outputs)

    @staticmethod
    def eliminate_common_subexpressions(graph: Graph) -> Graph:
        pure_ops = {"add", "mul", "relu", "fused_relu_affine"}
        commutative_ops = {"add", "mul"}
        aliases = {}
        signatures = {}
        nodes = []

        def resolve(name):
            while name in aliases:
                name = aliases[name]
            return name

        for node in graph.nodes:
            inputs = tuple(resolve(name) for name in node.inputs)
            rewritten = Node(node.name, node.op, inputs, node.value)
            if node.op not in pure_ops:
                nodes.append(rewritten)
                continue
            key_inputs = tuple(sorted(inputs)) if node.op in commutative_ops else inputs
            key = (node.op, key_inputs, node.value)
            if key in signatures:
                aliases[node.name] = signatures[key]
            else:
                signatures[key] = node.name
                nodes.append(rewritten)

        outputs = tuple(resolve(name) for name in graph.outputs)
        return Graph(tuple(nodes), outputs)

    @staticmethod
    def eliminate_dead_code(graph: Graph) -> Graph:
        by_name = {node.name: node for node in graph.nodes}
        live = set()

        def mark(name):
            if name in live:
                return
            live.add(name)
            for dependency in by_name[name].inputs:
                mark(dependency)

        for output in graph.outputs:
            mark(output)
        return Graph(
            tuple(node for node in graph.nodes if node.name in live),
            graph.outputs,
        )

    @staticmethod
    def fuse_relu_affine(graph: Graph) -> Graph:
        by_name = {node.name: node for node in graph.nodes}
        users = {node.name: 0 for node in graph.nodes}
        for node in graph.nodes:
            for dependency in node.inputs:
                users[dependency] += 1
        for output in graph.outputs:
            users[output] += 1

        replacements = {}
        for relu_node in graph.nodes:
            if relu_node.op != "relu":
                continue
            add_node = by_name[relu_node.inputs[0]]
            if add_node.op != "add" or users[add_node.name] != 1:
                continue
            mul_node = None
            bias_node = None
            for name in add_node.inputs:
                candidate = by_name[name]
                if candidate.op == "mul":
                    mul_node = candidate
                elif candidate.op == "const" and isinstance(candidate.value, Number):
                    bias_node = candidate
            if mul_node is None or bias_node is None or users[mul_node.name] != 1:
                continue
            x_name = None
            scale_node = None
            for name in mul_node.inputs:
                candidate = by_name[name]
                if candidate.op == "const" and isinstance(candidate.value, Number):
                    scale_node = candidate
                else:
                    x_name = name
            if x_name is None or scale_node is None:
                continue
            replacements[relu_node.name] = Node(
                relu_node.name,
                "fused_relu_affine",
                (x_name, scale_node.name, bias_node.name),
            )

        nodes = tuple(replacements.get(node.name, node) for node in graph.nodes)
        fused = GraphOptimizationSolution.eliminate_dead_code(Graph(nodes, graph.outputs))
        return fused
```


## 執行最小示例：融合如何改變圖

下面的 DAG 故意包含相同分支、可折疊的常量和一段未連到輸出的死碼。依序進行常量折疊、CSE、DCE 與融合，再比較輸出及結構代理值。


```python
example = Graph((
    Node("x", "input"),
    Node("two_a", "const", value=2),
    Node("two_b", "const", value=2),
    Node("scale", "mul", ("two_a", "two_b")),
    Node("bias", "const", value=1),
    Node("scaled_a", "mul", ("x", "scale")),
    Node("shifted_a", "add", ("scaled_a", "bias")),
    Node("relu_a", "relu", ("shifted_a",)),
    Node("scaled_b", "mul", ("x", "scale")),
    Node("shifted_b", "add", ("scaled_b", "bias")),
    Node("relu_b", "relu", ("shifted_b",)),
    Node("dead", "mul", ("x", "two_b")),
    Node("out", "add", ("relu_a", "relu_b")),
), ("out",))

sample = {"x": (-2, -1, 0, 1, 2)}
optimized = GraphOptimizationSolution.constant_fold(example)
optimized = GraphOptimizationSolution.eliminate_common_subexpressions(optimized)
optimized = GraphOptimizationSolution.eliminate_dead_code(optimized)
optimized = GraphOptimizationSolution.fuse_relu_affine(optimized)

print("改寫前：")
print(graph_text(example))
print("\n改寫後：")
print(graph_text(optimized))

before_value = run_graph(example, sample)
after_value = run_graph(optimized, sample)
assert before_value == after_value
print("\n輸出相同：", before_value == after_value, before_value)

print("\n改寫前結構代理值：", graph_cost_report(example, sample))
print("改寫後結構代理值：", graph_cost_report(optimized, sample))
```


### 如何解讀這個結果？

在這個例子中，兩個相同分支先由 CSE 合併；算術子圖再由一個融合節點表示。原圖有未使用的 dead 節點，DCE 會將它移除。範例以同一組輸入檢查改寫前後的輸出完全相同。

operation_nodes 代表 IR 中的運算節點數。intermediate_vector_elements_proxy 將每個非輸出向量節點的元素數相加，模擬「假設每個中間結果各自實體化一次」的結構成本。真實執行可能重用 buffer，也可能因 backend、layout、排程或其他 pass 產生不同結果；這個數字不是 HBM byte 數，也不是延遲或效能提升百分比。

實際融合還要看資料型別與精度契約。例如，把乘加合成單一指令可能改變浮點捨入；如果要求逐位相同，就必須限制這類改寫。若允許一定誤差，也要用符合 workload 的容差及邊界輸入驗證。layout 轉換、暫存器壓力和 kernel 設定則需在目標硬體上量測。

## 相关阅读與來源

- [Part 1E：編譯最佳化與硬體生態](../../01_Hardware_Math_and_Systems/1E.md)
- [PyTorch FX：Graph 與 Node](https://docs.pytorch.org/docs/stable/fx.html)
- [PyTorch torch.compile 執行路徑](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler.html)
- [MLIR Passes：CSE、DCE 等改寫](https://mlir.llvm.org/docs/Passes/)
- [MLIR Canonicalization：正規化與常量折疊](https://mlir.llvm.org/docs/Canonicalization/)