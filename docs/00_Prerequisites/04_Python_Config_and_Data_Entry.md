# 04. Python Config and Data Entry | Python 配置与数据入口

**难度：** Easy | **环境：** CPU-first | **标签：** `Python`, `配置`, `I/O` | **目标人群：** Part 2-4 前置补课者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/04_Python_Config_and_Data_Entry.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


本页聚焦：会把路径、配置和临时文件放进稳定的工程结构；会用 `with` 处理文件生命周期；会把读写动作写成可验证的小代码；会把原始输入整理成后面可复用的数据入口。

**关键词：** `config`, `path`, `with`

## 前置阅读
**导语：** 先看 0A 组页，把 Python 对象封装和文件操作的边界对齐，再进入这一页会更顺。
- [03. Python OOP and Utility Patterns | Python 面向对象与工具模式](./03_Python_OOP_and_Utility_Patterns.md)
- [0A 组页](./0A.md)

## 相关阅读
**导语：** 本页先把配置合并、路径读写、临时目录和数据 schema 的最小判断讲清楚；如果想继续把张量和模型前置补完整，可以顺着看下面这些页。
- [05. PyTorch Tensor Fundamentals | PyTorch 张量基础操作](./05_PyTorch_Tensor_Fundamentals.md)

## Q1：配置、路径和资源管理分别解决什么问题？

这页先把三个东西分清：配置负责参数入口，路径负责文件定位，资源管理负责打开和关闭。后面一旦涉及训练脚本、实验目录、数据缓存和临时结果，就要先回到这三个判断。
这里可以先把配置理解成“数据和实验的说明书”：它会先决定数据从哪里读、缓存放在哪里、输出往哪写。


```python
from pathlib import Path
from tempfile import TemporaryDirectory


def merge_config(base, overrides):
    merged = base.copy()
    merged.update(overrides)
    return merged


def write_text_file(path, text):
    path.write_text(text, encoding='utf-8')
    return path.read_text(encoding='utf-8')


def describe_run(base, overrides):
    config = merge_config(base, overrides)
    return f"{config['name']}|{config['precision']}|{config['batch_size']}"

```

## Q1验证：配置合并和路径读写是否稳定？

这里直接检查两个最常见动作：配置能不能覆盖，文件能不能写回再读回。只要这两个动作稳定，后面的实验目录、日志记录和缓存读写就会好管理很多。


```python
base = {'name': 'TinyLM', 'precision': 'bf16', 'batch_size': 4}
overrides = {'precision': 'fp16', 'batch_size': 8}
config = merge_config(base, overrides)
print('合并后的配置：', config)
assert config == {'name': 'TinyLM', 'precision': 'fp16', 'batch_size': 8}

with TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / 'run.txt'
    text = write_text_file(path, describe_run(base, overrides))
    print('文件读回：', text)
    assert text == 'TinyLM|fp16|8'
print('✅ 配置和 I/O 通过')

```

## Q2：什么时候该用临时目录和 `with`？

当你只是临时检查输出、保存中间结果或做一次性调试时，临时目录比手动创建和清理目录更稳，也更不容易残留脏文件。对数据工程来说，这一层常常就是预处理缓存、临时导出和一次性验证的入口。


```python
with TemporaryDirectory() as tmpdir:
    root = Path(tmpdir)
    report = root / 'report.txt'
    report.write_text('hello io', encoding='utf-8')
    print('临时文件路径：', report)
    print('临时文件内容：', report.read_text(encoding='utf-8'))
print('✅ 临时目录已自动收尾')

```

## Q2验证：临时目录是否自动收尾？

这里不用手动删目录，只要看到 `with` 退出后资源不再需要继续维护，就说明 I/O 的生命周期已经对齐。对后面的缓存和实验目录管理来说，这一步很关键。


```python
with TemporaryDirectory() as tmpdir:
    root = Path(tmpdir)
    report = root / 'check.txt'
    report.write_text('done', encoding='utf-8')
    assert report.read_text(encoding='utf-8') == 'done'
    print('✅ 写入和读取通过')

```

## Q3：什么时候该把配置读取和路径操作封装成小工具？

当同一套配置和路径逻辑会在训练、验证、保存里反复出现时，最小封装比到处复制粘贴更稳，也更容易检查输出是否一致。这里再往前想一步，就是把原始输入先整理成固定 schema，再把这种 schema 用同一套路径规则保存下来；后面进入 Tensor、batch 和 checkpoint 时，这个习惯会直接复用。


```python
from pathlib import Path


def build_run_path(root, name, suffix='txt'):
    # 统一路径拼接方式，后面缓存、日志和预处理输出都可以复用。
    root = Path(root)
    return root / f'{name}.{suffix}'


def render_report(config):
    # 用稳定字符串描述一次运行，便于后面做缓存名或日志名。
    return f"{config['name']}|{config['precision']}|{config['batch_size']}"


cfg = {'name': 'TinyLM', 'precision': 'bf16', 'batch_size': 4}
path = build_run_path('/tmp', cfg['name'])
print('run_path:', path)
print('report:', render_report(cfg))
assert str(path).endswith('TinyLM.txt')
assert render_report(cfg) == 'TinyLM|bf16|4'
print('✅ path_tool 通过')

```

### 本节小结

- `config` 负责入口约定，`path` 负责把约定落到文件位置。
- `with` 和临时目录负责控制生命周期，避免脏文件和资源泄漏。
- 配置、I/O 和 schema 一起看，后面进入 Tensor、batch 和 checkpoint 时会更顺。
