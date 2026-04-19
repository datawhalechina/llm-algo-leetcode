import os
import glob
import shutil
import json
import re

def generate_cloud_env_block(ipynb_path):
    """生成云端运行环境区块"""
    rel_path = ipynb_path.replace('\\', '/')
    cloud_env = f"""
> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lynnyulinlin-debug/llm-algo-leetcode/blob/main/{rel_path})
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*
"""
    return cloud_env

def extract_english_title_from_filename(filename):
    """从文件名提取编号和英文标题（不带编号）"""
    name = filename.replace('.ipynb', '')
    parts = name.split('_')
    if len(parts) < 2:
        return None, None
    number = parts[0]
    title = ' '.join(parts[1:])
    return number, title

def process_first_cell(source_text, ipynb_path):
    """处理第一个 Markdown cell，重构标题并插入云端环境"""
    lines = source_text.split('\n')

    # 1. 处理标题：提取编号和英文标题
    filename = os.path.basename(ipynb_path)
    number, english_title = extract_english_title_from_filename(filename)

    if lines and lines[0].startswith('# '):
        original_title = lines[0][2:].strip()

        # 提取中文部分（去除已有的英文部分和编号）
        chinese_title = original_title
        if ' | ' in original_title:
            chinese_title = original_title.split(' | ')[-1].strip()

        # 去除中文部分的编号（如 "01. "）
        match = re.match(r'^\d+\.\s+(.+)$', chinese_title)
        if match:
            chinese_title = match.group(1)

        # 构建标准双语标题
        if number and english_title:
            lines[0] = f"# {number}. {english_title} | {chinese_title}"

    # 2. 插入云端运行环境：在难度标签下方
    cloud_env = generate_cloud_env_block(ipynb_path)

    # 寻找难度标签的行号
    insert_idx = -1
    for i, line in enumerate(lines):
        if "**难度：**" in line or "**Difficulty:**" in line:
            insert_idx = i + 1  # 插入在难度标签的下一行
            break

    if insert_idx != -1:
        # 如果找到难度标签，插入云端环境
        lines.insert(insert_idx, cloud_env)
    else:
        # 如果没找到，插在标题（第0行）后面
        lines.insert(1, cloud_env)

    return '\n'.join(lines)

def process_markdown_file(md_path, out_path):
    """处理纯 Markdown 文件，主要是合并双语标题"""
    filename = os.path.basename(md_path)
    # 对于 md 文件，提取方式和 ipynb 略有不同（后缀不同）
    name = filename.replace('.md', '')
    parts = name.split('_')
    if len(parts) < 2:
        shutil.copy2(md_path, out_path)
        return

    number = parts[0]
    english_title = ' '.join(parts[1:])

    with open(md_path, "r", encoding="utf-8") as f:
        source_text = f.read()

    lines = source_text.split('\n')

    # 处理双语标题
    if lines and lines[0].startswith('# '):
        original_title = lines[0][2:].strip()

        chinese_title = original_title
        if ' | ' in original_title:
            chinese_title = original_title.split(' | ')[-1].strip()

        # 去除中文标题中常见的开头："讨论题 01："、"01. " 等
        chinese_title = re.sub(r'^(讨论题\s*\d+[：:]\s*|\d+\.\s+)', '', chinese_title)

        if number and english_title:
            lines[0] = f"# {number}. {english_title} | {chinese_title}"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write('\n'.join(lines))

def main():
    print("=" * 60)
    print("开始构建文档站点...")
    print("=" * 60)

    # 1. Clear out all docs chapter directories
    for d in ["docs/00_Prerequisites", "docs/01_Hardware_Math_and_Systems", "docs/02_PyTorch_Algorithms", "docs/03_CUDA_and_Triton_Kernels"]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    print("✅ 目录清理完成")

    # 2. Iterate through all IPYNB notebooks
    converted_count = 0
    for ipynb_path in sorted(
        glob.glob("00_Prerequisites/*.ipynb") +
        glob.glob("01_Hardware_Math_and_Systems/*.ipynb") +
        glob.glob("02_PyTorch_Algorithms/*.ipynb") +
        glob.glob("03_CUDA_and_Triton_Kernels/*.ipynb")
    ):
        out_path = os.path.join("docs", ipynb_path.replace(".ipynb", ".md"))

        with open(ipynb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        md_lines = []
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'markdown':
                source = "".join(cell['source'])

                # 处理链接替换
                source = re.sub(r'(\]\([^)]+)\.ipynb\)', r'\1.md)', source)

                # 如果是第一个 cell，执行特殊处理（标题重构 + 插入云端环境）
                if i == 0:
                    source = process_first_cell(source, ipynb_path)

                md_lines.append(source)

            elif cell['cell_type'] == 'code':
                source = "".join(cell['source'])
                if source.strip():
                    md_lines.append("\n```python\n" + source + "\n```\n")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        converted_count += 1

    print(f"✅ 转换完成: {converted_count} 个 Notebook")

    # 3. Process and Copy ALL raw Markdown files
    md_count = 0
    for md_path in sorted(glob.glob("*/*.md")):
        if md_path.startswith("docs/") or md_path.startswith("scripts/"):
            continue
        out_path = os.path.join("docs", md_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # 走处理逻辑而不是直接复制
        process_markdown_file(md_path, out_path)
        md_count += 1

    print(f"✅ 处理并复制完成: {md_count} 个 Markdown 文件")
    print("=" * 60)
    print("文档构建全部完成！")

if __name__ == "__main__":
    main()