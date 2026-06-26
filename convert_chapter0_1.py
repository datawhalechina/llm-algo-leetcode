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
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/{rel_path})
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

    filename = os.path.basename(ipynb_path)
    number, english_title = extract_english_title_from_filename(filename)

    if lines and lines[0].startswith('# '):
        original_title = lines[0][2:].strip()

        chinese_title = original_title
        if ' | ' in original_title:
            chinese_title = original_title.split(' | ')[-1].strip()

        match = re.match(r'^\d+\.\s+(.+)$', chinese_title)
        if match:
            chinese_title = match.group(1)

        if number and english_title:
            lines[0] = f"# {number}. {english_title} | {chinese_title}"

    cloud_env = generate_cloud_env_block(ipynb_path)

    insert_idx = -1
    for i, line in enumerate(lines):
        if "**难度：**" in line or "**Difficulty:**" in line:
            insert_idx = i + 1
            break

    if insert_idx != -1:
        lines.insert(insert_idx, cloud_env)
    else:
        lines.insert(1, cloud_env)

    return '\n'.join(lines)


def normalize_markdown_links(source_text):
    """将 source 侧 Markdown 中的常见链接改写为 docs 镜像可用的路径。"""
    source_text = re.sub(r'(\]\([^)]+)\.ipynb\)', r'\1.md)', source_text)
    source_text = re.sub(
        r'(https://colab\.research\.google\.com/github/datawhalechina/llm-algo-leetcode/blob/main/[^)\s]+)\.md\)',
        r'\1.ipynb)',
        source_text,
    )
    source_text = re.sub(r'\[([^\]]+)\.ipynb\]\(([^)]+)\.md\)', r'[\1.md](\2.md)', source_text)
    source_text = source_text.replace('../docs/', '../')
    return source_text


def process_markdown_file(md_path, out_path):
    """处理纯 Markdown 文件，主要是合并双语标题"""
    filename = os.path.basename(md_path)
    name = filename.replace('.md', '')
    parts = name.split('_')
    if len(parts) < 2:
        with open(md_path, "r", encoding="utf-8") as f:
            source_text = f.read()

        source_text = normalize_markdown_links(source_text)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(source_text)
        return

    number = parts[0]
    english_title = ' '.join(parts[1:])

    with open(md_path, "r", encoding="utf-8") as f:
        source_text = f.read()

    source_text = normalize_markdown_links(source_text)
    lines = source_text.split('\n')

    if lines and lines[0].startswith('# '):
        original_title = lines[0][2:].strip()

        chinese_title = original_title
        if ' | ' in original_title:
            chinese_title = original_title.split(' | ')[-1].strip()

        chinese_title = re.sub(r'^(讨论题\s*\d+[：:]\s*|\d+\.\s+)', '', chinese_title)

        if number and english_title:
            lines[0] = f"# {number}. {english_title} | {chinese_title}"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write('\n'.join(lines))


def main():
    print("=" * 60)
    print("开始构建第零/第一部分文档站点...")
    print("=" * 60)

    for d in ["docs/00_Prerequisites", "docs/01_Hardware_Math_and_Systems"]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    print("✅ 目录清理完成")

    converted_count = 0
    for ipynb_path in sorted(
        glob.glob("00_Prerequisites/*.ipynb") +
        glob.glob("01_Hardware_Math_and_Systems/*.ipynb")
    ):
        out_path = os.path.join("docs", ipynb_path.replace(".ipynb", ".md"))

        with open(ipynb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        md_lines = []
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'markdown':
                source = "".join(cell['source'])
                source = re.sub(r'(\]\([^)]+)\.ipynb\)', r'\1.md)', source)
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

    md_count = 0
    for md_path in sorted(glob.glob("*/*.md")):
        if md_path.startswith("docs/") or md_path.startswith("scripts/"):
            continue
        if not (
            md_path.startswith("00_Prerequisites/")
            or md_path.startswith("01_Hardware_Math_and_Systems/")
        ):
            continue
        out_path = os.path.join("docs", md_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        process_markdown_file(md_path, out_path)
        md_count += 1

    print(f"✅ 处理并复制完成: {md_count} 个 Markdown 文件")
    print("=" * 60)
    print("文档构建全部完成！")


if __name__ == "__main__":
    main()
