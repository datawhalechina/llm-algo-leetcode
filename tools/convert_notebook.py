import os
import glob
import shutil
import json
import re
import argparse

SOURCE_DIRS = [
    "00_Prerequisites",
    "01_Hardware_Math_and_Systems",
    "02_PyTorch_Algorithms",
    "03_Triton_Kernels",
    "04_CUDA_and_System_Optimization",
]

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

def normalize_markdown_links(source_text):
    """将 source 侧 Markdown 中的常见链接改写为 docs 镜像可用的路径。"""
    source_text = re.sub(r'(\]\([^)]+)\.ipynb\)', r'\1.md)', source_text)
    source_text = source_text.replace('../docs/', '../')
    return source_text

def process_markdown_file(md_path, out_path):
    """处理纯 Markdown 文件，主要是合并双语标题"""
    filename = os.path.basename(md_path)
    # Group pages like 2_1.md / 3_1.md / 4_1.md should keep their
    # hand-written section titles instead of being rewritten as "2. 1 | ...".
    if re.fullmatch(r"\d+_\d+\.md", filename):
        with open(md_path, "r", encoding="utf-8") as f:
            source_text = f.read()

        source_text = normalize_markdown_links(source_text)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(source_text)
        return

    # 对于 md 文件，提取方式和 ipynb 略有不同（后缀不同）
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

    # 先把指向源码/镜像目录的常见链接统一改写到 docs 侧可用的相对路径
    source_text = normalize_markdown_links(source_text)
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


def collect_targets(args):
    """收集需要处理的源文件。"""
    targets = []

    if args.files:
        targets.extend(args.files)

    if args.dirs:
        for d in args.dirs:
            targets.extend(sorted(glob.glob(os.path.join(d, "*.ipynb"))))
            targets.extend(sorted(glob.glob(os.path.join(d, "*.md"))))

    if not targets:
        for d in SOURCE_DIRS:
            targets.extend(sorted(glob.glob(os.path.join(d, "*.ipynb"))))
        for d in SOURCE_DIRS:
            targets.extend(sorted(glob.glob(os.path.join(d, "*.md"))))

    deduped = []
    seen = set()
    for item in targets:
        norm = os.path.normpath(item)
        if norm in seen:
            continue
        if not (
            norm.startswith("00_Prerequisites" + os.sep)
            or norm.startswith("01_Hardware_Math_and_Systems" + os.sep)
            or norm.startswith("02_PyTorch_Algorithms" + os.sep)
            or norm.startswith("03_Triton_Kernels" + os.sep)
            or norm.startswith("04_CUDA_and_System_Optimization" + os.sep)
        ):
            continue
        seen.add(norm)
        deduped.append(norm)
    return deduped


def clean_full_docs_tree():
    for d in [
        "docs/00_Prerequisites",
        "docs/01_Hardware_Math_and_Systems",
        "docs/02_PyTorch_Algorithms",
        "docs/03_Triton_Kernels",
        "docs/04_CUDA_and_System_Optimization",
    ]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)


def output_path_for_source(source_path):
    return os.path.join("docs", source_path.replace(".ipynb", ".md"))


def write_text(path, content, dry_run=False):
    if dry_run:
        print(f"[dry-run] write {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def main():
    parser = argparse.ArgumentParser(description="Convert Chapter 2/3 notebooks and markdown to docs mirror.")
    parser.add_argument("--dir", dest="dirs", action="append", help="Only convert the specified source directory.")
    parser.add_argument("--file", dest="files", action="append", help="Only convert the specified source file.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without writing files.")
    args = parser.parse_args()

    print("=" * 60)
    print("开始构建文档站点...")
    print("=" * 60)

    targets = collect_targets(args)
    if not args.dirs and not args.files:
        if args.dry_run:
            print("[dry-run] 清理 docs/02_PyTorch_Algorithms 和 docs/03_Triton_Kernels")
        else:
            clean_full_docs_tree()
        print("✅ 目录清理完成")
    else:
        print("✅ 局部模式：跳过整树清理")

    converted_count = 0
    md_count = 0
    for source_path in targets:
        out_path = output_path_for_source(source_path)

        if source_path.endswith(".ipynb"):
            if args.dry_run:
                print(f"[dry-run] convert notebook {source_path} -> {out_path}")
                converted_count += 1
                continue

            with open(source_path, "r", encoding="utf-8") as f:
                nb = json.load(f)

            md_lines = []
            for i, cell in enumerate(nb['cells']):
                if cell['cell_type'] == 'markdown':
                    source = "".join(cell['source'])
                    source = re.sub(r'(\]\([^)]+)\.ipynb\)', r'\1.md)', source)
                    if i == 0:
                        source = process_first_cell(source, source_path)
                    md_lines.append(source)

                elif cell['cell_type'] == 'code':
                    source = "".join(cell['source'])
                    if source.strip():
                        md_lines.append("\n```python\n" + source + "\n```\n")

            write_text(out_path, "\n".join(md_lines))
            converted_count += 1
        else:
            if args.dry_run:
                print(f"[dry-run] copy markdown {source_path} -> {out_path}")
                md_count += 1
                continue
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            process_markdown_file(source_path, out_path)
            md_count += 1

    print(f"✅ 转换完成: {converted_count} 个 Notebook")

    print(f"✅ 处理并复制完成: {md_count} 个 Markdown 文件")
    print("=" * 60)
    print("文档构建全部完成！")

if __name__ == "__main__":
    main()
