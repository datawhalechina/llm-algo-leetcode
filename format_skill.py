import glob
import json
import re
import argparse
import sys

def process_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            if filepath.endswith(".ipynb"):
                data = json.load(f)
            else:
                data = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False

    modified = False

    def clean_text(text):
        original = text
        # 1. Remove special emojis globally
        text = re.sub(r'[🟢🔴🟡💻📖🎯🚀💡⚡⚠️🏆🔥📐🔍]', '', text)
        
        # 2. Fix Metadata Markdown syntax
        # The user reported that some lines ended up like: 难度： Medium | 标签： ... | **目标人群：**
        # We need to ensure that the line STRICTLY matches:
        # **难度：** Medium | **标签：** ... | **目标人群：** ...
        if "难度：" in text and "标签：" in text and "目标人群：" in text:
            # Strip all asterisks first to parse cleanly
            clean_meta = text.replace("**", "").replace("*", "")
            diff_match = re.search(r"难度：\s*([^\|]+)\|", clean_meta)
            tags_match = re.search(r"标签：\s*([^\|]+)\|", clean_meta)
            target_match = re.search(r"目标人群：\s*(.*?)(?:\n|$)", clean_meta)
            
            if diff_match and tags_match and target_match:
                diff = diff_match.group(1).strip()
                tags = tags_match.group(1).strip()
                target = target_match.group(1).strip()
                
                new_meta = f"**难度：** {diff} | **标签：** {tags} | **目标人群：** {target}"
                if text.endswith("\n"):
                    new_meta += "\n"
                text = new_meta

        # 3. Fix LaTeX corruption caused by Python escape sequences
        
        # Fractions (\f -> \x0c -> \frac)
        text = text.replace("\x0crac", "\\frac")
        
        # Right parens / brackets (\r -> carriage return)
        text = re.sub(r'\\r(?:\\r)*\\right', r'\\right', text)
        text = text.replace("\r\n\\right)", "\\right)")
        text = text.replace("\\r\",\n    \"\\right)", "}\\right)")
        text = text.replace("ight)", "\\right)")
        text = text.replace("ight]", "\\right]")
        text = re.sub(r'\\r\n\\right', r'\\right', text)
        
        # Left parens (\left instead of eft)
        text = re.sub(r'\\l(?:\\l)*\\left', r'\\left', text)
        text = text.replace("\\left(", "\\left(")
        text = text.replace("eft(", "\\left(")
        
        # Times (\t -> tab -> \times)
        text = re.sub(r'\\t(?:\\t)*\\times', r'\\times', text)
        text = text.replace("\x09imes", "\\times")
        text = text.replace("imes", "\\times")
        
        # Otimes (\o\t\t\times -> \otimes)
        text = re.sub(r'\\o[\\a-z\x00-\x1f]*\\times', r'\\otimes', text)
        text = text.replace("\\o\\t\\t\\times", "\\otimes")
        text = text.replace("\\o\\times", "\\otimes")
        
        # Final cleanup for multiple escapes
        text = text.replace("\\\\l\\\\left", "\\\\left")
        text = text.replace("\\\\r\\\\right", "\\\\right")

        return text, text != original

    if filepath.endswith(".ipynb"):
        for cell in data.get("cells", []):
            if cell.get("cell_type") == "markdown":
                for i, line in enumerate(cell.get("source", [])):
                    cleaned, changed = clean_text(line)
                    if changed:
                        cell["source"][i] = cleaned
                        modified = True
    else:
        cleaned, changed = clean_text(data)
        if changed:
            data = cleaned
            modified = True

    if modified:
        with open(filepath, "w", encoding="utf-8") as f:
            if filepath.endswith(".ipynb"):
                json.dump(data, f, ensure_ascii=False, indent=1)
            else:
                f.write(data)
        print(f"Cleaned: {filepath}")
    
    return modified

if __name__ == "__main__":
    """
    ===============================================================
    常见错误说明和处理方式 (维护在 format_skill.py)
    ===============================================================
    
    当 Python 的 JSON 库或者一些基于字符串的批量替换工具在处理 Markdown 文本时，
    极容易因为转义字符引发以下毁坏性的 LaTeX / 排版错误：
    
    1. LaTeX 公式符号被误解析为 Python 控制符：
       - `\frac` 被吃掉首字母，变成了 `\f` (换页符, hex: \x0c) + `rac`。
       - `\right` 被当成了 `\r` (回车符, hex: \x0d)，导致换行并剩下 `ight`。
       - `\times` 被当成了 `\t` (制表符, hex: \x09) 加上 `imes`。
       - `\left` 被吞噬 `\l` 变成了 `eft`。
       - `\otimes` 变成了 `\o` + tab + tab + `times`。
       处理方式：在 Python 替换中，用具体的正则表达式或者替换串反向映射，例如：
       `text.replace("\x0crac", "\\frac")` 和 `text.replace("ight)", "\\right)")`。
       
    2. 反复格式化叠加带来的残渣：
       - 多次处理后可能会留下 `\l\l\l\left` 或是 `\r\r\r\right` 等重复反斜杠的尸体。
       处理方式：使用 `re.sub(r'\\r(?:\\r)*\\right', r'\\right', text)` 强行清洗掉前置累赘。
       
    3. Metadata 行 Markdown 粗体标记残缺：
       - 之前用正则表达式全局去除 emoji 时，不慎切断了前置的 `**难度：**` 和 `**标签：**` 等。
       - 产生了 `难度： Medium | 标签：... | **目标人群：**` 这种粗细不均的排版。
       处理方式：在本脚本中定义一个全局的拦截器，只要某行包含 "难度"、"标签" 和 "目标人群"，
       就先用 `.replace("**", "")` 脱掉所有星号，最后使用一致的模板重新拼装：
       `f"**难度：** {diff} | **标签：** {tags} | **目标人群：** {target}"`
       
    """
    
    parser = argparse.ArgumentParser(description="Format LaTeX and remove special symbols from markdown/notebooks")
    parser.add_argument("path", help="File or directory path to process (e.g. 03_CUDA_and_Triton_Kernels/)")
    args = parser.parse_args()
    
    path = args.path
    if path.endswith(".ipynb") or path.endswith(".md"):
        files = [path]
    else:
        files = glob.glob(f"{path}/**/*.ipynb", recursive=True) + glob.glob(f"{path}/**/*.md", recursive=True)
        
    count = 0
    for f in files:
        if "node_modules" not in f and process_file(f):
            count += 1
            
    print(f"Total files updated: {count}")
