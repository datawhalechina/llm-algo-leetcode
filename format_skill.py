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
        # 1. Remove special symbols globally
        text = re.sub(r'[🟢🔴🟡💻📖🎯🚀💡⚡⚠️🏆🔥📐🔍]', '', text)
        
        # 2. Fix LaTeX corruption caused by Python escape sequences
        # \f turning into \x0c
        text = text.replace("\x0crac", "\\frac")
        # \r turning into \r or carriage return split
        text = text.replace("\r\n\\right)", "\\right)")
        text = text.replace("\\r\",\n    \"\\right)", "}\\right)")
        # common \left \right issues
        text = text.replace("\\left(", "\\left(")  # ensuring no \l loss
        text = text.replace("eft(", "\\left(")
        text = text.replace("ight)", "\\right)")
        text = text.replace("ight]", "\\right]")
        text = text.replace("\x09imes", "\\times")
        text = text.replace("imes", "\\times")
        # clean stray newlines before \right
        text = re.sub(r'\\r\n\\right', r'\\right', text)

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
        if process_file(f):
            count += 1
            
    print(f"Total files updated: {count}")
