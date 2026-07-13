#!/usr/bin/env python3
"""
通用的 Notebook 答案验证脚本

用法:
    python tools/test_notebook_answers.py 00_PyTorch_Warmup.ipynb
    python tools/test_notebook_answers.py --all  # 测试所有notebook
"""

import sys
import argparse
import nbformat
from pathlib import Path
import subprocess
import tempfile

# 理论章节（无代码实现，只需占位符测试）
THEORY_CHAPTERS = [
    '21_CUDA_vs_Triton_vs_PyTorch.ipynb',
]

# 分布式多进程章节（需要特殊环境，测试脚本中只验证结构）
DISTRIBUTED_CHAPTERS = [
    '19_Distributed_Communication_Primitives.ipynb',
]


def extract_question_code(notebook_path):
    """从notebook中提取题目区的代码（TODO部分）"""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    question_code = []
    in_question_section = False

    for cell in nb.cells:
        # 检测题目区开始（import之后，STOP HERE之前）
        if cell.cell_type == 'code' and 'import' in cell.source and not in_question_section:
            in_question_section = True
            continue

        # 检测题目区结束
        if cell.cell_type == 'markdown' and 'STOP HERE' in cell.source:
            break

        # 在题目区且是代码cell
        if in_question_section and cell.cell_type == 'code':
            # 跳过测试代码
            if 'def test_' in cell.source or 'test_' in cell.source and '()' in cell.source:
                continue
            question_code.append(cell.source)

    return '\n\n'.join(question_code)


def extract_answer_code(notebook_path):
    """从notebook中提取答案区的代码"""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    answer_code = []
    in_answer_section = False

    for cell in nb.cells:
        # 检测答案区开始
        if cell.cell_type == 'markdown' and '参考代码与解析' in cell.source:
            in_answer_section = True
            continue

        # 在答案区且是代码cell
        if in_answer_section and cell.cell_type == 'code':
            # 提取所有代码（包括测试函数）
            answer_code.append(cell.source)

    return '\n\n'.join(answer_code)


def extract_test_code(notebook_path, test_mode='answer'):
    """从notebook中提取测试代码

    Args:
        notebook_path: notebook文件路径
        test_mode: 'answer' 或 'question'，用于判断是否需要特殊处理
    """
    notebook_name = Path(notebook_path).name

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # 检查是否是理论章节
    if notebook_name in THEORY_CHAPTERS:
        # 理论章节返回占位符测试
        return """
def test_theory():
    print("✅ 理论章节，无需代码测试")
    return True

test_theory()
"""

    in_answer_section = False
    fallback_test_code = None
    question_candidates = []

    for cell in nb.cells:
        if cell.cell_type == 'markdown' and '参考代码与解析' in cell.source:
            in_answer_section = True
            continue
        if cell.cell_type == 'markdown' and 'STOP HERE' in cell.source and test_mode == 'question':
            break
        if cell.cell_type == 'code' and 'def test_' in cell.source:
            if test_mode == 'answer' and not in_answer_section:
                fallback_test_code = fallback_test_code or cell.source
                continue
            return cell.source
        if test_mode == 'question' and cell.cell_type == 'code' and not in_answer_section:
            question_candidates.append(cell.source)

    if test_mode == 'question' and question_candidates:
        return question_candidates[-1]

    return fallback_test_code


def extract_imports(notebook_path):
    """从notebook中提取import语句"""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # 查找第一个代码cell（通常是import cell）
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'import' in cell.source:
            return cell.source

    return ""


def test_notebook_answers(notebook_path, test_mode='answer'):
    """测试notebook的答案代码

    Args:
        notebook_path: notebook文件路径
        test_mode: 'answer' 测试答案区代码, 'question' 测试题目区代码
    """
    mode_name = "答案区" if test_mode == 'answer' else "题目区"
    print(f"\n{'='*60}")
    print(f"测试: {notebook_path} ({mode_name})")
    print(f"{'='*60}\n")

    # 提取代码
    imports = extract_imports(notebook_path)

    if test_mode == 'answer':
        code = extract_answer_code(notebook_path)
    else:
        code = extract_question_code(notebook_path)

    test_code = extract_test_code(notebook_path, test_mode)

    if not code:
        print(f"❌ 未找到{mode_name}代码")
        return False

    if not test_code:
        print(f"❌ 未找到测试代码")
        return False

    # 组合完整的测试脚本
    full_script = f"""
{imports}

{code}

{test_code}
"""

    # 写入临时文件并执行
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(full_script)
        temp_path = f.name

    try:
        result = subprocess.run(
            ['python', temp_path],
            capture_output=True,
            text=True,
            timeout=30
        )

        print(result.stdout)
        if result.stderr:
            print("错误输出:")
            print(result.stderr)

        skip_markers = [
            '无 GPU，完成结构检查；运行级验证需要 GPU。',
            '结构检查通过',
            '忽略测试：无 GPU',
            '忽略测试：此环境没有 NVIDIA GPU',
            '无 GPU，跳过运行级验证。',
        ]
        if result.returncode == 0 and any(marker in result.stdout for marker in skip_markers):
            print(f"\n⏭️ {notebook_path} {mode_name}仅完成结构检查，已跳过运行级验证（无 GPU）")
            return 'skip'

        # 答案区的通过标准以进程退出码为准。
        # `✅` 只作为人类可读的结果展示，不应影响判定。
        success = result.returncode == 0

        if success:
            print(f"\n✅ {notebook_path} {mode_name}测试通过")
            return 'pass'

        if test_mode == 'question' and 'NotImplementedError' in result.stderr:
            print(f"\n❌ {notebook_path} {mode_name}按预期失败（未完成题目）")
            return 'expected_fail'

        print(f"\n❌ {notebook_path} {mode_name}测试失败")
        return 'fail'

    except subprocess.TimeoutExpired:
        print(f"❌ 测试超时")
        return 'fail'
    except Exception as e:
        print(f"❌ 执行出错: {e}")
        return 'fail'
    finally:
        Path(temp_path).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description='测试Notebook答案代码')
    parser.add_argument('notebook', nargs='?', help='Notebook文件路径')
    parser.add_argument('--all', action='store_true', help='测试所有notebook')
    parser.add_argument('--dir', default='.', help='Notebook目录路径')
    parser.add_argument('--mode', choices=['answer', 'question', 'both'], default='answer',
                        help='测试模式: answer=答案区, question=题目区, both=两者都测试')

    args = parser.parse_args()

    if args.all:
        # 测试目录下所有notebook
        notebook_dir = Path(args.dir)
        notebooks = sorted(notebook_dir.glob('*.ipynb'))

        if not notebooks:
            print(f"未找到notebook文件: {notebook_dir}")
            return 1

        results = {}
        for nb_path in notebooks:
            # 跳过checkpoint文件
            if '.ipynb_checkpoints' in str(nb_path):
                continue
            if args.mode == 'both':
                # 测试题目区和答案区
                q_success = test_notebook_answers(nb_path, 'question')
                a_success = test_notebook_answers(nb_path, 'answer')
                results[nb_path.name] = {'question': q_success, 'answer': a_success}
            else:
                success = test_notebook_answers(nb_path, args.mode)
                results[nb_path.name] = success

        # 汇总结果
        print(f"\n{'='*60}")
        print("测试汇总")
        print(f"{'='*60}")

        if args.mode == 'both':
            for name, result in results.items():
                def fmt(status):
                    return {
                        'pass': "✅",
                        'skip': "⏭️ 跳过（无 GPU）",
                        'expected_fail': "❌ 按预期失败（未完成题目）",
                        'fail': "❌",
                    }.get(status, "❓")

                q_status = fmt(result['question'])
                a_status = fmt(result['answer'])
                print(f"{name}:")
                print(f"  题目区: {q_status}")
                print(f"  答案区: {a_status}")

            total = len(results)
            q_passed = sum(1 for r in results.values() if r['question'] == 'pass')
            q_expected_fail = sum(1 for r in results.values() if r['question'] == 'expected_fail')
            a_passed = sum(1 for r in results.values() if r['answer'] == 'pass')
            a_expected_fail = sum(1 for r in results.values() if r['answer'] == 'expected_fail')
            q_skipped = sum(1 for r in results.values() if r['question'] == 'skip')
            a_skipped = sum(1 for r in results.values() if r['answer'] == 'skip')
            print(f"\n题目区通过: {q_passed}/{total}")
            print(f"题目区按预期失败: {q_expected_fail}/{total}")
            print(f"题目区跳过: {q_skipped}/{total}")
            print(f"答案区通过: {a_passed}/{total}")
            print(f"答案区按预期失败: {a_expected_fail}/{total}")
            print(f"答案区跳过: {a_skipped}/{total}")
            return 0 if all(r['question'] != 'fail' and r['answer'] != 'fail' for r in results.values()) else 1
        else:
            for name, success in results.items():
                status = {
                    'pass': "✅ 通过",
                    'skip': "⏭️ 跳过（无 GPU）",
                    'expected_fail': "🟡 按预期失败",
                    'fail': "❌ 失败",
                }.get(success, "❓ 未知")
                print(f"{status} - {name}")

            total = len(results)
            passed = sum(1 for v in results.values() if v == 'pass')
            skipped = sum(1 for v in results.values() if v == 'skip')
            print(f"\n总计: {passed}/{total} 通过, {skipped}/{total} 跳过")
            return 0 if all(v != 'fail' for v in results.values()) else 1

    elif args.notebook:
        # 测试单个notebook
        if args.mode == 'both':
            q_success = test_notebook_answers(args.notebook, 'question')
            a_success = test_notebook_answers(args.notebook, 'answer')
            print(f"\n{'='*60}")
            print("测试结果")
            print(f"{'='*60}")
            q_label = {
                'pass': '✅ 通过',
                'skip': '⏭️ 跳过（无 GPU）',
                'expected_fail': '❌ 按预期失败（未完成题目）',
                'fail': '❌ 失败',
            }.get(q_success, '❓ 未知')
            a_label = {
                'pass': '✅ 通过',
                'skip': '⏭️ 跳过（无 GPU）',
                'expected_fail': '❌ 按预期失败（未完成题目）',
                'fail': '❌ 失败',
            }.get(a_success, '❓ 未知')
            print(f"题目区: {q_label}")
            print(f"答案区: {a_label}")
            return 0 if q_success != 'fail' and a_success != 'fail' else 1
        else:
            success = test_notebook_answers(args.notebook, args.mode)
            return 0 if success != 'fail' else 1

    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
