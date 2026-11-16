"""Regression fixtures for GitHub risks that local MathJax cannot detect."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.check_math_formula_symbols import github_compatibility_issues, scan_file


class MathCompatibilityTests(unittest.TestCase):
    """Cover the observed rendering failures without network or GPU dependencies."""

    def messages(self, text):
        return [message for _, message in github_compatibility_issues(Path('example.md'), text)]

    def test_chinese_punctuation_before_formula(self):
        for punctuation in '，。、；：（）':
            with self.subTest(punctuation=punctuation):
                self.assertTrue(any('边界风险' in s for s in self.messages('说明' + punctuation + '$p_1,p_2$')))

    def test_spaced_formula_and_sequence(self):
        self.assertEqual(self.messages(r'用 $[p_1,p_2,a_1,a_2,\mathrm{EOS}]$ 表示： $p_1,p_2$。'), [])

    def test_block_and_inline_operatorname(self):
        for text in (r'值 $\operatorname{SiLU}(x)$', '$$\n\\operatorname{offset}(i,j)\n$$'):
            self.assertTrue(any('宏兼容' in s for s in self.messages(text)))

    def test_escaped_punctuation(self):
        for command in (r'\,', r'\!', r'\;', r'\:', r'\%', r'\#', r'\{', r'\}'):
            with self.subTest(command=command):
                self.assertTrue(any('转义风险' in s for s in self.messages('$a' + command + 'b$')))

    def test_aligned_row_separator_is_not_single_escape(self):
        self.assertEqual(self.messages('$$\n\\begin{aligned}\na&=b,\\\\\nc&=d.\n\\end{aligned}\n$$'), [])

    def test_star_and_safe_replacement(self):
        self.assertTrue(any('星号' in s for s in self.messages('$z^*$')))
        self.assertEqual(self.messages(r'$z^{\ast}$'), [])

    def test_accent_subscript(self):
        self.assertTrue(any('下标边界' in s for s in self.messages(r'$\bar{s}_{\pi}$')))
        self.assertEqual(self.messages(r'$\bar s_\pi=\frac{s_\pi}{N}$'), [])

    def test_parenthesis_delimiters(self):
        self.assertTrue(any('分隔符' in s for s in self.messages(r'这里 \(x,y\)')))

    def test_fences_and_inline_code_are_not_prose(self):
        text = '```python\n$\\operatorname{x}$\n```\n~~~text\n，$x$\n~~~\n' + r'`$\operatorname{x}$`'
        self.assertEqual(self.messages(text), [])

    def test_long_fence_containing_short_fence(self):
        self.assertEqual(self.messages('````md\n```\n，$x$\n```\n````\n'), [])

    def test_notebook_prose_only_and_cell_location(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'example.ipynb'
            path.write_text(json.dumps({'cells': [
                {'cell_type': 'code', 'source': ['，$ignored$']},
                {'cell_type': 'markdown', 'source': ['第一行\n', '说明，$x$']},
            ]}))
            before = path.read_bytes()
            with patch('tools.check_math_formula_symbols.validate_formulas', side_effect=AssertionError('Node must not run')):
                issues = scan_file(path, github_compatible=True, static_only=True)
            self.assertEqual(len(issues), 1)
            self.assertEqual((issues[0][0].cell, issues[0][0].line), (2, 2))
            self.assertEqual(before, path.read_bytes())

    def test_static_cli_exit_status(self):
        script = ROOT / 'tools/check_math_formula_symbols.py'
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'example.md'
            for text, status in [('说明，$x$', 1), ('说明， $x$', 0)]:
                path.write_text(text)
                result = subprocess.run([sys.executable, str(script), '--github-compatible',
                                         '--static-only', '--file', str(path)], capture_output=True, text=True)
                self.assertEqual(result.returncode, status, result.stdout + result.stderr)
                if status == 0:
                    self.assertIn('final GitHub display not verified', result.stdout)


if __name__ == '__main__':
    unittest.main()
