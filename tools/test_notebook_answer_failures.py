"""Regression tests for expected exercise failures in the notebook runner."""

import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import nbformat

from tools.test_notebook_answers import (
    has_explicit_todo_failure,
    strict_question_failure_issues,
    test_notebook_answers as run_notebook_answers,
)


class ExpectedTodoFailureTests(unittest.TestCase):
    """Keep incomplete exercises distinct from broken implementations."""

    def test_python_todo_guard(self):
        self.assertTrue(has_explicit_todo_failure('NotImplementedError: TODO 1：填写索引'))

    def test_local_triton_todo_assertion(self):
        stderr = (
            'triton.compiler.errors.CompileTimeAssertionFailure: at 7:4:\n'
            '    tl.static_assert(False, "TODO 2：补全加载")\n'
            'TODO 2：补全加载\n'
        )
        self.assertTrue(has_explicit_todo_failure(stderr))

    def test_other_failures_are_not_expected(self):
        cases = (
            'triton.compiler.errors.CompilationError: TODO 1\nAttributeError: missing API',
            'triton.compiler.errors.CompileTimeAssertionFailure: at 7:4:\nBLOCK_SIZE must be positive',
            'AssertionError: incorrect output\n# TODO 1：load input',
            'RuntimeError: CUDA compilation failed\n# raise NotImplementedError("TODO")',
        )
        for stderr in cases:
            with self.subTest(stderr=stderr):
                self.assertFalse(has_explicit_todo_failure(stderr))

    def test_secondary_errors_remain_failures(self):
        self.assertEqual(
            strict_question_failure_issues('NameError: x\nNotImplementedError: TODO 1'),
            ['NameError'],
        )


class NotebookSubprocessImportTests(unittest.TestCase):
    """Temporary extracted scripts must retain the notebook checkout imports."""

    def test_part03_and_part04_receive_their_checkout_root(self):
        for part in ('03_Triton_Kernels', '04_CUDA_and_System_Optimization'):
            with self.subTest(part=part), TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / part / '01_Import_Probe.ipynb'
                source.parent.mkdir()
                (root / 'runner_checkout_probe.py').write_text('VALUE = 7\n', encoding='utf-8')
                cells = [
                    ('tutorial-imports', 'import sys'),
                    ('tutorial-question', 'raise NotImplementedError("TODO 1")'),
                    ('tutorial-test', 'from runner_checkout_probe import VALUE\nassert VALUE == 7'),
                    ('tutorial-answer', 'answer = 7'),
                ]
                notebook = nbformat.v4.new_notebook(cells=[
                    nbformat.v4.new_code_cell(code, metadata={'tags': [tag]})
                    for tag, code in cells
                ])
                nbformat.write(notebook, source)
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    status = run_notebook_answers(source, 'answer')
                self.assertEqual(status, 'pass', output.getvalue())


if __name__ == '__main__':
    unittest.main()
