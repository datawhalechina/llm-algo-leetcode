"""Regression checks for foundation notebook links and clean mirror ordering."""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import check_chapter_links as links


class FoundationLinkTests(unittest.TestCase):
    def test_notebook_prose_is_checked_without_code_or_outputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "01_Hardware_Math_and_Systems"
            source.mkdir()
            path = source / "example.ipynb"
            path.write_text(json.dumps({"cells": [
                {"cell_type": "markdown", "source": ["[old](../03_Triton_Kernels/old.ipynb)"]},
                {"cell_type": "code", "source": ["[not_a_link](code.py)"],
                 "outputs": [{"text": "[not_a_link](output.py)"}]},
            ]}), encoding="utf-8")
            with patch.object(links, "ROOT", root), patch.object(links, "DOC_ROOT", root / "docs"):
                items = list(links.iter_link_sources("source"))
                self.assertEqual(len(items), 1)
                self.assertEqual(links.LINK_RE.findall(items[0][1]), ["../03_Triton_Kernels/old.ipynb"])
                self.assertFalse(any(p.exists() for p in links.resolve_candidates(path, "../03_Triton_Kernels/old.ipynb")))
                target = root / "03_Triton_Kernels/old.ipynb"
                target.parent.mkdir()
                target.write_text("{}")
                self.assertTrue(any(p.exists() for p in links.resolve_candidates(path, "../03_Triton_Kernels/old.ipynb")))

    def test_verify_generates_cross_part_mirrors_before_docs_check(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("repository_verify", root / "verify.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.object(module, "run_python") as run, patch.object(module, "run_docs_build") as build:
            module.verify_chapter0_1(build_docs=False)
        calls = [call.args for call in run.call_args_list]
        self.assertEqual(calls[0], ("tools/check_chapter_links.py", "--scope", "source"))
        conversion = ("tools/convert_notebook.py", "--dir", "03_Triton_Kernels", "--dir", "04_CUDA_and_System_Optimization")
        self.assertLess(calls.index(conversion), calls.index(("tools/check_chapter_links.py", "--scope", "docs")))
        build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
