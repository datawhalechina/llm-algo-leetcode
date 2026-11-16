"""Compatibility entry for test_math_formula_compatibility.py; implementation lives in tools/tests."""

from pathlib import Path
import runpy


def load_tests(loader, tests, pattern):
    """Forward explicit module runs; let discovery collect the canonical file."""
    if pattern is not None:
        return tests
    import sys

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return loader.loadTestsFromName("tools.tests.test_math_formula_compatibility")


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "tests" / Path(__file__).name),
        run_name="__main__",
    )
