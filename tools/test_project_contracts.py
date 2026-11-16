"""Compatibility entry aggregating the relocated project contract checks."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUITES = (
    "tools.tests.test_project_contracts",
    "tools.tests.test_part02_project_structure",
    "tools.tests.test_quantization_project_contracts",
)


def load_tests(loader, tests, pattern):
    """Run all legacy checks explicitly; avoid duplicate discovery."""
    if pattern is not None:
        return tests
    return loader.loadTestsFromNames(SUITES)


def main():
    """Preserve the callable smoke entry and raise on failed assertions."""
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromNames(SUITES))
    if not result.wasSuccessful():
        raise AssertionError("project contract regression failed")


if __name__ == "__main__":
    main()
