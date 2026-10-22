"""Compatibility launcher for the relocated foundation regression tests."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "tests" / Path(__file__).name),
        run_name="__main__",
    )
