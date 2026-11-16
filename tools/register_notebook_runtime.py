#!/usr/bin/env python3
"""Compatibility entry; implementation lives in tools.environment.register_notebook_runtime."""

from importlib import import_module
from pathlib import Path
import sys

if __package__ in (None, ""):
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)

_implementation = import_module("tools.environment.register_notebook_runtime")

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
else:
    sys.modules[__name__] = _implementation
