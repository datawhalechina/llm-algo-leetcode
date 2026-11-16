"""Create v1 sidecar results for legacy training-memory measurements.

Usage:
    python tools/normalize_training_memory_results.py \
      benchmarks/results/73_real_gpu_training.json \
      benchmarks/results/76_real_gpu_memory.json \
      benchmarks/results/75_memory_budget_decision.json

The command preserves raw inputs and conflicting outputs, including when an
explicit output directory is supplied. Identical sidecars can be reused.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .training_memory_result_schema import normalize_file
except ImportError:  # Direct script execution.
    from training_memory_result_schema import normalize_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Legacy result JSON files")
    parser.add_argument("--output-dir", type=Path, help="Directory for generated sidecar files")
    parser.add_argument("--dry-run", action="store_true", help="Validate outputs without writing files")
    args = parser.parse_args()

    for source in args.inputs:
        output = None
        if args.output_dir:
            output = args.output_dir / f"{source.stem}_v1.json"
        target = normalize_file(source, output, dry_run=args.dry_run)
        print(f"[dry-run] {target}" if args.dry_run else target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
