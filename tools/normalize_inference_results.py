"""Convert legacy 66--70 backend JSON into the shared result envelope.

The input file is never overwritten by default. This keeps historical raw
measurements intact while producing a schema-compatible companion report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from inference_result_schema import from_backend_report
    from result_migration_io import save_migration_report
except ModuleNotFoundError:  # pragma: no cover - direct repository execution
    from tools.inference_result_schema import from_backend_report
    from tools.result_migration_io import save_migration_report


def normalize_file(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    project: str = "66",
    strategy: str | None = None,
    dry_run: bool = False,
) -> Path:
    source = Path(input_path)
    report = json.loads(source.read_text(encoding="utf-8"))
    if "normalized_result" in report:
        normalized = report["normalized_result"]
    else:
        normalized = from_backend_report(
            report,
            project=project,
            strategy=strategy,
            backend="vllm",
            dtype="unknown",
        )

    target = Path(output_path) if output_path else source.with_name(f"{source.stem}_normalized.json")
    return save_migration_report(source, target, normalized, dry_run=dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize a legacy inference result without overwriting it")
    parser.add_argument("input", help="Legacy backend JSON path")
    parser.add_argument("--output", help="Output normalized JSON path")
    parser.add_argument("--project", default="66", help="Project number")
    parser.add_argument("--strategy", help="Optional strategy name")
    parser.add_argument("--dry-run", action="store_true", help="Validate output without writing files")
    args = parser.parse_args()
    target = normalize_file(args.input, output_path=args.output, project=args.project,
                            strategy=args.strategy, dry_run=args.dry_run)
    print(f"[dry-run] {target}" if args.dry_run else target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
