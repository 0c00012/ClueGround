#!/usr/bin/env python
"""Check local import closure and parse every published CSV artifact."""

from __future__ import annotations

import ast
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOTS = ("scripts", "src", "baselines", "tools")


def module_names() -> set[str]:
    names: set[str] = set()
    for package in PACKAGE_ROOTS:
        for path in (ROOT / package).rglob("*.py"):
            names.add(".".join(path.relative_to(ROOT).with_suffix("").parts))
    return names


def check_imports() -> list[str]:
    available = module_names()
    missing: list[str] = []
    for package in PACKAGE_ROOTS:
        for path in (ROOT / package).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported = [node.module]
                    if node.module in PACKAGE_ROOTS:
                        imported.extend(f"{node.module}.{alias.name}" for alias in node.names)
                else:
                    continue
                for name in imported:
                    if not name.startswith(tuple(f"{p}." for p in PACKAGE_ROOTS)):
                        continue
                    if not any(
                        candidate == name
                        or candidate.startswith(f"{name}.")
                        or name.startswith(f"{candidate}.")
                        for candidate in available
                    ):
                        missing.append(f"{path.relative_to(ROOT)}: {name}")
    return sorted(set(missing))


def check_csvs() -> tuple[int, list[str]]:
    failures: list[str] = []
    paths = sorted((ROOT / "results").rglob("*.csv"))
    paths.extend([ROOT / "docs" / "PAPER_RESULT_LINEAGE.csv", ROOT / "docs" / "SPLIT_FINGERPRINTS.csv"])
    for path in paths:
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                list(csv.reader(handle))
        except Exception as exc:  # pragma: no cover - release audit path
            failures.append(f"{path.relative_to(ROOT)}: {exc}")
    return len(paths), failures


def main() -> None:
    missing = check_imports()
    csv_count, csv_failures = check_csvs()
    failures = missing + csv_failures
    if failures:
        raise SystemExit("Release structure audit failed:\n" + "\n".join(failures))
    print(f"PASS: local import closure and {csv_count} CSV files verified")


if __name__ == "__main__":
    main()
