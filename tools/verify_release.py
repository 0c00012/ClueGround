#!/usr/bin/env python
"""Verify the SHA-256 manifest of the public ClueGround release."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "checksums" / "RELEASE_MANIFEST.sha256"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    failures: list[str] = []
    checked = 0
    for raw in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        expected, relative = raw.split("  ", 1)
        path = ROOT / relative
        checked += 1
        if not path.is_file():
            failures.append(f"missing: {relative}")
        elif sha256(path) != expected:
            failures.append(f"hash mismatch: {relative}")
    if failures:
        raise SystemExit("Release verification failed:\n" + "\n".join(failures))
    print(f"PASS: verified {checked} files")


if __name__ == "__main__":
    main()
