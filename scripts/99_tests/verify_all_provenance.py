#!/usr/bin/env python3
"""
Verify SHA-256 checksums of every data_raw subfolder against provenance.yml.

Run periodically to detect data drift, accidental modification, or corruption.

Usage:
    python3 scripts/99_tests/verify_all_provenance.py
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib.provenance import verify_checksums


def main():
    raw_dir = PROJECT_ROOT / "data_raw"
    folders = [d for d in raw_dir.iterdir()
                 if d.is_dir() and (d / "provenance.yml").exists()]

    print("=" * 70)
    print(f"Verifying SHA-256 checksums for {len(folders)} data sources")
    print("=" * 70)

    all_ok = True
    for folder in sorted(folders):
        print(f"\n[*] {folder.name} ...")
        try:
            ok = verify_checksums(folder)
            if ok:
                print(f"    ✓ ALL VALID")
            else:
                print(f"    ✗ MISMATCH FOUND")
                all_ok = False
        except Exception as e:
            print(f"    ! ERROR: {type(e).__name__}: {e}")
            all_ok = False

    print("\n" + "=" * 70)
    if all_ok:
        print("VERIFICATION PASSED — all data files match recorded checksums.")
        return 0
    else:
        print("VERIFICATION FAILED — see details above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
