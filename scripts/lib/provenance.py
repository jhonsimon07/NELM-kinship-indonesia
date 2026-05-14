"""
Provenance tracking utility for PJ2.

Every raw dataset must have a sibling provenance.yml documenting:
  - source URL
  - access date
  - retrieval method
  - license
  - SHA-256 checksum of every file

Usage:

    from scripts.lib.provenance import write_provenance, verify_checksums

    write_provenance(
        target_dir=Path("data_raw/bps_susenas_profilmigran"),
        dataset_name="bps_profil_migran_2024",
        source_url="https://www.bps.go.id/.../profil-migran-2024.html",
        license="Public domain (BPS Indonesia)",
        access_method="WebFetch + manual PDF download",
        intended_use=["PJ2b time-series migration analysis"],
        notes="Free PDF download from BPS publication catalog",
    )

    # Later, to verify integrity:
    ok = verify_checksums(Path("data_raw/bps_susenas_profilmigran"))
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml


# ---------------------------------------------------------------------------

def sha256_file(path: Path, chunk_size: int = 8192) -> str:
    """Return SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def file_size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 3)


def list_data_files(target_dir: Path,
                     extensions: Iterable[str] = (".csv", ".dta", ".sav",
                                                    ".xlsx", ".xls", ".pdf",
                                                    ".json", ".parquet",
                                                    ".tsv", ".txt", ".html",
                                                    ".zip", ".tar.gz")) -> list[Path]:
    """List all data files in target_dir, excluding provenance.yml itself."""
    files = []
    for p in sorted(target_dir.rglob("*")):
        if p.is_file() and p.name != "provenance.yml":
            if any(p.name.lower().endswith(ext) for ext in extensions):
                files.append(p)
    return files


# ---------------------------------------------------------------------------

def write_provenance(
    target_dir: Path,
    dataset_name: str,
    source_url: str,
    license: str,
    access_method: str,
    intended_use: list[str],
    source_organisation: str = "",
    license_url: str = "",
    notes: str = "",
    extra_fields: dict | None = None,
) -> Path:
    """
    Write provenance.yml in target_dir documenting all data files.

    Computes SHA-256 + size for every file. Returns path to provenance.yml.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    files = list_data_files(target_dir)
    file_records = []
    for f in files:
        rel = f.relative_to(target_dir.parent)
        file_records.append({
            "path": str(rel),
            "name": f.name,
            "size_mb": file_size_mb(f),
            "sha256": sha256_file(f),
        })

    record = {
        "dataset_name": dataset_name,
        "source_organisation": source_organisation,
        "source_url": source_url,
        "license": license,
        "license_url": license_url,
        "access_method": access_method,
        "access_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": file_records,
        "n_files": len(file_records),
        "total_size_mb": round(sum(r["size_mb"] for r in file_records), 3),
        "intended_use": intended_use,
        "notes": notes,
    }
    if extra_fields:
        record.update(extra_fields)

    out = target_dir / "provenance.yml"
    with out.open("w") as f:
        yaml.safe_dump(record, f, sort_keys=False, allow_unicode=True,
                       default_flow_style=False, width=88)
    return out


# ---------------------------------------------------------------------------

def verify_checksums(target_dir: Path) -> tuple[bool, list[str]]:
    """
    Re-compute SHA-256 for every file in provenance.yml; return (ok, mismatches).

    Returns:
        ok: True if all checksums match
        mismatches: list of file paths whose current hash differs from recorded
    """
    target_dir = Path(target_dir)
    prov_path = target_dir / "provenance.yml"
    if not prov_path.exists():
        return False, [f"provenance.yml not found in {target_dir}"]
    with prov_path.open() as f:
        record = yaml.safe_load(f)
    mismatches = []
    for entry in record.get("files", []):
        rel = Path(entry["path"])
        full = target_dir.parent / rel
        if not full.exists():
            mismatches.append(f"MISSING: {rel}")
            continue
        actual = sha256_file(full)
        if actual != entry["sha256"]:
            mismatches.append(f"HASH_MISMATCH: {rel} "
                              f"(expected {entry['sha256'][:12]}, "
                              f"got {actual[:12]})")
    return len(mismatches) == 0, mismatches


# ---------------------------------------------------------------------------

def cli_audit_all(data_root: Path):
    """Walk data_root, audit every subdirectory with provenance.yml."""
    issues = {}
    for subdir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        prov = subdir / "provenance.yml"
        if not prov.exists():
            issues[subdir.name] = ["NO_PROVENANCE_YML"]
            continue
        ok, msgs = verify_checksums(subdir)
        if not ok:
            issues[subdir.name] = msgs
    return issues


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.lib.provenance audit <data_raw_path>")
        sys.exit(1)
    if sys.argv[1] == "audit":
        root = Path(sys.argv[2])
        issues = cli_audit_all(root)
        if not issues:
            print(f"OK — all provenance.yml files in {root} verified")
            sys.exit(0)
        for name, msgs in issues.items():
            print(f"\n=== {name} ===")
            for m in msgs:
                print(f"  {m}")
        sys.exit(1)
