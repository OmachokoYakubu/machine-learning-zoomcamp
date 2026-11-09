#!/usr/bin/env python3
"""Download and SHA-256 verify the UCI #275 Capital Bikeshare hourly dataset."""

from __future__ import annotations

import argparse
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
EXPECTED_ZIP_SHA256 = "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401"
EXPECTED_HOUR_CSV_SHA256 = (
    "e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "bike_sharing_hour.csv"


def verify_sha256(content: bytes, expected: str, label: str) -> str:
    digest = hashlib.sha256(content).hexdigest()
    if digest != expected:
        raise ValueError(
            f"SHA-256 mismatch for {label}: expected {expected}, got {digest}"
        )
    return digest


def ensure_dataset(output_path: Path = DEFAULT_OUTPUT, force: bool = False) -> Path:
    if output_path.exists() and not force:
        existing_bytes = output_path.read_bytes()
        digest = verify_sha256(
            existing_bytes, EXPECTED_HOUR_CSV_SHA256, output_path.name
        )
        print(f"[OK] Verified existing {output_path} (sha256={digest[:16]}...)")
        return output_path

    print(f"Downloading UCI #275 archive from {UCI_ZIP_URL} ...")
    req = urllib.request.Request(UCI_ZIP_URL, headers={"User-Agent": "BikePulse/1.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        zip_bytes = response.read()

    verify_sha256(zip_bytes, EXPECTED_ZIP_SHA256, "bike+sharing+dataset.zip")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        hour_bytes = archive.read("hour.csv")

    digest = verify_sha256(hour_bytes, EXPECTED_HOUR_CSV_SHA256, "hour.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(hour_bytes)
    print(
        f"[OK] Saved {output_path} ({len(hour_bytes):,} bytes, sha256={digest[:16]}...)"
    )
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and verify UCI #275 Bike Sharing hourly dataset"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if local file matches"
    )
    args = parser.parse_args()
    ensure_dataset(args.output, force=args.force)

