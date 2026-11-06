"""准备 MovieLens-1M：校验压缩包哈希并解压（幂等）。

用法：
    python scripts/prepare_data.py
"""

from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ZIP = DATA / "ml-1m.zip"
TARGET = DATA / "ml-1m"

EXPECTED_SHA256 = "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"
URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    if not ZIP.exists():
        print(f"downloading {URL} ...")
        import urllib.request

        urllib.request.urlretrieve(URL, ZIP)

    digest = sha256(ZIP)
    if digest != EXPECTED_SHA256:
        print(f"sha256 mismatch:\n  expected {EXPECTED_SHA256}\n  actual   {digest}")
        return 1

    if not (TARGET / "ratings.dat").exists():
        print("extracting ...")
        with zipfile.ZipFile(ZIP) as z:
            z.extractall(DATA)

    n = sum(1 for _ in open(TARGET / "ratings.dat", encoding="latin-1"))
    print(f"ok: {n:,} ratings in {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())