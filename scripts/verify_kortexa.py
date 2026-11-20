"""Verify the downloaded expert files of a kortexa EXL3 pack against checkpoint.json's sizes and sha256 (parallel)."""
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main(root: Path) -> int:
    files = json.loads((root / "checkpoint.json").read_text())["files"]
    want = {k: v for k, v in files.items() if k.startswith("layers/")}
    missing = [k for k in want if not (root / k).exists()]
    wrong_size = [k for k in want if (root / k).exists() and (root / k).stat().st_size != want[k]["bytes"]]
    present = [k for k in want if k not in missing and k not in wrong_size]
    with ProcessPoolExecutor(8) as pool:
        digests = dict(zip(present, pool.map(digest, [str(root / k) for k in present], chunksize=8)))
    bad = [k for k, d in digests.items() if d != want[k]["sha256"]]
    print(json.dumps({"expected": len(want), "missing": len(missing), "wrong_size": len(wrong_size), "bad_sha256": len(bad),
                      "examples": (missing + wrong_size + bad)[:5]}))
    return 1 if missing or wrong_size or bad else 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
