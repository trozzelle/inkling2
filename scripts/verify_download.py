"""Verify a local Hugging Face snapshot: every LFS file's sha256 and size against the Hub at a pinned revision."""
import hashlib
import sys
from pathlib import Path

from huggingface_hub import HfApi


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    repo, rev, root = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    bad = []
    for e in HfApi().list_repo_tree(repo, revision=rev, recursive=True, expand=True):
        lfs = getattr(e, "lfs", None)
        if lfs is None:
            continue
        p = root / e.path
        ok = p.exists() and p.stat().st_size == lfs.size and sha256(p) == lfs.sha256
        if not ok:
            bad.append(e.path)
        print(("ok  " if ok else "BAD ") + e.path, flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
