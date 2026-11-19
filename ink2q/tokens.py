"""Token-set files written by Baekpica's build_calib_inkling2.py: input_ids int32 [N], offsets int64 [n+1]."""
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file


def load_tokens(path: Path) -> list[np.ndarray]:
    t = load_file(str(path))
    ids, off = t["input_ids"].astype(np.int64), t["offsets"].astype(np.int64)
    return [ids[off[i] : off[i + 1]] for i in range(len(off) - 1)]


def load_manifest(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text())["samples"]


def subset(seqs, manifest, max_per_seq: int):
    """First max_per_seq tokens of each sequence, metadata copied."""
    return ([s[:max_per_seq] for s in seqs],
            [dict(m, n_tokens=min(m.get("n_tokens", 10**9), max_per_seq)) for m in manifest])


def write_tokens(path: Path, seqs) -> None:
    ids = np.concatenate(seqs).astype(np.int32)
    off = np.concatenate([[0], np.cumsum([len(s) for s in seqs])]).astype(np.int64)
    save_file({"input_ids": ids, "offsets": off}, str(path))


def write_manifest(path: Path, manifest) -> None:
    Path(path).write_text(json.dumps({"samples": manifest}) + "\n")
