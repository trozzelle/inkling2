"""Token-set files written by Baekpica's build_calib_inkling2.py: input_ids int32 [N], offsets int64 [n+1]."""
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


def load_tokens(path: Path) -> list[np.ndarray]:
    t = load_file(str(path))
    ids, off = t["input_ids"].astype(np.int64), t["offsets"].astype(np.int64)
    return [ids[off[i] : off[i + 1]] for i in range(len(off) - 1)]


def load_manifest(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text())["samples"]
