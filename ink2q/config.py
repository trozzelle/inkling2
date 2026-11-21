"""Pinned revisions and filesystem layout."""
import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPO = "thinkingmachines/Inkling-2-Small-MXFP8"
SOURCE_REV = "f69fa26848416b8f69a0576ef29278b0d3e92d2d"
OQE_REPO = "Hygaard/Inkling-2-Small-oQe-3.07bpw-MLX"
HELDOUT_SHA256 = "83d7bf03af780618c45fafe128124b0adfc37f4451632cb3ad9e643a88adca5f"
CALIB_SHA256 = "d5aeda5741ebf2151c8205a7d3c9b2dda9f9482d4d00776824fdf55a8409ec21"


@dataclass(frozen=True)
class Paths:
    root: Path
    source: Path
    data: Path

    @classmethod
    def from_env(cls) -> "Paths":
        root = Path(os.environ.get("INK2Q_ROOT", REPO_ROOT / "work"))
        source = Path(os.environ.get("INK2Q_SOURCE", root / "source"))
        data = Path(os.environ.get("INK2Q_DATA", root / "data"))
        return cls(root, source, data)
