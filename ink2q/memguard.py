"""MLX memory limits and a free-memory floor (a GPU watchdog panic left 11 MB free)."""
import json
import re
import subprocess
from dataclasses import dataclass, field
from typing import Callable

GIB = 1 << 30


def apply_limits(memory_gib: float, cache_gib: float) -> None:
    import mlx.core as mx

    mx.set_memory_limit(int(memory_gib * GIB))
    mx.set_cache_limit(int(cache_gib * GIB))


def parse_vm_stat(text: str) -> int:
    page = int(re.search(r"page size of (\d+) bytes", text).group(1))

    def pages(name: str) -> int:
        return int(re.search(rf"^Pages {name}:\s+(\d+)\.", text, re.M).group(1))

    return (pages("free") + pages("inactive") + pages("speculative")) * page


def available_bytes() -> int:
    return parse_vm_stat(subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout)


class MemoryFloorError(RuntimeError):
    pass


@dataclass
class Floor:
    gib: float
    read: Callable[[], int] = field(default=available_bytes)

    def check(self, phase: str) -> int:
        avail = self.read()
        if avail < self.gib * GIB:
            raise MemoryFloorError(f"{phase}: {avail / GIB:.1f} GiB available, floor {self.gib} GiB")
        print(json.dumps({"phase": phase, "available_gib": round(avail / GIB, 1)}), flush=True)
        return avail
