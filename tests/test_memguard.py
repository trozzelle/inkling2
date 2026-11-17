import pytest

from ink2q.memguard import Floor, MemoryFloorError, parse_vm_stat

VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                 13189198.
Pages active:                                1538665.
Pages inactive:                              1494728.
Pages speculative:                             65583.
Pages throttled:                                   0.
Pages wired down:                             418578.
Pages purgeable:                              198946.
"""


def test_parse_vm_stat():
    assert parse_vm_stat(VM_STAT) == (13189198 + 1494728 + 65583) * 16384


def test_floor_passes_and_fails():
    gib = 1 << 30
    assert Floor(10, read=lambda: 11 * gib).check("ok") == 11 * gib
    with pytest.raises(MemoryFloorError, match="layer 3"):
        Floor(24, read=lambda: 23 * gib).check("layer 3")
