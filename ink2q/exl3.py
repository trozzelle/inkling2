"""EXL3 trellis decoding (ExLlamaV3's format, as specified by TensorFold's docs/recipes/exl3.md and its scalar
reference zig/src/core/exl3_format.zig). A layer is W[K, N] = diag(suh) · H_K · W_q · H_N · diag(svh), with W_q decoded
from 16x16 tiles of a circular bitstream and H the 128x128 Sylvester Hadamard matrix / sqrt(128).

Two paths: numpy (every codebook, whole and half widths; the oracle tests) and MLX (whole widths, all codebooks are
supported for states, the mul1 codebook for values: what the kortexa Inkling pack uses) for whole layers."""
import mlx.core as mx
import numpy as np

INST3_MUL, INST3_ADD, MCG_MUL, MUL1_MUL = 89226354, 64248484, 0xCBAC1FED, 0x83DCD12D
MASK, FLIP = 0x8FFF8FFF, 0x3B603B60
MUL1_SCALE = float(np.array([0x1EEE], np.uint16).view(np.float16)[0])
MUL1_BIAS = float(np.array([0xC931], np.uint16).view(np.float16)[0])
CODEBOOKS = ("inst3", "mcg", "mul1")
HD = 128


def stream_ends(halves: int) -> np.ndarray:
    """E(p), p = 0..255: where value p's 16-bit window ends in the tile's bitstream (exclusive)."""
    p1 = np.arange(1, 257, dtype=np.int64)
    if halves % 2 == 0:
        return p1 * halves // 2
    ka = (halves - 1) // 2
    return (p1 * (2 * ka + 1) - (p1 % 2)) // 2


def _windows(halves: int):
    ring = 256 * halves // 2
    start = (stream_ends(halves) + ring - 16) % ring
    return start, ring


def tile_states_np(halves: int, tiles: np.ndarray) -> np.ndarray:
    """tiles int16 [T, 8*halves] -> states uint32 [T, 256] (stream order)."""
    start, ring = _windows(halves)
    words = tiles.view(np.uint16).astype(np.uint32)
    w32 = words[:, 0::2] | (words[:, 1::2] << 16)                  # little-endian pairs of int16 words
    n32 = w32.shape[1]
    bit_idx = (start[:, None] + np.arange(16)[None, :]) % ring     # [256, 16], first bit most significant
    word = w32[:, bit_idx // 32]                                    # [T, 256, 16]
    bits = (word >> (31 - (bit_idx % 32)).astype(np.uint32)) & 1
    weights = (1 << np.arange(15, -1, -1)).astype(np.uint32)
    assert n32 * 32 == ring or halves % 2 == 1
    return (bits * weights).sum(-1).astype(np.uint32)


def codebook_np(codebook: str, s: np.ndarray) -> np.ndarray:
    """16-bit states -> fp16 values, one rounding per state."""
    s = s.astype(np.uint64)
    if codebook == "mul1":
        x = (s * MUL1_MUL) & 0xFFFFFFFF
        h = 1024 + (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255)
        return (h.astype(np.float64) * MUL1_SCALE + MUL1_BIAS).astype(np.float16)
    x = (s * (MCG_MUL if codebook == "mcg" else INST3_MUL) + (0 if codebook == "mcg" else INST3_ADD)) & 0xFFFFFFFF
    m = ((x & MASK) ^ FLIP).astype(np.uint32)
    lo = (m & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float64)
    hi = (m >> 16).astype(np.uint16).view(np.float16).astype(np.float64)
    return (lo + hi).astype(np.float16)


def placement() -> tuple[np.ndarray, np.ndarray]:
    p = np.arange(256)
    lane, j = p // 8, p % 8
    return 2 * (lane % 4) + (j & 1) + 8 * ((j >> 1) & 1), lane // 4 + 8 * (j >> 2)


def unpack_np(halves: int, codebook: str, trellis: np.ndarray) -> np.ndarray:
    """trellis int16 [K/16, N/16, 8*halves] -> W_q fp16 [K, N] (rotated domain)."""
    kt, nt, words = trellis.shape
    vals = codebook_np(codebook, tile_states_np(halves, trellis.reshape(-1, words))).reshape(kt, nt, 256)
    row, col = placement()
    tile = np.empty((kt, nt, 16, 16), np.float16)
    tile[:, :, row, col] = vals
    return tile.transpose(0, 2, 1, 3).reshape(kt * 16, nt * 16)


def hadamard(n: int = HD) -> np.ndarray:
    i = np.arange(n)
    par = np.vectorize(lambda v: bin(v).count("1") & 1)(i[:, None] & i[None, :])
    return np.where(par == 1, -1.0, 1.0) / np.sqrt(n)


def dequant_np(halves: int, codebook: str, trellis, suh, svh) -> np.ndarray:
    """W [K, N] float64 = diag(suh) H_K W_q H_N diag(svh)."""
    wq = unpack_np(halves, codebook, trellis).astype(np.float64)
    k, n = wq.shape
    h = hadamard()
    wq = np.einsum("ij,bjn->bin", h, wq.reshape(k // HD, HD, n)).reshape(k, n) * np.asarray(suh, np.float64)[:, None]
    wq = np.einsum("bmj,ij->bmi", wq.reshape(k, n // HD, HD), h).reshape(k, n) * np.asarray(svh, np.float64)[None, :]
    return wq


# ---- MLX path (whole widths, mul1 values) -------------------------------------------------------------------------

def _fwht(x: mx.array, axis: int) -> mx.array:
    """Blockwise Sylvester Hadamard along `axis` (length a multiple of 128), scaled by 1/sqrt(128). Butterflies only."""
    x = mx.moveaxis(x, axis, -1)
    shape = x.shape
    y = x.reshape(-1, HD)
    h = 1
    while h < HD:
        y = y.reshape(-1, HD // (2 * h), 2, h)
        a, b = y[:, :, 0, :], y[:, :, 1, :]
        y = mx.stack([a + b, a - b], axis=2)
        h *= 2
    y = y.reshape(shape) * (1.0 / np.sqrt(HD))
    return mx.moveaxis(y, -1, axis)


def dequant_mx(bits: int, trellis: mx.array, suh: mx.array, svh: mx.array) -> mx.array:
    """mul1, whole bits: trellis int16 [K/16, N/16, 16*bits] -> W [K, N] float32."""
    kt, nt, words = trellis.shape
    start, ring = _windows(2 * bits)
    w16 = trellis.reshape(-1, words).view(mx.uint16).astype(mx.uint32)
    w32 = w16[:, 0::2] | (w16[:, 1::2] << 16)                        # [T, 8*bits]
    n32 = ring // 32
    w0 = mx.array((start // 32).astype(np.int32))
    off = mx.array((start % 32).astype(np.uint32))
    a = w32[:, w0]                                                    # [T, 256]
    b = w32[:, mx.array(((start // 32 + 1) % n32).astype(np.int32))]
    # the 16 bits from `off` (MSB-first) across words a then b
    hi_part = mx.where(off <= 16, a >> (16 - mx.minimum(off, 16)), a << (mx.maximum(off, 16) - 16))
    lo_part = mx.where(off > 16, b >> (48 - mx.maximum(off, 17)), mx.zeros_like(b))
    s = (hi_part | lo_part) & 0xFFFF
    x = s * MUL1_MUL                                                  # uint32 wraps mod 2^32
    h = 1024 + (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24)
    # exact in fp32 (22 significant bits), then one rounding to fp16
    vals = (h.astype(mx.float32) * MUL1_SCALE + MUL1_BIAS).astype(mx.float16).astype(mx.float32)
    row, col = placement()
    inv = np.empty(256, np.int64)
    inv[row * 16 + col] = np.arange(256)                              # tile position -> stream index
    tile = vals[:, mx.array(inv.astype(np.int32))].reshape(kt, nt, 16, 16)
    wq = tile.transpose(0, 2, 1, 3).reshape(kt * 16, nt * 16)
    w = _fwht(wq, 0) * suh.astype(mx.float32)[:, None]
    return _fwht(w, 1) * svh.astype(mx.float32)[None, :]


def expert_weight(f: dict, e: int, proj: str) -> mx.array:
    """One expert projection of a kortexa file as our [out, in] float32 (W[K, N] transposed)."""
    t = f[f"experts.{e}.{proj}.trellis"]
    bits = t.shape[-1] // 16
    return dequant_mx(bits, t, f[f"experts.{e}.{proj}.suh"], f[f"experts.{e}.{proj}.svh"]).T


def kortexa_spec(root, layer: int):
    """All of one layer's routed experts from a kortexa-inkling2-exl3-v1 pack, decoded to bf16 SwitchLinear modules
    (gate_up [E, 4608, 4608] with interleaved gate/up rows as in the source; down [E, 4608, 2304])."""
    import json
    from pathlib import Path

    from mlx import nn
    from mlx_lm.models.switch_layers import SwitchLinear

    from ink2q.experts import ExpertSpec

    root = Path(root)
    ck = json.loads((root / "checkpoint.json").read_text())
    batches = dict(ck["layers"])[str(layer)]["batches"]
    parts = {"w13": [], "w2": []}
    for b in batches:
        f = mx.load(str(root / b["file"]))
        for e in range(b["start"], b["end"]):
            for proj in parts:
                w = expert_weight(f, e - b["start"] if f"experts.{e - b['start']}.w13.trellis" in f else e, proj)
                parts[proj].append(w.astype(mx.bfloat16))
        mx.eval(parts["w13"][-8:], parts["w2"][-8:])
        del f
    mods = []
    for proj in ("w13", "w2"):
        m = SwitchLinear.__new__(SwitchLinear)
        nn.Module.__init__(m)
        m.weight = mx.stack(parts[proj])
        parts[proj] = []
        mx.eval(m.weight)
        m.freeze()
        mods.append(m)
    return ExpertSpec(mods[0], mods[1], f"exl3:{root.name}")
