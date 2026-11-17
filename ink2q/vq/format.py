"""R13+8 expert code layout and a numpy reference decoder.

Per matrix [N rows, K cols], G = K // 8 groups per row (G % 16 == 0):
  lo12 uint8 [N, G*3//2]  low 12 bits of the stage-1 index, pairs (a, b) packed as
                          [a & 255, ((a >> 8) & 15) | ((b & 15) << 4), (b >> 4) & 255]
  hi1  uint8 [N, G//8]    bit 12 of index g is bit g % 8 of byte g // 8, LSB first
  i2   uint8 [N, G]       stage-2 index
  gain float16 [N]
  c1   uint8 [8192, 8] E4M3, c2 uint8 [256, 8] E4M3
  W[r, 8g+d] = f32(gain[r]) * (e4m3(c1[i1[r,g], d]) + e4m3(c2[i2[r,g], d]))
Stacked experts add a leading E axis to lo12, hi1, i2 and gain.
"""
import numpy as np


def e4m3_table() -> np.ndarray:
    b = np.arange(256, dtype=np.uint32)
    e = (b >> 3) & 15
    m = (b & 7).astype(np.float64)
    v = np.where(e == 0, m * 2.0**-9, (1.0 + m / 8.0) * np.exp2(e.astype(np.float64) - 7.0))
    v = np.where(b & 128, -v, v)
    v[(b & 127) == 127] = np.nan
    return v.astype(np.float32)


def pack(i1: np.ndarray, i2: np.ndarray):
    assert i1.shape == i2.shape and i1.shape[-1] % 16 == 0
    lo = i1.astype(np.uint32) & 0xFFF
    a, b = lo[..., 0::2], lo[..., 1::2]
    lo12 = np.stack([a & 255, ((a >> 8) & 15) | ((b & 15) << 4), (b >> 4) & 255], axis=-1)
    lo12 = lo12.astype(np.uint8).reshape(*i1.shape[:-1], -1)
    hi1 = np.packbits(((i1 >> 12) & 1).astype(np.uint8), axis=-1, bitorder="little")
    return lo12, hi1, i2.astype(np.uint8)


def unpack(lo12: np.ndarray, hi1: np.ndarray, i2: np.ndarray):
    t = lo12.reshape(*lo12.shape[:-1], -1, 3).astype(np.uint16)
    a = t[..., 0] | ((t[..., 1] & 15) << 8)
    b = (t[..., 1] >> 4) | (t[..., 2] << 4)
    low = np.stack([a, b], axis=-1).reshape(*lo12.shape[:-1], -1)
    hi = np.unpackbits(hi1, axis=-1, bitorder="little")[..., : low.shape[-1]].astype(np.uint16)
    return low | (hi << 12), i2


def decode(lo12, hi1, i2, gain, c1, c2) -> np.ndarray:
    t = e4m3_table()
    j1, j2 = unpack(lo12, hi1, i2)
    v = t[c1[j1]] + t[c2[j2]]                      # [..., N, G, 8]
    w = v.reshape(*v.shape[:-2], -1)               # [..., N, K]
    return (w * gain.astype(np.float32)[..., None]).astype(np.float32)
