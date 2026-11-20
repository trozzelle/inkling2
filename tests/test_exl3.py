"""EXL3 decoder against TensorFold's oracle fixture (tests/fixtures/exl3_oracle.bin, from TensorFold's
tools/exl3_fixture.py, Apache-2.0): per-tile states for 27 codebook/width cases, codebook spot values, and one real
mcg 8.00 bpw block unpacked and dequantized."""
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np

from ink2q import exl3 as E

FIX = Path(__file__).parent / "fixtures" / "exl3_oracle.bin"


def u(fmt, b, at):
    return struct.unpack_from("<" + fmt, b, at)[0]


def records(b):
    assert b[:8] == b"EXL3ORAC"
    at, out = 16, []
    for _ in range(u("Q", b, 8)):
        halves, cb, kt, nt = u("H", b, at), b[at + 2], u("I", b, at + 4), u("I", b, at + 8)
        words = 8 * halves
        tb = at + 12
        sb = tb + kt * nt * words * 2
        out.append((halves, cb, np.frombuffer(b, np.int16, kt * nt * words, tb).reshape(kt * nt, words),
                    np.frombuffer(b, np.uint32, kt * nt * 256, sb).reshape(kt * nt, 256)))
        at = sb + kt * nt * 256 * 4
    return out, at


def test_tile_states_match_oracle_for_every_codebook_and_width():
    b = FIX.read_bytes()
    recs, _ = records(b)
    tiles = 0
    for halves, _, trellis, want in recs:
        np.testing.assert_array_equal(E.tile_states_np(halves, trellis), want)
        tiles += len(trellis)
    assert tiles == 27 * 6


def test_codebook_spot_values_match_oracle():
    b = FIX.read_bytes()
    _, at = records(b)
    for i in range(u("I", b, at)):
        o = at + 4 + 6 * i
        cb, state, want = u("H", b, o), u("H", b, o + 2), u("H", b, o + 4)
        got = E.codebook_np(E.CODEBOOKS[cb], np.array([state])).view(np.uint16)[0]
        assert got == want


def test_real_mcg_block_unpacks_and_dequantizes():
    b = FIX.read_bytes()
    _, at = records(b)
    at += 4 + u("I", b, at) * 6
    halves, kt, nt, tlen = u("H", b, at), u("I", b, at + 2), u("I", b, at + 6), u("I", b, at + 10)
    at += 14
    trellis = np.frombuffer(b, np.int16, tlen // 2, at).reshape(kt, nt, -1)
    at += tlen

    def blob(dtype):
        nonlocal at
        n = u("I", b, at)
        a = np.frombuffer(b, dtype, n // np.dtype(dtype).itemsize, at + 4)
        at += 4 + n
        return a

    suh, svh = blob(np.float16), blob(np.float16)
    wq_want, dq_want = blob(np.uint16), blob(np.float64)
    np.testing.assert_array_equal(E.unpack_np(halves, "mcg", trellis).view(np.uint16).ravel(), wq_want)
    dq = E.dequant_np(halves, "mcg", trellis, suh, svh).ravel()
    assert np.abs(dq - dq_want).max() <= 1e-9 * (1 + np.abs(dq_want).max())


def test_mlx_mul1_path_matches_numpy():
    rng = np.random.default_rng(0)
    for bits in (2, 3, 4):
        trellis = rng.integers(-32768, 32767, size=(16, 8, 16 * bits), dtype=np.int16)
        suh = rng.uniform(0.5, 1.5, 256).astype(np.float16)
        svh = rng.uniform(0.001, 0.01, 128).astype(np.float16)
        want = E.dequant_np(2 * bits, "mul1", trellis, suh, svh)
        got = np.array(E.dequant_mx(bits, mx.array(trellis), mx.array(suh), mx.array(svh)))
        np.testing.assert_allclose(got, want, rtol=2e-6, atol=2e-6 * np.abs(want).max())
