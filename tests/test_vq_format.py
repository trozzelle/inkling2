import numpy as np

from ink2q.vq.format import decode, e4m3_table, pack, unpack


def test_e4m3_known_values():
    t = e4m3_table()
    assert t[0x38] == 1.0
    assert t[0xB8] == -1.0
    assert t[0x01] == 2.0**-9
    assert t[0x7E] == 448.0
    assert np.isnan(t[0x7F]) and np.isnan(t[0xFF])


def test_pack_roundtrip():
    rng = np.random.default_rng(0)
    i1 = rng.integers(0, 8192, size=(3, 5, 32), dtype=np.uint16)
    i2 = rng.integers(0, 256, size=(3, 5, 32), dtype=np.uint8)
    lo12, hi1, j2 = pack(i1, i2)
    assert lo12.shape == (3, 5, 48) and hi1.shape == (3, 5, 4) and j2.shape == (3, 5, 32)
    u1, u2 = unpack(lo12, hi1, j2)
    np.testing.assert_array_equal(u1, i1)
    np.testing.assert_array_equal(u2, i2)


def test_pack12_byte_layout_matches_baekpica():
    i1 = np.array([[0xABC, 0x123] + [0] * 14], dtype=np.uint16)
    lo12, _, _ = pack(i1, np.zeros_like(i1, dtype=np.uint8))
    assert list(lo12[0, :3]) == [0xBC, 0x3A, 0x12]


def test_decode_one_group():
    t = e4m3_table()
    c1 = np.zeros((8192, 8), np.uint8)
    c1[4097] = 0x38  # 1.0 in every dimension
    c2 = np.zeros((256, 8), np.uint8)
    c2[3, 0] = 0x30  # 0.5 in dimension 0
    i1 = np.zeros((1, 16), np.uint16)
    i1[0, 0] = 4097
    i2 = np.zeros((1, 16), np.uint8)
    i2[0, 0] = 3
    lo12, hi1, j2 = pack(i1, i2)
    w = decode(lo12, hi1, j2, np.array([2.0], np.float16), c1, c2)
    assert w.shape == (1, 128)
    assert w[0, 0] == 2.0 * (1.0 + t[0x30])
    np.testing.assert_array_equal(w[0, 1:8], 2.0)
    np.testing.assert_array_equal(w[0, 8:], 0.0)
