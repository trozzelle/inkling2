import json

import numpy as np
import pytest

from ink2q import tfdspark

V = 64  # stand-in vocab: ids 56.. masked to -inf like the real 200058..


def to_bits(x: np.ndarray) -> np.ndarray:
    """f32 -> bf16 bits, round to nearest even."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def make_logits(rng, holds=16):
    x = (rng.standard_normal((holds, 7, V)) * 5).astype(np.float32)
    x[..., 56:] = -np.inf
    return to_bits(x)


def write_seq(tf, ref, seq, props_ref, props_tf, logits_ref, logits_tf):
    name = f"seq-{seq:05d}"
    np.save(ref / f"{name}.props.npy", props_ref.astype(np.int32))
    props_tf.astype("<i4").tofile(tf / f"{name}.props.bin")
    np.save(ref / f"{name}.logits.npy", logits_ref)
    pad = np.zeros((16 - len(logits_tf), 7, V), np.uint16)
    np.concatenate([logits_tf, pad]).astype("<u2").tofile(tf / f"{name}.logits.bin")


@pytest.fixture
def dirs(tmp_path):
    tf, ref = tmp_path / "tf", tmp_path / "ref"
    tf.mkdir(), ref.mkdir()
    return tf, ref


def props_ref(T=40, holds=(10, 30), rng=None):
    p = np.full((T, 7), -1, np.int32)
    p[holds[0]:holds[1]] = (rng or np.random.default_rng(0)).integers(0, 50, (holds[1] - holds[0], 7))
    return p


@pytest.fixture
def identical(dirs):
    tf, ref = dirs
    rng = np.random.default_rng(1)
    for seq in (0, 1):
        p, lg = props_ref(rng=rng), make_logits(rng)
        write_seq(tf, ref, seq, p, p.copy(), lg, lg.copy())
    return tf, ref


def test_parse_seqs_ranges_are_inclusive():
    assert tfdspark.parse_seqs("0..3") == [0, 1, 2, 3]
    assert tfdspark.parse_seqs("5") == [5]
    assert tfdspark.parse_seqs("0..1,7,9..10") == [0, 1, 7, 9, 10]


def test_identical_data_passes_with_exact_logits(identical):
    tf, ref = identical
    r = tfdspark.compare(tf, ref, [0, 1])
    assert r["pass"] and r["agreement"]["total"] == 1.0 and r["rows"] == 40
    assert r["logits"]["rel_rms"] == 0 and r["logits"]["max_abs"] == 0 and r["logits"]["exact_frac"] == 1.0
    assert r["logits"]["holds"] == 32 and r["logits"]["floor"] > 0
    assert r["agreement"]["conditional"] == [1.0] * 7 and r["agreement"]["mean_agreed_prefix"] == 7.0


def test_floor_is_bf16_rounding_scale():
    # x in [1, 2): ulp = 2^-7, rms error ulp/sqrt(12); rel to rms(x) for a constant 1.5
    bits = to_bits(np.full((7, 8), 1.5, np.float32))
    s = tfdspark.logits_stats(bits, bits)
    assert s["sum_floor2"] / s["sum_r2"] == pytest.approx((2.0 ** -7) ** 2 / 12 / 1.5 ** 2)
    # exact power of two: ulp 2^-7 * 2^k
    assert tfdspark.bf16_ulp_sq(np.array([4.0, -6.0, 0.0], np.float32)).tolist() == [(2.0 ** -5) ** 2, (2.0 ** -5) ** 2, 0.0]


def test_one_ulp_error_everywhere_fails_the_gate(identical):
    tf, ref = identical
    lg = np.load(ref / "seq-00000.logits.npy")
    r = tfdspark.bf16_to_f32(lg)
    fin = np.isfinite(r)
    bumped = lg.copy()
    bumped[fin] += 1
    bumped.astype("<u2").tofile(tf / "seq-00000.logits.bin")
    rep = tfdspark.compare(tf, ref, [0])
    assert rep["logits"]["rel_rms"] > 0 and rep["gate"]["rel_rms"] == (rep["logits"]["rel_rms"] <= 2 * rep["logits"]["floor"])
    # one ulp of every entry has rms ulp (not ulp/sqrt(12)): sqrt(12) times the floor, so the gate fails it
    assert rep["logits"]["rel_rms"] / rep["logits"]["floor"] == pytest.approx(12 ** 0.5, rel=0.2)
    assert not rep["pass"]


def test_half_ulp_noise_is_within_gate(identical):
    tf, ref = identical
    lg = np.load(ref / "seq-00000.logits.npy")
    rng = np.random.default_rng(5)
    f = tfdspark.bf16_to_f32(lg)
    noisy = f + (rng.uniform(-0.5, 0.5, f.shape) * np.sqrt(tfdspark.bf16_ulp_sq(np.where(np.isfinite(f), f, 0)))).astype(np.float32)
    noisy[~np.isfinite(f)] = -np.inf
    to_bits(noisy).astype("<u2").tofile(tf / "seq-00000.logits.bin")
    rep = tfdspark.compare(tf, ref, [0])
    assert 0 < rep["logits"]["rel_rms"] <= 2 * rep["logits"]["floor"] and rep["pass"]


def test_planted_disagreement_counts_per_position_and_conditionally(dirs):
    tf, ref = dirs
    rng = np.random.default_rng(2)
    p = props_ref(T=20, holds=(0, 10), rng=rng)
    q = p.copy()
    q[0:5, 2] += 1          # rows 0..4 miss at position 3 only
    q[5:7, 0] += 1          # rows 5,6 miss everywhere after the first token (shift all)
    q[5:7, 1:] += 1
    lg = make_logits(rng)
    write_seq(tf, ref, 0, p, q, lg, lg)
    r = tfdspark.compare(tf, ref, [0])["agreement"]
    assert r["total"] == pytest.approx((70 - 5 - 14) / 70)
    assert r["per_position"] == pytest.approx([8 / 10, 8 / 10, 3 / 10, 8 / 10, 8 / 10, 8 / 10, 8 / 10])
    assert r["first"] == pytest.approx(0.8)
    # given the first two agreed (8 rows), position 3 misses in 5 of them; given first three agreed (3 rows), the
    # remaining positions agree on all of them
    assert r["conditional"][1] == pytest.approx(1.0)
    assert r["conditional"][2] == pytest.approx(3 / 8)
    assert r["conditional"][3] == pytest.approx(1.0)
    assert r["mean_agreed_prefix"] == pytest.approx((5 * 2 + 3 * 7 + 2 * 0) / 10)
    assert not tfdspark.compare(tf, ref, [0])["pass"]


def test_minus_one_rows_are_ignored(dirs):
    tf, ref = dirs
    rng = np.random.default_rng(3)
    p = props_ref(T=40, holds=(10, 30), rng=rng)
    q = p.copy()
    q[:10] = -1
    q[30:] = -1
    lg = make_logits(rng)
    write_seq(tf, ref, 0, p, q, lg, lg)
    r = tfdspark.compare(tf, ref, [0])
    assert r["rows"] == 20 and r["agreement"]["total"] == 1.0 and r["pass"]


def test_hold_set_mismatch_fails_gate_but_agreement_ignores_those_rows(dirs):
    tf, ref = dirs
    rng = np.random.default_rng(4)
    p = props_ref(T=40, holds=(10, 30), rng=rng)
    q = p.copy()
    q[10:12] = -1           # tf skipped two holds the reference ran
    lg = make_logits(rng)
    write_seq(tf, ref, 0, p, q, lg, lg)
    r = tfdspark.compare(tf, ref, [0])
    assert r["rows"] == 18 and r["rows_ref_only"] == 2 and r["rows_tf_only"] == 0
    assert r["agreement"]["total"] == 1.0 and not r["gate"]["holds_match"] and not r["pass"]


def test_finiteness_mismatch_fails_mask_check(identical):
    tf, ref = identical
    lg = np.load(ref / "seq-00000.logits.npy")
    lg[3, 2, 60] = to_bits(np.array([1.0], np.float32))[0]   # a masked id left open on tf's side
    lg.astype("<u2").tofile(tf / "seq-00000.logits.bin")
    r = tfdspark.compare(tf, ref, [0])
    assert r["logits"]["nonfinite_mismatch"] == 1 and not r["gate"]["mask_match"] and not r["pass"]


def test_props_shape_mismatch_raises(identical):
    tf, ref = identical
    np.zeros((39, 7), "<i4").tofile(tf / "seq-00000.props.bin")
    with pytest.raises(ValueError, match="props shape mismatch"):
        tfdspark.compare(tf, ref, [0])
    np.zeros(40 * 7 + 3, "<i4").tofile(tf / "seq-00000.props.bin")
    with pytest.raises(ValueError, match="multiple of 7"):
        tfdspark.compare(tf, ref, [0])


def test_logits_shape_mismatch_raises(identical):
    tf, ref = identical
    np.zeros(15 * 7 * V, "<u2").tofile(tf / "seq-00000.logits.bin")
    with pytest.raises(ValueError, match="expected 16x7x64"):
        tfdspark.compare(tf, ref, [0])
    with pytest.raises(ValueError, match="logits shape mismatch"):
        tfdspark.logits_stats(np.zeros((7, 8), np.uint16), np.zeros((7, 9), np.uint16))


def test_only_first_reference_holds_are_compared(dirs):
    tf, ref = dirs
    rng = np.random.default_rng(6)
    p = props_ref(T=20, holds=(0, 3), rng=rng)
    lg_ref = make_logits(rng, holds=3)
    write_seq(tf, ref, 0, p, p, lg_ref, lg_ref)
    assert tfdspark.compare(tf, ref, [0])["logits"]["holds"] == 3


def test_cli_writes_report_and_exit_code(identical, capsys, monkeypatch, tmp_path):
    tf, ref = identical
    out = tmp_path / "r.json"
    monkeypatch.setattr("sys.argv", ["tfdspark", "compare", "--tf", str(tf), "--ref", str(ref), "--seqs", "0..1",
                                     "--out", str(out)])
    with pytest.raises(SystemExit) as e:
        tfdspark.main()
    assert e.value.code == 0
    assert json.loads(out.read_text())["pass"] is True and json.loads(capsys.readouterr().out)["pass"] is True
