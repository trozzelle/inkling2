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
    lg = make_logits(rng, holds=10)
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


def test_gate_ignores_steps_after_a_divergent_proposal(dirs):
    tf, ref = dirs
    rng = np.random.default_rng(7)
    p = props_ref(rng=rng)
    pt = p.copy()
    pt[10, 2] = (pt[10, 2] + 1) % 50  # hold 0 diverges at step 2: steps 0..2 comparable
    lg = make_logits(rng)
    lt = lg.copy()
    lt[0, 3:] = to_bits(np.random.default_rng(9).standard_normal((4, V)).astype(np.float32) * 5)  # garbage after
    write_seq(tf, ref, 0, p, pt, lg, lt)
    r = tfdspark.compare(tf, ref, [0])
    assert r["logits"]["rel_rms"] == 0 and r["gate"]["rel_rms"]
    assert r["logits_all_steps"]["rel_rms"] > 0
    assert r["logits"]["n"] < r["logits_all_steps"]["n"]


def test_comparable_steps():
    a = np.arange(7)
    assert tfdspark.comparable_steps(a, a) == 7
    b = a.copy(); b[0] = 9
    assert tfdspark.comparable_steps(a, b) == 1
    b = a.copy(); b[4] = 9
    assert tfdspark.comparable_steps(a, b) == 5


def _write_accept(tmp_path, ref_props, tf_props, tokens, L):
    tf, ref, taps = tmp_path / "tf", tmp_path / "ref", tmp_path / "taps"
    for d in (tf, ref, taps):
        d.mkdir()
    np.save(ref / "seq-00000.props.npy", ref_props.astype(np.int32))
    tf_props.astype("<i4").tofile(tf / "seq-00000.props.bin")
    (taps / "seq-00000.tokens.json").write_text(json.dumps({"prompt": L, "tokens": tokens.tolist()}))
    return tf, ref, taps


def test_accept_identical_and_perturbed(tmp_path):
    L, T = 4, 60
    tokens = np.arange(T + 1) % 11
    props = np.full((T, 7), -1, np.int32)
    for p in range(L - 1, T):
        props[p] = [tokens[p + 2 + k] if p + 2 + k < len(tokens) else 0 for k in range(7)]
    meta = [{"seq": 0, "set": "bench"}]
    tf, ref, taps = _write_accept(tmp_path, props, props, tokens, L)
    r = tfdspark.accept(tf, ref, taps, meta, min_rows=0)
    assert r["pass"] and r["ref"]["all"]["tokens_per_cycle"][5] > 5.9
    assert r["tf"]["bench"]["per_position"]["accept_k"] == r["ref"]["bench"]["per_position"]["accept_k"]
    bad = props.copy()
    bad[L - 1 :, 1] = -7  # TF never accepts beyond one draft token
    (tmp_path / "tf" / "seq-00000.props.bin").write_bytes(bad.astype("<i4").tobytes())
    r = tfdspark.accept(tf, ref, taps, meta, min_rows=0)
    assert not r["pass"] and r["tokens_per_cycle_rel_diff"]["all"][5] < -0.01
    # a set below the row minimum is reported but not gated, and with nothing gated there is no pass
    r = tfdspark.accept(tf, ref, taps, meta)
    assert r["gated_sets"] == [] and not r["pass"] and r["tokens_per_cycle_rel_diff"]["bench"][5] < -0.01


def _write_replay(tf, taps, seq, accepted, probs_row, L=10, T=60):
    """One replayed sequence: from row L - 1 on every hold's first `accepted` proposals are the recorded next tokens, then a
    miss, each proposal with the probability in `probs_row`; -1 before the prompt's end. 44 scored rows at the defaults."""
    name = f"seq-{seq:05d}"
    tokens = list(range(100, 100 + T + 1))
    (taps / f"{name}.tokens.json").write_text(json.dumps({"prompt": L, "tokens": tokens}))
    props = np.full((T, 7), -1, np.int32)
    probs = np.full((T, 7), -1, np.float32)
    for p in range(L - 1, T):
        row = [tokens[p + 2 + k] if p + 2 + k < len(tokens) else -2 for k in range(7)]
        if accepted < 7:
            row[accepted] = -9
        props[p] = row
        probs[p] = probs_row
    props.astype("<i4").tofile(tf / f"{name}.props.bin")
    probs.astype("<f4").tofile(tf / f"{name}.probs.bin")


@pytest.fixture
def replay(tmp_path):
    tf, taps = tmp_path / "tf", tmp_path / "taps"
    tf.mkdir(), taps.mkdir()
    _write_replay(tf, taps, 0, 3, [0.95, 0.95, 0.95, 0.2, 0.2, 0.2, 0.2])   # bench: three land, the fourth is doubtful
    _write_replay(tf, taps, 1, 1, [0.9, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])      # chat: one lands, the second is doubtful
    _write_replay(tf, taps, 2, 7, [0.95] * 7)                              # agentic: all seven land
    meta = [{"seq": 0, "set": "bench"}, {"seq": 1, "set": "chat"}, {"seq": 2, "set": "agentic"}]
    (tmp_path / "prompts.json").write_text(json.dumps(meta))
    return tf, taps, meta, tmp_path / "prompts.json"


def test_depthsim_rates_each_set_and_the_gate_sets_together(replay):
    tf, taps, meta, _ = replay
    rep = tfdspark.depthsim(tf, taps, meta)
    assert list(rep["sets"]) == ["bench", "chat", "agentic", "gate", "all"]
    assert [rep["sets"][k]["rows"] for k in rep["sets"]] == [44, 44, 44, 88, 132]
    assert rep["sets"]["gate"]["sequences"] == [0, 1] and rep["sets"]["all"]["sequences"] == [0, 1, 2]
    g = rep["sets"]["gate"]
    # depth 3: bench rounds take 4 rows (11 rounds), chat rounds 2 (22): 33 rounds, 88 tokens, 3 drafts each
    assert g["fixed"]["3"]["rounds"] == 33 and g["fixed"]["3"]["tokens_per_round"] == pytest.approx(88 / 33)
    assert g["fixed"]["3"]["tok_s"] == pytest.approx(88000 / (33 * 31.1 + 5.95 * 99))
    # the floor verifies 3 drafts of bench's holds and 1 of chat's: 55 drafts for the same rounds and tokens
    assert g["floor"]["0.70"]["rounds"] == 33 and g["floor"]["0.70"]["drafts_per_round"] == pytest.approx(55 / 33)
    assert g["floor"]["0.70"]["tok_s"] == pytest.approx(88000 / (33 * 31.1 + 5.95 * 55))
    assert g["best_floor"]["tok_s"] >= g["floor"]["0.70"]["tok_s"] and g["best_floor"]["vs_best_fixed_pct"] > 0
    # agentic's seven land: the floor takes all seven where depth 7 does (6 rounds of 8 rows over 44)
    a = rep["sets"]["agentic"]
    assert a["floor"]["0.70"]["tok_s"] == pytest.approx(a["fixed"]["7"]["tok_s"]) and a["best_fixed"]["depth"] == 7
    assert rep["cost"] == {"verify_ms": 25.2, "row_ms": 5.95, "hold_ms": 5.9}


def test_depthsim_takes_the_cost_model_and_the_floors_it_is_given(replay):
    from ink2q import tfdepth
    tf, taps, meta, _ = replay
    rep = tfdspark.depthsim(tf, taps, meta, tfdepth.Cost(10.0, 1.0, 2.0), [0.4, 0.8])
    assert rep["thresholds"] == [0.4, 0.8] and list(rep["sets"]["gate"]["floor"]) == ["0.40", "0.80"]
    assert rep["sets"]["gate"]["fixed"]["3"]["tok_s"] == pytest.approx(88000 / (33 * 12 + 1.0 * 99))


def test_depthsim_refuses_a_replay_with_rows_missing_or_without_probabilities(replay):
    tf, taps, meta, _ = replay
    (tf / "seq-00001.probs.bin").write_bytes(np.full((59, 7), 0.9, "<f4").tobytes())
    with pytest.raises(ValueError, match="seq-00001.*59 probs rows, expected 60"):
        tfdspark.depthsim(tf, taps, meta)
    np.full((60, 7), -1, "<f4").tofile(tf / "seq-00001.probs.bin")   # a replay that never kept them
    with pytest.raises(ValueError, match="seq 1"):
        tfdspark.depthsim(tf, taps, meta)
    (tf / "seq-00001.probs.bin").write_bytes(b"\0" * 7)
    with pytest.raises(ValueError, match="multiple of 7"):
        tfdspark.depthsim(tf, taps, meta)
    (tf / "seq-00001.probs.bin").unlink()
    with pytest.raises(FileNotFoundError):
        tfdspark.depthsim(tf, taps, meta)


def test_depthsim_command_prints_the_table_and_writes_the_report(replay, tmp_path, monkeypatch, capsys):
    tf, taps, _, prompts = replay
    out = tmp_path / "depthsim.json"
    argv = ["tfdspark", "depthsim", "--tf", str(tf), "--taps", str(taps), "--prompts", str(prompts), "--verify-ms", "25.2",
            "--row-ms", "5.95", "--hold-ms", "5.9", "--thresholds", "0.5,0.7,0.9", "--out", str(out)]
    monkeypatch.setattr("sys.argv", argv)
    tfdspark.main()
    text = capsys.readouterr().out
    assert "fixed 1" in text and "fixed 7" in text and "floor 0.70" in text and "floor 0.30" not in text and "best floor" in text
    rep = json.loads(out.read_text())
    assert rep["thresholds"] == [0.5, 0.7, 0.9] and rep["sets"]["gate"]["rows"] == 88
    assert rep["sets"]["gate"]["floor"]["0.70"]["tok_s"] == pytest.approx(88000 / (33 * 31.1 + 5.95 * 55))
