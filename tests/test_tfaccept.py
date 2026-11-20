import json

import numpy as np

from ink2q import tfaccept, tftaps


def test_accept_lengths_count_leading_matches_only():
    tokens = np.arange(20)
    props = np.full((20, 7), -1)
    props[3] = tokens[5:12]                 # all 7 right
    props[4] = [6, 7, 99, 9, 10, 11, 12]    # 2 right, then a miss
    a = tfaccept.accept_lengths(props, tokens, 3, 5)
    assert a.tolist() == [7, 2]


def test_simulate_steps_by_accepted_plus_one():
    a = np.array([2, 0, 0, 5, 0, 0, 0, 0, 0, 1, 0, 0])
    r = tfaccept.simulate(a, depth=3)
    # p=0: take 3 -> p=3: take min(5,3)+1=4 -> p=7: take 1 -> p=8: take 1 -> p=9: take 2 -> p=11: take 1 -> end
    assert r["cycles"] == 6 and r["tokens"] == 12 and abs(r["tokens_per_cycle"] - 2.0) < 1e-9


def test_per_position_rates():
    a = np.array([0, 1, 2, 7])
    r = tfaccept.per_position(a, depth=7)
    assert r["accept_k"][0] == 0.75 and r["accept_k"][1] == 0.5 and r["cond_k"][1] == 2 / 3


def test_load_seq_round_trips(tmp_path):
    T, H = 3, tftaps.H
    (tmp_path / "seq-00000.tokens.json").write_text(json.dumps({"prompt": 2, "tokens": [1, 2, 3, 4]}))
    taps = np.arange(T * 8 * H, dtype=np.uint16).reshape(T, 8, H)
    eids = np.arange(T * 48 * 8, dtype=np.uint32).reshape(T, 48, 8) % 384
    xm = np.ones((T, 2, H), np.uint16)
    taps.tofile(tmp_path / "seq-00000.taps.bin")
    eids.tofile(tmp_path / "seq-00000.eids.bin")
    xm.tofile(tmp_path / "seq-00000.xm.bin")
    s = tftaps.load_seq(tmp_path, 0)
    assert s["prompt"] == 2 and np.array_equal(s["taps"], taps) and np.array_equal(s["eids"], eids)
