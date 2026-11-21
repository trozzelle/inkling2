import numpy as np
import pytest

from ink2q import tfaccept, tfdepth
from ink2q.tfdepth import Cost, Tally, kept_fixed, kept_floor, walk


def test_kept_floor_verifies_the_first_draft_then_stops_at_the_first_one_under_the_floor():
    probs = np.array([
        [0.125, 0.875, 0.5, 0.99, 0.99, 0.99, 0.99],    # a draft under the floor ends it, though the rest are over
        [0.125, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],    # a first draft under the floor is still verified: all seven
        [0.99, 0.25, 0.99, 0.99, 0.99, 0.99, 0.99],     # the second under: the first alone
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],            # a draft exactly at the floor passes
        [0.5, 0.5, 0.5, 0.5, 0.484375, 0.5, 0.5],       # one just under it does not
        [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],     # none passes: still the first
    ], np.float32)
    assert kept_floor(probs, 0.75).tolist() == [2, 7, 1, 1, 1, 1]
    assert kept_floor(probs, 0.5).tolist() == [7, 7, 1, 7, 4, 1]
    assert kept_floor(probs, 0.5).dtype.kind == "i"


def test_kept_floor_compares_each_probability_as_the_engine_does_in_f64():
    # the replay's f32 0.7 is 0.69999999..: the lane core compares that value, widened, with its f64 floor 0.7
    probs = np.full((1, 7), 0.7, np.float32)
    assert kept_floor(probs, 0.7).tolist() == [1]
    assert kept_floor(probs, float(np.float32(0.7))).tolist() == [7]


def test_kept_floor_and_fixed_refuse_what_is_not_a_block_of_seven():
    with pytest.raises(ValueError):
        kept_floor(np.ones((3, 6), np.float32), 0.5)
    for d in (0, 8):
        with pytest.raises(ValueError):
            kept_fixed(4, d)


def test_walk_commits_the_accepted_drafts_it_verified_plus_one_and_moves_on_by_that_many_rows():
    a = np.array([2, 0, 0, 5, 0, 0, 0, 0, 0, 1, 0, 0])
    k = np.array([3, 9, 9, 2, 9, 9, 1, 1, 1, 4, 9, 1])
    t = walk(a, k)
    # rows 0 (takes 3), 3 (min(5, 2) + 1 = 3), 6, 7, 8 (one each), 9 (2), 11 (1)
    assert (t.cycles, t.tokens, t.drafts) == (7, 12, 3 + 2 + 1 + 1 + 1 + 4 + 1)
    with pytest.raises(ValueError):
        walk(a, k[:5])


def test_walk_at_a_fixed_depth_is_the_acceptance_module_simulation():
    a = np.random.default_rng(0).integers(0, 8, 500)
    for d in tfdepth.FIXED_DEPTHS:
        t = walk(a, kept_fixed(len(a), d))
        s = tfaccept.simulate(a, d)
        assert (t.cycles, t.tokens) == (s["cycles"], s["tokens"]) and t.drafts == d * t.cycles


def test_a_round_costs_the_window_the_hold_and_each_draft_verified():
    c = Cost()
    assert (c.verify_ms, c.row_ms, c.hold_ms) == (25.2, 5.95, 5.9)
    t = Tally(cycles=2, tokens=7, drafts=5)
    assert t.ms(c) == pytest.approx(2 * (25.2 + 5.9) + 5.95 * 5)
    assert t.tok_s(c) == pytest.approx(7000 / (2 * 31.1 + 29.75))
    assert t.ms(Cost(10, 1, 2)) == pytest.approx(2 * 12 + 5)
    assert np.isnan(Tally().tok_s(c))


def two_regions():
    """80 rows where every draft lands (all confident), then 100 where one does and the second is a coin toss (not confident)."""
    lengths = np.array([7] * 80 + [1] * 100)
    probs = np.array([[0.95] * 7] * 80 + [[0.9] + [0.2] * 6] * 100, np.float32)
    return lengths, probs


def test_a_floor_follows_the_rows_where_every_fixed_depth_compromises():
    seqs = {0: two_regions()}
    rep = tfdepth.simulate(seqs, {"gate": [0]}, Cost(), thresholds=(0.5, 0.7, 0.99))
    s = rep["sets"]["gate"]
    assert s["rows"] == 180 and s["sequences"] == [0]
    # depth 7: 10 rounds of 8 tokens in the first region, 50 of 2 in the second, 7 drafts each: 60 rounds, 420 drafts
    assert s["fixed"]["7"]["tok_s"] == pytest.approx(180000 / (60 * 31.1 + 5.95 * 420))
    assert s["fixed"]["7"]["rounds"] == 60 and s["fixed"]["7"]["drafts_per_round"] == pytest.approx(7)
    # depth 1: 40 rounds of 2 tokens, then 50: 90 rounds, one draft each
    assert s["fixed"]["1"]["tok_s"] == pytest.approx(180000 / (90 * 31.1 + 5.95 * 90))
    # the floor at 0.7 verifies all seven where they land and one where the second is not trusted: 60 rounds, 70 + 50 drafts
    f = s["floor"]["0.70"]
    assert f["tok_s"] == pytest.approx(180000 / (60 * 31.1 + 5.95 * 120))
    assert f["rounds"] == 60 and f["tokens_per_round"] == pytest.approx(3) and f["drafts_per_round"] == pytest.approx(2)
    # at 0.5 the same (0.2 stays under); at 0.99 nothing passes (0.95 is under it): one draft a round, as fixed depth 1
    assert s["floor"]["0.50"]["tok_s"] == pytest.approx(f["tok_s"])
    assert s["floor"]["0.99"]["tok_s"] == pytest.approx(s["fixed"]["1"]["tok_s"])
    # no fixed depth does as well, and the best of them is named with the floor's gain over it and over depth 4
    assert all(f["tok_s"] > v["tok_s"] for v in s["fixed"].values())
    assert s["best_floor"]["threshold"] in (0.5, 0.7) and s["best_floor"]["tok_s"] == pytest.approx(f["tok_s"])
    assert s["best_floor"]["vs_best_fixed_pct"] == pytest.approx(100 * (f["tok_s"] / s["best_fixed"]["tok_s"] - 1))
    assert s["best_floor"]["vs_fixed_4_pct"] == pytest.approx(100 * (f["tok_s"] / s["fixed"]["4"]["tok_s"] - 1))
    assert s["best_floor"]["vs_best_fixed_pct"] > 0


def test_sets_pool_their_sequences_rounds_and_skip_the_empty():
    seqs = {0: two_regions(), 1: two_regions(), 2: (np.array([3] * 40), np.array([[0.9] * 7] * 40, np.float32))}
    rep = tfdepth.simulate(seqs, {"a": [0], "b": [0, 1], "c": [2], "empty": []}, Cost(), thresholds=(0.7,))
    assert set(rep["sets"]) == {"a", "b", "c"}
    # two identical sequences rate as one
    for key in ("fixed", "floor"):
        for rule, v in rep["sets"]["a"][key].items():
            assert rep["sets"]["b"][key][rule]["tok_s"] == pytest.approx(v["tok_s"])
            assert rep["sets"]["b"][key][rule]["rounds"] == 2 * v["rounds"]
    assert rep["sets"]["b"]["rows"] == 360
    # a set pools rounds, not rates: tokens over the time of all of them
    both = tfdepth.simulate(seqs, {"x": [0, 2]}, Cost(), thresholds=(0.7,))["sets"]["x"]["fixed"]["4"]
    one, two = rep["sets"]["a"]["fixed"]["4"], rep["sets"]["c"]["fixed"]["4"]
    ms = lambda e: e["rounds"] * 31.1 + 5.95 * e["rounds"] * e["drafts_per_round"]
    tokens = lambda e: e["rounds"] * e["tokens_per_round"]
    assert both["tok_s"] == pytest.approx(1000 * (tokens(one) + tokens(two)) / (ms(one) + ms(two)))


def test_a_replay_without_probabilities_or_with_a_hold_missing_is_refused():
    lengths, probs = two_regions()
    bad = probs.copy()
    bad[3, 2] = -1
    with pytest.raises(ValueError, match="seq 0"):
        tfdepth.simulate({0: (lengths, bad)}, {"gate": [0]})
    with pytest.raises(ValueError, match="accepted lengths"):
        tfdepth.simulate({0: (lengths[:10], probs)}, {"gate": [0]})
    nan = probs.copy()
    nan[0, 0] = np.nan
    with pytest.raises(ValueError):
        tfdepth.simulate({0: (lengths, nan)}, {"gate": [0]})


def test_the_report_names_every_rule_and_the_best_of_each_kind():
    rep = tfdepth.simulate({0: two_regions()}, {"gate": [0], "all": [0]})
    text = tfdepth.format_report(rep)
    for d in range(1, 8):
        assert f"fixed {d}" in text
    for t in tfdepth.THRESHOLDS:
        assert f"floor {t:.2f}" in text
    assert "gate" in text and "best floor" in text and "best fixed depth" in text
    assert len(tfdepth.THRESHOLDS) == 14 and tfdepth.THRESHOLDS[0] == 0.3 and tfdepth.THRESHOLDS[-1] == 0.95
