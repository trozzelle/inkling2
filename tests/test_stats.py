import numpy as np

from ink2q.score import slice_of
from ink2q.stats import bootstrap_ci, paired_bootstrap


def rows(vals, gen=10):
    return [{"positions": gen, "generated": gen, "kl": v * gen, "kl_gen": v * gen} for v in vals]


def test_slice_of():
    assert slice_of({"format": "raw", "domain": "web_en"}) == "raw"
    assert slice_of({"format": "chat", "domain": "chat_ko"}) == "chat"
    assert slice_of({"format": "chat", "domain": "agent_swe"}) == "agentic"
    assert slice_of({"format": "chat", "domain": "x", "source": "session"}) == "agentic"


def test_bootstrap_detects_consistent_improvement():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.02, 0.08, size=40)
    r = paired_bootstrap(rows(base), rows(base * 0.8), "kl", gen=True, n=2000)
    assert r["diff"] < 0 and r["hi"] < 0 and r["p_b_better"] > 0.99


def test_bootstrap_no_difference_straddles_zero():
    rng = np.random.default_rng(1)
    a = rng.uniform(0.02, 0.08, size=40)
    b = a + rng.normal(0, 0.01, size=40)
    r = paired_bootstrap(rows(a), rows(b), "kl", gen=True, n=2000)
    assert r["lo"] < 0 < r["hi"]


def test_bootstrap_ci_brackets_the_pooled_mean():
    rng = np.random.default_rng(2)
    r = bootstrap_ci(rows(rng.uniform(0.02, 0.08, size=40)), "kl", gen=True, n=2000)
    assert r["lo"] < r["mean"] < r["hi"] and r["hi"] - r["lo"] < 0.02
