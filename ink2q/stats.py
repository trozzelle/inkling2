"""Paired cluster bootstrap over sequences for pooled per-position metrics."""
import numpy as np


def paired_bootstrap(a: list[dict], b: list[dict], key: str, gen: bool, n: int = 10000, seed: int = 0) -> dict:
    assert len(a) == len(b)
    k, d = (key + "_gen", "generated") if gen else (key, "positions")
    sa = np.array([r[k] for r in a], np.float64)
    sb = np.array([r[k] for r in b], np.float64)
    cnt = np.array([r[d] for r in a], np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    den = cnt[idx].sum(1)
    ma, mb = sa[idx].sum(1) / den, sb[idx].sum(1) / den
    diff = mb - ma
    lower_is_better = key in ("kl", "nll")
    better = diff < 0 if lower_is_better else diff > 0
    tot = cnt.sum()
    return {"a": sa.sum() / tot, "b": sb.sum() / tot, "diff": (sb.sum() - sa.sum()) / tot,
            "lo": float(np.percentile(diff, 2.5)), "hi": float(np.percentile(diff, 97.5)),
            "p_b_better": float(better.mean())}
