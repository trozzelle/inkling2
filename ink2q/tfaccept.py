"""Draft acceptance from proposals against the target's greedy tokens: accepted lengths per row, per-position rates,
and tokens per verify cycle when a greedy run drafts at a fixed depth."""
import numpy as np

BLOCK = 7


def accept_lengths(props: np.ndarray, tokens: np.ndarray, lo: int, hi: int) -> np.ndarray:
    out = np.zeros(hi - lo, np.int64)
    for j, p in enumerate(range(lo, hi)):
        n = 0
        while n < BLOCK and p + 2 + n < len(tokens) and props[p, n] == tokens[p + 2 + n]:
            n += 1
        out[j] = n
    return out


def per_position(a: np.ndarray, depth: int = BLOCK) -> dict:
    ge = [float((a >= k).mean()) for k in range(1, depth + 1)]
    cond = [ge[0]] + [ge[k] / ge[k - 1] if ge[k - 1] > 0 else 0.0 for k in range(1, depth)]
    return {"rows": int(len(a)), "accept_k": ge, "cond_k": cond, "mean_len": float(np.minimum(a, depth).mean())}


def simulate(a: np.ndarray, depth: int) -> dict:
    p, cycles, tokens = 0, 0, 0
    while p < len(a):
        take = int(min(a[p], depth)) + 1
        cycles += 1
        tokens += take
        p += take
    return {"depth": depth, "cycles": cycles, "tokens": tokens, "tokens_per_cycle": tokens / max(cycles, 1)}
