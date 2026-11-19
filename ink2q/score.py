"""Teacher-forced scoring identical in definition to Baekpica's inkling2_eval.py (generated mask, KL, top-1, NLL)."""
import mlx.core as mx
import numpy as np

MSG_MODEL, END_MSG, END_SAMPLING = 200001, 200010, 200006
KEYS = ("kl", "top1", "nll_teacher", "nll")


def generated_mask(ids: np.ndarray, chat: bool) -> np.ndarray:
    t = len(ids)
    if not chat:
        return np.ones(t - 1, dtype=bool)
    inside = np.zeros(t, dtype=bool)
    on = False
    for j, tok in enumerate(ids.tolist()):
        if tok == MSG_MODEL:
            on = True
            continue
        if on:
            inside[j] = True
            if tok in (END_MSG, END_SAMPLING):
                on = tok == END_MSG and j + 1 < t and ids[j + 1] == MSG_MODEL
                if tok == END_SAMPLING:
                    on = False
    return inside[1:]


def head_logits(final: mx.array, head: mx.array, vocab_real: int = 200058, cap: float = 30.0) -> mx.array:
    z = (final @ head.T).astype(mx.float32)
    return (mx.tanh(z / cap) * cap)[:, :vocab_real]


def score_sequence(ids: np.ndarray, chat: bool, zt_fn, zv_fn, chunk: int = 1024, positions: bool = False) -> dict:
    n = len(ids) - 1
    gen = generated_mask(ids, chat)
    tgt = mx.array(ids[1:])
    acc = {k: 0.0 for k in KEYS} | {k + "_gen": 0.0 for k in KEYS}
    kls, tops = [], []
    for c0 in range(0, n, chunk):
        c1 = min(n, c0 + chunk)
        zt, zv = zt_fn(c0, c1), zv_fn(c0, c1)
        lt = zt - mx.logsumexp(zt, axis=-1, keepdims=True)
        lv = zv - mx.logsumexp(zv, axis=-1, keepdims=True)
        t = tgt[c0:c1, None]
        vals = {
            "kl": (mx.exp(lt) * (lt - lv)).sum(-1),
            "top1": (mx.argmax(zt, -1) == mx.argmax(zv, -1)).astype(mx.float32),
            "nll_teacher": -mx.take_along_axis(lt, t, -1)[:, 0],
            "nll": -mx.take_along_axis(lv, t, -1)[:, 0],
        }
        g = mx.array(gen[c0:c1].astype(np.float32))
        for k, v in vals.items():
            acc[k] += float(v.sum())
            acc[k + "_gen"] += float((v * g).sum())
        if positions:
            kls.append(np.array(vals["kl"]))
            tops.append(np.array(vals["top1"]) > 0.5)
    out = {"positions": n, "generated": int(gen.sum()), **acc}
    if positions:
        out["pos"] = {"kl": np.concatenate(kls).astype(np.float32), "top1": np.concatenate(tops).astype(bool), "gen": gen}
    return out


AGENTIC_DOMAINS = {"agent_tool", "agent_swe", "ds4_prompts"}


def slice_of(meta: dict) -> str:
    if meta.get("source") == "session" or meta.get("domain") in AGENTIC_DOMAINS:
        return "agentic"
    return "raw" if meta.get("format", "raw") == "raw" else "chat"


def summarize(rows: list[dict]) -> dict:
    n = sum(r["positions"] for r in rows)
    g = sum(r["generated"] for r in rows)
    out = {"positions": n, "generated": g}
    out |= {k: sum(r[k] for r in rows) / max(n, 1) for k in KEYS}
    if g:
        out |= {k + "_gen": sum(r[k + "_gen"] for r in rows) / g for k in KEYS}
    return out
