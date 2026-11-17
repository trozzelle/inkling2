import math

import mlx.core as mx
import numpy as np

from ink2q.score import generated_mask, score_sequence, summarize

SYS, MODEL, END, ENDS = 200002, 200001, 200010, 200006


def test_generated_mask_raw_is_all_true():
    assert generated_mask(np.array([5, 6, 7, 8]), chat=False).tolist() == [True, True, True]


def test_generated_mask_chat_matches_baekpica_rules():
    ids = np.array([SYS, 5, 6, MODEL, 7, 8, END, 9])
    assert generated_mask(ids, chat=True).tolist() == [False, False, False, True, True, True, False]
    # END followed by MODEL keeps the model turn open, but the MODEL token itself is not marked
    ids = np.array([SYS, MODEL, 7, END, MODEL, 8, ENDS, 9])
    assert generated_mask(ids, chat=True).tolist() == [False, True, True, False, True, True, False]


def test_score_identical_logits():
    ids = np.array([1, 2, 0, 3])
    z = mx.array(np.log(np.array([[0.1, 0.2, 0.3, 0.4]] * 3, np.float32)))
    row = score_sequence(ids, False, lambda a, b: z[a:b], lambda a, b: z[a:b], chunk=2)
    assert row["positions"] == 3 and row["generated"] == 3
    assert abs(row["kl"]) < 1e-6 and row["top1"] == 3
    expected = -(math.log(0.3) + math.log(0.1) + math.log(0.4))
    assert abs(row["nll"] - expected) < 1e-5 and abs(row["nll_teacher"] - expected) < 1e-5


def _lsm(z):
    z = z.astype(np.float64)
    m = z.max(-1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(-1, keepdims=True))


def test_score_differing_logits_on_chat_row():
    # chat row: SYS MODEL 5 END -> targets MODEL, 5, END; generated mask [F, T, T]
    ids = np.array([SYS, MODEL, 5, END])
    V = END + 1
    rng = np.random.default_rng(3)
    zt = rng.normal(size=(3, V)).astype(np.float32)
    zv = (zt + rng.normal(scale=0.5, size=(3, V))).astype(np.float32)
    zt[0, 7], zv[0, 8] = 40.0, 40.0          # top-1 disagrees at position 0 only
    zt[1:, 9] += 40.0
    zv[1:, 9] += 40.0
    row = score_sequence(ids, True, lambda a, b: mx.array(zt[a:b]), lambda a, b: mx.array(zv[a:b]))
    lt, lv = _lsm(zt), _lsm(zv)
    tgt = ids[1:]
    kl = (np.exp(lt) * (lt - lv)).sum(-1)    # KL(teacher || variant)
    nt = -lt[np.arange(3), tgt]
    nv = -lv[np.arange(3), tgt]
    g = np.array([0.0, 1.0, 1.0])
    assert row["generated"] == 2 and row["top1"] == 2 and row["top1_gen"] == 2
    for key, val in (("kl", kl), ("nll_teacher", nt), ("nll", nv)):
        assert abs(row[key] - val.sum()) < 1e-3 * max(1.0, abs(val.sum())), key
        assert abs(row[key + "_gen"] - (val * g).sum()) < 1e-3 * max(1.0, abs((val * g).sum())), key
    assert abs(row["nll"] - row["nll_teacher"]) > 1e-2


def test_summarize_means():
    rows = [{"positions": 2, "generated": 1, "kl": 0.2, "top1": 2, "nll_teacher": 1.0, "nll": 1.2,
             "kl_gen": 0.1, "top1_gen": 1, "nll_teacher_gen": 0.5, "nll_gen": 0.6}]
    s = summarize(rows)
    assert s["kl"] == 0.1 and s["top1"] == 1.0 and s["nll_gen"] == 0.6
