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


def test_summarize_means():
    rows = [{"positions": 2, "generated": 1, "kl": 0.2, "top1": 2, "nll_teacher": 1.0, "nll": 1.2,
             "kl_gen": 0.1, "top1_gen": 1, "nll_teacher_gen": 0.5, "nll_gen": 0.6}]
    s = summarize(rows)
    assert s["kl"] == 0.1 and s["top1"] == 1.0 and s["nll_gen"] == 0.6
