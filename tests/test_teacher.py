import json

import mlx.core as mx
import numpy as np
from safetensors.numpy import save_file

from ink2q import evaluate, teacher
from ink2q.source import SourceIndex
from tests.helpers.toy import CONFIG, checkpoint, create_model, quantize_experts_mxfp8, write_checkpoint


def test_teacher_writes_final_and_scores_itself(tmp_path):
    full = create_model()
    quantize_experts_mxfp8(full)
    write_checkpoint(checkpoint(full), tmp_path / "src", CONFIG)
    rng = np.random.default_rng(0)
    seqs = [rng.integers(0, 120, size=(n,)) for n in (12, 20)]
    teacher.run(SourceIndex(tmp_path / "src"), seqs, tmp_path / "t")
    assert (tmp_path / "t" / "done.json").exists()
    lines = (tmp_path / "t" / "layers.jsonl").read_text().splitlines()
    assert len(lines) == CONFIG["text_config"]["num_hidden_layers"]

    ids = np.concatenate(seqs).astype(np.int32)
    offsets = np.array([0, 12, 32], np.int64)
    save_file({"input_ids": ids, "offsets": offsets}, str(tmp_path / "tok.safetensors"))
    manifest = {"samples": [{"i": 0, "format": "raw"}, {"i": 1, "format": "raw"}]}
    (tmp_path / "m.json").write_text(json.dumps(manifest))
    res = evaluate.main_args([
        "--source", str(tmp_path / "src"), "--tokens", str(tmp_path / "tok.safetensors"),
        "--manifest", str(tmp_path / "m.json"), "--teacher", str(tmp_path / "t"),
        "--out", str(tmp_path / "e.json"), "--vocab-real", "120",
    ])
    s = res["summary"]
    assert s["positions"] == 30 and abs(s["kl"]) < 1e-6 and s["top1"] == 1.0
    full_logits = full(mx.array(seqs[0])[None])[0]
    lp = full_logits - mx.logsumexp(full_logits, axis=-1, keepdims=True)
    nll0 = -float(mx.take_along_axis(lp[:-1], mx.array(seqs[0][1:])[:, None], -1).sum())
    assert abs(res["rows"][0]["nll_teacher"] - nll0) < 1e-3

    # a perturbed variant must score worse than the teacher
    var = tmp_path / "v"
    var.mkdir()
    for i in range(2):
        f = mx.load(str(tmp_path / "t" / f"seq-{i:05d}.safetensors"))["final"]
        noisy = f + 0.5 * mx.random.normal(f.shape, key=mx.random.key(i)).astype(f.dtype)
        mx.save_safetensors(str(var / f"seq-{i:05d}.safetensors"), {"final": noisy})
    res_v = evaluate.main_args([
        "--source", str(tmp_path / "src"), "--tokens", str(tmp_path / "tok.safetensors"),
        "--manifest", str(tmp_path / "m.json"), "--teacher", str(tmp_path / "t"), "--variant", str(var),
        "--out", str(tmp_path / "ev.json"), "--vocab-real", "120",
    ])
    sv = res_v["summary"]
    assert sv["kl"] > 1e-3 and sv["top1"] < 1.0
    assert abs(sv["nll_teacher"] - s["nll_teacher"]) < 1e-6


def test_variant_run_equals_model_with_swapped_experts(tmp_path):
    from ink2q import experts as X
    from ink2q import stream as S
    full = create_model()
    quantize_experts_mxfp8(full)
    write_checkpoint(checkpoint(full), tmp_path / "src", CONFIG)
    seq = np.random.default_rng(3).integers(0, 120, size=(16,))
    teacher.run(SourceIndex(tmp_path / "src"), [seq], tmp_path / "v",
                variant=lambda mod, l: X.rtn(mod, 4, 4, group_size=32) if l == 3 else None)
    got = mx.load(str(tmp_path / "v" / "seq-00000.safetensors"))["final"]
    L3 = full.model.language_model.layers[3]
    with X.install(L3, X.rtn(L3, 4, 4, group_size=32)):
        st = S.embed(full, mx.array(seq))
        for l in range(CONFIG["text_config"]["num_hidden_layers"]):
            st = S.step(full, l, st)
        want = S.final_hidden(full, st)
    assert mx.array_equal(got, want)
    teacher.run(SourceIndex(tmp_path / "src"), [seq], tmp_path / "t")
    assert not mx.array_equal(got, mx.load(str(tmp_path / "t" / "seq-00000.safetensors"))["final"])


def test_run_sets_equals_separate_runs(tmp_path):
    full = create_model()
    quantize_experts_mxfp8(full)
    write_checkpoint(checkpoint(full), tmp_path / "src", CONFIG)
    rng = np.random.default_rng(5)
    a, b = [rng.integers(0, 120, size=(n,)) for n in (7, 11)], [rng.integers(0, 120, size=(9,))]
    src = SourceIndex(tmp_path / "src")
    teacher.run_sets(src, [(a, tmp_path / "A"), (b, tmp_path / "B")])
    teacher.run(src, b, tmp_path / "B1")
    f = lambda d, i: mx.load(str(tmp_path / d / f"seq-{i:05d}.safetensors"))["final"]
    assert mx.array_equal(f("B", 0), f("B1", 0))
    assert (tmp_path / "A" / "seq-00001.safetensors").exists() and not (tmp_path / "A" / "seq-00002.safetensors").exists()
