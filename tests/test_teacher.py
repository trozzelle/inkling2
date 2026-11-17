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
