import json

import mlx.core as mx
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ink2q import assemble as AS  # noqa: E402
from ink2q import collect as C  # noqa: E402
from ink2q import experts as X  # noqa: E402
from ink2q import model as M  # noqa: E402
from ink2q import stream as S  # noqa: E402
from ink2q.quant import gptq as G  # noqa: E402
from ink2q.quant.hessian import expert_h  # noqa: E402
from ink2q.quant.layer import convert_layer  # noqa: E402
from ink2q.quant.pack import pack_affine  # noqa: E402
from ink2q.source import SourceIndex  # noqa: E402
from tests.helpers.toy import CONFIG, checkpoint, create_model, quantize_experts_mxfp8, write_checkpoint  # noqa: E402

CFG = {"kind": "raw", "shrink": 64.0, "power": 1.0, "damp": 0.01, "refit": 1, "group": 32, "batch": 2}


def setup(tmp_path):
    full = create_model()
    quantize_experts_mxfp8(full)
    write_checkpoint(checkpoint(full), tmp_path / "src", CONFIG)
    src = SourceIndex(tmp_path / "src")
    m = M.build(src.config())
    M.load_globals(m, src)
    for l in range(CONFIG["text_config"]["num_hidden_layers"]):
        M.load_layer(m, src, l)
    rng = np.random.default_rng(0)
    seqs = [rng.integers(0, 120, size=(n,)) for n in (24, 31, 18)]
    return m, [S.embed_lean(m, mx.array(s)) for s in seqs]


def test_chunked_stats_equal_one_update(tmp_path):
    m, states = setup(tmp_path)
    mod = m.model.language_model.layers[0]
    a = C.ExpertStats(mod, None, raw=True, normalized=False)
    b = C.ExpertStats(mod, None, raw=True, normalized=False)
    new = C.step_with_stats(m, 0, states, a, chunk_tokens=20)
    xs = []
    for st in states:
        with C.capture(mod) as sink:
            S.step_lean(m, 0, st)
            xs.append(sink[-1][0])
    b.add(mx.concatenate(xs))
    ha, hb = a.to_host(), b.to_host()
    np.testing.assert_array_equal(ha["count"], hb["count"])
    np.testing.assert_allclose(ha["xx_raw"], hb["xx_raw"], rtol=1e-5, atol=1e-4)
    assert all(mx.array_equal(x.h, S.step_lean(m, 0, st).h) for x, st in zip(new, states))


def test_convert_layer_matches_direct_gptq_and_round_trips(tmp_path):
    m, states = setup(tmp_path)
    mod = m.model.language_model.layers[2]
    for l in range(2):
        states = [S.step_lean(m, l, st) for st in states]
    st = C.ExpertStats(mod, None, raw=True, normalized=False)
    C.step_with_stats(m, 2, states, st)
    host = st.to_host(np.float32, release=True)
    assert not any(st.g.values())
    rec = convert_layer(mod, 2, host, {"w13": [3], "w2": [2, 4]}, CFG, tmp_path / "conv")
    assert set(rec["h_err"]) == {"w13_b3", "w2_b2", "w2_b4"}
    H = expert_h(torch.from_numpy(host["aa_raw"].astype(np.float64)), torch.from_numpy(host["count"]), None, 64.0, 1.0)
    W = torch.from_numpy(np.array(X.source_bf16(mod, "down", slice(None)).astype(mx.float32), np.float64))
    Q, Sc, Bi, _ = G.convert(W, H, 4, group=32, damp=0.01, refit_passes=1)
    words, _, _ = pack_affine(Q.numpy(), Sc.numpy(), Bi.numpy(), 4, 32)
    got = mx.load(str(AS.expert_file(tmp_path / "conv", 2, "w2", 4)))
    assert mx.array_equal(got["model.layers.2.mlp.experts.w2.weight"], words)
    AS.write_expert_index({"2": {"w13": 3, "w2": 4}}, tmp_path / "conv", tmp_path / "cand", group_size=32)
    spec = X.from_mlx_dir(tmp_path / "cand", 2)
    assert spec.gate_up.bits == 3 and spec.down.bits == 4
    assert mx.array_equal(spec.down.weight, words)
    q = json.loads((tmp_path / "cand" / "config.json").read_text())["quantization"]
    assert q["model.layers.2.mlp.experts.w13"] == {"bits": 3, "group_size": 32, "mode": "affine"}
