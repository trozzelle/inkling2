import mlx.core as mx
import numpy as np

from ink2q import experts as X
from ink2q import harness as Hn
from ink2q import stream as S
from ink2q.source import SourceIndex
from tests.helpers.toy import CONFIG, checkpoint, create_model, quantize_experts_mxfp8, write_checkpoint


def setup(tmp_path):
    full = create_model()
    quantize_experts_mxfp8(full)
    write_checkpoint(checkpoint(full), tmp_path / "src", CONFIG)
    rng = np.random.default_rng(0)
    return full, [rng.integers(0, 120, size=(n,)) for n in (9, 14)]


def direct(full, seq, swaps):
    st = S.embed_lean(full, mx.array(seq))
    for l in range(CONFIG["text_config"]["num_hidden_layers"]):
        mod = full.model.language_model.layers[l]
        if l in swaps:
            with X.install(mod, swaps[l](mod)):
                st = S.step_lean(full, l, st)
        else:
            st = S.step_lean(full, l, st)
    return S.final_hidden_lean(full, st)


def test_variants_match_direct_forward_and_teacher(tmp_path):
    full, seqs = setup(tmp_path)
    rtn2 = lambda mod: X.rtn(mod, 2, 2, group_size=32)
    rtn4 = lambda mod: X.rtn(mod, 4, 4, group_size=32)
    vs = [Hn.Variant("a", {5: rtn2}), Hn.Variant("b", {1: rtn4, 6: rtn2}), Hn.Variant("same", {})]
    Hn.run(SourceIndex(tmp_path / "src"), seqs, vs, tmp_path / "out", budget_gib=1.0)
    for i, seq in enumerate(seqs):
        f = lambda name: mx.load(str(tmp_path / "out" / name / f"seq-{i:05d}.safetensors"))["final"]
        assert mx.array_equal(f("teacher"), direct(full, seq, {}))
        assert mx.array_equal(f("a"), direct(full, seq, {5: rtn2}))
        assert mx.array_equal(f("b"), direct(full, seq, {1: rtn4, 6: rtn2}))
        assert mx.array_equal(f("same"), f("teacher"))
        assert not mx.array_equal(f("a"), f("teacher"))


def test_source_variant_equals_teacher(tmp_path):
    full, seqs = setup(tmp_path)
    import json
    (tmp_path / "v.json").write_text(json.dumps([{"name": "null", "layers": {"0": {"kind": "source"}, "4": {"kind": "source"}}}]))
    Hn.run(SourceIndex(tmp_path / "src"), seqs, Hn.load_variants(tmp_path / "v.json"), tmp_path / "out", budget_gib=1.0)
    for i in range(len(seqs)):
        f = lambda name: mx.load(str(tmp_path / "out" / name / f"seq-{i:05d}.safetensors"))["final"]
        assert mx.array_equal(f("null"), f("teacher"))


def test_multiple_passes_give_same_results(tmp_path):
    full, seqs = setup(tmp_path)
    rtn2 = lambda mod: X.rtn(mod, 2, 2, group_size=32)
    vs = [Hn.Variant(f"v{l}", {l: rtn2}) for l in (0, 3, 7)]
    tiny = Hn.stream_bytes(seqs, CONFIG["text_config"]["hidden_size"], CONFIG["text_config"]["num_hidden_layers"],
                           CONFIG["text_config"]["n_layers_per_attn_res_block"])
    Hn.run(SourceIndex(tmp_path / "src"), seqs, vs, tmp_path / "out", budget_gib=2.5 * tiny / Hn.GIB)
    import json
    assert len(json.loads((tmp_path / "out" / "passes.json").read_text())["passes"]) == 3
    for i, seq in enumerate(seqs):
        for l in (0, 3, 7):
            got = mx.load(str(tmp_path / "out" / f"v{l}" / f"seq-{i:05d}.safetensors"))["final"]
            assert mx.array_equal(got, direct(full, seq, {l: rtn2}))


def test_plan_passes_respects_budget():
    vs = [Hn.Variant(str(i), {i: None}) for i in range(10)]
    passes = Hn.plan_passes(vs, bytes_per_stream=10, budget_bytes=35)
    assert all(len(p) + 1 <= 3 for p in passes)           # +1: the teacher stream
    assert sorted(v.name for p in passes for v in p) == sorted(v.name for v in vs)
