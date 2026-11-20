import json

import mlx.core as mx
import numpy as np

from ink2q import tfio
from ink2q.tokens import load_manifest, load_tokens, write_manifest, write_tokens


def test_bf16_bits_round_trip_exactly():
    x = mx.array([1.0, -2.5, 3.140625, 1e-3, 65280.0], dtype=mx.bfloat16)
    bits = np.array(x.view(mx.uint16))
    assert np.array_equal(tfio.bf16_bits_to_f32(bits), np.array(x.astype(mx.float32)))


def test_export_tokens_writes_one_json_list_a_sequence(tmp_path):
    seqs = [np.arange(10), np.arange(5) + 100]
    write_tokens(tmp_path / "t.safetensors", seqs)
    n = tfio.export_tokens(tmp_path / "t.safetensors", tmp_path / "ids", max_tokens=8)
    assert n == 2
    lines = (tmp_path / "ids" / "index.txt").read_text().split()
    assert json.loads(open(lines[0]).read()) == list(range(8))
    assert json.loads(open(lines[1]).read()) == [100, 101, 102, 103, 104]


def test_import_finals_reads_the_tools_bf16_rows(tmp_path):
    rows = mx.random.normal((7, 16)).astype(mx.bfloat16)
    (tmp_path / "bin").mkdir()
    np.array(rows.view(mx.uint16)).astype("<u2").tofile(tmp_path / "bin" / "seq-00000.bin")
    assert tfio.import_finals(tmp_path / "bin", tmp_path / "out", hidden=16) == 1
    got = mx.load(str(tmp_path / "out" / "seq-00000.safetensors"))["final"]
    assert got.dtype == mx.bfloat16 and mx.array_equal(got, rows)


def test_subset_slices_tokens_manifest_and_named_finals(tmp_path):
    seqs = [np.arange(10), np.arange(4)]
    write_tokens(tmp_path / "t.safetensors", seqs)
    write_manifest(tmp_path / "m.json", [{"case": "a", "n_tokens": 10}, {"case": "b", "n_tokens": 4}])
    fin = tmp_path / "teacher"
    fin.mkdir()
    for i, s in enumerate(seqs):
        mx.save_safetensors(str(fin / f"seq-{i:05d}.safetensors"), {"final": mx.arange(len(s) * 2).reshape(len(s), 2)})
    tfio.make_subset(tmp_path / "t.safetensors", tmp_path / "m.json", 6, {"teacher": fin}, tmp_path / "sub")
    assert [len(s) for s in load_tokens(tmp_path / "sub" / "tokens.safetensors")] == [6, 4]
    assert load_manifest(tmp_path / "sub" / "manifest.json")[0]["n_tokens"] == 6
    got = mx.load(str(tmp_path / "sub" / "teacher" / "seq-00000.safetensors"))["final"]
    assert got.shape == (6, 2) and mx.array_equal(got, mx.arange(12).reshape(6, 2))


def test_verdict_passes_only_when_the_upper_bound_is_below_the_margin(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"slices": {"all": {"kl": {"diff": 0.0005, "lo": -0.001, "hi": 0.0015}}}}))
    assert tfio.verdict(p, 0.002)["pass"]
    p.write_text(json.dumps({"slices": {"all": {"kl": {"diff": 0.002, "lo": 0.001, "hi": 0.0031}}}}))
    assert not tfio.verdict(p, 0.002)["pass"]
