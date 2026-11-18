"""G1a: per-layer hidden states of the MLX streamed forward against Baekpica's torch reference (CPU), real weights.

Two valid bf16 implementations do not agree bit for bit: a last-bit difference in a layer's MoE input flips the
expert set of a few percent of tokens, and those tokens dominate the RMS difference. The gate is therefore
relative to a measured noise floor: the reference against itself with its short convolutions run in bf16 (as the
Transformers 5.23.0 fallback does) instead of FP32. Each layer passes when
rel_rms(MLX, reference) <= MAX_RATIO * rel_rms(reference bf16-conv, reference).
"""
import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor/ds4-dfm-rs/tools/vq_calib"))
import inkling2_ref as R  # noqa: E402

from ink2q import model as M  # noqa: E402
from ink2q import stream as S  # noqa: E402
from ink2q.source import SourceIndex  # noqa: E402
from ink2q.tokens import load_manifest, load_tokens  # noqa: E402


MAX_RATIO = 1.5
_short_conv_fp32 = R.short_conv


def _short_conv_bf16(x, w):
    t, c = x.shape
    y = torch.nn.functional.conv1d(x.to(w.dtype).t().unsqueeze(0), w.reshape(c, 1, -1),
                                   padding=w.shape[-1] - 1, groups=c)[0, :, :t].t()
    return ((y.float() + x.float()) / (w.shape[-1] + 1) ** 0.5).to(x.dtype)


def rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).mean() / (b**2).mean()))


def reference(source: Path, ids: np.ndarray, layers: int, conv_bf16: bool) -> list[np.ndarray]:
    R.short_conv = _short_conv_bf16 if conv_bf16 else _short_conv_fp32
    src = R.Source(source)
    glob = R.Globals(src, "cpu", head=False)
    h = glob.embed_ids(torch.tensor(ids))
    blocks, pos, out = [], torch.arange(len(ids)), []
    for l in range(layers):
        lw = R.LayerWeights(l, src.read(src.layer_names(l)), "cpu")
        resid, mb = R.attn_part(lw, h, blocks, pos, "bf16")
        y, _ = R.moe(lw, R.mlp_in(lw, resid, mb))
        new = R.finish_layer(lw, resid, y)
        if R.is_boundary(l):
            blocks = blocks + [h]
        h = new
        out.append(h.float().numpy())
        del lw
        print(json.dumps({"reference_layer": l, "conv_bf16": conv_bf16}), flush=True)
    R.short_conv = _short_conv_fp32
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/calib"))
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--case", default="en_web")
    ap.add_argument("--out", type=Path, default=Path("reports/g1a.json"))
    a = ap.parse_args()
    seqs = load_tokens(a.data / "heldout.safetensors")
    manifest = load_manifest(a.data / "heldout_manifest.json")
    i = next(k for k, m in enumerate(manifest) if m.get("case") == a.case)
    ids = seqs[i][: a.tokens]

    ref32 = reference(a.source, ids, a.layers, conv_bf16=False)   # Baekpica's reference as published
    ref16 = reference(a.source, ids, a.layers, conv_bf16=True)    # same, convolution as Transformers 5.23.0

    idx = SourceIndex(a.source)
    m = M.build(idx.config())
    M.load_globals(m, idx, final=False)
    st = S.embed(m, mx.array(ids))
    rows = []
    for l in range(a.layers):
        M.load_layer(m, idx, l)
        st = S.step(m, l, st)
        mx.eval(st.h, st.blocks)
        M.drop_layer(m, l)
        ours = np.array(st.h[0].astype(mx.float32))
        r32, r16 = ref32[l], ref16[l]
        floor = rel_rms(r16, r32)
        rows.append({"layer": l, "rel_rms": rel_rms(ours, r32), "noise_floor": floor,
                     "ratio": rel_rms(ours, r32) / floor, "max_abs": float(np.abs(ours - r32).max())})
        print(json.dumps(rows[-1]), flush=True)
    res = {"case": a.case, "tokens": len(ids), "layers": rows, "max_ratio": MAX_RATIO,
           "pass": all(r["ratio"] <= MAX_RATIO for r in rows)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"G1a_pass": res["pass"]}))
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
