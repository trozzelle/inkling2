"""G1a: per-layer hidden states of the MLX streamed forward against Baekpica's torch reference (CPU), real weights."""
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

    src = R.Source(a.source)
    glob = R.Globals(src, "cpu", head=False)
    h = glob.embed_ids(torch.tensor(ids))
    blocks, pos, ref = [], torch.arange(len(ids)), []
    for l in range(a.layers):
        lw = R.LayerWeights(l, src.read(src.layer_names(l)), "cpu")
        resid, mb = R.attn_part(lw, h, blocks, pos, "bf16")
        y, _ = R.moe(lw, R.mlp_in(lw, resid, mb))
        new = R.finish_layer(lw, resid, y)
        if R.is_boundary(l):
            blocks = blocks + [h]
        h = new
        ref.append(h.float().numpy())
        del lw
        print(json.dumps({"reference_layer": l}), flush=True)

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
        d = ours - ref[l]
        rows.append({"layer": l, "rel_rms": float(np.sqrt((d**2).mean() / (ref[l] ** 2).mean())),
                     "max_abs": float(np.abs(d).max())})
        print(json.dumps(rows[-1]), flush=True)
    res = {"case": a.case, "tokens": len(ids), "layers": rows, "pass": all(r["rel_rms"] <= 4e-3 for r in rows)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"G1a_pass": res["pass"]}))
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
