"""Convert one layer's routed experts with FP64 GPTQ from host statistics; write oMLX-layout files per width."""
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from ink2q import experts as X
from ink2q.assemble import expert_file
from ink2q.quant import gptq as G
from ink2q.quant.hessian import expert_h
from ink2q.quant.pack import pack_affine

PROJ = (("w13", "gate_up", "xx"), ("w2", "down", "aa"))


def convert_layer(mod, layer: int, host: dict, widths: dict, cfg: dict, out: Path) -> dict:
    """host: ExpertStats.to_host() over all experts; widths: {"w13": [bits...], "w2": [bits...]};
    cfg: kind, shrink, power, damp, refit, group, batch. Returns a receipt (h_err per width, counts, seconds)."""
    t0 = time.monotonic()
    receipt = {"layer": layer, "h_err": {}}
    n = torch.from_numpy(np.asarray(host["count"]))
    for w, which, key in PROJ:
        if not widths.get(w):
            continue
        Sg = host[f"{key}_{cfg['kind']}"]
        prior = None
        if w == "w13":   # gate/up experts share the hidden space: shrink toward the layer mean
            prior = torch.from_numpy(Sg.sum(0, dtype=np.float64)) / max(int(n.sum()), 1)
        acc = {b: {"Q": [], "S": [], "B": [], "err": []} for b in widths[w]}
        for e0 in range(0, Sg.shape[0], cfg["batch"]):
            sl = slice(e0, e0 + cfg["batch"])
            H = expert_h(torch.from_numpy(np.asarray(Sg[sl], np.float64)), n[sl], prior, cfg["shrink"], cfg["power"])
            W = torch.from_numpy(np.array(X.source_bf16(mod, which, sl).astype(mx.float32), np.float64))
            for b in widths[w]:
                Q, Sc, Bi, err = G.convert(W, H, b, group=cfg["group"], damp=cfg["damp"], refit_passes=cfg["refit"])
                a = acc[b]
                a["Q"].append(Q.numpy()); a["S"].append(Sc.numpy()); a["B"].append(Bi.numpy()); a["err"].append(err.numpy())
        for b, a in acc.items():
            words, s, bb = pack_affine(np.concatenate(a["Q"]), np.concatenate(a["S"]), np.concatenate(a["B"]), b,
                                       cfg["group"])
            f = expert_file(out, layer, w, b)
            f.parent.mkdir(parents=True, exist_ok=True)
            prefix = f"model.layers.{layer}.mlp.experts.{w}"
            tmp = f.with_name(f.stem + ".tmp.safetensors")
            mx.save_safetensors(str(tmp), {f"{prefix}.weight": words, f"{prefix}.scales": s, f"{prefix}.biases": bb})
            tmp.replace(f)
            err = np.concatenate(a["err"])
            receipt["h_err"][f"{w}_b{b}"] = {"mean": float(err.mean()), "p90": float(np.percentile(err, 90))}
    c = np.asarray(host["count"])
    receipt["counts"] = {"min": int(c.min()), "median": float(np.median(c)), "max": int(c.max()),
                         "unrouted": int((c == 0).sum())}
    receipt["convert_seconds"] = round(time.monotonic() - t0, 1)
    return receipt
