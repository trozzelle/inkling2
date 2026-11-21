"""Layer-0 bake-off — RTN vs oQe vs FP64 GPTQ-affine (H variants) on held-out MoE output error."""
import argparse
import itertools
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from ink2q import collect as C
from ink2q import experts as X
from ink2q import model as M
from ink2q import stream as S
from ink2q.memguard import apply_limits
from ink2q.quant import gptq as G
from ink2q.quant.hessian import expert_h
from ink2q.quant.pack import pack_affine
from ink2q.source import SourceIndex
from ink2q.tokens import load_tokens

SEED, TARGET = 20261119, 1_000_000
EXPERTS = list(range(0, 384, 6))
CHUNK = 65536          # tokens per statistics update: one Gram update per expert per chunk, not per sequence


def calib_subset(seqs, target=TARGET, seed=SEED):
    order = np.random.default_rng(seed).permutation(len(seqs))
    pick, n = [], 0
    for i in order:
        pick.append(int(i))
        n += len(seqs[i])
        if n >= target:
            break
    return sorted(pick)


def moe_inputs(m, L0, seqs):
    """Layer-0 MoE inputs of every sequence, concatenated [T, H] bf16 (layer 0 needs only the embedding)."""
    xs = []
    for s in seqs:
        with C.capture(L0) as sink:
            S.step_lean(m, 0, S.embed_lean(m, mx.array(s)))
            x = sink[-1][0]
            mx.eval(x)
        xs.append(x)
    return mx.concatenate(xs)


def to_t(a):                       # mlx -> torch float64 (CPU), exact through float32
    return torch.from_numpy(np.array(a.astype(mx.float32), np.float64))


def rel_out(orig, hats, held, which):
    """Σ_e ‖X_e Wᵀ − X_e Ŵᵀ‖² / Σ_e ‖X_e Wᵀ‖²; both matmuls on bf16-exact operands."""
    num = den = 0.0
    for e, xe, act in held:
        Xe = (xe if which == "gate_up" else act).astype(mx.float32)
        Y = Xe @ orig[e].astype(mx.float32).T
        Yh = Xe @ hats[e].astype(mx.float32).T
        num += float(((Y - Yh) ** 2).sum())
        den += float((Y ** 2).sum())
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("work/source-l0"))
    ap.add_argument("--oqe", type=Path, default=Path("work/oqe-l0"))
    ap.add_argument("--data", type=Path, default=Path("data/calib"))
    ap.add_argument("--out", type=Path, default=Path("reports/e2.json"))
    a = ap.parse_args()
    apply_limits(100, 4)
    t_start = time.monotonic()
    src = SourceIndex(a.source)
    m = M.build(src.config())
    M.load_globals(m, src, final=False)
    M.load_layer(m, src, 0)
    L0 = m.model.language_model.layers[0]
    cal = load_tokens(a.data / "calib.safetensors")
    pick = calib_subset(cal)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    (a.out.parent / "e2-calib-subset.json").write_text(json.dumps(
        {"seed": SEED, "target": TARGET, "indices": pick, "tokens": int(sum(len(cal[i]) for i in pick))}))

    st = C.ExpertStats(L0, EXPERTS, raw=True, normalized=True)
    xc = moe_inputs(m, L0, [cal[i] for i in pick])
    for c0 in range(0, xc.shape[0], CHUNK):
        st.add(xc[c0:c0 + CHUNK])
    del xc
    st.held(moe_inputs(m, L0, load_tokens(a.data / "heldout.safetensors")))
    host = st.to_host()
    st.drop_device()
    held = st.held_rows
    print(json.dumps({"stats_seconds": round(time.monotonic() - t_start, 1),
                      "count_min": int(host["count"].min()), "count_median": float(np.median(host["count"])),
                      "held_rows": int(sum(x.shape[0] for _, x, _ in held))}), flush=True)
    orig = {w: {e: X.source_bf16(L0, w, slice(e, e + 1))[0] for e in EXPERTS} for w in ("gate_up", "down")}
    mx.eval([v for d in orig.values() for v in d.values()])
    rows = []

    def record(method, hats, **extra):
        r = {"method": method, **extra,
             "rel_out_w13": rel_out(orig["gate_up"], hats["gate_up"], held, "gate_up"),
             "rel_out_w2": rel_out(orig["down"], hats["down"], held, "down"),
             "elapsed_s": round(time.monotonic() - t_start, 1)}
        rows.append(r)
        print(json.dumps(r), flush=True)

    def deq(words, s, b, bits, group=128):
        return mx.dequantize(words, s, b, group_size=group, bits=bits)

    # (a) RTN on the harness's grid
    for bits in (2, 3, 4):
        hats = {w: {e: deq(*mx.quantize(orig[w][e], group_size=128, bits=bits), bits) for e in EXPERTS} for w in orig}
        record("rtn", hats, bits=bits, bpw=bits + 0.25)
    # (b) oQe's own tensors (3/3 at layer 0)
    oq = X.from_mlx_dir(a.oqe, 0)
    hats = {w: {e: deq(mod.weight[e], mod.scales[e], mod.biases[e], mod.bits, mod.group_size) for e in EXPERTS}
            for w, mod in (("gate_up", oq.gate_up), ("down", oq.down))}
    record("oqe", hats, bits=oq.gate_up.bits, bits_w2=oq.down.bits, group=oq.gate_up.group_size,
           bpw=oq.gate_up.bits + 32 / oq.gate_up.group_size)
    del oq, hats

    # (c, d) FP64 GPTQ under each Hessian configuration
    count = torch.from_numpy(host["count"])

    def gptq_hats(kind, shrink, power, bits, damp, refit):
        hats, herr = {}, {}
        for w, key in (("gate_up", f"xx_{kind}"), ("down", f"aa_{kind}")):
            Sg = torch.from_numpy(host[key])
            prior = (Sg.sum(0) / count.sum().clamp_min(1)) if w == "gate_up" else None
            hats[w], errs = {}, []
            for b0 in range(0, len(EXPERTS), 8):
                es = EXPERTS[b0:b0 + 8]
                H = expert_h(Sg[b0:b0 + 8], count[b0:b0 + 8], prior, shrink, power)
                W = torch.stack([to_t(orig[w][e]) for e in es])
                Q, Sc, Bi, err = G.convert(W, H, bits, group=128, damp=damp, refit_passes=refit)
                words, s, b = pack_affine(Q.numpy(), Sc.numpy(), Bi.numpy(), bits, 128)
                for j, e in enumerate(es):
                    hats[w][e] = deq(words[j], s[j], b[j], bits)
                errs.append(err)
            herr[w] = float(torch.cat(errs).mean())
        return hats, herr

    def run(kind, shrink, power, bits, damp, refit):
        hats, herr = gptq_hats(kind, shrink, power, bits, damp, refit)
        record("gptq", hats, kind=kind, shrink=shrink, power=power, bits=bits, damp=damp, refit=refit,
               bpw=bits + 0.25, h_err_w13=herr["gate_up"], h_err_w2=herr["down"])

    def best_of(rows):
        g3 = [r for r in rows if r["method"] == "gptq" and r["bits"] == 3]
        return min(g3, key=lambda r: r["rel_out_w13"] + r["rel_out_w2"])

    def save(final):
        best = best_of(rows)
        a.out.write_text(json.dumps({"rows": rows, "partial": not final,
                                     "best": {k: best[k] for k in ("kind", "shrink", "power", "damp", "refit")}},
                                    indent=1) + "\n")

    # 1. every H configuration at 3 bits, where `best` is chosen
    for kind, shrink, power in itertools.product(("raw", "norm"), (1024.0, 4096.0, 16384.0), (1.0, 0.75)):
        run(kind, shrink, power, 3, 0.01, 1)
        save(False)
    # 2. damp and refit ablations on the best configuration; `best` is chosen again over all 3-bit rows
    b = best_of(rows)
    for damp, refit in ((0.01, 0), (0.0, 1), (0.1, 1)):
        run(b["kind"], b["shrink"], b["power"], 3, damp, refit)
        save(False)
    # 3. the final best configuration at 2 and 4 bits (gain over RTN and oQe per width)
    b = best_of(rows)
    for bits in (2, 4):
        run(b["kind"], b["shrink"], b["power"], bits, b["damp"], b["refit"])
        save(False)
    save(True)


if __name__ == "__main__":
    main()
