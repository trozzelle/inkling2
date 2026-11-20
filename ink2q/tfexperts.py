"""Per-projection reference for the routed experts of one layer on recorded MoE inputs: w13 outputs and w2 outputs
in FP64 from the EXL3 pack (dequantized in f32, exact to about 1e-7 for these fp16 products), the bf16 SwiGLU input
every implementation's w2 is fed, the largest GEMV sums before the output Hadamard (the fp16 range question), and
the comparison of M1's and Sushi's outputs against it."""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from ink2q.exl3 import dequant_mx
from ink2q.tfio import bf16_bits_to_f32
from ink2q.tftaps import load_seq

HD = 128


def fwht(x: np.ndarray) -> np.ndarray:
    """Blockwise normalized Sylvester Hadamard over the last axis (a multiple of 128), in f64."""
    y = x.reshape(*x.shape[:-1], -1, HD).astype(np.float64).copy()
    h = 1
    while h < HD:
        y = y.reshape(*y.shape[:-1], HD // (2 * h), 2, h)
        a, b = y[..., 0, :].copy(), y[..., 1, :].copy()
        y[..., 0, :], y[..., 1, :] = a + b, a - b
        y = y.reshape(*y.shape[:-3], HD)
        h *= 2
    return (y / np.sqrt(HD)).reshape(x.shape)


def pre_rotation(out: np.ndarray, svh: np.ndarray) -> np.ndarray:
    return fwht(out / svh)


def to_bf16(x: np.ndarray) -> np.ndarray:
    return np.array(mx.array(x.astype(np.float32)).astype(mx.bfloat16).astype(mx.float32)).astype(np.float64)


def capped_swiglu(gu: np.ndarray, gate_cap: float = 4.0, up_cap: float = 25.0) -> np.ndarray:
    g, u = gu[..., 0::2], gu[..., 1::2]
    return gate_cap * np.tanh(g / gate_cap) / (1 + np.exp(-g)) * up_cap * np.tanh(u / up_cap)


def load_expert(kortexa: Path, layer: int, e: int):
    ck = json.loads((kortexa / "checkpoint.json").read_text())
    batch = next(b for b in dict(ck["layers"])[str(layer)]["batches"] if b["start"] <= e < b["end"])
    f = mx.load(str(kortexa / batch["file"]))
    idx = e - batch["start"] if f"experts.{e - batch['start']}.w13.trellis" in f else e
    out = []
    for proj in ("w13", "w2"):
        t = f[f"experts.{idx}.{proj}.trellis"]
        suh, svh = f[f"experts.{idx}.{proj}.suh"], f[f"experts.{idx}.{proj}.svh"]
        w = dequant_mx(t.shape[-1] // 16, t, suh, svh)             # W [K, N] f32
        out += [np.array(w.T).astype(np.float64), np.array(svh.astype(mx.float32)).astype(np.float64)]
    return out                                                        # w13 [N,K], svh13, w2 [H,I], svh2


def reference(taps: Path, seq: int, layer: int, rows: int, row0: int, kortexa: Path, out: Path) -> dict:
    s = load_seq(taps, seq)
    slot = {0: 0, 30: 1}[layer]
    x = bf16_bits_to_f32(s["xm"][row0 : row0 + rows, slot]).astype(np.float64)
    eids = s["eids"][row0 : row0 + rows, layer]
    S = rows * 8
    gu = np.zeros((S, 4608))
    y = np.zeros((S, 4608))
    act = np.zeros((S, 2304))
    pre13, pre2 = 0.0, 0.0
    cache = {}
    for r in range(rows):
        for k in range(8):
            e = int(eids[r, k])
            if e not in cache:
                cache[e] = load_expert(kortexa, layer, e)
            w13, svh13, w2, svh2 = cache[e]
            j = r * 8 + k
            gu[j] = w13 @ x[r]
            act[j] = to_bf16(capped_swiglu(to_bf16(gu[j])))
            y[j] = w2 @ act[j]
            pre13 = max(pre13, float(np.abs(pre_rotation(gu[j], svh13)).max()))
            pre2 = max(pre2, float(np.abs(pre_rotation(y[j], svh2)).max()))
    gu.tofile(f"{out}.gu.f64")
    y.tofile(f"{out}.y.f64")
    np.array(mx.array(act.astype(np.float32)).astype(mx.bfloat16).view(mx.uint16)).tofile(f"{out}.act.bf16")
    res = {"layer": layer, "rows": rows, "slots": S, "experts": len(cache), "w13_pre_max": pre13, "w2_pre_max": pre2,
           "fp16_max": 65504.0}
    Path(f"{out}.range.json").write_text(json.dumps(res) + "\n")
    return res


def errors(got: np.ndarray, want: np.ndarray) -> dict:
    d = got - want
    return {"rel_rms": float(np.sqrt((d ** 2).mean() / (want ** 2).mean())),
            "max_rel": float((np.abs(d) / np.maximum(np.abs(want), np.abs(want).mean())).max())}


def compare(ref: str, m1: str, sushi: str, kortexa: Path, layer: int, taps: Path, seq: int, rows: int, row0: int) -> dict:
    S = rows * 8
    gu = np.fromfile(f"{ref}.gu.f64").reshape(S, 4608)
    y = np.fromfile(f"{ref}.y.f64").reshape(S, 4608)
    res = {"m1": {"w13": errors(bf16_bits_to_f32(np.fromfile(f"{m1}.gu.bin", "<u2")).reshape(S, 4608), gu),
                  "w2": errors(bf16_bits_to_f32(np.fromfile(f"{m1}.y.bin", "<u2")).reshape(S, 4608), y)}}
    if sushi:
        eids = load_seq(taps, seq)["eids"][row0 : row0 + rows, layer].reshape(-1)
        g = np.fromfile(f"{sushi}.g.f32", np.float32).reshape(S, 2, 2304).astype(np.float64).sum(1)
        u = np.fromfile(f"{sushi}.u.f32", np.float32).reshape(S, 2, 2304).astype(np.float64).sum(1)
        out = np.zeros((S, 4608))
        for j, e in enumerate(eids):
            svh13 = load_expert(kortexa, layer, int(e))[1]
            out[j] = np.concatenate([fwht(g[j]), fwht(u[j])]) * svh13
        res["sushi"] = {"w13": errors(out, gu),
                        "w2_proxy": errors(np.fromfile(f"{sushi}.y.f16", np.float16).reshape(S, 4608).astype(np.float64), y)}
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("ref", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--taps", type=Path, required=True)
        p.add_argument("--seq", type=int, required=True)
        p.add_argument("--layer", type=int, required=True)
        p.add_argument("--rows", type=int, default=64)
        p.add_argument("--row0", type=int, required=True)
        p.add_argument("--kortexa", type=Path, required=True)
    sub.choices["ref"].add_argument("--out", required=True)
    for k in ("--ref", "--m1"):
        sub.choices["compare"].add_argument(k, required=True)
    sub.choices["compare"].add_argument("--sushi", default="")
    a = ap.parse_args(argv)
    if a.cmd == "ref":
        print(json.dumps(reference(a.taps, a.seq, a.layer, a.rows, a.row0, a.kortexa, Path(a.out))))
    else:
        print(json.dumps(compare(a.ref, a.m1, a.sushi, a.kortexa, a.layer, a.taps, a.seq, a.rows, a.row0), indent=1))


if __name__ == "__main__":
    main()
