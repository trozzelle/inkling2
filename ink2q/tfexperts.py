"""Per-projection reference for the routed experts of one layer on recorded MoE inputs: w13 outputs and w2 outputs
in FP64 from the EXL3 pack (dequantized in f32, exact to about 1e-7 for these fp16 products), the bf16 SwiGLU input
every implementation's w2 is fed, the largest GEMV sums before the output Hadamard (the fp16 range question), and
the comparison of M1's, Sushi's and the decode chain's outputs against it."""
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
    """rel_rms: rms of the difference over rms of want. max_rel: the largest difference over max(|want|, mean |want|).
    The _big forms leave out the outputs under 1e-3 of want's rms: a sum that cancels to near zero is many bf16 steps
    off at no cost to the rest, so its steps say little."""
    d = got - want
    big = np.abs(want) >= 1e-3 * np.sqrt((want ** 2).mean())
    return {"rel_rms": float(np.sqrt((d ** 2).mean() / (want ** 2).mean())),
            "max_rel": float((np.abs(d) / np.maximum(np.abs(want), np.abs(want).mean())).max()),
            "rel_rms_big": float(np.sqrt((d[big] ** 2).mean() / (want[big] ** 2).mean())),
            "max_rel_big": float((np.abs(d[big]) / np.abs(want[big])).max())}


MOE_WEIGHT = 0.125


def moe_rows(ey: np.ndarray, rows: int) -> np.ndarray:
    """ik_combine's weighted sum of each row's eight per-pick outputs ([rows * 8, N]) at weight MOE_WEIGHT: the products
    added in f32 in pick order, one bf16 rounding (the product is exact at a power-of-two weight, so each add rounds once,
    as the kernel's fma does)."""
    e = ey.reshape(rows, 8, -1).astype(np.float32)
    acc = np.zeros((rows, e.shape[-1]), np.float32)
    for k in range(8):
        acc = acc + e[:, k] * np.float32(MOE_WEIGHT)
    return to_bf16(acc)


def compare(ref: str, m1: str, sushi: str, kortexa: Path, layer: int, taps: Path, seq: int, rows: int, row0: int,
            xd: str = "") -> dict:
    S = rows * 8
    gu = np.fromfile(f"{ref}.gu.f64").reshape(S, 4608)
    y = np.fromfile(f"{ref}.y.f64").reshape(S, 4608)
    m1_y = bf16_bits_to_f32(np.fromfile(f"{m1}.y.bin", "<u2")).reshape(S, 4608)
    res = {"bf16_floor": {"w13": errors(to_bf16(gu), gu), "w2": errors(to_bf16(y), y)},
           "m1": {"w13": errors(bf16_bits_to_f32(np.fromfile(f"{m1}.gu.bin", "<u2")).reshape(S, 4608), gu),
                  "w2": errors(m1_y, y)}}
    if sushi:
        eids = load_seq(taps, seq)["eids"][row0 : row0 + rows, layer].reshape(-1)
        g = np.fromfile(f"{sushi}.g.f32", np.float32).reshape(S, 2, 2304).astype(np.float64).sum(1)
        u = np.fromfile(f"{sushi}.u.f32", np.float32).reshape(S, 2, 2304).astype(np.float64).sum(1)
        out = np.zeros((S, 4608))
        for j, e in enumerate(eids):
            svh13 = load_expert(kortexa, layer, int(e))[1]
            out[j] = np.concatenate([fwht(g[j]), fwht(u[j])]) * svh13
        y2 = np.fromfile(f"{sushi}.y.f16", np.float16).reshape(S, 4608).astype(np.float64)
        res["sushi"] = {"w13": errors(out, gu), "w2_proxy": errors(y2, y)}
        # the same outputs at M1's output format: the excess over bf16_floor is what the sums' precision costs
        res["sushi_bf16"] = {"w13": errors(to_bf16(out), gu), "w2_proxy": errors(to_bf16(y2), y)}
    if xd:
        # The taps carry no router weights: every pick weighs 1/8, in the reference and in each path's weighted sum.
        moe = (y.reshape(rows, 8, -1) * MOE_WEIGHT).sum(1)
        read = lambda name, shape: bf16_bits_to_f32(np.fromfile(f"{xd}.{name}.bin", "<u2")).reshape(shape).astype(np.float64)
        res["bf16_floor"]["moe"] = errors(to_bf16(moe), moe)
        res["m1"]["moe"] = errors(moe_rows(m1_y, rows), moe)
        # w2 and moe on the act the reference gives every path (as m1's), then on each path's own act
        res["xd"] = {"w13": errors(read("gu", (S, 4608)), gu), "w2": errors(read("y", (S, 4608)), y),
                     "moe": errors(read("moe", (rows, 4608)), moe),
                     "w2_own_act": errors(read("y_own", (S, 4608)), y), "moe_own_act": errors(read("moe_own", (rows, 4608)), moe)}
        res["m1"]["w2_own_act"] = errors(read("proj_y_own", (S, 4608)), y)
        res["m1"]["moe_own_act"] = errors(read("proj_moe_own", (rows, 4608)), moe)
        parts = ("w13", "w2", "moe", "w2_own_act", "moe_own_act")
        floor = lambda k: res["bf16_floor"][k.removesuffix("_own_act")]["rel_rms"]
        res["xd_over_m1"] = {k: res["xd"][k]["rel_rms"] / res["m1"][k]["rel_rms"] for k in parts}
        res["over_floor"] = {n: {k: res[n][k]["rel_rms"] / floor(k) - 1 for k in parts} for n in ("m1", "xd")}
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
    sub.choices["compare"].add_argument("--xd", default="", help="prefix of the decode chain's `tf-inkling-bench xd-dump` files")
    a = ap.parse_args(argv)
    if a.cmd == "ref":
        print(json.dumps(reference(a.taps, a.seq, a.layer, a.rows, a.row0, a.kortexa, Path(a.out))))
    else:
        print(json.dumps(compare(a.ref, a.m1, a.sushi, a.kortexa, a.layer, a.taps, a.seq, a.rows, a.row0, a.xd), indent=1))


if __name__ == "__main__":
    main()
