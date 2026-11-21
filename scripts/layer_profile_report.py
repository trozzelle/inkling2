"""Layer-profile report (markdown to stdout) from reports/w2: the fair baseline per set with CIs, KL against position,
served oQe against it, the per-layer profile table, the RTN linearity check, the NVFP4 yardstick and the layer 40-47
ranking."""
import json
import sys
from pathlib import Path

import numpy as np

from ink2q.score import slice_of
from ink2q.stats import bootstrap_ci

BINS = [0, 1024, 2048, 4096, 8192, 16384, 32768]


def spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def kl_by_position(npz: Path, rows: list[dict], domain: str | None) -> list[tuple[str, float, int]]:
    z = np.load(npz)
    acc = np.zeros(len(BINS) - 1)
    cnt = np.zeros(len(BINS) - 1)
    for r in rows:
        if domain and r.get("domain") != domain:
            continue
        kl = z[f"{r['i']}_kl"]
        for j in range(len(BINS) - 1):
            seg = kl[BINS[j]:BINS[j + 1]]
            acc[j] += seg.sum()
            cnt[j] += len(seg)
    return [(f"{BINS[j]}-{BINS[j + 1]}", acc[j] / cnt[j], int(cnt[j])) for j in range(len(BINS) - 1) if cnt[j]]


def main(w2: Path, oqe_config: Path) -> None:
    out = ["## Fair baseline, per-layer profile, NVFP4 yardstick", ""]
    out += ["### Fair baseline (oQe experts, source non-experts, ink2q forward)", "",
            "| Set | Slice | KL (95% CI) | top-1 (95% CI) | positions |", "|---|---|---|---|---|"]
    for s in ("heldout", "agentic", "long"):
        ev = json.loads((w2 / f"bprime-{s}.json").read_text())
        rows = ev["rows"]
        gen = all(r["generated"] for r in rows)
        groups = {"all": rows}
        for r in rows:
            groups.setdefault(slice_of(r), []).append(r)
        for name, rs in groups.items():
            g = gen and all(r["generated"] for r in rs)
            k, t = bootstrap_ci(rs, "kl", g), bootstrap_ci(rs, "top1", g)
            n = sum(r["generated" if g else "positions"] for r in rs)
            out.append(f"| {s} | {name}{' (gen)' if g else ''} | {k['mean']:.4f} ({k['lo']:.4f}–{k['hi']:.4f}) | "
                       f"{t['mean']:.4f} ({t['lo']:.4f}–{t['hi']:.4f}) | {n} |")
    out += ["", "KL against position (all positions):", "", "| Set | Positions | KL | n |", "|---|---|---|---|"]
    for s, dom in (("heldout", "long_en_doc"), ("long", None)):
        rows = json.loads((w2 / f"bprime-{s}.json").read_text())["rows"]
        for b, v, n in kl_by_position(w2 / f"bprime-{s}.npz", rows, dom):
            out.append(f"| {s}{'/' + dom if dom else ''} | {b} | {v:.4f} | {n} |")
    cmp = json.loads((w2 / "b-vs-bprime.json").read_text())["slices"]
    out += ["", "Served oQe (8-bit non-experts) against the fair baseline (source non-experts), held-out:", "",
            "| Slice | metric | served | fair | fair−served (95% CI) |", "|---|---|---|---|---|"]
    for name, d in cmp.items():
        for k in ("kl", "top1"):
            r = d[k]
            out.append(f"| {name} | {k} | {r['a']:.4f} | {r['b']:.4f} | {r['diff']:+.4f} ({r['lo']:+.4f}–{r['hi']:+.4f}) |")

    ev = w2 / "profile" / "eval"
    kl = {p.stem: json.loads(p.read_text())["summary"]["kl"] for p in ev.glob("*.json")}
    q = json.loads(oqe_config.read_text())["quantization"]
    bits = [(q[f"model.layers.{l}.mlp.experts.w13"]["bits"], q[f"model.layers.{l}.mlp.experts.w2"]["bits"])
            for l in range(48)]
    oq = np.array([kl[f"oqe_L{l}"] for l in range(48)])
    r2 = np.array([kl[f"rtn2_L{l}"] for l in range(48)])
    r4 = np.array([kl[f"rtn4_L{l}"] for l in range(48)])
    out += ["", "### Per-layer profile (held-out, 512 tokens per sequence, KL over all positions)", "",
            "| Layer | oQe bits (w13/w2) | KL oQe | KL RTN2 | KL RTN4 | RTN2/RTN4 | oQe rank |", "|---|---|---|---|---|---|---|"]
    rank = {l: i + 1 for i, l in enumerate(np.argsort(-oq))}
    for l in range(48):
        out.append(f"| {l} | {bits[l][0]}/{bits[l][1]} | {oq[l]:.5f} | {r2[l]:.5f} | {r4[l]:.5f} | {r2[l] / r4[l]:.1f} | {rank[l]} |")
    ratio = r2 / r4
    out += ["", f"- Sum of single-layer KLs: oQe {oq.sum():.4f}, RTN2 {r2.sum():.4f}, RTN4 {r4.sum():.4f}.",
            f"- RTN2/RTN4 KL ratio across layers: median {np.median(ratio):.1f}, IQR {np.percentile(ratio, 25):.1f}–"
            f"{np.percentile(ratio, 75):.1f} (weight-MSE ratio of RTN 2-bit to 4-bit grids is about 25).",
            f"- Spearman across layers: RTN2 vs RTN4 {spearman(r2, r4):.3f}; oQe vs RTN4 {spearman(oq, r4):.3f}; "
            f"oQe vs RTN2 {spearman(oq, r2):.3f}.",
            f"- NVFP4 on layers 0–39: KL {kl['nvfp4_0_39']:.5f}."]
    late = [l for l in range(40, 48) if 2 in bits[l]]
    out.append(f"- Layers 40–47 at 2 bits in oQe: {late}; their oQe sensitivity ranks: {[rank[l] for l in late]}; "
               f"RTN4 ranks: {[int(np.argsort(np.argsort(-r4))[l]) + 1 for l in late]}.")
    print("\n".join(out))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
