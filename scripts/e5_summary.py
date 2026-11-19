"""E5 summary: GPTQ gain g(b) = KL_gptq(b/b) / KL_rtn(b/b) and the w13 share of a layer's sensitivity, per layer and
as the median the Plan 3 allocator reads; also the RTN 3-bit check of the allocator's geometric interpolation."""
import json
import statistics
import sys
from pathlib import Path


def main(eval_dir: Path, layers: list[int], out: Path) -> dict:
    kl = {p.stem: json.loads(p.read_text())["summary"]["kl"] for p in eval_dir.glob("*.json")}
    per = {}
    for l in layers:
        g = {b: kl[f"gptq_{b}_{b}_L{l}"] / kl[f"rtn{b}_L{l}"] for b in (2, 3, 4)}
        d13 = kl[f"gptq_2_3_L{l}"] - kl[f"gptq_3_3_L{l}"]
        d2 = kl[f"gptq_3_2_L{l}"] - kl[f"gptq_3_3_L{l}"]
        geo3 = (kl[f"rtn2_L{l}"] * kl[f"rtn4_L{l}"]) ** 0.5
        per[l] = {"g": g, "w13_share": d13 / (d13 + d2) if d13 + d2 > 0 else None,
                  "rtn3_over_geometric": kl[f"rtn3_L{l}"] / geo3,
                  "kl": {k: v for k, v in kl.items() if k.endswith(f"_L{l}")}}
    gain = {b: statistics.median(per[l]["g"][b] for l in layers) for b in (2, 3, 4)}
    shares = [per[l]["w13_share"] for l in layers if per[l]["w13_share"] is not None]
    out.mkdir(parents=True, exist_ok=True)
    (out / "gain.json").write_text(json.dumps({"g": gain, "per_layer": {l: per[l]["g"] for l in layers}}, indent=1) + "\n")
    (out / "proj_ratio.json").write_text(json.dumps({"w13_share": statistics.median(shares),
                                                     "per_layer": {l: per[l]["w13_share"] for l in layers}}, indent=1) + "\n")
    res = {"gain": gain, "w13_share": statistics.median(shares), "per_layer": per}
    (out / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main(Path(sys.argv[1]), [int(x) for x in sys.argv[2].split(",")], Path(sys.argv[3]))
