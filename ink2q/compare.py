"""Paired comparison of two evaluations of the same token set against the same teacher."""
import argparse
import json
from pathlib import Path

from ink2q.score import slice_of
from ink2q.stats import paired_bootstrap


def main_args(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=10000)
    x = ap.parse_args(argv)
    ra, rb = json.loads(x.a.read_text())["rows"], json.loads(x.b.read_text())["rows"]
    assert [r["i"] for r in ra] == [r["i"] for r in rb], "different token sets"
    groups = {"all": list(range(len(ra)))}
    for j, (r, s) in enumerate(zip(ra, rb)):        # served-path rows carry no format; use the side that does
        groups.setdefault(slice_of(r if "format" in r else s), []).append(j)
    out = {"a": str(x.a), "b": str(x.b), "slices": {}}
    for name, js in groups.items():
        sel_a, sel_b = [ra[j] for j in js], [rb[j] for j in js]
        gen = all(r["generated"] for r in sel_a)
        out["slices"][name] = {k: paired_bootstrap(sel_a, sel_b, k, gen=gen, n=x.n) for k in ("kl", "top1")}
    x.out.parent.mkdir(parents=True, exist_ok=True)
    x.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main_args()
