"""TensorFold run-tool files: token sets as one JSON id list a sequence, the tool's bf16 finals back as ink2q finals,
held-out subsets (the first N tokens of each sequence) with finals sliced to them, and the model gate's verdict."""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from ink2q.tokens import load_manifest, load_tokens, subset, write_manifest, write_tokens


def bf16_bits_to_f32(u16: np.ndarray) -> np.ndarray:
    return (np.asarray(u16).astype(np.uint32) << 16).view(np.float32)


def export_tokens(tokens: Path, out: Path, limit: int = 0, max_tokens: int = 0) -> int:
    seqs = load_tokens(tokens)
    seqs = seqs[:limit] if limit else seqs
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, s in enumerate(seqs):
        p = (out / f"seq-{i:05d}.json").resolve()
        p.write_text(json.dumps([int(t) for t in (s[:max_tokens] if max_tokens else s)]))
        paths.append(str(p))
    (out / "index.txt").write_text("\n".join(paths) + "\n")
    return len(seqs)


def import_finals(src: Path, out: Path, hidden: int) -> int:
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("seq-*.bin"))
    for p in files:
        u = np.fromfile(p, "<u2").reshape(-1, hidden)
        mx.save_safetensors(str(out / f"{p.stem}.safetensors"), {"final": mx.array(u).view(mx.bfloat16)})
    return len(files)


def make_subset(tokens: Path, manifest: Path, max_tokens: int, finals: dict[str, Path], out: Path) -> dict:
    seqs, man = subset(load_tokens(tokens), load_manifest(manifest), max_tokens)
    out.mkdir(parents=True, exist_ok=True)
    write_tokens(out / "tokens.safetensors", seqs)
    write_manifest(out / "manifest.json", man)
    for name, d in finals.items():
        od = out / name
        od.mkdir(exist_ok=True)
        for i, s in enumerate(seqs):
            f = mx.load(str(d / f"seq-{i:05d}.safetensors"))["final"][: len(s)]
            mx.save_safetensors(str(od / f"seq-{i:05d}.safetensors"), {"final": f})
    return {"sequences": len(seqs), "tokens": int(sum(len(s) for s in seqs)), "finals": sorted(finals)}


def verdict(compare_json: Path, max_hi: float) -> dict:
    kl = json.loads(Path(compare_json).read_text())["slices"]["all"]["kl"]
    return {"diff": kl["diff"], "lo": kl["lo"], "hi": kl["hi"], "max_hi": max_hi, "pass": kl["hi"] <= max_hi}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tokens")
    t.add_argument("--tokens", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--limit", type=int, default=0)
    t.add_argument("--max-tokens", type=int, default=0)
    f = sub.add_parser("finals")
    f.add_argument("--src", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--hidden", type=int, default=4608)
    s = sub.add_parser("subset")
    s.add_argument("--tokens", type=Path, required=True)
    s.add_argument("--manifest", type=Path, required=True)
    s.add_argument("--max-tokens", type=int, required=True)
    s.add_argument("--finals", nargs="*", default=[], help="NAME=DIR pairs")
    s.add_argument("--out", type=Path, required=True)
    v = sub.add_parser("verdict")
    v.add_argument("--compare", type=Path, required=True)
    v.add_argument("--max-hi", type=float, default=0.002)
    a = ap.parse_args(argv)
    if a.cmd == "tokens":
        print(json.dumps({"sequences": export_tokens(a.tokens, a.out, a.limit, a.max_tokens)}))
    elif a.cmd == "finals":
        print(json.dumps({"finals": import_finals(a.src, a.out, a.hidden)}))
    elif a.cmd == "subset":
        pairs = dict(p.split("=", 1) for p in a.finals)
        print(json.dumps(make_subset(a.tokens, a.manifest, a.max_tokens, {k: Path(v) for k, v in pairs.items()}, a.out)))
    else:
        res = verdict(a.compare, a.max_hi)
        print(json.dumps(res))
        raise SystemExit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
