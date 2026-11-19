"""Session evaluation sets from the eval split (by session group): agentic windows and long-context renders."""
import argparse
import hashlib
import json
import socket
from pathlib import Path

import numpy as np

from ink2q.data import render as R
from ink2q.data import sessions as SS
from ink2q.score import generated_mask
from ink2q.tokens import write_manifest, write_tokens


def eval_files(dirs: list[Path]) -> list[tuple[str, Path]]:
    """(group, path) for every session file in the eval split, ordered by sha256 of group then path."""
    out = []
    for d in dirs:
        for p in sorted(Path(d).expanduser().rglob("*.jsonl")):
            g = SS.session_group(p)
            if R.split(g) == "eval":
                out.append((g, p))
    return sorted(out, key=lambda gp: (hashlib.sha256(gp[0].encode()).hexdigest(), str(gp[1])))


def read(path: Path, host: str):
    claude = ".claude" in path.parts
    try:
        return SS.read_claude(path, host) if claude else SS.read_omp(path, host)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=Path, nargs="+", required=True)
    ap.add_argument("--model", type=Path, default=Path("data/model-files"))
    ap.add_argument("--out", type=Path, default=Path("data/sessions"))
    ap.add_argument("--agentic-gen", type=int, default=50_000)
    ap.add_argument("--agentic-len", type=int, default=8192)
    ap.add_argument("--per-group", type=int, default=2)
    ap.add_argument("--long-ctx", type=int, default=32768)
    ap.add_argument("--long-n", type=int, default=4)
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(a.model))
    defang = R.Defang(a.model / "tokenizer.json")
    host = socket.gethostname().split(".")[0]
    files = eval_files(a.sessions)
    groups = sorted({g for g, _ in files})
    agentic, a_man, gen_total = [], [], 0
    long, l_man, long_groups, agentic_groups = [], [], set(), set()
    for g, p in files:
        if gen_total >= a.agentic_gen and len(long) >= a.long_n:
            break
        conv = read(p, host)
        if conv is None:
            continue
        msgs, tools = R.to_inkling(conv, defang)
        if g in long_groups:
            continue                           # a long-context group is not reused for the agentic set
        if len(long) < a.long_n and g not in agentic_groups:
            for w in R.windows(msgs, tools, tok, 2 * a.long_ctx):
                if len(w) >= a.long_ctx:
                    long.append(w[: a.long_ctx])
                    l_man.append({"i": len(l_man), "source": "session", "group": g, "format": "chat",
                                  "domain": "session_long", "n_tokens": a.long_ctx})
                    long_groups.add(g)
                    break
            if g in long_groups:
                continue
        if gen_total < a.agentic_gen and sum(m["group"] == g for m in a_man) < a.per_group:
            ws = R.windows(msgs, tools, tok, a.agentic_len)
            if not ws:
                continue
            take = sorted({int(round(x)) for x in np.linspace(0, len(ws) - 1, min(a.per_group, len(ws)))})
            for k in take:
                if sum(m["group"] == g for m in a_man) >= a.per_group:
                    break
                w = ws[k]
                gen = int(generated_mask(w, chat=True).sum())
                if gen == 0:
                    continue
                agentic.append(w)
                a_man.append({"i": len(a_man), "source": "session", "group": g, "format": "chat",
                              "domain": "session_agent", "n_tokens": int(len(w)), "generated": gen})
                gen_total += gen
                agentic_groups.add(g)
    a.out.mkdir(parents=True, exist_ok=True)
    write_tokens(a.out / "agentic.safetensors", agentic)
    write_manifest(a.out / "agentic_manifest.json", a_man)
    if long:
        write_tokens(a.out / "long.safetensors", long)
        write_manifest(a.out / "long_manifest.json", l_man)
    summary = {"eval_groups": len(groups), "eval_files": len(files),
               "agentic": {"sequences": len(agentic), "tokens": int(sum(len(w) for w in agentic)),
                           "generated": gen_total, "groups": len(agentic_groups)},
               "long": {"sequences": len(long), "tokens_each": a.long_ctx, "groups": len(long_groups)}}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
