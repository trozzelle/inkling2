"""Multi-stream splice harness: each layer is loaded once per pass and applied to the teacher stream and to every
variant stream; a variant forks from the teacher at its first modified layer."""
import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mlx.core as mx

from ink2q import experts as X
from ink2q import model as M
from ink2q import stream as S
from ink2q.memguard import Floor, apply_limits
from ink2q.source import SourceIndex
from ink2q.tokens import load_manifest, load_tokens, subset, write_manifest, write_tokens

GIB = 1 << 30


@dataclass
class Variant:
    name: str
    layers: dict = field(default_factory=dict)   # layer -> callable(layer_module) -> ExpertSpec

    @property
    def first(self) -> int:
        return min(self.layers) if self.layers else 10**9


def plan_passes(variants, bytes_per_stream: int, budget_bytes: int):
    per_pass = max(1, budget_bytes // bytes_per_stream - 1)
    ordered = sorted(variants, key=lambda v: v.first)
    return [ordered[i:i + per_pass] for i in range(0, len(ordered), per_pass)]


def stream_bytes(seqs, hidden: int, n_layers: int, block: int) -> int:
    """Upper bound on one stream's live bytes: h, its successor, and every boundary snapshot (bf16)."""
    snaps = -(-n_layers // block)
    return int(sum(len(s) for s in seqs) * hidden * 2 * (snaps + 2))


def _save(out: Path, name: str, finals):
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(finals):
        mx.save_safetensors(str(d / f"seq-{i:05d}.safetensors"), {"final": f})


def _step_all(m, layer, states):
    states = [S.step_lean(m, layer, st) for st in states]
    mx.eval([st.h for st in states])
    return states


def run(src: SourceIndex, seqs, variants, out: Path, budget_gib: float, floor: Floor | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    m = M.build(src.config())
    M.load_globals(m, src)
    text = m.args.text
    per = stream_bytes(seqs, text.hidden_size, text.num_hidden_layers, text.n_layers_per_attn_res_block)
    passes = plan_passes(variants, per, int(budget_gib * GIB))
    log = {"bytes_per_stream": per, "budget_bytes": int(budget_gib * GIB),
           "passes": [[v.name for v in p] for p in passes]}
    (out / "passes.json").write_text(json.dumps(log, indent=1) + "\n")
    for pi, group in enumerate(passes or [[]]):
        teacher = [S.embed_lean(m, mx.array(s)) for s in seqs]
        live: dict[str, list] = {}
        for layer in range(text.num_hidden_layers):
            t0 = time.monotonic()
            M.load_layer(m, src, layer)
            mod = m.model.language_model.layers[layer]
            for v in group:
                if v.first == layer:
                    live[v.name] = list(teacher)          # fork: shares buffers until it diverges
            teacher = _step_all(m, layer, teacher)
            for v in group:
                if v.name not in live:
                    continue
                if layer in v.layers:
                    spec = v.layers[layer](mod)
                    with X.install(mod, spec):
                        live[v.name] = _step_all(m, layer, live[v.name])
                    del spec
                else:
                    live[v.name] = _step_all(m, layer, live[v.name])
            M.drop_layer(m, layer)
            mx.clear_cache()
            print(json.dumps({"pass": pi, "layer": layer, "live": len(live),
                              "seconds": round(time.monotonic() - t0, 1)}), flush=True)
            if floor is not None:
                floor.check(f"pass {pi} layer {layer}")
        if pi == 0:
            _save(out, "teacher", [S.final_hidden_lean(m, st) for st in teacher])
        for v in group:
            _save(out, v.name, [S.final_hidden_lean(m, st) for st in live.get(v.name, teacher)])
        live.clear()


def _builder(spec: dict) -> Callable:
    kind = spec["kind"]
    g = spec.get("group_size", 128)
    if kind == "rtn":
        return lambda mod: X.rtn(mod, spec["bits_gu"], spec["bits_dn"], group_size=g)
    if kind == "nvfp4":
        return lambda mod: X.nvfp4(mod)
    if kind == "mlx":
        layer = spec["layer"]
        return lambda mod: X.from_mlx_dir(Path(spec["dir"]), layer)
    raise ValueError(kind)


def load_variants(path: Path) -> list[Variant]:
    vs = []
    for v in json.loads(Path(path).read_text()):
        layers = {int(l): _builder({**spec, "layer": int(l)}) for l, spec in v["layers"].items()}
        vs.append(Variant(v["name"], layers))
    return vs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--tokens", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--variants", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-per-seq", type=int, default=512)
    ap.add_argument("--budget-gib", type=float, default=110)
    a = ap.parse_args()
    apply_limits(min(180, a.budget_gib + 40), 4)
    seqs, man = subset(load_tokens(a.tokens), load_manifest(a.manifest), a.max_per_seq)
    a.out.mkdir(parents=True, exist_ok=True)
    write_tokens(a.out / "tokens.safetensors", seqs)
    write_manifest(a.out / "manifest.json", man)
    run(SourceIndex(a.source), seqs, load_variants(a.variants), a.out, a.budget_gib, Floor(24))


if __name__ == "__main__":
    main()
