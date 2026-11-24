"""Numerical oracle and parity gates for Inkling 2 Mini (inkling2_loop) via mlx-lm."""
import argparse
import contextlib
import json
import math
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

from ink2q.memguard import apply_limits
from ink2q.score import head_logits, score_sequence, summarize
from ink2q.tfio import bf16_bits_to_f32
from ink2q.tokens import load_manifest, load_tokens

MAX_RATIO = 1.5


def rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).mean() / (b**2).mean()))


def read_dump(path: Path, tokens: int, hidden: int) -> np.ndarray:
    return bf16_bits_to_f32(np.fromfile(path, "<u2")).reshape(-1, tokens, hidden)


@contextlib.contextmanager
def fp32_static_convs():
    """Run Inkling 2 static ShortConvolution in fp32 to measure the BF16 rounding noise floor."""
    from mlx_lm.models.inkling2 import layers as ik_layers

    orig = ik_layers.ShortConvolution.__call__

    def call(self, hidden_states, conv_state, conv_mask):
        k = self.kernel_size
        residual = hidden_states.astype(mx.float32)
        window = ik_layers._conv_input(hidden_states, conv_state, conv_mask, k)
        out = mx.conv1d(
            window.astype(mx.float32),
            self.conv1d.weight.astype(mx.float32),
            groups=self.conv1d.groups,
        )
        return ((out + residual) / math.sqrt(k + 1)).astype(hidden_states.dtype), window

    ik_layers.ShortConvolution.__call__ = call
    try:
        yield
    finally:
        ik_layers.ShortConvolution.__call__ = orig


def _defining_call_class(obj) -> type:
    for cls in type(obj).__mro__:
        if "__call__" in cls.__dict__:
            return cls
    raise TypeError(f"{type(obj).__name__} is not callable")


def trace_virtual_layers(model, ids: np.ndarray) -> tuple[np.ndarray, mx.array]:
    """Run one teacher-forced forward and capture all virtual-layer outputs [V, T, H] F32 plus BF16 finals [T, H]."""
    lm = model.model.language_model
    target_ids = {id(layer) for layer in lm.layers}
    captured: list[np.ndarray] = []
    patched: list[tuple[type, object]] = []
    seen_classes: set[type] = set()

    for layer in lm.layers:
        cls = _defining_call_class(layer)
        if cls in seen_classes:
            continue
        seen_classes.add(cls)
        orig_call = cls.__dict__["__call__"]
        patched.append((cls, orig_call))

        def make_wrap(fn):
            def wrapped(self, *args, **kwargs):
                out = fn(self, *args, **kwargs)
                if id(self) in target_ids:
                    h = out[0] if isinstance(out, tuple) else out
                    mx.eval(h)
                    captured.append(np.array(h[0].astype(mx.float32)))
                return out

            return wrapped

        cls.__call__ = make_wrap(orig_call)

    try:
        tokens = mx.array(ids[None, :])
        finals = lm(tokens)[0]
        mx.eval(finals)
    finally:
        for cls, orig_call in patched:
            cls.__call__ = orig_call

    return np.stack(captured), finals


def gate_layers(ref: np.ndarray, floor: np.ndarray, tf: np.ndarray, max_ratio: float = MAX_RATIO) -> dict:
    rows = []
    for v in range(tf.shape[0]):
        fl = rel_rms(floor[v], ref[v])
        d = rel_rms(tf[v], ref[v])
        rows.append({
            "virtual_layer": v,
            "rel_rms": d,
            "noise_floor": fl,
            "ratio": d / fl if fl > 0 else math.inf,
            "max_abs": float(np.abs(tf[v] - ref[v]).max()),
        })
    return {"layers": rows, "max_ratio": max_ratio, "pass": all(r["ratio"] <= max_ratio for r in rows)}


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def score_finals(
    head: mx.array,
    ids: np.ndarray,
    ref_finals: mx.array,
    tf_finals: mx.array,
    vocab_real: int,
    max_kl_upper: float = 0.002,
    min_top1: float = 0.98,
) -> dict:
    row = score_sequence(
        ids,
        False,
        lambda c0, c1: head_logits(ref_finals[c0:c1], head, vocab_real),
        lambda c0, c1: head_logits(tf_finals[c0:c1], head, vocab_real),
        positions=True,
    )
    pos_kl = np.asarray(row.pop("pos")["kl"], dtype=np.float64)
    summary = summarize([row])
    lo, hi = bootstrap_ci(pos_kl)
    passed = bool(hi <= max_kl_upper and summary["top1"] >= min_top1)
    return {
        **summary,
        "kl_ci95_lower": lo,
        "kl_ci95_upper": hi,
        "max_kl_upper": max_kl_upper,
        "min_top1": min_top1,
        "pass": passed,
    }


def _load_unembed(model_dir: Path) -> mx.array:
    key = "model.llm.unembed.weight"
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        shard = json.loads(index_path.read_text())["weight_map"][key]
        return mx.load(str(model_dir / shard))[key]
    for name in ("model-00001-of-00001.safetensors", "model.safetensors"):
        p = model_dir / name
        if p.exists():
            return mx.load(str(p))[key]
    raise FileNotFoundError(f"no safetensors file containing {key} in {model_dir}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("ref")
    r.add_argument("--model", type=Path, required=True)
    r.add_argument("--tokens", type=Path, required=True)
    r.add_argument("--manifest", type=Path, required=True)
    r.add_argument("--case", default="en_web")
    r.add_argument("--max-tokens", type=int, default=512)
    r.add_argument("--out", type=Path, required=True)

    g = sub.add_parser("gate")
    g.add_argument("--ref", type=Path, required=True)
    g.add_argument("--tf", type=Path, required=True)
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--max-ratio", type=float, default=MAX_RATIO)

    k = sub.add_parser("kl")
    k.add_argument("--model", type=Path, required=True)
    k.add_argument("--ref", type=Path, required=True)
    k.add_argument("--tf-finals", type=Path, required=True)
    k.add_argument("--out", type=Path, required=True)
    k.add_argument("--max-kl-upper", type=float, default=0.002)
    k.add_argument("--min-top1", type=float, default=0.98)

    a = ap.parse_args(argv)
    if a.cmd == "ref":
        from mlx_lm.utils import load_model

        apply_limits(80, 4)
        seqs, man = load_tokens(a.tokens), load_manifest(a.manifest)
        i = next(idx for idx, m in enumerate(man) if m.get("case") == a.case)
        ids = seqs[i][: a.max_tokens]
        model, _ = load_model(a.model)
        ref, finals = trace_virtual_layers(model, ids)
        with fp32_static_convs():
            flo, _ = trace_virtual_layers(model, ids)
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "ids.json").write_text(json.dumps([int(t) for t in ids]))
        np.save(a.out / "ref.npy", ref)
        np.save(a.out / "floor.npy", flo)
        mx.save_safetensors(str(a.out / "finals.safetensors"), {"final": finals})
        print(json.dumps({"case": a.case, "tokens": len(ids), "virtual_layers": int(ref.shape[0]), "hidden": int(ref.shape[2])}))
        return

    if a.cmd == "gate":
        ref, flo = np.load(a.ref / "ref.npy"), np.load(a.ref / "floor.npy")
        tf = read_dump(a.tf, ref.shape[1], ref.shape[2])
        res = gate_layers(ref[: tf.shape[0]], flo[: tf.shape[0]], tf, a.max_ratio)
        for row in res["layers"]:
            print(json.dumps(row))
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps({"layer_gate_pass": res["pass"]}))
        sys.exit(0 if res["pass"] else 1)

    apply_limits(48, 4)
    cfg = json.loads((a.model / "config.json").read_text())
    vocab_real = int(cfg["text_config"]["unpadded_vocab_size"])
    hidden = int(cfg["text_config"]["hidden_size"])
    ids = np.asarray(json.loads((a.ref / "ids.json").read_text()), dtype=np.int32)
    ref_finals = mx.load(str(a.ref / "finals.safetensors"))["final"]
    tf_raw = bf16_bits_to_f32(np.fromfile(a.tf_finals, "<u2")).reshape(len(ids), hidden)
    tf_finals = mx.array(tf_raw).astype(mx.bfloat16)
    head = _load_unembed(a.model)
    res = score_finals(head, ids, ref_finals, tf_finals, vocab_real, a.max_kl_upper, a.min_top1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
