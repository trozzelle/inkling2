"""Agreement of TensorFold's DSpark drafter with oMLX's, on the same recorded taps.

Both sides replay a sequence row by row. Per sequence `seq-NNNNN`:
  props   int32 [T][7], row p = the proposals made after committed row p, -1 where no hold ran.
          TensorFold: `.props.bin` (raw little-endian); oMLX: `.props.npy`.
  logits  bf16 bits [n][7][201024], the post-bias, post-softcap, vocab-masked step logits of the first holds, in order
          (masked ids are -inf). TensorFold: `.logits.bin` (always 16 holds); oMLX: `.logits.npy` (uint16 bits,
          min(16, holds) holds). Only the first n holds the reference saved are compared.

Noise floor: the reference values are bf16, so a bf16 implementation that rounds the same f32 value independently
differs by up to one ulp per entry. For a value x the rounding error of bf16 (8 significant bits) is uniform over one
ulp(x) = 2^(floor(log2|x|) - 7), with rms ulp(x)/sqrt(12). The floor is sqrt(sum ulp^2/12) / sqrt(sum x^2) over the
reference's finite entries: the rel-RMS of a single bf16 rounding of the reference's own scale. The gate allows 2x it.

A hold's step k is conditioned on the proposal of step k-1 (`prev`), so once the two sides propose different tokens
the later steps of that hold are not comparable. The gate pools only the comparable steps of each hold: steps 0 up to
and including the first step whose proposal differs (all 7 when the proposals agree). The all-steps figure is reported
as information only.

`accept` scores both sides' proposals against the recorded greedy tokens (tokens per cycle at depths 3/5/7 and
per-position acceptance, as scripts/dspark_accept.py summarises them); it passes when TensorFold's tokens per cycle are
within 1% of the reference's at every depth for every set of at least 1,000 scored rows. Smaller sets (one sequence, a
few hundred rows) are reported but not gated: a single divergent proposal moves them by more than 1%.

`depthsim` replays a draft depth policy on TensorFold's own replay: `seq-NNNNN.props.bin` and `seq-NNNNN.probs.bin` (f32 [T][7], the
top-1 probability of each proposal under its step's logits, -1 where no hold ran) of every hold, all of them, against the recorded
tokens. It reports tokens per second of fixed depths 1 to 7 and of a confidence floor (the first draft verified, each next one while its
own probability is at least the floor) on each set of the prompts file (the gate's sets, bench and chat, as `gate`, and `all`), by a
cost model of a round: window, each verified draft, drafter hold. See ink2q/tfdepth.py.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

BLOCK = 7
HOLDS = 16
AGREEMENT_MIN = 0.99
FLOOR_FACTOR = 2.0


def parse_seqs(spec: str) -> list[int]:
    """'0..20' is inclusive; commas join: '0..3,7,9..10'."""
    out: list[int] = []
    for part in spec.split(","):
        a, sep, b = part.partition("..")
        lo, hi = int(a), int(b) if sep else int(a)
        if hi < lo:
            raise ValueError(f"empty range {part!r}")
        out.extend(range(lo, hi + 1))
    return out


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def bf16_ulp_sq(x: np.ndarray) -> np.ndarray:
    """Squared bf16 ulp of each finite value of x (f32); zero entries contribute nothing."""
    _, e = np.frexp(x)  # |x| = m * 2^e with m in [0.5, 1): floor(log2|x|) = e - 1
    ulp = np.ldexp(np.float32(1.0), e - 8)
    return np.where(x == 0, 0.0, ulp.astype(np.float64) ** 2)


def load_props_tf(path: Path) -> np.ndarray:
    raw = np.fromfile(path, "<i4")
    if raw.size % BLOCK:
        raise ValueError(f"{path}: {raw.size} int32 values is not a multiple of {BLOCK}")
    return raw.reshape(-1, BLOCK)


def load_props_ref(path: Path) -> np.ndarray:
    a = np.load(path)
    if a.ndim != 2 or a.shape[1] != BLOCK:
        raise ValueError(f"{path}: shape {a.shape}, expected [T][{BLOCK}]")
    return a.astype(np.int32, copy=False)


def compare_props(tf: np.ndarray, ref: np.ndarray) -> dict:
    """Counts over the proposal positions where both sides have a token. Rows are holds on one side or both; a row
    where only one side ran a hold is counted in `rows_tf_only` / `rows_ref_only` and takes no part in agreement."""
    if tf.shape != ref.shape:
        raise ValueError(f"props shape mismatch: tensorfold {tf.shape} vs reference {ref.shape}")
    both = (tf >= 0) & (ref >= 0)
    eq = both & (tf == ref)
    hold_tf, hold_ref = (tf >= 0).any(axis=1), (ref >= 0).any(axis=1)
    # conditional: position k counts only in rows whose positions 0..k-1 were all present and equal
    prefix = np.ones(len(tf), bool)
    cond_n, cond_eq = np.zeros(BLOCK, np.int64), np.zeros(BLOCK, np.int64)
    for k in range(BLOCK):
        live = prefix & both[:, k]
        cond_n[k], cond_eq[k] = live.sum(), (live & eq[:, k]).sum()
        prefix = live & eq[:, k]
    return {
        "rows": int((hold_tf & hold_ref).sum()),
        "rows_tf_only": int((hold_tf & ~hold_ref).sum()),
        "rows_ref_only": int((hold_ref & ~hold_tf).sum()),
        "n": both.sum(axis=0).astype(np.int64),
        "eq": eq.sum(axis=0).astype(np.int64),
        "cond_n": cond_n,
        "cond_eq": cond_eq,
        "prefix_len_sum": int(eq.cumprod(axis=1).sum()),
    }


def _frac(num, den) -> float:
    return float(num) / float(den) if den else float("nan")


def logits_stats(tf_bits: np.ndarray, ref_bits: np.ndarray) -> dict:
    """Sums for one hold ([7][V] bf16 bits each). Entries finite in both sides are compared; entries whose finiteness
    differs (a masked id left open or the reverse) are counted as `nonfinite_mismatch`."""
    if tf_bits.shape != ref_bits.shape:
        raise ValueError(f"logits shape mismatch: tensorfold {tf_bits.shape} vs reference {ref_bits.shape}")
    t, r = bf16_to_f32(tf_bits), bf16_to_f32(ref_bits)
    ft, fr = np.isfinite(t), np.isfinite(r)
    ok = ft & fr
    rv, d = r[ok].astype(np.float64), t[ok].astype(np.float64) - r[ok].astype(np.float64)
    return {
        "n": int(ok.sum()),
        "sum_d2": float((d * d).sum()),
        "sum_r2": float((rv * rv).sum()),
        "sum_floor2": float(bf16_ulp_sq(r[ok]).sum() / 12.0),
        "max_abs": float(np.abs(d).max()) if d.size else 0.0,
        "exact": int((d == 0).sum()),
        "nonfinite_mismatch": int((ft != fr).sum()),
    }


def _rel_rms(sum_d2: float, sum_r2: float) -> float:
    return float(np.sqrt(sum_d2 / sum_r2)) if sum_r2 > 0 else float("nan")


def _pool(stats: list[dict]) -> dict:
    s = {k: sum(x[k] for x in stats) for k in ("n", "sum_d2", "sum_r2", "sum_floor2", "exact", "nonfinite_mismatch")}
    return {
        "holds": len(stats),
        "n": s["n"],
        "rel_rms": _rel_rms(s["sum_d2"], s["sum_r2"]),
        "max_abs": max((x["max_abs"] for x in stats), default=0.0),
        "floor": _rel_rms(s["sum_floor2"], s["sum_r2"]),
        "exact_frac": _frac(s["exact"], s["n"]),
        "nonfinite_mismatch": s["nonfinite_mismatch"],
    }


def _agreement(n: np.ndarray, eq: np.ndarray, cond_n: np.ndarray, cond_eq: np.ndarray) -> dict:
    return {
        "total": _frac(eq.sum(), n.sum()),
        "per_position": [_frac(eq[k], n[k]) for k in range(BLOCK)],
        "first": _frac(eq[0], n[0]),
        "conditional": [_frac(cond_eq[k], cond_n[k]) for k in range(BLOCK)],
    }


def comparable_steps(tf_row: np.ndarray, ref_row: np.ndarray) -> int:
    """Steps of a hold whose logits are comparable: up to and including the first differing proposal."""
    diff = np.nonzero(tf_row != ref_row)[0]
    return BLOCK if diff.size == 0 else int(diff[0]) + 1


def compare_seq(tf_dir: Path, ref_dir: Path, seq: int) -> tuple[dict, list[dict], list[dict]]:
    name = f"seq-{seq:05d}"
    props_tf, props_ref = load_props_tf(tf_dir / f"{name}.props.bin"), load_props_ref(ref_dir / f"{name}.props.npy")
    props = compare_props(props_tf, props_ref)
    ref = np.load(ref_dir / f"{name}.logits.npy", mmap_mode="r")
    if ref.dtype != np.uint16 or ref.ndim != 3 or ref.shape[1] != BLOCK:
        raise ValueError(f"{name}.logits.npy: dtype {ref.dtype} shape {ref.shape}, expected uint16 [n][{BLOCK}][vocab]")
    n, vocab = ref.shape[0], ref.shape[2]
    path = tf_dir / f"{name}.logits.bin"
    tf = np.memmap(path, "<u2", "r")
    if tf.size != HOLDS * BLOCK * vocab:
        raise ValueError(f"{path}: {tf.size} bf16 values, expected {HOLDS}x{BLOCK}x{vocab}")
    if n > HOLDS:
        raise ValueError(f"{name}.logits.npy has {n} holds, more than the {HOLDS} TensorFold writes")
    tf = tf.reshape(HOLDS, BLOCK, vocab)
    rows = np.nonzero((props_ref >= 0).any(axis=1))[0][:n]
    if len(rows) < n:
        raise ValueError(f"{name}: reference saved {n} holds of logits but its props have {len(rows)} hold rows")
    holds, comparable = [], []
    for i, row in enumerate(rows):
        t, r = np.asarray(tf[i]), np.asarray(ref[i])
        holds.append(logits_stats(t, r))
        m = comparable_steps(props_tf[row], props_ref[row])
        comparable.append(logits_stats(t[:m], r[:m]))
    return props, holds, comparable


def compare(tf_dir: Path, ref_dir: Path, seqs: list[int]) -> dict:
    per_seq, all_holds, all_comp = {}, [], []
    tot = {k: np.zeros(BLOCK, np.int64) for k in ("n", "eq", "cond_n", "cond_eq")}
    only_tf = only_ref = rows = prefix = 0
    for seq in seqs:
        p, holds, comp = compare_seq(tf_dir, ref_dir, seq)
        for k in tot:
            tot[k] += p[k]
        only_tf, only_ref, rows, prefix = (only_tf + p["rows_tf_only"], only_ref + p["rows_ref_only"],
                                           rows + p["rows"], prefix + p["prefix_len_sum"])
        all_holds.extend(holds)
        all_comp.extend(comp)
        per_seq[seq] = {
            "rows": p["rows"],
            "agreement": _agreement(p["n"], p["eq"], p["cond_n"], p["cond_eq"]),
            "logits": {**_pool(comp), "per_hold": [{"rel_rms": _rel_rms(h["sum_d2"], h["sum_r2"]),
                                                     "max_abs": h["max_abs"]} for h in comp]},
            "logits_all_steps": _pool(holds),
        }
    agreement = _agreement(tot["n"], tot["eq"], tot["cond_n"], tot["cond_eq"])
    logits = _pool(all_comp)
    logits_all = _pool(all_holds)
    checks = {
        "agreement": agreement["total"] >= AGREEMENT_MIN,
        "rel_rms": logits["rel_rms"] <= FLOOR_FACTOR * logits["floor"],
        "holds_match": only_tf == 0 and only_ref == 0,
        "mask_match": logits_all["nonfinite_mismatch"] == 0,
    }
    return {
        "seqs": seqs,
        "rows": rows,
        "rows_tf_only": only_tf,
        "rows_ref_only": only_ref,
        "agreement": {**agreement, "mean_agreed_prefix": _frac(prefix, rows)},
        "logits": logits,
        "logits_all_steps": logits_all,
        "gate": {"agreement_min": AGREEMENT_MIN, "rel_rms_max_over_floor": FLOOR_FACTOR, **checks},
        "pass": bool(all(checks.values())),
        "per_seq": per_seq,
    }


ACCEPT_REL_TOL = 0.01
ACCEPT_MIN_ROWS = 1000
DEPTHS = (3, 5, 7)
SETS = ("bench", "chat", "agentic", "all")


def _summarize(lengths: dict, prompts: list[dict]) -> dict:
    from ink2q.tfaccept import per_position, simulate
    out = {}
    for kind in SETS:
        segs = [lengths[m["seq"]] for m in prompts if kind == "all" or m["set"] == kind]
        if segs:
            out[kind] = {"rows": int(sum(len(g) for g in segs)),
                         "per_position": per_position(np.concatenate(segs)),
                         "tokens_per_cycle": {d: float(np.mean([simulate(g, d)["tokens_per_cycle"] for g in segs]))
                                              for d in DEPTHS}}
    return out


def accept(tf_dir: Path, ref_dir: Path, taps: Path, prompts: list[dict], min_rows: int = ACCEPT_MIN_ROWS) -> dict:
    from ink2q.tfaccept import accept_lengths
    lens = {"tf": {}, "ref": {}}
    for meta in prompts:
        seq = meta["seq"]
        meta_t = json.loads((taps / f"seq-{seq:05d}.tokens.json").read_text())
        L, tokens = meta_t["prompt"], np.array(meta_t["tokens"], np.int64)
        T = len(tokens) - 1
        name = f"seq-{seq:05d}"
        for side, props in (("tf", load_props_tf(tf_dir / f"{name}.props.bin")),
                            ("ref", load_props_ref(ref_dir / f"{name}.props.npy"))):
            if len(props) != T:
                raise ValueError(f"{name} {side}: {len(props)} props rows, expected {T}")
            lens[side][seq] = accept_lengths(props, tokens, L - 1, T - 7)
    tf, ref = _summarize(lens["tf"], prompts), _summarize(lens["ref"], prompts)
    rel = {k: {d: tf[k]["tokens_per_cycle"][d] / ref[k]["tokens_per_cycle"][d] - 1 for d in DEPTHS} for k in ref}
    gated = [k for k in ref if ref[k]["rows"] >= min_rows]
    ok = bool(gated) and all(abs(rel[k][d]) <= ACCEPT_REL_TOL for k in gated for d in DEPTHS)
    return {"tf": tf, "ref": ref, "tokens_per_cycle_rel_diff": rel, "tolerance": ACCEPT_REL_TOL,
            "min_rows": min_rows, "gated_sets": gated, "pass": ok}


def load_probs_tf(path: Path) -> np.ndarray:
    """f32 [T][7]: each proposal's probability under its step's logits, -1 where no hold ran."""
    raw = np.fromfile(path, "<f4")
    if raw.size % BLOCK:
        raise ValueError(f"{path}: {raw.size} float32 values is not a multiple of {BLOCK}")
    return raw.reshape(-1, BLOCK)


def depthsim(tf_dir: Path, taps: Path, prompts: list[dict], cost=None, thresholds=None) -> dict:
    """Rates of fixed depths and of confidence floors on every set of `prompts` (see ink2q/tfdepth.py): the gate's sets pooled as
    `gate`, every sequence as `all`, and each set of its own."""
    from ink2q import tfdepth
    from ink2q.tfaccept import accept_lengths
    seqs = {}
    for meta in prompts:
        seq = meta["seq"]
        name = f"seq-{seq:05d}"
        meta_t = json.loads((taps / f"{name}.tokens.json").read_text())
        L, tokens = meta_t["prompt"], np.array(meta_t["tokens"], np.int64)
        T = len(tokens) - 1
        props = load_props_tf(tf_dir / f"{name}.props.bin")
        probs = load_probs_tf(tf_dir / f"{name}.probs.bin")
        if len(props) != T or len(probs) != T:
            raise ValueError(f"{name}: {len(props)} props rows and {len(probs)} probs rows, expected {T}")
        seqs[seq] = (accept_lengths(props, tokens, L - 1, T - 7), probs[L - 1:T - 7])
    sets = {k: [m["seq"] for m in prompts if m["set"] == k] for k in ("bench", "chat", "agentic")}
    sets["gate"] = [m["seq"] for m in prompts if m["set"] in tfdepth.GATE_SETS]
    sets["all"] = [m["seq"] for m in prompts]
    return tfdepth.simulate(seqs, sets, cost or tfdepth.Cost(), tfdepth.THRESHOLDS if thresholds is None else thresholds)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare", help="TensorFold replay output against the oMLX reference")
    c.add_argument("--tf", type=Path, required=True, help="directory with seq-NNNNN.{props,logits}.bin")
    c.add_argument("--ref", type=Path, required=True, help="directory with seq-NNNNN.{props,logits}.npy")
    c.add_argument("--seqs", default="0..20", help="inclusive ranges and lists: 0..20 or 0..3,7")
    c.add_argument("--out", type=Path, help="also write the JSON report here")
    c = sub.add_parser("accept", help="acceptance of both sides' proposals against the recorded greedy tokens")
    c.add_argument("--tf", type=Path, required=True)
    c.add_argument("--ref", type=Path, required=True)
    c.add_argument("--taps", type=Path, required=True)
    c.add_argument("--prompts", type=Path, required=True)
    c.add_argument("--out", type=Path)
    c = sub.add_parser("depthsim", help="tokens per second of fixed draft depths and of a confidence floor, simulated on a replay")
    c.add_argument("--tf", type=Path, required=True, help="directory with seq-NNNNN.{props,probs}.bin")
    c.add_argument("--taps", type=Path, required=True, help="directory with seq-NNNNN.tokens.json")
    c.add_argument("--prompts", type=Path, required=True)
    c.add_argument("--verify-ms", type=float, default=25.2, help="a window of the pending row alone")
    c.add_argument("--row-ms", type=float, default=5.95, help="each draft verified past it")
    c.add_argument("--hold-ms", type=float, default=5.9, help="the drafter's hold (and its encode), the same at any depth")
    c.add_argument("--thresholds", default=None, help="comma list of floors (0.30 to 0.95 by 0.05 when not given)")
    c.add_argument("--out", type=Path, help="also write the JSON report here")
    a = ap.parse_args()
    if a.cmd == "depthsim":
        from ink2q import tfdepth
        floors = [float(x) for x in a.thresholds.split(",")] if a.thresholds else None
        rep = depthsim(a.tf, a.taps, json.loads(a.prompts.read_text()), tfdepth.Cost(a.verify_ms, a.row_ms, a.hold_ms), floors)
        if a.out:
            a.out.write_text(json.dumps(rep, indent=1) + "\n")
        print(tfdepth.format_report(rep))
        return
    if a.cmd == "accept":
        rep = accept(a.tf, a.ref, a.taps, json.loads(a.prompts.read_text()))
        text = json.dumps(rep, indent=1) + "\n"
        if a.out:
            a.out.write_text(text)
        print(text, end="")
        sys.exit(0 if rep["pass"] else 1)
    rep = compare(a.tf, a.ref, parse_seqs(a.seqs))
    text = json.dumps(rep, indent=1) + "\n"
    if a.out:
        a.out.write_text(text)
    short = {k: rep[k] for k in ("rows", "rows_tf_only", "rows_ref_only", "agreement", "logits", "logits_all_steps",
                                  "gate", "pass")}
    print(json.dumps(short, indent=1))
    sys.exit(0 if rep["pass"] else 1)


if __name__ == "__main__":
    main()
