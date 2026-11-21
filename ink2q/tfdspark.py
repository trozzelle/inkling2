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


def compare_seq(tf_dir: Path, ref_dir: Path, seq: int) -> tuple[dict, list[dict]]:
    name = f"seq-{seq:05d}"
    props = compare_props(load_props_tf(tf_dir / f"{name}.props.bin"), load_props_ref(ref_dir / f"{name}.props.npy"))
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
    holds = [logits_stats(np.asarray(tf[i]), np.asarray(ref[i])) for i in range(n)]
    return props, holds


def compare(tf_dir: Path, ref_dir: Path, seqs: list[int]) -> dict:
    per_seq, all_holds = {}, []
    tot = {k: np.zeros(BLOCK, np.int64) for k in ("n", "eq", "cond_n", "cond_eq")}
    only_tf = only_ref = rows = prefix = 0
    for seq in seqs:
        p, holds = compare_seq(tf_dir, ref_dir, seq)
        for k in tot:
            tot[k] += p[k]
        only_tf, only_ref, rows, prefix = (only_tf + p["rows_tf_only"], only_ref + p["rows_ref_only"],
                                           rows + p["rows"], prefix + p["prefix_len_sum"])
        all_holds.extend(holds)
        per_seq[seq] = {
            "rows": p["rows"],
            "agreement": _agreement(p["n"], p["eq"], p["cond_n"], p["cond_eq"]),
            "logits": {**_pool(holds), "per_hold": [{"rel_rms": _rel_rms(h["sum_d2"], h["sum_r2"]),
                                                      "max_abs": h["max_abs"]} for h in holds]},
        }
    agreement = _agreement(tot["n"], tot["eq"], tot["cond_n"], tot["cond_eq"])
    logits = _pool(all_holds)
    checks = {
        "agreement": agreement["total"] >= AGREEMENT_MIN,
        "rel_rms": logits["rel_rms"] <= FLOOR_FACTOR * logits["floor"],
        "holds_match": only_tf == 0 and only_ref == 0,
        "mask_match": logits["nonfinite_mismatch"] == 0,
    }
    return {
        "seqs": seqs,
        "rows": rows,
        "rows_tf_only": only_tf,
        "rows_ref_only": only_ref,
        "agreement": {**agreement, "mean_agreed_prefix": _frac(prefix, rows)},
        "logits": logits,
        "gate": {"agreement_min": AGREEMENT_MIN, "rel_rms_max_over_floor": FLOOR_FACTOR, **checks},
        "pass": bool(all(checks.values())),
        "per_seq": per_seq,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare", help="TensorFold replay output against the oMLX reference")
    c.add_argument("--tf", type=Path, required=True, help="directory with seq-NNNNN.{props,logits}.bin")
    c.add_argument("--ref", type=Path, required=True, help="directory with seq-NNNNN.{props,logits}.npy")
    c.add_argument("--seqs", default="0..20", help="inclusive ranges and lists: 0..20 or 0..3,7")
    c.add_argument("--out", type=Path, help="also write the JSON report here")
    a = ap.parse_args()
    rep = compare(a.tf, a.ref, parse_seqs(a.seqs))
    text = json.dumps(rep, indent=1) + "\n"
    if a.out:
        a.out.write_text(text)
    short = {k: rep[k] for k in ("rows", "rows_tf_only", "rows_ref_only", "agreement", "logits", "gate", "pass")}
    print(json.dumps(short, indent=1))
    sys.exit(0 if rep["pass"] else 1)


if __name__ == "__main__":
    main()
