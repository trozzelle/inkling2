"""Draft depth for DSpark by a confidence floor, simulated on a TensorFold replay's recorded proposals.

A replay (`tf-inkling-run <pack> dspark ...`) holds a block after every committed row of a recorded greedy run and writes each
row's proposals (`props.bin`, int32 [T][7]) and the probability of each proposal under its step's logits (`probs.bin`, f32
[T][7]: the top-1 probability of the softcapped, vocabulary-masked row). A drafted round starting at row p verifies n of the 7
proposals and commits min(a_p, n) + 1 tokens, a_p being the accepted length of the hold at row p (its leading proposals that equal
the recorded tokens); the next round starts that many rows on. Two ways to choose n:

  fixed d   n = d every round.
  floor t   n = 1 + the leading proposals after the first whose own probability is at least t, up to 7: what the engine runs with
            INKLING_DRAFT_CONFIDENCE=t (the lane core compares each draft's chance with the floor from the second on and stops at
            the first that falls short; the first draft is always verified).

A round costs verify_ms + row_ms * n + hold_ms: the window forward over the pending row alone, each draft verified past it, and the
drafter's hold, which is the same for any n (its block pass is fixed). A set's rate pools every round of its sequences: tokens
over time.
"""
from dataclasses import asdict, dataclass

import numpy as np

BLOCK = 7
FIXED_DEPTHS = tuple(range(1, BLOCK + 1))
THRESHOLDS = tuple(round(0.30 + 0.05 * i, 2) for i in range(14))  # 0.30 .. 0.95
GATE_SETS = ("bench", "chat")
SET_NAMES = ("bench", "chat", "agentic", "gate", "all")
BASELINE_DEPTH = 4  # the served default


@dataclass(frozen=True)
class Cost:
    """Milliseconds of a round: the window of the pending row alone, each draft verified past it, the drafter's hold."""
    verify_ms: float = 25.2
    row_ms: float = 5.95
    hold_ms: float = 5.9


@dataclass
class Tally:
    """Rounds, tokens committed and drafts verified."""
    cycles: int = 0
    tokens: int = 0
    drafts: int = 0

    def add(self, other: "Tally") -> None:
        self.cycles += other.cycles
        self.tokens += other.tokens
        self.drafts += other.drafts

    def ms(self, cost: Cost) -> float:
        return self.cycles * (cost.verify_ms + cost.hold_ms) + cost.row_ms * self.drafts

    def tok_s(self, cost: Cost) -> float:
        return 1000.0 * self.tokens / self.ms(cost) if self.cycles else float("nan")

    def entry(self, cost: Cost) -> dict:
        return {"tok_s": self.tok_s(cost), "rounds": self.cycles,
                "tokens_per_round": self.tokens / self.cycles if self.cycles else float("nan"),
                "drafts_per_round": self.drafts / self.cycles if self.cycles else float("nan")}


def kept_fixed(rows: int, depth: int) -> np.ndarray:
    """The drafts a fixed depth verifies at each of `rows` rows."""
    if not 1 <= depth <= BLOCK:
        raise ValueError(f"depth {depth}: 1 to {BLOCK}")
    return np.full(rows, depth, np.int64)


def kept_floor(probs: np.ndarray, floor: float) -> np.ndarray:
    """The drafts the floor lets through at each row of probs [rows][7]: the first, then each next one while its own probability
    is at least `floor`, stopping at the first that is not (a later one over the floor does not count). Each f32 probability is
    widened before it meets the f64 floor, as the lane core does."""
    if probs.ndim != 2 or probs.shape[1] != BLOCK:
        raise ValueError(f"probs {probs.shape}: [rows][{BLOCK}] expected")
    return 1 + np.cumprod(probs[:, 1:].astype(np.float64) >= floor, axis=1, dtype=np.int64).sum(axis=1)


def walk(lengths: np.ndarray, kept: np.ndarray) -> Tally:
    """One greedy run round by round from its first row: lengths[p] the accepted length of the hold at row p, kept[p] the drafts a
    round starting there verifies."""
    if len(lengths) != len(kept):
        raise ValueError(f"{len(lengths)} lengths, {len(kept)} kept")
    a, k = lengths.tolist(), kept.tolist()
    t = Tally()
    p = 0
    while p < len(a):
        take = min(a[p], k[p]) + 1
        t.cycles += 1
        t.tokens += take
        t.drafts += k[p]
        p += take
    return t


def check_rows(lengths: np.ndarray, probs: np.ndarray, name: str = "") -> None:
    """A sequence's scored rows: every hold has its seven probabilities (a replay writes -1 where no hold ran)."""
    if len(lengths) != len(probs):
        raise ValueError(f"{name}: {len(lengths)} accepted lengths, {len(probs)} probability rows")
    if probs.ndim != 2 or probs.shape[1] != BLOCK:
        raise ValueError(f"{name}: probs {probs.shape}: [rows][{BLOCK}] expected")
    if not np.all((probs > 0) & (probs <= 1)):
        raise ValueError(f"{name}: a scored row has a probability outside (0, 1] (a replay without probabilities, or a hold missing)")


def simulate(seqs: dict, sets: dict, cost: Cost = Cost(), thresholds=THRESHOLDS) -> dict:
    """Rates of every fixed depth and every floor on each set. `seqs`: sequence -> (accepted lengths [rows], probabilities
    [rows][7]) of its scored rows; `sets`: name -> its sequences."""
    for seq, (lengths, probs) in seqs.items():
        check_rows(lengths, probs, f"seq {seq}")
    out = {"cost": asdict(cost), "thresholds": [float(t) for t in thresholds], "sets": {}}
    for name, members in sets.items():
        if not any(len(seqs[s][0]) for s in members):
            continue
        fixed = {d: Tally() for d in FIXED_DEPTHS}
        floors = {float(t): Tally() for t in thresholds}
        for seq in members:
            lengths, probs = seqs[seq]
            for d, tally in fixed.items():
                tally.add(walk(lengths, kept_fixed(len(lengths), d)))
            for t, tally in floors.items():
                tally.add(walk(lengths, kept_floor(probs, t)))
        best_fixed = max(fixed, key=lambda d: fixed[d].tok_s(cost))
        best_floor = max(floors, key=lambda t: floors[t].tok_s(cost)) if floors else None
        entry = {"sequences": list(members), "rows": int(sum(len(seqs[s][0]) for s in members)),
                 "fixed": {str(d): v.entry(cost) for d, v in fixed.items()},
                 "floor": {f"{t:.2f}": v.entry(cost) for t, v in floors.items()},
                 "best_fixed": {"depth": best_fixed, "tok_s": fixed[best_fixed].tok_s(cost)}}
        if best_floor is not None:
            base = fixed[BASELINE_DEPTH].tok_s(cost)
            top = floors[best_floor].tok_s(cost)
            entry["best_floor"] = {"threshold": best_floor, "tok_s": top,
                                   "vs_fixed_4_pct": 100 * (top / base - 1),
                                   "vs_best_fixed_pct": 100 * (top / fixed[best_fixed].tok_s(cost) - 1)}
        out["sets"][name] = entry
    return out


def format_report(rep: dict) -> str:
    """The rates of `simulate`'s report as a table: a row for each rule, a column for each set."""
    names = [n for n in SET_NAMES if n in rep["sets"]]
    c = rep["cost"]
    lines = [f"cost model: window {c['verify_ms']} ms + {c['row_ms']} ms a verified draft + hold {c['hold_ms']} ms; tok/s, pooled over each set's rounds",
             "sets: " + ", ".join(f"{n} {len(rep['sets'][n]['sequences'])} sequences / {rep['sets'][n]['rows']} rows" for n in names),
             f"{'rule':<14}" + "".join(f"{n:>10}" for n in names)]
    for d in FIXED_DEPTHS:
        lines.append(f"{'fixed ' + str(d):<14}" + "".join(f"{rep['sets'][n]['fixed'][str(d)]['tok_s']:>10.1f}" for n in names))
    for t in rep["thresholds"]:
        lines.append(f"{'floor %.2f' % t:<14}" + "".join(f"{rep['sets'][n]['floor'][f'{t:.2f}']['tok_s']:>10.1f}" for n in names))
    for n in names:
        s = rep["sets"][n]
        line = f"{n}: best fixed depth {s['best_fixed']['depth']} at {s['best_fixed']['tok_s']:.1f} tok/s"
        if "best_floor" in s:
            b = s["best_floor"]
            line += (f"; best floor {b['threshold']:.2f} at {b['tok_s']:.1f} tok/s, {b['vs_fixed_4_pct']:+.1f}% on fixed {BASELINE_DEPTH}"
                     f" and {b['vs_best_fixed_pct']:+.1f}% on the best fixed depth")
        lines.append(line)
    return "\n".join(lines)
