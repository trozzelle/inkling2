"""E5 layer choice from W2's profile, restricted to layers where single-layer KL tracks perturbation size.

W2d showed that below about layer 30 any single-layer perturbation (RTN 8 included) grows into a KL floor of
0.005-0.012, so single-layer KL there does not measure quantization quality. Among layers >= FIRST: the oQe
2-bit layer with the largest KL, the oQe 3-bit layer with the median KL, and the layer in 42-47 with the largest KL
(each distinct). Prints the comma-separated layers."""
import json
import sys
from pathlib import Path

FIRST = 30


def choose(eval_dir: Path, oqe_config: Path) -> list[int]:
    q = json.loads(oqe_config.read_text())["quantization"]
    kl = {l: json.loads((eval_dir / f"oqe_L{l}.json").read_text())["summary"]["kl"] for l in range(48)}
    bits = {l: (q[f"model.layers.{l}.mlp.experts.w13"]["bits"], q[f"model.layers.{l}.mlp.experts.w2"]["bits"])
            for l in range(48)}
    late = range(FIRST, 48)
    l4 = max(range(42, 48), key=lambda l: kl[l])
    l2 = max((l for l in late if bits[l] == (2, 2) and l != l4), key=lambda l: kl[l])
    three = sorted((l for l in late if bits[l] == (3, 3) and l not in (l2, l4)), key=lambda l: kl[l])
    l3 = three[len(three) // 2]
    return sorted({l2, l3, l4})


if __name__ == "__main__":
    print(",".join(map(str, choose(Path(sys.argv[1]), Path(sys.argv[2])))))
