"""E5 layer choice from W2's profile: the oQe 2-bit layer with the largest KL, the oQe 3-bit layer with the median KL,
and the layer in 42-47 with the largest KL (each distinct). Prints the comma-separated layers."""
import json
import sys
from pathlib import Path


def choose(eval_dir: Path, oqe_config: Path) -> list[int]:
    q = json.loads(oqe_config.read_text())["quantization"]
    kl = {l: json.loads((eval_dir / f"oqe_L{l}.json").read_text())["summary"]["kl"] for l in range(48)}
    bits = {l: (q[f"model.layers.{l}.mlp.experts.w13"]["bits"], q[f"model.layers.{l}.mlp.experts.w2"]["bits"])
            for l in range(48)}
    two = [l for l in range(48) if 2 in bits[l]]
    three = sorted((l for l in range(48) if bits[l] == (3, 3)), key=lambda l: kl[l])
    l2 = max(two, key=lambda l: kl[l])
    l3 = next(l for l in three[len(three) // 2:] if l != l2)
    l4 = max((l for l in range(42, 48) if l not in (l2, l3)), key=lambda l: kl[l])
    return sorted({l2, l3, l4})


if __name__ == "__main__":
    print(",".join(map(str, choose(Path(sys.argv[1]), Path(sys.argv[2])))))
