"""W2d diagnostic variants: a null swap (harness offset), RTN 8 vs RTN 4 at single layers (chaotic floor), and
all-oQe with one layer restored (marginal sensitivity at the oQe operating point)."""
import json
import sys

OQE = "$INK2Q_ROOT/models/Inkling-2-Small-oQe-3.07bpw-MLX"
PROBE = [0, 4, 12, 20, 30, 40, 47]
LOO = [0, 4, 6, 12, 18, 24, 30, 36, 42, 46]


def variants(oqe: str = OQE) -> list[dict]:
    mlx = {"kind": "mlx", "dir": oqe}
    out = [{"name": "null_L0", "layers": {"0": {"kind": "source"}}}]
    for l in PROBE:
        for b in (8, 4):
            out.append({"name": f"rtn{b}_L{l}", "layers": {str(l): {"kind": "rtn", "bits_gu": b, "bits_dn": b}}})
    out.append({"name": "oqe_all", "layers": {str(l): mlx for l in range(48)}})
    for k in LOO:
        out.append({"name": f"oqe_all_but_L{k}", "layers": {str(l): mlx for l in range(48) if l != k}})
    return out


if __name__ == "__main__":
    json.dump(variants(*sys.argv[1:2]), sys.stdout, indent=1)
