"""W2 profile variants: oQe experts, RTN 2/2 and RTN 4/4 at one layer each, plus NVFP4 on layers 0-39."""
import json
import sys

OQE = "$INK2Q_ROOT/models/Inkling-2-Small-oQe-3.07bpw-MLX"


def variants(oqe: str = OQE, layers: int = 48) -> list[dict]:
    out = []
    for l in range(layers):
        out.append({"name": f"oqe_L{l}", "layers": {str(l): {"kind": "mlx", "dir": oqe}}})
        out.append({"name": f"rtn2_L{l}", "layers": {str(l): {"kind": "rtn", "bits_gu": 2, "bits_dn": 2}}})
        out.append({"name": f"rtn4_L{l}", "layers": {str(l): {"kind": "rtn", "bits_gu": 4, "bits_dn": 4}}})
    out.append({"name": "nvfp4_0_39", "layers": {str(l): {"kind": "nvfp4"} for l in range(40)}})
    return out


if __name__ == "__main__":
    json.dump(variants(*sys.argv[1:2]), sys.stdout, indent=1)
