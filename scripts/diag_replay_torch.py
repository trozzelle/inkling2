"""Diagnostic: replay layer-30 w13 GPTQ for experts 16-31 with torch only (no MLX in the process)."""
import sys
import time

import numpy as np
import torch

from ink2q.quant import gptq as G
from ink2q.quant.hessian import expert_h

d = sys.argv[1]
threads = int(sys.argv[2]) if len(sys.argv) > 2 else 0
if threads:
    torch.set_num_threads(threads)
z = np.load(f"{d}/L30_e16-31.npz")
Wall = torch.from_numpy(np.load(f"{d}/W_L30_e16-31.npy").astype(np.float64))
prior = torch.from_numpy(z["all"] / float(z["tokens"]))
n = torch.from_numpy(z["count"])
for rep in range(3):
    for b0 in (0, 8):
        t = time.perf_counter()
        sl = slice(b0, b0 + 8)
        H = expert_h(torch.from_numpy(z["xx_raw"][sl].astype(np.float64)), n[sl], prior, 16384.0, 1.0)
        fac = G.factor(H, 0.1)
        errs = []
        for bits in (2, 3, 4):
            _, _, _, err = G.convert(Wall[sl], H, bits, group=128, damp=0.1, refit_passes=1, fac=fac)
            errs.append(float(err.max()))
        print(f"threads {torch.get_num_threads()} rep {rep} experts {16 + b0}-{23 + b0}: max h_err b2/b3/b4 "
              f"{errs[0]:.4f}/{errs[1]:.4f}/{errs[2]:.4f}  U finite {bool(torch.isfinite(fac[0]).all())}  "
              f"{time.perf_counter() - t:.0f}s", flush=True)
