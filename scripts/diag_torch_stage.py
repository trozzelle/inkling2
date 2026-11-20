"""Diagnostic: run one GPTQ stage at full size on synthetic data, to find which one corrupts the process."""
import sys

import torch

from ink2q.quant import gptq as G

stage, K = sys.argv[1], int(sys.argv[2])
torch.manual_seed(0)
X = torch.randn(8, 2 * K, K, dtype=torch.float64)
H = X.mT @ X / (2 * K)
W = torch.randn(8, 4608, K, dtype=torch.float64) * 0.02
fac = G.factor(H, 0.1)
if stage in ("quantize", "refit", "h_error", "convert"):
    Q, S, B = G.quantize(W, H, 3, 128, 0.1, fac=fac)
if stage in ("refit", "convert"):
    S, B = G.refit(W, Q, H, 128, S, B)
if stage == "h_error":
    G.h_error(W, G.dequant(Q, S, B, 128), H)
if stage == "convert":
    G.convert(W, H, 3, 128, 0.1, 1, fac=fac)
print("ok", stage, flush=True)
