"""Diagnostic: which batched float64 LAPACK call corrupts the process (run one case per process; check exit code)."""
import sys

import torch

case, threads, P, K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
torch.set_num_threads(threads)
torch.manual_seed(0)
X = torch.randn(P, 2 * K, K, dtype=torch.float64)
H = X.mT @ X / (2 * K)
if case == "chol":
    torch.linalg.cholesky(H)
elif case == "inv":
    torch.cholesky_inverse(torch.linalg.cholesky(H))
elif case == "upper":
    torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
elif case == "loop":   # one matrix at a time
    for p in range(P):
        L = torch.linalg.cholesky(H[p])
        torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
elif case == "lowerT":  # upper factor of H⁻¹ as the transpose of its lower factor
    torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H))).mT.contiguous()
print("ok")
