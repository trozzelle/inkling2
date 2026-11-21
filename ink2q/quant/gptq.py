"""FP64 GPTQ (Frantar et al. 2022, Algorithm 1) for MLX affine groups, with a block-diagonal least-squares refit."""
import torch


def bf16(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16).to(torch.float64)


def factor(H: torch.Tensor, damp: float):
    """(upper Cholesky factor of the damped H⁻¹, dead-column mask); reusable for every width and refit pass."""
    H = H.clone()
    d = H.diagonal(dim1=-2, dim2=-1)
    dead = d <= 0
    d.masked_fill_(dead, 1.0)
    d.add_(damp * d.mean(-1, keepdim=True))
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    # U = Lᵀ for H⁻¹ = L Lᵀ, so H⁻¹ = UᵀU with U upper. Not cholesky(upper=True): at 4608² with 12 threads that
    # LAPACK path corrupts the heap (macOS 27.0.1, torch 2.13; wrong batches, then a segfault).
    return torch.linalg.cholesky(Hinv).mT.contiguous(), dead


def quantize(W, H, bits, group=128, damp=0.01, block=128, scales=None, biases=None, fac=None):
    """W [P,R,K], H [P,K,K] -> codes uint8 [P,R,K], scales and biases [P,R,K/group] (bf16 values in float64).
    Group affine parameters come from the error-updated weights at group entry unless given. `fac` is
    factor(H, damp), computed here if not given."""
    P, R, K = W.shape
    qmax = 2**bits - 1
    G = K // group
    U, dead = fac if fac is not None else factor(H, damp)
    W = W.clone()
    W.masked_fill_(dead[:, None, :], 0.0)
    Q = torch.zeros(P, R, K, dtype=torch.uint8)
    S = torch.empty(P, R, G, dtype=torch.float64) if scales is None else scales.clone()
    B = torch.empty(P, R, G, dtype=torch.float64) if biases is None else biases.clone()
    for i1 in range(0, K, block):
        i2 = min(i1 + block, K)
        W1 = W[:, :, i1:i2].clone()
        E1 = torch.zeros_like(W1)
        U1 = U[:, i1:i2, i1:i2]
        for j in range(i2 - i1):
            col = i1 + j
            g = col // group
            if col % group == 0 and scales is None:
                wg = W[:, :, col:col + group].clone()
                n_in = min(group, i2 - col)
                wg[:, :, :n_in] = W1[:, :, j:j + n_in]
                lo, hi = wg.amin(-1), wg.amax(-1)
                S[:, :, g] = bf16(torch.where(hi > lo, (hi - lo) / qmax, torch.ones_like(hi)))
                B[:, :, g] = bf16(lo)
            s, b = S[:, :, g], B[:, :, g]
            w = W1[:, :, j]
            q = torch.clamp(torch.round((w - b) / s), 0, qmax)
            Q[:, :, col] = q.to(torch.uint8)
            err = (w - (s * q + b)) / U1[:, j, j][:, None]
            W1[:, :, j:] -= err[:, :, None] * U1[:, j, j:][:, None, :]
            E1[:, :, j] = err
        W[:, :, i2:] -= E1 @ U[:, i1:i2, i2:]
    return Q, S, B


def dequant(Q, S, B, group):
    P, R, K = Q.shape
    return (Q.double().view(P, R, -1, group) * S[..., None] + B[..., None]).view(P, R, K)


def h_error(W, What, H):
    D = W - What
    num = ((D @ H) * D).sum((-1, -2))
    den = ((W @ H) * W).sum((-1, -2))
    return num / den


def refit(W, Q, H, group, S_old, B_old):
    """Per-row, per-group least squares of (scale, bias) under the diagonal block H[g,g], rounded to bf16."""
    P, R, K = W.shape
    S, B = S_old.clone(), B_old.clone()
    for g in range(K // group):
        sl = slice(g * group, (g + 1) * group)
        Hg = H[:, sl, sl]
        q = Q[:, :, sl].double()
        w = W[:, :, sl]
        Hq = q @ Hg
        h1 = Hg.sum(-1)
        a11 = (Hq * q).sum(-1)
        a12 = (q * h1[:, None, :]).sum(-1)
        a22 = h1.sum(-1)[:, None].expand_as(a11)
        r1 = (Hq * w).sum(-1)
        r2 = (w * h1[:, None, :]).sum(-1)
        det = a11 * a22 - a12 * a12
        ok = det.abs() > 1e-12 * (a11 * a22).abs().clamp_min(1e-300)
        safe = torch.where(ok, det, torch.ones_like(det))
        s = (r1 * a22 - r2 * a12) / safe
        b = (a11 * r2 - a12 * r1) / safe
        good = ok & torch.isfinite(s) & torch.isfinite(b)
        S[:, :, g] = torch.where(good, bf16(s), S[:, :, g])
        B[:, :, g] = torch.where(good, bf16(b), B[:, :, g])
    return S, B


def convert(W, H, bits, group=128, damp=0.01, refit_passes=1, fac=None):
    """GPTQ, then refit_passes rounds of (refit, GPTQ with the refit grid fixed), then a final refit."""
    fac = fac if fac is not None else factor(H, damp)
    Q, S, B = quantize(W, H, bits, group, damp, fac=fac)
    for _ in range(refit_passes):
        S, B = refit(W, Q, H, group, S, B)
        Q, S, B = quantize(W, H, bits, group, damp, scales=S, biases=B, fac=fac)
    S, B = refit(W, Q, H, group, S, B)
    # receipt metric: rows contribute independently, so about 512 evenly spaced rows estimate the ratio at 1/9
    # the cost of the full (R x K) @ (K x K) product
    r = slice(None, None, max(1, W.shape[1] // 512))
    return Q, S, B, h_error(W[:, r], dequant(Q, S, B, group)[:, r], H)
