# correlation() and hessian() are copied verbatim from ds4-dfm-rs tools/vq_calib/ds41_hvq.py.
#
# MIT License
#
# Copyright (c) 2026 The ds4.c authors
# Copyright (c) 2026 Entrpi <entrpi@proton.me> (batched-serving fork modifications)
# Copyright (c) 2023-2026 The ggml authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
"""Shrunk per-expert correlation and Hessian, FP64 (torch, CPU)."""
import torch


def correlation(s, count, prior, shrink):
    """Shrunk correlation matrices, FP64.

    s [P, C, C] token-normalized sums, count [P] routed tokens, prior [C, C] (per token) or
    None for the identity scaled to the expert's own mean diagonal. Returns (R [P, C, C]
    with unit diagonal, the shrunk diagonal [P, C])."""
    s = s.double()
    n = count.double().view(-1, 1, 1)
    if prior is None:
        mean = s.diagonal(dim1=-2, dim2=-1).mean(-1) / n.view(-1).clamp_min(1.0)
        mean = torch.where(count.view(-1) > 0, mean, torch.ones_like(mean))
        cov = s / (n + shrink)
        cov.diagonal(dim1=-2, dim2=-1).add_((shrink * mean / (n.view(-1) + shrink))[:, None])
    else:
        cov = (s + shrink * prior.double()) / (n + shrink)
    cov = 0.5 * (cov + cov.mT)
    diag = cov.diagonal(dim1=-2, dim2=-1).clone()
    scale = torch.where(diag > 0, diag.clamp_min(1e-300).rsqrt(), torch.zeros_like(diag))
    r = cov * scale[:, :, None] * scale[:, None, :]
    r.diagonal(dim1=-2, dim2=-1).fill_(1.0)
    return r, diag


def hessian(col, r, t, damp):
    """H_t [P, C, C] FP64 from column weights col [P, C] (F32) and correlations r.
    The diagonal is col (+ damp * mean(col)) exactly; the off-diagonal part is
    t * sqrt(col_i col_j) r_ij, exact zeros at t = 0."""
    d = col.double()
    root = d.sqrt()
    h = r * (t * root[:, :, None]) * root[:, None, :]
    h.diagonal(dim1=-2, dim2=-1).copy_(d + damp * d.mean(-1, keepdim=True))
    return h


def expert_h(S, n, prior, shrink: float, power: float):
    """Per-expert Hessian from Gram sums S [P,C,C] over n [P] tokens: shrunk correlation, diagonal raised to power."""
    r, diag = correlation(S, n, prior, shrink)
    col = diag.clamp_min(0.0) ** power
    return hessian(col, r, 1.0, 0.0)
