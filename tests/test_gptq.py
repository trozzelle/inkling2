import torch

from ink2q.quant import gptq as G
from ink2q.quant.hessian import expert_h


def corr_h(K, rho=0.9):
    idx = torch.arange(K, dtype=torch.float64)
    return rho ** (idx[:, None] - idx[None, :]).abs()


def test_identity_h_equals_rtn_on_the_same_grid():
    torch.manual_seed(0)
    W = torch.randn(1, 8, 64, dtype=torch.float64)
    H = torch.eye(64, dtype=torch.float64)[None]
    Q, S, B = G.quantize(W, H, bits=3, group=32, damp=0.0, block=32)
    rtn = torch.clamp(torch.round((W.view(1, 8, 2, 32) - B[..., None]) / S[..., None]), 0, 7).view(1, 8, 64)
    assert torch.equal(Q.double(), rtn)


def test_feedback_lowers_h_error_and_refit_lowers_it_further():
    torch.manual_seed(1)
    K = 128
    W = torch.randn(1, 16, K, dtype=torch.float64)
    H = corr_h(K)[None]
    Qr, Sr, Br = G.quantize(W, torch.eye(K, dtype=torch.float64)[None], 2, group=32, damp=0.0, block=32)
    Qg, Sg, Bg = G.quantize(W, H, 2, group=32, damp=0.01, block=32)
    e_rtn = G.h_error(W, G.dequant(Qr, Sr, Br, 32), H)
    e_gptq = G.h_error(W, G.dequant(Qg, Sg, Bg, 32), H)
    assert float(e_gptq) < 0.8 * float(e_rtn)
    _, _, _, e2 = G.convert(W, H, 2, group=32, damp=0.01, refit_passes=1)
    assert float(e2) <= float(e_gptq) + 1e-12


def test_block_size_does_not_change_the_result():
    torch.manual_seed(2)
    W = torch.randn(2, 8, 128, dtype=torch.float64)
    H = corr_h(128)[None].expand(2, -1, -1).clone()
    a = G.quantize(W, H, 3, group=32, block=32)
    b = G.quantize(W, H, 3, group=32, block=128)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_expert_h_is_symmetric_positive_definite_for_unrouted_experts():
    torch.manual_seed(3)
    X = torch.randn(2, 50, 16, dtype=torch.float64)
    S = X.mT @ X
    S[1] = 0.0
    n = torch.tensor([50, 0])
    H = expert_h(S, n, None, shrink=64.0, power=1.0)
    assert torch.allclose(H, H.mT)
    assert bool((torch.linalg.eigvalsh(H) > 0).all())
