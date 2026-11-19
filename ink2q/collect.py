"""Per-expert calibration statistics for one MoE layer (mlx-lm Inkling 2), Gram matrices from bf16-exact operands."""
import contextlib

import mlx.core as mx
import numpy as np
from mlx_lm.models.inkling2 import _softcapped_swiglu

from ink2q.experts import source_bf16


class _Capture:
    def __init__(self, inner, sink):
        self.inner, self.sink = inner, sink

    def __call__(self, x):
        self.sink.append(x)
        return self.inner(x)

    def __getattr__(self, k):
        return getattr(self.inner, k)


@contextlib.contextmanager
def capture(layer_module):
    """Record every input of layer_module.mlp (bf16 [1, T, H]) while it runs unchanged."""
    sink, orig = [], layer_module.mlp
    layer_module.mlp = _Capture(orig, sink)
    try:
        yield sink
    finally:
        layer_module.mlp = orig


def gram_bf16x3(u: mx.array) -> mx.array:
    """fp32 uᵀu for fp32 u, as three matmuls of bf16-exact pieces (exact products on the GPU). The omitted lo·lo
    term bounds the error at about 2^-16 of the matrix scale, far below what the Hessian shrinkage resolves."""
    hi = u.astype(mx.bfloat16)
    lo = (u - hi.astype(mx.float32)).astype(mx.bfloat16)
    hf, lf = hi.astype(mx.float32), lo.astype(mx.float32)
    return hf.T @ hf + hf.T @ lf + lf.T @ hf


class ExpertStats:
    def __init__(self, layer_module, experts=None, raw=True, normalized=True):
        self.L = layer_module
        act = layer_module.mlp.experts.act_fn
        self.gcap, self.ucap = act.gate_softcap, act.up_softcap
        gu = layer_module.mlp.experts.gate_up_proj.weight
        n = gu.shape[0]
        self.experts = list(range(n)) if experts is None else list(experts)
        self.raw, self.normalized = raw, normalized
        self.count = np.zeros(n, np.int64)
        self.H = layer_module.post_attention_layernorm.weight.shape[0]
        self.F = gu.shape[1] // 2
        self.x2 = np.zeros((n, self.H))
        self.a2 = np.zeros((n, self.F))
        self.g: dict[str, dict[int, mx.array]] = {k: {} for k in ("xx_raw", "aa_raw", "xx_norm", "aa_norm")}
        self.held_rows: list = []
        self._w = {}

    def _weights(self, e):
        if e not in self._w:
            self._w[e] = (source_bf16(self.L, "gate_up", slice(e, e + 1))[0],
                          source_bf16(self.L, "down", slice(e, e + 1))[0])
        return self._w[e]

    def _route(self, x):
        inds, _ = self.L.mlp.gate(x)
        return np.array(inds)

    def _act(self, xe, e):
        gu, _ = self._weights(e)
        y = xe @ gu.T
        return _softcapped_swiglu(y[:, 0::2], y[:, 1::2], self.gcap, self.ucap)

    def _acc(self, key, e, m):
        self.g[key][e] = self.g[key][e] + m if e in self.g[key] else m

    def add(self, x: mx.array) -> None:
        x = x.reshape(-1, x.shape[-1])
        inds = self._route(x)
        for e in self.experts:
            rows = np.where((inds == e).any(-1))[0]
            if len(rows) == 0:
                continue
            xe = x[mx.array(rows)]
            act = self._act(xe, e)
            self.count[e] += len(rows)
            xf, af = xe.astype(mx.float32), act.astype(mx.float32)
            self.x2[e] += np.array((xf * xf).sum(0), np.float64)
            self.a2[e] += np.array((af * af).sum(0), np.float64)
            if self.raw:
                self._acc("xx_raw", e, xf.T @ xf)
                self._acc("aa_raw", e, af.T @ af)
            if self.normalized:
                ux = xf * mx.rsqrt(mx.maximum((xf * xf).mean(-1, keepdims=True), 1e-30))
                ua = af * mx.rsqrt(mx.maximum((af * af).mean(-1, keepdims=True), 1e-30))
                self._acc("xx_norm", e, gram_bf16x3(ux))
                self._acc("aa_norm", e, gram_bf16x3(ua))
            mx.eval([self.g[k][e] for k in self.g if e in self.g[k]])

    def held(self, x: mx.array) -> None:
        x = x.reshape(-1, x.shape[-1])
        inds = self._route(x)
        for e in self.experts:
            rows = np.where((inds == e).any(-1))[0]
            if len(rows):
                xe = x[mx.array(rows)]
                self.held_rows.append((e, xe, self._act(xe, e)))

    def to_host(self, dtype=np.float64) -> dict:
        out = {"count": self.count, "x2": self.x2, "a2": self.a2}
        n = len(self.count)
        for k, d in self.g.items():
            if not d:
                continue
            dim = self.H if k.startswith("xx") else self.F
            arr = np.zeros((n, dim, dim), dtype)
            for e, g in d.items():
                arr[e] = np.array(g, dtype)
            out[k] = arr
        return out

    def drop_device(self) -> None:
        for d in self.g.values():
            d.clear()
        mx.clear_cache()
