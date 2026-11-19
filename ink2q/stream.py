"""Layer-streamed forward of one sequence; mirrors TextModel.__call__ and Model.__call__ at the pinned commit."""
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm.models.inkling2 import _unweighted_rms_norm


@dataclass
class SeqState:
    h: mx.array                  # [1, L, H]
    blocks: mx.array | None      # [n_blocks, 1, L, H] float32


def embed(model, ids: mx.array) -> SeqState:
    return SeqState(model.model.language_model.embed(ids[None]), None)


def step(model, layer: int, st: SeqState) -> SeqState:
    h, blocks = model.model.language_model.layers[layer](st.h, st.blocks)
    return SeqState(h, blocks)


def final_hidden(model, st: SeqState) -> mx.array:
    tm = model.model.language_model
    normed = _unweighted_rms_norm(st.h.astype(mx.float32))
    h = tm.final_input(st.blocks, normed)[..., 0, :]
    return tm.norm(h).astype(st.h.dtype)[0]


def logits(model, final: mx.array) -> mx.array:
    text = model.args.text
    z = model.lm_head(final).astype(mx.float32)
    cap = text.final_logit_softcapping
    return (mx.tanh(z / cap) * cap)[..., : text.unpadded_vocab_size]


@dataclass
class LeanState:
    h: mx.array            # [1, L, H] bf16
    snaps: list            # bf16 inputs of boundary layers passed so far


def _blocks(snaps: list) -> mx.array | None:
    if not snaps:
        return None
    return mx.stack([_unweighted_rms_norm(s.astype(mx.float32)) for s in snaps])


def embed_lean(model, ids: mx.array) -> LeanState:
    return LeanState(model.model.language_model.embed(ids[None]), [])


def step_lean(model, layer: int, st: LeanState, layer_module=None) -> LeanState:
    mod = layer_module or model.model.language_model.layers[layer]
    h, _ = mod(st.h, _blocks(st.snaps))
    snaps = st.snaps + [st.h] if mod.is_block_boundary else st.snaps
    return LeanState(h, snaps)


def final_hidden_lean(model, st: LeanState) -> mx.array:
    tm = model.model.language_model
    normed = _unweighted_rms_norm(st.h.astype(mx.float32))
    h = tm.final_input(_blocks(st.snaps), normed)[..., 0, :]
    return tm.norm(h).astype(st.h.dtype)[0]
