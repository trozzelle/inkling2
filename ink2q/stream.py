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
