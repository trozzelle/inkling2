"""Build the mlx-lm Inkling 2 model and load or drop one decoder layer at a time."""
import mlx.core as mx
from mlx_lm.models import inkling2

from ink2q.source import SourceIndex


def build(config: dict) -> inkling2.Model:
    return inkling2.Model(inkling2.ModelArgs.from_dict({**config, "vision_config": None}))


def _load(model: inkling2.Model, raw: dict[str, mx.array]) -> None:
    weights = model.sanitize(raw)
    model.load_weights(list(weights.items()), strict=False)


def load_layer(model: inkling2.Model, src: SourceIndex, layer: int) -> None:
    _load(model, src.load(src.layer_names(layer)))


def load_globals(model: inkling2.Model, src: SourceIndex, *, final: bool = True) -> None:
    _load(model, src.load(src.global_names(final)))


def drop_layer(model: inkling2.Model, layer: int) -> None:
    layers = model.model.language_model.layers
    layers[layer] = type(layers[layer])(model.args.text, layer)
