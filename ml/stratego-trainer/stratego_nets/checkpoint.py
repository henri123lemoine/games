"""safetensors save/load of model weights AND optimizer state.

MLX's `mx.save_safetensors` only stores a flat {str: array} map, so nested param
trees and the AdamW optimizer state are flattened with dotted keys (via
mlx.utils.tree_flatten) and re-nested on load. Model params, EMA shadow, and
optimizer state round-trip exactly.
"""

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


def _flatten(tree, prefix):
    return {f"{prefix}.{k}": v for k, v in tree_flatten(tree)}


def _collect(flat, prefix):
    plen = len(prefix) + 1
    sub = {k[plen:]: v for k, v in flat.items() if k.startswith(prefix + ".")}
    return tree_unflatten(list(sub.items()))


def save(path, model, optimizer=None, ema=None, metadata=None):
    """Write model (+ optional optimizer state + EMA shadow) to a .safetensors file."""
    flat = _flatten(model.parameters(), "model")
    if optimizer is not None:
        flat.update(_flatten(optimizer.state, "opt"))
    if ema is not None:
        flat.update(_flatten(ema.shadow_params(), "ema"))
    meta = {k: str(v) for k, v in (metadata or {}).items()}
    mx.eval(flat)
    mx.save_safetensors(path, flat, metadata=meta)


def load(path, model, optimizer=None, ema=None):
    """Load weights (+ optional optimizer state + EMA shadow) from a .safetensors file.

    Mutates ``model`` (and ``optimizer``/``ema`` if given) in place. Returns the
    raw flat dict so callers can inspect metadata-adjacent tensors if needed.
    """
    flat = mx.load(path)
    model.update(_collect(flat, "model"))
    if optimizer is not None and any(k.startswith("opt.") for k in flat):
        optimizer.state = _collect(flat, "opt")
    if ema is not None and any(k.startswith("ema.") for k in flat):
        ema.load_flat(dict(tree_flatten(_collect(flat, "ema"))))
    mx.eval(model.parameters())
    return flat
