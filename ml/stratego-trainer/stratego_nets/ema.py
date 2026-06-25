"""Exponential moving average of model parameters (the .pthm "magnet" weights).

Reference rule (`exponential_weighted_average.py`): ema <- decay*ema + (1-decay)*orig.
Decays: 0.999 move/setup, 0.99 belief (see spec.EMA_DECAY_*).
"""

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map, tree_unflatten


class EMA:
    """Tracks a shadow copy of a model's parameters under an EMA update.

    ``update(model)`` folds the model's current params into the shadow.
    ``shadow_params()`` returns the EMA param tree (e.g. to load into an eval copy).
    """

    def __init__(self, model, decay):
        self.decay = decay
        # Deep-copy the current params as the initial shadow.
        self.shadow = tree_map(lambda p: mx.array(p), model.parameters())
        mx.eval(self.shadow)

    def update(self, model):
        d = self.decay
        params = model.parameters()
        self.shadow = tree_map(
            lambda e, p: d * e + (1.0 - d) * p, self.shadow, params
        )
        mx.eval(self.shadow)

    def shadow_params(self):
        return self.shadow

    def flat(self):
        """Flat {name: array} dict of the shadow params (for save/load)."""
        return dict(tree_flatten(self.shadow))

    def load_flat(self, flat):
        self.shadow = tree_unflatten(list(flat.items()))
        mx.eval(self.shadow)
