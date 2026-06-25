"""Run directory: metrics.jsonl, periodic safetensors checkpoints, STOP sentinel,
work-seconds budget, keep-best — the operational contract from `aztrainer/rundir.rs`
and `ml/slither-ppo`, reproduced in Python.

Layout (`runs/<name>/`):
  metrics.jsonl       one JSON object per iteration (losses, schedules, eval, counts)
  latest.safetensors  most-recent working + EMA + optimizer checkpoint (move + setup)
  ckpt_<step>.safetensors   periodic snapshots
  best.safetensors    best-by-eval EMA snapshot (keep-best)
  STOP                touch this file to request a clean stop at the next iter boundary
"""

import json
import os
import time


class RunDir:
    def __init__(self, root: str, name: str, work_seconds: float = 0.0):
        self.path = os.path.join(root, name)
        os.makedirs(self.path, exist_ok=True)
        self.metrics_path = os.path.join(self.path, "metrics.jsonl")
        self.stop_path = os.path.join(self.path, "STOP")
        self.work_seconds = work_seconds
        self.start = time.time()
        self.best_eval = float("-inf")
        # Truncate metrics on a fresh run so a re-run does not append to stale curves.
        open(self.metrics_path, "w").close()

    def elapsed(self) -> float:
        return time.time() - self.start

    def should_stop(self) -> bool:
        if os.path.exists(self.stop_path):
            return True
        return self.work_seconds > 0 and self.elapsed() >= self.work_seconds

    def log(self, record: dict) -> None:
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()

    def _ckpt(self, fname, move, move_opt, move_ema, setup, setup_opt, setup_ema, step):
        """One safetensors file holding both nets' working + EMA + optimizer state."""
        path = os.path.join(self.path, fname)
        # Save with prefixed keys so both nets coexist in one file. We reuse the
        # net checkpoint flatten convention but under move_*/setup_* namespaces.
        import mlx.core as mx
        from mlx.utils import tree_flatten

        flat = {}

        def add(prefix, tree):
            for k, v in tree_flatten(tree):
                flat[f"{prefix}.{k}"] = v

        add("move.model", move.parameters())
        add("move.opt", move_opt.state)
        add("move.ema", move_ema.shadow_params())
        add("setup.model", setup.parameters())
        add("setup.opt", setup_opt.state)
        add("setup.ema", setup_ema.shadow_params())
        mx.eval(flat)
        mx.save_safetensors(path, flat, metadata={"step": str(step)})
        return path

    def save_latest(self, *args, step):
        return self._ckpt("latest.safetensors", *args, step=step)

    def save_periodic(self, *args, step):
        self._ckpt(f"ckpt_{step}.safetensors", *args, step=step)
        return self.save_latest(*args, step=step)

    def maybe_save_best(self, eval_score, *args, step):
        if eval_score > self.best_eval:
            self.best_eval = eval_score
            self._ckpt("best.safetensors", *args, step=step)
            return True
        return False


def load_checkpoint(path, move=None, setup=None, move_opt=None, setup_opt=None,
                    move_ema=None, setup_ema=None, prefer_ema=False):
    """Load a RunDir checkpoint into the given nets/optimizers/EMAs (in place).

    `prefer_ema=True` loads the EMA shadow into the live net params too (for
    evaluating the magnet/EMA policy from a reloaded checkpoint).
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    flat = mx.load(path)

    def collect(prefix):
        plen = len(prefix) + 1
        sub = {k[plen:]: v for k, v in flat.items() if k.startswith(prefix + ".")}
        return tree_unflatten(list(sub.items())) if sub else None

    for net, opt, ema, ns in (
        (move, move_opt, move_ema, "move"),
        (setup, setup_opt, setup_ema, "setup"),
    ):
        if net is None:
            continue
        params = collect(f"{ns}.ema") if prefer_ema else collect(f"{ns}.model")
        if params is not None:
            net.update(params)
        if opt is not None:
            st = collect(f"{ns}.opt")
            if st is not None:
                opt.state = st
        if ema is not None:
            sh = collect(f"{ns}.ema")
            if sh is not None:
                ema.load_flat(dict(tree_flatten(sh)))
        mx.eval(net.parameters())
    return flat
