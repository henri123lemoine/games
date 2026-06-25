# stratego-trainer (milestone 3)

The three Ataraxos Stratego transformers in **MLX (Python)**, scaled for tabula-rasa
training in ~1 day on the target **Apple M5 Max (64 GB, MLX 0.31.2)**. Faithful to
`games/stratego/ATARAXOS_SPEC.md` §3-§4 and `ENCODING_SPEC.md` §0, cross-checked
against the reference repo (`pyengine/networks/*`).

This milestone builds + unit-tests the nets against **synthetic** inputs; the live
Rust sim wires in at a later milestone. No Rust dependencies. MLX framework choice
is settled by `BENCHMARK.md` (MLX 1.6-2.4x faster than PyTorch-MPS for this net).

## Layout

```
stratego_nets/
  spec.py         shapes / channel widths / EMA decays (single source of truth)
  action_map.py   the (92x92) move-head -> 1800-action mapping (LogitConverter port)
  nets.py         MoveTransformer, ArrangementTransformer, BeliefTransformer + blocks
  ema.py          EMA shadow (decay 0.999 move/setup, 0.99 belief)
  checkpoint.py   safetensors save/load of weights + optimizer state + EMA
  config.py       sizing presets (default ~1-day budget; *_REF reproduce paper sizes)
tests/            pytest: shapes, mapping vs action.rs, train step, EMA, checkpoint
bench_step.py     synthetic fwd+bwd+AdamW+EMA step-time / peak-mem at b=1024/2048
BENCHMARK.md      the MLX-vs-PyTorch decision record (copied verbatim)
```

## Setup

```bash
cd ml/stratego-trainer
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest         # 26 tests
.venv/bin/python bench_step.py     # b=1024 step time + peak mem
```

## Net API

All nets are `mlx.nn.Module`s. Construct from a config preset or directly:

```python
import stratego_nets as S
move   = S.MoveTransformer.from_config(S.MoveConfig())          # ~5.06M
setup  = S.ArrangementTransformer.from_config(S.SetupConfig())  # ~3.18M
belief = S.BeliefTransformer.from_config(S.BeliefConfig())      # ~9.70M
# *_REF presets reproduce paper sizes: 14.78M / 12.65M / 57.18M
```

### MoveTransformer (encoder-only)
```python
out = move(obs, legal_mask=None)
#   obs        (B, 92, 643) f32          per-token sim obs (orchestrated, lakes dropped)
#   legal_mask (B, 1800) bool, optional  sim legal-action mask; illegal -> -inf
# returns:
#   out["move_logits"] (B, 1800)  env action logits; lake/illegal slots = dtype min
#   out["value_logp"]  (B, 3)     log-softmax over CATEGORICAL_AGGREGATION [-1,0,1]
```
Value token is prepended at index 0; the key-query head emits a (B,92,92) src->dst
grid mapped to the 1800-slot env action space (see "move-head mapping" below).

### ArrangementTransformer / setup (decoder-only, causal)
```python
import mlx.core as mx
pc  = mx.array(list(S.spec.CLASSIC_PIECE_COUNTS), dtype=mx.float32)  # (14,)
out = setup(seq, pc)
#   seq (B, T<=39, 14) one-hot placement prefix; a learned-zero start token is
#       prepended and the sequence truncated to 40 (each position predicts the next).
#   pc  (14,) remaining-type budget used to mask the placement head.
# returns per slot:
#   out["logits"]   (B, 40, 14)  next placement, illegal types -> -inf
#                                (0-budget types; flag restricted to the right half
#                                 under force_handedness)
#   out["value"]    (B, 40, 3)   W/L/D value
#   out["ent_pred"] (B, 40, 1)   conditional-entropy prediction
```
Placement order is bottom-to-top, left-to-right over the 4x10 home grid.
Handedness must be randomized post-generation (M5's loop), per the reference.

### BeliefTransformer (encoder-decoder, dropout 0.2)
```python
logits = belief(obs, unknown_pos_onehot, unknown_type_onehot)
#   obs                 (B, 92, 697) f32     belief-variant obs (86 history planes)
#   unknown_pos_onehot  (B, n_piece, 92)     one-hot cell of each unknown opp piece
#                                            (row-major; all-zero rows = padding)
#   unknown_type_onehot (B, n_piece, 14)     teacher-forcing target type one-hots
# returns:
#   logits (B, n_piece, 14)  per-hidden-piece type logits (autoregressive, causal)
```
Encoder embeddings at unknown-opponent-piece cells are gathered as decoder memory;
the decoder predicts each piece's type with a right-shifted (zero-start) teacher
forcing. Mask the per-piece CE loss to still-active pieces (see `bench_step.py`).

### EMA + checkpoints
```python
ema = S.EMA(move, decay=S.spec.EMA_DECAY_MOVE)   # 0.999 (0.99 for belief)
# each train step, after opt.update(net, grads):
ema.update(net)                                  # ema <- d*ema + (1-d)*param
S.save("ckpt.safetensors", net, optimizer=opt, ema=ema, metadata={"step": t})
S.load("ckpt.safetensors", net, optimizer=opt, ema=ema)   # round-trips all three
```

## Move-head -> 1800-action mapping

The move head's (B,92,92) grid is "piece at occupiable cell i -> occupiable cell j".
The Rust sim indexes actions in the **1800-slot src-displacement** space of
`games/stratego/src/action.rs`: `action = 100*c + src_pov`, `c in [0,18)`
(`c<9` vertical destination-row, `c>=9` horizontal destination-col, the source
row/col skipped). `action_map.create_srcdst_to_env_action_index` builds the gather
index that scatters the flattened 92*92 grid into the 1800 vector, reproducing the
reference `LogitConverter`. It matches `action.rs::from_abs` exactly:

```
env action index = c * 100 + src_cell, with
  c = new_row        if new_row < src_row else new_row - 1            (vertical)
  c = 9 + (new_col   if new_col < src_col else new_col - 1)           (horizontal)
```

1544 of the 1800 slots are representable; the 256 lake-touching slots are filled
with the dtype min (read as -inf by the softmax) before any legal mask. The
converter is POV-agnostic: the env index already encodes `src_pov`, and the sim
emits obs/masks in the acting player's POV, so the same converter aligns for both
players — exactly as the reference does. `tests/test_action_map.py` checks ten
known src/dst pairs against `action.rs`.

## Obs / array contract (what M5 must feed)

The Rust sim does the FeatureOrchestrator work (plane filtering, piece-id one-hot,
lake-cell drop) and hands the nets **already-tokenized** arrays. All arrays are
numpy / MLX with batch leading; obs are `float32`.

| feed | dtype | shape | notes |
|---|---|---|---|
| move obs | f32 | `(B, 92, 643)` | 355 board + 32 history + 256 piece-id, lakes dropped, acting-player POV |
| move legal_mask | bool | `(B, 1800)` | env action space; True = legal |
| setup seq | f32 | `(B, T<=39, 14)` | one-hot placements, bottom-left first; one-hot(14) per slot |
| setup piece_counts | f32 | `(14,)` | classic = `(1,8,5,4,4,4,3,2,1,1,1,6,0,0)` |
| belief obs | f32 | `(B, 92, 697)` | as move but 86 history planes |
| belief unknown_pos_onehot | f32 | `(B, n_piece, 92)` | row-major one-hot cell per unknown opp piece; zero rows pad |
| belief unknown_type_onehot | f32 | `(B, n_piece, 14)` | teacher-forcing targets (train); generate() samples at inference |

Token order is the 92 occupiable cells in row-major order with the 8 lakes
(`{42,43,46,47,52,53,56,57}`) removed. The 643/697 channel order is the
ENCODING_SPEC §2 board layout (355) + history + 256 piece-id one-hot, matching
`feature_orchestration.py`.

## Sizing

Defaults target the ~1-day / 64 GB budget; batch 1024-2048 (b=4096 is memory-bound
on 64 GB, see BENCHMARK.md). Measured on the **M5 Max (64 GB)**, bf16, fwd+bwd+AdamW+EMA:

| net | params | b=1024 | b=2048 |
|---|---|---|---|
| move   | 5.06M | 254 ms / 8.7 GB | 477 ms / 17.3 GB |
| setup  | 3.18M |  84 ms / 8.7 GB | 162 ms / 17.3 GB |
| belief | 9.70M | 366 ms / 13.2 GB | 703 ms / 25.4 GB |

Move-net b=1024 (254 ms) lands in the benchmark's predicted 190-450 ms band.
`config.py` exposes depth/width as constants for M5 to tune.
