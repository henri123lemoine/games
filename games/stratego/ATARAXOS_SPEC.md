# Ataraxos reimplementation — build bible

Goal: a **from-scratch, functionally-equivalent reimplementation of Ataraxos** (Sokota, Vinitsky,
Hu, Kolter, Farina — arXiv:2511.07312, "Superhuman AI for Stratego"), every optimization,
**trained tabula-rasa on an Apple M5 Max in ~1 day, no borrowed weights.**

Reference (MIT, © 2023 Gabriele Farina): `github.com/AtaraxosAI/stratego`.
Local clone for source-of-truth lookups (file:line citations below are into it):
`…/scratchpad/ataraxos-ref`. The paper has NO hyperparameter appendix — **the code is the
authoritative source for exact numbers**; the paper confirms shapes/algorithm only.

The encoder spec lives in `ENCODING_SPEC.md` (the 355/387/643-channel input). This file covers
rules engine, simulator, the three transformers, training loops, test-time search, setup phase,
and the paper↔code differences.

---

## 0. Target hardware & scaling (M5 Max, ~1 day)

Stack decision (matches the repo's proven Metal path — `ml/aztrainer`, `ml/ppo-core`,
`ml/slither-ppo`): **`tch` (libtorch 2.x) on `Device::Mps`** unless the MLX investigation shows a
decisive tensor-unit throughput win. New crates: `games/stratego` (rules, `game-core` only, reused
by sim + trainer + `lab`) and `ml/stratego-trainer` (standalone `[workspace]`, sibling to
`aztrainer`). Simulator = **rayon-parallel cheap-clone Rust CPU sim** (the GPU, not the sim, is the
bottleneck on this hardware — the opposite of their 16×H100 setup, so a ~1–5M updates/s CPU sim
keeps the MPS nets fed). bf16 autocast + **fp32 softmax/loss**; **bucket every batch to ×256**
(carries the repo's MPS shape-cache-leak fix from `aztrainer/net.rs:458-461`); one GPU thread owns
all forwards.

We are at ~1% of their FLOP budget (3072 H100-hours vs ~1 M5-Max-day). Keep **every algorithmic
optimization**; tune only size/quantity knobs: smaller nets (move ~3–8M, setup ~2–4M, belief
deferred or ~10M), 2k–8k parallel envs, ~0.5–2M games, ~100k–400k grad steps. Memory (128 GB
unified) is a non-constraint — spend it on batch width for MPS utilization.

---

## 1. RULES ENGINE + SIMULATOR

### 1.1 Board & state
- 10×10 = 100 cells, row-major `cell = 10*row + col`. 8 lakes at abs indices `{42,43,46,47,52,53,56,57}`
  (`constants.py:53`), baked in as `LAKE`-color pieces; **92 occupiable** (`constants.py:16`).
  Setup region 4 rows × 10 cols = 40/player; rows 0–3 red, 6–9 blue, rows 4–5 middle w/ lakes at
  cols 2,3,6,7 (`stratego_board.cu:119-136`).
- **`Piece` = 16 bytes** bitfield (`stratego_board.h:44-69`): `type:4` (SPY=0…MARSHAL=9, FLAG=10,
  BOMB=11, LAKE=12, EMPTY=13, HIDDEN_PIECE=15 sentinel), `color:2` (0=empty,1=red,2=blue,3=lake),
  `visible:1`, `has_moved:1`, `piece_id` (0–39, 0xff empty), and per-piece 2-byte **bitsets**
  `threatened[2]/evaded[2]/actively_adjacent[2]/protected_[2]/protected_against[2]/`
  `was_protected_by[2]/was_protected_against[2]` (the substrate for the threat/protection channels).
- **`StrategoBoard` = 1920 bytes, `alignas(128)`** (`stratego_board.h:128-145`): `Piece pieces[10][10]`
  (1600B); `num_hidden[2][12]`, `num_hidden_unmoved[2]`; `prev_dst_abs`, `prev_prev_dst_abs`,
  `last_moved_piece_type`; `deaths[2][5]` (bitset over piece_id of dead); `DeathStatus death_status[2][40]`.
  `static_assert(sizeof==1920)`. **Rust: `#[repr(C, align(128))]` + bitfield helpers; exact byte layout
  matters (games are snapshotted/serialized as raw bytes).**
- `DeathStatus` = 2 bytes (`stratego_board.h:95-104`): `is_dead:1, death_reason:3, piece_type:4,
  death_location:8`. `DeathReason` (`:71-93`) = the six causes-of-death (see ENCODING_SPEC §3.6).
- Ranking = numeric `type` 0..9, **higher wins** except specials. `CLASSIC_INITIAL_COUNTS =
  {1,8,5,4,4,4,3,2,1,1,1,6}` (spy..bomb, 40 total) (`stratego_board.h:165`).
- All rule logic is **integer (uint8/int32)**; float appears only in emitted feature tensors. Keep the
  Rust simulator fully integer; emit bf16 only at the NN boundary.

### 1.2 Move & attack resolution (`action_kernels.cu`)
- Action decode (`:129-147`): `NUM_ACTIONS=1800`, `c=action//100 ∈[0,17]`, `cell=action%100`.
  `c∈[0,8]`=vertical (dest row, skipping source row), `c∈[9,17]`=horizontal. Acting-player POV;
  blue mirrored `9-idx` to absolute. Converters: `ActionsToAbsCoordinates` /
  `AbsCoordinatesToActions` (`stratego.cu:2094-2182`).
- Legality (`LegalActionsMaskKernel:5-105`): source legal iff `type<FLAG` and `color==for_player`.
  Non-scouts ±1; **scouts** slide while next cell empty, capture at the stop, blocked by own/lakes.
- **Battle table** (`:333-338`):
  `to_wins (defender wins) iff (to.type<FLAG && to.type>from.type && !(to.type==MARSHAL &&
  from.type==SPY)) || (to.type==BOMB && from.type!=MINER)`; `tie iff to.type==from.type`; else
  `from wins`. Encodes all specials: **spy beats marshal only when attacking**; **miner defuses bomb**;
  attacking **flag** always wins → `d_flag_captured[env]=player` (`:699-702`).
- Reveal-on-attack (`:273-316`): mover sets `has_moved`; on attack both pieces become visible
  (decrement `num_hidden`); a ≥2-square scout move force-reveals.
- Captures update `deaths`/`death_status` per outcome (`:342-417`). **Protection channels**
  (`UPDATE_PROTECT:448-696`) = intricate 5-case cell geometry → must be reproduced geometry-for-geometry
  for byte-exact features (port the Python tracker oracle).
- Move summary (6 bytes/move, `:159-164`): `[src_rel, dst_rel, src_code, dst_code, src_id, dst_id]`,
  `code = type|(visible<<4)|(has_moved<<5)`; `29` = EMPTY-visible = the "non-attack" marker the chase
  rule keys on. Basis for the 32 history planes.

### 1.3 Continuous-chasing rule — exact state machine (`chase_state.cu/.h`)
`MAX_CHASE_LENGTH=210`. Per color/env: `last_dst_pos[2]`, `last_src_pos[2]` (0xee=missing),
`int32 chase_length[2]`.
- Update (`UpdateChaseStateKernel:28-96`): `p`=mover. `is_attack = (move_summary[+3] != 29)`.
  (1) record `last_dst/src_pos[p]`; (2) opponent chase: if `IS_ADJACENT(abs_src, last_dst_pos[~p])`
  then `chase_length[~p]++` else `=0`; (3) on attack: both → 0; (4) player chase: if `abs_dst`
  adjacent to any opponent piece, `chase_length[p]++` else `=0`.
- Illegal-move computation (`ComputeIllegalChaseMovesKernel:98-286`): for each `delta∈[1,chase_length)`,
  reconstruct the board `delta` moves ago from the circular history, diff vs current to find the single
  moved piece; if it's `p`'s piece on a straight-line move that **re-threatens** the opponent (dst
  adjacent to enemy) **and is not a simple revert of the immediately previous move**
  (`src==last_dst && dst==last_src` is allowed), that action is illegal → cleared from the mask.
  Plain English: **during a chase you may not reproduce an earlier threatening position, except you may
  undo your last move.** Two easy-to-miss carve-outs: "revert allowed" and "must-be-a-threat."

### 1.4 Two-square rule — exact state machine (`twosquare_state.cu/.h`)
State = last 4 cells of the tracked piece in **relative coords** `{A(newest),B,C,D(oldest)}`, 0xff=missing.
- Update (`:11-55`): chain continues iff `state.A==src_cell` AND same axis as previous move; then shift
  `D=C;C=B`, else reset `D=C=0xff`; always `A=dst;B=src`. D/C populate only when the **same piece
  oscillates in the same axis**.
- Death reset (`:57-76`). Triggered (`IsTwosquareRuleTriggered:88-98`): `D!=0xff` and the 4 cells form
  a strict single-axis zig-zag ("crossed the same border 3× in a row"). **Scouts**:
  `IsTwosquareRulePrecludingDirection` forbids only moves *past* the prior turning point, not the whole
  axis. Removal masks the forbidden destination(s) (scout: range beyond `min/max(COL(A),COL(C))`).
  Needs `move_memory≥6`.

### 1.5 K-move rule + termination/reward (`termination_kernels.cu`)
- Two counters per half-move: `num_moves` (total), `num_moves_since_last_attack` (reset on attack).
  **Final run: `max_num_moves=4000`, `max_num_moves_between_attacks=100`, `move_memory=86`**
  (`rl_main.py:50-52`; train.log shows 4000 — the 2000/200 defaults in `stratego_conf.h:43,47` are
  overridden). NOTE: ENCODING agent flagged `max_num_moves_between_attacks` not echoed in train.log —
  confirm (200 default vs 100). For the **move-net history** config `move_memory=32` (see ENCODING_SPEC §0).
- Terminal (`IncrementTerminationCounterKernel:141-171`) iff `has_legal_movement<3` (a player can't
  move) OR `flag_captured` OR `num_moves>max_num_moves` OR
  `num_moves_since_last_attack>max_num_moves_between_attacks` OR already terminal. `terminated_since`
  saturating; env auto-plays ~3 no-ops then auto-resets (fused, no host round-trip).
- **Reward for player 0** (`ComputeRewardPl0Kernel:173-223`), 0 unless terminal, timeout-rewind: if
  `not_timeout && flag_captured`: **+1 red captured / −1 blue captured**; elif `not_timeout &&
  terminated`: `x=has_legal_movement` → **0 (both stuck), +1 (only red moves), −1 (only blue)**; else
  (timeout) **0**. Python decomposes into `is_flag_capture/is_wipe_out/is_battle_timeout/
  is_gamelen_timeout/is_kamikaze` (`env.py:272-305`).

### 1.6 GPU-resident simulator (paper §2.6: ~10M updates/s on H100) → Apple Silicon
- All state is `(buf_size, num_envs, …)` device tensors; `current_step_` indexes mod `buf_size`
  (circular). One thread/env per kernel; **no warp shuffles / shared mem / atomics** — flat per-env maps.
- **The replay buffer IS the circular board buffer**; trajectories are compact 1920-byte boards;
  **history reconstructed on demand** (`SnapshotState`/`SnapshotEnvHistory`, `stratego.cu:1649-1980`)
  walking the ring via `num_moves`/`num_moves_since_reset` + reset prehistory. `ComputeInfostateTensor`
  **recomputes** the 355-ch tensor per query rather than storing it.
- Per iter: `traj_len_per_player=101` → `2*101+2` rows/env; `num_envs=1600`.
- **Rust port:** rayon par-iter over envs; `Vec`-backed ring of `repr(C,align(128))` 1920-byte boards;
  `copy_from_slice` for D2D roll-forward; integer history reconstruction ported exactly (ring ≥210 for
  chase); native `u128` for the arrangement ranking (`stratego_board.h:172-219`). Correctness hinges on
  **branch-logic fidelity** (battle table, protection geometry, chase diff, two-square zig-zag), not GPU
  concurrency. Match the RNG stream order for determinism.

---

## 2. INPUT ENCODING
See `ENCODING_SPEC.md` (full byte-exact spec: 355 board planes → 387 raw infostate (+32 history) →
643 per-token move-net input (+256 piece-id one-hots); the combinatorial posterior math; the
threat/evade/protection trackers; the 180° POV + parity-flip conventions; the reference test oracles).

---

## 3. THE THREE TRANSFORMERS

Shared (`transformer_basics.py`): MHSA (separate q/k/v/out Linears, SDPA, dropout_p=0 inside).
**Pre-LN** block: `x = x + drop(mha(LN1(x))); x = x + drop(ff(LN2(x)))`, `ff = lin2(relu(lin1(x)))`.
**Activation = ReLU**; `ff_factor=4`; dropout residual-only; `norm_out` after the stack. `DecoderBlock`
= causal self-attn layer + cross-attn layer (each with own FF → 4 residual sub-blocks). `Linear
bias=True`, `LayerNorm eps=1e-5`. All positional/temporal embeddings learned absolute,
`trunc_normal_(std=0.1)`.

- **MOVE net** `MoveTransformer` (~14.7M): `depth=8, n_head=8, embed_dim=384` (head_dim 48), FF=1536,
  dropout=0, `plane_history_len=32`, `use_cat_vf=True`. `embedder=Linear(643→384)`; prepend 1 learned
  **value token** → 93 tokens; learned pos-emb `(1,93,384)`. **Encoder-only, bidirectional.**
  **Key-query move head** (`make_action_logits:136-144`): `q_proj,k_proj=Linear(384,384)`;
  `attn=(q@kᵀ)/sqrt(384)` → `(B,92,92)` logit "piece at i → cell j"; `LogitConverter` (`utils.py:85-110`)
  maps 92×92 → the 1800-action src-displacement param (lake moves excluded); illegal → `finfo.min`;
  `Categorical(logits)`. **Value head** `Linear(384,3)` on value token, `log_softmax` over
  `CATEGORICAL_AGGREGATION=[-1,0,1]` (lose/tie/win).
- **SETUP net** `ArrangementTransformer` (~12.6M): `depth=4, n_head=8, embed_dim=512` (head_dim 64),
  FF=2048, dropout=0, `force_handedness=True`. **Decoder-only, causal.** token = one-hot over 14 piece
  types per slot; `embedder=Linear(14→512)`; learned zero start-token, truncated to 40; pos-emb
  `(1,40,512)`. **Three heads**: next-placement `Linear(512,14)` (legal-type masked); W/L/D value
  `Linear(512,3)`; conditional-entropy `Linear(512,1)`.
- **BELIEF net** `BeliefTransformer` (~57.1M): `n_encoder_layer=6, n_decoder_block=6, num_head=8,
  embed_dim=512` (head_dim 64), FF=2048, **dropout=0.2** (only net with dropout), `plane_history_len=86`.
  Encoder: FeatureOrchestrator `in_channels=697`, 92 cell tokens, bidirectional, pos-emb `(92,512)`.
  **Encoder-token filtering** (`extract_mem`, `utils.py:113-126`): keep only embeddings at squares with
  **unknown opponent pieces**, row-major → decoder memory. Decoder: predicts each unknown piece's type
  autoregressively (teacher forcing); 6 `DecoderBlock`s; `final_linear=Linear(512,14)`; `generate()`
  samples respecting per-type counts + movability.

EMA (`exponential_weighted_average.py`): `ema ← decay·ema + (1-decay)·orig`. Move/setup **0.999**
(`rl.py:53`), belief **0.99** (`belief.py:43`). Checkpoints ship `.pthw` (working) + `.pthm` (EMA/magnet).

Rust/Metal notes: code forces `SDPBackend.EFFICIENT_ATTENTION` — implement standard SDPA (numerically
equiv). `torch_compile` is perf-only. `BeliefTransformer.generate()` has a 2-arg bug (`:139-141`);
follow the `sampling_loop` contract. `TemporalBeliefTransformer.forward_sequential` is an empty stub.

---

## 4. TRAINING LOOPS — exact losses, schedules, hyperparameters

### 4.1 Move-RL (`pyengine/core/rl.py`, `core/buffer.py`) — RLConfig defaults (`rl.py:47-107`)
| Param | Value | | Param | Value |
|---|---|---|---|---|
| `clip_range` (PPO ε) | 0.2 | | `kl_coef` (rev-KL to data policy) | 0.1 |
| `ema_decay` | 0.999 | | `adv_filt_rate` (top-quantile) | 0.75 |
| `td_lambda` (value λ) | **0.8** | | `adv_filt_thresh` (abs floor) | 0.01 |
| `gae_lambda` (advantage λ) | **0.5** | | `lr_coef` | 0.5 |
| `vf_coef` | 1.0 | | `lr_decay` (exp) | 1.1 |
| `policy_coef` | 1.0 | | `lr_ceil`/`lr_floor` | 1e-4 / 5e-6 |
| `temperature_coef` (magnet) | 0.05 | | `weight_decay` | 0.0 |
| `temperature_decay` (exp) | 0.3 | | `max_grad_norm` | 0.267 |
| `temperature_ceil`/`floor` | 0.1 / 0.001 | | `dtype` | bfloat16 |
| `num_envs` | 1600 | | `train_every_per_player` | 101 |

λ-returns (`buffer.py:194-247,351-357`): no γ beyond λ (finite-horizon, undiscounted);
`returns = segmented_discounted_cumsum(δ, td_lambda·~terminal) + values` (td_lambda **0.8**);
`advantages = segmented_discounted_cumsum(scalar_δ, gae_lambda·~terminal)` (gae_lambda **0.5**) — distinct
λ. Value target = same player's prediction two plies later, or one-hot terminal reward.

Loss (`rl.py:547-579`):
```
ratio = exp(logp - old_logp)
policy_loss = -min(adv·ratio, adv·clamp(ratio, 1±0.2)).mean()        # PPO-clip
kl_loss     = (probs·(logp - data_logp)).sum(-1).mean()              # rev-KL to data policy
value_loss  = -(returns · log_softmax(value_pred)).sum(-1).mean()    # categorical W/L/D CE
magnet_kl   = (xe_to_magnet - entropy).mean()                        # rev-KL to magnet
loss = 1.0·policy_loss + temperature(t)·magnet_kl + 1.0·value_loss + 0.1·kl_loss
```
clip_grad_norm_ 0.267 → AdamW.step() → EMA update.

Magnet = `legal/legal.sum()` flat-uniform (`uniform_magnet=True` default). **Note: paper describes the
per-piece-then-per-move uniform** (`get_weighted_uniform_policy`, the `uniform_magnet=False` branch).
Shipped default differs from paper — choose deliberately.

Advantage filtering (`buffer.py:233-241`): keep iff `|adv| ≥ max(quantile(|adv|,0.75), 0.01)` (~top 25%,
abs floor 0.01). Paper §2.3: ~2.5× wall-clock cut, *increased* sample efficiency.

**Dynamic damping** — the headline. `power_schedule(coef,step,decay,ceil,floor) =
clip(coef/(step+1)^decay, floor, ceil)` (`rl.py:756-760`), per iteration:
- LR = `clip(0.5/(t+1)^1.1, 5e-6, 1e-4)` (starts at ceiling, anneals ~t≈55).
- Magnet-KL coef = `clip(0.05/(t+1)^0.3, 0.001, 0.1)`.
- The **other 3 of the paper's "four update-size mechanisms" are CONSTANTS in code**: PPO-clip 0.2,
  grad-norm 0.267, data-policy-KL 0.1. (`RLResumeConfig` exposes these for manual re-tuning across
  resume segments — so they were hand-bumped between segments, not scheduled.)

Optimizer AdamW (`train_container.py:34`), 2 param groups (no WD on bias), betas/eps assumed PyTorch
defaults (0.9,0.999,1e-8 — not overridden; verify if exactness critical). **1 pass/iter**. ~200 env-steps
× 1600 envs ≈ **320k transitions/iter**, ~25% trained. Shipped run = **42,400 iters**.

### 4.2 Setup loop (co-trained in `rl.py`; `arrangement/buffer.py`)
**Pure MC**: `arr_td_lambda=1.0, arr_gae_lambda=1.0`, no filtering. Advantage = MC-outcome-adv +
`reg_temp·entropy-adv` (`reg_norm=10.0`). Three head losses (`rl.py:616-702`, 5 epochs, batch 1024):
PPO-clip policy (`arr_clip=0.2`); W/L/D value CE; conditional-entropy MSE; rev-KL to data policy. Coefs
`arr_policy_coef=1.0, arr_ent_pred_coef=1.0, arr_vf_coef=0.5, arr_kl_coef=0.1`; grad-norm 0.5; AdamW
**constant lr 5e-5**. Setup temperature `clip(0.1/(t+1)^0.3, 0.001, 1.0)`. `n_arr=1024` setups
regenerated **every iteration**; reset-distribution swapped each iter; `ArrangementBuffer` age-filtered
(`storage_duration=2·101+max_num_moves`), deduped by combinatorial id; EMA setup policy checkpointed.
`remember_main.py` is **dead code** — the setup net is co-trained inside the move-RL loop, not separately.

### 4.3 Belief loop (`belief_main.py`, `belief/belief.py`)
**Adam** (not AdamW), constant lr **5e-5**, grad-norm 0.5, EMA **0.99**, dropout 0.2, num_envs 1024.
Loss = autoregressive per-piece CE of predicted type vs true hidden type, masked to **active (still-hidden)
pieces only** (`belief.py:231,279-291`). Data from **frozen** trained move+setup nets. Trains *after* RL.

---

## 5. TEST-TIME SEARCH (`pyengine/core/search.py`)
**Not MCTS — no tree, no iteration loop.** One call = sample beliefs → shallow rollouts → **one
closed-form MMD update** → sample one action.
- Constructor (`:66-113`): `depth` even (loop runs `depth//2` pairs); `num_envs≥100`;
  `categorical_aggregation=[-1,0,1]`.
- Belief-sample count (`:129`): `n_sample = min(num_envs // L, max_num_samples)`, `L`=#legal actions.
  Each action tried in ≈`num_envs/L` worlds (`sample_deterministic(uniform,num_envs)`).
- Belief sampling (`:281-370`): `belief_model.generate(...)` (count+movability-masked); flavors
  MARGINALIZED_UNIFORM (reads the `their_*_prob` channels), UNIFORM (combinatorial), PLANAR/TEMPORAL.
  Sampled types written via `assign_opponent_hidden_pieces`. `belief_model=None` → ground-truth
  ("perfect search").
- Rollouts (`:180-279`): `num_envs` total, `depth` plies, **both players act with the move net**,
  sampling from its policy (not argmax). λ-return leaf target **td_lambda=1.0**: terminal reward if ended,
  else bootstrap with the value head only at final depth.
- q̂ (`:234-272`): categorical value head softmaxed, terminal one-hot; accumulate per root action via
  `scatter_add_`, `q=cum/counts`, **`scalar_q = q @ [-1,0,1]`**.
- **MMD closed form** (`compute_search_policy:431-454`):
  `π_search ∝ exp((log π_bp + α·q̂ + (α·τ)·log π_magnet) / (1 + α·τ))`, α=stepsize, τ=temperature,
  π_bp = move-net behavior policy, π_magnet = `get_weighted_uniform_policy`. `uniform_magnet=True` drops
  the magnet term: `exp((log π_bp + α·q̂)/(1+α·τ))`. **Exactly one MMD step.** Then `Categorical(π_search).sample()`.
- Eval-script defaults: **depth=10, α=10, τ≈0.001, td_lambda=1.0, max_num_samples=100–200, num_envs≈1024,
  bf16**. Paper §2.6 deployed eval = **40-ply, 1000-rollout, ~1.26s/move** (heavier than the script default).
  Ablation grid `(d,k)∈{(10,200),(10,1000),(40,200),(40,1000)}`.
- Stale: `belief/naive.py` (beta-belief baseline) ABSENT; `eval_search_vs_search.py` passes nonexistent
  `iterations=`. `test_search.py:14-187` is the Q-estimator oracle (rollout returns match an independent
  `CircularBuffer.process_data` to 1e-6) — reproduce to validate.

---

## 6. SETUP-PHASE SPECIFICS
- Placement order (`arrangement_transformer.py:31-40,93-124`): **square-by-square, bottom-to-top,
  left-to-right** (row-major over the 4×10 home grid; first slot = bottom-left). One type token/square;
  legal-mask zeros 0-count types and (under `force_handedness`) the flag off non-right squares.
- Mirror randomization: `force_handedness=True` forces the flag to the **right half** during generation,
  then **~50% of samples are flipped** (`arrangement/sampling.py:102-107`; `flip_arrangements` =
  `.flip(-2)` across the middle column). `needs_flip` carried into the buffer to reconcile
  network-orientation log-probs vs env-orientation arrangements.
- Pooling: `n_arr=1024` regenerated **every RL iteration** from the live actor; env reset-distribution
  swapped (even→red, odd→blue); `ArrangementBuffer` variable-size, age-filtered, deduped by combinatorial
  id; running-mean MC outcome per setup; EMA-net setups checkpointed.
- Board mapping (`stratego_board.cu:84-149`): rows 0–3 red, 6–9 blue, lakes cols 2,3,6,7 in rows 4–5.
  `piece_id` POV `i*10+j` red, `99-(i*10+j)` blue. **Arrangement string** = 40 chars `A`–`M`
  (`A=EMPTY,B=BOMB,C=SPY,D=SCOUT,E=MINER,F=SERGEANT,G=LIEUTENANT,H=CAPTAIN,I=MAJOR,J=COLONEL,K=GENERAL,
  L=MARSHAL,M=FLAG`) — **differs from Python `PIECE_INDICES`**; implement both + the bijection.
  Classic = 40 pieces, barrage = 8 + 32 empties (sum 40). `is_terminal_arrangement` filters trivially-lost setups.

---

## 7. DIFFERENCES FROM PAPER + CUDA TO RE-ENGINEER
1. **"456 channels" is not in the code** — code-true is 355 board (+32 history +256 piece-id = 643). Target the code.
2. Of the four damping mechanisms, **only LR is power-law annealed**; PPO-clip (0.2), grad-norm (0.267),
   data-policy-KL (0.1) are constants. The magnet regularizer is separately annealed. Plan for manual
   re-tuning across segments (`RLResumeConfig` hooks).
3. **Default magnet is flat-uniform, not the paper's per-piece uniform** (`uniform_magnet=True`).
4. **Setup uses pure MC (λ=1.0, no filtering)**; move uses λ=0.8/0.5 + top-25% filtering. Don't cross-apply.
5. Move RL does **1 pass/iter** (not multi-epoch PPO); setup 5 epochs; belief straight supervised CE.
6. Eval scripts default to **depth-10**; paper's 40-ply/1000-rollout is the heavier deployed setting.
7. **Stale/missing files**: `remember_main.py`, `core/remember.py`, `networks/rl_planar.py`,
   `rl_temporal.py`, `belief/naive.py` absent/broken — the repo is a partial de-synced dump. Authoritative
   runnable files: `rl.py`, `search.py`, `belief/{sampling,uniform,masking}.py`, `arrangement/*`, the
   `*_transformer.py` clean family, the C++ env.
8. Adam betas/eps assumed PyTorch defaults (verify if exactness critical).

**CUDA to re-engineer:** the whole simulator (per-env kernels, fused circular replay buffer, on-demand
history reconstruction, `assign_opponent_hidden_pieces`, `snapshot_env_history`, 128-bit arrangement
ranking). All sim logic is integer, embarrassingly parallel, no atomics/shuffles → CPU rayon (or Metal);
correctness is branch-logic fidelity. Nets are 100% standard torch ops (no custom kernels) → tch-on-MPS
(or MLX). Search/training tensor ops (`scatter_add_`, `bincount`, `softmax`, `Categorical`) all have
MPS/CPU equivalents. Match RNG stream order for determinism.

**Verification oracles** (reference `tests/`): `test_*_planes.py`, `test_cemetery.py`, `test_infostate.py`,
`test_unknown_piece_counts.py`, `test_move_summary.py`, `test_search.py` — port as Rust tests.
