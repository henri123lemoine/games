# Byte-exact input-encoding spec — Ataraxos Stratego

Reference clone (file:line cites are into it): `…/scratchpad/ataraxos-ref`.
Companion: `ATARAXOS_SPEC.md` (rules, nets, training, search). This file is the encoder.

## 0. The "456" reconciliation (read first)
**456 appears nowhere in the code; the input is NOT 456 channels.** Code-true counts (final-run config):
- **Boardstate = exactly 355 channels.** `stratego_conf.h:59` `NUM_BOARD_STATE_CHANNELS=355`;
  `stratego.cu:2067` asserts `INFOSTATE_CHANNEL_DESCRIPTION.size()==355`.
- **Raw infostate** appends history (`stratego.cu:87`):
  `NUM_INFOSTATE_CHANNELS = 355 + move_memory*(4*enable_hidden_and_types + enable_src_dst + enable_dm)`.
  Training config (`stratego_conf.h:86-92`, confirmed by `pretrained/final_run/train.log`):
  `enable_src_dst_planes=true, enable_hidden_and_types_planes=false, enable_dm_planes=false,
  move_memory=32` → **387 = 355 + 32·1**.
- **Move-net per-token input** (`feature_orchestration.py:48-50`):
  `in_channels = sum(plane_mask) + plane_history_len + (N_PIECE_ID if use_piece_ids else 0)`.
  Move net `use_planes=True` → all 355 kept, `plane_history_len=32`, `N_PIECE_ID=256` → **643**.

So: **355 (board) → 387 (raw infostate incl. 32 history) → 643 (per-token incl. 256 piece-id one-hots).**
"456" is likely a paper-appendix count under a different config (e.g. `use_planes=False`). **Code to
643/387/355, not 456.** `N_PIECE_ID=256`, valid ids `[0,40]` ours, `[60,100]` theirs, `255` empty
(`constants.py:40-43`).

## 1. Memory / coordinate conventions
- Board 10×10 row-major `cell=10*row+col`; lakes `{42,43,46,47,52,53,56,57}` (`constants.py:53`); 92 occupiable.
- `Piece` (`stratego_board.h:44-68`, 16 bytes): `type:4` color:2(0=EMPTY,1=RED,2=BLUE,3=LAKE) visible:1
  has_moved:1; `piece_id`; eight 2-byte bitsets `threatened/evaded/actively_adjacent/protected_/
  protected_against/was_protected_by/was_protected_against` indexed by piece type
  (`field[type/8] & (1<<(type%8))`). `HIDDEN_PIECE=15` = the "unknown" bit (byte1 bit7).
- PieceType: SPY=0,SCOUT=1,MINER=2,SERGEANT=3,LIEUTENANT=4,CAPTAIN=5,MAJOR=6,COLONEL=7,GENERAL=8,
  MARSHAL=9,FLAG=10,BOMB=11,LAKE=12,EMPTY=13.
- `piece_id` = starting square [0,39] ours; tensors: ours 0–39, opp 60–99, empty 255.
- **POV = 180° point-reflection** for player 2: `pov_cell = 99 - cell` (`infostate_kernels.cu:258,286,…`)
  — NOT an independent row/col flip.
- Output tensor `(num_envs, NUM_INFOSTATE_CHANNELS, 10,10)`, channel-major, `out[channel*100 + pov_cell]`,
  `INFOSTATE_STRIDE = NUM_INFOSTATE_CHANNELS*100`. **Tensor zeroed first** (`stratego.cu:1252`) — cells not
  explicitly written stay 0.0.

## 2. Exact channel index ranges (each ×100 cells; [start,end] inclusive)
Order from `boardstate_channels.h` (`GenerateBoardstateChannelDescriptions`), mirrored `stratego.cu:2013-2066`,
every index confirmed against test asserts.

| Ch | N | Group | Kernel |
|---|---|---|---|
| 0–11 | 12 | `our_<piece>` one-hot (spy..bomb) | `infostate_kernels.cu:246-270` |
| 12–23 | 12 | `their_<piece>_prob` (opp posterior, my view) | `:272-380` shift 1200 |
| 24–35 | 12 | `our_<piece>_prob` (opp's posterior over me, rotated) | `:272-380` shift 2400 |
| 36 | 1 | `our_hidden_bool` | `:405` |
| 37 | 1 | `their_hidden_bool` | `:406` |
| 38 | 1 | `empty_bool` | `:407` |
| 39 | 1 | `our_moved_bool` | `:408` |
| 40 | 1 | `their_moved_bool` | `:409` |
| 41 | 1 | `max_num_moves_frac` (constant plane) | `:410` |
| 42 | 1 | `max_num_moves_between_attacks_frac` | `:411` |
| 43–53 | 11 | `we_threatened_<p\|unknown>` | `:437` `(43+i)` |
| 54–64 | 11 | `we_evaded_*` | `:438` |
| 65–75 | 11 | `we_actively_adj_*` | `:439` |
| 76–86 | 11 | `they_threatened_*` | `:440` |
| 87–97 | 11 | `they_evaded_*` | `:441` |
| 98–108 | 11 | `they_actively_adj_*` | `:442` |
| 109–119 | 11 | `our_dead_<spy..marshal,bomb>` | `:477` |
| 120–130 | 11 | `their_dead_*` | `:478` |
| 131–190 | 60 | `our_deathstatus_<reason>_<piece>` (6×10) | `:521` `(131+reason*10+ptype)` |
| 191–250 | 60 | `their_deathstatus_*` | `:540` `(191+…)` |
| 251–263 | 13 | `our_protected_<spy..bomb,empty,unknown>` | `:567` |
| 264–276 | 13 | `our_protected_against_*` | `:568` |
| 277–289 | 13 | `our_was_protected_by_*` | `:569` |
| 290–302 | 13 | `our_was_protected_against_*` | `:570` |
| 303–315 | 13 | `their_protected_*` | `:571` |
| 316–328 | 13 | `their_protected_against_*` | `:572` |
| 329–341 | 13 | `their_was_protected_by_*` | `:573` |
| 342–354 | 13 | `their_was_protected_against_*` | `:574` |
| 355–386 | 32 | `src_dst_cell[-32..-1]` move history | `stratego.cu:2072`; `infostate_kernels.cu:4-76` |

Boardstate total **355**. Index oracles: `test_attack_planes.py:181-439`, `test_cemetery.py:21,61`,
`test_protect_planes.py:352-598`.
`pieces_with_extras` (13 protection buckets) = `{spy,scout,miner,sergeant,lieutenant,captain,major,
colonel,general,marshal,bomb,empty,unknown}` → type values `{0..9, 11, 13, 15}` (`infostate_kernels.cu:563`).
**No flag plane** in protection or cemetery (cemetery 109–130 = 11 buckets `{spy..marshal,bomb}`; flag
death = terminal).

## 3. Exact per-group construction
### 3.1 `our_<piece>` (0–11), `BoardStateKernel__OwnPieceTypes` (`:246-270`)
Cell holding a piece of color `for_player` with `type<LAKE`: `out[100*type + pov] = 1.0`. (FLAG, BOMB
get planes here too → 12.)

### 3.2 "If-uniform-random" opponent posterior (12–23, 24–35), `BoardStateKernel__ProbTypes` (`:272-380`)
Analytic combinatorial belief, NOT the neural belief net. Invoked twice (`stratego.cu:1199,1202`):
12–23 = opponent's pieces in my POV (`for_player=to_play, rotate=false, shift 1200`); 24–35 = opponent's
posterior over my pieces, rotated back (`for_player=3-to_play, rotate=true, shift 2400`),
`pov = ((for_player==2) ^ rotate) ? 99-cell : cell`. Only opponent-relative cells with `type<LAKE` written
(`:299`). Counters (`stratego_board.h:131-132`): `num_hidden[t]` t∈[0,12), `num_hidden_unmoved`,
`total_num_hidden = Σ num_hidden[t]`, `denom = total_num_hidden − num_hidden[FLAG] − num_hidden[BOMB]`.
Three cases:
1. **Visible opp piece**: one-hot true type `out[100*piece.type+pov]=1.0`.
2. **Hidden & has_moved** (can't be FLAG/BOMB, `denom>0`): movables t∈{SPY..MARSHAL}
   `out[100*t+pov] = num_hidden[t]/denom`.
3. **Hidden & never moved**: `norm_factor = (num_hidden_unmoved − num_hidden[FLAG] − num_hidden[BOMB]) /
   (num_hidden_unmoved * denom)` (applied only if `total_num_hidden != num_hidden[FLAG]+num_hidden[BOMB]`);
   movables `out[100*t+pov] = num_hidden[t]*norm_factor`; always `out[100*FLAG+pov]=num_hidden[FLAG]/
   num_hidden_unmoved`, `out[100*BOMB+pov]=num_hidden[BOMB]/num_hidden_unmoved`.
   **All divisions f32; compute f32 then cast to storage dtype to match bf16 parity.** `denom==0` only when
   sole remaining hidden are FLAG/BOMB (movable writes guarded). Oracles `test_unknown_piece_counts.py`,
   `test_infostate.py`.

### 3.3 Basic state (36–42), `BoardStateKernel__InvisiblesEmptyAndMoved` (`:382-412`)
36 `our_hidden`=(!visible & color==for_player); 37 `their_hidden`; 38 `empty`=(type==EMPTY);
39 `our_moved`=(has_moved & ours); 40 `their_moved`; 41 `num_moves/max_num_moves` (constant plane);
42 `num_moves_since_last_attack/max_num_moves_between_attacks`. Training `max_num_moves=4000` (default 2000);
`max_num_moves_between_attacks` default **200** but not echoed in train.log — **confirm** (ch 42 denom).

### 3.4 Threat/Evade/Active-adj (43–108), `:414-444`
Types `{SPY..MARSHAL, HIDDEN_PIECE}` (11; FLAG/BOMB never threatened). `we_*` written only if piece is
**ours AND hidden**; `they_*` only if **theirs AND hidden**. Value `!!(piece.<field>[t/8] & (1<<(t%8)))`.
Semantics (Python trackers `test_attack_planes.py:57-168`): threatened = opp types this piece moved adjacent
to on its move (opp hidden → bit 10); evaded = alive opp types that moved adjacent last move then we moved
away (needs `is_adjacent(last_dst,src)`); actively_adjacent = opp types adjacent during its turn. Bitsets
**accumulated during env steps** (`action_kernels.cu`), reported at the piece's current cell.

### 3.5 Cemetery (109–130), `BoardStateKernel__Deaths` (`:446-480`)
**Per-square at the dead piece's STARTING cell (piece_id), one-hot by type.** Skip middle rows [40,60).
`rel = index<40 ? index : 99-index`. `is_dead = deaths[index>=60][rel/8] & (1<<(rel%8))`. **Type read from
`d_zero_boards`** (initial arrangement). For i∈{SPY..MARSHAL, BOMB}: our `out[(109+i)*100+pov]`,
their `out[(120+i)*100+pov]`. Flag masked. Oracle `test_cemetery.py`.

### 3.6 Death reasons / six causes (131–250), `:482-542`
6 reasons × 10 types (SPY..MARSHAL) × 2 sides. Reasons (`stratego_board.h:71-93`):
0 ATTACKED_VISIBLE_STRONGER, 1 ATTACKED_VISIBLE_TIE, 2 ATTACKED_HIDDEN, 3 VISIBLE_DEFENDED_WEAKER,
4 VISIBLE_DEFENDED_TIE, 5 HIDDEN_DEFENDED. Marked at the **death LOCATION** (`death_status[side][i].
death_location`, POV-mapped), not the starting square. our `(131+reason*10+ptype)`, their `(191+…)`.
`DeathStatus{is_dead:1,death_reason:3,piece_type:4,death_location:8}`. Oracles `test_battle_planes.py`,
`test_new_planes.py`.

### 3.7 Protection (251–354), `BoardStateKernel__Protections` (`:544-575`)
8 fields × 13 types. Nonzero only for **hidden** own (`our_*`) / **hidden** opp (`their_*`) pieces. Order:
four `our_` then four `their_` of `{protected_, protected_against, was_protected_by, was_protected_against}`.
**Port the 200-line `MyProtectTracker` (`test_protect_planes.py:132-339`) verbatim** — active protection
(`:212-250`), passive (`:290-339`), reveal/clear. Buckets: movables 0–9, bomb→BOMB, empty→EMPTY,
unknown→HIDDEN.

### 3.8 32 move-history planes (355–386), `src_dst_cell`, `InjectInfostateSrcDstKernel` (`:4-76`)
Plane i∈[0,32) ↔ `delta = move_memory − (idx/num_envs) ∈ [1,32]`; written at
`(idx/num_envs)*100`, so **oldest→newest, most recent at ch 386**. Per plane: reconstructed move sets
**src cell −1, dst cell +1**, else 0. Decode: `from_cell=action%100`; `direction=action>=900`;
`new_coord=(action/100)%9`; dst via skip-own-index `new_coord + (new_coord>=from_idx)`. **Parity flip**:
`requires_flip = delta%2` (moves alternate players, all rendered in current POV) → `pov = 9-idx` per row/col.
Boundary guards: skip if `delta>num_moves` or `delta<terminated_since`; read `action_history` if
`delta<=moves_since_reset` else `action_prehistory`. FeatureOrchestrator keeps the **last 32**
(`feature_orchestration.py:93-97`).
**Optional history planes (in code, OFF in final run):** `enable_hidden_and_types` → +4×32;
`enable_dm` → +32 DeepMind-style (src non-attack −1, else `−(2 + (type+1)/12)`; `test_dm_planes.py:46`).

## 4. Piece-id planes (256), `feature_orchestration.py:58,101-111`
`piece_ids (B,10,10)` (0–39 ours, 60–99 theirs, 255 empty) **one-hot to 256** via
`F.embedding(piece_ids, eye(256))`, permuted to channel dim, concatenated last. Encodes piece **identity**,
not type. Final move-net token = **355 + 32 + 256 = 643**; flatten + permute, drop 8 lake cells → `(B,92,643)`.

## 5. `LastMovesEncoder` / move_summary (BELIEF net only — not the 32 planes)
`TRAILING_LAST_MOVES_DIM=12`, `LAST_MOVES_DICTIONARY_SIZE=256`, `NOTMOVE_ID=254`. `encodings:(12,256,embed_dim)`,
input `(B,12)` → `Σ_d encodings[d, last_moves[:,d]]`. Belief net only. 6-byte `move_summary` =
`[src_cell, dst_cell, src_piece_enc, dst_piece_enc, src_piece_id, dst_piece_id]`; `encode_piece =
type_index + 16*visible + 32*immobile` (hidden-unknown `0b00001111`/`0b00101111`). NOT the move-net input.

## 6. Rust reproduction checklist (Metal/CPU)
1. Build 355 boardstate channels at the exact offsets in §2, then 32 `src_dst` history planes = 387.
   Zero-fill whole tensor first; write only cells each rule touches.
2. POV = 180° point-reflection (`99−cell`) for player 2, plus the `^rotate` XOR for ch 24–35, plus the
   per-move parity flip (`9−row`,`9−col` when `delta` odd) for history planes.
3. Posterior math in f32 with exact `denom`/`norm_factor` (§3.2); FLAG/BOMB use `num_hidden_unmoved`,
   movables use `denom`, never-moved movables scale by `norm_factor`. f32→storage cast to match bf16.
4. Cemetery = STARTING square (piece_id), type from the zero/initial board; death-reason = DEATH LOCATION.
5. Threat/evade/active-adj/protect = accumulated bitsets maintained during env steps; reported for
   **hidden** pieces at current cell (port `action_kernels.cu` + the Python test trackers).
6. Piece-id = 256-way one-hot of starting-slot identity (0–39/60–99/255), concatenated last → 643-dim
   tokens; drop 8 lake cells → 92 tokens.
7. Verify against repo tests (assert indices AND per-cell values): `test_attack_planes.py`,
   `test_protect_planes.py`, `test_cemetery.py`, `test_battle_planes.py`, `test_new_planes.py`,
   `test_infostate.py`, `test_unknown_piece_counts.py`, `test_far_action_history.py`, `test_dm_planes.py`.

## 7. Open flags
- (a) "456" not in code → use 643/387/355; reconcile against paper appendix, don't code to 456.
- (b) `max_num_moves_between_attacks` training value not echoed (only `max_num_moves:4000`); default 200 —
  confirm before fixing ch 42's denominator.
