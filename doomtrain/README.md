# doomtrain — DFP/GRU trainer for the doomrl 1v1 deathmatch substrate

M3 of "RL on Doom": a reinforcement-learning trainer over the `doomrl` 1v1
deathmatch substrate (M1/M2). Standalone `tch` crate, like `azt`/`azgo` — its
own `[workspace]` so `libtorch` never touches the root `cargo test`.

## STRATEGIC REDESIGN 2026-06-20 (branch `doom-strategic`) — RETRAIN REQUIRED

The substrate, map, obs/action space and reward were rebuilt for *strategic* 1v1
deathmatch (AltDeath item economy). The previous bot CANNOT play this — its obs
had no item/map info and its action space could not strafe or switch weapons. The
old checkpoints are stale; a full retrain on the new env is required.

What changed (see `doomrl/STRATEGIC_CONTRACT.md` for the authoritative spec):
- Map: the asymmetric **dumbbell** arena (`doomrl/assets/dumbbell.wad`, built by
  `tools/make_arena_wad.py` + a real BSP node builder `tools/nodebuild.py`) —
  two pockets + central hall, a sunken rocket-launcher altar, a raised megaarmor
  ledge, a soulsphere, two LOS-breaking cover blocks, a choke.
- Match runs **`-altdeath`**: items respawn on 30 s timers.
- `OBS_DIM = 40`: self economy (armor type, all ammo, ready-weapon, position) +
  opponent + 3 key items (available / respawn / bearing-by-distance).
- `NUM_ACTIONS = 486`: 9 turn × 3 forward × **3 strafe** × 2 fire × **3 weapon**.
- Reward: item-control shaping (grab/hold rocket, megaarmor, soulsphere) kept
  well below the +5 frag.

### Launch the full retrain (single command)

```bash
cd doomtrain
DOOMTRAIN_MPS=1 ./run.sh ppo \
    --iters=500 --steps=1024 --bc-iters=200 --self-play-at=0.6 \
    --save=runs/strategic.ot --best=runs/strategic_best.ot
# then export the deploy weights (round-trip verified):
./run.sh ppo-export --net=runs/strategic_best.ot --out=doomppo_strategic.bin
```

This runs BC warmstart (clone the item-seeking scripted hunter) → curriculum
(spawn distance + opponent skill ramp) → self-play, eval-gated keep-best, on the
strategic env. A tiny CPU smoke of this exact entrypoint runs clean (finite
losses, correct 40/486 shapes); strategic *play quality* requires the full GPU
run (not yet done).

## STATUS — WORKING BOT 2026-06-20 (branch `doomrl-m1-substrate`, not merged)

There is now a **genuinely winning 1v1 deathmatch bot**, trained with the
research blueprint's proven recipe (PPO self-play + BC warmstart + curriculum).
Run it with the `ppo` subcommand.

**The recipe (what made it work):**
1. **PPO** actor-critic on the GRU trunk (`ppo_net.rs`, `ppo.rs`): GAE(γ.99,
   λ.95) + clipped surrogate (ε.2) + value + entropy, BPTT over 32-step windows.
2. **Curriculum** (`env.rs`, `main.rs`): a **BeatableBot** (the scripted hunter
   weakened with aim noise + reaction delay + fire-prob, skill 0→1); **spawn-near**
   (`doomrl_dm_spawn_near`, held close for the first 60% so kills happen before
   navigation matters, then widening); **Arnold-style shaped reward** (+5 frag,
   −death/−suicide, +damage, +small dist-moved anti-camp); a finer **54-action**
   space (9 turns) so aim can track; ramp to a frozen **self-play** snapshot.
3. **BC warmstart + BC anchor** (the keystone): undirected RL can't land the 7+
   accurate shots a kill needs, so it never frags and never reinforces +frag —
   the cold-start that pinned `frag_share` at 0 for *both* DFP and raw PPO. Fix:
   clone the scripted hunter's aim (supervised CE) **before** PPO, and keep a
   small DAgger-style **BC-anchor** CE term in the PPO loss so RL refines rather
   than drifts off the cloned aim.

**Results (live MPS run, eval-gated keep-best):**
- `frag_share` vs the beatable curriculum bot: BC 0.33 → PPO **~0.70–0.73**,
  ~50–60 net frags per 12-episode eval (winning ~2.5:1), **holding** (no
  collapse).
- The keep-best checkpoint (`runs/.../best.ot`), still mid-training, already
  scores **0.73 vs the mid bot and ~0.47 vs the PERFECT-aim hunter** (nearly
  even with flawless aim — the benchmark DFP could never beat, where it scored 0).

**Commands:**
```bash
./run.sh ppo --bc-iters=200 --iters=500 --steps=1024 --self-play-at=0.6 \
             --save=run/final.ot --best=run/best.ot     # train (DOOMTRAIN_MPS=1 for Metal)
./run.sh ppo-eval  --net=run/best.ot --eval-skill=1.0   # benchmark vs the perfect hunter
./run.sh ppo-export --net=run/best.ot --out=doom.bin    # portable weights (round-trip verified)
```

**M4 (WASM deploy):** still needs an **`emsdk` install** (`emcc` not on PATH).
The `ppo-export` flat file (`DOOMDFP1` magic, no-tch read) is the bridge: a
browser forward loads it and drives a `doomrl` WASM build's ticcmd.

Full run findings + recipe are in memory `doom-rl-training`. (The earlier DFP
path — `smoke`/`train`/`eval`/`export` subcommands — is retained but superseded;
its diagnostics are documented in the memory file.)

---

(Original M3 design notes below.)

## Approach (per the M3 blueprint)

- **Substrate**: links `../doomrl/build/libdoomrl.a` via FFI (`build.rs` runs
  `doomrl/build.sh`). Drives both deathmatch seats from one deterministic
  process (`doomrl_dm_step`), reads per-seat `P_CheckSight`-gated state, loops
  episodes with `doomrl_reset`.
- **Observation** (`OBS_DIM=18`, egocentric, normalized): own health, armor,
  ammo, facing (sin/cos), momentum; opponent visible flag, bearing (sin/cos),
  distance, relative velocity, health; and the last-seen memory block (valid,
  ticks-since-seen, last bearing sin/cos). No omniscience — only what
  `P_CheckSight` allows.
- **Memory**: a **GRU** (`GRU_HIDDEN=128`) carries hidden state across steps for
  partial observability (DRQN-style), so the policy can act on a target that has
  left the field of view.
- **Action**: a flattened multi-discrete set — turn ∈ 5 rates × forward/back ∈ 3
  × fire ∈ 2 = **30 actions** (`env::decode_action`), each mapped straight onto
  a `doomrl` ticcmd.
- **Objective — DFP** (Direct Future Prediction): predict the *future change* of
  the measurement vector `[health, ammo, frags]` at offsets {1,2,4,8,16,32},
  per action, via a dueling expectation+advantage head. Action = argmax over
  actions of `goal · predicted-future`. **No hand-designed reward shaping** —
  the measurements are the goal, exactly as the brief specifies. The substrate's
  scalar `reward` is still exposed (frag ±1 + health/armor shaping) for an
  APPO/PPO path later, but DFP does not use it.

## The map (why a custom arena)

DFP needs frags to actually happen or there is no learning signal. The shareware
maps are poor for 1v1 DM: I surveyed E1M1–E1M9 deathmatch-start floor-height
spread — E1M5 is flattest (104 units) but has bad sightlines; E1M1 spreads 160
and its DM starts sit on platforms 96–144 units apart vertically, so the
pistol's auto-aim misses and naive agents barely frag. So `doomrl` ships a
generated **flat single-room arena**, `doomrl/assets/flatarena.wad`
(`doomrl/tools/make_arena_wad.py`): one convex sector, four DM starts at the
same floor height, hand-built minimal BSP/blockmap/reject. On it a scripted
hunter baseline hits **~65 frags/sec with 100% mutual visibility** (vs 0 on the
shareware maps) — the clean signal DFP wants. Loaded via `-file` (the shareware
"cannot -file" guard is gated behind `DOOMRL_ALLOW_FILE`, set automatically).

## Build & run

`tch`'s `download-libtorch` drops the dylibs in the build dir without an rpath
(same as `azt`), so use the wrapper, which sets `DYLD_LIBRARY_PATH`:

```bash
cd doomtrain
./run.sh smoke                                   # BPTT smoke: 3 eps, asserts sanity
./run.sh train --iters=20 --steps=256 --eval-every=5 --save=doomdfp.ot
./run.sh eval  --net=doomdfp.ot --episodes=10    # greedy DFP vs the scripted hunter
./run.sh export --net=doomdfp.ot --out=doomdfp.bin   # portable weights + round-trip check
```

Subcommands:

- **`smoke`** — collects episodes, fills the replay buffer with BPTT chunks, runs
  `train_chunk` updates, and asserts shape / finiteness / non-constant-obs. Prints
  `bptt_loss` (falling), `chunks_in_replay`, `obs_std`, and the action histogram.
- **`train`** — the full loop: per iteration, collect an episode (epsilon decays
  `--eps-start`→`--eps-end`), push BPTT chunks into a capped replay buffer, run
  `--updates` BPTT minibatch updates of `--batch` chunks, and every
  `--eval-every` iters run an eval. Saves a tch checkpoint to `--save`.
- **`eval`** — loads `--net`, plays the greedy DFP policy (seat 0, GRU state
  carried) vs the fixed scripted hunter (seat 1, observation-only, no
  ground-truth) over `--episodes`, and reports frags/deaths + `net_frag_share` —
  the measurable "is it learning to fight" signal.
- **`export`** — loads `--net`, writes a portable flat file (`DOOMDFP1` magic +
  named fp32 tensors, no tch needed to read it) for the eventual wasm/in-browser
  forward, and verifies the round-trip against the live VarStore.

Set `DOOMTRAIN_MPS=1` to run on Metal (the net/rollout are device-generic).
`frags=0` from the untrained net is expected — the eval/`net_frag_share` is the
number that should climb during a real GPU run.

## Path to the GPU run (M3 continued)

- **Device**: `DOOMTRAIN_MPS=1` flips to Metal (already wired); a real run wants
  more iters/steps and tuned `lr`/epsilon/window.
- **Scale**: many env instances in parallel (the engine is ~3–6k tics/sec
  headless) feeding one replay buffer; longer rollouts.
- **Self-play / PBT**: APPO + population-based training (Sample Factory-style) is
  the strength path; both seats already step from the same net, so freezing one
  seat as a past-snapshot opponent (instead of the scripted hunter) is a small
  change to `collect_episode`.
- **Deploy**: the `export` flat file is the bridge to M4 — a browser forward
  loads it and drives a `doomrl` WASM build's ticcmd. M4 still needs an `emsdk`
  install for the engine side.

## Files

- `build.rs` — runs `doomrl/build.sh`, links `libdoomrl.a` (reruns if it's gone).
- `src/ffi.rs` — `#[repr(C)]` structs + `extern "C"` bindings + safe `Engine`.
- `src/env.rs` — `DoomEnv`, observation/measurement encoders, action decode, the
  scripted hunter baseline.
- `src/net.rs` — the GRU + dueling DFP heads; `step` (single tic) and
  `forward_seq` (whole-window BPTT via the GRU's `seq_init`).
- `src/dfp.rs` — rollout collection, `chunk_rollout` (BPTT windows), the
  `ReplayBuffer`, `train_chunk` (BPTT update), `eval_episode`.
- `src/export.rs` — portable weight export + round-trip verify + checkpoint load.
- `src/main.rs` — CLI: `smoke | train | eval | export`.
- `run.sh` — build + run with `DYLD_LIBRARY_PATH` set.
