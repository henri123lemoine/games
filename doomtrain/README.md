# doomtrain — DFP/GRU trainer for the doomrl 1v1 deathmatch substrate

M3 of "RL on Doom": a reinforcement-learning trainer over the `doomrl` 1v1
deathmatch substrate (M1/M2). Standalone `tch` crate, like `azt`/`azgo` — its
own `[workspace]` so `libtorch` never touches the root `cargo test`.

**Status: complete trainer, CPU-validated.** It compiles, links the C engine,
and runs the full DFP loop — collect → replay → **truncated-BPTT** updates with
epsilon decay — plus an `eval` (vs a fixed scripted hunter) and a portable
weight `export`. The remaining work is the long **GPU** run itself (flip the
device flag); the loss falls, eval runs, and export round-trips on CPU.

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
