# doomtrain — DFP/GRU trainer for the doomrl 1v1 deathmatch substrate

M3 of "RL on Doom": a reinforcement-learning trainer over the `doomrl` 1v1
deathmatch substrate (M1/M2). Standalone `tch` crate, like `azt`/`azgo` — its
own `[workspace]` so `libtorch` never touches the root `cargo test`.

**Status: harness + CPU smoke.** It compiles, links the C engine, steps the
env, builds DFP targets, and runs training steps with the loss decreasing — the
GPU-free prep. The real training run (longer, on GPU) comes later.

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
./run.sh smoke                       # default: 3 episodes × 256 steps on the arena
./run.sh smoke --episodes=5 --steps=512 --epsilon=0.4
```

The smoke prints, per episode: frag count, batch size, DFP loss (decreasing),
target stats, observation spread, and the action histogram, and asserts shape /
finiteness / non-constant-observation sanity. Example:

```
engine up: num_players=2
net params: 377966
ep0: ... loss=0.098 tgt_absmax=0.060 obs_std=0.934 distinct_actions=30/30
...
smoke OK: compiled, env stepped, DFP targets built, train steps ran
```

`frags=0` in the smoke is expected: the **untrained** DFP policy has not learned
to aim or close distance yet (the scripted hunter frags because its aim is
hand-coded). The learning signal is present — loss falls and the frag
measurement is in the DFP target.

## Path to the real run (M3 continued, GPU)

- **Device**: flip `Device::Cpu` → `Device::Mps` (Metal) in `main.rs`, as `azt`
  does; the net and rollout are already device-generic.
- **Scale**: many env instances in parallel (the engine is ~3–6k tics/sec
  headless), a replay buffer, longer rollouts, epsilon decay.
- **BPTT**: the smoke runs the GRU stateless per step for the train pass
  (simplest correct form); real training should backprop through time over
  rollout chunks (carry `DfpNet::zero_state`-seeded hidden state across the
  chunk). The forward already threads explicit state for this.
- **Self-play / PBT**: APPO + population-based training (Sample Factory-style) is
  the strength path; both seats already step from the same net, so freezing one
  seat as a past-snapshot opponent is a small change.
- **Eval / export**: add a greedy-eval subcommand (frags/episode vs a fixed
  opponent) and a weight export for a future browser bot (M4 WASM still needs an
  `emsdk` install for the engine side).

## Files

- `build.rs` — runs `doomrl/build.sh`, links `libdoomrl.a`.
- `src/ffi.rs` — `#[repr(C)]` structs + `extern "C"` bindings + safe `Engine`.
- `src/env.rs` — `DoomEnv`, observation/measurement encoders, action decode.
- `src/net.rs` — the GRU + dueling DFP heads.
- `src/dfp.rs` — rollout collection, future-offset target build, train step.
- `src/main.rs` — CLI + `smoke`.
- `run.sh` — build + run with `DYLD_LIBRARY_PATH` set.
