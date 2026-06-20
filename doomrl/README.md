# doomrl — a controllable Doom substrate for RL

M1 of "RL on Doom": a headless, deterministic, instrumented build of vanilla
Doom with game-state read-out and input injection. The eventual goal is a 1v1
deathmatch bot trained against this substrate and deployed in a WASM build; this
milestone delivers the substrate only (no learning, no arcade wiring).

This is **not** a `Game`-trait game and is **outside the cargo workspace** — it is
standalone C, like `azgo`. It never touches `cargo test`.

## What this is

`doomgeneric` (vanilla Doom source with a 6-function platform seam) compiled
with a custom **headless** platform layer plus a thin **RL control surface**:

- step the engine one tic at a time, injecting a `ticcmd` for the player;
- read out player state (position, angle, momentum, health, armor, weapon,
  ammo, kill/item counts) and a **line-of-sight-gated** enemy list — only
  monsters the player can actually see (engine `P_CheckSight`), each with
  relative bearing, distance, and relative velocity, plus last-seen memory for
  a target that left the field of view;
- deterministic and side-effect-free read-out: same input sequence reproduces
  the same trajectory, and reading state never perturbs the simulation.

## Why doomgeneric (substrate decision)

Two hard constraints had to hold simultaneously:

1. **Instrumentable for RL** — native, headless, fast, with direct access to
   `players[]` / `mobj_t` state and the ability to inject input per tic.
2. **WASM-deployable later** — the final trained bot must ship in a browser
   build (this rules out ViZDoom / ZDoom C++, which does not compile to clean
   WASM).

doomgeneric wins on both: it is the *vanilla* Doom source (so `players[]`,
`mobj_t`, the thinker list, and the deterministic `M_Random` table are all
right there), it reduces the platform to six `DG_*` functions (so a headless
port is ~120 lines), and it is WASM-proven (it ships a
`doomgeneric_emscripten.c` and a `Makefile.emscripten`). The arcade's existing
browser Doom is a *Chocolate-Doom* Emscripten port — good proof the dynamics
compile to WASM, but a black box with no state seam. doomgeneric gives us the
same family of vanilla dynamics with the seam we need.

Vanilla Doom is single-player-deterministic by design (fixed-point math, a
static random table), which is exactly what reproducible RL rollouts want.

## Layout

```
doomrl/
  doomrl.h              control-surface API (state struct, action struct)
  doomrl.c              headless DG_* platform layer + RL API + ticcmd override
  doomrl_sound_null.c   null sound/music modules (no SDL dependency)
  driver.c              demo driver: warp E1M1, step N tics, print state, bench
  build.sh              compiles the vanilla core + our 3 files, native
  vendor/doomgeneric/   ozkl/doomgeneric @ dcb7a8d (vendored, GPLv2)
```

The only edit to vendored source is a two-line hook in `vendor/.../d_loop.c`:
a `__attribute__((weak)) DGRL_OverrideTiccmd()` (default no-op) called right
after the engine builds the player's ticcmd, so our strong override in
`doomrl.c` can substitute the RL action. Everything else is unmodified vanilla.

## Build & run (native, headless)

```bash
cd doomrl
./build.sh                                   # -> build/doomrl_driver
./build/doomrl_driver --tics=2000 --print-every=70
./build/doomrl_driver --tics=10000 --bench   # measured ~6400 tics/sec (~183x realtime)
```

The driver loads the shareware IWAD already vendored for the arcade
(`../web/app/public/doom/doom1.wad`), warps to **E1M1** at skill 3, and steps
deterministically. It does **not** open a window, an audio device, or render
(`-nodraw` short-circuits `D_Display`).

## Control surface (`doomrl.h`)

```c
void doomrl_init(int argc, char **argv);   // boot engine into a level (uses -warp/-skill)
void doomrl_step(const doomrl_action_t*);  // inject action, advance exactly one tic
void doomrl_get_state(doomrl_state_t*);    // read player + enemy state
int  doomrl_tic(void);                     // current gametic
```

Action (one tic): `forward` / `side` (signed, Doom move units), `turn`
(`angleturn`, `<<16` = full circle), `fire`, `use`, `weapon` (1-8, 0 = no
change). These map straight onto `ticcmd_t`, so movement is the engine's own
analog-ish step values, not key edges.

State: `tic`, `gamestate`, player `x/y/z` (float map units), `angle_deg`,
`momx/momy`, `health`, `armor`/`armortype`, `ready_weapon`, `ammo[4]`,
`killcount`/`itemcount`/`secretcount`, level totals, `alive`, plus
`num_visible_enemies` and up to 64 enemy slots — each `{type, x, y, z,
angle_deg, health, dist, bearing_deg, rel_vx, rel_vy, awake}`. Enemies are the
live `MF_COUNTKILL` mobjs from the thinker list (corpses excluded), **gated by
the engine's own `P_CheckSight(player, enemy)`** so only monsters with an
actual unobstructed line of sight are reported — no omniscience, no
through-wall entities. `bearing_deg` is the enemy's angle relative to the
player's facing (`[-180,180]`, 0 = dead ahead); `rel_vx/rel_vy` are the
enemy's velocity minus the player's; `awake==1` when the monster is targeting
the player.

The state also carries a `target` block (`doomrl_target_memory_t`): the nearest
currently-visible enemy is latched as the pursued target, and once it leaves
line of sight the block persists its `last_bearing_deg` / `last_dist` and counts
`ticks_since_seen` — the "pursue a target that left the FOV" hook. It resets on
level change.

## Determinism

`I_GetTime` is driven by a software clock (`DG_GetTicksMs`) rather than
wall-clock, so engine busy-waits (the screen-wipe loop, the `TryRunTics` wait)
terminate without real time passing. With `singletics=true`, each
`doomrl_step` runs **exactly one** game tic regardless of that clock. Same seed
+ same action sequence ⇒ byte-identical final state (verified: repeated runs
land on identical `pos/angle/hp/items`). Reproducibility comes from Doom's fixed
`M_Random` table plus identical injected ticcmds, not from the timer.

`P_CheckSight` mutates the engine's shared `validcount`, so the read-out
saves and restores it around the LOS scan — `doomrl_get_state` is therefore
**side-effect-free**: reading the observation 0, 1, or 20 times between steps
yields bit-identical subsequent dynamics (verified).

## Verified behavior (M1 acceptance)

- Boots E1M1, player spawns at `(1056, -3616)` facing 90° with 100 hp / pistol /
  50 bullets — the canonical E1M1 start.
- `forward` moves the player (and `-forward` reverses); turning sweeps the
  angle; the player picks up E1M1's armor/health bonuses as it walks.
- LOS gating verified: at the E1M1 spawn `num_visible_enemies==0` (the 6
  monsters are behind walls — the old omniscient scan reported all 6); walking
  forward until an imp (`MT_TROOP`, hp 60) clears a doorway flips it to 1 with a
  sensible bearing/distance, and after it leaves sight the `target` block keeps
  `last_bearing`/`last_dist` and increments `ticks_since_seen`.
- `get_state` is side-effect-free: 0 vs 20 extra reads per tic give identical
  trajectories.
- **~5800 tics/sec single-threaded stepping-only, ~5600 with a full LOS-gated
  `get_state` every tic (~160x realtime at Doom's 35 Hz); `P_CheckSight` adds
  ~3%.**

## Licenses

- **doomgeneric / vanilla Doom source** (`vendor/doomgeneric/`): **GPLv2** — see
  `vendor/doomgeneric/LICENSE`. Source: github.com/ozkl/doomgeneric @
  `dcb7a8dbc7a16ce3dda29382ac9aae9d77d21284`. Our headless port and RL surface
  link against this and are therefore GPLv2 as well.
- **`doom1.wad`**: id Software's freely-redistributable Doom shareware IWAD
  (not committed here; reused from `web/app/public/doom/`).

## Path to M2–M4

**M2 — 1v1 deathmatch driver (two ticcmds, reward signal).**
Vanilla Doom already supports up to 4 players via `players[MAXPLAYERS]`; a
deathmatch match is just two ticcmds fed per tic instead of one. The plan:

- Compile with `FEATURE_MULTIPLAYER` *off* and instead drive both players
  locally — extend `DGRL_OverrideTiccmd` (and the `ticdata[...].cmds[i]` write
  in `d_loop.c`) to fill `cmds[0]` and `cmds[1]` from two actions, set
  `playeringame[0..1]=true`, and start with `-deathmatch -warp` on a DM map.
  This sidesteps the netcode entirely (no packets, no sync) — both avatars live
  in one deterministic process, which is exactly what self-play wants.
- Reward: per-tic delta of `players[i].frags[]` (kill = +1, death = -1), plus
  shaping from `health`/`armor`/`damagecount` deltas and
  `P_AimLineAttack`-style line-of-sight to the opponent. All readable from the
  same structs already exposed.
- Add `doomrl_reset()` (re-`G_InitNew` the map) so episodes loop without
  restarting the process.

Note that vanilla doomgeneric is fundamentally single-player; the
two-local-players hack above is the **M2 long-pole** (the engine's spawn,
`P_SetupLevel`, and respawn paths assume one local player, and a clean
deathmatch needs DM thing-spawns and frag scoring wired through). ViZDoom
already solves multiplayer and is a viable *training-only* fallback if the
two-player hack proves too invasive — but it doesn't compile to clean WASM, so
M4's browser-deploy goal keeps us on doomgeneric.

**M3 — training loop.**
Observation = the `doomrl_state_t` vector — self features plus the already
**LOS-gated** nearest-K visible enemies in egocentric polar coords
(`bearing_deg`, `dist`, `rel_vx/rel_vy`) and the `target` last-seen block, so
the policy only ever sees what a screen player would. Small enough for a fast
MLP/GRU; later, the `DG_ScreenBuffer` for a conv policy. Self-play with the
existing `azt`/`azgo`-style infra (the repo already has AlphaZero plumbing,
momentum-SGD, gradient clipping). Many headless instances at ~5.6k tics/sec
each parallelize rollouts cheaply. Frame-skip (act every 4th tic) is one
`--tics` loop change.

**M4 — WASM deploy.**
The same instrumented build compiles to WASM via the vendored
`Makefile.emscripten` + `doomgeneric_emscripten.c` (swap our headless `DG_*` for
a canvas/keyboard variant, or keep headless and feed the policy net's ticcmd
through `DGRL_OverrideTiccmd`). Export `doomrl_step`/`doomrl_get_state` via
`EXPORTED_FUNCTIONS`; run the policy net in JS/WASM and drive the bot's ticcmd
each tic. `emcc` is not currently on this machine's PATH, so M4 needs an emsdk
install (`emscripten 6.x`, matching the arcade's existing port) before building.
```
