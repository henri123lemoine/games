# ML layer refactor: generic by-algorithm engines

**Status:** design only. Do not execute the migration from this document — the
move is a single exclusive pass that must run *after* the in-flight game-fix
work lands (see [Migration plan](#migration-plan)).

## Why

The lab's thesis (`ARCHITECTURE.md`): *algorithms written once against a shared
`Game` trait, applied to many games.* Search, CFR, rollout, alpha-beta all honor
this — there is exactly one `solvers::azero::Search`, one `AlphaBeta`, one
`Rollout`, generic over `Game` + capability traits.

The **ML layer does not.** It is sprawled by *game* instead of by *algorithm*:

| concern | chess | go | snake | slither | doom |
|---|---|---|---|---|---|
| tch-free forward + format | `azinfer` | `goinfer` | `snakeinfer` | `slitherinfer` | — |
| AlphaZero trainer (tch) | `azt` | `azgo` | `azsnake` | — | — |
| PPO trainer (tch) | — | — | — | `slither-ppo` | `doomrl` (being built) |
| browser WebGPU evaluator | `chess/azgpu.ts` | `go/azgpu.ts` | `snake/azgpu.ts` | (PPO path) | — |

Three near-identical resnet forwards in Rust, three near-identical resnet
forwards in tch, three byte-identical conv shaders in WGSL, and three trainer
crates whose `Cargo.toml`s differ only in which `(game, infer)` pair they name.
Adding the next learned game means copy-pasting a forward, a trainer, an export
format, and a WebGPU driver. That is precisely the per-game algorithm
duplication the architecture exists to forbid.

This refactor applies the thesis to the ML layer: **the forward pass, the
self-play loop, the PPO math, and the weights format are algorithms — they
become generic. Encoders (which plane means what) are game knowledge — they
stay per-game, exactly where they already live.**

## What is already correct (do not touch)

Two seams are already in the right place and the design builds on them rather
than replacing them:

1. **`game_core::PolicyValueEncoder`** (`game-core/src/lib.rs:173`) is the
   existing generic boundary between game knowledge and net machinery:

   ```rust
   pub trait PolicyValueEncoder<G: Game>: Sync {
       fn input_len(&self) -> usize;                                    // feature width
       fn policy_len(&self) -> usize;                                   // policy head width
       fn encode_state(&self, game: &G, state: &G::State) -> Vec<f32>;  // game knowledge
       fn action_index(&self, game: &G, state: &G::State, a: G::Action) -> usize;
   }
   ```

   The per-game encoders **already live in the game crates**:
   `games/chess/src/encode.rs` (`PlanesEncoder`), `games/go/src/encode.rs`
   (`GoEncoder`), `games/snake/src/encode.rs` (`SnakeEncoder`). The `*infer`
   crates do **not** own the encoders — they own the forward and the format.
   This is the load-bearing fact: only the forward + format are duplicated; the
   game knowledge is already factored correctly.

2. **`solvers::azero::Search`** (`solvers/src/azero/search.rs`) is the one PUCT
   implementation, generic over `Game`. Its evaluator contract is already
   game-agnostic:

   ```rust
   pub struct EvalRequest { pub features: Vec<f32>, pub support: Vec<u16> }
   pub struct EvalResult { pub priors: Vec<f32>, pub value: f32 }
   ```

   `azt`, `azgo`, `azsnake` all drive *this same* `Search` — none forks it. The
   only thing each supplies is *who answers `EvalRequest`* (a GPU batch through
   its `*infer` instantiation). That answerer is what a generic inference engine
   becomes.

So the refactor changes the two duplicated layers (forward+format, trainer
harness) and leaves the two correct seams (encoder trait, Search) untouched.

---

## Step 1 — shared vs per-game analysis

### 1a. The four inference forwards

All four are tch-free, pure-`f32` reference forwards of the **same network
family**: a 3×3 conv stem, a residual tower of `blocks × (conv3×3, conv3×3)`
pairs with BN folded into the conv weights, then a policy head and a value head.
The `AZSNK1` parser (`snakeinfer/src/model.rs:96`) is the canonical shape — stem,
`tower: Vec<(Conv, Conv)>`, policy head, value head, read sequentially by a
`Reader` with `floats()/conv()/linear()` primitives.

**Common (becomes the generic engine):**

- Conv stem (3×3), residual tower of `blocks` paired 3×3 convs, BN-folded.
- Architecture dims (`blocks`, `channels`, board `size`) are **already
  config-driven**, read from the header, not hardcoded — `goinfer`/`snakeinfer`
  validate them with plausibility bounds.
- The `Reader` primitive set (`floats`, `conv`, `linear`) and the
  "no trailing bytes" integrity check.
- Global-pool nets are **board-size-agnostic**: the same conv weights run at any
  `size` (`snakeinfer/src/model.rs:147`, "forward at an arbitrary board size").

**Per-game (stays as plugged-in parameters):**

| axis | chess (`azinfer`) | go (`goinfer`) | snake (`snakeinfer`) | slither (`slitherinfer`) |
|---|---|---|---|---|
| input planes | chess planes ×64 | go planes ×size² | snake planes ×size² | semantic grid + **scalars** |
| **policy head** | **flat** (dense → fixed move space) | **global-pool** (1×1 conv → spatial + pass) | **global-pool** (3C pool → MLP → 4 dirs) | global-pool → small action set |
| value head | scalar tanh | scalar (+ optional **ownership** head, GO3) | scalar | scalar |
| consumer | PUCT `EvalRequest` | PUCT `EvalRequest` | PUCT `EvalRequest` | **greedy action** (PPO), `grid+scalars` |

The two genuine variation axes are **head type** (flat vs global-pool, plus
go's optional ownership head) and **whether there is a scalars side-input**
(slither only). Both are finite enumerations, not open-ended per-game logic.

**Export formats — confirmed structural siblings, one container is feasible.**

| crate | magic | header after magic | notes |
|---|---|---|---|
| `azinfer` | `AZWEB001` | `blocks, channels` (u32) | flat policy; fixed board |
| `goinfer` | `AZWEBGO2` / `AZWEBGO3` | `blocks, channels, size` (u32) | GO3 = GO2 + ownership head — **versioned evolution already exists** |
| `snakeinfer` | `AZSNK1` | `blocks, channels, size` (u32) | global-pool policy |
| `slitherinfer` | `SLNET1` | `blocks, channels, size` (u32) + scalar dims | grid+scalars; PPO net |

Every format is `magic ∥ {dims as u32 LE} ∥ {BN-folded f32 weights, in fixed
layer order}` with a plausibility check and a no-trailing-bytes check. They
differ only in the magic string, which dims appear, and which heads' weights
follow. **One unified versioned container subsumes all four** (see
[the unified format](#the-unified-format-aznet1)). Go's GO2→GO3 already proves
the extension mechanism: bump a version/flag, append a head's weights.

**Cargo / membership:** `azinfer`, `goinfer`, `snakeinfer` are main-workspace
members (tch-free, so they don't pull libtorch). `slitherinfer` is **not** a
workspace member — it depends on `slither-rl` (the env), and lives on the PPO
side; it is consumed by `slither-ppo` (standalone) and the slither wasm engine.

### 1b. The three AlphaZero trainers

`azt`, `azgo`, `azsnake` are **structurally identical** standalone crates. Each:

- uses the empty-`[workspace]` table trick to stay out of the main workspace
  (keeps libtorch off `cargo test` at root) — *verbatim the same comment in all
  three*;
- has the dependency skeleton `{game} + {infer} + game-core + solvers + tch +
  rayon + serde_json`, differing **only** in the `(game, infer)` pair:
  - `azt`: `chess + azinfer`
  - `azgo`: `go + goinfer`
  - `azsnake`: `snake + snakeinfer`
- drives the one shared `solvers::azero::Search` (none forks it);
- defines a `net.rs` (tch resnet: conv stem + residual tower + heads), an
  `export.rs` doing `magic ∥ dims ∥ BN-folded floats`, and a self-play /
  replay / optimizer / run-dir harness.

**The tch `net.rs` modules are the same resnet** differing only in plane count,
policy width, board dims, and head type — the *training-side mirror* of the same
forward the `*infer` crates evaluate. `export.rs` is the same `to_le_bytes`
weight-dump in all three (`azt/src/export.rs:59`, `azgo/src/export.rs:74`,
`azsnake/src/export.rs:66`).

**Shared harness (becomes the generic trainer):** self-play batching across
concurrent games, leaf-eval batching, replay buffer of `(encoding, visit
distribution, outcome)`, optimizer (SGD+momentum, gradient clipping, LR
warmup — see note below), loss (policy cross-entropy + value MSE + L2), the
run-dir contract (`metrics.jsonl` + dashboard + `STOP` file), and the export
step.

**Per-game seams (stay as parameters):** the `Game` + `PolicyValueEncoder`, the
`NetConfig` (planes/channels/blocks/policy width/head type), and any
game-specific eval gauge (e.g. azgo's GNU Go Elo anchor, azsnake's heuristic
seat). These are configuration and harness plug-ins, not algorithm forks.

> **Drift note:** recent commits landed "SGD+momentum, gradient clipping, LR
> warmup" in `azgo`. Whether `azt`/`azsnake` got the same refinement is exactly
> the kind of divergence a single trainer eliminates — one optimizer, fixed
> once. (Confirm parity at migration time; the generic trainer adopts the
> most-evolved version.)

### 1c. The two PPO trainers

- **`slither-ppo`** (standalone, tch): trains a 3-conv CNN policy/value net over
  `slither-rl`'s egocentric semantic grid; vectorized rollouts over many
  parallel `slither-rl` arenas; GAE(λ) + clipped surrogate + value loss +
  entropy; a PFSP-lite opponent pool seeded with the hand-coded heuristic.
  Depends on `slither-rl` (env) + `slitherinfer` (reference forward / export).
- **`doomrl`** is, today, **only the C `doomgeneric` substrate** — there is *no
  Rust trainer yet*. The doom-dynamics agent is building the PPO trainer now.
  The design therefore **must not depend on doomrl internals**; instead the
  generic PPO core is shaped so doomrl's future trainer is a thin adapter from
  day one (nothing to retrofit).

**Shared PPO core (becomes `ppo-core`):** rollout collection, GAE(λ) advantage
estimation, advantage normalization, the clipped surrogate objective, (clipped)
value loss, entropy bonus, minibatch/epoch loop. This is standard PPO math —
identical up to hyperparameters — so a shared core is clearly viable.

**Per-env adapter:** what the core needs from an environment is small and
uniform: `reset`, `step → (obs, reward, done)`, an action space, and an obs
shape. `slither-rl` is a pure-Rust env; doom wraps C via FFI. A generic `Env`
trait (below) captures exactly this, and each substrate implements it once.

The PPO **net** is conv-based (same trunk family as the AZ side), but on the
*training* side it is tch; its tch-free reference forward (for the browser /
greedy play) is the slither path through the unified inference engine. So PPO
reuses the *forward engine* but has its own *training algorithm* and its own
*evaluator contract* (greedy `argmax` over `grid+scalars`, not PUCT
`EvalRequest`).

### 1d. The WebGPU evaluators (TypeScript)

`web/app/src/frontends/{chess,go,snake}/azgpu.ts` triplicate the same thing.
The conv shader's WGSL is **byte-identical** across all three — same bindings
`x/w/bias/res/y/P` at `@binding(0..5)`, `@compute @workgroup_size(64)`, entry
`main` (`chess:8`, `go:11`, `snake:28`). There is **no shared WGSL/driver
module** today (a search for a common `wgsl`/`resnet`/`NetWeights` helper finds
nothing). Each file is ~360–414 lines, the bulk being the duplicated conv/resnet
WGSL + the buffer-allocation/dispatch driver + a header parser that
re-implements, in TS, the same byte layout the matching Rust `*infer` crate
parses. (slither has no `azgpu.ts` — its PPO path uses a different evaluator.)

**Common (becomes a shared TS module):** the conv/residual-block WGSL, the
global-pool/head WGSL (parameterized by head type), the GPU buffer allocation
and the conv→tower→heads dispatch loop, and the binary header parser (one
parser for the unified format).

**Per-game (stays):** the board→tensor input encoder (game knowledge,
mirrors the Rust `encode_state`), the policy width, board dims, and head type —
all read from the unified header, so even these collapse to *data* rather than
*code*.

A unified format + one shared TS driver collapses three ~380-line files to one
shared driver plus a per-game encoder of a few dozen lines each.

---

## Step 2 — the generic engines

### The unified format: `AZNET1`

One versioned container for every exported net, replacing `AZWEB001` /
`AZWEBGO2/3` / `AZSNK1` / `SLNET1`:

```
magic    : "AZNET1\0\0"                 (8 bytes)
arch     : u32 blocks
           u32 channels
           u32 planes                   (input feature channels)
           u32 size                     (board side; 1 for fixed/flat games)
           u32 scalars                  (side-input scalar count; 0 if none)
           u32 policy_kind              (0 = flat, 1 = global-pool spatial)
           u32 policy_len               (flat policy width; 0 when spatial)
           u32 heads_flags              (bit0 = value, bit1 = ownership, …)
weights  : BN-folded f32 (LE), in fixed layer order:
           stem, tower[blocks]×(conv,conv), policy-head, value-head,
           [ownership-head if flag set]
```

- Self-describing: parser + WebGPU driver branch on `policy_kind` /
  `heads_flags` / `scalars`, never on a game identity.
- Backward extension is the GO2→GO3 mechanism generalized: new heads add a flag
  bit and append weights; old readers reject unknown flags cleanly.
- One Rust parser, one TS parser. The `Reader{floats,conv,linear}` primitives
  already exist (`snakeinfer/src/model.rs:62`) and are lifted verbatim.

### Generic inference engine: `nn-infer` (tch-free, workspace member)

Replaces `azinfer` / `goinfer` / `snakeinfer` / `slitherinfer`'s forward+format.

```rust
pub struct Arch {
    pub blocks: usize, pub channels: usize, pub planes: usize,
    pub size: usize, pub scalars: usize,
    pub policy: PolicyKind, pub heads: HeadFlags,
}
pub enum PolicyKind { Flat { len: usize }, GlobalPool }

pub struct Net { /* stem, tower, heads — all config-driven */ }
impl Net {
    pub fn parse(bytes: &[u8]) -> Result<Net, String>;          // AZNET1
    pub fn arch(&self) -> &Arch;
    /// Spatial features (planes·size²) + optional scalars → policy logits + value.
    pub fn forward(&self, planes: &[f32], scalars: &[f32]) -> Output;
    pub fn forward_at(&self, planes: &[f32], scalars: &[f32], size: usize) -> Output;
}
pub struct Output { pub policy: Vec<f32>, pub value: f32, pub ownership: Option<Vec<f32>> }
```

The PUCT bridge — turning a `Net` into the answerer for `Search`'s
`EvalRequest` — is a thin generic adapter (`encode_state` already produced the
`features`; `support` selects/softmaxes the legal policy subset). PPO uses the
same `Net::forward` for greedy `argmax`.

**Per-game encoders are unchanged** and stay in `games/*/src/encode.rs`. slither's
`obs` encoder stays in `slither-rl`/`slitherinfer` (it is env knowledge, not a
`PolicyValueEncoder`). The engine consumes encoders; it does not own them.

### Generic AlphaZero trainer: `aztrainer` (standalone, tch)

One standalone crate replacing `azt` / `azgo` / `azsnake`. Keeps the empty
-`[workspace]` libtorch-isolation trick. Parameterized at the binary entry by
the `(Game, PolicyValueEncoder, NetConfig)` triple:

```rust
pub struct TrainSpec<'a, G: Game, E: PolicyValueEncoder<G>> {
    pub game: &'a G,
    pub encoder: &'a E,
    pub net: NetConfig,          // blocks, channels, planes, size, head kind, heads
    pub search: PuctConfig,      // solvers::azero::PuctConfig
    pub run: RunDir,             // metrics.jsonl + dashboard + STOP contract
    pub gauge: Option<Box<dyn EloGauge<G>>>,  // azgo's GNU Go anchor, etc.
}
pub fn train<G, E>(spec: TrainSpec<G, E>);
```

- One tch resnet (`net.rs`), config-driven by `NetConfig`, the training mirror of
  `nn-infer::Net`. One optimizer (the most-evolved: SGD+momentum, grad clip, LR
  warmup). One self-play/replay/loss harness. One `export()` writing `AZNET1`.
- Per-game `main.rs` shims (chess/go/snake) construct the `TrainSpec` and call
  `train` — a few dozen lines each, the only per-game code on the trainer side.
- Game-specific gauges (GNU Go Elo, heuristic seat) plug in via the `EloGauge`
  trait — game *knowledge*, not trainer forks.

### Generic PPO trainer: `ppo-core` + per-env adapters (standalone, tch)

```rust
pub trait Env: Sync {
    type Obs;                                   // e.g. { grid: Vec<f32>, scalars: Vec<f32> }
    fn reset(&mut self, seed: u64);
    fn step(&mut self, actions: &[usize]) -> Vec<Step>;   // (obs, reward, done) per actor
    fn obs(&self, actor: usize) -> Self::Obs;
    fn num_actors(&self) -> usize;
    fn action_space(&self) -> usize;
    fn obs_shape(&self) -> ObsShape;            // planes·size² + scalars
}

pub struct PpoConfig { /* clip, gae_lambda, gamma, vf_coef, ent_coef, epochs, minibatch, lr */ }
pub fn train<E: Env>(env_pool: Vec<E>, cfg: PpoConfig, net: NetConfig, run: RunDir);
```

`ppo-core` owns rollout collection, GAE(λ), advantage normalization, the clipped
surrogate, (clipped) value loss, entropy, and the minibatch/epoch loop — the
math, once. `slither-ppo` becomes a thin `Env` adapter over `slither-rl` (plus
its PFSP opponent-pool logic, which is slither-specific league policy and stays
with it). `doomrl`'s future trainer implements `Env` over the C substrate via
FFI — and because the contract is fixed now, it is an adapter from the start.

The PPO net reuses the `NetConfig`/resnet shape; its tch-free reference forward
is `nn-infer::Net` (slither's `grid+scalars` path).

### Generic WebGPU driver: `web/app/src/engine/aznet.ts`

One module exporting: the conv/residual-block WGSL, the head WGSL
(branching on `policy_kind`/`heads_flags`), the buffer-allocation + dispatch
driver, and the `AZNET1` header parser. Each frontend keeps only its
board→tensor encoder (mirror of `encode_state`) and calls the shared driver. The
three ~380-line `azgpu.ts` files collapse to one driver + three small encoders.

### Target layout (by algorithm, not game)

```
nn-infer/                tch-free generic forward + AZNET1 parser   [workspace member]
solvers/src/azero/       unchanged (the one Search)
games/*/src/encode.rs    unchanged (per-game PolicyValueEncoder)
slither-rl/              unchanged env (+ canonical deploy config, see below)

aztrainer/               generic AlphaZero trainer (tch)            [standalone, empty [workspace]]
  src/{net,selfplay,replay,optimizer,export,rundir}.rs
  src/bin/{chess,go,snake}.rs   per-game TrainSpec shims
ppo-core/                generic PPO math (tch)                     [standalone]
  + per-env adapters: slither (over slither-rl), doom (over doomgeneric FFI)

web/app/src/engine/aznet.ts     shared WGSL + driver + parser
web/app/src/frontends/*/        per-game encoder only

# removed after migration:
azinfer goinfer snakeinfer slitherinfer  azt azgo azsnake
chess/azgpu.ts go/azgpu.ts snake/azgpu.ts (logic → aznet.ts)
```

---

## Train↔Deploy parity contract

A first-class invariant the generic engine must enforce, motivated by a real
bug: the slither browser deploy diverged from the trainer **not** in the shared
dynamics code (`slither-rl`'s `World` is genuinely shared by `slither-ppo` and
the wasm engine) but in **config** and **deploy-only behavior**:

- `web/app/src/frontends/slither/index.ts:34-35` **hardcoded** `WORMS = 6`,
  `PELLETS = 250`; an earlier version had `700` (the comment at `:29-30` admits
  it ran "~3x denser" than training). The trainer's real config lives in
  `slither-rl/src/world.rs:176` (`WorldConfig`, `Default` `pellet_target = 250`).
  Two independent specifications of one thing → guaranteed eventual drift.
- `index.ts:11-12` documents a **deploy-only** round-robin bot-forward throttle
  (~7.5 Hz vs training's 30 Hz) — a perf hack that changed the bot's effective
  reaction rate, i.e. dynamics-relevant behavior, on the deploy side only.

Shared dynamics *code* is necessary but **not sufficient**: the same state
machine fed different inputs, or wrapped in deploy-only behavior, still diverges.

**The contract (applies to every learned game — snake/go/chess AZ nets too):**

1. **Single source of truth for env/world config.** The trained run's config is
   **exported alongside the weights** — it rides in the `AZNET1` header as a
   config block (the env/world parameters + board/encoder identity the net was
   trained on). The deployed wasm **reads that block**; it must not carry
   hardcoded `WORMS`/`PELLETS`/board-size/plane constants. For games whose
   config is a fixed canonical value, the shared env/game crate exposes a single
   `deploy_config()` (or `WorldConfig::canonical()`) that **both** the trainer
   and the browser call — never a re-specified literal on the deploy side.

2. **Single source of truth for decision/step rate.** The decision rate the net
   trained at is part of that exported config and is honored verbatim at deploy.
   No deploy-side throttle/round-robin may alter the *effective* decision or step
   rate of a learned agent.

3. **Deploy-only logic is render/cosmetic only.** Interpolation, glow,
   DPR caps, sprite caching — fine. Anything touching dynamics-relevant behavior
   (decision rate, env parameters, action timing) is forbidden on the deploy
   side; it must come from the shared config/code path.

4. **Parity is testable.** A deploy reading config from the header cannot
   silently disagree with training; a CI check can assert the wasm engine
   instantiates from the exported config and that no dynamics constant is
   hardcoded in a frontend.

The generic refactor is the natural enforcement point: one exported format
carrying one config, consumed identically by the trainer (writer) and the wasm
engine (reader), makes this bug class **structurally impossible** rather than a
discipline everyone must remember per game.

---

## Migration plan

### Sequencing reality (the hard constraint)

The trainer/infer crates are **being actively edited right now** by in-flight
game-fix agents: snake-speed (→ `snakeinfer`/web), slither-fix (→ `slither-rl`),
slither-strength (→ `slither-ppo`), doom-dynamics (→ building `doomrl`'s PPO).
Moving or merging these crates mid-edit would break their work and produce
brutal conflicts.

**Therefore the trainer/PPO merge runs as ONE exclusive big-bang pass, started
only AFTER all four game-fix agents land.** Do not interleave it with their work.

### What can stage incrementally (before the exclusive pass)

These are additive and don't move the crates the agents are editing:

1. **Land the `AZNET1` container + `nn-infer` forward** as a *new* workspace
   crate, with each existing `*infer` crate's tests ported as parity tests
   (parse a real exported net, assert forward matches the old crate bit-for-bit).
   Old crates keep working; nothing is deleted yet.
2. **Add `nn-infer`'s exporter** and teach trainers to *also* emit `AZNET1`
   alongside their current format (dual-write), so a real trained net exists in
   the new format to test against. No behavior change.
3. **Build `web/app/src/engine/aznet.ts`** (shared WGSL + driver + parser) and
   migrate **one** frontend (snake is the safest once snake-speed lands) to it,
   leaving the other two on their own `azgpu.ts`. Validate browser parity, then
   migrate the rest.

### The exclusive pass (after the agents land)

4. **Merge the three AZ trainers into `aztrainer`** with per-game `bin/` shims;
   adopt the most-evolved optimizer; emit only `AZNET1`. Delete `azt`/`azgo`/
   `azsnake`.
5. **Factor `ppo-core` out of `slither-ppo`**; reduce `slither-ppo` to an `Env`
   adapter + its PFSP league policy. Coordinate the `Env` trait shape with
   doom-dynamics so doomrl's trainer adopts it directly.
6. **Cut the remaining frontends** to `aznet.ts`; delete the three `azgpu.ts`
   logic bodies. Delete `azinfer`/`goinfer`/`snakeinfer`/`slitherinfer` once
   `nn-infer` fully subsumes them.
7. **Bake the parity contract**: move the slither deploy config into a shared
   `deploy_config()`/`WorldConfig::canonical()`, drop the hardcoded frontend
   constants and the bot-forward throttle, and route the wasm engine through the
   `AZNET1` config block. Add the CI parity check.

### Risks

- **tch isolation must hold.** `aztrainer` and `ppo-core` stay standalone with
  the empty-`[workspace]` trick; `nn-infer` stays tch-free and a workspace
  member. If any tch dep leaks into the main workspace, `cargo test` at root
  pulls libtorch — a hard regression. Guard with a CI check that the root
  workspace has no tch in its lockfile.
- **Forward parity is non-negotiable.** Every weight already trained must load
  and evaluate identically under `nn-infer` (BN-folding order, layer order,
  global-pool reduction). Gate the merge on bit-for-bit parity tests against the
  old crates' outputs before deleting anything (steps 1–2 exist for this).
- **doomrl is a moving target.** Its PPO trainer doesn't exist yet; the `Env`
  trait must be agreed *with* doom-dynamics, not imposed on a finished crate.
- **Go's ownership head + slither's scalars** are the two format features that
  exercise the `heads_flags`/`scalars` generality — get them into the `AZNET1`
  design from day one (they are, above) so the container isn't retrofitted.
- **Head-type enumeration must stay closed.** If a future game needs a head that
  is neither flat nor global-pool, that's a new `PolicyKind` variant + a version
  bump — a deliberate, reviewed extension, not a per-game escape hatch.
