# aztrainer

One AlphaZero trainer for every learned game. A config-driven tch resnet plus a
shared self-play / replay / optimizer / run-dir harness, parameterized by a
`(Game, PolicyValueEncoder, NetConfig)` triple — replacing the near-identical
`azt` (chess), `azgo` (go), and `azsnake` (snake) crates. The algorithm is
written once; only game knowledge lives per game, under
`src/games/{chess,four_player_chess,go,snake}/`.

Standalone on purpose (empty `[workspace]` table): keeps libtorch off the main
workspace's `cargo test`. Build from this directory.

## What is shared vs per-game

- **Shared core** (`src/`): the resnet (`net.rs`, keyed on `nn_infer::HeadKind`,
  so a checkpoint's architecture selects both the policy and value head), the
  optimizer + training step (`train.rs`: AdamW or SGD+Nesterov, gradient
  clipping, SWA, the value-mix z/q target blend, go's ownership/score auxiliary
  losses), the replay buffer, the run-dir contract (`rundir.rs`:
  `metrics.jsonl` / `latest.ot` / `ckpt-*` / `dashboard.html` / `STOP` / the LR
  schedule), the dual-write export (`export.rs`), and the export parity check
  (`verify.rs`).
- **Per game** (`src/games/<game>/`): the `NetConfig`, the replay `Sample`
  codec, the self-play reward shaping (chess repetition / go komi-ownership-score
  + dihedral + mixed-size / Battlesnake joint-equilibrium backups + chance
  resolution + discount + all-perspective dihedral augmentation),
  the eval ladder, the strength gauge, and any serving commands.

## Binaries

```
cargo run --release --bin chess -- <run|bench|play|uci|elo|calibrate|export|verify-export>
cargo run --release --bin four-player-chess -- <run|eval|export|verify-export>
cargo run --release --bin go    -- <run|bench|elo|rate|calibrate|calibrate-pass|export|verify-export>
cargo run --release --bin snake -- <run|bench|rate|elo|field|compare|field-compare|split-compare|export|verify-export>
```

A short example:

```
cargo run --release --bin go -- run --dir ../../data/azgo/run1 --hours 5 --size 9
cargo run --release --bin four-player-chess -- run --dir ../../runs/four-player-chess --hours 8
cargo run --release --bin four-player-chess -- eval --net ../../runs/four-player-chess/latest.ot
cargo run --release --bin go -- verify-export --net ../../data/azgo/run1/latest.ot
cargo run --release --bin snake -- run --method logit --players 4 --hours 8 --dir runs/battlesnake/logit-p4
cargo run --release --bin snake -- field --net runs/battlesnake/logit-p4/latest.ot --method logit
```

## Running the built binaries: the libtorch dylib path

`tch`'s `download-libtorch` feature fetches libtorch into the build directory
but bakes **no rpath** into the binary, so running it directly fails with
`Library not loaded: @rpath/libtorch_cpu.dylib`. Point `DYLD_LIBRARY_PATH`
(macOS) / `LD_LIBRARY_PATH` (Linux) at the downloaded lib dir — the same gotcha
the original trainers had:

```sh
LIBT=$(find target/release/build -name libtorch_cpu.dylib | head -1 | xargs dirname)
DYLD_LIBRARY_PATH="$LIBT" ./target/release/go verify-export --net ../../data/azgo/run1/latest.ot
```

(`cargo run` injects this automatically; only the standalone binary needs it.)

## Export formats

Export dual-writes the legacy per-game container (`AZWEB001` / `AZWEBGO2-3` /
`AZSNK1`) and the unified `AZNET1` container. The BN-folded weight body is
byte-identical between them, so the deployed browser nets keep loading while
`AZNET1` ships additively. `verify-export` checks the tch forward against both
the legacy `*infer` reference and `nn-infer`'s `AZNET1` forward.
