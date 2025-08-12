# Twenty-One RL Environment (Rust) + Python Bridge

This repo contains a fast Rust implementation of the two-player Twenty-One game (Rules in `AGENTS.md`), with:

- A compact, allocation-conscious environment suitable for RL.
- Deterministic test hooks and a Criterion benchmark.
- A JSON stdin/stdout bridge binary for Python interop.
- Python scripts for simple bot play, MCCFR training, and playing against a trained agent.

## Build & Test

- Build: `cargo build`
- Lint: `cargo clippy -- -D warnings`
- Tests: `cargo test`
- Bench: `cargo bench`

## Running the Bridge + Scripts

All scripts automatically locate the Rust bridge binary, building it if needed. You can override the binary path with `TWENTYONE_BRIDGE_BIN=/absolute/path/to/twentyone_bridge`.

From `scripts/`:

1. Quick bot-vs-bot verification

   - `uv run run_basic.py`
     - This will build the bridge if missing, run a full game between two simple bots, and print the outcome.

2. Train a simple MCCFR policy (single-round game abstraction)

   - `uv run mccfr_agent.py`
     - Saves a JSON policy at `scripts/data/policy_mccfr.json` (directory created on demand).

3. Play against the agent

   - `uv run play_vs_agent.py scripts/data/policy_mccfr.json 42`
     - You’ll be prompted to draw or stand each turn.
     - The agent uses the trained policy and falls back to a simple heuristic if a state isn’t in the policy.

## Notes

- The Rust env models full game hearts and rounds. The MCCFR trainer provided here learns a decent draw/stand policy on a single-round abstraction for demonstration; stronger performance can be achieved with a deeper, state-aware trainer (e.g., multi-round modeling or snapshot/restore of the Rust env).
- The bridge JSON protocol is documented in `src/bin/bridge.rs` and summarized below:
  - `{"cmd":"new","seed":<u64>}`
  - `{"cmd":"start_round"}`
  - `{"cmd":"current_player"}` → `{current_player}`
  - `{"cmd":"observation","player":0|1}` → full observation for that POV
  - `{"cmd":"step","action":"draw"|"stand"}` → step result, round/game status, outcome
  - `{"cmd":"hearts"}`, `{"cmd":"round"}`
  - `{"cmd":"quit"}`

## Troubleshooting

- If a script reports `FileNotFoundError: target/debug/twentyone_bridge`, it is likely being run from a subdir; the updated scripts auto-build and locate the binary. Alternatively, run `cargo build --bin twentyone_bridge` once from repo root.
- To use a custom build dir, set `TWENTYONE_BRIDGE_BIN` to the built binary path.
