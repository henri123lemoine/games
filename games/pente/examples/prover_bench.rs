//! Deterministic correctness + throughput corpus for the Pente forcing prover.
//!
//! This is deliberately model-free: it isolates the CPU-side tactical work
//! that runs at every AlphaZero leaf. The checksum and per-position proof labels
//! make before/after binaries directly comparable, while repeated wall time
//! measures the implementation rather than training or GPU variance.
//!
//! Run:
//!   cargo run --release -p pente --example prover_bench -- \
//!     positions=16 repeats=3 depth=7 nodes=1500 vct=1

use std::hint::black_box;
use std::time::Instant;

use game_core::{Game, GameUi, Rng};
use pente::{Pente, PenteState, VcfConfig, winning_move};

fn arg<T: std::str::FromStr>(key: &str, default: T) -> T {
    let prefix = format!("{key}=");
    std::env::args()
        .skip(1)
        .find_map(|value| value.strip_prefix(&prefix).and_then(|raw| raw.parse().ok()))
        .unwrap_or(default)
}

fn corpus(game: &Pente, count: usize) -> Vec<PenteState> {
    let mut rng = Rng::new(0x5eed_f00d_cafe_babe);
    let mut states = Vec::with_capacity(count);
    let mut attempt = 0usize;
    while states.len() < count {
        let mut state = game.initial_state();
        // Cover sparse openings through tactically denser middlegames without
        // depending on any bot implementation under comparison.
        let target_ply = 4 + ((attempt * 11 + states.len() * 7) % 49);
        for _ in 0..target_ply {
            let actions = game.legal_actions(&state);
            let action = actions[rng.below(actions.len())];
            game.apply(&mut state, action);
            if game.is_terminal(&state) {
                break;
            }
        }
        if !game.is_terminal(&state) {
            states.push(state);
        }
        attempt += 1;
    }
    states
}

fn main() {
    let size = arg("size", 19usize);
    let positions = arg("positions", 16usize);
    let repeats = arg("repeats", 3usize);
    let depth = arg("depth", 7u32);
    let nodes = arg("nodes", 1500u64);
    let vct = arg("vct", 1u32) != 0;
    let game = Pente::new(size);
    let cfg = VcfConfig::for_leaf(depth, nodes, vct);
    let states = corpus(&game, positions);

    // Warm instruction/data caches without including the warm-up in totals.
    let _ = black_box(winning_move(&game, &states[0], cfg));

    let started = Instant::now();
    let mut checksum = 0xcbf2_9ce4_8422_2325u64;
    let mut proofs = 0usize;
    let mut slowest_ms = 0.0f64;
    for (index, state) in states.iter().enumerate() {
        let position_started = Instant::now();
        let mut expected = None;
        for repeat in 0..repeats {
            let result = black_box(winning_move(&game, black_box(state), cfg));
            if repeat == 0 {
                expected = result;
            } else {
                assert_eq!(
                    result, expected,
                    "non-deterministic proof at corpus index {index}"
                );
            }
        }
        let elapsed_ms = position_started.elapsed().as_secs_f64() * 1e3;
        slowest_ms = slowest_ms.max(elapsed_ms / repeats as f64);
        proofs += usize::from(expected.is_some());
        let proof_id = expected.map_or(0, |action| u64::from(action.0) + 1);
        let key = game.state_key(state).expect("Pente has state keys");
        checksum ^= key.rotate_left((index % 64) as u32) ^ proof_id;
        checksum = checksum.wrapping_mul(0x100_0000_01b3);
        println!(
            "position={index} ply={} legal={} key={key:016x} proof={} avg_ms={:.3}",
            state.moves(),
            game.legal_actions(state).len(),
            expected
                .map(|action| game.action_label(state, action))
                .unwrap_or_else(|| "-".to_string()),
            elapsed_ms / repeats as f64,
        );
    }
    let total_ms = started.elapsed().as_secs_f64() * 1e3;
    println!(
        "summary positions={positions} repeats={repeats} proofs={proofs} checksum={checksum:016x} total_ms={total_ms:.3} avg_ms={:.3} slowest_ms={slowest_ms:.3}",
        total_ms / (positions * repeats) as f64,
    );
}
