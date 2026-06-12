//! Behaviour-locking tests for the game engine.
//!
//! These guard the *rules* against accidental change while the engine is
//! optimized:
//!
//! * `golden_master_trajectory_is_stable` replays a deterministic battery of
//!   games and hashes the entire observable trajectory. Any change to dealing,
//!   stepping, outcome scoring, or damage flips the hash and fails the test —
//!   forcing the change to be acknowledged (and the constant updated on purpose).
//! * `rules_hold_over_random_play` checks the actual rule invariants
//!   (winner = closest to 21 without busting, damage = round number, hearts
//!   bounds, game-over condition) over many random games, so a refactor that
//!   keeps the hash by coincidence still can't violate the rules.

use twentyone::{Action, Env};

/// Simple deterministic policy: draw while below 17, else stand.
fn threshold_action(total: u8) -> Action {
    if total < 17 {
        Action::Draw
    } else {
        Action::Stand
    }
}

struct Fnv(u64);
impl Fnv {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    fn write_u8(&mut self, b: u8) {
        self.0 ^= b as u64;
        self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
    }
    fn write(&mut self, v: u64) {
        for b in v.to_le_bytes() {
            self.write_u8(b);
        }
    }
}

/// Reconstruct both players' final totals at a round end from the public up
/// cards and the revealed down cards.
fn final_totals(env: &Env) -> (u8, u8) {
    let (u0, l0, u1, l1) = env.last_public_up().expect("up cards at round end");
    let [d0, d1] = env.last_reveal().expect("down cards at round end");
    let s0: u8 = u0[..l0 as usize].iter().sum::<u8>() + d0;
    let s1: u8 = u1[..l1 as usize].iter().sum::<u8>() + d1;
    (s0, s1)
}

#[test]
fn golden_master_trajectory_is_stable() {
    let mut h = Fnv::new();
    for seed in 0..256u64 {
        let mut env = Env::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        while !env.is_game_over() {
            env.start_new_round().unwrap();
            h.write(env.round() as u64);
            loop {
                let p = env.current_player();
                let obs = env.observation(p);
                h.write_u8(p as u8);
                h.write_u8(obs.self_total);
                h.write_u8(obs.opp_face_up);
                h.write_u8(obs.deck_count);
                let action = threshold_action(obs.self_total);
                h.write_u8(matches!(action, Action::Draw) as u8);
                let res = env.step(action).unwrap();
                if let Some(o) = res.outcome {
                    h.write_u8(o.winner.map(|w| w as u8).unwrap_or(2));
                    h.write_u8(o.damage);
                    h.write_u8(env.hearts(0));
                    h.write_u8(env.hearts(1));
                }
                if res.round_over || res.game_over {
                    break;
                }
            }
        }
    }
    // Frozen characterization value. If an intentional rules/engine change moves
    // it, re-run and update this constant deliberately.
    assert_eq!(
        h.0, 0xdd0a_6796_cd90_88b0,
        "trajectory digest changed: {:#018x}",
        h.0
    );
}

#[test]
fn rules_hold_over_random_play() {
    const START_HEARTS: u8 = 6;
    for seed in 0..2000u64 {
        let mut env = Env::new(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) | 1);
        let mut prev_round = 0u8;
        let mut steps_guard = 0u32;
        while !env.is_game_over() {
            let h0_before = env.hearts(0);
            let h1_before = env.hearts(1);
            assert!(h0_before <= START_HEARTS && h1_before <= START_HEARTS);
            assert!(
                h0_before > 0 && h1_before > 0,
                "active game has both players alive"
            );

            let round = env.round();
            assert!(round >= prev_round, "round number never decreases");
            prev_round = round;

            env.start_new_round().unwrap();
            let outcome = loop {
                steps_guard += 1;
                assert!(steps_guard < 100_000, "game made no progress");
                let p = env.current_player();
                let obs = env.observation(p);
                let res = env.step(threshold_action(obs.self_total)).unwrap();
                if res.round_over || res.game_over {
                    break res.outcome;
                }
            };

            let (h0_after, h1_after) = (env.hearts(0), env.hearts(1));
            let outcome = outcome.expect("a finished round reports an outcome");
            let (t0, t1) = final_totals(&env);

            // Winner rule: if exactly one player busts, the other wins. Otherwise
            // (both safe, or both bust) the smaller distance to 21 wins, and equal
            // distance is a tie. Note the both-bust case is decided by who busted
            // by less, not treated as a draw.
            let busted = |t: u8| t > 21;
            let expected_winner = match (busted(t0), busted(t1)) {
                (false, true) => Some(0),
                (true, false) => Some(1),
                _ => {
                    let d0 = t0.abs_diff(21);
                    let d1 = t1.abs_diff(21);
                    match d0.cmp(&d1) {
                        std::cmp::Ordering::Less => Some(0),
                        std::cmp::Ordering::Greater => Some(1),
                        std::cmp::Ordering::Equal => None,
                    }
                }
            };
            assert_eq!(
                outcome.winner, expected_winner,
                "winner must be closest to 21 without busting (totals {t0}/{t1})"
            );

            // Damage equals the round number; the loser loses exactly that many
            // hearts, the winner none; a tie changes nothing.
            assert_eq!(outcome.damage, round, "damage equals the round number");
            match outcome.winner {
                Some(0) => {
                    assert_eq!(h0_after, h0_before);
                    assert_eq!(h1_after, h1_before.saturating_sub(round));
                }
                Some(1) => {
                    assert_eq!(h1_after, h1_before);
                    assert_eq!(h0_after, h0_before.saturating_sub(round));
                }
                Some(w) => panic!("winner must be 0 or 1, got {w}"),
                None => {
                    assert_eq!((h0_after, h1_after), (h0_before, h1_before));
                }
            }
        }
        // The game ends exactly when a player has been reduced to zero hearts.
        assert!(
            env.hearts(0) == 0 || env.hearts(1) == 0,
            "game over implies a dead player"
        );
    }
}
