//! End-to-end self-play tests: drive the real [`Simulator`] with the uniform
//! evaluator into a [`ReplayBuffer`], then exercise the buffer's on-demand
//! reconstruction and λ-return / advantage processing over *sim-collected*
//! transitions — proving the evaluator, sim, and buffer compose, not just that
//! each works against synthetic inputs.

use crate::buffer::{ReplayBuffer, Snapshot};
use crate::encode::{EncoderConfig, encode_tokens};
use crate::evaluator::{Phase, UniformEvaluator};
use crate::rules;
use crate::sim::Simulator;

const CAP: usize = 128;

fn collect(num_envs: usize, steps: usize, seed: u64) -> (Simulator, ReplayBuffer) {
    let cfg = EncoderConfig::default();
    let mut sim = Simulator::new(num_envs, cfg, seed, 4000);
    let mut buffer = ReplayBuffer::new(num_envs, CAP, cfg);
    sim.run(&UniformEvaluator, &mut buffer, steps);
    (sim, buffer)
}

#[test]
fn collected_transitions_reencode_to_their_live_obs() {
    // Deployment is 80 placements; 110 steps (< ring capacity) keeps both the
    // late-deployment and the first move-phase transitions resident.
    let (_sim, buffer) = collect(6, 110, 0x5151);
    let cfg = EncoderConfig::default();

    let mut checked_play = 0;
    let mut checked_deploy = 0;
    for env in 0..buffer.num_envs() {
        for step in 0..buffer.capacity() {
            let Some(t) = buffer.get(env, step) else {
                continue;
            };
            let view = buffer.encode_view(env, step).expect("resident slot");
            assert_eq!(view.phase, t.phase);
            assert_eq!(view.player, t.player);

            match &t.snapshot {
                Snapshot::Play { board, to_play } => {
                    let live_obs = encode_tokens(board, *to_play, &cfg);
                    assert_eq!(view.obs, live_obs, "obs reconstruction mismatch");
                    let live_legal: Vec<u16> = rules::legal_mask(board, *to_play)
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &b)| b.then_some(i as u16))
                        .collect();
                    assert_eq!(view.legal, live_legal, "legal mask mismatch");
                    assert!(t.legal.contains(&t.action), "chosen action must be legal");
                    checked_play += 1;
                }
                Snapshot::Deploy { current, .. } => {
                    let live_legal: Vec<u16> =
                        current.legal_types().iter().map(|&p| p as u16).collect();
                    assert_eq!(view.legal, live_legal);
                    assert!(t.legal.contains(&t.action));
                    checked_deploy += 1;
                }
            }
        }
    }
    assert!(checked_play > 0, "expected move-phase transitions");
    assert!(checked_deploy > 0, "expected deployment transitions");
}

#[test]
fn process_data_over_sim_trajectory_is_finite_and_well_formed() {
    let (_sim, buffer) = collect(4, 80, 0x2727);

    let mut total = 0;
    for env in 0..buffer.num_envs() {
        let targets = buffer.process_data(env, 0.8, 0.5);
        for (slot, t) in &targets {
            assert!(
                t.ret.iter().all(|v| v.is_finite()),
                "categorical return must be finite"
            );
            assert!(t.advantage.is_finite(), "advantage must be finite");
            assert!(
                buffer.get(env, *slot).is_some(),
                "target keys a resident slot"
            );
        }
        total += targets.len();
    }
    assert!(total > 0, "expected processed transitions");
}

#[test]
fn old_log_probs_are_a_normalized_log_distribution() {
    let (_sim, buffer) = collect(4, 40, 0x9999);

    for env in 0..buffer.num_envs() {
        for step in 0..buffer.capacity() {
            let Some(t) = buffer.get(env, step) else {
                continue;
            };
            assert_eq!(t.old_log_probs.len(), t.legal.len());
            let mass: f32 = t.old_log_probs.iter().map(|&lp| lp.exp()).sum();
            assert!(
                (mass - 1.0).abs() < 1e-4,
                "log-probs over the legal set must normalize, got {mass}"
            );
            assert!(t.chosen < t.legal.len());
            assert_eq!(t.legal[t.chosen], t.action);
        }
    }
}

#[test]
fn terminal_transitions_carry_zero_sum_signed_rewards() {
    let (_sim, buffer) = collect(8, 400, 0x3131);

    let mut terminals = 0;
    for env in 0..buffer.num_envs() {
        for step in 0..buffer.capacity() {
            let Some(t) = buffer.get(env, step) else {
                continue;
            };
            if t.is_terminating_action {
                assert!(
                    t.terminal_reward.abs() <= 1.0 + 1e-6,
                    "terminal reward in [-1, 1]"
                );
                if t.phase == Phase::Move {
                    terminals += 1;
                }
            } else {
                assert_eq!(t.terminal_reward, 0.0, "non-terminal carries no reward");
            }
        }
    }
    assert!(terminals > 0, "expected recorded terminal transitions");
}
