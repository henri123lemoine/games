//! Replay samples for four-player chess. Values are absolute-seat win-share
//! distributions, not a mover-relative scalar: one terminal-placement target
//! each for Red, Blue, Yellow, and Green. States are retained directly so the
//! score, delayed-check credit, and en-passant state remain exact.

use std::collections::VecDeque;

use four_player_chess::encode::{FourPlayerChessEncoder, POLICY_LEN};
use four_player_chess::{FourPlayerChess, State};
use game_core::{PolicyValueEncoder, Rng};

use crate::net::NetConfig;
use crate::train::{Batch, TrainSample};

pub struct Sample {
    pub state: State,
    pub policy: Vec<(u16, f32)>,
    pub z: [f32; 4],
}

impl TrainSample for Sample {
    fn expand(
        buf: &VecDeque<Self>,
        n: usize,
        cfg: &NetConfig,
        _value_mix: f32,
        rng: &mut Rng,
    ) -> Vec<Batch> {
        let game = FourPlayerChess::default();
        let enc = FourPlayerChessEncoder;
        let plane_len = cfg.planes as usize * 14 * 14;
        let mut planes = vec![0.0; n * plane_len];
        let mut policy = vec![0.0; n * POLICY_LEN];
        let mut value = vec![0.0; n * 4];
        for row in 0..n {
            let sample = &buf[rng.below(buf.len())];
            let x = enc.encode_state(&game, &sample.state);
            planes[row * plane_len..(row + 1) * plane_len].copy_from_slice(&x);
            for &(index, probability) in &sample.policy {
                policy[row * POLICY_LEN + usize::from(index)] = probability;
            }
            for seat in 0..4 {
                value[row * 4 + seat] = sample.z[seat];
            }
        }
        vec![Batch {
            n,
            size: 14,
            planes,
            policy,
            value,
            ownership: Vec::new(),
            score: Vec::new(),
        }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use four_player_chess::encode::PLANE_COUNT;
    use game_core::Game;

    #[test]
    fn expanded_value_target_is_a_distribution() {
        let game = FourPlayerChess::with_ply_cap(1);
        let sample = Sample {
            state: game.initial_state(),
            policy: vec![(0, 1.0)],
            z: [1.0, 0.0, 0.0, 0.0],
        };
        let mut buf = VecDeque::new();
        buf.push_back(sample);
        let cfg = super::super::run::config(1, 8);
        let batch = Sample::expand(&buf, 1, &cfg, 0.2, &mut Rng::new(1));
        assert_eq!(batch[0].planes.len(), PLANE_COUNT * 14 * 14);
        assert!((batch[0].value.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(batch[0].value, [1.0, 0.0, 0.0, 0.0]);
    }
}
