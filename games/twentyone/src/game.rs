//! Twenty-One expressed as a [`cfr_core::Game`], adapting [`crate::env::Env`].
//!
//! The engine's controllable-chance API (`deal_specific` / `draw_specific` /
//! `stand`) lets us surface chance explicitly: between rounds a chance node deals
//! the four opening cards, and a `Draw` decision leads to a chance node that
//! reveals the drawn card. Information sets reuse the engine's lossless
//! `sufficient_key`. The full game is far too large for the generic full-tree
//! solver — its purpose is to host Twenty-One in the same framework (and validate
//! the wrapper); solving the real game uses the specialized decomposed solver in [`crate::solver`].

use crate::env::Env;
use cfr_core::{Game, Turn};

#[derive(Clone, Copy, Debug)]
pub enum Action {
    Stand,
    Draw,
    /// Chance: the specific card revealed by a draw.
    DrawCard(u8),
    /// Chance: the four opening cards (p0 up, p1 up, p0 down, p1 down).
    Deal(u8, u8, u8, u8),
}

#[derive(Clone)]
pub struct T21State {
    env: Env,
    /// True at the chance node that resolves a just-chosen `Draw`.
    drawing: bool,
}

/// Twenty-One with a configurable starting heart count (6 = the full game).
pub struct TwentyOne {
    pub start_hearts: u8,
}

impl Default for TwentyOne {
    fn default() -> Self {
        Self { start_hearts: 6 }
    }
}

impl TwentyOne {
    pub fn new(start_hearts: u8) -> Self {
        Self { start_hearts }
    }
}

impl Game for TwentyOne {
    type State = T21State;
    type Action = Action;

    fn initial_state(&self) -> T21State {
        T21State {
            env: Env::from_state([self.start_hearts, self.start_hearts], 1),
            drawing: false,
        }
    }

    fn turn(&self, s: &T21State) -> Turn {
        if s.env.is_game_over() {
            return Turn::Player(0); // unused; is_terminal guards
        }
        if s.drawing || !s.env.round_active() {
            Turn::Chance
        } else {
            Turn::Player(s.env.current_player())
        }
    }

    fn is_terminal(&self, s: &T21State) -> bool {
        s.env.is_game_over()
    }

    fn returns(&self, s: &T21State, player: usize) -> f64 {
        s.env.utility(player)
    }

    fn legal_actions(&self, s: &T21State) -> Vec<Action> {
        if s.env.deck_mask() != 0 {
            vec![Action::Draw, Action::Stand]
        } else {
            vec![Action::Stand]
        }
    }

    fn chance_outcomes(&self, s: &T21State) -> Vec<(Action, f64)> {
        if s.drawing {
            let mask = s.env.deck_mask();
            let n = mask.count_ones() as f64;
            (0..11u8)
                .filter(|&c| mask & (1 << c) != 0)
                .map(|c| (Action::DrawCard(c + 1), 1.0 / n))
                .collect()
        } else {
            // Deal four distinct opening cards from 1..=11, each ordering uniform.
            let p = 1.0 / (11.0 * 10.0 * 9.0 * 8.0);
            let mut out = Vec::with_capacity(7920);
            for a in 1..=11u8 {
                for b in 1..=11u8 {
                    if b == a {
                        continue;
                    }
                    for c in 1..=11u8 {
                        if c == a || c == b {
                            continue;
                        }
                        for d in 1..=11u8 {
                            if d == a || d == b || d == c {
                                continue;
                            }
                            out.push((Action::Deal(a, b, c, d), p));
                        }
                    }
                }
            }
            out
        }
    }

    fn apply(&self, s: &mut T21State, a: Action) {
        match a {
            Action::Deal(p0u, p1u, p0d, p1d) => {
                s.env.deal_specific([p0u, p1u, p0d, p1d]).unwrap();
            }
            Action::Draw => {
                s.drawing = true;
            }
            Action::DrawCard(card) => {
                s.env.draw_specific(card).unwrap();
                s.drawing = false;
            }
            Action::Stand => {
                s.env.stand().unwrap();
            }
        }
    }

    fn infoset_key(&self, s: &T21State, player: usize) -> u64 {
        s.env.sufficient_key(player)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A uniform-random play-through reaches a terminal with valid zero-sum
    /// utilities — exercising the chance/decision wiring against the real engine.
    #[test]
    fn random_playthrough_is_well_formed() {
        let game = TwentyOne::new(6);
        let mut rng: u64 = 0x1234_5678;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for _ in 0..200 {
            let mut s = game.initial_state();
            let mut steps = 0;
            while !game.is_terminal(&s) {
                steps += 1;
                assert!(steps < 10_000, "game should terminate");
                match game.turn(&s) {
                    Turn::Chance => {
                        let outs = game.chance_outcomes(&s);
                        let total: f64 = outs.iter().map(|(_, p)| p).sum();
                        assert!((total - 1.0).abs() < 1e-9, "chance sums to 1");
                        let a = outs[(next() as usize) % outs.len()].0;
                        game.apply(&mut s, a);
                    }
                    Turn::Player(_) => {
                        let acts = game.legal_actions(&s);
                        let a = acts[(next() as usize) % acts.len()];
                        game.apply(&mut s, a);
                    }
                }
            }
            assert_eq!(
                game.returns(&s, 0) + game.returns(&s, 1),
                0.0,
                "Twenty-One is zero-sum at terminal"
            );
        }
    }

    /// The same game-agnostic arena that runs on Liar's Dice also runs on
    /// Twenty-One: two random agents play to a valid win rate near 0.5.
    #[test]
    fn arena_runs_on_twentyone() {
        let game = TwentyOne::new(6);
        let rando = |g: &TwentyOne, s: &T21State, _p: usize, r: f64| {
            let n = g.legal_actions(s).len();
            ((r * n as f64) as usize).min(n - 1)
        };
        let wr = cfr_core::win_rate(&game, &rando, &rando, 400, 1);
        assert!((0.3..0.7).contains(&wr), "random vs random ~0.5, got {wr}");
    }
}
