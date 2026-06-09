//! Game-agnostic head-to-head evaluation: play any two agents in any [`Game`].

use crate::{Game, Turn};

/// Picks an action at a decision node — the index into
/// [`Game::legal_actions`] — given the state and which player is to move.
/// `r` is a fresh uniform random in `[0, 1)` for mixed strategies.
pub trait Agent<G: Game> {
    fn act(&mut self, game: &G, state: &G::State, player: usize, r: f64) -> usize;
}

impl<G: Game, F: FnMut(&G, &G::State, usize, f64) -> usize> Agent<G> for F {
    fn act(&mut self, game: &G, state: &G::State, player: usize, r: f64) -> usize {
        self(game, state, player, r)
    }
}

/// A minimal reproducible PRNG so matches are deterministic given a seed.
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    pub fn unit(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Play one game between `a` (player 0) and `b` (player 1); returns the utility
/// to player 0 (chance sampled with `rng`).
pub fn play<G: Game>(game: &G, a: &mut impl Agent<G>, b: &mut impl Agent<G>, rng: &mut Rng) -> f64 {
    let mut s = game.initial_state();
    loop {
        if game.is_terminal(&s) {
            return game.returns(&s, 0);
        }
        match game.turn(&s) {
            Turn::Chance => {
                let outs = game.chance_outcomes(&s);
                let r = rng.unit();
                let mut acc = 0.0;
                let mut chosen = outs[outs.len() - 1].0;
                for (act, p) in &outs {
                    acc += *p;
                    if r < acc {
                        chosen = *act;
                        break;
                    }
                }
                game.apply(&mut s, chosen);
            }
            Turn::Player(p) => {
                let actions = game.legal_actions(&s);
                let r = rng.unit();
                let i = if p == 0 {
                    a.act(game, &s, 0, r)
                } else {
                    b.act(game, &s, 1, r)
                };
                game.apply(&mut s, actions[i]);
            }
        }
    }
}

/// Player 0's win rate (draws count as half) over `games` matches with seats
/// swapped each game to cancel any first-mover bias.
pub fn win_rate<G: Game>(
    game: &G,
    a: &mut impl Agent<G>,
    b: &mut impl Agent<G>,
    games: u32,
    seed: u64,
) -> f64 {
    let mut rng = Rng::new(seed);
    let mut score = 0.0;
    for g in 0..games {
        let r0 = if g % 2 == 0 {
            play(game, a, b, &mut rng)
        } else {
            -play(game, b, a, &mut rng)
        };
        score += if r0 > 0.0 {
            1.0
        } else if r0 < 0.0 {
            0.0
        } else {
            0.5
        };
    }
    score / games as f64
}
