//! Game-agnostic head-to-head evaluation: play any agents in any [`Game`],
//! for two or more players.

use crate::{Game, Turn};

/// Picks an action at a decision node — the index into [`Game::legal_actions`] —
/// given the state, which player is to move, and a private stream of
/// randomness for mixed strategies and stochastic search. Deterministic agents
/// ignore `rng`. Immutable so agents can be shared across seats/games.
pub trait Agent<G: Game> {
    fn act(&self, game: &G, state: &G::State, player: usize, rng: &mut Rng) -> usize;
}

impl<G: Game, F: Fn(&G, &G::State, usize, &mut Rng) -> usize> Agent<G> for F {
    fn act(&self, game: &G, state: &G::State, player: usize, rng: &mut Rng) -> usize {
        self(game, state, player, rng)
    }
}

/// Plays uniformly at random — the universal baseline for any game.
pub struct RandomAgent;

impl<G: Game> Agent<G> for RandomAgent {
    fn act(&self, game: &G, state: &G::State, _player: usize, rng: &mut Rng) -> usize {
        rng.below(game.legal_actions(state).len())
    }
}

/// A minimal reproducible PRNG so matches are deterministic given a seed.
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    /// The next raw 64-bit draw — for deriving reproducible sub-seeds (e.g.
    /// per-rollout streams in parallel simulations).
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform integer in `[0, n)`. `n` must be positive.
    pub fn below(&mut self, n: usize) -> usize {
        debug_assert!(n > 0);
        ((self.unit() * n as f64) as usize).min(n - 1)
    }

    /// Samples an index proportionally to non-negative `weights` (a mixed
    /// strategy, a chance distribution). Robust to weights that do not sum
    /// to exactly 1; floating-point shortfall lands on the last index.
    pub fn pick(&mut self, weights: &[f64]) -> usize {
        debug_assert!(!weights.is_empty());
        let total: f64 = weights.iter().sum();
        let mut target = self.unit() * total;
        for (i, w) in weights.iter().enumerate() {
            target -= w;
            if target < 0.0 {
                return i;
            }
        }
        weights.len() - 1
    }
}

/// Play one game from the initial state; `agents[p]` controls player `p`.
/// Returns the terminal state — score it with [`Game::returns`], [`winner`],
/// or [`win_share`].
pub fn play_n<G: Game>(game: &G, agents: &[&dyn Agent<G>], rng: &mut Rng) -> G::State {
    playout_from(game, game.initial_state(), agents, rng)
}

/// Like [`play_n`] but starting from an arbitrary (e.g. mid-game) state — used
/// for Monte-Carlo rollouts.
pub fn playout_from<G: Game>(
    game: &G,
    mut s: G::State,
    agents: &[&dyn Agent<G>],
    rng: &mut Rng,
) -> G::State {
    while !game.is_terminal(&s) {
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
                let i = agents[p].act(game, &s, p, rng);
                game.apply(&mut s, actions[i]);
            }
        }
    }
    s
}

/// The unique winner of a terminal state — the player with strictly greatest
/// return — or `None` when the top return is shared (a draw).
pub fn winner<G: Game>(game: &G, terminal: &G::State) -> Option<usize> {
    let (best, _, count) = top_return(game, terminal);
    if count == 1 { Some(best) } else { None }
}

/// `player`'s share of the win at a terminal state: 1 for a sole win, `1/k`
/// for a k-way tie at the top, 0 otherwise. Sums to 1 across players, so a
/// field where everyone always draws scores everyone at the fair share.
pub fn win_share<G: Game>(game: &G, terminal: &G::State, player: usize) -> f64 {
    let (_, best_v, count) = top_return(game, terminal);
    if game.returns(terminal, player) == best_v {
        1.0 / count as f64
    } else {
        0.0
    }
}

fn top_return<G: Game>(game: &G, terminal: &G::State) -> (usize, f64, usize) {
    let mut best = 0;
    let mut best_v = game.returns(terminal, 0);
    let mut count = 1;
    for p in 1..game.num_players() {
        let v = game.returns(terminal, p);
        if v > best_v {
            best_v = v;
            best = p;
            count = 1;
        } else if v == best_v {
            count += 1;
        }
    }
    (best, best_v, count)
}

/// Play one two-player game; returns the utility to player 0 (e.g. +1 win,
/// 0 draw, -1 loss).
pub fn play<G: Game>(game: &G, a: &impl Agent<G>, b: &impl Agent<G>, rng: &mut Rng) -> f64 {
    let agents: [&dyn Agent<G>; 2] = [a, b];
    let terminal = play_n(game, &agents, rng);
    game.returns(&terminal, 0)
}

/// Player 0's score rate over `games` two-player matches, seats swapped each
/// game to cancel first-mover bias. Wins count 1, draws ½, losses 0.
pub fn win_rate<G: Game>(
    game: &G,
    a: &impl Agent<G>,
    b: &impl Agent<G>,
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
        } else if r0 == 0.0 {
            0.5
        } else {
            0.0
        };
    }
    score / games as f64
}

/// `hero`'s win share against a field of `baseline` opponents, rotating the
/// hero through every seat so position is unbiased. Ties at the top split
/// credit, so "fair" is exactly `1/players` even in drawish games.
pub fn winrate_vs_field<G: Game>(
    game: &G,
    hero: &dyn Agent<G>,
    baseline: &dyn Agent<G>,
    games: u32,
    seed: u64,
) -> f64 {
    let n = game.num_players();
    let mut rng = Rng::new(seed);
    let mut share = 0.0;
    for g in 0..games {
        let hero_seat = (g as usize) % n;
        let agents: Vec<&dyn Agent<G>> = (0..n)
            .map(|p| if p == hero_seat { hero } else { baseline })
            .collect();
        let terminal = play_n(game, &agents, &mut rng);
        share += win_share(game, &terminal, hero_seat);
    }
    share / games as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Game;

    /// Every game ends immediately in an all-way draw.
    struct AlwaysDraw(usize);

    impl Game for AlwaysDraw {
        type State = bool; // played?
        type Action = u8;

        fn num_players(&self) -> usize {
            self.0
        }
        fn initial_state(&self) -> bool {
            false
        }
        fn turn(&self, _s: &bool) -> Turn {
            Turn::Player(0)
        }
        fn is_terminal(&self, s: &bool) -> bool {
            *s
        }
        fn returns(&self, _s: &bool, _player: usize) -> f64 {
            0.0
        }
        fn legal_actions(&self, _s: &bool) -> Vec<u8> {
            vec![0]
        }
        fn chance_outcomes(&self, _s: &bool) -> Vec<(u8, f64)> {
            Vec::new()
        }
        fn apply(&self, s: &mut bool, _a: u8) {
            *s = true;
        }
        fn infoset_key(&self, s: &bool, _player: usize) -> u64 {
            *s as u64
        }
    }

    #[test]
    fn draws_score_half_not_a_seat_zero_win() {
        let game = AlwaysDraw(2);
        let mut rng = Rng::new(1);
        assert_eq!(play(&game, &RandomAgent, &RandomAgent, &mut rng), 0.0);
        assert_eq!(win_rate(&game, &RandomAgent, &RandomAgent, 10, 3), 0.5);
    }

    #[test]
    fn ties_split_win_share_and_have_no_winner() {
        let game = AlwaysDraw(3);
        let mut rng = Rng::new(1);
        let agents: [&dyn Agent<AlwaysDraw>; 3] = [&RandomAgent, &RandomAgent, &RandomAgent];
        let terminal = play_n(&game, &agents, &mut rng);
        assert_eq!(winner(&game, &terminal), None);
        for p in 0..3 {
            assert!((win_share(&game, &terminal, p) - 1.0 / 3.0).abs() < 1e-12);
        }
        let share = winrate_vs_field(&game, &RandomAgent, &RandomAgent, 9, 5);
        assert!(
            (share - 1.0 / 3.0).abs() < 1e-12,
            "an all-draw field scores everyone at fair share, got {share}"
        );
    }
}
