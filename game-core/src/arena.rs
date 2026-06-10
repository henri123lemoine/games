//! Game-agnostic head-to-head evaluation: play any agents in any [`Game`],
//! for two or more players.

use crate::{Game, Turn};

/// Picks an action at a decision node — the index into [`Game::legal_actions`] —
/// given the state, which player is to move, and a fresh uniform `r` in `[0, 1)`
/// for mixed strategies. Immutable so agents can be shared across seats/games.
pub trait Agent<G: Game> {
    fn act(&self, game: &G, state: &G::State, player: usize, r: f64) -> usize;
}

impl<G: Game, F: Fn(&G, &G::State, usize, f64) -> usize> Agent<G> for F {
    fn act(&self, game: &G, state: &G::State, player: usize, r: f64) -> usize {
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

/// Play one game; `agents[p]` controls player `p`. Returns the winning player
/// (greatest terminal return; ties broken by lowest index).
pub fn play_n<G: Game>(game: &G, agents: &[&dyn Agent<G>], rng: &mut Rng) -> usize {
    playout_from(game, game.initial_state(), agents, rng)
}

/// Like [`play_n`] but starting from an arbitrary (e.g. mid-game) state — used
/// for Monte-Carlo rollouts.
pub fn playout_from<G: Game>(
    game: &G,
    mut s: G::State,
    agents: &[&dyn Agent<G>],
    rng: &mut Rng,
) -> usize {
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
                let i = agents[p].act(game, &s, p, rng.unit());
                game.apply(&mut s, actions[i]);
            }
        }
    }
    let mut best = 0;
    let mut best_v = game.returns(&s, 0);
    for p in 1..game.num_players() {
        let v = game.returns(&s, p);
        if v > best_v {
            best_v = v;
            best = p;
        }
    }
    best
}

/// Play one two-player game; returns the utility to player 0 (+1/-1).
pub fn play<G: Game>(game: &G, a: &impl Agent<G>, b: &impl Agent<G>, rng: &mut Rng) -> f64 {
    let agents: [&dyn Agent<G>; 2] = [a, b];
    if play_n(game, &agents, rng) == 0 {
        1.0
    } else {
        -1.0
    }
}

/// Player 0's win rate over `games` two-player matches, seats swapped each game
/// to cancel first-mover bias.
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
        score += if r0 > 0.0 { 1.0 } else { 0.0 };
    }
    score / games as f64
}

/// `hero`'s win rate against a field of `baseline` opponents, rotating the hero
/// through every seat so position is unbiased. Returns wins / games.
pub fn winrate_vs_field<G: Game>(
    game: &G,
    hero: &dyn Agent<G>,
    baseline: &dyn Agent<G>,
    games: u32,
    seed: u64,
) -> f64 {
    let n = game.num_players();
    let mut rng = Rng::new(seed);
    let mut wins = 0u32;
    for g in 0..games {
        let hero_seat = (g as usize) % n;
        let agents: Vec<&dyn Agent<G>> = (0..n)
            .map(|p| if p == hero_seat { hero } else { baseline })
            .collect();
        if play_n(game, &agents, &mut rng) == hero_seat {
            wins += 1;
        }
    }
    wins as f64 / games as f64
}
