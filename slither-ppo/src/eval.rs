//! Eval panel: drop the greedy learner into fresh arenas against a single
//! opponent kind and measure the things the project actually cares about — does it
//! get kills (via cut-off: a foe head hitting the learner's body), does it win the
//! match, does it survive. Run vs the random floor (must crush) and vs the
//! heuristic teacher (the gating opponent; rising from ~0 is the real success
//! signal).
//!
//! Greedy (argmax) actions, no exploration noise, on the same even-self-play
//! geometry the late curriculum trains on, so eval numbers track deployment.

use slither_rl::env::{Action, Env};
use slither_rl::world::{START_LENGTH, World, WorldConfig};
use slither_rl::{Heuristic, Rng, random_action};

use tch::Device;

use crate::net::Policy;
use crate::obs_batch;

#[derive(Clone, Copy, Debug, Default)]
pub struct EvalResult {
    /// Overall win share: a kill, or outliving/out-growing the field. The stable
    /// headline metric, comparable across runs (the guardrail).
    pub winrate: f32,
    /// Decisive-predation win share: fraction of games with at least one kill.
    /// This is the encircle/cut-off signal we want to *push* — winning by
    /// trapping a foe, not by farming food and outlasting.
    pub kill_winrate: f32,
    pub learner_kills_per_game: f32,
    pub opp_kills_per_game: f32,
    pub mean_lifespan: f32,
    pub mean_final_len: f32,
}

#[derive(Clone, Copy)]
pub enum Opp {
    Random,
    Heuristic,
}

/// Evaluate the greedy learner over `games` independent 1-v-rest arenas against
/// `opp`. Learner is seat 0; all other seats run `opp`. Forwards are batched
/// across all still-alive games each step — one GPU call per step total.
///
/// `symmetric` picks the world: `true` is the TRUE deployment config — every worm
/// the same `START_LENGTH`, no jitter — so the score is the real win-share a human
/// faces, with NO size head-start for the learner. `false` keeps the old favorable
/// config (learner oversized +30, opponents small) — reported alongside only to
/// show how much the head-start inflated the headline. Keep-best must gate on the
/// symmetric number.
pub fn evaluate(
    policy: &Policy,
    device: Device,
    games: usize,
    steps: usize,
    opp: Opp,
    seed: u64,
    symmetric: bool,
) -> EvalResult {
    let cfg = if symmetric {
        WorldConfig {
            worms: 6,
            seat0_length: START_LENGTH,
            prey_jitter: 0.0,
            ..WorldConfig::default()
        }
    } else {
        WorldConfig {
            worms: 6,
            seat0_length: START_LENGTH + 30.0,
            prey_jitter: 40.0,
            ..WorldConfig::default()
        }
    };

    let mut envs: Vec<Env> = Vec::with_capacity(games);
    let mut heurs: Vec<Vec<Heuristic>> = Vec::with_capacity(games);
    let mut rngs: Vec<Rng> = Vec::with_capacity(games);
    let mut last_obs: Vec<Vec<_>> = Vec::with_capacity(games);
    let mut learner_kills = vec![0u32; games];
    let mut opp_kills = vec![0u32; games];
    let mut lifespan = vec![0u32; games];
    let mut alive = vec![true; games];
    let mut final_len = vec![0.0f32; games];

    for g in 0..games {
        let mut env = Env::new(cfg);
        let world = World::new(seed.wrapping_add(g as u64 * 0x9e3779b9), cfg);
        let n = world.worms.len();
        let obs = env.reset_world(world);
        last_obs.push(obs);
        heurs.push(
            (0..n)
                .map(|s| Heuristic::new(seed ^ (g as u64) ^ (s as u64 * 7)))
                .collect(),
        );
        rngs.push(Rng::new(seed ^ 0xBEEF ^ g as u64));
        envs.push(env);
    }

    for _ in 0..steps {
        // Batch the learner obs of every still-alive game for one forward.
        let mut batch_obs = Vec::with_capacity(games);
        let mut batch_game = Vec::with_capacity(games);
        for g in 0..games {
            if alive[g] {
                batch_obs.push(last_obs[g][0].clone());
                batch_game.push(g);
            }
        }
        if batch_obs.is_empty() {
            break;
        }
        let (grid, scalars) = obs_batch::pack(&batch_obs, device);
        let (turns, boosts) = policy.act_greedy(&grid, &scalars);
        let turns: Vec<i64> = turns.try_into().unwrap();
        let boosts: Vec<i64> = boosts.try_into().unwrap();

        let mut learner_action = vec![Action::default(); games];
        for (row, &g) in batch_game.iter().enumerate() {
            learner_action[g] = Action {
                turn: turns[row] as u8,
                boost: boosts[row] != 0,
            };
        }

        for g in 0..games {
            if !alive[g] {
                continue;
            }
            let env = &mut envs[g];
            let game_heurs = &mut heurs[g];
            let game_rng = &mut rngs[g];
            let n = env.num_worms();
            let mut actions = Vec::with_capacity(n);
            actions.push(learner_action[g]);
            for (seat, heur) in game_heurs.iter_mut().enumerate().take(n).skip(1) {
                let a = match opp {
                    Opp::Random => random_action(game_rng),
                    Opp::Heuristic => heur.act(env.world(), seat),
                };
                actions.push(a);
            }
            let out = env.step(&actions);
            last_obs[g] = out.obs;
            learner_kills[g] += out.kills[0];
            opp_kills[g] += out.kills[1..].iter().sum::<u32>();
            if out.done[0] {
                alive[g] = false;
                final_len[g] = env.world().worms[0].length;
            } else {
                lifespan[g] += 1;
                final_len[g] = env.world().worms[0].length;
            }
        }
    }

    // Win = learner ended ahead: got a kill, or outlived/out-grew the field.
    // kill_wins = the decisive-predation subset (at least one kill).
    let mut wins = 0;
    let mut kill_wins = 0;
    for g in 0..games {
        let me = &envs[g].world().worms[0];
        let biggest_foe = envs[g]
            .world()
            .worms
            .iter()
            .enumerate()
            .filter(|(j, w)| *j != 0 && !w.dead)
            .map(|(_, w)| w.length)
            .fold(0.0f32, f32::max);
        let killed = learner_kills[g] > 0;
        let won = killed || (!me.dead && me.length > biggest_foe * 1.1);
        if won {
            wins += 1;
        }
        if killed {
            kill_wins += 1;
        }
    }

    let gf = games as f32;
    EvalResult {
        winrate: wins as f32 / gf,
        kill_winrate: kill_wins as f32 / gf,
        learner_kills_per_game: learner_kills.iter().sum::<u32>() as f32 / gf,
        opp_kills_per_game: opp_kills.iter().sum::<u32>() as f32 / gf,
        mean_lifespan: lifespan.iter().sum::<u32>() as f32 / gf,
        mean_final_len: final_len.iter().sum::<f32>() / gf,
    }
}
