//! Batched four-player self-play. Every leaf carries absolute-seat values.
//! Network win shares are mapped to the game's zero-sum returns before PUCT
//! backup. With a league checkpoint, one current seat plays three past-policy
//! seats; only current-seat decisions are recorded into replay.

use std::collections::HashMap;

use four_player_chess::encode::{FourPlayerChessEncoder, move_index, shares_to_returns};
use four_player_chess::{EndReason, FourPlayerChess, State};
use game_core::rand::sample_visits;
use game_core::{Game, Rng};
use rayon::prelude::*;
use solvers::azero::{self, EvalRequest, EvalResult, Gather, PuctConfig, Value, argmax};

use super::sample::Sample;
use crate::net::Infer;

#[derive(Clone, Copy)]
pub struct SelfPlayConfig {
    pub puct: PuctConfig,
    pub concurrent: usize,
    pub temp_plies: u16,
    pub ply_cap: u16,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        SelfPlayConfig {
            puct: PuctConfig {
                sims: 96,
                max_leaves: 8,
                cycle_draws: true,
                ..PuctConfig::default()
            },
            concurrent: 64,
            temp_plies: 48,
            ply_cap: 320,
        }
    }
}

#[derive(Default, Clone, Copy)]
pub struct SelfPlayStats {
    pub games: u32,
    pub league_games: u32,
    pub plies: u64,
    pub capped: u32,
    pub cpu_secs: f32,
    pub gpu_secs: f32,
    pub batches: u32,
    pub evals: u64,
}

impl SelfPlayStats {
    pub fn avg_plies(self) -> f32 {
        if self.games == 0 {
            0.0
        } else {
            self.plies as f32 / self.games as f32
        }
    }
}

struct Record {
    state: State,
    policy: Vec<(u16, f32)>,
}

struct Worker {
    state: State,
    search: azero::Search<FourPlayerChess>,
    rng: Rng,
    keys: HashMap<u64, u8>,
    records: Vec<Record>,
    plies: u16,
    past_mask: u8,
    search_uses_past: bool,
}

enum WorkerStep {
    Requests {
        requests: Vec<EvalRequest>,
        past: bool,
    },
    Finished {
        samples: Vec<Sample>,
        plies: u16,
        capped: bool,
        league: bool,
    },
}

type WorkerOutcome = (
    Option<(Vec<Sample>, u16, bool, bool)>,
    Vec<EvalRequest>,
    bool,
);

fn outcome_shares(state: &State) -> [f32; 4] {
    let best = *state.scores.iter().max().expect("four scores");
    let winners = state.scores.iter().filter(|&&score| score == best).count();
    let mut shares = [0.0; 4];
    for (seat, share) in shares.iter_mut().enumerate() {
        if state.scores[seat] == best {
            *share = 1.0 / winners as f32;
        }
    }
    shares
}

fn shares_from(value: Value) -> [f32; 4] {
    match value {
        Value::Seats(values) => values[..4].try_into().expect("four seat values"),
        Value::Mover(_) => panic!("four-player net returned a scalar value"),
    }
}

fn map_results(results: &mut [EvalResult]) {
    for result in results {
        let shares = shares_from(result.value);
        result.value = Value::Seats(shares_to_returns(&shares));
    }
}

impl Worker {
    fn new(seed: u64, cfg: &SelfPlayConfig, league: bool) -> Worker {
        let mut worker = Worker {
            state: FourPlayerChess::with_ply_cap(cfg.ply_cap).initial_state(),
            search: azero::Search::new(None),
            rng: Rng::new(seed),
            keys: HashMap::new(),
            records: Vec::new(),
            plies: 0,
            past_mask: 0,
            search_uses_past: false,
        };
        worker.reset(cfg, league);
        worker
    }

    fn reset(&mut self, cfg: &SelfPlayConfig, league: bool) {
        let game = FourPlayerChess::with_ply_cap(cfg.ply_cap);
        self.state = game.initial_state();
        self.search = azero::Search::new(None);
        self.keys.clear();
        self.keys.insert(self.state.repetition_key(), 1);
        self.records.clear();
        self.plies = 0;
        self.past_mask = if league {
            let current = self.rng.below(4);
            0b1111 & !(1 << current)
        } else {
            0
        };
        self.search_uses_past = self.past_mask & 1 != 0;
    }

    fn advance(&mut self, cfg: &SelfPlayConfig, mut results: Vec<EvalResult>) -> WorkerStep {
        map_results(&mut results);
        let game = FourPlayerChess::with_ply_cap(cfg.ply_cap);
        loop {
            let gather = self.search.advance(
                &game,
                &FourPlayerChessEncoder,
                &self.state,
                &cfg.puct,
                &mut self.rng,
                std::mem::take(&mut results),
                &|key| self.keys.get(&key).copied().unwrap_or(0) > 0,
                None,
            );
            match gather {
                Gather::Requests(requests) => {
                    return WorkerStep::Requests {
                        requests,
                        past: self.search_uses_past,
                    };
                }
                Gather::Done => {
                    if let Some(finished) = self.play_move(&game, cfg) {
                        return finished;
                    }
                }
            }
        }
    }

    fn play_move(&mut self, game: &FourPlayerChess, cfg: &SelfPlayConfig) -> Option<WorkerStep> {
        let visits = self.search.root_visits().to_vec();
        let actions = self.search.root_actions().to_vec();
        let total: u32 = visits.iter().sum();
        let actor = self.state.to_move.index();
        if self.past_mask & (1 << actor) == 0 {
            let policy = actions
                .iter()
                .zip(&visits)
                .map(|(&action, &count)| (move_index(action) as u16, count as f32 / total as f32))
                .collect();
            self.records.push(Record {
                state: self.state.clone(),
                policy,
            });
        }

        let choice = if self.plies < cfg.temp_plies {
            sample_visits(&visits, &mut self.rng)
        } else {
            argmax(&visits)
        };
        game.apply(&mut self.state, actions[choice]);
        self.plies += 1;
        self.search = azero::Search::new(None);
        if !game.is_terminal(&self.state) {
            self.search_uses_past = self.past_mask & (1 << self.state.to_move.index()) != 0;
        }
        *self.keys.entry(self.state.repetition_key()).or_insert(0) += 1;

        if game.is_terminal(&self.state) {
            let z = outcome_shares(&self.state);
            let samples = self
                .records
                .drain(..)
                .map(|record| Sample {
                    state: record.state,
                    policy: record.policy,
                    z,
                })
                .collect();
            return Some(WorkerStep::Finished {
                samples,
                plies: self.plies,
                capped: self.state.end == EndReason::PlyCap,
                league: self.past_mask != 0,
            });
        }
        None
    }
}

pub struct SelfPlay {
    cfg: SelfPlayConfig,
    workers: Vec<Worker>,
    results: Vec<Vec<EvalResult>>,
    league: bool,
}

impl SelfPlay {
    pub fn new(cfg: SelfPlayConfig, seed: u64) -> SelfPlay {
        let workers = (0..cfg.concurrent)
            .map(|index| Worker::new(mix(seed, index as u64), &cfg, false))
            .collect();
        SelfPlay {
            cfg,
            workers,
            results: (0..cfg.concurrent).map(|_| Vec::new()).collect(),
            league: false,
        }
    }

    fn set_league(&mut self, active: bool) {
        if self.league == active {
            return;
        }
        self.league = active;
        for (worker, results) in self.workers.iter_mut().zip(&mut self.results) {
            worker.reset(&self.cfg, active);
            results.clear();
        }
    }

    pub fn collect(
        &mut self,
        current: &Infer,
        past: Option<&Infer>,
        target_samples: usize,
    ) -> (Vec<Sample>, SelfPlayStats) {
        self.set_league(past.is_some());
        let mut samples = Vec::with_capacity(target_samples + 1024);
        let mut stats = SelfPlayStats::default();
        while samples.len() < target_samples {
            let cpu_start = std::time::Instant::now();
            let cfg = self.cfg;
            let outcomes: Vec<WorkerOutcome> = self
                .workers
                .par_iter_mut()
                .zip(self.results.par_iter_mut())
                .map(
                    |(worker, results)| match worker.advance(&cfg, std::mem::take(results)) {
                        WorkerStep::Requests { requests, past } => (None, requests, past),
                        WorkerStep::Finished {
                            samples,
                            plies,
                            capped,
                            league,
                        } => {
                            worker.reset(&cfg, self.league);
                            let WorkerStep::Requests { requests, past } =
                                worker.advance(&cfg, Vec::new())
                            else {
                                unreachable!("fresh game requests a root evaluation")
                            };
                            (Some((samples, plies, capped, league)), requests, past)
                        }
                    },
                )
                .collect();

            let mut current_reqs = Vec::new();
            let mut past_reqs = Vec::new();
            let mut spans = Vec::with_capacity(outcomes.len());
            for (finished, requests, uses_past) in outcomes {
                if let Some((new, plies, capped, league)) = finished {
                    samples.extend(new);
                    stats.games += 1;
                    stats.league_games += u32::from(league);
                    stats.plies += u64::from(plies);
                    stats.capped += u32::from(capped);
                }
                let target = if uses_past {
                    &mut past_reqs
                } else {
                    &mut current_reqs
                };
                spans.push((uses_past, target.len(), requests.len()));
                target.extend(requests);
            }
            stats.cpu_secs += cpu_start.elapsed().as_secs_f32();
            stats.batches += 1;
            stats.evals += (current_reqs.len() + past_reqs.len()) as u64;

            let gpu_start = std::time::Instant::now();
            let mut current_out = current.forward_batch(&current_reqs);
            let mut past_out = if past_reqs.is_empty() {
                Vec::new()
            } else {
                past.expect("past requests require a league net")
                    .forward_batch(&past_reqs)
            };
            stats.gpu_secs += gpu_start.elapsed().as_secs_f32();
            for (index, (uses_past, start, len)) in spans.into_iter().enumerate().rev() {
                let outputs = if uses_past {
                    &mut past_out
                } else {
                    &mut current_out
                };
                self.results[index] = outputs.split_off(start);
                debug_assert_eq!(self.results[index].len(), len);
            }
        }
        (samples, stats)
    }
}

pub fn mix(a: u64, b: u64) -> u64 {
    game_core::hash::combine(a, b)
}
