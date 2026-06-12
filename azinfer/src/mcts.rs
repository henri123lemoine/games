//! The chess instantiation of `solvers::azero`'s batched park/resume PUCT
//! search (where the algorithm lives — see its docs for the search itself).
//! This wrapper binds [`Chess`] + [`PlanesEncoder`], keeps the trainer-facing
//! `Board`/history API, and turns on cycle draws: a position that already
//! occurred in the game (`history`) or earlier on the current descent path
//! backs up a draw — without it, self-play shuffles into threefold draws the
//! tree cannot see.

use std::collections::HashMap;

use chess::encode::PlanesEncoder;
use chess::{Board, Chess, Move};
use game_core::Rng;
use solvers::azero::{self, PuctConfig};

pub use solvers::azero::Gather;

#[derive(Clone, Copy)]
pub struct MctsConfig {
    pub sims: u32,
    pub c_puct: f32,
    pub fpu: f32,
    pub dirichlet_alpha: f64,
    /// Weight of Dirichlet noise mixed into the root prior; 0 disables.
    pub root_noise: f32,
    /// Leaves gathered per `advance` call (virtual-loss parallelism).
    pub max_leaves: u32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig {
            sims: 320,
            c_puct: 1.6,
            fpu: 0.25,
            dirichlet_alpha: 0.3,
            root_noise: 0.25,
            max_leaves: 8,
        }
    }
}

pub type Tree = azero::Tree<Chess>;

pub struct Search(azero::Search<Chess>);

impl Search {
    /// Starts a search, optionally seeded with a reused subtree.
    pub fn new(reuse: Option<Tree>) -> Search {
        Search(azero::Search::new(reuse))
    }

    /// Resolves `results` (aligned with the previous `Gather::Requests`),
    /// then gathers the next batch of leaves or finishes. `history` maps
    /// position keys to occurrence counts in the game so far.
    pub fn advance(
        &mut self,
        board: &Board,
        history: &HashMap<u64, u8>,
        cfg: &MctsConfig,
        rng: &mut Rng,
        results: Vec<crate::EvalResult>,
    ) -> Gather {
        let cfg = PuctConfig {
            sims: cfg.sims,
            c_puct: cfg.c_puct,
            fpu: cfg.fpu,
            dirichlet_alpha: cfg.dirichlet_alpha,
            root_noise: cfg.root_noise,
            max_leaves: cfg.max_leaves,
            cycle_draws: true,
        };
        self.0
            .advance(&Chess, &PlanesEncoder, board, &cfg, rng, results, &|key| {
                history.get(&key).copied().unwrap_or(0) > 0
            })
    }

    /// Visit counts over the root's moves, aligned with `root_moves`.
    pub fn root_visits(&self) -> &[u32] {
        self.0.root_visits()
    }

    pub fn root_moves(&self) -> &[Move] {
        self.0.root_actions()
    }

    /// Visit-weighted mean value of the root position (player to move):
    /// the search's estimate of the position itself, for value targets.
    pub fn root_value(&self) -> f64 {
        self.0.root_value()
    }

    pub fn root_q(&self) -> f64 {
        self.0.root_q()
    }

    /// Extracts the subtree under the root's edge `e` for reuse after that
    /// move is played. Returns `None` if the child was never expanded.
    pub fn extract_child(self, e: usize) -> Option<Tree> {
        self.0.extract_child(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EvalResult, argmax};
    use chess::legal_moves;

    fn drive_with_uniform_net(board: &Board, cfg: &MctsConfig, rng: &mut Rng) -> Search {
        let mut search = Search::new(None);
        let history = HashMap::new();
        let mut results: Vec<EvalResult> = Vec::new();
        loop {
            match search.advance(board, &history, cfg, rng, std::mem::take(&mut results)) {
                Gather::Requests(reqs) => {
                    results = reqs
                        .iter()
                        .map(|r| EvalResult {
                            priors: vec![1.0 / r.support.len() as f32; r.support.len()],
                            value: 0.0,
                        })
                        .collect();
                }
                Gather::Done => return search,
            }
        }
    }

    #[test]
    fn finds_back_rank_mate_with_uniform_net() {
        let b = Board::from_fen("6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1").unwrap();
        let cfg = MctsConfig {
            sims: 256,
            root_noise: 0.0,
            ..MctsConfig::default()
        };
        let mut rng = Rng::new(7);
        let search = drive_with_uniform_net(&b, &cfg, &mut rng);
        let best = search.root_moves()[argmax(search.root_visits())];
        assert_eq!(best, "e1e8".parse().unwrap());
    }

    #[test]
    fn repeated_position_backs_up_draw() {
        // The only winning try for White is Re8#; if the game history says
        // the position after a rook shuffle already occurred, search must
        // treat that branch as a draw, not as fresh territory.
        let b = Board::from_fen("6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1").unwrap();
        let cfg = MctsConfig {
            sims: 128,
            root_noise: 0.0,
            ..MctsConfig::default()
        };
        let mut rng = Rng::new(3);
        // Mark every possible successor as already seen except the mate.
        let mut history = HashMap::new();
        for m in legal_moves(&b) {
            if m != "e1e8".parse().unwrap() {
                let mut nb = b.clone();
                nb.apply(m);
                history.insert(nb.key(), 1);
            }
        }
        let mut search = Search::new(None);
        let mut results: Vec<EvalResult> = Vec::new();
        while let Gather::Requests(reqs) =
            search.advance(&b, &history, &cfg, &mut rng, std::mem::take(&mut results))
        {
            results = reqs
                .iter()
                .map(|r| EvalResult {
                    priors: vec![1.0 / r.support.len() as f32; r.support.len()],
                    value: 0.0,
                })
                .collect();
        }
        let best = search.root_moves()[argmax(search.root_visits())];
        assert_eq!(best, "e1e8".parse().unwrap());
    }

    #[test]
    fn extract_child_preserves_subtree() {
        let b = Board::start();
        let cfg = MctsConfig {
            sims: 128,
            root_noise: 0.0,
            ..MctsConfig::default()
        };
        let mut rng = Rng::new(11);
        let search = drive_with_uniform_net(&b, &cfg, &mut rng);
        let choice = argmax(search.root_visits());
        let child_visits = search.root_visits()[choice];
        assert!(child_visits > 1, "best move got visits");
        let tree = search.extract_child(choice).expect("best child expanded");
        let reused = Search(azero::Search::new(Some(tree)));
        let total: u32 = reused.root_visits().iter().sum();
        assert_eq!(
            total + 1,
            child_visits,
            "extracted subtree keeps every visit except the expansion eval"
        );
    }
}
