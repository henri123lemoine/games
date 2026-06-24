//! The chess instantiation of `solvers::azero`'s batched park/resume PUCT
//! search, for the in-wasm CPU bot. A thin wrapper binding [`Chess`] +
//! [`PlanesEncoder`], keeping the `Board`/history API, and turning on cycle
//! draws: a position already seen in the game (`history`) or earlier on the
//! current descent backs up a draw, so the deployed bot sees threefold
//! repetition the way self-play did. The algorithm itself lives in
//! `solvers::azero` — this only adapts it to chess's board and history.

use std::collections::HashMap;

use chess::encode::PlanesEncoder;
use chess::{Board, Chess, Move};
use game_core::Rng;
use solvers::azero::{self, PuctConfig};

pub use solvers::azero::Gather;

/// Search knobs for the chess bot. `cycle_draws` is forced on (chess needs
/// repetition awareness); the rest mirror `solvers::azero::PuctConfig`.
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

    /// Resolves `results` (aligned with the previous `Gather::Requests`), then
    /// gathers the next batch of leaves or finishes. `history` maps position
    /// keys to occurrence counts in the game so far.
    pub fn advance(
        &mut self,
        board: &Board,
        history: &HashMap<u64, u8>,
        cfg: &MctsConfig,
        rng: &mut Rng,
        results: Vec<azero::EvalResult>,
    ) -> Gather {
        let cfg = PuctConfig {
            sims: cfg.sims,
            c_puct: cfg.c_puct,
            fpu: cfg.fpu,
            dirichlet_alpha: cfg.dirichlet_alpha,
            root_noise: cfg.root_noise,
            max_leaves: cfg.max_leaves,
            cycle_draws: true,
            forced_playouts_k: 0.0,
        };
        self.0.advance(
            &Chess,
            &PlanesEncoder,
            board,
            &cfg,
            rng,
            results,
            &|key| history.get(&key).copied().unwrap_or(0) > 0,
            None,
        )
    }

    /// Visit counts over the root's moves, aligned with `root_moves`.
    pub fn root_visits(&self) -> &[u32] {
        self.0.root_visits()
    }

    pub fn root_moves(&self) -> &[Move] {
        self.0.root_actions()
    }

    /// Visit-weighted mean value of the root position (player to move).
    pub fn root_value(&self) -> f64 {
        self.0.root_value()
    }

    /// Extracts the subtree under the root's edge `e` for reuse after that move
    /// is played. Returns `None` if the child was never expanded.
    pub fn extract_child(self, e: usize) -> Option<Tree> {
        self.0.extract_child(e)
    }
}

/// Drives `search` to its visit budget against a synchronous evaluator — the
/// advance/evaluate loop the in-wasm CPU bot runs.
pub fn run_to_done(
    search: &mut Search,
    board: &Board,
    history: &HashMap<u64, u8>,
    cfg: &MctsConfig,
    rng: &mut Rng,
    mut eval: impl FnMut(&[azero::EvalRequest]) -> Vec<azero::EvalResult>,
) {
    let mut results = Vec::new();
    while let Gather::Requests(reqs) =
        search.advance(board, history, cfg, rng, std::mem::take(&mut results))
    {
        results = eval(&reqs);
    }
}
