//! The AlphaZero chess bot's wasm surface. The batched park/resume PUCT
//! search runs here, weight-free; the page evaluates parked leaves with
//! WebGPU and feeds the results back (`advance` → `batch_*` → `advance` …
//! until it returns 0, then `best`). One instance mirrors one game: `push`
//! every applied move — both sides' — so repetition awareness sees the real
//! game history and the searched subtree carries over between turns.

use std::collections::HashMap;

use azinfer::mcts::{Gather, MctsConfig, Search};
use azinfer::{EvalRequest, EvalResult, argmax};
use chess::{Board, Move, legal_moves};
use game_core::Rng;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct AzChessBot {
    board: Board,
    history: HashMap<u64, u8>,
    search: Search,
    cfg: MctsConfig,
    rng: Rng,
    /// Requests parked by the last `advance`, awaiting page-side evaluation.
    batch: Vec<EvalRequest>,
    /// The tree holds at least an expanded root (safe to read/extract).
    has_tree: bool,
    /// The last search ran to its visit budget (best move is readable).
    done: bool,
}

#[wasm_bindgen]
impl AzChessBot {
    /// A fresh bot at the standard start position. Play is deterministic
    /// argmax without root noise; `seed` only feeds chance-free tie paths.
    #[wasm_bindgen(constructor)]
    pub fn new(sims: u32, max_leaves: u32, seed: u32) -> AzChessBot {
        let board = Board::start();
        let mut history = HashMap::new();
        history.insert(board.key(), 1);
        AzChessBot {
            board,
            history,
            search: Search::new(None),
            cfg: MctsConfig {
                sims,
                max_leaves,
                root_noise: 0.0,
                ..MctsConfig::default()
            },
            rng: Rng::new(u64::from(seed)),
            batch: Vec::new(),
            has_tree: false,
            done: false,
        }
    }

    /// Mirrors an applied move (either side's): advances the internal board,
    /// records the position for repetition awareness, and reuses the
    /// searched subtree under that move when there is one.
    pub fn push(&mut self, uci: &str) -> Result<(), JsError> {
        if !self.batch.is_empty() {
            return Err(JsError::new("push while evaluations are in flight"));
        }
        let mv: Move = uci.trim().parse().map_err(|e: String| JsError::new(&e))?;
        let moves = legal_moves(&self.board);
        let Some(idx) = moves.iter().position(|&m| m == mv) else {
            return Err(JsError::new(&format!("'{uci}' is not legal here")));
        };
        let reuse = if self.has_tree {
            let search = std::mem::replace(&mut self.search, Search::new(None));
            search.extract_child(idx)
        } else {
            None
        };
        self.has_tree = reuse.is_some();
        self.search = Search::new(reuse);
        self.done = false;
        self.board.apply(mv);
        *self.history.entry(self.board.key()).or_insert(0) += 1;
        Ok(())
    }

    /// Resumes the search with the page's evaluations for the previous batch
    /// (pass empty arrays on the first call), gathers the next batch, and
    /// returns its size — 0 means the search is done and `best` is ready.
    /// `priors` is the flat concatenation over the batch, aligned with
    /// `batch_offsets`; `values` holds one entry per request.
    pub fn advance(&mut self, priors: &[f32], values: &[f32]) -> Result<u32, JsError> {
        let results = if self.batch.is_empty() {
            if !priors.is_empty() || !values.is_empty() {
                return Err(JsError::new("no batch outstanding, expected empty results"));
            }
            Vec::new()
        } else {
            if values.len() != self.batch.len() {
                return Err(JsError::new(&format!(
                    "expected {} values, got {}",
                    self.batch.len(),
                    values.len()
                )));
            }
            let mut out = Vec::with_capacity(self.batch.len());
            let mut off = 0usize;
            for (req, &value) in self.batch.iter().zip(values) {
                let k = req.support.len();
                if off + k > priors.len() {
                    return Err(JsError::new("priors shorter than the batch support"));
                }
                out.push(EvalResult {
                    priors: priors[off..off + k].to_vec(),
                    value,
                });
                off += k;
            }
            if off != priors.len() {
                return Err(JsError::new("priors longer than the batch support"));
            }
            out
        };
        self.batch.clear();
        match self.search.advance(
            &self.board,
            &self.history,
            &self.cfg,
            &mut self.rng,
            results,
        ) {
            Gather::Requests(reqs) => {
                self.has_tree = true;
                self.batch = reqs;
                Ok(self.batch.len() as u32)
            }
            Gather::Done => {
                self.has_tree = true;
                self.done = true;
                Ok(0)
            }
        }
    }

    /// Features of the pending batch, flat `[n × 18·64]` (board planes).
    pub fn batch_features(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.batch.iter().map(|r| r.features.len()).sum());
        for r in &self.batch {
            out.extend_from_slice(&r.features);
        }
        out
    }

    /// Legal policy indices of the pending batch, flat; `batch_offsets`
    /// delimits the per-request runs.
    pub fn batch_support(&self) -> Vec<u16> {
        let mut out = Vec::with_capacity(self.batch.iter().map(|r| r.support.len()).sum());
        for r in &self.batch {
            out.extend_from_slice(&r.support);
        }
        out
    }

    /// `n + 1` prefix offsets into `batch_support` / the flat priors.
    pub fn batch_offsets(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.batch.len() + 1);
        let mut off = 0u32;
        out.push(0);
        for r in &self.batch {
            off += r.support.len() as u32;
            out.push(off);
        }
        out
    }

    /// The searched move as UCI (argmax over root visits).
    pub fn best(&self) -> Result<String, JsError> {
        if !self.done {
            return Err(JsError::new("search is not done"));
        }
        let moves = self.search.root_moves();
        Ok(moves[argmax(self.search.root_visits())].to_string())
    }

    /// `{"value":…,"sims":…}` — the root's searched value (side to move)
    /// and total visits, for a thinking readout.
    pub fn stats(&self) -> String {
        let sims: u32 = if self.has_tree {
            self.search.root_visits().iter().sum()
        } else {
            0
        };
        let value = if self.has_tree {
            self.search.root_value()
        } else {
            0.0
        };
        format!("{{\"value\":{value},\"sims\":{sims}}}")
    }
}
