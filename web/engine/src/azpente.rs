//! The AlphaZero Pente bot's wasm surface. Mirrors `azgo`: the batched
//! park/resume PUCT search runs here, the page evaluates parked leaves with
//! WebGPU and feeds them back (`advance` → `batch_*` → `advance` … until it
//! returns 0, then `best`); without a GPU `play_cpu` runs the whole search
//! in-wasm against `nn-infer`'s reference forward. One instance mirrors one
//! game: `push` every applied move — both sides' — so the searched subtree
//! carries over between turns (Pente is deterministic, no chance node).
//!
//! The difference from go is the solver: this bot wires the sound, capture-aware
//! VCF+VCT forcing solver (`pente::PenteProver`) into the search as its
//! [`game_core::TerminalProver`], so it proves a forced win at every leaf the
//! search expands and the MCTS-solver backs it up as an exact ±1 — the same
//! integration the native lab bot uses. The solver is pure Rust, so it compiles
//! to wasm and the browser bot plays the identical search the native one does.
//! Pente has no pass and no ownership head, so there is no pass/adjudication
//! logic.

use game_core::{Game, GameUi, PolicyValueEncoder, Proof, Rng};
use nn_infer::Net;
use pente::encode::PenteEncoder;
use pente::{Pente, PenteProver, PenteState, VcfConfig};
use solvers::azero::{EvalRequest, Gather, PuctConfig, Search, argmax};
use wasm_bindgen::prelude::*;

use crate::eval_batch;

#[wasm_bindgen]
pub struct AzPenteBot {
    game: Pente,
    enc: PenteEncoder,
    state: PenteState,
    search: Search<Pente>,
    cfg: PuctConfig,
    vcf: VcfConfig,
    rng: Rng,
    /// The net, loaded for CPU play; unused on the GPU path (the page evaluates
    /// leaves). `None` until `load_weights`.
    model: Option<Net>,
    /// Requests parked by the last `advance`, awaiting page-side evaluation.
    batch: Vec<EvalRequest>,
    /// The tree holds at least an expanded root (safe to read/extract).
    has_tree: bool,
    /// The last search ran to its visit budget (best move is readable).
    done: bool,
}

#[wasm_bindgen]
impl AzPenteBot {
    /// A fresh bot at the empty `size`×`size` board. Play is deterministic
    /// argmax over visit counts — or the root's proven win when the solver
    /// proves one — with no root noise (full strength); `seed` only feeds
    /// chance-free tie paths. `vcf_depth`/`vcf_nodes` bound the *per-leaf*
    /// forcing solver wired in as the search's prover (the native bot's
    /// per-leaf defaults: depth 7, ~1500 nodes); keep them conservative, the
    /// solver runs at every expanded leaf.
    #[wasm_bindgen(constructor)]
    pub fn new(
        sims: u32,
        max_leaves: u32,
        seed: u32,
        size: usize,
        vcf_depth: u32,
        vcf_nodes: u32,
    ) -> AzPenteBot {
        let game = Pente::new(size);
        let state = game.initial_state();
        AzPenteBot {
            game,
            enc: PenteEncoder::new(size),
            state,
            search: Search::new(None),
            cfg: PuctConfig {
                sims,
                max_leaves,
                root_noise: 0.0,
                ..PuctConfig::default()
            },
            vcf: VcfConfig::for_leaf(vcf_depth, u64::from(vcf_nodes), true),
            rng: Rng::new(u64::from(seed)),
            model: None,
            batch: Vec::new(),
            has_tree: false,
            done: false,
        }
    }

    /// Loads the `AZNET1` net (CPU leaf evaluation needs it; the GPU path does
    /// not, but loading is harmless).
    pub fn load_weights(&mut self, weights: &[u8]) -> Result<(), JsError> {
        let net = Net::parse(weights).map_err(|e| JsError::new(&e))?;
        self.model = Some(net);
        Ok(())
    }

    /// Runs the whole search to its visit budget in-wasm — the forcing solver
    /// proving every expanded leaf — evaluating each parked leaf with the
    /// reference forward, and returns the chosen move.
    pub fn play_cpu(&mut self) -> Result<String, JsError> {
        if !self.batch.is_empty() {
            return Err(JsError::new("play_cpu while evaluations are in flight"));
        }
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| JsError::new("CPU weights not loaded"))?;
        let prover = PenteProver { cfg: self.vcf };
        let mut results = Vec::new();
        while let Gather::Requests(reqs) = self.search.advance(
            &self.game,
            &self.enc,
            &self.state,
            &self.cfg,
            &mut self.rng,
            std::mem::take(&mut results),
            &|_| false,
            Some(&prover),
        ) {
            results = eval_batch(model, &reqs);
        }
        self.has_tree = true;
        self.done = true;
        self.best()
    }

    /// Mirrors an applied move (either side's): advances the internal board and
    /// reuses the searched subtree under that move when there is one. `coord` is
    /// a board label (`"k10"`).
    pub fn push(&mut self, coord: &str) -> Result<(), JsError> {
        if !self.batch.is_empty() {
            return Err(JsError::new("push while evaluations are in flight"));
        }
        let action = self
            .game
            .parse_action(&self.state, coord)
            .ok_or_else(|| JsError::new(&format!("'{coord}' is not legal here")))?;
        let idx = self
            .game
            .legal_actions(&self.state)
            .iter()
            .position(|&a| a == action)
            .expect("parse_action returns a legal action");
        let reuse = if self.has_tree {
            let search = std::mem::replace(&mut self.search, Search::new(None));
            search.extract_child(idx)
        } else {
            None
        };
        self.has_tree = reuse.is_some();
        self.search = Search::new(reuse);
        self.done = false;
        self.game.apply(&mut self.state, action);
        Ok(())
    }

    /// Resumes the search with the page's evaluations for the previous batch
    /// (pass empty arrays on the first call), gathers the next batch, and
    /// returns its size — 0 means the move is ready and `best` is readable. The
    /// forcing solver proves every expanded leaf as the search runs; a root the
    /// solver proves a win ends the search early (the next batch is empty).
    pub fn advance(&mut self, priors: &[f32], values: &[f32]) -> Result<u32, JsError> {
        let results = crate::unpack_eval_results(&self.batch, priors, values)
            .map_err(|e| JsError::new(&e))?;
        self.batch.clear();
        let prover = PenteProver { cfg: self.vcf };
        match self.search.advance(
            &self.game,
            &self.enc,
            &self.state,
            &self.cfg,
            &mut self.rng,
            results,
            &|_| false,
            Some(&prover),
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

    /// Features of the pending batch, flat `[n × PLANES·size²]`.
    pub fn batch_features(&self) -> Vec<f32> {
        crate::batch_features(&self.batch)
    }

    /// Legal policy indices of the pending batch, flat; `batch_offsets`
    /// delimits the per-request runs.
    pub fn batch_support(&self) -> Vec<u16> {
        crate::batch_support(&self.batch)
    }

    /// `n + 1` prefix offsets into `batch_support` / the flat priors.
    pub fn batch_offsets(&self) -> Vec<u32> {
        crate::batch_offsets(&self.batch)
    }

    /// The chosen move as a board label (`"k10"`): the root's solver-proven
    /// forced win when the search proved one, otherwise the argmax over root
    /// visits.
    pub fn best(&self) -> Result<String, JsError> {
        if !self.done {
            return Err(JsError::new("search is not done"));
        }
        // A solver-proven root win is exact — play the proven move over the visit
        // argmax. `best_proven_action` is correct for both a proof bubbled up
        // from a winning child and a root the prover proves *directly* (its
        // witnessing move pins the edge in the search's `resolve`).
        let idx = self
            .search
            .best_proven_action()
            .unwrap_or_else(|| argmax(self.search.root_visits()));
        let action = self.search.root_actions()[idx];
        Ok(self.game.action_label(&self.state, action))
    }

    /// `""` — Pente has no ownership head, so no adjudicated result; the
    /// engine's literal result stands. Kept for the shared `azFinalResult` op.
    pub fn final_result(&self) -> String {
        String::new()
    }

    /// `{"value":…,"pairs":[b,w]}` for the position-quality readout: one net
    /// forward on the current root (no search, mirroring Go's snapshot).
    /// `value` is Black's win probability in `[0,1]` (the net's mover-POV value
    /// flipped to Black and mapped from `[-1,1]`). Pente has no score head, so
    /// the captured-pair counts (Black, White) stand in for a material readout.
    /// Empty string until the net is loaded (the GPU path skips `load_weights`).
    pub fn eval(&self) -> String {
        let Some(model) = self.model.as_ref() else {
            return String::new();
        };
        let planes = self.enc.encode_state(&self.game, &self.state);
        let value = f64::from(model.forward(&planes, &[]).value);
        let to_black = if self.state.to_move() == 0 { 1.0 } else { -1.0 };
        let black_value = (value * to_black).clamp(-1.0, 1.0);
        let win_prob = ((black_value + 1.0) / 2.0).clamp(0.0, 1.0);
        let pairs = self.state.pairs();
        format!(
            "{{\"value\":{win_prob},\"pairs\":[{},{}]}}",
            pairs[0], pairs[1]
        )
    }

    /// `{"value":…,"sims":…}` — the root's searched value (side to move) and
    /// total visits, for a thinking readout. A solver-proven root win reports a
    /// decisive value.
    pub fn stats(&self) -> String {
        if self.search.root_proof() == Some(Proof::Win) {
            let sims: u32 = self.search.root_visits().iter().sum();
            return crate::stats_json(1.0, sims);
        }
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
        crate::stats_json(value, sims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pente::PenteAction;

    /// A synthetic `AZNET1` Pente net of the right shape (8 planes, no
    /// ownership head): uniform tiny fill, just enough to forward.
    fn synth_net(blocks: usize, c: usize, size: usize) -> Vec<u8> {
        let planes = pente::encode::PLANES;
        let floats = c * planes * 9
            + c
            + blocks * 2 * (c * c * 9 + c)
            + (c * c + c)
            + (3 * c * c + c)
            + c
            + (3 * c + 1)
            + (c * c + c)
            + (128 * 3 * c + 128)
            + (128 + 1);
        let arch = nn_infer::Arch {
            blocks,
            channels: c,
            planes,
            size,
            scalars: 0,
            head: nn_infer::HeadKind::GlobalPoolSpatial,
            policy_len: 0,
            flags: nn_infer::HeadFlags(0),
            value_seats: 1,
        };
        let mut b = arch.header_bytes();
        for _ in 0..floats {
            b.extend_from_slice(&0.02f32.to_le_bytes());
        }
        b
    }

    #[test]
    fn cpu_play_returns_a_legal_move_and_mirrors() {
        let mut bot = AzPenteBot::new(8, 8, 7, 9, 8, 4000);
        bot.load_weights(&synth_net(2, 6, 9)).unwrap();
        // Black's forced opening is the center; the bot must return it.
        let center = bot
            .game
            .action_label(&bot.state, PenteAction(bot.game.center()));
        let first = bot.play_cpu().unwrap();
        assert_eq!(first, center, "Black opens at the forced center");
        bot.push(&first).unwrap();
        // White replies somewhere legal.
        let reply = bot.play_cpu().unwrap();
        let action = bot.game.parse_action(&bot.state, &reply);
        assert!(action.is_some(), "reply '{reply}' is a legal move");
    }

    #[test]
    fn plays_a_solver_proven_forcing_win() {
        // A position with four black stones in a row and both ends open: the
        // forcing solver — wired into the search as its prover — proves the root
        // a one-move win, and the bot plays the proof-witnessing move (the
        // winning placement) rather than the visit argmax. The solver runs at
        // the root leaf, so the win is proven on the first net round-trip even
        // with a uniform (uninformative) synthetic net.
        let game = Pente::new(9);
        let state = game.parse_state(
            &[
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . X X X X . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
            ],
            0,
            [0, 0],
        );
        let mut bot = AzPenteBot::new(64, 8, 1, 9, 8, 4000);
        bot.load_weights(&synth_net(2, 6, 9)).unwrap();
        bot.game = game;
        bot.enc = PenteEncoder::new(9);
        bot.state = state;
        let mv = bot.play_cpu().unwrap();
        let completes = ["b5", "g5"]; // either open end completes the five
        assert!(
            completes.contains(&mv.as_str()),
            "the bot plays the solver-proven winning move, got {mv}"
        );
        assert_eq!(
            bot.search.root_proof(),
            Some(Proof::Win),
            "the root must be proven a win"
        );
    }
}
