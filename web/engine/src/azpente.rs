//! The AlphaZero Pente bot's wasm surface. Mirrors `azgo`: the batched
//! park/resume PUCT search runs here, the page evaluates parked leaves with
//! WebGPU and feeds them back (`advance` → `batch_*` → `advance` … until it
//! returns 0, then `best`); without a GPU `play_cpu` runs the whole search
//! in-wasm against `nn-infer`'s reference forward. One instance mirrors one
//! game: `push` every applied move — both sides' — so the searched subtree
//! carries over between turns (Pente is deterministic, no chance node).
//!
//! The difference from go is the hybrid: before deferring to the net-MCTS move
//! this bot runs the same sound, capture-aware VCF root solver the native lab
//! bot does (`pente::winning_move` with a move-time `VcfConfig`) and plays a
//! proven forced win immediately. The VCF is pure Rust, so it compiles to wasm
//! and the browser bot plays the identical hybrid the native one does. Pente
//! has no pass and no ownership head, so there is no pass/adjudication logic.

use game_core::{Game, GameUi, PolicyValueEncoder, Rng};
use nn_infer::Net;
use pente::encode::PenteEncoder;
use pente::{Pente, PenteAction, PenteState, VcfConfig};
use solvers::azero::{EvalRequest, EvalResult, Gather, PuctConfig, Search, argmax};
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
    /// A VCF-proven forced win for the side to move, found once when a move
    /// begins and played in place of the search (cleared by `push`).
    forced: Option<PenteAction>,
    /// The tree holds at least an expanded root (safe to read/extract).
    has_tree: bool,
    /// The last search ran to its visit budget (best move is readable).
    done: bool,
}

#[wasm_bindgen]
impl AzPenteBot {
    /// A fresh bot at the empty `size`×`size` board. Play is deterministic
    /// argmax over visit counts with no root noise — full strength; `seed` only
    /// feeds chance-free tie paths. `vcf_depth`/`vcf_nodes` bound the move-time
    /// forcing solver (the native bot's defaults: depth 8, ~4000 nodes).
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
            vcf: VcfConfig {
                max_depth: vcf_depth,
                max_nodes: u64::from(vcf_nodes),
                ..VcfConfig::default()
            },
            rng: Rng::new(u64::from(seed)),
            model: None,
            batch: Vec::new(),
            forced: None,
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

    /// Whether a sound forcing line wins for the side to move right now; caches
    /// the winning move in `forced` so `best` can return it without searching.
    fn check_forced(&mut self) -> bool {
        if self.forced.is_none() {
            self.forced = pente::winning_move(&self.game, &self.state, self.vcf);
        }
        self.forced.is_some()
    }

    /// Runs the whole search to its visit budget in-wasm, evaluating every
    /// parked leaf with the reference forward, and returns the chosen move —
    /// unless the VCF proves a forced win first, in which case it plays that.
    pub fn play_cpu(&mut self) -> Result<String, JsError> {
        if !self.batch.is_empty() {
            return Err(JsError::new("play_cpu while evaluations are in flight"));
        }
        if self.check_forced() {
            self.done = true;
            return self.best();
        }
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| JsError::new("CPU weights not loaded"))?;
        let mut results = Vec::new();
        while let Gather::Requests(reqs) = self.search.advance(
            &self.game,
            &self.enc,
            &self.state,
            &self.cfg,
            &mut self.rng,
            std::mem::take(&mut results),
            &|_| false,
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
        self.forced = None;
        self.done = false;
        self.game.apply(&mut self.state, action);
        Ok(())
    }

    /// Resumes the search with the page's evaluations for the previous batch
    /// (pass empty arrays on the first call), gathers the next batch, and
    /// returns its size — 0 means the move is ready and `best` is readable. The
    /// first call also runs the VCF: when it proves a forced win the search is
    /// skipped entirely and this returns 0 immediately.
    pub fn advance(&mut self, priors: &[f32], values: &[f32]) -> Result<u32, JsError> {
        if self.batch.is_empty() && self.check_forced() {
            if !priors.is_empty() || !values.is_empty() {
                return Err(JsError::new("no batch outstanding, expected empty results"));
            }
            self.done = true;
            return Ok(0);
        }
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
            &self.game,
            &self.enc,
            &self.state,
            &self.cfg,
            &mut self.rng,
            results,
            &|_| false,
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

    /// The chosen move as a board label (`"k10"`): a VCF-proven forced win when
    /// there is one, otherwise the argmax over root visits.
    pub fn best(&self) -> Result<String, JsError> {
        if let Some(action) = self.forced {
            return Ok(self.game.action_label(&self.state, action));
        }
        if !self.done {
            return Err(JsError::new("search is not done"));
        }
        let idx = argmax(self.search.root_visits());
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
    /// total visits, for a thinking readout. A VCF win reports a decisive value
    /// with no visits.
    pub fn stats(&self) -> String {
        if self.forced.is_some() {
            return "{\"value\":1,\"sims\":0}".to_string();
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
        format!("{{\"value\":{value},\"sims\":{sims}}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn plays_a_proven_forcing_win_without_the_net() {
        // A position with four black stones in a row and both ends open: the
        // VCF proves a one-move win, which the bot must play even though no
        // weights are loaded (the forcing solver is net-free).
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
        let mut bot = AzPenteBot::new(1, 8, 1, 9, 8, 4000);
        bot.game = game;
        bot.enc = PenteEncoder::new(9);
        bot.state = state;
        // No weights loaded: only the VCF can produce a move here.
        assert!(bot.model.is_none());
        let mv = bot.play_cpu().unwrap();
        let completes = ["b5", "g5"]; // either open end completes the five
        assert!(
            completes.contains(&mv.as_str()),
            "VCF plays the winning move, got {mv}"
        );
    }
}
