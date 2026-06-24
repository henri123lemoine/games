//! The AlphaZero go bot's wasm surface. The batched park/resume PUCT search
//! runs here. With a GPU the page evaluates parked leaves with WebGPU and
//! feeds the results back (`advance` → `batch_*` → `advance` … until it
//! returns 0, then `best`). Without a GPU, `load_weights` hands the same
//! `.azweb` net to this bot and `play_cpu` runs the whole search in-wasm
//! against `nn-infer`'s reference forward — identical net, identical search,
//! just no WebGPU (so it stays at the trivial visit budget to keep moves
//! responsive). One instance mirrors one game: `push` every applied move —
//! both sides' — so the searched subtree carries over between turns. Unlike
//! chess there is no repetition history: go's ko lives in the state and cycle
//! draws are off.

use game_core::{Game, GameUi, PolicyValueEncoder, Rng};
use go::encode::GoEncoder;
use go::{Go, GoAction, GoState};
use nn_infer::Net;
use solvers::azero::{EvalRequest, EvalResult, Gather, PuctConfig, Search, argmax};
use wasm_bindgen::prelude::*;

use crate::eval_batch;

/// Ownership past this magnitude assigns a point to a color (for the
/// literal-board agreement check and for adjudicated scoring).
const TAU: f32 = 0.5;

#[wasm_bindgen]
pub struct AzGoBot {
    game: Go,
    enc: GoEncoder,
    size: usize,
    state: GoState,
    search: Search<Go>,
    cfg: PuctConfig,
    rng: Rng,
    /// Requests parked by the last `advance`, awaiting page-side evaluation.
    batch: Vec<EvalRequest>,
    /// The reference net, loaded in both modes: CPU play uses it for leaf
    /// evaluations, and either mode uses its ownership head for the pass
    /// decision. `None` until `load_weights` (and ownership-less if the net
    /// carries no ownership head).
    model: Option<Net>,
    /// The tree holds at least an expanded root (safe to read/extract).
    has_tree: bool,
    /// The last search ran to its visit budget (best move is readable).
    done: bool,
}

#[wasm_bindgen]
impl AzGoBot {
    /// A fresh bot at the empty `size`×`size` board. Play is deterministic
    /// argmax over visit counts with no root noise — full strength; `seed` only
    /// feeds chance-free tie paths.
    #[wasm_bindgen(constructor)]
    pub fn new(sims: u32, max_leaves: u32, seed: u32, size: usize) -> AzGoBot {
        let game = Go::new(size);
        let state = game.initial_state();
        AzGoBot {
            game,
            enc: GoEncoder::new(size),
            size,
            state,
            search: Search::new(None),
            cfg: PuctConfig {
                sims,
                max_leaves,
                root_noise: 0.0,
                ..PuctConfig::default()
            },
            rng: Rng::new(u64::from(seed)),
            batch: Vec::new(),
            model: None,
            has_tree: false,
            done: false,
        }
    }

    /// Loads the `AZNET1` net (CPU leaf evaluation and the ownership pass
    /// decision both need it, so both modes call this).
    pub fn load_weights(&mut self, weights: &[u8]) -> Result<(), JsError> {
        let net = Net::parse(weights).map_err(|e| JsError::new(&e))?;
        self.model = Some(net);
        Ok(())
    }

    /// Runs the whole search to its visit budget in-wasm, evaluating every
    /// parked leaf with the reference forward, and returns the chosen move.
    /// The GPU `advance`/`best` loop and this share one search and one tree, so
    /// `push` reuse works the same either way.
    pub fn play_cpu(&mut self) -> Result<String, JsError> {
        if !self.batch.is_empty() {
            return Err(JsError::new("play_cpu while evaluations are in flight"));
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

    /// Mirrors an applied move (either side's): advances the internal board
    /// and reuses the searched subtree under that move when there is one.
    /// `coord` is a board label (`"c3"`) or `"pass"`.
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

    /// Absolute (Black-positive) ownership of the current position from the
    /// net's mover-view head, or `None` without an ownership-carrying net.
    fn ownership_abs(&self) -> Option<Vec<f32>> {
        let model = self.model.as_ref()?;
        let planes = self.enc.encode_state(&self.game, &self.state);
        let mover = model.forward_at(&planes, &[], self.size).ownership?;
        let sign = if self.state.to_move() == 0 { 1.0 } else { -1.0 };
        Some(mover.iter().map(|o| o * sign).collect())
    }

    fn decided(&self) -> bool {
        self.ownership_abs()
            .is_some_and(|own| self.game.result_decided(&own, TAU))
    }

    /// `{"value":…,"scoreLead":…}` for the position-quality readout: one net
    /// forward on the current root. `value` is Black's win probability in
    /// `[0,1]` (the net's mover-view value, flipped to Black and mapped from
    /// `[-1,1]`); `scoreLead` is the expected Black−White point lead from the
    /// summed Black-positive ownership minus komi, Black POV. Empty string
    /// without an ownership-carrying net (no usable score signal).
    pub fn eval(&self) -> String {
        let Some(model) = self.model.as_ref() else {
            return String::new();
        };
        let planes = self.enc.encode_state(&self.game, &self.state);
        let out = model.forward_at(&planes, &[], self.size);
        let Some(mover_own) = out.ownership else {
            return String::new();
        };
        let to_black = if self.state.to_move() == 0 { 1.0 } else { -1.0 };
        let black_value = f64::from(out.value) * to_black;
        let win_prob = (black_value + 1.0) / 2.0;
        let owned: f64 = mover_own.iter().map(|o| f64::from(*o) * to_black).sum();
        let score_lead = owned - self.game.komi();
        format!("{{\"value\":{win_prob},\"scoreLead\":{score_lead}}}")
    }

    /// The adjudicated final result (dead stones scored by ownership) as display
    /// text, or `""` when there is no ownership net or the board is not settled
    /// enough to trust — in which case the engine's literal score stands.
    pub fn final_result(&self) -> String {
        let Some(own) = self.ownership_abs() else {
            return String::new();
        };
        if !self.game.result_decided(&own, TAU) {
            return String::new();
        }
        let (b, w) = self.game.adjudicated_area(&own, TAU);
        let komi = self.game.komi();
        let margin = b as f64 - w as f64 - komi;
        let (winner, by) = if margin > 0.0 {
            ("Black", margin)
        } else {
            ("White", -margin)
        };
        format!("Black {b} — White {w} (+{komi} komi). {winner} wins by {by:.1}.")
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

    /// Features of the pending batch, flat `[n × 9·size²]` (board planes).
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

    /// The searched move as a board label (`"c3"` / `"pass"`), argmax over root
    /// visits — except once the ownership head says the result is decided (the
    /// lead exceeds every uncertain point), when the bot passes instead of
    /// filling settled territory, ending the game rather than playing it to the
    /// last eye.
    pub fn best(&self) -> Result<String, JsError> {
        // Concede a won board only once the opponent has also passed; while
        // they keep playing, fall through to the search so the bot defends
        // invasions instead of passing its territory away.
        if self.state.consecutive_passes() >= 1 && self.decided() {
            return Ok(self.game.action_label(&self.state, GoAction::Pass));
        }
        if !self.done {
            return Err(JsError::new("search is not done"));
        }
        // Mirror self-play/eval: never pass while a productive move remains,
        // so the deployed bot doesn't hand a human the game on a sparse board.
        let mut visits = self.search.root_visits().to_vec();
        let actions = self.search.root_actions();
        go::mask_pass_visits(&self.game, &self.state, actions, &mut visits);
        let action = actions[argmax(&visits)];
        Ok(self.game.action_label(&self.state, action))
    }

    /// `{"value":…,"sims":…}` — the root's searched value (side to move) and
    /// total visits, for a thinking readout.
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

#[cfg(test)]
mod tests {
    use super::*;
    use go::encode::PLANES;

    /// A synthetic `AZNET1` go net of the right shape (uniform-fill heads).
    /// `ownership` fills the ownership conv (`None` omits the head); a large
    /// fill saturates it so every point reads as Black-owned ("decided").
    fn synth_net(blocks: usize, c: usize, size: usize, ownership: Option<f32>) -> Vec<u8> {
        let floats = c * PLANES * 9
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
            planes: PLANES,
            size,
            scalars: 0,
            head: nn_infer::HeadKind::GlobalPoolSpatial,
            policy_len: 0,
            flags: nn_infer::HeadFlags(if ownership.is_some() {
                nn_infer::HeadFlags::OWNERSHIP
            } else {
                0
            }),
        };
        let mut b = arch.header_bytes();
        for _ in 0..floats {
            b.extend_from_slice(&0.02f32.to_le_bytes());
        }
        if let Some(w) = ownership {
            for _ in 0..c {
                b.extend_from_slice(&w.to_le_bytes());
            }
        }
        b
    }

    #[test]
    fn ownership_abs_flips_with_the_side_to_move() {
        let mut bot = AzGoBot::new(4, 8, 7, 3);
        bot.load_weights(&synth_net(2, 6, 3, Some(0.1))).unwrap();
        let mover = |bot: &AzGoBot| {
            let planes = bot.enc.encode_state(&bot.game, &bot.state);
            bot.model
                .as_ref()
                .unwrap()
                .forward_at(&planes, &[], bot.size)
                .ownership
                .unwrap()
        };
        assert_eq!(bot.state.to_move(), 0);
        let abs0 = bot.ownership_abs().unwrap();
        assert_eq!(abs0, mover(&bot), "Black to move: absolute == mover view");
        bot.push("b2").unwrap();
        assert_eq!(bot.state.to_move(), 1);
        let abs1 = bot.ownership_abs().unwrap();
        let m1 = mover(&bot);
        assert!(
            abs1.iter().zip(&m1).all(|(a, m)| *a == -*m),
            "White to move: negated"
        );
    }

    #[test]
    fn azwebgo2_net_offers_no_ownership() {
        let mut bot = AzGoBot::new(4, 8, 7, 3);
        bot.load_weights(&synth_net(2, 6, 3, None)).unwrap();
        assert!(bot.ownership_abs().is_none());
        assert_eq!(bot.final_result(), "");
    }

    /// `eval`'s score lead must equal the summed Black-positive ownership minus
    /// komi, and its value must be the Black-POV win probability — both
    /// expressed via the already-trusted `ownership_abs` + the raw mover value,
    /// so a sign or POV regression in `eval` is caught regardless of the net's
    /// particular ownership pattern.
    #[test]
    fn eval_matches_black_pov_ownership_and_value() {
        let mut bot = AzGoBot::new(4, 8, 7, 3);
        bot.load_weights(&synth_net(2, 6, 3, Some(0.1))).unwrap();
        let check = |bot: &AzGoBot| {
            let j: serde_json::Value = serde_json::from_str(&bot.eval()).unwrap();
            let (value, lead) = (
                j["value"].as_f64().unwrap(),
                j["scoreLead"].as_f64().unwrap(),
            );
            let own: f64 = bot
                .ownership_abs()
                .unwrap()
                .iter()
                .map(|o| f64::from(*o))
                .sum();
            assert!((lead - (own - bot.game.komi())).abs() < 1e-4);

            let planes = bot.enc.encode_state(&bot.game, &bot.state);
            let raw = f64::from(
                bot.model
                    .as_ref()
                    .unwrap()
                    .forward_at(&planes, &[], bot.size)
                    .value,
            );
            let to_black = if bot.state.to_move() == 0 { 1.0 } else { -1.0 };
            let expect = (raw * to_black + 1.0) / 2.0;
            assert!(
                (value - expect).abs() < 1e-4,
                "value={value} expect={expect}"
            );
            assert!(
                (0.0..=1.0).contains(&value),
                "win prob in [0,1], got {value}"
            );
        };
        assert_eq!(bot.state.to_move(), 0);
        check(&bot);
        bot.push("b2").unwrap();
        assert_eq!(bot.state.to_move(), 1);
        check(&bot);
    }

    #[test]
    fn eval_is_empty_without_an_ownership_net() {
        let mut bot = AzGoBot::new(4, 8, 7, 3);
        bot.load_weights(&synth_net(2, 6, 3, None)).unwrap();
        assert_eq!(bot.eval(), "");
    }

    #[test]
    fn does_not_pass_an_undecided_board() {
        let mut bot = AzGoBot::new(8, 8, 7, 9);
        bot.load_weights(&synth_net(2, 6, 9, Some(0.1))).unwrap();
        // The opening is not decided (no lead exceeds the open board), so the
        // bot plays rather than passing, and shows no adjudicated result.
        assert!(!bot.decided());
        assert_eq!(bot.final_result(), "");
    }

    #[test]
    fn waits_for_the_opponents_pass_before_conceding_a_decided_board() {
        let mut bot = AzGoBot::new(16, 8, 7, 3);
        // Saturated ownership: every point reads Black-owned, so always decided.
        bot.load_weights(&synth_net(2, 6, 3, Some(20.0))).unwrap();
        assert!(bot.decided());

        // Opponent still playing: the bot plays on rather than conceding.
        assert_eq!(bot.state.consecutive_passes(), 0);
        assert_ne!(bot.play_cpu().unwrap(), "pass");

        // Opponent passes: now the bot passes back to score the won board.
        bot.push("b2").unwrap();
        bot.push("pass").unwrap();
        assert_eq!(bot.state.consecutive_passes(), 1);
        assert!(bot.decided());
        assert_eq!(
            bot.best().unwrap(),
            "pass",
            "end the game once the opponent has also stopped playing"
        );
    }
}
