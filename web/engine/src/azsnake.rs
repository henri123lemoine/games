//! The AlphaZero snake bot's wasm surface. The batched park/resume PUCT search
//! runs in-wasm; leaf evaluation runs either on the GPU (the page answers each
//! parked batch via WebGPU — `set_state` → `advance`/`batch_*` → `advance` … →
//! `best`) or in-wasm against `snakeinfer`'s reference forward (`play_cpu`),
//! the no-GPU fallback. Same net and same search either way, like go and chess.
//!
//! Snake's food placement is a chance node the match engine resolves with its
//! own RNG, so a move-mirroring bot (like `AzGoBot`) would diverge from the
//! engine on every spawn. Instead this bot reconstructs its search root from the
//! engine's authoritative view JSON before each move (`set_state`), which also
//! discards the prior tree — chance nodes make subtree reuse across the engine's
//! RNG unsound, and one search per move is cheap. The reconstruction is exact:
//! the view carries both worms, the food, health, the step count, and seat 0's
//! pending heading.

use game_core::{Game, GameUi, Rng, Turn};
use snake::duel::{Dir, MAX_HEALTH, Worm};
use snake::encode::SnakeEncoder;
use snake::{Duel, DuelState};
use snakeinfer::model::Model;
use snakeinfer::{EvalRequest, EvalResult, Gather, PuctConfig, Search, argmax};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct AzSnakeBot {
    game: Duel,
    enc: SnakeEncoder,
    state: DuelState,
    search: Search<Duel>,
    cfg: PuctConfig,
    rng: Rng,
    /// The reference net; `None` until `load_weights`. The CPU path needs it for
    /// every leaf evaluation; the GPU path never loads it.
    model: Option<Model>,
    /// Requests parked by the last `advance`, awaiting page-side (GPU)
    /// evaluation. Empty between moves and on the CPU path.
    batch: Vec<EvalRequest>,
    /// The last search ran to its visit budget, so `best` is readable.
    done: bool,
}

#[wasm_bindgen]
impl AzSnakeBot {
    /// A fresh bot at the opening position. Play is deterministic argmax over
    /// visit counts with no root noise — full strength; `seed` only feeds
    /// chance-free tie paths and the in-tree chance sampling.
    #[wasm_bindgen(constructor)]
    pub fn new(sims: u32, max_leaves: u32, seed: u32) -> AzSnakeBot {
        let game = Duel::new();
        let state = game.initial_state();
        AzSnakeBot {
            game,
            enc: SnakeEncoder::new(),
            state,
            search: Search::new(None),
            cfg: PuctConfig {
                sims,
                max_leaves,
                root_noise: 0.0,
                ..PuctConfig::default()
            },
            rng: Rng::new(u64::from(seed)),
            model: None,
            batch: Vec::new(),
            done: false,
        }
    }

    /// Loads the `.azweb` net; only the in-wasm CPU leaf evaluation needs it
    /// (the GPU path evaluates page-side).
    pub fn load_weights(&mut self, weights: &[u8]) -> Result<(), JsError> {
        self.model = Some(Model::parse(weights).map_err(|e| JsError::new(&e))?);
        Ok(())
    }

    /// Resets the search root to the position described by the engine's view
    /// JSON (the `snake` frontend contract). Discards the prior tree — snake's
    /// chance nodes make subtree reuse across the engine's RNG unsound, and one
    /// search per move is cheap. Both backends start each move here.
    pub fn set_state(&mut self, view_json: &str) -> Result<(), JsError> {
        if !self.batch.is_empty() {
            return Err(JsError::new("set_state while evaluations are in flight"));
        }
        let state = parse_state(&self.game, view_json).map_err(|e| JsError::new(&e))?;
        if self.game.is_terminal(&state) || !matches!(self.game.turn(&state), Turn::Player(_)) {
            return Err(JsError::new("set_state must leave a player to move"));
        }
        self.state = state;
        self.search = Search::new(None);
        self.done = false;
        Ok(())
    }

    /// Runs the whole search to its visit budget in-wasm, evaluating every
    /// parked leaf with the reference forward, and returns the chosen move as a
    /// heading label (`"up"`/`"right"`/`"down"`/`"left"`). The no-GPU fallback;
    /// requires `load_weights` and a `set_state` leaving a player to move.
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
            results = model.eval(&reqs);
        }
        self.done = true;
        self.best()
    }

    /// Resumes the search with the page's evaluations for the previous batch
    /// (pass empty arrays on the first call after `set_state`), gathers the next
    /// batch, and returns its size — 0 means the search is done and `best` is
    /// ready. `priors` is the flat concatenation over the batch, aligned with
    /// `batch_offsets`; `values` holds one entry per request. The GPU path.
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
                self.batch = reqs;
                Ok(self.batch.len() as u32)
            }
            Gather::Done => {
                self.done = true;
                Ok(0)
            }
        }
    }

    /// Features of the pending batch, flat `[n × 18·area]` (board planes).
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

    /// The searched move as a heading label (`"up"`/`"right"`/`"down"`/`"left"`),
    /// argmax over root visits. Readable once a search has run to its budget.
    pub fn best(&self) -> Result<String, JsError> {
        if !self.done {
            return Err(JsError::new("search is not done"));
        }
        let visits = self.search.root_visits();
        let actions = self.search.root_actions();
        let action = actions[argmax(visits)];
        Ok(self.game.action_label(&self.state, action))
    }

    /// `{"value":…,"sims":…}` — the root's searched value (side to move) and
    /// total visits, for a thinking readout.
    pub fn stats(&self) -> String {
        let sims: u32 = self.search.root_visits().iter().sum();
        let value = if sims > 0 {
            self.search.root_value()
        } else {
            0.0
        };
        format!("{{\"value\":{value},\"sims\":{sims}}}")
    }
}

/// Rebuilds a `DuelState` from the `snake` frontend view JSON. Exact: the view
/// carries both worms (cells head-first, heading, health), the food, the step
/// count, and seat 0's pending heading (present only when seat 1 is to move).
fn parse_state(game: &Duel, view_json: &str) -> Result<DuelState, String> {
    let v: serde_json::Value =
        serde_json::from_str(view_json).map_err(|e| format!("view json: {e}"))?;
    let side = game.side();
    let dir_of = |s: &str| -> Result<Dir, String> {
        Ok(match s {
            "n" | "up" => Dir::Up,
            "e" | "right" => Dir::Right,
            "s" | "down" => Dir::Down,
            "w" | "left" => Dir::Left,
            other => return Err(format!("bad heading '{other}'")),
        })
    };
    let snakes = v["snakes"]
        .as_array()
        .filter(|a| a.len() == 2)
        .ok_or("view needs two snakes")?;
    let mut worms = Vec::with_capacity(2);
    for s in snakes {
        let cells = s["cells"]
            .as_array()
            .ok_or("snake needs cells")?
            .iter()
            .map(|c| {
                let xy = c.as_array().ok_or("cell is [x,y]")?;
                let x = xy[0].as_u64().ok_or("cell x")? as usize;
                let y = xy[1].as_u64().ok_or("cell y")? as usize;
                if x >= side || y >= side {
                    return Err(format!("cell ({x},{y}) off the {side}-grid"));
                }
                Ok((x, y))
            })
            .collect::<Result<Vec<_>, String>>()?;
        if cells.is_empty() {
            return Err("snake has no cells".into());
        }
        let heading = dir_of(s["dir"].as_str().ok_or("snake needs dir")?)?;
        let alive = s["alive"].as_bool().unwrap_or(true);
        let health = s["health"].as_u64().unwrap_or(u64::from(MAX_HEALTH)) as u8;
        worms.push(Worm::from_parts(&cells, heading, alive, health));
    }
    let food = match &v["food"] {
        serde_json::Value::Array(a) if a.len() == 2 => {
            let x = a[0].as_u64().ok_or("food x")? as usize;
            let y = a[1].as_u64().ok_or("food y")? as usize;
            Some((x, y))
        }
        _ => None,
    };
    let pending = match v["pending"].as_str() {
        Some(s) => Some(dir_of(s)?),
        None => None,
    };
    let steps = v["step"].as_u64().unwrap_or(0) as u32;
    let w1 = worms.pop().expect("two worms");
    let w0 = worms.pop().expect("two worms");
    Ok(Duel::state_from_parts([w0, w1], food, pending, steps))
}
