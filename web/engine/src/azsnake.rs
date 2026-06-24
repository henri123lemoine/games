//! The AlphaZero snake bot's wasm surface. The batched park/resume PUCT search
//! runs in-wasm; leaf evaluation runs either on the GPU (the page answers each
//! parked batch via WebGPU — `set_state` → `advance`/`batch_*` → `advance` … →
//! `best`) or in-wasm against `nn-infer`'s reference forward (`play_cpu`),
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

use game_core::{Game, GameUi, PolicyValueEncoder, Rng, Turn};
use nn_infer::Net;
use snake::duel::{Dir, DuelAction, MAX_HEALTH, Worm};
use snake::encode::SnakeEncoder;
use snake::{Duel, DuelState};
use solvers::azero::{EvalRequest, EvalResult, Gather, PuctConfig, Search, argmax};
use wasm_bindgen::prelude::*;

use crate::eval_batch;

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
    model: Option<Net>,
    /// Requests parked by the last `advance`, awaiting page-side (GPU)
    /// evaluation. Empty between moves and on the CPU path.
    batch: Vec<EvalRequest>,
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
        }
    }

    /// Loads the `AZNET1` net; only the in-wasm CPU leaf evaluation needs it
    /// (the GPU path evaluates page-side).
    pub fn load_weights(&mut self, weights: &[u8]) -> Result<(), JsError> {
        let net = Net::parse(weights).map_err(|e| JsError::new(&e))?;
        self.model = Some(net);
        Ok(())
    }

    /// Resets the search root to the position described by the engine's view
    /// JSON (the `snake` frontend contract). Discards the prior tree — snake's
    /// chance nodes make subtree reuse across the engine's RNG unsound, and one
    /// search per move is cheap. Both backends start each move here.
    ///
    /// Any batch parked by a *time-budgeted* search that was stopped mid-flight
    /// (the anytime GPU path returns best-so-far at a deadline rather than
    /// running every batch) is simply dropped — the whole tree is being
    /// discarded anyway, so a leftover parked batch is stale and harmless.
    pub fn set_state(&mut self, view_json: &str) -> Result<(), JsError> {
        self.batch.clear();
        let state = parse_state(&self.game, view_json).map_err(|e| JsError::new(&e))?;
        if self.game.is_terminal(&state) || !matches!(self.game.turn(&state), Turn::Player(_)) {
            return Err(JsError::new("set_state must leave a player to move"));
        }
        self.state = state;
        self.search = Search::new(None);
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
            None,
        ) {
            results = eval_batch(model, &reqs);
        }
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
            None,
        ) {
            Gather::Requests(reqs) => {
                self.batch = reqs;
                Ok(self.batch.len() as u32)
            }
            Gather::Done => Ok(0),
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
    /// argmax over root visits. Readable as soon as the root has any visits, so
    /// a time-budgeted (anytime) search can stop at its deadline and read the
    /// best move SO FAR — the argmax-over-visits move is well-defined from the
    /// first simulation and only sharpens with more, so an early read is the
    /// same kind of answer, just from a shallower search.
    pub fn best(&self) -> Result<String, JsError> {
        if !self.search.has_root() {
            return Err(JsError::new("search has not expanded the root yet"));
        }
        let visits = self.search.root_visits();
        if visits.iter().all(|&v| v == 0) {
            return Err(JsError::new("search has no visits yet"));
        }
        let actions = self.search.root_actions();
        let action = actions[argmax(visits)];
        Ok(self.game.action_label(&self.state, action))
    }

    /// A competent move from a SINGLE policy-head forward — the always-available
    /// fast floor for real-time play, ~1-2ms on the CPU/wasm forward (NO WebGPU
    /// needed). One net eval gives the policy over the four headings; we pick the
    /// highest-policy heading that does NOT immediately die (a cheap 1-ply
    /// safety: simulate the tick and require the acting seat to survive). If
    /// every heading dies, take the top-policy one. Requires `load_weights` and a
    /// `set_state`-shaped view leaving a player to move.
    ///
    /// This is what the real-time driver applies every tick, so the bot always
    /// makes a real, non-suicidal move instead of coasting straight; the heavy
    /// background search only refines on top when it has time.
    pub fn policy_move(&mut self, view_json: &str) -> Result<String, JsError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| JsError::new("policy weights not loaded"))?;
        let state = parse_state(&self.game, view_json).map_err(|e| JsError::new(&e))?;
        if self.game.is_terminal(&state) {
            return Err(JsError::new("policy_move on a terminal state"));
        }
        let Turn::Player(seat) = self.game.turn(&state) else {
            return Err(JsError::new("policy_move needs a player to move"));
        };

        // One forward: policy over the four headings (support is all of Dir::ALL,
        // whose action indices are 0..4 — see SnakeEncoder::action_index).
        let features = self.enc.encode_state(&self.game, &state);
        let support: Vec<u16> = (0..Dir::ALL.len() as u16).collect();
        let (priors, _value) = model.forward_support(&features, &[], &support);

        // Rank headings by policy, highest first.
        let mut order: Vec<usize> = (0..Dir::ALL.len()).collect();
        order.sort_by(|&a, &b| {
            priors[b]
                .partial_cmp(&priors[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Pick the best heading whose immediate resolution keeps the acting seat
        // alive; fall back to the top-policy heading if all are fatal.
        let chosen = order
            .iter()
            .copied()
            .find(|&i| self.survives(&state, seat, Dir::ALL[i]))
            .unwrap_or(order[0]);
        Ok(self
            .game
            .action_label(&state, DuelAction::Move(Dir::ALL[chosen])))
    }

    /// Whether `seat` is still alive one tick after committing `dir`. Seat 1's
    /// commit resolves the tick, so we can read survival directly. Seat 0's
    /// commit only records a pending heading (no resolution yet), so we
    /// conservatively check that its next head cell is in-bounds and not on a
    /// blocking body segment — a cheap geometric guard.
    fn survives(&self, state: &DuelState, seat: usize, dir: Dir) -> bool {
        if seat == 1 {
            let mut s = state.clone();
            self.game.apply(&mut s, DuelAction::Move(dir));
            return s.worm(1).alive();
        }
        // Seat 0: geometric next-head safety against both snakes' bodies.
        let side = self.game.side() as i32;
        let worm = state.worm(seat);
        let (hx, hy) = worm.head();
        let (dx, dy) = match dir {
            Dir::Up => (0, -1),
            Dir::Right => (1, 0),
            Dir::Down => (0, 1),
            Dir::Left => (-1, 0),
        };
        let (nx, ny) = (hx as i32 + dx, hy as i32 + dy);
        if nx < 0 || ny < 0 || nx >= side || ny >= side {
            return false;
        }
        let (nx, ny) = (nx as usize, ny as usize);
        // The acting snake's own tail vacates this tick (unless it eats), so the
        // last cell is not a blocker; every other body cell of either snake is.
        let eats = state.food() == Some((nx, ny));
        for s in 0..2 {
            let w = state.worm(s);
            let last = w.len().saturating_sub(1);
            for (i, c) in w.cells().enumerate() {
                let vacates = s == seat && i == last && !eats;
                if !vacates && c == (nx, ny) {
                    return false;
                }
            }
        }
        true
    }

    /// `{"value":…,"sims":…}` — the root's searched value (side to move) and
    /// total visits, for a thinking readout.
    pub fn stats(&self) -> String {
        if !self.search.has_root() {
            return "{\"value\":0,\"sims\":0}".to_string();
        }
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
