//! The AlphaZero snake bot's wasm surface. The batched park/resume PUCT search
//! and the reference forward both run in-wasm: the snake net is tiny (4×64), so
//! the CPU forward is fast enough — no WebGPU path, unlike go.
//!
//! Snake's food placement is a chance node the match engine resolves with its
//! own RNG, so a move-mirroring bot (like `AzGoBot`) would diverge from the
//! engine on every spawn. Instead this bot reconstructs its search root from the
//! engine's authoritative view JSON before each move (`set_state`), then runs
//! the whole search to its visit budget against `snakeinfer`'s forward
//! (`play_cpu`). The reconstruction is exact: the view carries both worms, the
//! food, health, the step count, and seat 0's pending heading.

use game_core::{Game, GameUi, Rng, Turn};
use snake::duel::{Dir, MAX_HEALTH, Worm};
use snake::encode::SnakeEncoder;
use snake::{Duel, DuelState};
use snakeinfer::model::Model;
use snakeinfer::{Gather, PuctConfig, Search, argmax};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct AzSnakeBot {
    game: Duel,
    enc: SnakeEncoder,
    state: DuelState,
    search: Search<Duel>,
    cfg: PuctConfig,
    rng: Rng,
    /// The reference net; `None` until `load_weights`. CPU play needs it for
    /// every leaf evaluation.
    model: Option<Model>,
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
        }
    }

    /// Loads the `.azweb` net; CPU leaf evaluation needs it.
    pub fn load_weights(&mut self, weights: &[u8]) -> Result<(), JsError> {
        self.model = Some(Model::parse(weights).map_err(|e| JsError::new(&e))?);
        Ok(())
    }

    /// Resets the search root to the position described by the engine's view
    /// JSON (the `snake` frontend contract). Discards the prior tree — snake's
    /// chance nodes make subtree reuse across the engine's RNG unsound, and one
    /// search per move is cheap on a 4×64 net.
    pub fn set_state(&mut self, view_json: &str) -> Result<(), JsError> {
        let state = parse_state(&self.game, view_json).map_err(|e| JsError::new(&e))?;
        self.state = state;
        self.search = Search::new(None);
        Ok(())
    }

    /// Runs the whole search to its visit budget in-wasm, evaluating every
    /// parked leaf with the reference forward, and returns the chosen move as a
    /// heading label (`"up"`/`"right"`/`"down"`/`"left"`). Requires the search
    /// root to be a position where it is a player's turn (the engine only drives
    /// the bot then).
    pub fn play_cpu(&mut self) -> Result<String, JsError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| JsError::new("CPU weights not loaded"))?;
        if self.game.is_terminal(&self.state)
            || !matches!(self.game.turn(&self.state), Turn::Player(_))
        {
            return Err(JsError::new("set_state must leave a player to move"));
        }
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
