//! The browser engine: wasm-bindgen bindings over the lab's registry and
//! type-erased matches (see WEB.md). Designed to run inside a Web Worker;
//! every value crossing the boundary is a JSON string, so the JS side stays
//! game-schema-free.

use lab::registry::{Opts, entries};
use lab::runner::{AnyMatch, MatchEvent};
use serde_json::{Value, json};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

fn opts_from_json(opts_json: &str) -> Result<Opts, String> {
    let mut map = std::collections::HashMap::new();
    if !opts_json.trim().is_empty() {
        let v: Value =
            serde_json::from_str(opts_json).map_err(|e| format!("bad opts JSON: {e}"))?;
        let obj = v.as_object().ok_or("opts must be a JSON object")?;
        for (k, val) in obj {
            let s = match val {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            map.insert(k.clone(), s);
        }
    }
    Ok(Opts::new(map))
}

fn event_json(e: &MatchEvent) -> Value {
    json!({
        "seat": e.seat,
        "label": e.label,
        "text": e.text,
        "detail": e.detail,
        "data": e
            .data
            .as_deref()
            .and_then(|d| serde_json::from_str::<Value>(d).ok()),
    })
}

/// The catalog: playable games (with their option help) and the bot specs
/// available for bot-vs-bot comparison per game.
#[wasm_bindgen]
pub fn list_games() -> String {
    let all = entries();
    let games: Vec<Value> = all
        .iter()
        .map(|e| json!({"id": e.id, "summary": e.summary, "opts": e.opts_help}))
        .collect();
    let compare: Vec<Value> = all
        .iter()
        .filter_map(|e| {
            let ev = e.eval.as_ref()?;
            Some(json!({"id": e.id, "bots": ev.bots_help, "field": ev.has_field}))
        })
        .collect();
    json!({"games": games, "compare": compare}).to_string()
}

/// Registers artifact bytes (a trained net, a solver table) under the
/// path-like id the registry will ask for (e.g. `data/azero/chess.bin`).
#[wasm_bindgen]
pub fn load_artifact(id: &str, bytes: &[u8]) {
    lab::artifacts::put(id, bytes.to_vec());
}

#[wasm_bindgen]
pub struct WebMatch {
    inner: Box<dyn AnyMatch>,
}

/// Builds a match from a game id and a JSON object of `key=value` options
/// (the same options as the terminal client, e.g. `{"players":5,"seat":
/// "watch","seed":12345}`). Hosts should always pass `seed`.
#[wasm_bindgen]
pub fn create_match(game: &str, opts_json: &str) -> Result<WebMatch, JsError> {
    let opts = opts_from_json(opts_json).map_err(|e| JsError::new(&e))?;
    let entry = entries()
        .into_iter()
        .find(|e| e.id == game)
        .ok_or_else(|| JsError::new(&format!("unknown game '{game}'")))?;
    let inner = (entry.make)(&opts).map_err(|e| JsError::new(&e))?;
    let unused = opts.unused();
    if !unused.is_empty() {
        return Err(JsError::new(&format!(
            "unused option(s): {} — opts for {}: {}",
            unused.join(", "),
            entry.id,
            entry.opts_help
        )));
    }
    Ok(WebMatch { inner })
}

#[wasm_bindgen]
impl WebMatch {
    /// One bot move (chance folded in) as event JSON, or `""` once it is the
    /// human's turn or the game is over. Clients call this in a loop and
    /// animate each event.
    pub fn step(&mut self) -> String {
        self.inner
            .step()
            .map(|e| event_json(&e).to_string())
            .unwrap_or_default()
    }

    pub fn is_over(&self) -> bool {
        self.inner.is_over()
    }

    /// The human's view as terminal text (the generic frontend's fallback).
    pub fn view(&self) -> String {
        self.inner.view()
    }

    /// The human's view as game-private JSON, when the game provides one.
    pub fn view_data(&self) -> Option<String> {
        self.inner.view_data()
    }

    /// JSON array of legal action labels, menu-ordered; indices are the wire
    /// format for moves.
    pub fn legal_labels(&self) -> String {
        serde_json::to_string(&self.inner.legal_labels()).unwrap_or_else(|_| "[]".into())
    }

    /// Apply human input (menu index or game-native text like `e2e4`);
    /// returns the applied move's event JSON, or throws to re-prompt.
    pub fn apply_human(&mut self, input: &str) -> Result<String, JsError> {
        self.inner
            .apply_human(input)
            .map(|e| event_json(&e).to_string())
            .map_err(|e| JsError::new(&e))
    }

    pub fn result_text(&self) -> String {
        self.inner.result_text()
    }

    /// Seat to act, or -1 (chance/terminal).
    pub fn to_act(&self) -> i32 {
        self.inner.to_act().map(|s| s as i32).unwrap_or(-1)
    }

    pub fn num_seats(&self) -> usize {
        self.inner.num_seats()
    }

    /// The human's seat, or -1 when spectating.
    pub fn human_seat(&self) -> i32 {
        self.inner.human_seat().map(|s| s as i32).unwrap_or(-1)
    }
}

/// Plays seat-swapped pairs `lo..hi` of `a` vs `b` and returns
/// `{"w":…,"d":…,"l":…}` from A's perspective. Workers call this in slices;
/// the same seed and slice always reproduce the same games.
#[wasm_bindgen]
pub fn play_pairs(
    game: &str,
    opts_json: &str,
    a: &str,
    b: &str,
    seed: u32,
    lo: u32,
    hi: u32,
) -> Result<String, JsError> {
    let opts = opts_from_json(opts_json).map_err(|e| JsError::new(&e))?;
    let eval = entries()
        .into_iter()
        .find(|e| e.id == game)
        .and_then(|e| e.eval)
        .ok_or_else(|| JsError::new(&format!("no compare support for '{game}'")))?;
    let (w, d, l) = (eval.pairs)(&opts, a, b, seed as u64, lo as u64..hi as u64)
        .map_err(|e| JsError::new(&e))?;
    Ok(json!({"w": w, "d": d, "l": l}).to_string())
}

/// N-player field games `lo..hi`: hero `a` rotated through seats against a
/// field of `b`; returns `{"wins":…,"losses":…}`.
#[wasm_bindgen]
pub fn play_field(
    game: &str,
    opts_json: &str,
    a: &str,
    b: &str,
    seed: u32,
    lo: u32,
    hi: u32,
) -> Result<String, JsError> {
    let opts = opts_from_json(opts_json).map_err(|e| JsError::new(&e))?;
    let eval = entries()
        .into_iter()
        .find(|e| e.id == game)
        .and_then(|e| e.eval)
        .ok_or_else(|| JsError::new(&format!("no compare support for '{game}'")))?;
    if !eval.has_field {
        return Err(JsError::new(&format!(
            "'{game}' has no field (N-player) mode"
        )));
    }
    let (wins, losses) = (eval.field)(&opts, a, b, seed as u64, lo as u64..hi as u64)
        .map_err(|e| JsError::new(&e))?;
    Ok(json!({"wins": wins, "losses": losses}).to_string())
}

/// Elo point estimate and 95% margin for a W-D-L record.
#[wasm_bindgen]
pub fn elo(w: u32, d: u32, l: u32) -> String {
    let e = game_core::stats::elo_estimate(w as u64, d as u64, l as u64);
    json!({"elo": e.elo, "margin": e.margin()}).to_string()
}

/// Bradley-Terry Elo table from a round-robin record matrix
/// (`records[i][j] = [w, d, l]` of i against j), mean-anchored at 0.
#[wasm_bindgen]
pub fn fit_elo_table(records_json: &str) -> Result<String, JsError> {
    let records: Vec<Vec<(u64, u64, u64)>> = serde_json::from_str(records_json)
        .map_err(|e| JsError::new(&format!("bad records: {e}")))?;
    serde_json::to_string(&game_core::stats::fit_elo(&records))
        .map_err(|e| JsError::new(&e.to_string()))
}
