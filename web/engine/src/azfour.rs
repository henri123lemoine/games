//! Four-player chess AlphaZero wasm surface. The search consumes absolute-seat
//! values mapped from the net's four win-share logits; one instance mirrors one
//! board and can drive any configured seat.

use std::collections::HashMap;

use four_player_chess::encode::{FourPlayerChessEncoder, shares_to_returns};
use four_player_chess::{FourPlayerChess, State};
use game_core::{Game, GameUi, Rng};
use nn_infer::Net;
use solvers::azero::{EvalRequest, EvalResult, Gather, PuctConfig, Search, Value, argmax};
use wasm_bindgen::prelude::*;

fn cpu_eval(net: &Net, requests: &[EvalRequest]) -> Vec<EvalResult> {
    requests
        .iter()
        .map(|request| {
            let (priors, shares) =
                net.forward_support_seats(&request.features, &[], &request.support);
            EvalResult {
                priors,
                value: Value::Seats(shares_to_returns(&shares)),
            }
        })
        .collect()
}

fn unpack(
    batch: &[EvalRequest],
    priors: &[f32],
    shares: &[f32],
) -> Result<Vec<EvalResult>, String> {
    if shares.len() != batch.len() * 4 {
        return Err(format!(
            "expected {} seat values, got {}",
            batch.len() * 4,
            shares.len()
        ));
    }
    let mut offset = 0;
    let mut results = Vec::with_capacity(batch.len());
    for (index, request) in batch.iter().enumerate() {
        let len = request.support.len();
        if offset + len > priors.len() {
            return Err("priors shorter than the batch support".into());
        }
        results.push(EvalResult {
            priors: priors[offset..offset + len].to_vec(),
            value: Value::Seats(shares_to_returns(&shares[index * 4..index * 4 + 4])),
        });
        offset += len;
    }
    if offset != priors.len() {
        return Err("priors longer than the batch support".into());
    }
    Ok(results)
}

#[wasm_bindgen]
pub struct AzFourPlayerBot {
    game: FourPlayerChess,
    state: State,
    search: Search<FourPlayerChess>,
    cfg: PuctConfig,
    rng: Rng,
    history: HashMap<u64, u8>,
    model: Option<Net>,
    batch: Vec<EvalRequest>,
    has_tree: bool,
    done: bool,
}

#[wasm_bindgen]
impl AzFourPlayerBot {
    #[wasm_bindgen(constructor)]
    pub fn new(sims: u32, max_leaves: u32, seed: u32) -> AzFourPlayerBot {
        let game = FourPlayerChess::default();
        let state = game.initial_state();
        let mut history = HashMap::new();
        history.insert(state.repetition_key(), 1);
        AzFourPlayerBot {
            game,
            state,
            search: Search::new(None),
            cfg: PuctConfig {
                sims,
                max_leaves,
                root_noise: 0.0,
                cycle_draws: true,
                ..PuctConfig::default()
            },
            rng: Rng::new(u64::from(seed)),
            history,
            model: None,
            batch: Vec::new(),
            has_tree: false,
            done: false,
        }
    }

    pub fn load_weights(&mut self, weights: &[u8]) -> Result<(), JsError> {
        let net = Net::parse(weights).map_err(|error| JsError::new(&error))?;
        if net.arch().value_seats != 4
            || net.arch().size != 14
            || net.arch().planes != four_player_chess::encode::PLANE_COUNT
        {
            return Err(JsError::new("incompatible four-player chess net"));
        }
        self.model = Some(net);
        Ok(())
    }

    pub fn play_cpu(&mut self) -> Result<String, JsError> {
        if !self.batch.is_empty() {
            return Err(JsError::new("play_cpu while evaluations are in flight"));
        }
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| JsError::new("CPU weights not loaded"))?;
        let mut results = Vec::new();
        while let Gather::Requests(requests) = self.search.advance(
            &self.game,
            &FourPlayerChessEncoder,
            &self.state,
            &self.cfg,
            &mut self.rng,
            std::mem::take(&mut results),
            &|key| self.history.get(&key).copied().unwrap_or(0) > 0,
            None,
        ) {
            results = cpu_eval(model, &requests);
        }
        self.has_tree = true;
        self.done = true;
        self.best()
    }

    pub fn push(&mut self, label: &str) -> Result<(), JsError> {
        if !self.batch.is_empty() {
            return Err(JsError::new("push while evaluations are in flight"));
        }
        let action = self
            .game
            .parse_action(&self.state, label)
            .ok_or_else(|| JsError::new(&format!("'{label}' is not legal here")))?;
        let actions = self.game.legal_actions(&self.state);
        let index = actions
            .iter()
            .position(|&candidate| candidate == action)
            .expect("parsed action is legal");
        let reuse = if self.has_tree {
            let search = std::mem::replace(&mut self.search, Search::new(None));
            search.extract_child(index)
        } else {
            None
        };
        self.has_tree = reuse.is_some();
        self.search = Search::new(reuse);
        self.done = false;
        self.game.apply(&mut self.state, action);
        *self.history.entry(self.state.repetition_key()).or_insert(0) += 1;
        Ok(())
    }

    pub fn advance(&mut self, priors: &[f32], values: &[f32]) -> Result<u32, JsError> {
        let results = unpack(&self.batch, priors, values).map_err(|error| JsError::new(&error))?;
        self.batch.clear();
        match self.search.advance(
            &self.game,
            &FourPlayerChessEncoder,
            &self.state,
            &self.cfg,
            &mut self.rng,
            results,
            &|key| self.history.get(&key).copied().unwrap_or(0) > 0,
            None,
        ) {
            Gather::Requests(requests) => {
                self.has_tree = true;
                self.batch = requests;
                Ok(self.batch.len() as u32)
            }
            Gather::Done => {
                self.has_tree = true;
                self.done = true;
                Ok(0)
            }
        }
    }

    pub fn batch_features(&self) -> Vec<f32> {
        crate::batch_features(&self.batch)
    }

    pub fn batch_support(&self) -> Vec<u16> {
        crate::batch_support(&self.batch)
    }

    pub fn batch_offsets(&self) -> Vec<u32> {
        crate::batch_offsets(&self.batch)
    }

    pub fn best(&self) -> Result<String, JsError> {
        if !self.done {
            return Err(JsError::new("search is not done"));
        }
        let action = self.search.root_actions()[argmax(self.search.root_visits())];
        Ok(self.game.action_label(&self.state, action))
    }

    pub fn final_result(&self) -> String {
        String::new()
    }

    pub fn stats(&self) -> String {
        let sims = if self.has_tree {
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
