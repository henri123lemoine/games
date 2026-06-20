//! The export parity gate: load a checkpoint, export it (legacy + `AZNET1`),
//! and check that the tch forward agrees with the tch-free reference forwards
//! over random positions. Guards the BN folding and the layer order.
//!
//! Two references are checked: the legacy per-game `*infer` crate (the historic
//! gate, parsing the legacy bytes) and `nn_infer` parsing the `AZNET1` bytes.
//! Agreement with both proves the dual-write is consistent and that the unified
//! container reproduces the per-game forwards bit-for-bit.

use std::path::Path;

use game_core::{Game, PolicyValueEncoder, Rng};

use crate::net::{EvalRequest, Infer, NetConfig};

/// Per-game hooks the parity walk needs: a fresh game and encoder for the
/// chosen architecture.
pub trait VerifyGame {
    type G: Game;
    type E: PolicyValueEncoder<Self::G>;
    fn game(cfg: &NetConfig) -> Self::G;
    fn encoder(cfg: &NetConfig) -> Self::E;
}

/// The legacy reference parsed from the exported bytes — one of the three
/// `*infer` models, by head kind. All three share `solvers::azero`'s
/// `EvalRequest`/`EvalResult`, so `eval` is uniform; only go carries ownership.
pub enum LegacyModel {
    Chess(azinfer::model::Model),
    Go(goinfer::model::Model),
    Snake(snakeinfer::model::Model),
}

impl LegacyModel {
    fn eval(&self, req: &EvalRequest) -> (Vec<f32>, f32) {
        let one = std::slice::from_ref(req);
        let r = match self {
            LegacyModel::Chess(m) => m.eval(one),
            LegacyModel::Go(m) => m.eval(one),
            LegacyModel::Snake(m) => m.eval(one),
        };
        let r0 = &r[0];
        (r0.priors.clone(), r0.value)
    }

    fn ownership(&self, features: &[f32], size: usize) -> Option<Vec<f32>> {
        match self {
            LegacyModel::Go(m) => m.ownership_at(features, size),
            _ => None,
        }
    }
}

/// Walks `positions` random legal positions and asserts tch ≡ legacy ≡ nn_infer.
/// `cfg` is the checkpoint architecture; `net_path` the checkpoint; the export
/// is written to `legacy_out` / `aznet1_out` first.
pub fn verify<V: VerifyGame>(
    net_path: &Path,
    cfg: NetConfig,
    legacy_out: &Path,
    aznet1_out: &Path,
    positions: usize,
) -> Result<(), String> {
    let body = crate::export::export_dual(net_path, cfg, legacy_out, aznet1_out)?;
    println!(
        "exported body {body} bytes -> {} (legacy) + {} (AZNET1)",
        legacy_out.display(),
        aznet1_out.display()
    );

    let infer = Infer::load(net_path, cfg, tch::Device::Cpu, tch::Kind::Float)
        .map_err(|e| format!("load checkpoint: {e}"))?;
    let legacy_bytes = std::fs::read(legacy_out).map_err(|e| format!("read legacy: {e}"))?;
    let aznet1_bytes = std::fs::read(aznet1_out).map_err(|e| format!("read aznet1: {e}"))?;
    let legacy = parse_legacy(&legacy_bytes, &cfg)?;
    let aznet1 = nn_infer::Net::parse(&aznet1_bytes).map_err(|e| format!("nn_infer parse: {e}"))?;

    let game = V::game(&cfg);
    let enc = V::encoder(&cfg);
    let mut rng = Rng::new(7);
    let mut state = game.initial_state();
    let (mut max_dp, mut max_dv, mut max_do) = (0.0f32, 0.0f32, 0.0f32);
    let (mut max_dp_az, mut max_dv_az, mut max_do_az) = (0.0f32, 0.0f32, 0.0f32);
    let mut seen = 0;
    while seen < positions {
        if game.is_terminal(&state) {
            state = game.initial_state();
            continue;
        }
        if matches!(game.turn(&state), game_core::Turn::Chance) {
            let outs = game.chance_outcomes(&state);
            let j = game_core::rand::sample_outcome(&outs, &mut rng);
            game.apply(&mut state, outs[j].0);
            continue;
        }
        let actions = game.legal_actions(&state);
        let support: Vec<u16> = actions
            .iter()
            .map(|&a| enc.action_index(&game, &state, a) as u16)
            .collect();
        let req = EvalRequest {
            features: enc.encode_state(&game, &state),
            support: support.clone(),
        };

        let tch_out = &infer.forward_batch(std::slice::from_ref(&req))[0];
        let (leg_priors, leg_value) = legacy.eval(&req);
        for (a, b) in tch_out.priors.iter().zip(&leg_priors) {
            max_dp = max_dp.max((a - b).abs());
        }
        max_dv = max_dv.max((tch_out.value - leg_value).abs());

        // nn_infer: full forward then restrict + softmax over the legal support,
        // matching forward_batch's gather.
        let az = aznet1.forward_at(&req.features, &[], cfg.size as usize);
        let mut az_priors: Vec<f32> = support.iter().map(|&s| az.policy[s as usize]).collect();
        nn_infer::softmax(&mut az_priors);
        for (a, b) in tch_out.priors.iter().zip(&az_priors) {
            max_dp_az = max_dp_az.max((a - b).abs());
        }
        max_dv_az = max_dv_az.max((tch_out.value - az.value).abs());

        if let Some(leg_own) = legacy.ownership(&req.features, cfg.size as usize) {
            let tch_own = infer.ownership(&req.features);
            for (a, b) in tch_own.iter().zip(&leg_own) {
                max_do = max_do.max((a - b).abs());
            }
            if let Some(az_own) = &az.ownership {
                for (a, b) in tch_own.iter().zip(az_own) {
                    max_do_az = max_do_az.max((a - b).abs());
                }
            }
        }

        let i = rng.below(actions.len());
        game.apply(&mut state, actions[i]);
        seen += 1;
    }

    println!(
        "vs legacy:   max |Δprior| {max_dp:.2e}, |Δvalue| {max_dv:.2e}, |Δownership| {max_do:.2e}"
    );
    println!(
        "vs nn_infer: max |Δprior| {max_dp_az:.2e}, |Δvalue| {max_dv_az:.2e}, |Δownership| {max_do_az:.2e}"
    );
    if max_dp >= 1e-3 || max_dv >= 1e-3 || max_do >= 1e-3 {
        return Err("export does not match the legacy reference".into());
    }
    if max_dp_az >= 1e-3 || max_dv_az >= 1e-3 || max_do_az >= 1e-3 {
        return Err("AZNET1 export does not match nn_infer".into());
    }
    println!("export verified over {positions} positions");
    Ok(())
}

fn parse_legacy(bytes: &[u8], cfg: &NetConfig) -> Result<LegacyModel, String> {
    use nn_infer::HeadKind;
    match cfg.head {
        HeadKind::FlatConv => azinfer::model::Model::parse(bytes)
            .map(LegacyModel::Chess)
            .map_err(|e| format!("azinfer parse: {e}")),
        HeadKind::GlobalPoolSpatial => goinfer::model::Model::parse(bytes)
            .map(LegacyModel::Go)
            .map_err(|e| format!("goinfer parse: {e}")),
        HeadKind::GlobalPoolDense => snakeinfer::model::Model::parse(bytes)
            .map(LegacyModel::Snake)
            .map_err(|e| format!("snakeinfer parse: {e}")),
    }
}
