//! The export parity gate: load a checkpoint, export it to `AZNET1`, and check
//! that the tch forward agrees with `nn-infer`'s tch-free `AZNET1` forward over
//! random positions. Guards the BN folding and the layer order — the proof that
//! a trained net exports to weights the browser plays identically.

use std::path::Path;

use game_core::{Game, PolicyValueEncoder, Rng};

use crate::net::{EvalRequest, Infer, NetConfig};

/// Per-game hooks the parity walk needs: a fresh game and encoder for the chosen
/// architecture.
pub trait VerifyGame {
    type G: Game;
    type E: PolicyValueEncoder<Self::G>;
    fn game(cfg: &NetConfig) -> Self::G;
    fn encoder(cfg: &NetConfig) -> Self::E;
}

/// Walks `positions` random legal positions and asserts tch ≡ `nn_infer(AZNET1)`
/// for policy, value, and (go) ownership. `cfg` is the checkpoint architecture;
/// `net_path` the checkpoint; the `AZNET1` export is written to `out` and parsed
/// back as the reference.
pub fn verify<V: VerifyGame>(
    net_path: &Path,
    cfg: NetConfig,
    out: &Path,
    positions: usize,
) -> Result<(), String> {
    let body = crate::export::export(net_path, cfg, out)?;
    println!("exported body {body} bytes -> {} (AZNET1)", out.display());

    let infer = Infer::load(net_path, cfg, tch::Device::Cpu, tch::Kind::Float)
        .map_err(|e| format!("load checkpoint: {e}"))?;
    let aznet1_bytes = std::fs::read(out).map_err(|e| format!("read aznet1: {e}"))?;
    let net = nn_infer::Net::parse(&aznet1_bytes).map_err(|e| format!("nn_infer parse: {e}"))?;

    let game = V::game(&cfg);
    let enc = V::encoder(&cfg);
    let mut rng = Rng::new(7);
    let mut state = game.initial_state();
    let (mut max_dp, mut max_dv, mut max_do) = (0.0f32, 0.0f32, 0.0f32);
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

        // nn_infer's PUCT bridge: forward, restrict to the legal support, softmax
        // — exactly what `Infer::forward_batch` does on the tch side.
        let (nn_priors, nn_value) = net.forward_support(&req.features, &[], &support);
        for (a, b) in tch_out.priors.iter().zip(&nn_priors) {
            max_dp = max_dp.max((a - b).abs());
        }
        max_dv = max_dv.max((tch_out.value - nn_value).abs());

        // Go ownership head, when present.
        let out = net.forward_at(&req.features, &[], cfg.size as usize);
        if let Some(nn_own) = &out.ownership {
            let tch_own = infer.ownership(&req.features);
            for (a, b) in tch_own.iter().zip(nn_own) {
                max_do = max_do.max((a - b).abs());
            }
        }

        let i = rng.below(actions.len());
        game.apply(&mut state, actions[i]);
        seen += 1;
    }

    println!(
        "tch vs nn_infer(AZNET1): max |Δprior| {max_dp:.2e}, |Δvalue| {max_dv:.2e}, \
         |Δownership| {max_do:.2e} over {positions} positions"
    );
    if max_dp >= 1e-3 || max_dv >= 1e-3 || max_do >= 1e-3 {
        return Err("AZNET1 export does not match the tch forward".into());
    }
    println!("export verified");
    Ok(())
}
