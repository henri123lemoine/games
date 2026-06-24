//! A frozen, self-generated position test set and a single-forward scorer for
//! it — a cheap whole-board strength proxy across training snapshots, no human
//! data. `gen` plays batched self-play games once and labels each sampled
//! position with its game's eventual outcome (value) and a search's best move
//! (policy). `score` runs one batched forward per checkpoint and reports value
//! MSE / sign-accuracy and policy top-1 against those frozen labels.
//!
//! Labels target the net's value/whole-board judgment rather than tactics: the
//! value label is the realized game `z` (mover-perspective), so a net's raw
//! value head is graded against how games it played actually ended, and the
//! policy label is the move a (deeper-`--sims`) search preferred at that
//! position.

use std::path::{Path, PathBuf};

use game_core::{Game, PolicyValueEncoder, Rng};
use go::encode::GoEncoder;
use go::{Go, GoState};
use rayon::prelude::*;
use solvers::azero::{self, Gather, PuctConfig, argmax};
use tch::Kind;

use super::run::{device_for, net_config_for, parse_arg as arg};
use super::sample::{compact, expand};
use super::selfplay::mix;
use crate::net::{EvalRequest, EvalResult, Infer};

/// One frozen test position: enough to re-encode the planes (packed board +
/// mover + komi + size) plus the value and best-move labels.
struct Position {
    planes: Box<[u64]>,
    stm_white: bool,
    komi: f32,
    size: u8,
    /// Game outcome from the mover's perspective at this position (+1 win).
    value: f32,
    /// Best move's policy index from the search at this position.
    best_move: u16,
}

/// One recorded ply before its outcome is known: packed planes, mover-is-white,
/// komi, board size, the searched best move, and the mover seat.
type PlyRecord = (Box<[u64]>, bool, f32, u8, u16, usize);

/// A self-play game that records one labeled position per ply and backfills the
/// mover-perspective outcome when it ends. Mirrors `eval::net_vs_net`'s batched
/// single-net loop, with komi randomized per game like self-play.
struct GenGame {
    go: Go,
    state: GoState,
    search: azero::Search<Go>,
    rng: Rng,
    records: Vec<PlyRecord>,
    plies: u16,
    done: bool,
}

impl GenGame {
    fn new(size: usize, komi: f64, seed: u64) -> GenGame {
        let go = Go::with_komi(size, komi);
        GenGame {
            state: go.initial_state(),
            go,
            search: azero::Search::new(None),
            rng: Rng::new(seed),
            records: Vec::new(),
            plies: 0,
            done: false,
        }
    }
}

/// The draw-guard ply cap, matching self-play's classification.
fn max_plies(size: usize) -> u16 {
    (4 * size * size) as u16
}

/// Generates a frozen test set: plays batched games with the net at `--sims`,
/// samples positions across them, and labels each with its game's outcome and
/// the searched best move. Writes a self-describing JSON file.
pub fn generate(args: &[String]) {
    let net_path: PathBuf = arg(
        args,
        "--net",
        PathBuf::from("../../data/azgo/run_moyo/latest_swa.ot"),
    );
    let out: PathBuf = arg(args, "--out", PathBuf::from("evalset.json"));
    let target: usize = arg(args, "--n", 1024);
    let sims: u32 = arg(args, "--sims", 400);
    let concurrent: usize = arg(args, "--concurrent", 128);
    // Keep ~one position per game to decorrelate the set; a game yields many
    // plies but neighboring plies share most of the board and the outcome.
    let per_game: usize = arg(args, "--per-game", 2);
    let temp_plies: u16 = arg(args, "--temp-plies", 12);
    let komi_range: i64 = arg(args, "--komi-range", 0);
    let seed: u64 = arg(args, "--seed", 0x0E5A_15E7);

    let dev = device_for();
    let cfg = net_config_for(args, &net_path);
    let size = cfg.size as usize;
    let infer = match Infer::load(&net_path, cfg, dev, Kind::Half) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("failed to load {}: {e}", net_path.display());
            std::process::exit(1);
        }
    };

    let enc = GoEncoder::new(size);
    let puct = PuctConfig {
        sims,
        root_noise: 0.0,
        ..PuctConfig::default()
    };

    let draw_komi = |rng: &mut Rng| -> f64 {
        if komi_range <= 0 {
            go::KOMI
        } else {
            let off = rng.below((2 * komi_range + 1) as usize) as i64 - komi_range;
            go::KOMI + off as f64
        }
    };

    let t = std::time::Instant::now();
    let mut positions: Vec<Position> = Vec::with_capacity(target);
    let mut games: Vec<GenGame> = (0..concurrent)
        .map(|i| {
            let mut rng = Rng::new(mix(seed, i as u64));
            let komi = draw_komi(&mut rng);
            GenGame::new(size, komi, mix(seed, (i as u64) << 16 | 1))
        })
        .collect();
    let mut next_seed = seed.wrapping_add(concurrent as u64);

    let mut results: Vec<Vec<EvalResult>> = (0..games.len()).map(|_| Vec::new()).collect();
    while positions.len() < target {
        let gathered: Vec<Vec<EvalRequest>> = games
            .par_iter_mut()
            .zip(results.par_iter_mut())
            .map(|(g, r)| {
                let mut pending = std::mem::take(r);
                loop {
                    if g.done {
                        return Vec::new();
                    }
                    let go = g.go;
                    match g.search.advance(
                        &go,
                        &enc,
                        &g.state,
                        &puct,
                        &mut g.rng,
                        std::mem::take(&mut pending),
                        &|_| false,
                        None,
                    ) {
                        Gather::Requests(reqs) => return reqs,
                        Gather::Done => {
                            let mut visits = g.search.root_visits().to_vec();
                            let actions = g.search.root_actions().to_vec();
                            go::mask_pass_visits(&go, &g.state, &actions, &mut visits);
                            let best = argmax(&visits);
                            let stm = g.state.to_move();
                            let (planes, stm_white) =
                                compact(&enc.encode_state(&go, &g.state), go.size());
                            let best_move = enc.action_index(&go, &g.state, actions[best]) as u16;
                            g.records.push((
                                planes,
                                stm_white,
                                go.komi() as f32,
                                go.size() as u8,
                                best_move,
                                stm,
                            ));
                            // Opening exploration only diversifies which
                            // positions we collect; later it plays the searched
                            // move so games stay realistic.
                            let choice = if g.plies < temp_plies {
                                game_core::rand::sample_visits(&visits, &mut g.rng)
                            } else {
                                best
                            };
                            go.apply(&mut g.state, actions[choice]);
                            g.plies += 1;
                            g.search = azero::Search::new(None);
                            if go.is_terminal(&g.state) || g.plies >= max_plies(go.size()) {
                                g.done = true;
                                return Vec::new();
                            }
                        }
                    }
                }
            })
            .collect();

        let mut flat: Vec<EvalRequest> = Vec::new();
        let mut spans: Vec<(usize, usize)> = Vec::with_capacity(gathered.len());
        for reqs in gathered {
            spans.push((flat.len(), reqs.len()));
            flat.extend(reqs);
        }
        if flat.is_empty() {
            // Every game finished this cycle: harvest their positions, then
            // refill the pool with fresh games and keep going.
            for g in &mut games {
                let z_black = g.go.returns(&g.state, 0) as f32;
                let recs = std::mem::take(&mut g.records);
                let n = recs.len();
                if n == 0 {
                    continue;
                }
                let step = (n / per_game.max(1)).max(1);
                for (j, (planes, stm_white, komi, sz, best_move, mover)) in
                    recs.into_iter().enumerate()
                {
                    if j % step != 0 || positions.len() >= target {
                        continue;
                    }
                    let z = if mover == 0 { z_black } else { -z_black };
                    positions.push(Position {
                        planes,
                        stm_white,
                        komi,
                        size: sz,
                        value: z,
                        best_move,
                    });
                }
            }
            if positions.len() >= target {
                break;
            }
            for (i, g) in games.iter_mut().enumerate() {
                let s = next_seed.wrapping_add(i as u64);
                let mut rng = Rng::new(mix(s, 7));
                let komi = draw_komi(&mut rng);
                *g = GenGame::new(size, komi, mix(s, 11));
            }
            next_seed = next_seed.wrapping_add(games.len() as u64);
            results = (0..games.len()).map(|_| Vec::new()).collect();
            continue;
        }
        let mut outs = infer.forward_batch(&flat);
        for (i, (start, len)) in spans.into_iter().enumerate().rev() {
            results[i] = outs.split_off(start);
            debug_assert_eq!(results[i].len(), len);
        }
    }
    positions.truncate(target);

    let json = serde_json::json!({
        "kind": "go-evalset/1",
        "size": size,
        "sims": sims,
        "net": net_path.display().to_string(),
        "n": positions.len(),
        "positions": positions
            .iter()
            .map(|p| serde_json::json!({
                "planes": p.planes.iter().collect::<Vec<_>>(),
                "stm_white": p.stm_white,
                "komi": p.komi,
                "size": p.size,
                "value": p.value,
                "best_move": p.best_move,
            }))
            .collect::<Vec<_>>(),
    });
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&out, json.to_string()).expect("write evalset");
    println!(
        "wrote {} positions ({size}x{size}, {sims} sims) to {} [{:.0}s]",
        positions.len(),
        out.display(),
        t.elapsed().as_secs_f32()
    );
}

/// Scores a checkpoint against a frozen set in one batched forward (no search,
/// no games): value MSE + value sign-accuracy vs the outcome labels, and policy
/// top-1 match vs the searched best-move labels. Prints JSON metrics.
pub fn score(args: &[String]) {
    let net_path: PathBuf = arg(
        args,
        "--net",
        PathBuf::from("../../data/azgo/run_moyo/latest_swa.ot"),
    );
    let set: PathBuf = arg(args, "--set", PathBuf::from("evalset.json"));

    let text = std::fs::read_to_string(&set).expect("read evalset");
    let doc: serde_json::Value = serde_json::from_str(&text).expect("parse evalset");
    let size = doc["size"].as_u64().expect("evalset size") as usize;
    let arr = doc["positions"].as_array().expect("evalset positions");

    let cfg = net_config_for(args, &net_path);
    assert_eq!(
        cfg.size as usize, size,
        "checkpoint size {} != evalset size {size}",
        cfg.size
    );
    let dev = device_for();
    let infer = match Infer::load(&net_path, cfg, dev, Kind::Half) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("failed to load {}: {e}", net_path.display());
            std::process::exit(1);
        }
    };

    let cells = size * size;
    let policy = (cells + 1) as u16;
    // Full-policy support per position: requesting priors over every index lets
    // the batched forward return the value and the full policy, so top-1 is
    // argmax over the returned priors mapped back to policy indices.
    let support: Vec<u16> = (0..policy).collect();

    let t = std::time::Instant::now();
    let plane_len = go::encode::PLANES * cells;
    let mut buf = vec![0.0f32; plane_len];
    let reqs: Vec<EvalRequest> = arr
        .iter()
        .map(|p| {
            let planes: Vec<u64> = p["planes"]
                .as_array()
                .expect("planes")
                .iter()
                .map(|w| w.as_u64().expect("plane word"))
                .collect();
            let stm_white = p["stm_white"].as_bool().expect("stm_white");
            let komi = p["komi"].as_f64().expect("komi") as f32;
            expand(&planes, stm_white, komi, 0, size, &mut buf);
            EvalRequest {
                features: buf.clone(),
                support: support.clone(),
            }
        })
        .collect();

    let outs = infer.forward_batch(&reqs);

    let n = arr.len();
    let mut sq_err = 0.0f64;
    let mut sign_ok = 0usize;
    let mut top1_ok = 0usize;
    for (p, out) in arr.iter().zip(&outs) {
        let value_label = p["value"].as_f64().expect("value") as f32;
        let best_label = p["best_move"].as_u64().expect("best_move") as usize;
        let d = (out.value - value_label) as f64;
        sq_err += d * d;
        // A label is ±1 (won/lost game), so sign-accuracy is the value head's
        // win-prediction rate; the rare exact-zero is counted correct either way.
        if (out.value >= 0.0) == (value_label >= 0.0) {
            sign_ok += 1;
        }
        // Priors are aligned with `support` = 0..policy, so argmax over them is
        // the predicted policy index directly.
        let pred = argmax_f32(&out.priors);
        if pred == best_label {
            top1_ok += 1;
        }
    }

    let metrics = serde_json::json!({
        "net": net_path.display().to_string(),
        "iter": crate::rundir::last_iter(
            &net_path.parent().unwrap_or(Path::new(".")).join("metrics.jsonl")
        ),
        "set": set.display().to_string(),
        "n": n,
        "value_mse": sq_err / n.max(1) as f64,
        "value_sign_acc": sign_ok as f64 / n.max(1) as f64,
        "policy_top1": top1_ok as f64 / n.max(1) as f64,
        "secs": t.elapsed().as_secs_f32(),
    });
    println!("{}", metrics);
}

/// Index of the max element (first on ties).
fn argmax_f32(xs: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in xs.iter().enumerate() {
        if x > xs[best] {
            best = i;
        }
    }
    best
}

/// `go evalset <gen|score>` dispatch.
pub fn main(args: &[String]) {
    match args.first().map(String::as_str) {
        Some("gen") => generate(&args[1..]),
        Some("score") => score(&args[1..]),
        other => {
            eprintln!("usage: go evalset <gen|score> [flags]\ngot: {other:?}");
            std::process::exit(2);
        }
    }
}
