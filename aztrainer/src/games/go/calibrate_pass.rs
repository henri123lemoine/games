//! Does the ownership-based "result decided" call agree with a classical
//! scorer? Plays self-play to its first decided position and compares the
//! adjudicated result with GNU Go's `final_score` (also at the terminal). Not
//! training — a calibration check. `go calibrate-pass [--net ...] [--games 4]
//! [--sims 8]`.

use game_core::{Game, GameUi, PolicyValueEncoder, Rng};
use go::encode::GoEncoder;
use go::{Go, GoAction, GoState, mask_pass_visits};
use nn_infer::{Legacy, Net};
use solvers::azero::{EvalResult, Gather, PuctConfig, Search, argmax};

use super::eval::gnugo_path;
use super::gtp::Gtp;
use super::run::parse_arg as arg;

const TAU: f32 = 0.5;
const SIZE: usize = 9;

fn own_abs(net: &Net, game: &Go, enc: &GoEncoder, s: &GoState) -> Vec<f32> {
    let out = net.forward_at(&enc.encode_state(game, s), &[], SIZE);
    let mover = out.ownership.expect("net has an ownership head");
    let sign = if s.to_move() == 0 { 1.0 } else { -1.0 };
    mover.iter().map(|o| o * sign).collect()
}

fn best_move(
    net: &Net,
    game: &Go,
    enc: &GoEncoder,
    s: &GoState,
    sims: u32,
    noise: f32,
    rng: &mut Rng,
) -> GoAction {
    let cfg = PuctConfig {
        sims,
        max_leaves: 8,
        root_noise: noise,
        ..PuctConfig::default()
    };
    let mut search = Search::new(None);
    let mut results = Vec::new();
    while let Gather::Requests(reqs) = search.advance(
        game,
        enc,
        s,
        &cfg,
        rng,
        std::mem::take(&mut results),
        &|_| false,
    ) {
        results = reqs
            .iter()
            .map(|r| {
                let (priors, value) = net.forward_support(&r.features, &[], &r.support);
                EvalResult { priors, value }
            })
            .collect();
    }
    let mut visits = search.root_visits().to_vec();
    let actions = search.root_actions();
    mask_pass_visits(game, s, actions, &mut visits);
    actions[argmax(&visits)]
}

/// GNU Go reports `B+17.5` / `W+3.5` / `0`; return Black's margin.
fn parse_score(s: &str) -> Option<f64> {
    if s.trim() == "0" {
        return Some(0.0);
    }
    let (color, num) = s.split_once('+')?;
    let v: f64 = num.trim().parse().ok()?;
    match color.trim() {
        "B" => Some(v),
        "W" => Some(-v),
        _ => None,
    }
}

pub fn run(args: &[String]) {
    let net_path: String = arg(
        args,
        "--net",
        "web/app/public/azero/azero-go.azweb".to_string(),
    );
    let games: usize = arg(args, "--games", 4);
    let sims: u32 = arg(args, "--sims", 8);
    let noise: f32 = arg(args, "--noise", 0.35);
    // Accept either a new AZNET1 export or a legacy `.azweb` (the deployed go
    // net, `AZWEBGO2/3`), which the generic engine loads via its legacy adapter.
    let data = std::fs::read(&net_path).expect("read net");
    let net = Net::parse(&data)
        .or_else(|_| {
            Legacy::GoSpatial {
                planes: go::encode::PLANES,
            }
            .load(&data)
        })
        .expect("parse net (AZNET1 or legacy AZWEBGO2/3)");
    let game = Go::new(SIZE);
    let enc = GoEncoder::new(SIZE);

    println!(
        "calibrate-pass: ownership adjudication vs GNU Go final_score ({games} games, {sims} sims)"
    );
    let (mut winner_agree, mut total, mut margin_err) = (0usize, 0usize, 0.0f64);

    for g in 0..games {
        let seed = 1000 + g as u64;
        let mut rng = Rng::new(seed);
        let mut gtp = Gtp::spawn_gnugo(&gnugo_path(), 10, seed as u32, SIZE).expect("spawn gnugo");
        let mut s = game.initial_state();
        let mut ply = 0;
        let mut decided = false;
        while !game.is_terminal(&s) && ply < 220 {
            let a = best_move(&net, &game, &enc, &s, sims, noise, &mut rng);
            let color = if s.to_move() == 0 { "black" } else { "white" };
            let label = game.action_label(&s, a);
            gtp.cmd(&format!("play {color} {}", label.to_uppercase()))
                .expect("gnugo play");
            game.apply(&mut s, a);
            ply += 1;
            if !decided {
                let oa = own_abs(&net, &game, &enc, &s);
                if game.result_decided(&oa, TAU) {
                    decided = true;
                    let (b, w) = game.adjudicated_area(&oa, TAU);
                    let adj = b as f64 - w as f64 - game.komi();
                    let gnugo = gtp.cmd("final_score").ok().and_then(|r| parse_score(&r));
                    print!("game {g}: decided @ply {ply}  adj {adj:+.1} (B{b} W{w})");
                    match gnugo {
                        Some(gs) => {
                            let agree = (adj > 0.0) == (gs > 0.0);
                            winner_agree += agree as usize;
                            total += 1;
                            margin_err += (adj - gs).abs();
                            println!(
                                "  gnugo {gs:+.1}  winner {}",
                                if agree { "AGREE" } else { "DIFFER" }
                            );
                        }
                        None => println!("  gnugo score N/A"),
                    }
                }
            }
        }
        let literal = game.score_margin(&s);
        let gnugo = gtp.cmd("final_score").ok().and_then(|r| parse_score(&r));
        println!(
            "  terminal @ply {ply}  literal {literal:+.1}{}",
            gnugo
                .map(|x| format!("  gnugo {x:+.1}"))
                .unwrap_or_default()
        );
    }
    if total > 0 {
        println!(
            "\nwinner agreement at decided point: {winner_agree}/{total}  ·  mean |adj − gnugo| = {:.1} pts",
            margin_err / total as f64
        );
    }
}
