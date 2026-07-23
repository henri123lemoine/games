//! Canonical simultaneous Battlesnake neural training and evaluation.

mod eval;
mod gauge;
mod run;
mod sample;
mod sim_selfplay;

use std::path::PathBuf;

use game_core::{Rng, SimultaneousGame, SimultaneousPolicyValueEncoder, SimultaneousTurn};
use snake::battlesnake::{Battlesnake, Direction, Rules};
use snake::battlesnake_encode::BattlesnakeEncoder;

pub use sample::Sample;

fn export(args: &[String]) {
    let net: PathBuf = run::parse_arg(
        args,
        "--net",
        PathBuf::from("runs/battlesnake/logit-p2/latest.ot"),
    );
    let out: PathBuf = run::parse_arg(args, "--out", net.with_extension("azweb"));
    let cfg = run::net_config_for(args, &net);
    let len = crate::export::export(&net, cfg, &out).expect("export");
    println!(
        "exported {}x{} canonical Battlesnake net: {} body bytes -> {}",
        cfg.blocks,
        cfg.channels,
        len,
        out.display(),
    );
}

fn verify_export(args: &[String]) {
    let net_path: PathBuf = run::parse_arg(
        args,
        "--net",
        PathBuf::from("runs/battlesnake/logit-p2/latest.ot"),
    );
    let out: PathBuf = run::parse_arg(args, "--out", net_path.with_extension("azweb"));
    let positions: usize = run::parse_arg(args, "--positions", 120);
    let cfg = run::net_config_for(args, &net_path);
    crate::export::export(&net_path, cfg, &out).expect("export");
    let infer = crate::net::Infer::load(&net_path, cfg, tch::Device::Cpu, tch::Kind::Float)
        .expect("load checkpoint");
    let bytes = std::fs::read(&out).expect("read export");
    let exported = nn_infer::Net::parse(&bytes).expect("parse export");
    let encoder = BattlesnakeEncoder;
    let mut rng = Rng::new(7);
    let mut game = Battlesnake::<2>::new(Rules {
        seed: 7,
        ..Rules::default()
    });
    let mut state = game.initial_state();
    let support = vec![0, 1, 2, 3];
    let (mut max_policy, mut max_value) = (0.0f32, 0.0f32);
    let mut seen = 0;
    while seen < positions {
        if game.is_terminal(&state) {
            game = Battlesnake::new(Rules {
                seed: rng.next_u64(),
                ..Rules::default()
            });
            state = game.initial_state();
            continue;
        }
        if game.turn(&state) == SimultaneousTurn::Chance {
            let action = game.sample_chance_action(&state, &mut rng);
            game.apply_chance(&mut state, action);
            continue;
        }
        for player in 0..2 {
            let features = encoder.encode_state(&game, &state, player);
            let request = crate::net::EvalRequest {
                features,
                support: support.clone(),
            };
            let tch = &infer.forward_batch(std::slice::from_ref(&request))[0];
            let (plain_policy, plain_value) =
                exported.forward_support(&request.features, &[], &support);
            for (&first, &second) in tch.priors.iter().zip(&plain_policy) {
                max_policy = max_policy.max((first - second).abs());
            }
            max_value = max_value.max((tch.value.as_mover() - plain_value).abs());
        }
        let joint = [Direction::ALL[rng.below(4)], Direction::ALL[rng.below(4)]];
        game.apply_joint(&mut state, &joint);
        seen += 1;
    }
    println!(
        "export parity over {positions} joint states: max |policy|={max_policy:.2e}, \
         max |value|={max_value:.2e}"
    );
    assert!(max_policy < 1e-3 && max_value < 1e-3);
}

pub fn main(args: &[String]) {
    match args.first().map(String::as_str) {
        Some("run") => run::run(&args[1..]),
        Some("bench") => run::bench(&args[1..]),
        Some("export") => export(&args[1..]),
        Some("verify-export") => verify_export(&args[1..]),
        Some("rate") => gauge::rate(&args[1..]),
        Some("elo") => gauge::elo_gauge(&args[1..]),
        Some("field") => gauge::field_gauge(&args[1..]),
        Some("compare") => gauge::compare(&args[1..]),
        Some("field-compare") => gauge::field_compare(&args[1..]),
        Some("split-compare") => gauge::split_compare(&args[1..]),
        other => {
            eprintln!(
                "usage: snake <run|bench|export|verify-export|rate|elo|field|compare|field-compare|split-compare> [flags]\n\
                 run methods: --method logit|maximin|policy --players 2|3|4\n\
                 got: {other:?}"
            );
            std::process::exit(2);
        }
    }
}
