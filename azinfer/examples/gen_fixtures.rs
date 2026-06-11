//! Writes WebGPU validation fixtures: positions (planes + legal indices)
//! with the reference model's expected priors and values.
//!
//! ```text
//! cargo run --release -p azinfer --example gen_fixtures -- \
//!     <export.bin> <out.json> [positions]
//! ```

use azinfer::EvalRequest;
use azinfer::model::Model;
use chess::encode::{az_move_index, encode_planes};
use chess::{Board, legal_moves};
use game_core::Rng;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let export = args.first().expect("usage: gen_fixtures <export.bin> <out.json> [n]");
    let out = args.get(1).expect("usage: gen_fixtures <export.bin> <out.json> [n]");
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(6);

    let data = std::fs::read(export).expect("read export");
    let model = Model::parse(&data).expect("parse export");

    let mut rng = Rng::new(20260611);
    let mut board = Board::start();
    let mut items = Vec::new();
    let mut plies = 0;
    while items.len() < n {
        let moves = legal_moves(&board);
        if moves.is_empty() || board.halfmove >= 100 || plies > 200 {
            board = Board::start();
            plies = 0;
            continue;
        }
        // Sample positions at varying depths.
        if plies % 17 == (items.len() * 3) % 17 {
            let req = EvalRequest {
                planes: encode_planes(&board),
                support: moves
                    .iter()
                    .map(|&m| az_move_index(m, board.stm) as u16)
                    .collect(),
            };
            let res = &model.eval(std::slice::from_ref(&req))[0];
            let join = |v: Vec<String>| v.join(",");
            items.push(format!(
                r#"{{"fen":"{}","planes":[{}],"support":[{}],"priors":[{}],"value":{}}}"#,
                board.to_fen(),
                join(req.planes.iter().map(|x| format!("{x}")).collect()),
                join(req.support.iter().map(|s| s.to_string()).collect()),
                join(res.priors.iter().map(|p| format!("{p}")).collect()),
                res.value,
            ));
        }
        let i = ((rng.unit() * moves.len() as f64) as usize).min(moves.len() - 1);
        board.apply(moves[i]);
        plies += 1;
    }
    std::fs::write(out, format!("[{}]", items.join(","))).expect("write fixtures");
    println!("wrote {n} fixtures to {out}");
}
