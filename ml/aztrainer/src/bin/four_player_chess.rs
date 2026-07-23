//! Four-player chess trainer binary — the standard Chess.com FFA game plugged
//! into the generic AlphaZero trainer.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    aztrainer::games::four_player_chess::main(&args);
}
