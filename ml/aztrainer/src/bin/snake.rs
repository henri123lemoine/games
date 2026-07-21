//! Canonical 2–4 player simultaneous Battlesnake training and evaluation.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    aztrainer::games::snake::main(&args);
}
