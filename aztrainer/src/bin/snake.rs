//! The snake (1v1 Duel, 20×20) trainer binary — a thin entry into the generic
//! `aztrainer` core with snake's game knowledge plugged in (see
//! `aztrainer::games::snake`).

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    aztrainer::games::snake::main(&args);
}
