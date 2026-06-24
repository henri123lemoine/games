//! The pente (5×5…19×19) trainer binary — a thin entry into the generic
//! `aztrainer` core with pente's game knowledge plugged in (see
//! `aztrainer::games::pente`).

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    aztrainer::games::pente::main(&args);
}
