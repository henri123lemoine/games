//! The go (9×9…19×19) trainer binary — a thin entry into the generic
//! `aztrainer` core with go's game knowledge plugged in (see
//! `aztrainer::games::go`).

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    aztrainer::games::go::main(&args);
}
