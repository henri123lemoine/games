//! The chess (8×8) trainer binary — a thin entry into the generic `aztrainer`
//! core with chess's game knowledge plugged in (see `aztrainer::games::chess`).

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    aztrainer::games::chess::main(&args);
}
