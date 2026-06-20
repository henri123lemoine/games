//! Headless check that killed bots respawn: build the real `SlitherGame`, force
//! a bot to die, tick once, and confirm the seat comes back alive at the start
//! size with the living-worm count restored.

use slither_engine::SlitherGame;

fn main() {
    let weights = std::fs::read("../app/public/slither/slither.weights")
        .or_else(|_| std::fs::read("web/app/public/slither/slither.weights"))
        .expect("slither.weights not found");

    let mut game = SlitherGame::new(&weights, 8, 700, 12345).expect("construct game");

    let n = game.worm_count();
    let alive0 = game.alive_count();
    println!("worms={n} alive_at_start={alive0}");
    assert_eq!(alive0, n, "everyone alive at start");

    // Kill bot seat 1 directly, then tick: the engine should respawn it.
    game.debug_kill(1);
    let alive_after_kill = game.alive_count();
    println!("alive_after_kill(before tick)={alive_after_kill}");
    assert_eq!(
        alive_after_kill,
        n - 1,
        "one bot dead before the respawn tick"
    );

    game.tick(0.0, false);
    let alive_after_tick = game.alive_count();
    let seat1_len = game.debug_worm_length(1);
    let seat1_dead = game.debug_worm_dead(1);
    println!("alive_after_tick={alive_after_tick} seat1_dead={seat1_dead} seat1_len={seat1_len}");
    assert!(!seat1_dead, "seat 1 must be alive again");
    assert_eq!(alive_after_tick, n, "population recovered after respawn");

    // Hammer it: kill a bunch over many ticks and confirm the count never collapses.
    let mut min_alive = alive_after_tick;
    for t in 0..2000u32 {
        if t % 25 == 0 {
            let victim = 1 + (t as usize % (n - 1));
            game.debug_kill(victim);
        }
        game.tick(0.0, false);
        min_alive = min_alive.min(game.alive_count());
    }
    println!("min_alive_over_run={min_alive} (n={n})");
    assert!(
        min_alive >= n - 1,
        "with steady respawns the arena should never drop more than the just-killed worm"
    );

    println!("OK: bots respawn and the population is maintained");
}
