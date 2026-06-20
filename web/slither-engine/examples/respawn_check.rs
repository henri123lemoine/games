//! Headless deploy-parity check for the in-browser slither world: confirms the
//! served engine matches the world the net trained on (6 worms, conservative
//! pellet_target=250 + trickle), that killed bots respawn, and that the body
//! radius follows the cube-root law. Built with `--features debug-hooks`.

use slither_engine::SlitherGame;

const WORMS: usize = 6;
const PELLETS: usize = 250;

fn main() {
    let weights = std::fs::read("../app/public/slither/slither.weights")
        .or_else(|_| std::fs::read("web/app/public/slither/slither.weights"))
        .expect("slither.weights not found");

    // Construct exactly as the page does: WORMS worms, PELLETS pellet target.
    let mut game = SlitherGame::new(&weights, WORMS, PELLETS, 12345).expect("construct game");

    // --- config parity: population and pellet target match the trainer ---
    let n = game.worm_count();
    assert_eq!(n, WORMS, "deploy world must have the trained worm count");
    assert_eq!(
        game.debug_pellet_target(),
        PELLETS,
        "deploy pellet target must match the conservative trainer default"
    );
    println!("worms={n} pellet_target={}", game.debug_pellet_target());

    // --- cube-root radius law (the shipped Rust dynamics) ---
    game.debug_set_length(1, 22.0);
    let r_start = game.debug_worm_radius(1);
    game.debug_set_length(1, 300.0);
    let r_mid = game.debug_worm_radius(1);
    game.debug_set_length(1, 3000.0);
    let r_big = game.debug_worm_radius(1);
    println!("radius: len22={r_start:.2} len300={r_mid:.2} len3000={r_big:.2}");
    assert!(
        (r_start - (5.0 + 3.6_f32)).abs() < 0.01,
        "len==START_LENGTH should give base+growth (cube-root law), got {r_start}"
    );
    assert!(
        r_big < 2.0 * r_mid,
        "10x length must less than double the radius (sublinear), got {r_mid}->{r_big}"
    );
    game.debug_set_length(1, 22.0);

    // --- pellet density settles near the conservative target, not ~3x it ---
    for _ in 0..400 {
        game.tick(0.0, false);
    }
    let pc = game.debug_pellet_count();
    println!("pellet_count_after_400_ticks={pc} (target {PELLETS})");
    assert!(
        pc <= (PELLETS as f32 * 1.6) as usize + 5,
        "pellet field must stay near the conservative target, got {pc}"
    );

    // --- decision rate: every living bot decides every tick (30 Hz == training) ---
    for _ in 0..30 {
        game.tick(0.0, false);
        assert_eq!(
            game.debug_decided_last_tick(),
            game.debug_living_bots(),
            "every living bot must run a forward every tick (no throttle)"
        );
    }
    println!(
        "decision_rate ok: decided {} bots/tick == living bots",
        game.debug_decided_last_tick()
    );

    // --- respawn: a killed bot returns alive next tick, population recovers ---
    let alive0 = game.alive_count();
    game.debug_kill(1);
    assert_eq!(game.alive_count(), alive0 - 1, "kill removed one bot");
    game.tick(0.0, false);
    assert!(!game.debug_worm_dead(1), "seat 1 respawned");
    assert_eq!(game.alive_count(), alive0, "population recovered");
    println!(
        "respawn ok: seat1 back at len={}",
        game.debug_worm_length(1)
    );

    // --- hammer: kill a bot every 25 ticks for a long run; never empties ---
    let mut min_alive = game.alive_count();
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
        "arena should never collapse under respawns"
    );

    println!(
        "OK: deploy world matches training (worms=6, pellets=250), respawn works, cube-root radius"
    );
}
