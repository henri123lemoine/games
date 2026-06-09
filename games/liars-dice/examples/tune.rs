//! Self-play tuning of the probabilistic agent: hill-climb its config by
//! repeatedly proposing a perturbed challenger and promoting it when it beats the
//! current champion head-to-head at the target configuration.
//!
//!     cargo run --release -p liars-dice --example tune [players] [dice] [faces] [steps] [games]

use cfr_core::winrate_vs_field;
use liars_dice::{LiarsDice, ProbConfig, ProbabilisticAgent, RandomAgent};

fn arg<T: std::str::FromStr>(i: usize, d: T) -> T {
    std::env::args()
        .nth(i)
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Symmetric perturbation in `[-m, m]`.
    fn jitter(&mut self, m: f64) -> f64 {
        (self.unit() - 0.5) * 2.0 * m
    }
}

fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

fn perturb(c: ProbConfig, rng: &mut Rng) -> ProbConfig {
    ProbConfig {
        liar_cut: clamp(c.liar_cut + rng.jitter(0.05), 0.05, 0.60),
        exact_cut: clamp(c.exact_cut + rng.jitter(0.05), 0.10, 0.70),
        safety: clamp(c.safety + rng.jitter(0.06), 0.10, 0.85),
        bluff: clamp(c.bluff + rng.jitter(0.04), 0.0, 0.35),
        bidder_bias: clamp(c.bidder_bias + rng.jitter(0.20), 0.0, 2.5),
    }
}

fn main() {
    let players: u8 = arg(1, 5);
    let dice: u8 = arg(2, 5);
    let faces: u8 = arg(3, 6);
    let steps: u32 = arg(4, 80);
    let games: u32 = arg(5, 3000);

    let game = LiarsDice::new(players, dice, faces);
    let default = ProbConfig::default();
    let mut champion = default;
    let mut rng = Rng(0x5EED_1234);

    let anchor = |c: ProbConfig| {
        let hero = ProbabilisticAgent::new(c);
        let base = RandomAgent;
        winrate_vs_field(&game, &hero, &base, 4000, 0x9999)
    };

    println!("Tuning {players}p{dice}d{faces}f — {steps} steps × {games} games/step");
    println!(
        "start: vs-random {:.3}  cfg {:?}",
        anchor(champion),
        champion
    );

    // A lone challenger against a field of champions beats its fair share 1/n
    // exactly when it is the stronger config; require a margin to clear noise.
    let fair = 1.0 / players as f64;
    let mut promotions = 0;
    for step in 0..steps {
        let challenger = perturb(champion, &mut rng);
        let hero = ProbabilisticAgent::new(challenger);
        let field = ProbabilisticAgent::new(champion);
        let wr = winrate_vs_field(&game, &hero, &field, games, 0x1357 + step as u64);
        if wr > fair + 0.012 {
            champion = challenger;
            promotions += 1;
            println!(
                "  step {step:>3}: promote (challenger {wr:.3} > fair {fair:.3})  liar={:.3} exact={:.3} safe={:.3} bluff={:.3} bias={:.2}",
                champion.liar_cut,
                champion.exact_cut,
                champion.safety,
                champion.bluff,
                champion.bidder_bias
            );
        }
    }

    // Final head-to-head: champion vs the original default.
    let champ = ProbabilisticAgent::new(champion);
    let def = ProbabilisticAgent::new(default);
    let vs_default = winrate_vs_field(&game, &champ, &def, 12000, 0x2024);
    println!("\n{promotions} promotions");
    println!("champion vs-random : {:.3}", anchor(champion));
    println!("champion vs-default: {:.3}  (0.5 = tie)", vs_default);
    println!("champion cfg: {champion:?}");
}
