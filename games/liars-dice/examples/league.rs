//! Robust self-play tuning against a *diverse league* of opponent styles, plus
//! champion snapshots (fictitious play). Maximizes the agent's average win share
//! across the whole panel so it doesn't merely overfit one opponent.
//!
//!     cargo run --release -p liars-dice --example league [players] [dice] [faces] [steps] [games]

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
    fn jitter(&mut self, m: f64) -> f64 {
        (self.unit() - 0.5) * 2.0 * m
    }
}

fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

fn perturb(c: ProbConfig, rng: &mut Rng) -> ProbConfig {
    ProbConfig {
        liar_cut: clamp(c.liar_cut + rng.jitter(0.04), 0.05, 0.60),
        exact_cut: clamp(c.exact_cut + rng.jitter(0.04), 0.10, 0.70),
        safety: clamp(c.safety + rng.jitter(0.05), 0.10, 0.85),
        bluff: clamp(c.bluff + rng.jitter(0.03), 0.0, 0.35),
        bidder_bias: clamp(c.bidder_bias + rng.jitter(0.15), 0.0, 2.5),
        open_frac: clamp(c.open_frac + rng.jitter(0.10), 0.0, 1.0),
        mix: clamp(c.mix + rng.jitter(0.03), 0.0, 0.25),
    }
}

fn cfg(liar: f64, exact: f64, safe: f64, bluff: f64, bias: f64, open: f64, mix: f64) -> ProbConfig {
    ProbConfig {
        liar_cut: liar,
        exact_cut: exact,
        safety: safe,
        bluff,
        bidder_bias: bias,
        open_frac: open,
        mix,
    }
}

fn main() {
    let players: u8 = arg(1, 5);
    let dice: u8 = arg(2, 5);
    let faces: u8 = arg(3, 6);
    let steps: u32 = arg(4, 150);
    let games: u32 = arg(5, 1200);

    let game = LiarsDice::new(players, dice, faces);

    // A spread of distinct styles the champion must beat on average.
    let mut league: Vec<ProbConfig> = vec![
        ProbConfig::default(),
        ProbConfig::baseline(),
        cfg(0.20, 0.45, 0.55, 0.03, 0.9, 0.3, 0.0), // conservative / trusting
        cfg(0.42, 0.30, 0.25, 0.20, 0.3, 0.7, 0.1), // aggressive bluffer
        cfg(0.22, 0.25, 0.40, 0.05, 1.3, 0.4, 0.05), // very trusting, exact-happy
        cfg(0.45, 0.40, 0.30, 0.02, 0.1, 0.6, 0.0), // paranoid caller
        cfg(0.30, 0.50, 0.45, 0.10, 0.6, 0.8, 0.08), // exact-focused
    ];

    let score = |c: ProbConfig, league: &[ProbConfig]| -> f64 {
        let hero = ProbabilisticAgent::new(c);
        let mut s = 0.0;
        for (i, m) in league.iter().enumerate() {
            let field = ProbabilisticAgent::new(*m);
            s += winrate_vs_field(&game, &hero, &field, games, 0x100 + i as u64);
        }
        s / league.len() as f64
    };

    let fair = 1.0 / players as f64;
    let mut champion = ProbConfig::default();
    let mut champ_score = score(champion, &league);
    let mut rng = Rng(0xA11CE);
    println!("League tuning {players}p{dice}d{faces}f — {steps} steps × {games} games");
    println!("start score {champ_score:.3} (fair {fair:.3})  cfg {champion:?}");

    let mut promotions = 0u32;
    for step in 0..steps {
        let challenger = perturb(champion, &mut rng);
        let sc = score(challenger, &league);
        if sc > champ_score + 0.003 {
            champion = challenger;
            champ_score = sc;
            promotions += 1;
            println!(
                "  step {step:>3}: score {sc:.3}  liar={:.3} exact={:.3} safe={:.3} bluff={:.3} bias={:.2}",
                champion.liar_cut,
                champion.exact_cut,
                champion.safety,
                champion.bluff,
                champion.bidder_bias
            );
            // Fictitious play: periodically add the champion to the league.
            if promotions % 4 == 0 {
                league.push(champion);
                champ_score = score(champion, &league);
            }
        }
    }

    let champ = ProbabilisticAgent::new(champion);
    let def = ProbabilisticAgent::new(ProbConfig::default());
    let rand = RandomAgent;
    let vs_default = winrate_vs_field(&game, &champ, &def, 12000, 0x2024);
    let vs_random = winrate_vs_field(&game, &champ, &rand, 8000, 0x9999);
    println!("\n{promotions} promotions, league size {}", league.len());
    println!("champion vs-random : {vs_random:.3}");
    println!("champion vs-default: {vs_default:.3}  (fair {fair:.3})");
    println!("champion cfg: {champion:?}");
}
