//! Robust self-play tuning against a *diverse league* of opponent styles, plus
//! champion snapshots (fictitious play). Maximizes the agent's average win share
//! across the whole panel so it doesn't merely overfit one opponent.
//!
//!     cargo run --release -p liars-dice --example league [players] [dice] [faces] [steps] [games]
//!     cargo run --release -p liars-dice --example league -- players=5 dice=5 faces=6 steps=150 games=1200
//!
//! Every run also writes structured metrics to
//! `metrics=runs/ld_league_metrics.jsonl` (`metrics=none` disables). The final
//! `league_final` row records the champion config and validation win-shares so
//! the tuned heuristic baseline can be reproduced in tournament comparisons.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

use game_core::{RandomAgent, winrate_vs_field};
use liars_dice::{LiarsDice, ProbConfig, ProbabilisticAgent};

#[derive(Debug)]
struct Args {
    players: u8,
    dice: u8,
    faces: u8,
    steps: u32,
    games: u32,
    seed: u64,
    metrics: Option<String>,
    append: bool,
}

impl Args {
    fn parse() -> Self {
        let mut positional = Vec::new();
        let mut keyed = HashMap::new();
        for raw in std::env::args().skip(1) {
            if let Some((key, value)) = raw.split_once('=') {
                keyed.insert(key.to_string(), value.to_string());
            } else {
                positional.push(raw);
            }
        }
        let metrics = keyed
            .get("metrics")
            .map(|s| (s != "none").then_some(s.clone()))
            .unwrap_or_else(|| Some("runs/ld_league_metrics.jsonl".to_string()));
        Self {
            players: get_arg(&keyed, "players")
                .or_else(|| pos_arg(&positional, 0))
                .unwrap_or(5),
            dice: get_arg(&keyed, "dice")
                .or_else(|| pos_arg(&positional, 1))
                .unwrap_or(5),
            faces: get_arg(&keyed, "faces")
                .or_else(|| pos_arg(&positional, 2))
                .unwrap_or(6),
            steps: get_arg(&keyed, "steps")
                .or_else(|| pos_arg(&positional, 3))
                .unwrap_or(150),
            games: get_arg(&keyed, "games")
                .or_else(|| pos_arg(&positional, 4))
                .unwrap_or(1200),
            seed: get_arg(&keyed, "seed").unwrap_or(0xA11CE),
            append: parse_bool(keyed.get("append").map(String::as_str).unwrap_or("0")),
            metrics,
        }
    }
}

fn get_arg<T: std::str::FromStr>(args: &HashMap<String, String>, key: &str) -> Option<T> {
    args.get(key).and_then(|s| s.parse().ok())
}

fn pos_arg<T: std::str::FromStr>(args: &[String], idx: usize) -> Option<T> {
    args.get(idx).and_then(|s| s.parse().ok())
}

fn parse_bool(s: &str) -> bool {
    matches!(s, "1" | "true" | "yes" | "on")
}

struct Metrics {
    file: Option<File>,
}

impl Metrics {
    fn open(path: Option<&str>, append: bool) -> io::Result<Self> {
        let Some(path) = path else {
            return Ok(Self { file: None });
        };
        if let Some(parent) = Path::new(path).parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(append)
            .truncate(!append)
            .open(path)?;
        Ok(Self { file: Some(file) })
    }

    fn write(&mut self, fields: Vec<String>) -> io::Result<()> {
        if let Some(file) = &mut self.file {
            writeln!(file, "{{{}}}", fields.join(","))?;
            file.flush()?;
        }
        Ok(())
    }
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
    let args = Args::parse();
    let players = args.players;
    let dice = args.dice;
    let faces = args.faces;
    let steps = args.steps;
    let games = args.games;
    let run_start = Instant::now();
    let mut metrics = Metrics::open(args.metrics.as_deref(), args.append).unwrap_or_else(|e| {
        eprintln!("failed to open metrics file: {e}");
        std::process::exit(1);
    });
    if let Some(path) = args.metrics.as_deref() {
        println!(
            "metrics: {path}{}",
            if args.append { " (append)" } else { "" }
        );
    }

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
    let mut rng = Rng(args.seed);
    println!("League tuning {players}p{dice}d{faces}f — {steps} steps × {games} games");
    println!("start score {champ_score:.3} (fair {fair:.3})  cfg {champion:?}");
    write_metric(
        &mut metrics,
        "league_config",
        vec![
            format!("\"players\":{players}"),
            format!("\"dice\":{dice}"),
            format!("\"faces\":{faces}"),
            format!("\"steps\":{steps}"),
            format!("\"games\":{games}"),
            format!("\"fair\":{fair:.6}"),
            format!("\"seed\":{}", args.seed),
            format!("\"league_size\":{}", league.len()),
            format!("\"start_score\":{champ_score:.6}"),
        ],
    );

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
            write_metric(
                &mut metrics,
                "league_promotion",
                with_config_fields(
                    vec![
                        format!("\"step\":{step}"),
                        format!("\"score\":{sc:.6}"),
                        format!("\"fair\":{fair:.6}"),
                        format!("\"promotions\":{promotions}"),
                        format!("\"league_size\":{}", league.len()),
                        format!("\"wall_s\":{:.6}", run_start.elapsed().as_secs_f64()),
                    ],
                    champion,
                ),
            );
            // Fictitious play: periodically add the champion to the league.
            if promotions.is_multiple_of(4) {
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
    write_metric(
        &mut metrics,
        "league_final",
        with_config_fields(
            vec![
                format!("\"players\":{players}"),
                format!("\"dice\":{dice}"),
                format!("\"faces\":{faces}"),
                format!("\"steps\":{steps}"),
                format!("\"games\":{games}"),
                format!("\"promotions\":{promotions}"),
                format!("\"league_size\":{}", league.len()),
                format!("\"champion_score\":{champ_score:.6}"),
                format!("\"vs_random\":{vs_random:.6}"),
                format!("\"vs_default\":{vs_default:.6}"),
                format!("\"fair\":{fair:.6}"),
                format!("\"wall_s\":{:.6}", run_start.elapsed().as_secs_f64()),
                format!("\"seed\":{}", args.seed),
            ],
            champion,
        ),
    );
}

fn write_metric(metrics: &mut Metrics, event: &str, mut fields: Vec<String>) {
    fields.insert(0, format!("\"event\":\"{event}\""));
    metrics.write(fields).unwrap_or_else(|e| {
        eprintln!("failed to write metrics: {e}");
        std::process::exit(1);
    });
}

fn with_config_fields(mut fields: Vec<String>, c: ProbConfig) -> Vec<String> {
    fields.extend([
        format!("\"liar_cut\":{:.12}", c.liar_cut),
        format!("\"exact_cut\":{:.12}", c.exact_cut),
        format!("\"safety\":{:.12}", c.safety),
        format!("\"bluff\":{:.12}", c.bluff),
        format!("\"bidder_bias\":{:.12}", c.bidder_bias),
        format!("\"open_frac\":{:.12}", c.open_frac),
        format!("\"mix\":{:.12}", c.mix),
    ]);
    fields
}
