//! AlphaZero module tests: backprop vs numerical gradients, checkpoint
//! roundtrip, PUCT legality on an untrained net, and a learning smoke test —
//! all on an inline tic-tac-toe.

use game_core::{Agent, Game, Rng, Turn};
use solvers::azero::{
    AzeroConfig, Mlp, PolicyValueEncoder, Puct, PuctAgent, Sample, SelfPlayTrainer,
};

struct Ttt;

#[derive(Clone)]
struct TttState {
    cells: [u8; 9],
    to_move: usize,
}

const LINES: [[usize; 3]; 8] = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
];

fn winner(cells: &[u8; 9]) -> Option<usize> {
    LINES.iter().find_map(|l| {
        let v = cells[l[0]];
        (v != 0 && cells[l[1]] == v && cells[l[2]] == v).then_some(v as usize - 1)
    })
}

impl Game for Ttt {
    type State = TttState;
    type Action = usize;

    fn initial_state(&self) -> TttState {
        TttState {
            cells: [0; 9],
            to_move: 0,
        }
    }

    fn turn(&self, s: &TttState) -> Turn {
        Turn::Player(s.to_move)
    }

    fn is_terminal(&self, s: &TttState) -> bool {
        winner(&s.cells).is_some() || s.cells.iter().all(|&c| c != 0)
    }

    fn returns(&self, s: &TttState, p: usize) -> f64 {
        match winner(&s.cells) {
            Some(w) if w == p => 1.0,
            Some(_) => -1.0,
            None => 0.0,
        }
    }

    fn legal_actions(&self, s: &TttState) -> Vec<usize> {
        (0..9).filter(|&i| s.cells[i] == 0).collect()
    }

    fn chance_outcomes(&self, _s: &TttState) -> Vec<(usize, f64)> {
        vec![]
    }

    fn apply(&self, s: &mut TttState, a: usize) {
        s.cells[a] = s.to_move as u8 + 1;
        s.to_move ^= 1;
    }

    fn infoset_key(&self, s: &TttState, _p: usize) -> u64 {
        s.cells.iter().fold(0u64, |k, &c| k * 3 + u64::from(c))
    }
}

struct TttEnc;

impl PolicyValueEncoder<Ttt> for TttEnc {
    fn input_len(&self) -> usize {
        19
    }

    fn policy_len(&self) -> usize {
        9
    }

    fn encode_state(&self, _g: &Ttt, s: &TttState) -> Vec<f32> {
        let mut x = vec![0.0; 19];
        for (i, &c) in s.cells.iter().enumerate() {
            match c {
                1 => x[i] = 1.0,
                2 => x[9 + i] = 1.0,
                _ => {}
            }
        }
        x[18] = s.to_move as f32;
        x
    }

    fn action_index(&self, _g: &Ttt, _s: &TttState, a: usize) -> usize {
        a
    }
}

#[test]
fn backprop_matches_numerical_gradients() {
    let mut net = Mlp::new(4, 6, 5, 7);
    let samples = [
        Sample {
            x: vec![0.3, -0.7, 1.2, 0.05],
            policy: vec![(0, 0.2), (2, 0.5), (4, 0.3)],
            z: 0.6,
        },
        Sample {
            x: vec![-1.0, 0.4, 0.0, 0.9],
            policy: vec![(1, 0.7), (3, 0.3)],
            z: -0.4,
        },
    ];
    let batch: Vec<&Sample> = samples.iter().collect();
    let mut analytic = Vec::new();
    net.grad(&batch, &mut analytic);

    let eps = 2e-3f32;
    let combined = |net: &Mlp| {
        let (pl, vl) = net.loss(&batch);
        pl + vl
    };
    for (i, &expected) in analytic.iter().enumerate() {
        let orig = net.params()[i];
        net.params_mut()[i] = orig + eps;
        let plus = combined(&net);
        net.params_mut()[i] = orig - eps;
        let minus = combined(&net);
        net.params_mut()[i] = orig;
        let numeric = (plus - minus) / (2.0 * eps);
        let tol = 1e-3 + 0.02 * numeric.abs().max(expected.abs());
        assert!(
            (numeric - expected).abs() <= tol,
            "param {i}: numeric {numeric} vs backprop {expected}"
        );
    }
}

#[test]
fn save_load_roundtrip() {
    let net = Mlp::new(11, 8, 7, 99);
    let path = std::env::temp_dir().join(format!("azero-roundtrip-{}.bin", std::process::id()));
    net.save(&path).unwrap();
    let back = Mlp::load(&path).unwrap();
    std::fs::remove_file(&path).ok();

    assert_eq!(
        (net.input_len(), net.hidden_len(), net.policy_len()),
        (back.input_len(), back.hidden_len(), back.policy_len())
    );
    assert_eq!(net.params(), back.params());
    let x: Vec<f32> = (0..11).map(|i| i as f32 / 7.0 - 0.6).collect();
    let (p1, v1) = net.policy_value(&x, &[0, 3, 6]);
    let (p2, v2) = back.policy_value(&x, &[0, 3, 6]);
    assert_eq!(p1, p2);
    assert_eq!(v1, v2);
}

#[test]
fn puct_with_untrained_net_returns_legal_indices() {
    let net = Mlp::new(19, 16, 9, 5);
    let game = &Ttt;
    let mut puct = Puct::new(game, &TttEnc, &net, 64);
    puct.root_noise = 0.25;
    puct.dirichlet_alpha = 0.6;

    let mut rng = Rng::new(3);
    let visits = puct.search(&game.initial_state(), &mut rng);
    assert_eq!(visits.len(), 9);
    assert_eq!(visits.iter().sum::<u32>(), 64);

    let agent = PuctAgent(puct);
    let mut s = game.initial_state();
    let mut r = Rng::new(11);
    while !game.is_terminal(&s) {
        let actions = game.legal_actions(&s);
        let Turn::Player(p) = game.turn(&s) else {
            unreachable!()
        };
        let i = agent.act(game, &s, p, r.unit());
        assert!(i < actions.len(), "illegal index {i} of {}", actions.len());
        game.apply(&mut s, actions[i]);
    }
}

#[test]
fn training_reduces_loss_on_tictactoe() {
    let cfg = AzeroConfig {
        hidden: 32,
        sims: 48,
        dirichlet_alpha: 0.6,
        temp_moves: 3,
        max_game_len: 9,
        games_per_iter: 24,
        replay_capacity: 4096,
        batch_size: 32,
        batches_per_iter: 50,
        lr: 0.05,
        ..AzeroConfig::default()
    };
    let mut trainer = SelfPlayTrainer::new(&Ttt, &TttEnc, cfg, 42);
    let first = trainer.iterate(1);
    let mut last = trainer.iterate(2);
    for seed in 3..=4 {
        last = trainer.iterate(seed);
    }
    assert!(first.total_loss().is_finite() && last.total_loss().is_finite());
    assert!(
        last.total_loss() < first.total_loss(),
        "200 SGD steps did not reduce loss: first {} last {}",
        first.total_loss(),
        last.total_loss()
    );
}
