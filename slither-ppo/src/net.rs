//! The slither policy/value net in tch: a 3-conv CNN over the egocentric
//! semantic grid (10×32×32), global-pooled and flattened, concatenated with the
//! scalar vector, then three heads — turn-bucket logits, a single boost logit,
//! and a scalar value. This is the agar.io-validated shape: a small CNN over a
//! low-res semantic grid beats an MLP or raw pixels.
//!
//! The two policy heads are independent: turn is a categorical over
//! [`TURN_BUCKETS`], boost is a Bernoulli over its one logit. They factor, so the
//! joint log-prob is the sum and the joint entropy is the sum — which is exactly
//! what PPO needs and what [`Policy::act`]/[`Policy::evaluate`] return.
//!
//! All tch calls for one VarStore must stay on one thread.

use slither_rl::env::SHAPES;
use tch::nn::{self, ConvConfig};
use tch::{Device, Kind, Tensor};

pub const TURN_BUCKETS: i64 = SHAPES.turn_buckets as i64;
pub const CHANNELS: i64 = SHAPES.grid.0 as i64;
pub const GRID: i64 = SHAPES.grid.1 as i64;
pub const SCALARS: i64 = SHAPES.scalars as i64;

const CONV1: i64 = 32;
const CONV2: i64 = 64;
const CONV3: i64 = 64;
const HIDDEN: i64 = 256;

/// VarStore tensor names in the exact order the `SLNET1` browser export writes
/// them — the contract `slitherinfer` parses against. Each conv/linear stores
/// `<path>.weight` then `<path>.bias`; the order matches the forward pass.
pub const EXPORT_ORDER: [&str; 14] = [
    "c1.weight",
    "c1.bias",
    "c2.weight",
    "c2.bias",
    "c3.weight",
    "c3.bias",
    "trunk.weight",
    "trunk.bias",
    "turn.weight",
    "turn.bias",
    "boost.weight",
    "boost.bias",
    "value.weight",
    "value.bias",
];

fn conv(p: nn::Path, cin: i64, cout: i64, k: i64, stride: i64) -> nn::Conv2D {
    let cfg = ConvConfig {
        stride,
        padding: (k - 1) / 2,
        ..Default::default()
    };
    nn::conv2d(p, cin, cout, k, cfg)
}

/// Flattened conv output width after the three strided convs over a `GRID×GRID`
/// input. Stride-2 on conv2 and conv3 halves the spatial extent twice: 32→32→16→8.
const CONV_OUT_HW: i64 = GRID / 4;
const CONV_FLAT: i64 = CONV3 * CONV_OUT_HW * CONV_OUT_HW;

pub struct Policy {
    c1: nn::Conv2D,
    c2: nn::Conv2D,
    c3: nn::Conv2D,
    trunk: nn::Linear,
    turn_head: nn::Linear,
    boost_head: nn::Linear,
    value_head: nn::Linear,
}

/// What a forward pass needs back for PPO: the per-action log-prob, the joint
/// entropy, and the value estimate. Built by [`Policy::evaluate`] over a minibatch.
pub struct Eval {
    pub log_prob: Tensor,
    pub entropy: Tensor,
    pub value: Tensor,
}

impl Policy {
    pub fn new(root: &nn::Path) -> Policy {
        Policy {
            c1: conv(root / "c1", CHANNELS, CONV1, 3, 1),
            c2: conv(root / "c2", CONV1, CONV2, 3, 2),
            c3: conv(root / "c3", CONV2, CONV3, 3, 2),
            trunk: nn::linear(
                root / "trunk",
                CONV_FLAT + SCALARS,
                HIDDEN,
                Default::default(),
            ),
            turn_head: nn::linear(root / "turn", HIDDEN, TURN_BUCKETS, Default::default()),
            boost_head: nn::linear(root / "boost", HIDDEN, 1, Default::default()),
            value_head: nn::linear(root / "value", HIDDEN, 1, Default::default()),
        }
    }

    /// Shared trunk: conv stack over the grid, flattened and concatenated with the
    /// scalars, through one hidden layer. `grid`: `[B, CHANNELS, GRID, GRID]`,
    /// `scalars`: `[B, SCALARS]` → `[B, HIDDEN]`.
    fn features(&self, grid: &Tensor, scalars: &Tensor) -> Tensor {
        let h = grid
            .apply(&self.c1)
            .relu()
            .apply(&self.c2)
            .relu()
            .apply(&self.c3)
            .relu()
            .flatten(1, -1);
        let h = Tensor::cat(&[h, scalars.shallow_clone()], 1);
        h.apply(&self.trunk).relu()
    }

    /// Raw head outputs for a batch: `(turn_logits [B, TURN_BUCKETS], boost_logit
    /// [B, 1], value [B])`.
    fn heads(&self, grid: &Tensor, scalars: &Tensor) -> (Tensor, Tensor, Tensor) {
        let f = self.features(grid, scalars);
        let turn = f.apply(&self.turn_head);
        let boost = f.apply(&self.boost_head);
        let value = f.apply(&self.value_head).squeeze_dim(-1);
        (turn, boost, value)
    }

    /// Raw head outputs for export verification: `(turn_logits [B,
    /// TURN_BUCKETS], boost_logit [B], value [B])`, no sampling, no graph. The
    /// browser export's reference forward must reproduce these.
    pub fn raw_heads(&self, grid: &Tensor, scalars: &Tensor) -> (Tensor, Tensor, Tensor) {
        tch::no_grad(|| {
            let (turn, boost, value) = self.heads(grid, scalars);
            (turn, boost.squeeze_dim(-1), value)
        })
    }

    /// Sample actions for a batch of observations (rollout / inference). Returns
    /// `(turn_idx [B] i64, boost [B] i64 in {0,1}, log_prob [B], value [B])`, all
    /// on CPU, detached — no graph is built. Used to step the env.
    pub fn act(&self, grid: &Tensor, scalars: &Tensor) -> (Tensor, Tensor, Tensor, Tensor) {
        tch::no_grad(|| {
            let (turn_logits, boost_logit, value) = self.heads(grid, scalars);

            let turn_probs = turn_logits.softmax(-1, Kind::Float);
            let turn_idx = turn_probs.multinomial(1, true).squeeze_dim(-1);

            let boost_p = boost_logit.squeeze_dim(-1).sigmoid();
            let boost = boost_p.bernoulli();

            let log_prob = joint_log_prob(&turn_logits, &boost_logit, &turn_idx, &boost);
            (
                turn_idx.to_device(Device::Cpu),
                boost.to_kind(Kind::Int64).to_device(Device::Cpu),
                log_prob.to_device(Device::Cpu),
                value.to_device(Device::Cpu),
            )
        })
    }

    /// Greedy (argmax turn, boost if p>0.5) actions for eval — no sampling noise.
    pub fn act_greedy(&self, grid: &Tensor, scalars: &Tensor) -> (Tensor, Tensor) {
        tch::no_grad(|| {
            let (turn_logits, boost_logit, _v) = self.heads(grid, scalars);
            let turn_idx = turn_logits.argmax(-1, false);
            let boost = boost_logit.squeeze_dim(-1).ge(0.0).to_kind(Kind::Int64);
            (
                turn_idx.to_device(Device::Cpu),
                boost.to_device(Device::Cpu),
            )
        })
    }

    /// Just the value estimate for a batch (bootstrap of the last rollout step).
    pub fn value(&self, grid: &Tensor, scalars: &Tensor) -> Tensor {
        tch::no_grad(|| {
            let (_t, _b, value) = self.heads(grid, scalars);
            value.to_device(Device::Cpu)
        })
    }

    /// Re-evaluate stored actions under the current params (PPO update): builds the
    /// graph and returns log-prob, entropy, value for the clipped surrogate.
    pub fn evaluate(
        &self,
        grid: &Tensor,
        scalars: &Tensor,
        turn_idx: &Tensor,
        boost: &Tensor,
    ) -> Eval {
        let (turn_logits, boost_logit, value) = self.heads(grid, scalars);
        let log_prob = joint_log_prob(&turn_logits, &boost_logit, turn_idx, boost);
        let entropy = categorical_entropy(&turn_logits) + bernoulli_entropy(&boost_logit);
        Eval {
            log_prob,
            entropy,
            value,
        }
    }
}

/// Joint log-prob of (turn bucket, boost bit) under the factored heads:
/// log p(turn) + log p(boost). `turn_idx`: `[B]` i64; `boost`: `[B]` float/int in
/// {0,1}.
fn joint_log_prob(
    turn_logits: &Tensor,
    boost_logit: &Tensor,
    turn_idx: &Tensor,
    boost: &Tensor,
) -> Tensor {
    let turn_logp = turn_logits
        .log_softmax(-1, Kind::Float)
        .gather(1, &turn_idx.unsqueeze(-1), false)
        .squeeze_dim(-1);
    // Bernoulli log-prob from the logit: b*logsigmoid(x) + (1-b)*logsigmoid(-x).
    let x = boost_logit.squeeze_dim(-1);
    let b = boost.to_kind(Kind::Float);
    let boost_logp = &b * x.log_sigmoid() + (1.0f64 - &b) * (-&x).log_sigmoid();
    turn_logp + boost_logp
}

/// Entropy of the categorical turn head, `[B]`.
fn categorical_entropy(logits: &Tensor) -> Tensor {
    let logp = logits.log_softmax(-1, Kind::Float);
    let p = logp.exp();
    -(p * logp).sum_dim_intlist(-1, false, Kind::Float)
}

/// Entropy of the Bernoulli boost head from its logit, `[B]`.
fn bernoulli_entropy(boost_logit: &Tensor) -> Tensor {
    let x = boost_logit.squeeze_dim(-1);
    let p = x.sigmoid();
    let log_p = x.log_sigmoid();
    let log_1mp = (-&x).log_sigmoid();
    -(&p * log_p + (1.0f64 - &p) * log_1mp)
}

/// Save the policy's weights to a `.ot` checkpoint.
pub fn save(vs: &nn::VarStore, path: &std::path::Path) -> Result<(), tch::TchError> {
    vs.save(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Tensor};

    fn policy() -> (nn::VarStore, Policy) {
        let vs = nn::VarStore::new(Device::Cpu);
        let p = Policy::new(&vs.root());
        (vs, p)
    }

    /// `act` returns the right shapes and dtypes, and the heads' value/log-prob are
    /// finite — a smoke check the conv arithmetic (32→16→8) lines up with the
    /// flatten width.
    #[test]
    fn act_shapes() {
        let (_vs, p) = policy();
        let b = 4i64;
        let grid = Tensor::randn([b, CHANNELS, GRID, GRID], (Kind::Float, Device::Cpu));
        let scalars = Tensor::randn([b, SCALARS], (Kind::Float, Device::Cpu));
        let (turn, boost, logp, value) = p.act(&grid, &scalars);
        assert_eq!(turn.size(), [b]);
        assert_eq!(boost.size(), [b]);
        assert_eq!(logp.size(), [b]);
        assert_eq!(value.size(), [b]);
        let turns: Vec<i64> = (&turn).try_into().unwrap();
        assert!(turns.iter().all(|&t| (0..TURN_BUCKETS).contains(&t)));
        let lp: Vec<f32> = (&logp).try_into().unwrap();
        assert!(lp.iter().all(|x| x.is_finite() && *x <= 0.0));
    }

    /// `evaluate`'s joint log-prob equals log p(turn) + log p(boost) computed by
    /// hand from the same logits — guards the factored-head math PPO relies on.
    #[test]
    fn joint_log_prob_factors_correctly() {
        let turn_logits = Tensor::from_slice(&[1.0f32, 0.0, -1.0, 2.0]).reshape([1, 4]);
        let boost_logit = Tensor::from_slice(&[0.5f32]).reshape([1, 1]);
        let turn_idx = Tensor::from_slice(&[2i64]);
        let boost = Tensor::from_slice(&[1.0f32]);

        let lp = joint_log_prob(&turn_logits, &boost_logit, &turn_idx, &boost);
        let got = f32::try_from(&lp).unwrap();

        // Manual: softmax over [1,0,-1,2], pick index 2; boost bit=1 → log σ(0.5).
        let logits = [1.0f32, 0.0, -1.0, 2.0];
        let max = logits.iter().cloned().fold(f32::MIN, f32::max);
        let denom: f32 = logits.iter().map(|x| (x - max).exp()).sum();
        let turn_lp = (logits[2] - max) - denom.ln();
        let boost_lp = (1.0 / (1.0 + (-0.5f32).exp())).ln();
        let expected = turn_lp + boost_lp;
        assert!(
            (got - expected).abs() < 1e-5,
            "got {got} expected {expected}"
        );
    }

    /// Entropy of a near-uniform categorical head is close to ln(TURN_BUCKETS); a
    /// peaked head is much lower. Sanity on the entropy term that drives
    /// exploration.
    #[test]
    fn categorical_entropy_bounds() {
        let uniform = Tensor::zeros([1, TURN_BUCKETS], (Kind::Float, Device::Cpu));
        let h_uniform = f32::try_from(&categorical_entropy(&uniform)).unwrap();
        let ln_k = (TURN_BUCKETS as f32).ln();
        assert!((h_uniform - ln_k).abs() < 1e-4);

        let mut peaked = vec![0.0f32; TURN_BUCKETS as usize];
        peaked[0] = 20.0;
        let peaked = Tensor::from_slice(&peaked).reshape([1, TURN_BUCKETS]);
        let h_peaked = f32::try_from(&categorical_entropy(&peaked)).unwrap();
        assert!(h_peaked < 0.01, "peaked entropy {h_peaked}");
    }

    /// A round-trip through save/load reproduces the same action distribution —
    /// checkpoints are faithful, so a pooled snapshot plays identically.
    #[test]
    fn checkpoint_round_trip() {
        let dir = std::env::temp_dir().join(format!("slither_ppo_ckpt_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.ot");

        let (vs, p) = policy();
        save(&vs, &path).unwrap();

        let grid = Tensor::ones([2, CHANNELS, GRID, GRID], (Kind::Float, Device::Cpu));
        let scalars = Tensor::ones([2, SCALARS], (Kind::Float, Device::Cpu));
        let (t0, _b0) = p.act_greedy(&grid, &scalars);

        let mut vs2 = nn::VarStore::new(Device::Cpu);
        let p2 = Policy::new(&vs2.root());
        vs2.load(&path).unwrap();
        let (t1, _b1) = p2.act_greedy(&grid, &scalars);

        let a: Vec<i64> = (&t0).try_into().unwrap();
        let b: Vec<i64> = (&t1).try_into().unwrap();
        assert_eq!(a, b);
        std::fs::remove_dir_all(&dir).ok();
    }
}
