//! PBS value-net encoding and the net-backed leaf.
//!
//! [`PbsNet`] wraps a [`RebelMlp`] behind a fixed, config-invariant encoding of a
//! public belief state. The encoding is seat-rotated so the `traverser` is the
//! reference seat 0, mirroring the reference ReBeL query
//! `[acting, traverser, last_bid, beliefs...]` but generalized to span every
//! supported config (2..=6 players, 2..=6 faces, 0..=`MAX_DICE` dice per seat).
//!
//! Input layout (length [`INPUT_DIM`]):
//!
//! ```text
//! PUBLIC block (PUBLIC_LEN = 37):
//!   [ 0.. 6) rotated dice_left / MAX_DICE   (rotated seat r = original (traverser+r) % players)
//!   [ 6..11) faces one-hot                  (faces in 2..=MAX_FACES → index faces-2)
//!   [11..16) players one-hot                (players in 2..=MAX_REBEL_SEATS → index players-2)
//!   [16..17) bid present flag
//!   [17..18) bid qty / (MAX_REBEL_SEATS*MAX_DICE)
//!   [18..24) bid face one-hot               (0-based face, width MAX_FACES)
//!   [24..25) first_round flag
//!   [25..31) acting seat one-hot, rotated relative to traverser
//!   [31..37) last_bidder one-hot, rotated relative to traverser
//! BELIEF block (MAX_REBEL_SEATS * H):
//!   per rotated seat r, that seat's belief embedded into the global H layout via
//!   `hands::global_index`; padding/eliminated seats contribute zeros.
//! ```
//!
//! Output is `H` raw per-hand values for the traverser (reference seat 0). The
//! traverser's actual hands (a `faces`-face enumeration) are read back from the
//! global slots via [`hands::global_index`]; for a `faces < MAX_FACES` config
//! only those slots carry signal, so [`decode`] gathers them rather than slicing
//! the whole block.

use crate::rebel::game::RebelGame;
use crate::rebel::hands::{self, H, MAX_DICE, MAX_FACES};
use crate::rebel::leaf::LeafValue;
use crate::rebel::pbs::{Belief, PublicState};

use solvers::rebel_mlp::{RebelMlp, RebelMlpConfig, Sample};

use std::io;
use std::path::Path;

/// Seats the encoding spans: covers every 2..=6 player config.
pub const MAX_REBEL_SEATS: usize = 6;

const FACES_OH: usize = MAX_FACES - 1;
const PLAYERS_OH: usize = MAX_REBEL_SEATS - 1;
const BID_LEN: usize = 2 + MAX_FACES;

const OFF_DICE: usize = 0;
const OFF_FACES: usize = OFF_DICE + MAX_REBEL_SEATS;
const OFF_PLAYERS: usize = OFF_FACES + FACES_OH;
const OFF_BID: usize = OFF_PLAYERS + PLAYERS_OH;
const OFF_FIRST: usize = OFF_BID + BID_LEN;
const OFF_ACTING: usize = OFF_FIRST + 1;
const OFF_BIDDER: usize = OFF_ACTING + MAX_REBEL_SEATS;

/// Length of the public block of the encoding.
pub const PUBLIC_LEN: usize = OFF_BIDDER + MAX_REBEL_SEATS;

/// Total network input width: public block plus one global hand layout per seat.
pub const INPUT_DIM: usize = PUBLIC_LEN + MAX_REBEL_SEATS * H;

/// Network output width: one raw value per global hand slot.
pub const OUTPUT_DIM: usize = H;

const QTY_SCALE: f32 = (MAX_REBEL_SEATS * MAX_DICE) as f32;

/// Seat-rotated, config-invariant encoding of `(public, traverser, belief)`.
pub fn encode(public: &PublicState, traverser: usize, belief: &Belief) -> Vec<f32> {
    let mut x = vec![0.0f32; INPUT_DIM];
    encode_into(&mut x, public, traverser, belief);
    x
}

/// [`encode`] into a caller-reused buffer of length [`INPUT_DIM`], zeroing it
/// first. The per-seat belief scatter reads each seat's cached global-index map
/// (no per-call hand enumeration or `global_index` recompute).
pub fn encode_into(x: &mut [f32], public: &PublicState, traverser: usize, belief: &Belief) {
    debug_assert_eq!(x.len(), INPUT_DIM, "encode_into buffer must be INPUT_DIM");
    x.fill(0.0);
    let players = public.players as usize;
    let faces = public.faces;

    for r in 0..players {
        let seat = (traverser + r) % players;
        x[OFF_DICE + r] = f32::from(public.dice_left[seat]) / MAX_DICE as f32;
    }

    x[OFF_FACES + (faces as usize - 2)] = 1.0;
    x[OFF_PLAYERS + (players - 2)] = 1.0;

    if let Some((qty, face)) = public.bid {
        x[OFF_BID] = 1.0;
        x[OFF_BID + 1] = f32::from(qty) / QTY_SCALE;
        x[OFF_BID + 2 + face as usize] = 1.0;
    }

    if public.first_round {
        x[OFF_FIRST] = 1.0;
    }

    let acting_rel = (public.turn + players - traverser) % players;
    x[OFF_ACTING + acting_rel] = 1.0;
    let bidder_rel = (public.last_bidder + players - traverser) % players;
    x[OFF_BIDDER + bidder_rel] = 1.0;

    for r in 0..players {
        let seat = (traverser + r) % players;
        let d = public.dice_left[seat];
        let base = PUBLIC_LEN + r * H;
        for (&g, &mass) in hands::tables(d, faces)
            .global_index
            .iter()
            .zip(&belief.per_seat[seat])
        {
            x[base + g] = mass as f32;
        }
    }
}

/// Read the traverser's per-hand values out of a length-[`H`] network output:
/// gather the global slots of the traverser's `faces`-face hands.
pub fn decode(public: &PublicState, traverser: usize, out: &[f32]) -> Vec<f64> {
    assert_eq!(out.len(), OUTPUT_DIM, "decode expects a length-H output");
    let d = public.dice_left[traverser];
    hands::tables(d, public.faces)
        .global_index
        .iter()
        .map(|&g| f64::from(out[g]))
        .collect()
}

/// A PBS value net: a [`RebelMlp`] plus the fixed PBS encoding/decoding.
pub struct PbsNet {
    net: RebelMlp,
}

impl PbsNet {
    /// A fresh net with `hidden`-wide, `n_layers`-deep body over the fixed PBS
    /// encoding.
    pub fn new(hidden: usize, n_layers: usize, seed: u64) -> PbsNet {
        let cfg = RebelMlpConfig {
            input_dim: INPUT_DIM,
            hidden,
            n_layers,
            output_dim: OUTPUT_DIM,
        };
        PbsNet {
            net: RebelMlp::new(cfg, seed),
        }
    }

    pub fn from_mlp(net: RebelMlp) -> PbsNet {
        assert_eq!(net.input_dim(), INPUT_DIM, "net input dim mismatch");
        assert_eq!(net.output_dim(), OUTPUT_DIM, "net output dim mismatch");
        PbsNet { net }
    }

    pub fn config(&self) -> RebelMlpConfig {
        self.net.config()
    }

    pub fn net(&self) -> &RebelMlp {
        &self.net
    }

    pub fn net_mut(&mut self) -> &mut RebelMlp {
        &mut self.net
    }

    /// The traverser's per-hand values at `(public, belief)`.
    pub fn evaluate(&self, public: &PublicState, traverser: usize, belief: &Belief) -> Vec<f64> {
        let out = self.net.forward(&encode(public, traverser, belief));
        decode(public, traverser, &out)
    }

    /// A training example mapping `(public, traverser, belief)` to the per-hand
    /// `root_values_mean` target, masked to the traverser's hand slots.
    pub fn to_sample(
        &self,
        public: &PublicState,
        traverser: usize,
        belief: &Belief,
        root_values_mean: &[f64],
    ) -> Sample {
        let input = encode(public, traverser, belief);
        let mut target = vec![0.0f32; OUTPUT_DIM];
        let mut mask = vec![0.0f32; OUTPUT_DIM];
        let d = public.dice_left[traverser];
        let table = hands::tables(d, public.faces);
        assert_eq!(
            table.hand_count(),
            root_values_mean.len(),
            "target length must match the traverser's hand count"
        );
        for (&g, &v) in table.global_index.iter().zip(root_values_mean) {
            target[g] = v as f32;
            mask[g] = 1.0;
        }
        Sample {
            input,
            target,
            mask,
        }
    }

    pub fn save(&self, path: &Path) -> io::Result<()> {
        self.net.save(path)
    }

    pub fn load(path: &Path) -> io::Result<PbsNet> {
        Ok(PbsNet::from_mlp(RebelMlp::load(path)?))
    }
}

/// A [`LeafValue`] backed by a [`PbsNet`]. Terminal leaves use the game's exact
/// per-hand payoffs; depth-limited (non-terminal) leaves use the net. Returned
/// values use the same normalized-belief convention as [`crate::rebel::leaf::TerminalLeaf`]
/// — the solver applies opponent-reach scaling.
pub struct NetLeaf<'a, G: RebelGame> {
    net: &'a PbsNet,
    game: &'a G,
}

impl<'a, G: RebelGame> NetLeaf<'a, G> {
    pub fn new(net: &'a PbsNet, game: &'a G) -> Self {
        Self { net, game }
    }
}

impl<G: RebelGame> LeafValue for NetLeaf<'_, G> {
    fn values(&self, public: &PublicState, traverser: usize, belief: &Belief) -> Vec<f64> {
        if self.game.is_terminal(public) {
            return self.game.terminal_cfv(public, traverser, belief);
        }
        self.net.evaluate(public, traverser, belief)
    }

    fn values_batch(
        &self,
        publics: &[PublicState],
        traverser: usize,
        beliefs: &[Belief],
    ) -> Vec<Vec<f64>> {
        let mut out = vec![Vec::new(); publics.len()];
        let mut net_rows: Vec<usize> = Vec::new();
        for (k, (public, belief)) in publics.iter().zip(beliefs).enumerate() {
            if self.game.is_terminal(public) {
                out[k] = self.game.terminal_cfv(public, traverser, belief);
            } else {
                net_rows.push(k);
            }
        }
        if !net_rows.is_empty() {
            let mut inputs = vec![0.0f32; net_rows.len() * INPUT_DIM];
            for (row, &k) in net_rows.iter().enumerate() {
                let dst = &mut inputs[row * INPUT_DIM..(row + 1) * INPUT_DIM];
                encode_into(dst, &publics[k], traverser, &beliefs[k]);
            }
            let raw = self.net.net().forward_batch(&inputs, net_rows.len());
            for (row, &k) in net_rows.iter().enumerate() {
                let slice = &raw[row * OUTPUT_DIM..(row + 1) * OUTPUT_DIM];
                out[k] = decode(&publics[k], traverser, slice);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rebel::cfr::{CfrParams, Solver};
    use crate::rebel::hands::global_index;
    use crate::rebel::pbs::MAX_SEATS;
    use crate::rebel::standard::StandardLiarsDice;

    #[test]
    fn input_and_output_dims_are_consistent() {
        assert_eq!(PUBLIC_LEN, 37);
        assert_eq!(OUTPUT_DIM, H);
        assert_eq!(INPUT_DIM, PUBLIC_LEN + MAX_REBEL_SEATS * H);
        let net = PbsNet::new(32, 2, 0);
        assert_eq!(net.config().input_dim, INPUT_DIM);
        assert_eq!(net.config().output_dim, OUTPUT_DIM);
    }

    fn mid_round_state() -> PublicState {
        let mut dice_left = [0u8; MAX_SEATS];
        dice_left[0] = 1;
        dice_left[1] = 1;
        PublicState {
            players: 2,
            faces: 4,
            dice_left,
            bid: Some((1, 2)),
            turn: 0,
            last_bidder: 1,
            first_round: false,
        }
    }

    #[test]
    fn encode_has_the_documented_layout() {
        let public = mid_round_state();
        let belief = Belief {
            per_seat: vec![vec![0.1, 0.2, 0.3, 0.4], vec![0.4, 0.3, 0.2, 0.1]],
        };
        let x = encode(&public, 0, &belief);
        assert_eq!(x.len(), INPUT_DIM);

        assert!((x[OFF_DICE] - 0.2).abs() < 1e-6);
        assert!((x[OFF_DICE + 1] - 0.2).abs() < 1e-6);
        assert_eq!(x[OFF_DICE + 2], 0.0);

        assert_eq!(x[OFF_FACES + 2], 1.0);
        assert_eq!(x[OFF_PLAYERS], 1.0);

        assert_eq!(x[OFF_BID], 1.0);
        assert!((x[OFF_BID + 1] - 1.0 / QTY_SCALE).abs() < 1e-6);
        assert_eq!(x[OFF_BID + 2 + 2], 1.0);

        assert_eq!(x[OFF_FIRST], 0.0);
        assert_eq!(x[OFF_ACTING], 1.0);
        assert_eq!(x[OFF_BIDDER + 1], 1.0);

        for (r, seat_belief) in belief.per_seat.iter().enumerate() {
            let base = PUBLIC_LEN + r * H;
            let block_sum: f32 = x[base..base + H].iter().sum();
            assert!((block_sum - 1.0).abs() < 1e-5);
            for (hand, &mass) in hands::enumerate(1, 4).iter().zip(seat_belief) {
                let g = global_index(hand, 1);
                assert!((x[base + g] - mass as f32).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn seat_rotation_places_traverser_first() {
        let public = mid_round_state();
        let belief = Belief {
            per_seat: vec![vec![0.1, 0.2, 0.3, 0.4], vec![0.7, 0.1, 0.1, 0.1]],
        };
        let from0 = encode(&public, 0, &belief);
        let from1 = encode(&public, 1, &belief);

        // Seat 1 as traverser sees its own belief in rotated-seat-0's block.
        for (hand, &mass) in hands::enumerate(1, 4).iter().zip(&belief.per_seat[1]) {
            let g = global_index(hand, 1);
            assert!((from1[PUBLIC_LEN + g] - mass as f32).abs() < 1e-6);
        }
        // Acting seat 0 is rotated to relative index 1 from seat 1's view.
        assert_eq!(from1[OFF_ACTING + 1], 1.0);
        assert_eq!(from0[OFF_ACTING], 1.0);
    }

    #[test]
    fn decode_inverts_a_scattered_target() {
        let public = mid_round_state();
        let values = vec![-0.5, 0.25, 0.75, -1.0];
        let net = PbsNet::new(16, 2, 1);
        let belief = Belief::uniform_prior(&public);
        let sample = net.to_sample(&public, 0, &belief, &values);
        let decoded = decode(&public, 0, &sample.target);
        assert_eq!(decoded.len(), values.len());
        for (d, v) in decoded.iter().zip(&values) {
            assert!((d - v).abs() < 1e-6);
        }
        let mask_sum: f32 = sample.mask.iter().sum();
        assert!((mask_sum - values.len() as f32).abs() < 1e-6);
    }

    #[test]
    fn net_leaf_drives_a_solver_on_1x4f() {
        let game = StandardLiarsDice::new(1, 4);
        let net = PbsNet::new(64, 2, 7);
        let leaf = NetLeaf::new(&net, &game);
        let params = CfrParams::default();
        let initial = Belief::uniform_prior(&game.root());
        let mut solver = Solver::new(&game, params, &leaf, initial);
        solver.multistep();
        let avg = solver.average_strategy();
        let tree = solver.tree();
        for (node, policy) in tree.nodes.iter().zip(avg) {
            if node.is_leaf {
                continue;
            }
            for row in policy {
                let sum: f64 = row.iter().sum();
                assert!((sum - 1.0).abs() < 1e-6, "row sums to {sum}");
                assert!(row.iter().all(|&p| (0.0..=1.0).contains(&p)));
            }
        }
    }
}
