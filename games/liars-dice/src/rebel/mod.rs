//! ReBeL foundation for Liar's Dice: multiset hand enumeration with a fixed
//! global index space, public belief states with Bayesian propagation, the
//! public-tree game interface the vector-CFR solver consumes, the reference
//! standard-rules game for the paper Table-2 exploitability gate, and a
//! depth-limited public tree builder.

pub mod cfr;
pub mod exploit;
pub mod game;
pub mod hands;
pub mod leaf;
pub mod pbs;
pub mod standard;
pub mod tree;

pub use cfr::{CfrParams, CfrVariant, Solver};
pub use exploit::{best_response, exploitability};
pub use game::{Bid, RebelGame};
pub use hands::{H, MAX_DICE, MAX_FACES};
pub use leaf::{LeafValue, PerfectOracleLeaf, TerminalLeaf};
pub use pbs::{Belief, Pbs, PublicState, bayes_update, propagate};
pub use standard::StandardLiarsDice;
pub use tree::{Node, Tree};
