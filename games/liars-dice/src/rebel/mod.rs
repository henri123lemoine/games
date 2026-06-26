//! ReBeL foundation for Liar's Dice: multiset hand enumeration with a fixed
//! global index space, public belief states with Bayesian propagation, the
//! public-tree game interface the vector-CFR solver consumes, the reference
//! standard-rules game for the paper Table-2 exploitability gate, a depth-limited
//! public tree builder, the PBS value-net encoding and net-backed leaf, the
//! recursive self-play data generator, the replay reservoir, and the self-play
//! training loop with its recursive-policy exploitability gate.

pub mod adapter;
pub mod agent;
pub mod buffer;
pub mod cfr;
pub mod deploy;
pub mod deploy_train;
pub mod exploit;
pub mod game;
pub mod hands;
pub mod leaf;
pub mod pbs;
pub mod selfplay;
pub mod standard;
pub mod train;
pub mod tree;
pub mod value_net;

pub use adapter::{LiarsDiceAdapter, principled_open_cap};
pub use agent::RebelAgent;
pub use buffer::Reservoir;
pub use cfr::{CfrParams, CfrVariant, Solver};
pub use deploy::{DeployCont, NetContinuation};
pub use deploy_train::{
    DeployReport, DeployRound, DeployTrainConfig, DeployTrainer, sample_deploy_round,
};
pub use exploit::{best_response, exploitability};
pub use game::{Bid, RebelGame};
pub use hands::{H, MAX_DICE, MAX_FACES};
pub use leaf::{LeafValue, PerfectOracleLeaf, RootedGame, TerminalLeaf};
pub use pbs::{Belief, Pbs, PublicState, bayes_update, propagate};
pub use selfplay::{SelfPlayParams, generate_episode};
pub use standard::StandardLiarsDice;
pub use train::{
    RebelTrainConfig, RebelTrainer, TrainReport, exact_full_nash, recursive_exploitability,
    recursive_strategy, stitch_exploitability,
};
pub use tree::{Node, Tree};
pub use value_net::{
    INPUT_DIM, MAX_REBEL_SEATS, NetLeaf, OUTPUT_DIM, PUBLIC_LEN, PbsNet, decode, encode,
};
