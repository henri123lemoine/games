//! Depth-limited public tree built by BFS from a game's root.
//!
//! One action is one public ply. A node is a leaf when it is terminal or when
//! its depth reaches `max_depth`; leaves carry no legal actions or children. The
//! per-hand strategy/regret/reach vectors are owned by the solver, not the tree.

use std::collections::VecDeque;

use crate::rebel::game::{Bid, RebelGame};
use crate::rebel::pbs::PublicState;

#[derive(Clone, Debug)]
pub struct Node {
    pub public: PublicState,
    /// The acting seat, meaningful only when the node is not terminal.
    pub acting: usize,
    /// Legal actions in stable order (empty at a leaf); aligned to `children`.
    pub legal: Vec<Bid>,
    pub children: Vec<usize>,
    pub parent: Option<usize>,
    pub depth: u32,
    pub is_leaf: bool,
    pub is_terminal: bool,
}

pub struct Tree {
    pub nodes: Vec<Node>,
    pub max_depth: u32,
}

impl Tree {
    pub fn build<G: RebelGame>(game: &G, max_depth: u32) -> Tree {
        let mut nodes = vec![make_node(game, game.root(), None, 0, max_depth)];
        let mut queue = VecDeque::from([0usize]);
        while let Some(idx) = queue.pop_front() {
            if nodes[idx].is_leaf {
                continue;
            }
            let legal = nodes[idx].legal.clone();
            let public = nodes[idx].public.clone();
            let depth = nodes[idx].depth;
            let mut children = Vec::with_capacity(legal.len());
            for &a in &legal {
                let child = game.apply(&public, a);
                let child_idx = nodes.len();
                nodes.push(make_node(game, child, Some(idx), depth + 1, max_depth));
                children.push(child_idx);
                queue.push_back(child_idx);
            }
            nodes[idx].children = children;
        }
        Tree { nodes, max_depth }
    }

    pub fn root(&self) -> &Node {
        &self.nodes[0]
    }

    pub fn node(&self, i: usize) -> &Node {
        &self.nodes[i]
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

fn make_node<G: RebelGame>(
    game: &G,
    public: PublicState,
    parent: Option<usize>,
    depth: u32,
    max_depth: u32,
) -> Node {
    let is_terminal = game.is_terminal(&public);
    let is_leaf = is_terminal || depth >= max_depth;
    let acting = game.acting(&public);
    let legal = if is_leaf {
        Vec::new()
    } else {
        game.legal_actions(&public)
    };
    Node {
        public,
        acting,
        legal,
        children: Vec::new(),
        parent,
        depth,
        is_leaf,
        is_terminal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rebel::standard::StandardLiarsDice;

    #[test]
    fn depth_two_tree_on_1x3f_has_the_expected_structure() {
        let game = StandardLiarsDice::new(1, 3);
        let tree = Tree::build(&game, 2);

        // Root: seat 0 to act, all 6 opening bids as children, not a leaf.
        let root = tree.root();
        assert_eq!(root.acting, 0);
        assert_eq!(root.depth, 0);
        assert!(!root.is_leaf);
        assert_eq!(root.children.len(), 6);
        assert!(root.legal.iter().all(|a| matches!(a, Bid::Raise { .. })));

        // Depth-1 nodes: seat 1 to act, decision nodes, children = higher bids + call.
        for &c in &root.children {
            let n = tree.node(c);
            assert_eq!(n.depth, 1);
            assert_eq!(n.acting, 1);
            assert!(!n.is_leaf);
            assert!(!n.is_terminal);
            assert_eq!(*n.legal.last().unwrap(), Bid::Call);
            assert_eq!(n.children.len(), n.legal.len());
        }

        let mut leaves = 0;
        let mut terminals = 0;
        for n in &tree.nodes {
            assert!(n.depth <= 2, "no node exceeds max_depth");
            if n.is_terminal {
                assert!(n.is_leaf, "every terminal node is a leaf");
                terminals += 1;
            }
            if n.is_leaf {
                assert_eq!(n.depth, 2, "leaves sit at the depth limit");
                assert!(n.children.is_empty());
                leaves += 1;
            }
        }
        // 6 openings; opening i (id i, ids 0..6) has (5 - i) higher bids + 1 call.
        // Total nodes = 1 + 6 + Σ_{i=0}^{5}(6 - i) = 1 + 6 + 21 = 28.
        assert_eq!(tree.len(), 28);
        assert_eq!(leaves, 21);
        assert_eq!(terminals, 6);
    }
}
