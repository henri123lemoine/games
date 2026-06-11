//! External-sampling MCCFR convergence on Kuhn poker — the same yardstick the
//! exact solver and the outcome-sampling variant are held to: the learned
//! average strategy's exact NashConv must approach zero.

use solvers::Mccfr;

mod common;
use common::{Kuhn, kuhn_nashconv};

#[test]
fn kuhn_converges_toward_nash() {
    let mut solver = Mccfr::new(Kuhn, 0xE5);
    solver.run(150_000);
    assert_eq!(solver.num_infosets(), 12, "Kuhn has 12 information sets");

    let nashconv = kuhn_nashconv(&|s, p| solver.policy(s, p));
    assert!(
        nashconv < 0.05,
        "external-sampling MCCFR should approach Nash on Kuhn, NashConv = {nashconv}"
    );
}
