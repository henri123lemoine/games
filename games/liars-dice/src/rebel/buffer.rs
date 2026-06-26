//! Uniform replay reservoir for value-net training samples.
//!
//! Vitter's Algorithm R: after seeing `n` items the buffer holds a uniform
//! random subset of size `min(n, cap)` of the stream. Matches the reservoir in
//! `solvers::deepcfr`, without the per-item iteration weight (ReBeL trains the
//! value net with priority off, uniform sampling).

use game_core::Rng;

use solvers::rebel_mlp::Sample;

/// A capped uniform sample of a stream of [`Sample`]s.
pub struct Reservoir {
    buf: Vec<Sample>,
    cap: usize,
    seen: u64,
}

impl Reservoir {
    pub fn new(cap: usize) -> Reservoir {
        Reservoir {
            buf: Vec::new(),
            cap: cap.max(1),
            seen: 0,
        }
    }

    /// Offer one item to the reservoir, keeping the sample uniform over the
    /// stream seen so far.
    pub fn push(&mut self, item: Sample, rng: &mut Rng) {
        self.seen += 1;
        if self.buf.len() < self.cap {
            self.buf.push(item);
        } else {
            let j = (rng.unit() * self.seen as f64) as usize;
            if j < self.cap {
                self.buf[j] = item;
            }
        }
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Total number of items ever offered to the reservoir.
    pub fn seen(&self) -> u64 {
        self.seen
    }

    /// A batch of `n` samples drawn uniformly at random with replacement. Empty
    /// when the reservoir is empty.
    pub fn sample_batch(&self, n: usize, rng: &mut Rng) -> Vec<Sample> {
        if self.buf.is_empty() {
            return Vec::new();
        }
        (0..n)
            .map(|_| self.buf[rng.below(self.buf.len())].clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(tag: f32) -> Sample {
        Sample {
            input: vec![tag],
            target: vec![tag],
            mask: vec![1.0],
        }
    }

    #[test]
    fn fills_then_caps_at_capacity() {
        let mut rng = Rng::new(1);
        let mut r = Reservoir::new(4);
        assert!(r.is_empty());
        for i in 0..10 {
            r.push(sample(i as f32), &mut rng);
        }
        assert_eq!(r.len(), 4);
        assert_eq!(r.seen(), 10);
    }

    #[test]
    fn sample_batch_draws_only_seen_items() {
        let mut rng = Rng::new(2);
        let mut r = Reservoir::new(100);
        for i in 0..20 {
            r.push(sample(i as f32), &mut rng);
        }
        let batch = r.sample_batch(64, &mut rng);
        assert_eq!(batch.len(), 64);
        for s in &batch {
            let tag = s.input[0];
            assert!((0.0..20.0).contains(&tag));
        }
    }

    #[test]
    fn keeps_an_approximately_uniform_sample() {
        let mut rng = Rng::new(3);
        let cap = 1000;
        let mut r = Reservoir::new(cap);
        let stream = 50_000usize;
        for i in 0..stream {
            r.push(sample(i as f32), &mut rng);
        }
        assert_eq!(r.len(), cap);
        let mean = r.buf.iter().map(|s| s.input[0] as f64).sum::<f64>() / cap as f64;
        let expected = (stream - 1) as f64 / 2.0;
        assert!(
            (mean - expected).abs() < 0.15 * expected,
            "reservoir mean {mean} far from uniform expectation {expected}"
        );
    }
}
