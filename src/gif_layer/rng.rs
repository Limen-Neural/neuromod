/// Deterministic SplitMix64 generator.
///
/// Used instead of a `rand` RNG so that generated topology is reproducible
/// across `rand` releases and across platforms: the whole state transition is
/// wrapping integer arithmetic with fixed constants.
#[derive(Clone, Copy, Debug)]
pub(super) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Sub-stream for one neuron. Mixing the index into the seed (rather than
    /// consuming a shared stream) keeps each neuron's topology independent of
    /// how many neurons precede it, so growing a layer does not reshuffle it.
    pub(super) fn for_neuron(seed: u64, neuron: usize) -> Self {
        Self::new(seed ^ (neuron as u64).wrapping_add(1).wrapping_mul(Self::GAMMA))
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(Self::GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform integer in `[0, bound)`, rejection-sampled so the distribution is
    /// unbiased (a bare `%` would over-weight small values).
    pub(super) fn next_bounded(&mut self, bound: u64) -> u64 {
        debug_assert!(bound > 0, "next_bounded requires a positive bound");
        let zone = (u64::MAX / bound) * bound;
        loop {
            let x = self.next_u64();
            if x < zone {
                return x % bound;
            }
        }
    }

    /// Uniform `f32` in `[0, 1)` with 24 bits of mantissa — the full precision
    /// an `f32` can represent in that interval, and exactly representable, so
    /// the value does not depend on rounding mode.
    pub(super) fn next_unit(&mut self) -> f32 {
        const SCALE: f32 = 1.0 / (1u32 << 24) as f32;
        ((self.next_u64() >> 40) as f32) * SCALE
    }
}
