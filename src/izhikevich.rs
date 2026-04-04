use serde::{Deserialize, Serialize};

/// Biologically plausible neuron model by Eugene M. Izhikevich (2003).
/// Reproduces many firing patterns (regular spiking, bursting, chattering,
/// fast-spiking interneurons) with only two equations and four parameters.
///
/// ANALOGY: A programmable oscillator. Changing `a,b,c,d` swaps the
/// oscillation pattern without changing the underlying circuit.
///
/// Reference: Izhikevich, E.M. (2003). Simple model of spiking neurons.
/// IEEE Transactions on Neural Networks, 14(6), 1569–1572.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct IzhikevichNeuron {
    // State variables
    pub v: f32, // Membrane potential (mV)
    pub u: f32, // Membrane recovery variable
    /// Timestep of the most recent spike (used by Hebbian STDP).
    pub last_spike_time: i64,

    // Parameters that define firing patterns
    pub a: f32, // Timescale of the recovery variable `u`
    pub b: f32, // Sensitivity of `u` to subthreshold fluctuations of `v`
    pub c: f32, // After-spike reset value of `v`
    pub d: f32, // After-spike reset of `u`
}

impl IzhikevichNeuron {
    /// Regular spiking (RS) — typical cortical excitatory neuron.
    /// Fires steadily with spike-frequency adaptation.
    pub fn new_regular_spiking() -> Self {
        let a = 0.02;
        let b = 0.2;
        let c = -65.0;
        Self { v: c, u: b * c, last_spike_time: -1, a, b, c, d: 8.0 }
    }

    /// Intrinsically bursting (IB) — fires a burst then switches to tonic spiking.
    /// Strong initial burst signals a salient event; useful for pattern detection.
    pub fn new_bursting() -> Self {
        let a = 0.02;
        let b = 0.2;
        let c = -55.0;
        Self { v: c, u: b * c, last_spike_time: -1, a, b, c, d: 4.0 }
    }

    /// Fast-spiking (FS) interneuron — high-frequency, no adaptation.
    /// Models inhibitory interneurons; ideal for global inhibition and bear/bull gating.
    pub fn new_fast_spiking() -> Self {
        let a = 0.1;
        let b = 0.2;
        let c = -65.0;
        Self { v: c, u: b * c, last_spike_time: -1, a, b, c, d: 2.0 }
    }

    /// Chattering (CH) neuron — rhythmic high-frequency bursts.
    /// Acts as an oscillator for temporal coding and clock-like spike trains.
    pub fn new_chattering() -> Self {
        let a = 0.02;
        let b = 0.2;
        let c = -50.0;
        Self { v: c, u: b * c, last_spike_time: -1, a, b, c, d: 2.0 }
    }

    /// Low-threshold spiking (LTS) interneuron — fires on weak inputs, strong adaptation.
    /// High sensitivity to low-amplitude signals; good for anomaly detection channels.
    pub fn new_low_threshold() -> Self {
        let a = 0.02;
        let b = 0.25;
        let c = -65.0;
        Self { v: c, u: b * c, last_spike_time: -1, a, b, c, d: 2.0 }
    }

    /// Simulates one timestep (1 ms) of the neuron's dynamics.
    /// Returns `true` if the neuron fired an action potential.
    ///
    /// Uses the half-step Euler method (two sub-steps per ms) for numerical stability,
    /// as recommended in the original Izhikevich (2003) paper.
    pub fn step(&mut self, i: f32) -> bool {
        self.step_with_time(i, 0)
    }

    /// Simulates one timestep (1 ms) with explicit time tracking for STDP.
    pub fn step_with_time(&mut self, i: f32, current_time: i64) -> bool {
        for _ in 0..2 {
            self.v += 0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + i;
        }
        self.u += self.a * (self.b * self.v - self.u);

        if self.v >= 30.0 {
            self.v = self.c;
            self.u += self.d;
            self.last_spike_time = current_time;
            true
        } else {
            false
        }
    }

    /// Reset state variables to resting equilibrium.
    pub fn reset(&mut self) {
        self.v = self.c;
        self.u = self.b * self.c;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regular_spiking_fires_under_sustained_input() {
        let mut n = IzhikevichNeuron::new_regular_spiking();
        let spikes: usize = (0..100).filter(|_| n.step(10.0)).count();
        assert!(spikes > 0, "RS neuron should fire under sustained 10 pA input");
    }

    #[test]
    fn test_fast_spiking_higher_rate_than_regular() {
        let mut rs = IzhikevichNeuron::new_regular_spiking();
        let mut fs = IzhikevichNeuron::new_fast_spiking();
        let rs_spikes: usize = (0..200).filter(|_| rs.step(10.0)).count();
        let fs_spikes: usize = (0..200).filter(|_| fs.step(10.0)).count();
        assert!(fs_spikes >= rs_spikes, "FS should fire at least as fast as RS");
    }

    #[test]
    fn test_reset_restores_resting_state() {
        let mut n = IzhikevichNeuron::new_regular_spiking();
        for _ in 0..50 { n.step(10.0); }
        n.reset();
        assert_eq!(n.v, n.c);
        assert!((n.u - n.b * n.c).abs() < 1e-6);
    }

    #[test]
    fn test_no_spike_at_rest() {
        let mut n = IzhikevichNeuron::new_regular_spiking();
        let spikes: usize = (0..10).filter(|_| n.step(0.0)).count();
        assert_eq!(spikes, 0, "No spike without input");
    }
}
