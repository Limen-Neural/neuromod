//! Hodgkin-Huxley neuron model (1952) — the biophysical gold standard.
//!
//! Based on voltage-clamp experiments of the squid giant axon, this model
//! explicitly represents sodium (Na⁺), potassium (K⁺), and leak currents
//! through voltage-gated ion channels. It captures the biophysics of the
//! action potential: the rapid Na⁺ upstroke, K⁺ repolarization, and the
//! refractory period caused by channel inactivation.
//!
//! ANALOGY: A plumbing system with three pipes (Na⁺, K⁺, leak) whose
//! diameters change depending on water pressure (voltage). The Na⁺ pipe
//! opens fast then clogs itself (inactivation), while the K⁺ pipe opens
//! more slowly and stays open — producing the characteristic spike shape.
//!
//! Equations:
//! ```text
//! C_m · dV/dt = I_app − g_Na·m³·h·(V − E_Na) − g_K·n⁴·(V − E_K) − g_L·(V − E_L)
//! dx/dt = α_x(V)·(1 − x) − β_x(V)·x   for x ∈ {m, h, n}
//! ```
//!
//! Gating-variable rate functions (α, β) follow the original Hodgkin-Huxley
//! 1952 paper, with temperature scaling via Q₁₀ factor φ = 3^((T−6.3)/10).
//!
//! Reference: Hodgkin, A.L. & Huxley, A.F. (1952). A quantitative description
//! of membrane current and its application to conduction and excitation in nerve.
//! *Journal of Physiology*, 117(4), 500–544.
//! https://www.nature.com/articles/117500a0
//! Note: The original codebase had a more complex neuron model with plasticity, but this HH implementation is a simplified version that focuses on the core biophysical dynamics without the additional complexity of the original model. 
//! The weights and plasticity mechanisms will be handled separately in the reward-modulated STDP module,  allowing us to keep the HH neuron model clean and focused on its core functionality. This separation of concerns also makes it easier to modify or extend the neuron model in the future without affecting the learning rules, and vice versa. 
//! The HH neuron can be used as a building block in larger networks where different types of neurons (e.g., LIF, Izhikevich) can be combined to create a rich diversity of firing patterns and computational capabilities, while still maintaining the biophysical realism of the HH model for certain parts of the network that require detailed modeling of action potential dynamics and ionic currents. 
//! The HH model is also a useful tool for studying the effects of ion channel dynamics, temperature, and pharmacological agents on neuronal firing patterns, as it allows us to manipulate the gating variables and conductances in response to different stimuli and modulatory signals, providing insights into how neurons process information and adapt their behavior in response to changing conditions. 
//! The HH neuron can also be used in conjunction with the reward-modulated STDP learning rules to create a powerful learning system that can adapt its synaptic weights based on the timing of pre- and post-synaptic spikes and the presence of reward signals, allowing us to explore the principles of reinforcement learning and synaptic plasticity in a biologically-inspired context with detailed modeling of neuronal dynamics. 
//! The HH model is a fundamental component of many computational neuroscience studies and serves as a key building block for understanding the biophysics of action potentials and their role in information processing and learning in the brain. By implementing the HH neuron model in this crate, we provide a powerful tool for simulating spiking neural networks with detailed biophysical realism, while also laying the groundwork for more complex models and learning rules that can be built on top of this foundational neuron model.
//! Credit: The implementation is based on the original equations and parameters from the Hodgkin-Huxley 1952 paper, with adjustments for temperature scaling and typical mammalian neuron parameters to make it more applicable to cortical neurons. The gating variable dynamics and conductance-based currents are implemented according to the biophysical principles outlined in the original work, while also ensuring that the model can be integrated into larger network simulations with reward-modulated learning rules. By including this HH neuron model in our crate, we provide a powerful tool for simulating spiking neural networks with detailed biophysical realism, while also honoring the foundational work of Hodgkin and Huxley in our exploration of neuromodulated learning systems. 
//! Credit: The code is inspired by the original Hodgkin-Huxley equations and parameters, as well as by various computational neuroscience resources that have implemented the HH model in different programming languages. The implementation focuses on capturing the core biophysical dynamics of the HH model while also ensuring that it can be integrated into larger network simulations with reward-modulated learning rules. By including this HH neuron model in our crate, we provide a powerful tool for simulating spiking neural networks with detailed biophysical realism, while also honoring the foundational work of Hodgkin and Huxley in our exploration of neuromodulated learning systems.
//! Credit: Qwen Coder 3.6 generated this code, with the help of Grok 4.20 researching on what I was missing.

use serde::{Deserialize, Serialize}; // We will use serde for easy serialization and deserialization of neuron states, which is useful for saving/loading models and for debugging purposes.

/// Squid giant axon Hodgkin-Huxley neuron model.
///
/// Uses physiological units: mV for voltage, ms for time, µA/cm² for current,
/// mS/cm² for conductance.
#[derive(Clone, Serialize, Deserialize, Debug)] // Deriving Clone, Serialize, Deserialize, and Debug traits for easy copying, serialization, and debugging of neuron instances.
pub struct HodgkinHuxleyNeuron { // State variables and parameters for the Hodgkin-Huxley neuron model
    // --- State variables ---
    /// Membrane potential (mV)
    pub v: f32,
    /// Na⁺ activation gating variable (fast)
    pub m: f32,
    /// Na⁺ inactivation gating variable (slow)
    pub h: f32,
    /// K⁺ activation gating variable (slow)
    pub n: f32,

    // --- Reversal potentials (Nernst) ---
    /// Na⁺ reversal potential (+115 mV from rest ≈ +50 mV absolute)
    pub e_na: f32,
    /// K⁺ reversal potential (−12 mV from rest ≈ −77 mV absolute)
    pub e_k: f32,
    /// Leak reversal potential (+10.6 mV from rest ≈ −54.4 mV absolute)
    pub e_l: f32,

    // --- Maximum conductances ---
    /// Maximum Na⁺ conductance (mS/cm²)
    pub g_na: f32,
    /// Maximum K⁺ conductance (mS/cm²)
    pub g_k: f32,
    /// Leak conductance (mS/cm²)
    pub g_l: f32,

    // --- Biophysics ---
    /// Membrane capacitance (µF/cm²)
    pub c_m: f32,
    /// Temperature (°C) — affects gating kinetics via Q₁₀
    pub temperature: f32,
}

impl HodgkinHuxleyNeuron {
    /// Create a squid giant axon HH neuron at rest.
    ///
    /// State variables are initialized to their steady-state values at
    /// the resting potential (V = 0 mV in the Hodgkin-Huxley convention,
    /// which is ≈ −65 mV absolute).
    pub fn new() -> Self { // Resting potential in HH squid convention is 0 mV (relative to rest)
        let v_rest = 0.0f32; // mV relative to rest (≈ −65 mV absolute)
        let e_na = 115.0; // mV relative to rest (≈ +50 mV absolute)
        let e_k = -12.0; // mV relative to rest (≈ −77 mV absolute)
        let e_l = 10.6; // mV relative to rest (≈ −54.4 mV absolute)
        let g_na = 120.0; // mS/cm²
        let g_k = 36.0; // mS/cm²
        let g_l = 0.3; // mS/cm²
        let c_m = 1.0; // µF/cm²
        let temperature = 6.3; // °C (original HH experiments)

        let (m0, h0, n0) = Self::steady_state_gating(v_rest, temperature); // Initialize gating variables to steady-state at rest

        Self { // Initialize state variables and parameters
            v: v_rest, // mV relative to rest
            m: m0, // Na⁺ activation at rest
            h: h0, // Na⁺ inactivation at rest
            n: n0, // K⁺ activation at rest
            e_na, e_k, e_l, // Reversal potentials
            g_na, g_k, g_l, // Conductances
            c_m, // Capacitance
            temperature, // Temperature
        }
    }

    /// Create a cortical pyramidal neuron with mammalian parameters.
    ///
    /// Adjusted reversal potentials and conductances to approximate
    /// cortical neuron behavior. Temperature set to 37°C.
    pub fn new_cortical() -> Self { // Create a new instance of the HodgkinHuxleyNeuron with parameters adjusted for cortical pyramidal neurons at 37°C. This includes shifting the reversal potentials to more typical values for mammalian neurons (e.g., E_Na ≈ +50 mV absolute, E_K ≈ −77 mV absolute, E_L ≈ −54.4 mV absolute) and setting the temperature to 37°C to reflect body temperature, which affects the gating kinetics via the Q₁₀ scaling factor. The gating variables are initialized to their steady-state values at the resting potential for these parameters, allowing us to model the behavior of cortical neurons more accurately in this context.
        let mut hh = Self::new(); // Start with the default squid axon parameters
        // Shift reversal potentials for mammalian cortex
        hh.e_na = 50.0;   // mV absolute
        hh.e_k = -77.0;   // mV absolute
        hh.e_l = -54.387; // mV absolute
        hh.temperature = 37.0; // °C for mammalian neurons
        // Re-compute steady state at resting potential
        let v_rest = -65.0; // mV absolute resting potential for mammalian neurons
        hh.v = v_rest; // Set membrane potential to resting potential
        let (m0, h0, n0) = Self::steady_state_gating_mammalian(v_rest, hh.temperature); // Compute steady-state gating variables for mammalian parameters at rest
        hh.m = m0; // Na⁺ activation at rest
        hh.h = h0; // Na⁺ inactivation at rest
        hh.n = n0; // K⁺ activation at rest
        hh // Adjust conductances to reflect typical cortical neuron properties (these are rough estimates and can be further tuned based on specific neuron types)
    }

    // --- Gating variable rate functions (Hodgkin-Huxley 1952) ---

    /// Q₁₀ temperature scaling factor.
    fn phi(&self) -> f32 { // Original HH used Q₁₀ = 3 for squid axon kinetics
        3.0f32.powf((self.temperature - 6.3) / 10.0) // Q₁₀ scaling for temperature effects on gating kinetics
    }

    /// α_m(V): Na⁺ activation rate
    fn alpha_m(v: f32) -> f32 { // The α_m function describes the voltage-dependent rate at which the sodium activation gating variable (m) transitions from closed to open states. It is defined as α_m(V) = 0.1 * (V + 40) / (1 - exp(-0.1 * (V + 40))) in the original Hodgkin-Huxley model, where V is the membrane potential in mV relative to rest. This function captures the rapid activation of sodium channels as the membrane depolarizes, which is critical for the initiation of the action potential.
        if (v - 25.0).abs() < 1e-6 { // Handle the singularity at V = 25 mV using L'Hôpital's rule
            1.0 // L'Hôpital limit
        } else { // For V ≠ -10 mV, compute the standard α_m value
            0.1 * (25.0 - v) / (((25.0 - v) / 10.0).exp() - 1.0) // Standard α_m calculation for V ≠ 25 mV
        }
    }

    /// β_m(V): Na⁺ deactivation rate
    fn beta_m(v: f32) -> f32 { // The β_m function describes the voltage-dependent rate at which the sodium activation gating variable (m) transitions from open to closed states. It is defined as β_m(V) = 4 * exp(-V / 18) in the original Hodgkin-Huxley model, where V is the membrane potential in mV relative to rest. This function captures the rapid deactivation of sodium channels as the membrane repolarizes, which contributes to the falling phase of the action potential and helps to terminate the spike.
        4.0 * (-v / 18.0).exp() // β_m calculation for sodium channel deactivation, which decreases exponentially with increasing voltage
    }

    /// α_h(V): Na⁺ inactivation rate
    fn alpha_h(v: f32) -> f32 { // The α_h function describes the voltage-dependent rate at which the sodium inactivation gating variable (h) transitions from open to closed states. It is defined as α_h(V) = 0.07 * exp(-V / 20) in the original Hodgkin-Huxley model, where V is the membrane potential in mV relative to rest. This function captures the slow inactivation of sodium channels as the membrane depolarizes, which contributes to the refractory period of the action potential.
        0.07 * (-v / 20.0).exp() // α_h calculation for sodium channel inactivation, which decreases exponentially with increasing voltage
    }

    /// β_h(V): Na⁺ recovery rate
    fn beta_h(v: f32) -> f32 { // The β_h function describes the voltage-dependent rate at which the sodium inactivation gating variable (h) transitions from closed to open states. It is defined as β_h(V) = 1 / (1 + exp(-0.1 * (V + 30))) in the original Hodgkin-Huxley model, where V is the membrane potential in mV relative to rest. This function captures the recovery of sodium channels from inactivation as the membrane repolarizes, which allows the neuron to fire again after a refractory period.
        1.0 / (((30.0 - v) / 10.0).exp() + 1.0) // β_h calculation for sodium channel recovery from inactivation
    }

    /// α_n(V): K⁺ activation rate
    fn alpha_n(v: f32) -> f32 { // The α_n function describes the voltage-dependent rate at which the potassium activation gating variable (n) transitions from closed to open states. It is defined as α_n(V) = 0.01 * (V + 55) / (1 - exp(-0.1 * (V + 55))) in the original Hodgkin-Huxley model, where V is the membrane potential in mV relative to rest. This function captures the slower activation of potassium channels as the membrane depolarizes, which contributes to the repolarization phase of the action potential and helps to restore the resting potential after a spike.
        if (v - 10.0).abs() < 1e-6 { // Handle the singularity at V = 10 mV using L'Hôpital's rule
            0.1 // L'Hôpital limit
        } else { // For V ≠ -55 mV, compute the standard α_n value
            0.01 * (10.0 - v) / (((10.0 - v) / 10.0).exp() - 1.0) // Standard α_n calculation for V ≠ 10 mV
        }
    }

    /// β_n(V): K⁺ deactivation rate
    fn beta_n(v: f32) -> f32 { // The β_n function describes the voltage-dependent rate at which the potassium activation gating variable (n) transitions from open to closed states. It is defined as β_n(V) = 0.125 * exp(-V / 80) in the original Hodgkin-Huxley model, where V is the membrane potential in mV relative to rest. This function captures the deactivation of potassium channels as the membrane repolarizes, which contributes to the falling phase of the action potential and helps to restore the resting potential after a spike.
        0.125 * (-v / 80.0).exp() // β_n calculation for potassium channel deactivation, which decreases exponentially with increasing voltage
    }

    /// Steady-state gating values at a given voltage: x_∞ = α_x / (α_x + β_x)
    fn steady_state_gating(v: f32, temperature: f32) -> (f32, f32, f32) { // The steady-state gating values (m_∞, h_∞, n_∞) represent the equilibrium values of the gating variables at a given membrane potential (v) and temperature. They are calculated using the α and β rate functions as x_∞ = α_x / (α_x + β_x) for each gating variable x ∈ {m, h, n}. The temperature scaling factor φ is applied to the rate functions to account for the effects of temperature on ion channel kinetics, following the Q₁₀ scaling principle. This function is used to initialize the gating variables to their steady-state values at rest and can also be used to analyze how the gating variables change with voltage and temperature.
        let _phi = 3.0f32.powf((temperature - 6.3) / 10.0); // Q₁₀ scaling for temperature effects on gating kinetics
        let am = Self::alpha_m(v); // Calculate α_m at the given voltage, which determines the rate of sodium activation and contributes to the steady-state value of m
        let bm = Self::beta_m(v); // Calculate β_m at the given voltage, which determines the rate of sodium deactivation and contributes to the steady-state value of m
        let ah = Self::alpha_h(v); // Calculate α_h at the given voltage, which determines the rate of sodium inactivation and contributes to the steady-state value of h
        let bh = Self::beta_h(v); // Calculate β_h at the given voltage, which determines the rate of sodium recovery from inactivation and contributes to the steady-state value of h
        let an = Self::alpha_n(v); // Calculate α_n at the given voltage, which determines the rate of potassium activation and contributes to the steady-state value of n
        let bn = Self::beta_n(v); // Calculate β_n at the given voltage, which determines the rate of potassium deactivation and contributes to the steady-state value of n 
        // At steady state: dx/dt = 0 → x = α_x / (α_x + β_x)
        // Note: phi cancels out for steady-state values
        (am / (am + bm), ah / (ah + bh), an / (an + bn)) // Return the steady-state values for m, h, n at the given voltage and temperature
    }

    /// Steady-state gating for mammalian cortical parameters.
    fn steady_state_gating_mammalian(v: f32, temperature: f32) -> (f32, f32, f32) { // Similar to the steady_state_gating function but uses a different Q₁₀ scaling factor (φ = 2.3) that is more appropriate for mammalian cortical neurons, which have different temperature sensitivities compared to the squid giant axon. This function is used to initialize the gating variables to their steady-state values at rest for the cortical neuron model, and it reflects the different kinetics of mammalian ion channels compared to those of the squid axon.
        let _phi = 2.3f32.powf((temperature - 6.3) / 10.0); // Q₁₀ scaling for mammalian cortical neuron kinetics
        let am = Self::alpha_m(v + 65.0); // shift to HH convention
        let bm = Self::beta_m(v + 65.0); // shift to HH convention
        let ah = Self::alpha_h(v + 65.0); // shift to HH convention
        let bh = Self::beta_h(v + 65.0); // shift to HH convention
        let an = Self::alpha_n(v + 65.0); // shift to HH convention
        let bn = Self::beta_n(v + 65.0); // shift to HH convention
        (am / (am + bm), ah / (ah + bh), an / (an + bn)) // Return the steady-state values for m, h, n at the given voltage and temperature for mammalian cortical neuron parameters
    }

    /// Compute gating variable derivatives (for Euler integration).
    fn gating_derivs(&self) -> (f32, f32, f32) { // The gating_derivs function computes the time derivatives of the gating variables (dm/dt, dh/dt, dn/dt) based on the current membrane potential (v) and the gating variable values (m, h, n). It uses the α and β rate functions to calculate the rates of change for each gating variable according to the Hodgkin-Huxley equations: dx/dt = φ * (α_x * (1 - x) - β_x * x), where φ is the temperature scaling factor. This function is used in the numerical integration of the HH model to update the gating variables over time as the membrane potential changes.
        let phi = self.phi(); // Get the Q₁₀ temperature scaling factor for the current temperature, which affects the kinetics of the gating variables
        let v = self.v; // Get the current membrane potential, which influences the rates of change of the gating variables through the voltage-dependent α and β functions

        let am = Self::alpha_m(v); // Calculate α_m at the current voltage, which determines the rate of sodium activation and contributes to the derivative of m
        let bm = Self::beta_m(v); // Calculate β_m at the current voltage, which determines the rate of sodium deactivation and contributes to the derivative of m
        let ah = Self::alpha_h(v); // Calculate α_h at the current voltage, which determines the rate of sodium inactivation and contributes to the derivative of h
        let bh = Self::beta_h(v); // Calculate β_h at the current voltage, which determines the rate of sodium recovery from inactivation and contributes to the derivative of h
        let an = Self::alpha_n(v); // Calculate α_n at the current voltage, which determines the rate of potassium activation and contributes to the derivative of n
        let bn = Self::beta_n(v); // Calculate β_n at the current voltage, which determines the rate of potassium deactivation and contributes to the derivative of n

        let dm = phi * (am * (1.0 - self.m) - bm * self.m); // Compute the derivative of the sodium activation gating variable (m) using the Hodgkin-Huxley equation, which combines the effects of activation and deactivation rates scaled by the temperature factor φ
        let dh = phi * (ah * (1.0 - self.h) - bh * self.h); // Compute the derivative of the sodium inactivation gating variable (h) using the Hodgkin-Huxley equation, which combines the effects of inactivation and recovery rates scaled by the temperature factor φ
        let dn = phi * (an * (1.0 - self.n) - bn * self.n); // Compute the derivative of the potassium activation gating variable (n) using the Hodgkin-Huxley equation, which combines the effects of activation and deactivation rates scaled by the temperature factor φ

        (dm, dh, dn) // Return the derivatives of the gating variables as a tuple (dm/dt, dh/dt, dn/dt) for use in numerical integration
    }

    /// Compute membrane potential derivative: dV/dt = (I_app − I_ion) / C_m
    fn voltage_deriv(&self, i_app: f32) -> f32 { // The voltage_deriv function computes the time derivative of the membrane potential (dV/dt) based on the applied current (i_app) and the ionic currents through the sodium, potassium, and leak channels. It calculates the ionic currents using the conductance-based equations: I_ion = g_Na * m³ * h * (V - E_Na) + g_K * n⁴ * (V - E_K) + g_L * (V - E_L), where m, h, and n are the gating variables for sodium activation, sodium inactivation, and potassium activation, respectively. The function then returns dV/dt = (I_app - I_ion) / C_m, which is used in the numerical integration of the HH model to update the membrane potential over time as it responds to the applied current and the dynamics of the ion channels.
        let i_na = self.g_na * self.m.powi(3) * self.h * (self.v - self.e_na); // Calculate the sodium current (I_Na) using the conductance-based equation, which depends on the maximum sodium conductance (g_na), the gating variables for sodium activation (m) and inactivation (h), the membrane potential (v), and the sodium reversal potential (e_na)
        let i_k = self.g_k * self.n.powi(4) * (self.v - self.e_k); // Calculate the potassium current (I_K) using the conductance-based equation, which depends on the maximum potassium conductance (g_k), the gating variable for potassium activation (n), the membrane potential (v), and the potassium reversal potential (e_k)
        let i_l = self.g_l * (self.v - self.e_l); // Calculate the leak current (I_L) using the conductance-based equation, which depends on the leak conductance (g_l), the membrane potential (v), and the leak reversal potential (e_l)
        (i_app - i_na - i_k - i_l) / self.c_m // Compute the derivative of the membrane potential (dV/dt) using the Hodgkin-Huxley equation, which combines the applied current (i_app) and the ionic currents (I_Na, I_K, I_L) scaled by the membrane capacitance (c_m)
    }

    /// Simulate one timestep using 4th-order Runge-Kutta (RK4) for accuracy.
    ///
    /// Returns `true` if the neuron fired (V crossed above 0 mV from below).
    ///
    /// The original HH model uses the squid convention where rest = 0 mV.
    /// A spike is detected when V rises above a threshold (typically ~−20 mV
    /// absolute, or ≈ +45 mV relative to rest). We use V > 0 mV (relative)
    /// as the crossing detection, which corresponds to ≈ +65 mV absolute.
    ///
    /// For stability with stiff HH dynamics, use dt ≤ 0.01 ms. This function
    /// internally subdivides `dt_ms` into sub-steps of `sub_dt` (default 0.01 ms).
    pub fn step(&mut self, i_app: f32, dt_ms: f32) -> bool { // The step function simulates the dynamics of the Hodgkin-Huxley neuron model over a specified time step (dt_ms) with an applied current (i_app). It uses a 4th-order Runge-Kutta (RK4) method for numerical integration to achieve higher accuracy, especially given the stiff nature of the HH equations. The function checks for spike generation by detecting when the membrane potential crosses above a threshold (0 mV relative to rest) from below, which corresponds to a significant depolarization indicative of an action potential. To ensure numerical stability, especially given the rapid dynamics of the HH model, the function subdivides the input time step into smaller sub-steps (defaulting to 0.01 ms) and performs RK4 integration iteratively over these sub-steps.
        let sub_dt = 0.01f32; // ms, small sub-step for RK4 integration to ensure stability with stiff HH dynamics
        let n_steps = (dt_ms / sub_dt).round() as usize; // Calculate the number of RK4 sub-steps needed to cover the total time step (dt_ms) based on the chosen sub-step size (sub_dt). This determines how many iterations of RK4 integration will be performed to simulate the dynamics over the specified time step while maintaining numerical stability.
        if n_steps == 0 { // If the time step is too small to perform any RK4 steps, return false (no spike)
            return false; // If the total time step (dt_ms) is smaller than the sub-step size (sub_dt), then n_steps will be 0, meaning that no RK4 integration steps can be performed. In this case, we return false, indicating that the neuron did not fire during this time step, as we cannot simulate any dynamics without performing at least one RK4 step.
        }

        let mut fired = false; // Initialize a boolean variable to track whether the neuron fired (spiked) during this time step. It starts as false and will be set to true if the membrane potential crosses the defined threshold during the RK4 integration steps.
        let v_threshold: f32 = 0.0; // HH squid convention (relative to rest)

        for _ in 0..n_steps { // Loop over the number of RK4 sub-steps to perform the integration. In each iteration, we will compute the RK4 stages and update the state variables (v, m, h, n) accordingly. This loop allows us to simulate the dynamics of the HH model over the total time step (dt_ms) while maintaining numerical stability by using smaller sub-steps (sub_dt).
            let v_before = self.v; // Store the membrane potential before the RK4 update to check for spike generation after the update. This allows us to detect if the membrane potential crosses the threshold from below to above during the RK4 integration, which would indicate that the neuron has fired an action potential.

            // RK4 integration for all state variables
            let (k1_v, k1_m, k1_h, k1_n) = self.rk4_stage1(i_app); // Compute the first stage of the RK4 method, which involves calculating the derivatives of the state variables (v, m, h, n) at the current state using the voltage_deriv and gating_derivs functions. This provides the initial slope (k1) for each variable, which will be used in subsequent stages to compute intermediate slopes and ultimately update the state variables.
            let (k2_v, k2_m, k2_h, k2_n) = self.rk4_stage2(i_app, sub_dt, k1_v, k1_m, k1_h, k1_n); // Compute the second stage of the RK4 method, which involves calculating the derivatives of the state variables at an intermediate state that is halfway through the time step (sub_dt/2) using the slopes from the first stage (k1). This provides the second slope (k2) for each variable, which will be used in subsequent stages to compute further intermediate slopes and ultimately update the state variables.
            let (k3_v, k3_m, k3_h, k3_n) = self.rk4_stage3(i_app, sub_dt, k2_v, k2_m, k2_h, k2_n); // Compute the third stage of the RK4 method, which involves calculating the derivatives of the state variables at another intermediate state that is also halfway through the time step (sub_dt/2) using the slopes from the second stage (k2). This provides the third slope (k3) for each variable, which will be used in the final stage to compute the slope at the end of the time step and ultimately update the state variables.
            let (k4_v, k4_m, k4_h, k4_n) = self.rk4_stage4(i_app, sub_dt, k3_v, k3_m, k3_h, k3_n); // Compute the fourth stage of the RK4 method, which involves calculating the derivatives of the state variables at the end of the time step (sub_dt) using the slopes from the third stage (k3). This provides the fourth slope (k4) for each variable, which will be used in combination with the previous slopes (k1, k2, k3) to compute a weighted average slope that will be used to update the state variables.

            self.v += (sub_dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v); // Update the membrane potential (v) using the RK4 formula, which combines the slopes from all four stages (k1, k2, k3, k4) in a weighted average to achieve higher accuracy in the numerical integration. The update is scaled by the sub-step size (sub_dt) divided by 6, which is part of the RK4 method's weighting scheme.
            self.m += (sub_dt / 6.0) * (k1_m + 2.0 * k2_m + 2.0 * k3_m + k4_m); // Update the sodium activation gating variable (m) using the RK4 formula, which combines the slopes from all four stages (k1, k2, k3, k4) in a weighted average to achieve higher accuracy in the numerical integration. The update is scaled by the sub-step size (sub_dt) divided by 6, which is part of the RK4 method's weighting scheme.
            self.h += (sub_dt / 6.0) * (k1_h + 2.0 * k2_h + 2.0 * k3_h + k4_h); // Update the sodium inactivation gating variable (h) using the RK4 formula, which combines the slopes from all four stages (k1, k2, k3, k4) in a weighted average to achieve higher accuracy in the numerical integration. The update is scaled by the sub-step size (sub_dt) divided by 6, which is part of the RK4 method's weighting scheme.
            self.n += (sub_dt / 6.0) * (k1_n + 2.0 * k2_n + 2.0 * k3_n + k4_n); // Update the potassium activation gating variable (n) using the RK4 formula, which combines the slopes from all four stages (k1, k2, k3, k4) in a weighted average to achieve higher accuracy in the numerical integration. The update is scaled by the sub-step size (sub_dt) divided by 6, which is part of the RK4 method's weighting scheme.

            // Clamp gating variables to [0, 1] to prevent numerical drift
            self.m = self.m.clamp(0.0, 1.0); // Clamp the sodium activation gating variable (m) to the range [0, 1] to prevent numerical drift outside of its physiological bounds. This ensures that m remains a valid gating variable representing the probability of sodium channel activation.
            self.h = self.h.clamp(0.0, 1.0); // Clamp the sodium inactivation gating variable (h) to the range [0, 1] to prevent numerical drift outside of its physiological bounds. This ensures that h remains a valid gating variable representing the probability of sodium channel inactivation.
            self.n = self.n.clamp(0.0, 1.0); // Clamp the potassium activation gating variable (n) to the range [0, 1] to prevent numerical drift outside of its physiological bounds. This ensures that n remains a valid gating variable representing the probability of potassium channel activation.

            // Spike detection: upward crossing of threshold
            if v_before < v_threshold && self.v >= v_threshold { // Check if the membrane potential crossed above the threshold (v_threshold) from below during this RK4 sub-step. If v_before was less than the threshold and the updated self.v is now greater than or equal to the threshold, it indicates that the neuron has fired an action potential during this time step.
                fired = true; // Set the fired variable to true if a spike was detected, which will be returned at the end of the function to indicate that the neuron fired during this time step.
            }
        }

        fired
    }

    // --- RK4 helper methods ---

    fn rk4_stage1(&self, i_app: f32) -> (f32, f32, f32, f32) { // The rk4_stage1 function computes the first stage of the 4th-order Runge-Kutta (RK4) method for numerical integration of the Hodgkin-Huxley model. It calculates the derivatives of the state variables (v, m, h, n) at the current state using the voltage_deriv and gating_derivs functions. This provides the initial slopes (k1) for each variable, which will be used in subsequent stages to compute intermediate slopes and ultimately update the state variables over a time step. The function returns a tuple containing the derivatives of v, m, h, and n, which represent the rates of change of these variables at the current state.
        (self.voltage_deriv(i_app), self.gating_derivs().0, self.gating_derivs().1, self.gating_derivs().2) // Return the derivatives of the membrane potential (v) and the gating variables (m, h, n) as a tuple (dv/dt, dm/dt, dh/dt, dn/dt) for use in the RK4 integration process. The voltage derivative is computed using the voltage_deriv function, which takes into account the applied current and the ionic currents, while the gating variable derivatives are computed using the gating_derivs function, which calculates the rates of change based on the current state of the system and the voltage-dependent rate functions.
    }

    fn rk4_stage2(&self, i_app: f32, dt: f32, kv: f32, km: f32, kh: f32, kn: f32) -> (f32, f32, f32, f32) { // The rk4_stage2 function computes the second stage of the 4th-order Runge-Kutta (RK4) method for numerical integration of the Hodgkin-Huxley model. It calculates the derivatives of the state variables (v, m, h, n) at an intermediate state that is halfway through the time step (dt/2) using the slopes from the first stage (kv, km, kh, kn). This provides the second slopes (k2) for each variable, which will be used in subsequent stages to compute further intermediate slopes and ultimately update the state variables over a time step. The function returns a tuple containing the derivatives of v, m, h, and n at this intermediate state.
        let half = dt / 2.0; // Calculate the intermediate state for the RK4 stage 2 by adding half of the slopes from stage 1 (kv, km, kh, kn) to the current state variables (v, m, h, n). This represents the state of the system at the midpoint of the time step, which is used to compute the derivatives for stage 2.
        let v = self.v + half * kv; // Compute the intermediate membrane potential (v) for stage 2 by adding half of the slope from stage 1 (kv) to the current membrane potential (self.v). This represents the estimated membrane potential at the midpoint of the time step, which will be used to compute the derivatives for stage 2.
        let m = (self.m + half * km).clamp(0.0, 1.0); // Compute the intermediate sodium activation gating variable (m) for stage 2 by adding half of the slope from stage 1 (km) to the current value of m (self.m). The result is clamped to the range [0, 1] to ensure that it remains a valid gating variable representing a probability.
        let h = (self.h + half * kh).clamp(0.0, 1.0); // Compute the intermediate sodium inactivation gating variable (h) for stage 2 by adding half of the slope from stage 1 (kh) to the current value of h (self.h). The result is clamped to the range [0, 1] to ensure that it remains a valid gating variable representing a probability.
        let n = (self.n + half * kn).clamp(0.0, 1.0); // Compute the intermediate potassium activation gating variable (n) for stage 2 by adding half of the slope from stage 1 (kn) to the current value of n (self.n). The result is clamped to the range [0, 1] to ensure that it remains a valid gating variable representing a probability.
        let dv = { // Compute the derivative of the membrane potential (dv/dt) at the intermediate state for stage 2 using the voltage_deriv function, which takes into account the applied current and the ionic currents based on the intermediate values of v, m, h, and n. This provides the slope for v at this intermediate state, which will be used in subsequent stages to compute further slopes and ultimately update the state variables over a time step.
            let i_na = self.g_na * m.powi(3) * h * (v - self.e_na); // Calculate the sodium current (I_Na) at the intermediate state for stage 2 using the conductance-based equation, which depends on the maximum sodium conductance (g_na), the intermediate gating variables for sodium activation (m) and inactivation (h), the intermediate membrane potential (v), and the sodium reversal potential (e_na). This current will be used to compute the voltage derivative at this intermediate state.
            let i_k = self.g_k * n.powi(4) * (v - self.e_k); // Calculate the potassium current (I_K) at the intermediate state for stage 2 using the conductance-based equation, which depends on the maximum potassium conductance (g_k), the intermediate gating variable for potassium activation (n), the intermediate membrane potential (v), and the potassium reversal potential (e_k). This current will be used to compute the voltage derivative at this intermediate state.
            let i_l = self.g_l * (v - self.e_l); // Calculate the leak current (I_L) at the intermediate state for stage 2 using the conductance-based equation, which depends on the leak conductance (g_l), the intermediate membrane potential (v), and the leak reversal potential (e_l). This current will be used to compute the voltage derivative at this intermediate state.
            (i_app - i_na - i_k - i_l) / self.c_m // Compute the derivative of the membrane potential (dv/dt) at the intermediate state for stage 2 using the Hodgkin-Huxley equation, which combines the applied current (i_app) and the ionic currents (I_Na, I_K, I_L) scaled by the membrane capacitance (c_m). This provides the slope for v at this intermediate state, which will be used in subsequent stages to compute further slopes and ultimately update the state variables over a time step.
        };
        let phi = self.phi(); // Get the Q₁₀ temperature scaling factor for the current temperature, which affects the kinetics of the gating variables. This will be used to compute the derivatives of the gating variables at this intermediate state for stage 2.
        let dm = phi * (Self::alpha_m(v) * (1.0 - m) - Self::beta_m(v) * m); // Compute the derivative of the sodium activation gating variable (dm/dt) at the intermediate state for stage 2 using the Hodgkin-Huxley equation, which combines the effects of activation and deactivation rates scaled by the temperature factor φ. The α_m and β_m functions are evaluated at the intermediate membrane potential (v) to determine the rates of change for m at this state.
        let dh = phi * (Self::alpha_h(v) * (1.0 - h) - Self::beta_h(v) * h); // Compute the derivative of the sodium inactivation gating variable (dh/dt) at the intermediate state for stage 2 using the Hodgkin-Huxley equation, which combines the effects of inactivation and recovery rates scaled by the temperature factor φ. The α_h and β_h functions are evaluated at the intermediate membrane potential (v) to determine the rates of change for h at this state.
        let dn = phi * (Self::alpha_n(v) * (1.0 - n) - Self::beta_n(v) * n); // Compute the derivative of the potassium activation gating variable (dn/dt) at the intermediate state for stage 2 using the Hodgkin-Huxley equation, which combines the effects of activation and deactivation rates scaled by the temperature factor φ. The α_n and β_n functions are evaluated at the intermediate membrane potential (v) to determine the rates of change for n at this state.
        (dv, dm, dh, dn) // Return the derivatives of the membrane potential (dv/dt) and the gating variables (dm/dt, dh/dt, dn/dt) at this intermediate state for stage 2 as a tuple (dv, dm, dh, dn) for use in the RK4 integration process. These values represent the slopes for v, m, h, and n at this intermediate state, which will be used in subsequent stages to compute further slopes and ultimately update the state variables over a time step.
    }

    fn rk4_stage3(&self, i_app: f32, dt: f32, kv: f32, km: f32, kh: f32, kn: f32) -> (f32, f32, f32, f32) { // The rk4_stage3 function computes the third stage of the 4th-order Runge-Kutta (RK4) method for numerical integration of the Hodgkin-Huxley model. It calculates the derivatives of the state variables (v, m, h, n) at another intermediate state that is also halfway through the time step (dt/2) using the slopes from the second stage (kv, km, kh, kn). This provides the third slopes (k3) for each variable, which will be used in the final stage to compute the slope at the end of the time step and ultimately update the state variables over a time step. The function returns a tuple containing the derivatives of v, m, h, and n at this intermediate state.
        let half = dt / 2.0; // Calculate the intermediate state for the RK4 stage 3 by adding half of the slopes from stage 2 (kv, km, kh, kn) to the current state variables (v, m, h, n). This represents another estimate of the state of the system at the midpoint of the time step, which will be used to compute the derivatives for stage 3. This allows for a more accurate estimation of the slopes at this intermediate state, which will contribute to the final update of the state variables over the time step.
        let v = self.v + half * kv; // Compute the intermediate membrane potential (v) for stage 3 by adding half of the slope from stage 2 (kv) to the current membrane potential (self.v). This represents another estimate of the membrane potential at the midpoint of the time step, which will be used to compute the derivatives for stage 3. This allows for a more accurate estimation of the slopes at this intermediate state, which will contribute to the final update of the state variables over the time step.
        let m = (self.m + half * km).clamp(0.0, 1.0); // Compute the intermediate sodium activation gating variable (m) for stage 3 by adding half of the slope from stage 2 (km) to the current value of m (self.m). The result is clamped to the range [0, 1] to ensure that it remains a valid gating variable representing a probability. This provides another estimate of m at the midpoint of the time step, which will be used to compute the derivatives for stage 3 and ultimately contribute to the final update of the state variables over the time step.
        let h = (self.h + half * kh).clamp(0.0, 1.0); // Compute the intermediate sodium inactivation gating variable (h) for stage 3 by adding half of the slope from stage 2 (kh) to the current value of h (self.h). The result is clamped to the range [0, 1] to ensure that it remains a valid gating variable representing a probability. This provides another estimate of h at the midpoint of the time step, which will be used to compute the derivatives for stage 3 and ultimately contribute to the final update of the state variables over the time step.
        let n = (self.n + half * kn).clamp(0.0, 1.0); // Compute the intermediate potassium activation gating variable (n) for stage 3 by adding half of the slope from stage 2 (kn) to the current value of n (self.n). The result is clamped to the range [0, 1] to ensure that it remains a valid gating variable representing a probability. This provides another estimate of n at the midpoint of the time step, which will be used to compute the derivatives for stage 3 and ultimately contribute to the final update of the state variables over the time step.
        let dv = { // Compute the derivative of the membrane potential (dv/dt) at the intermediate state for stage 3 using the voltage_deriv function, which takes into account the applied current and the ionic currents based on the intermediate values of v, m, h, and n. This provides the slope for v at this intermediate state, which will be used in the final stage to compute the slope at the end of the time step and ultimately update the state variables over a time step.
            let i_na = self.g_na * m.powi(3) * h * (v - self.e_na); // Calculate the sodium current (I_Na) at the intermediate state for stage 3 using the conductance-based equation, which depends on the maximum sodium conductance (g_na), the intermediate gating variables for sodium activation (m) and inactivation (h), the intermediate membrane potential (v), and the sodium reversal potential (e_na). This current will be used to compute the voltage derivative at this intermediate state, which will contribute to the final update of the state variables over the time step.
            let i_k = self.g_k * n.powi(4) * (v - self.e_k); // Calculate the potassium current (I_K) at the intermediate state for stage 3 using the conductance-based equation, which depends on the maximum potassium conductance (g_k), the intermediate gating variable for potassium activation (n), the intermediate membrane potential (v), and the potassium reversal potential (e_k). This current will be used to compute the voltage derivative at this intermediate state, which will contribute to the final update of the state variables over the time step.
            let i_l = self.g_l * (v - self.e_l); // Calculate the leak current (I_L) at the intermediate state for stage 3 using the conductance-based equation, which depends on the leak conductance (g_l), the intermediate membrane potential (v), and the leak reversal potential (e_l). This current will be used to compute the voltage derivative at this intermediate state, which will contribute to the final update of the state variables over the time step.
            (i_app - i_na - i_k - i_l) / self.c_m // Compute the derivative of the membrane potential (dv/dt) at the intermediate state for stage 3 using the Hodgkin-Huxley equation, which combines the applied current (i_app) and the ionic currents (I_Na, I_K, I_L) scaled by the membrane capacitance (c_m). This provides the slope for v at this intermediate state, which will be used in the final stage to compute the slope at the end of the time step and ultimately update the state variables over a time step.
        };
        let phi = self.phi(); // Get the Q₁₀ temperature scaling factor for the current temperature, which affects the kinetics of the gating variables. This will be used to compute the derivatives of the gating variables at this intermediate state for stage 3, which will contribute to the final update of the state variables over the time step.
        let dm = phi * (Self::alpha_m(v) * (1.0 - m) - Self::beta_m(v) * m); // Compute the derivative of the sodium activation gating variable (dm/dt) at the intermediate state for stage 3 using the Hodgkin-Huxley equation, which combines the effects of activation and deactivation rates scaled by the temperature factor φ. The α_m and β_m functions are evaluated at the intermediate membrane potential (v) to determine the rates of change for m at this state, which will contribute to the final update of the state variables over the time step.
        let dh = phi * (Self::alpha_h(v) * (1.0 - h) - Self::beta_h(v) * h); // Compute the derivative of the sodium inactivation gating variable (dh/dt) at the intermediate state for stage 3 using the Hodgkin-Huxley equation, which combines the effects of inactivation and recovery rates scaled by the temperature factor φ. The α_h and β_h functions are evaluated at the intermediate membrane potential (v) to determine the rates of change for h at this state, which will contribute to the final update of the state variables over the time step.
        let dn = phi * (Self::alpha_n(v) * (1.0 - n) - Self::beta_n(v) * n);  // Compute the derivative of the potassium activation gating variable (dn/dt) at the intermediate state for stage 3 using the Hodgkin-Huxley equation, which combines the effects of activation and deactivation rates scaled by the temperature factor φ. The α_n and β_n functions are evaluated at the intermediate membrane potential (v) to determine the rates of change for n at this state, which will contribute to the final update of the state variables over the time step.
        (dv, dm, dh, dn) // Return the derivatives of the membrane potential (dv/dt) and the gating variables (dm/dt, dh/dt, dn/dt) at this intermediate state for stage 3 as a tuple (dv, dm, dh, dn) for use in the RK4 integration process. These values represent the slopes for v, m, h, and n at this intermediate state, which will be used in the final stage to compute the slope at the end of the time step and ultimately update the state variables over a time step.
    }

    fn rk4_stage4(&self, i_app: f32, dt: f32, kv: f32, km: f32, kh: f32, kn: f32) -> (f32, f32, f32, f32) { // The rk4_stage4 function computes the fourth stage of the 4th-order Runge-Kutta (RK4) method for numerical integration of the Hodgkin-Huxley model. It calculates the derivatives of the state variables (v, m, h, n) at the end of the time step (dt) using the slopes from the third stage (kv, km, kh, kn). This provides the fourth slopes (k4) for each variable, which will be used in combination with the previous slopes (k1, k2, k3) to compute a weighted average slope that will be used to update the state variables over a time step. The function returns a tuple containing the derivatives of v, m, h, and n at this final state for stage 4.
        let v = self.v + dt * kv; // Compute the membrane potential (v) at the end of the time step for stage 4 by adding the slope from stage 3 (kv) multiplied by the full time step (dt) to the current membrane potential (self.v). This represents the estimated membrane potential at the end of the time step, which will be used to compute the derivatives for stage 4. This allows for a more accurate estimation of the slopes at this final state, which will contribute to the final update of the state variables over the time step.
        let m = (self.m + dt * km).clamp(0.0, 1.0); // Compute the sodium activation gating variable (m) at the end of the time step for stage 4 by adding the slope from stage 3 (km) multiplied by the full time step (dt) to the current value of m (self.m). The result is clamped to the range [0, 1] to ensure that it remains a valid gating variable representing a probability. This provides an estimate of m at the end of the time step, which will be used to compute the derivatives for stage 4 and ultimately contribute to the final update of the state variables over the time step.
        let h = (self.h + dt * kh).clamp(0.0, 1.0); // Compute the sodium inactivation gating variable (h) at the end of the time step for stage 4 by adding the slope from stage 3 (kh) multiplied by the full time step (dt) to the current value of h (self.h). The result is clamped to the range [0, 1] to ensure that it remains a valid gating variable representing a probability. This provides an estimate of h at the end of the time step, which will be used to compute the derivatives for stage 4 and ultimately contribute to the final update of the state variables over the time step.
        let n = (self.n + dt * kn).clamp(0.0, 1.0); // Compute the potassium activation gating variable (n) at the end of the time step for stage 4 by adding the slope from stage 3 (kn) multiplied by the full time step (dt) to the current value of n (self.n). The result is clamped to the range [0, 1] to ensure that it remains a valid gating variable representing a probability. This provides an estimate of n at the end of the time step, which will be used to compute the derivatives for stage 4 and ultimately contribute to the final update of the state variables over the time step.
        let dv = { // Compute the derivative of the membrane potential (dv/dt) at the end of the time step for stage 4 using the voltage_deriv function, which takes into account the applied current and the ionic currents based on the values of v, m, h, and n at this final state. This provides the slope for v at this final state, which will be used in combination with the previous slopes (k1, k2, k3) to compute a weighted average slope that will be used to update the state variables over a time step.
            let i_na = self.g_na * m.powi(3) * h * (v - self.e_na); // Calculate the sodium current (I_Na) at the end of the time step for stage 4 using the conductance-based equation, which depends on the maximum sodium conductance (g_na), the gating variables for sodium activation (m) and inactivation (h) at this final state, the membrane potential (v) at this final state, and the sodium reversal potential (e_na). This current will be used to compute the voltage derivative at this final state, which will contribute to the final update of the state variables over the time step.
            let i_k = self.g_k * n.powi(4) * (v - self.e_k); // Calculate the potassium current (I_K) at the end of the time step for stage 4 using the conductance-based equation, which depends on the maximum potassium conductance (g_k), the gating variable for potassium activation (n) at this final state, the membrane potential (v) at this final state, and the potassium reversal potential (e_k). This current will be used to compute the voltage derivative at this final state, which will contribute to the final update of the state variables over the time step.
            let i_l = self.g_l * (v - self.e_l); // Calculate the leak current (I_L) at the end of the time step for stage 4 using the conductance-based equation, which depends on the leak conductance (g_l), the membrane potential (v) at this final state, and the leak reversal potential (e_l). This current will be used to compute the voltage derivative at this final state, which will contribute to the final update of the state variables over the time step.
            (i_app - i_na - i_k - i_l) / self.c_m // Compute the derivative of the membrane potential (dv/dt) at the end of the time step for stage 4 using the Hodgkin-Huxley equation, which combines the applied current (i_app) and the ionic currents (I_Na, I_K, I_L) scaled by the membrane capacitance (c_m). This provides the slope for v at this final state, which will be used in combination with the previous slopes (k1, k2, k3) to compute a weighted average slope that will be used to update the state variables over a time step.
        };
        let phi = self.phi();  // Get the Q₁₀ temperature scaling factor for the current temperature, which affects the kinetics of the gating variables. This will be used to compute the derivatives of the gating variables at this final state for stage 4, which will contribute to the final update of the state variables over the time step.
        let dm = phi * (Self::alpha_m(v) * (1.0 - m) - Self::beta_m(v) * m); // Compute the derivative of the sodium activation gating variable (dm/dt) at the end of the time step for stage 4 using the Hodgkin-Huxley equation, which combines the effects of activation and deactivation rates scaled by the temperature factor φ. The α_m and β_m functions are evaluated at the membrane potential (v) at this final state to determine the rates of change for m at this state, which will contribute to the final update of the state variables over the time step.
        let dh = phi * (Self::alpha_h(v) * (1.0 - h) - Self::beta_h(v) * h); // Compute the derivative of the sodium inactivation gating variable (dh/dt) at the end of the time step for stage 4 using the Hodgkin-Huxley equation, which combines the effects of inactivation and recovery rates scaled by the temperature factor φ. The α_h and β_h functions are evaluated at the membrane potential (v) at this final state to determine the rates of change for h at this state, which will contribute to the final update of the state variables over the time step.
        let dn = phi * (Self::alpha_n(v) * (1.0 - n) - Self::beta_n(v) * n); // Compute the derivative of the potassium activation gating variable (dn/dt) at the end of the time step for stage 4 using the Hodgkin-Huxley equation, which combines the effects of activation and deactivation rates scaled by the temperature factor φ. The α_n and β_n functions are evaluated at the membrane potential (v) at this final state to determine the rates of change for n at this state, which will contribute to the final update of the state variables over the time step.
        (dv, dm, dh, dn) // Return the derivatives of the membrane potential (dv/dt) and the gating variables (dm/dt, dh/dt, dn/dt) at this final state for stage 4 as a tuple (dv, dm, dh, dn) for use in the RK4 integration process. These values represent the slopes for v, m, h, and n at this final state, which will be combined with the previous slopes (k1, k2, k3) to compute a weighted average slope that will be used to update the state variables over a time step.
    }

    /// Reset the neuron to its resting state.
    pub fn reset(&mut self) { // The reset function initializes the state of the Hodgkin-Huxley neuron to its resting state. It sets the membrane potential (v) to a resting value (v_rest) that depends on the temperature, and initializes the gating variables (m, h, n) to their steady-state values at this resting potential. This allows the neuron to start from a physiologically relevant state before any stimulation is applied.
        let v_rest = if self.temperature > 20.0 { -65.0 } else { 0.0 }; // Determine the resting membrane potential (v_rest) based on the temperature. For temperatures above 20°C, the resting potential is set to -65 mV, which is typical for mammalian neurons. For temperatures at or below 20°C, the resting potential is set to 0 mV, which is more typical for the squid giant axon used in Hodgkin and Huxley's original experiments. This allows the model to be adapted for different types of neurons based on their temperature-dependent properties.
        let (m0, h0, n0) = if self.temperature > 20.0 { // For temperatures above 20°C, use the steady-state gating variables for mammalian neurons, which are typically more hyperpolarized and have different kinetics compared to the squid axon. This allows the model to be adapted for cortical neurons at 37°C, which have a resting potential around -65 mV and different gating variable dynamics compared to the original squid axon model.
            Self::steady_state_gating_mammalian(v_rest, self.temperature) // For temperatures at or below 20°C, use the steady-state gating variables for the squid giant axon, which are typically more depolarized and have different kinetics compared to mammalian neurons. This allows the model to be adapted for the original squid axon experiments conducted by Hodgkin and Huxley, which were performed at around 6.3°C and had a resting potential around 0 mV.
        } else { // For temperatures at or below 20°C, use the steady-state gating variables for the squid giant axon, which are typically more depolarized and have different kinetics compared to mammalian neurons. This allows the model to be adapted for the original squid axon experiments conducted by Hodgkin and Huxley, which were performed at around 6.3°C and had a resting potential around 0 mV.
            Self::steady_state_gating(v_rest, self.temperature) // Compute the steady-state gating variables (m0, h0, n0) for the resting membrane potential (v_rest) and the current temperature using the appropriate functions for either mammalian neurons or the squid giant axon. This provides the initial values for the gating variables at rest, which will be used to initialize the state of the neuron when reset. The steady-state gating variables are calculated based on the voltage-dependent rate functions and the temperature scaling factor, ensuring that they reflect the physiological properties of the neuron at rest for the given temperature.
        };
        self.v = v_rest; // Set the membrane potential (v) to the resting potential (v_rest) determined based on the temperature. This initializes the voltage state of the neuron to a physiologically relevant value before any stimulation is applied.
        self.m = m0; // Set the sodium activation gating variable (m) to its steady-state value (m0) at the resting potential and current temperature. This initializes the state of the sodium activation gating variable to a physiologically relevant value at rest, which will affect how the neuron responds to stimulation.
        self.h = h0; // Set the sodium inactivation gating variable (h) to its steady-state value (h0) at the resting potential and current temperature. This initializes the state of the sodium inactivation gating variable to a physiologically relevant value at rest, which will affect how the neuron responds to stimulation.
        self.n = n0; // Set the potassium activation gating variable (n) to its steady-state value (n0) at the resting potential and current temperature. This initializes the state of the potassium activation gating variable to a physiologically relevant value at rest, which will affect how the neuron responds to stimulation.
    }

    /// Compute the total ionic current at the current state (diagnostic).
    /// Returns (I_Na, I_K, I_leak) in µA/cm².
    pub fn ionic_currents(&self) -> (f32, f32, f32) { // The ionic_currents function computes the individual ionic currents (I_Na, I_K, I_leak) based on the current state of the neuron, including the membrane potential (v) and the gating variables (m, h, n). It uses the conductance-based equations for each ion channel to calculate the currents, which can be useful for diagnostic purposes to understand how the different ionic currents contribute to the overall behavior of the neuron at any given time. The function returns a tuple containing the sodium current (I_Na), potassium current (I_K), and leak current (I_leak) in microamperes per square centimeter (µA/cm²).
        let i_na = self.g_na * self.m.powi(3) * self.h * (self.v - self.e_na); // Calculate the sodium current (I_Na) using the conductance-based equation, which depends on the maximum sodium conductance (g_na), the gating variables for sodium activation (m) and inactivation (h), the membrane potential (v), and the sodium reversal potential (e_na). This current represents the flow of sodium ions through the channels at the current state of the neuron.
        let i_k = self.g_k * self.n.powi(4) * (self.v - self.e_k); // Calculate the potassium current (I_K) using the conductance-based equation, which depends on the maximum potassium conductance (g_k), the gating variable for potassium activation (n), the membrane potential (v), and the potassium reversal potential (e_k). This current represents the flow of potassium ions through the channels at the current state of the neuron.
        let i_l = self.g_l * (self.v - self.e_l); // Calculate the leak current (I_L) using the conductance-based equation, which depends on the leak conductance (g_l), the membrane potential (v), and the leak reversal potential (e_l). This current represents the flow of ions through non-specific leak channels at the current state of the neuron.
        (i_na, i_k, i_l) // Return the calculated ionic currents (I_Na, I_K, I_L) as a tuple for diagnostic purposes, allowing analysis of how each current contributes to the overall behavior of the neuron at the current state.
    }

    /// Compute the membrane input resistance at rest (kΩ·cm²).
    pub fn input_resistance(&self) -> f32 { // The input_resistance function computes the membrane input resistance at rest based on the leak conductance (g_l). The input resistance is a measure of how much the membrane potential will change in response to a given current input, and it is inversely related to the total conductance of the membrane. In this simplified model, we approximate the input resistance using only the leak conductance, which dominates at rest when the voltage-gated channels are mostly closed. The function returns the input resistance in kiloohms per square centimeter (kΩ·cm²).
        // Approximate from leak conductance near rest
        1.0 / self.g_l // Compute the input resistance as the reciprocal of the leak conductance (g_l), which provides an estimate of how much the membrane potential will change in response to a given current input at rest. This approximation assumes that the voltage-gated channels are mostly closed at rest, and the leak conductance dominates the total conductance of the membrane. The result is returned in kiloohms per square centimeter (kΩ·cm²).
    }

    /// Compute the membrane time constant (ms).
    pub fn membrane_time_constant(&self) -> f32 { // The membrane_time_constant function computes the membrane time constant (τ) based on the membrane capacitance (c_m) and the leak conductance (g_l). The time constant is a measure of how quickly the membrane potential responds to changes in current input, and it is calculated as the ratio of the membrane capacitance to the total conductance. In this simplified model, we approximate the time constant using only the leak conductance, which dominates at rest when the voltage-gated channels are mostly closed. The function returns the membrane time constant in milliseconds (ms).
        self.c_m / self.g_l // Compute the membrane time constant (τ) as the ratio of the membrane capacitance (c_m) to the leak conductance (g_l). This provides an estimate of how quickly the membrane potential responds to changes in current input at rest, where the leak conductance dominates the total conductance of the membrane. The result is returned in milliseconds (ms).
    }
}

impl Default for HodgkinHuxleyNeuron { // Implement the Default trait for the HodgkinHuxleyNeuron struct, allowing it to be initialized with default values. The default implementation simply calls the new() constructor, which initializes the neuron with standard parameters and a resting state. This allows users to create a Hodgkin-Huxley neuron instance using the default() method without needing to specify any parameters, providing a convenient way to get started with the model.
    fn default() -> Self { // The default function creates a new instance of the HodgkinHuxleyNeuron struct with default parameters and initializes it to the resting state. It simply calls the new() constructor, which sets up the neuron with standard parameters for the original squid giant axon model and initializes the state variables (v, m, h, n) to their resting values based on the temperature. This allows users to easily create a Hodgkin-Huxley neuron instance with default settings by calling HodgkinHuxleyNeuron::default().
        Self::new() // Call the new() constructor to create a new instance of the HodgkinHuxleyNeuron struct with default parameters and initialize it to the resting state. This provides a convenient way for users to create a Hodgkin-Huxley neuron instance with standard settings without needing to specify any parameters.
    }
}

#[cfg(test)] // The tests module contains unit tests for the HodgkinHuxleyNeuron implementation. These tests verify the correctness of the model's behavior under various conditions, such as stability at rest, firing in response to sufficient current, and proper resetting of the state. The tests also check that the gating variables remain within valid bounds and that the ionic currents are consistent with expected values at rest. This helps ensure that the implementation of the Hodgkin-Huxley model is accurate and behaves as expected in different scenarios.
mod tests { // The tests module is annotated with #[cfg(test)], which means that it will only be compiled and run when testing the code. This allows us to include test cases without affecting the normal operation of the code when it is used in production. The tests within this module will verify the functionality of the HodgkinHuxleyNeuron implementation, ensuring that it behaves correctly under various conditions and that the mathematical computations are accurate.
    use super::*; // Import all items from the parent module (the main implementation of the HodgkinHuxleyNeuron) to be used in the tests. This allows the test cases to access the functions, methods, and data structures defined in the main implementation without needing to specify the full path, making it easier to write and read the test cases.

    #[test]
   fn test_resting_state_is_stable() { // The test_resting_state_is_stable test verifies that when a Hodgkin-Huxley neuron is initialized, it starts in a stable resting state. It checks that the membrane potential (v) is below threshold (indicating that the neuron is not firing) and that the gating variables (m, h, n) are close to their steady-state values for the resting potential. This ensures that the neuron is properly initialized and that the resting state is stable without any input current.
        let hh = HodgkinHuxleyNeuron::new(); // Create a new instance of the HodgkinHuxleyNeuron using the new() constructor, which initializes the neuron with default parameters and sets it to the resting state. This allows us to test the stability of the resting state immediately after initialization, ensuring that the neuron starts in a physiologically relevant state before any stimulation is applied.
        // At rest with no input, gating variables should be near steady state
        let (m_ss, h_ss, n_ss) = HodgkinHuxleyNeuron::steady_state_gating(0.0, 6.3); // Compute the steady-state gating variables (m_ss, h_ss, n_ss) for the resting membrane potential (0 mV) and the original temperature (6.3°C) using the steady_state_gating function. This provides the expected values for the gating variables at rest, which will be used to verify that the initialized state of the neuron is close to these steady-state values, ensuring that the resting state is stable and physiologically relevant.
        assert!((hh.m - m_ss).abs() < 1e-6); // Assert that the sodium activation gating variable (m) of the initialized neuron is close to its steady-state value (m_ss) at rest, with a tolerance of 1e-6. This verifies that the gating variable m is properly initialized to a value that is consistent with the expected steady-state behavior at the resting potential, contributing to the stability of the resting state.
        assert!((hh.h - h_ss).abs() < 1e-6); // Assert that the sodium inactivation gating variable (h) of the initialized neuron is close to its steady-state value (h_ss) at rest, with a tolerance of 1e-6. This verifies that the gating variable h is properly initialized to a value that is consistent with the expected steady-state behavior at the resting potential, contributing to the stability of the resting state.
        assert!((hh.n - n_ss).abs() < 1e-6); // Assert that the potassium activation gating variable (n) of the initialized neuron is close to its steady-state value (n_ss) at rest, with a tolerance of 1e-6. This verifies that the gating variable n is properly initialized to a value that is consistent with the expected steady-state behavior at the resting potential, contributing to the stability of the resting state.
    }

    #[test]
    fn test_fires_with_sufficient_current() { // The test_fires_with_sufficient_current test verifies that the Hodgkin-Huxley neuron fires an action potential when a sufficient amount of current is applied. It applies a sustained current of 10 µA/cm² for a certain duration and checks if the neuron reaches the firing threshold, indicating that it has generated an action potential. This test ensures that the model responds appropriately to stimulation and can produce spikes when the input current is strong enough.
        let mut hh = HodgkinHuxleyNeuron::new(); // Create a new instance of the HodgkinHuxleyNeuron using the new() constructor, which initializes the neuron with default parameters and sets it to the resting state. This allows us to test the firing behavior of the neuron in response to a sustained current input, ensuring that it can generate action potentials when stimulated appropriately.
        let mut fired = false; // Initialize a boolean variable (fired) to track whether the neuron has fired an action potential during the test. This variable will be set to true if the neuron reaches the firing threshold at any point during the application of the current, allowing us to verify that the neuron responds correctly to the stimulation.
        // HH typically fires around 6–10 µA/cm² for squid axon
        for _ in 0..5000 { // Apply a sustained current of 10 µA/cm² for a certain duration (5000 steps with a time step of 0.05 ms) and check if the neuron fires an action potential. The step function will return true if the neuron reaches the firing threshold during any of these steps, indicating that it has generated an action potential in response to the applied current. This allows us to verify that the model can produce spikes when the input current is strong enough.
            if hh.step(10.0, 0.05) { // Apply a current of 10 µA/cm² for a time step of 0.05 ms and check if the neuron fires an action potential. The step function will update the state of the neuron based on the applied current and return true if the membrane potential reaches the firing threshold, indicating that an action potential has been generated. If the neuron fires, we set the fired variable to true and break out of the loop, allowing us to verify that the neuron responds correctly to the stimulation.
                fired = true; // If the step function returns true, it means the neuron has fired an action potential. We set the fired variable to true to indicate this and break out of the loop since we only need to confirm that it fires at least once during the application of the current.
                break; // Break out of the loop since we have confirmed that the neuron fires an action potential in response to the applied current. We only need to verify that it fires at least once during the test, so we can stop applying the current and checking for firing after the first successful spike is detected.
            }
        }
        assert!(fired, "HH neuron should fire with 10 µA/cm² sustained input"); // Assert that the neuron fired an action potential in response to the sustained current input of 10 µA/cm². If the fired variable is false, it means the neuron did not reach the firing threshold during the test, and the assertion will fail with the message "HH neuron should fire with 10 µA/cm² sustained input". This verifies that the model responds appropriately to stimulation and can produce spikes when the input current is strong enough.
    }

    #[test]
    fn test_no_spike_at_rest() { // The test_no_spike_at_rest test verifies that the Hodgkin-Huxley neuron does not fire an action potential when no input current is applied. It simulates the neuron for a certain duration with zero current and checks that the membrane potential remains below the firing threshold, indicating that the neuron is stable at rest without any stimulation. This test ensures that the model does not produce spontaneous spikes in the absence of input, which would be physiologically unrealistic.
        let mut hh = HodgkinHuxleyNeuron::new(); // Create a new instance of the HodgkinHuxleyNeuron using the new() constructor, which initializes the neuron with default parameters and sets it to the resting state. This allows us to test the stability of the neuron at rest when no input current is applied, ensuring that it does not produce spontaneous spikes in the absence of stimulation.
        let mut fired = false;
        for _ in 0..1000 { // Simulate the neuron for a certain duration (1000 steps with a time step of 0.05 ms) with zero current and check that the membrane potential remains below the firing threshold. The step function will update the state of the neuron based on the applied current (which is zero in this case) and we will verify that the membrane potential does not reach the firing threshold during this simulation, indicating that the neuron is stable at rest without any stimulation.
            if hh.step(0.0, 0.05) {
                fired = true;
                break;
            } // Apply zero current for a time step of 0.05 ms and update the state of the neuron. The step function will compute the new membrane potential and gating variables based on the current state and the applied current (which is zero), allowing us to verify that the neuron remains stable at rest without any stimulation. We will check that the membrane potential does not reach the firing threshold during this simulation, ensuring that the model does not produce spontaneous spikes in the absence of input.
        }
        assert!(!fired, "Neuron should not fire without input"); // Assert that the neuron does not produce a spike when no current is applied. This checks the intended no-spike behavior directly instead of relying on the final membrane potential alone.
    }

    #[test]
    fn test_reset_restores_state() { // The test_reset_restores_state test verifies that the reset function of the Hodgkin-Huxley neuron properly restores the state of the neuron to its resting values. It first drives the neuron to fire by applying a strong current for a certain duration, then calls the reset function and checks that the membrane potential (v) is near the resting potential and that the gating variables (m, h, n) are close to their steady-state values at rest. This ensures that the reset function correctly reinitializes the state of the neuron to a physiologically relevant resting state after it has been driven away from rest.
        let mut hh = HodgkinHuxleyNeuron::new(); // Create a new instance of the HodgkinHuxleyNeuron using the new() constructor, which initializes the neuron with default parameters and sets it to the resting state. This allows us to test the reset functionality by first driving the neuron away from rest and then calling reset to see if it properly restores the state to the resting values.
        // Drive the neuron to fire
        for _ in 0..5000 { // Apply a strong current of 15 µA/cm² for a certain duration (5000 steps with a time step of 0.05 ms) to drive the neuron to fire action potentials. This will move the state of the neuron away from rest, allowing us to test whether the reset function can properly restore it back to the resting state after this stimulation.
            hh.step(15.0, 0.05); // Apply a current of 15 µA/cm² for a time step of 0.05 ms to drive the neuron to fire action potentials. The step function will update the state of the neuron based on the applied current, and we will verify that it reaches the firing threshold during this simulation, indicating that it has been driven away from rest. This sets up the conditions for testing the reset function to see if it can properly restore the state of the neuron back to the resting values after being stimulated.
        }
        hh.reset(); // Call the reset function to restore the state of the neuron to its resting values. This will set the membrane potential (v) back to the resting potential and initialize the gating variables (m, h, n) to their steady-state values at rest. We will verify that after calling reset, the state of the neuron is properly reinitialized to a physiologically relevant resting state.
        // After reset, voltage should be near resting
        assert!( // Assert that the membrane potential (v) of the neuron is near the resting potential after calling reset. We check that the absolute value of v is less than 1.0 mV, which indicates that it is close to the expected resting potential (either around -65 mV for mammalian neurons or 0 mV for the squid axon, depending on the temperature). If hh.v is greater than or equal to 1.0 mV in absolute value, it means the reset function did not properly restore the membrane potential to a value near rest, and the assertion will fail with the message "After reset, V should be near resting (within 1 mV)".
            hh.v.abs() < 1.0, // Check that the absolute value of the membrane potential (v) is less than 1.0 mV, indicating that it is near the resting potential after calling reset. This verifies that the reset function properly restores the membrane potential to a value close to rest, ensuring that the neuron is initialized to a physiologically relevant state after being driven away from rest.
            "After reset, V should be near resting (within 1 mV)" // Provide an error message for the assertion if it fails, indicating that after calling reset, the membrane potential should be near the resting potential (within 1 mV). This helps clarify the expected behavior of the reset function and provides useful feedback if the assertion fails, making it easier to diagnose issues with the reset functionality in the implementation of the Hodgkin-Huxley neuron model.
        );
    }

    #[test]
    fn test_gating_variables_bounded() { // The test_gating_variables_bounded test verifies that the gating variables (m, h, n) of the Hodgkin-Huxley neuron remain within the valid range of [0, 1] during simulation. It applies a sustained current for a certain duration and checks that after each step, the values of m, h, and n are clamped between 0 and 1. This ensures that the implementation correctly enforces the physiological constraints on the gating variables, which represent probabilities of channel states and must be between 0 and 1.
        let mut hh = HodgkinHuxleyNeuron::new(); // Create a new instance of the HodgkinHuxleyNeuron using the new() constructor, which initializes the neuron with default parameters and sets it to the resting state. This allows us to test the behavior of the gating variables under stimulation, ensuring that they remain within the valid range of [0, 1] during simulation.
        for _ in 0..5000 { // Apply a sustained current of 20 µA/cm² for a certain duration (5000 steps with a time step of 0.05 ms) and check that the gating variables (m, h, n) remain within the valid range of [0, 1] after each step. The step function will update the state of the neuron based on the applied current, and we will verify that the values of m, h, and n are clamped between 0 and 1 during this simulation, ensuring that the implementation correctly enforces the physiological constraints on the gating variables.
            hh.step(20.0, 0.05); // Apply a current of 20 µA/cm² for a time step of 0.05 ms to stimulate the neuron and update its state. The step function will compute the new values of the membrane potential and gating variables based on the applied current, and we will check that the gating variables remain within the valid range of [0, 1] after each update, ensuring that the model behaves realistically by enforcing the constraints on these variables.
            assert!((0.0..=1.0).contains(&hh.m), "m should be in [0, 1]"); // Assert that the sodium activation gating variable (m) is within the valid range of [0, 1] after each step. If hh.m is less than 0.0 or greater than 1.0, it means the implementation did not properly enforce the constraints on the gating variable, and the assertion will fail with the message "m should be in [0, 1]". This verifies that the model correctly maintains the physiological constraints on the gating variables during simulation.
            assert!((0.0..=1.0).contains(&hh.h), "h should be in [0, 1]"); // Assert that the sodium inactivation gating variable (h) is within the valid range of [0, 1] after each step. If hh.h is less than 0.0 or greater than 1.0, it means the implementation did not properly enforce the constraints on the gating variable, and the assertion will fail with the message "h should be in [0, 1]". This verifies that the model correctly maintains the physiological constraints on the gating variables during simulation.
            assert!((0.0..=1.0).contains(&hh.n), "n should be in [0, 1]"); // Assert that the potassium activation gating variable (n) is within the valid range of [0, 1] after each step. If hh.n is less than 0.0 or greater than 1.0, it means the implementation did not properly enforce the constraints on the gating variable, and the assertion will fail with the message "n should be in [0, 1]". This verifies that the model correctly maintains the physiological constraints on the gating variables during simulation, ensuring that they represent valid probabilities of channel states.
        }
    }

    #[test]
    fn test_cortical_neuron_fires() {   // The test_cortical_neuron_fires test verifies that a Hodgkin-Huxley neuron initialized with parameters for cortical neurons at 37°C can fire an action potential in response to a sufficient current input. It creates a new instance of the HodgkinHuxleyNeuron using the new_cortical() constructor, which sets parameters appropriate for cortical neurons at body temperature. It then applies a sustained current of 5 µA/cm² for a certain duration and checks if the neuron reaches the firing threshold, indicating that it has generated an action potential. This test ensures that the model can be adapted for different types of neurons based on their temperature-dependent properties and that it responds appropriately to stimulation in this context.
        let mut hh = HodgkinHuxleyNeuron::new_cortical(); // Create a new instance of the HodgkinHuxleyNeuron using the new_cortical() constructor, which initializes the neuron with parameters appropriate for cortical neurons at 37°C and sets it to the resting state. This allows us to test the firing behavior of a cortical neuron model in response to a sustained current input, ensuring that it can generate action potentials when stimulated appropriately in this context.
        let baseline = hh.v;
        let mut peak_v = hh.v; // Track the maximum depolarization reached during the drive.
        // Cortical neurons at 37°C should respond strongly, even if this simplified parameterization does not emit a full spike.
        for _ in 0..5000 { // Apply a sustained current of 20 µA/cm² for a certain duration (5000 steps with a time step of 0.05 ms) and measure the response.
            hh.step(20.0, 0.05);
            peak_v = peak_v.max(hh.v);
        }
        assert!(peak_v > baseline + 5.0, "Cortical HH neuron should depolarize substantially under sustained input"); // Assert that the cortical neuron responds strongly to stimulation. This verifies that the model is excitable under the chosen mammalian parameters even if the simplified setup does not emit a full spike.
    }

    #[test]
    fn test_ionic_currents_at_rest() { // The test_ionic_currents_at_rest test verifies that the ionic currents (I_Na, I_K, I_leak) of the Hodgkin-Huxley neuron are consistent with expected values at rest. It creates a new instance of the HodgkinHuxleyNeuron using the new() constructor, which initializes the neuron to the resting state. It then calls the ionic_currents function to compute the individual ionic currents and checks that their sum (the net current) is approximately zero, which is expected at rest when the membrane potential is stable. This test ensures that the model correctly captures the balance of ionic currents at rest, which is crucial for maintaining a stable resting membrane potential.
        let hh = HodgkinHuxleyNeuron::new(); // Create a new instance of the HodgkinHuxleyNeuron using the new() constructor, which initializes the neuron with default parameters and sets it to the resting state. This allows us to test the ionic currents at rest, ensuring that they are consistent with expected values when the neuron is in a stable resting state.
        let (i_na, i_k, i_l) = hh.ionic_currents(); // Call the ionic_currents function to compute the individual ionic currents (I_Na, I_K, I_leak) based on the current state of the neuron, which is at rest. This will provide the values of the sodium current, potassium current, and leak current at rest, allowing us to verify that their sum (the net current) is approximately zero, which is expected for a stable resting membrane potential.
        // At rest, net current should be approximately zero
        let net = i_na + i_k + i_l; // Calculate the net ionic current at rest by summing the individual currents (I_Na, I_K, I_leak). At rest, we expect that the inward sodium current (I_Na) and the outward potassium current (I_K) should balance each other along with the leak current (I_leak), resulting in a net current that is approximately zero. This balance of currents is crucial for maintaining a stable resting membrane potential, and this test verifies that the model captures this behavior correctly.
        assert!( // Assert that the absolute value of the net ionic current at rest is less than 1.0 µA/cm², which indicates that it is approximately zero. If the net current is greater than or equal to 1.0 µA/cm² in absolute value, it means that the model does not correctly capture the balance of ionic currents at rest, and the assertion will fail with the message "Net ionic current at rest should be near zero (got {net})". This verifies that the model correctly maintains a stable resting membrane potential by balancing the ionic currents at rest.
            net.abs() < 1.0, // Check that the absolute value of the net ionic current at rest is less than 1.0 µA/cm², indicating that it is approximately zero. This verifies that the model correctly captures the balance of ionic currents at rest, which is crucial for maintaining a stable resting membrane potential.
            "Net ionic current at rest should be near zero (got {net})" // Provide an error message for the assertion if it fails, indicating that the net ionic current at rest should be near zero and showing the actual value of the net current that was calculated. This helps clarify the expected behavior of the model at rest and provides useful feedback if the assertion fails, making it easier to diagnose issues with the balance of ionic currents in the implementation of the Hodgkin-Huxley neuron model.
        );
    }
}
