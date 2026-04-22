# Neuromod — General-Purpose Spiking Neural Network Library

[![Crates.io](https://img.shields.io/crates/v/neuromod.svg)](https://crates.io/crates/neuromod)
[![Docs.rs](https://img.shields.io/badge/docs.rs-neuromod-blue.svg)](https://docs.rs/neuromod)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL_3.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![GitHub](https://img.shields.io/badge/GitHub-Spikenaut/neuromod-black.svg)](https://github.com/Spikenaut/neuromod)

**v0.3.0 — Neuroscience Foundation**

A lightweight, zero-unsafe Rust crate for **general-purpose Spiking Neural Network (SNN)** research and engineering. `neuromod` provides a biologically grounded SNN engine designed for rigorous scientific experimentation, embedded deployment on reconfigurable hardware, and exploratory neuromorphic applications.

Optimised for high-performance execution on **Fedora 43 (Ship of Theseus)** workstations and bare-metal FPGA targets.

---

## Vision

> *The lion does not compromise fidelity for convenience.*

`neuromod` is built on a philosophy of **sovereign, high-fidelity engineering** — every neuron model, every plasticity rule, and every hardware export honours the precision demanded by both neuroscience and systems engineering. There are no shortcuts: the mathematics are correct, the memory footprint is minimal, and the abstractions are thin enough to run on bare metal.

This library is intended as a research-grade foundation for the scientific and engineering community — from cortical dynamics simulation to neuromorphic hardware experiments.

---

## Features

- **Five canonical neuron models**: Lapicque (1907), LIF, Hodgkin-Huxley (1952), FitzHugh-Nagumo (1961), Izhikevich (2003)
- Reward-Modulated STDP (R-STDP) learning
- Classical Hebbian STDP (unmodulated, honouring Hebb 1949)
- Full neuromodulator system (dopamine, cortisol, acetylcholine, tempo)
- Sub-1 µs modulator updates
- ~1.6 KB memory footprint for the full 16-channel network
- `no_std` + Q8.8 fixed-point `.mem` export for FPGA synthesis
- `jlrs` zero-copy interop for Julia-based offline training pipelines

---

## Core Components

### 16-Channel SNN Bank

The engine operates a **16-channel spiking neuron bank** structured around a coincidence detector and a global inhibitory interneuron. Each channel is independently configurable with any supported neuron model, enabling heterogeneous network topologies for multi-modal sensory encoding, population coding, and closed-loop feedback experiments.

```rust
use neuromod::{SpikingNetwork, NeuroModulators};

let mut network = SpikingNetwork::new();
let stimuli = [0.5f32; 16]; // 16-channel sensory input
let modulators = NeuroModulators::default();
let spikes = network.step(&stimuli, &modulators);
```

### Integrated Neuron Models

| Model | Module | Variables | Biological Realism | Primary Use Case |
|---|---|---|---|---|
| **Lapicque** — Classic LIF | [`lapicque`](src/lapicque.rs) | 1 | Foundational | Large-scale SNNs, FPGA baseline |
| **LIF** — Leaky Integrate-and-Fire | [`lif`](src/lif.rs) | 1 | Low–Medium | Hardware-friendly, low-power embedded |
| **Izhikevich** — Biological Firing Patterns | [`izhikevich`](src/izhikevich.rs) | 2 | Medium–High | Cortical pattern matching, burst detection |
| **FitzHugh-Nagumo** — Non-linear Oscillators | [`fitzhugh_nagumo`](src/fitzhugh_nagumo.rs) | 2 | Medium | Phase-plane analysis, oscillatory circuits |
| **Hodgkin-Huxley** — Conductance-Based Dynamics | [`hodgkin_huxley`](src/hodgkin_huxley.rs) | 4 | High | Biophysical simulation, ion-channel studies |

### Plasticity Rules

| Rule | Description |
|---|---|
| **Hebbian Baseline STDP** | Unsupervised Hebbian learning — "neurons that fire together, wire together" (Hebb, 1949) |
| **Reward-Modulated STDP (R-STDP)** | Dopamine-gated synaptic update; links spike timing to behavioural outcome |

---

## Legends of Neuromorphic Computing

This crate explicitly honours the foundational scientists whose work spans over a century of neuroscience:

| Scientist | Year | Module | Contribution |
|---|---|---|---|
| **Louis Lapicque** | 1907 | `lapicque` | Original Integrate-and-Fire model |
| **Donald O. Hebb** | 1949 | `hebbian` | "Neurons that fire together wire together" |
| **Alan Hodgkin & Andrew Huxley** | 1952 | `hodgkin_huxley` | Biophysical gold standard with explicit ion channels |
| **Richard FitzHugh & Jin-ichi Nagumo** | 1961/62 | `fitzhugh_nagumo` | Classic 2D relaxation oscillator |
| **Eugene Izhikevich** | 2003 | `izhikevich` | Programmable spiking neuron; reproduces 20+ cortical firing patterns |

---

## Neuron Model Reference

### Hodgkin-Huxley (1952) — Conductance-Based Dynamics

```rust
use neuromod::HodgkinHuxleyNeuron;

let mut hh = HodgkinHuxleyNeuron::new();               // squid giant axon (6.3 °C)
let mut cortical = HodgkinHuxleyNeuron::new_cortical(); // mammalian cortex (37 °C)
let fired = hh.step(10.0, 0.05);  // 10 µA/cm², dt = 50 µs
```

### Izhikevich (2003) — Biological Firing Patterns

```rust
use neuromod::IzhikevichNeuron;

let mut rs = IzhikevichNeuron::new_regular_spiking();  // RS cortical excitatory
let mut ib = IzhikevichNeuron::new_bursting();
let fired = rs.step(10.0);
```

### FitzHugh-Nagumo (1961) — Non-linear Oscillators

```rust
use neuromod::FitzHughNagumoNeuron;

let mut excitable  = FitzHughNagumoNeuron::new();             // driven excitable regime
let mut oscillator = FitzHughNagumoNeuron::new_oscillatory(); // spontaneous limit cycle
let fired = excitable.step(0.7, 0.5);
```

### Reward-Modulated STDP (R-STDP)

```rust
use neuromod::{SpikingNetwork, NeuroModulators, StdpParams};

let mut network = SpikingNetwork::new();
let params = StdpParams::default();
// dopamine signal in NeuroModulators gates the Hebbian weight update
network.apply_stdp(&params);
```

### Classical Hebbian STDP

```rust
use neuromod::{SpikingNetwork, StdpParams};

let mut network = SpikingNetwork::new();
let params = StdpParams::default();
network.apply_stdp(&params);
```

---

## Neuromodulator System

| Modulator | Role |
|---|---|
| **Dopamine** | Reward signal; gates R-STDP weight updates |
| **Cortisol** | Stress-driven inhibition; scales firing thresholds |
| **Acetylcholine** | Attentional gain; improves signal-to-noise ratio |
| **Tempo** | Global clock scaling for variable time-step integration |

---

## Performance

| Metric | Value |
|---|---|
| Step latency | **< 1 µs** per network step |
| Memory footprint | **~1.6 KB** (full 16-channel network) |
| Throughput | **> 1 M steps/sec** on a single core |
| FPGA export | Q8.8 fixed-point `.mem` (Xilinx-compatible) |
| Build target | Fedora 43 (Ship of Theseus) · `x86_64` & `armv7` cross-compilation |

---

## Comparison to Other Neuromorphic Crates

| Crate | Focus | Neuromodulators | Hardware / FPGA | Biological Models | Plasticity |
|---|---|---|---|---|---|
| **neuromod** | General-purpose SNN research | Dopamine, Cortisol, ACh, Tempo | Q8.8 `.mem` export, `no_std`, Artix-7 ready | 5 canonical models | Hebbian + R-STDP |
| `spiking_neural_networks` | General biophysical simulation | Basic reward only | None | High-fidelity | Limited |
| `omega-snn` | Cognitive SNN architecture | Dopamine + NE + Serotonin + ACh | None | Population coding | Sparse |
| `neuburn` | GPU training (Burn framework) | None | GPU training only | Spiking LSTM | Surrogate gradients |

---

## Roadmap

### v0.4.0 — Hardware Integration
- [ ] **Intel Lava Framework integration** — Python/Rust bridge for execution on Intel Loihi 2 neuromorphic chips via the Lava runtime

### v0.5.0 — Exploratory Extensions
- [ ] **Neural Bridge** *(exploratory)* — a low-latency, closed-loop interface layer for Brain-Computer Interface (BCI) research; targets real-time bidirectional communication between decoded neural spike trains and SNN actuator networks
- [ ] Spike sorting pre-processing pipeline for Utah Array / Neuropixels data ingestion
- [ ] Hardware-in-the-loop (HIL) validation against recorded cortical datasets

### Beyond
- [ ] PyO3 Python bindings for notebook-driven experimentation
- [ ] WASM target for browser-based neural simulation demos
- [ ] Integration with the [SpiNNaker 2](https://www.humanbrainproject.eu/en/silicon-brains/spinnaker-2/) platform

---

## Links

- Crates.io: <https://crates.io/crates/neuromod>
- Docs: <https://docs.rs/neuromod>
- Spikenaut Model Hub: <https://huggingface.co/rmems/Spikenaut-SNN-v2>

---

*`neuromod` — sovereign, high-fidelity neuromorphic engineering. Built to last.*
