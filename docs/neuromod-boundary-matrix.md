# neuromod: Runtime/Deployment Boundary Matrix

> Part of the [LIM-9](https://linear.app/saaq-spiking-adaptive-activity/issue/LIM-9/plan-rust-runtime-and-deployment-repo-boundary-matrix) Rust runtime boundary planning work.
> Tracked by: [neuromod #11](https://github.com/Limen-Neural/neuromod/issues/11), [neuromod #25](https://github.com/Limen-Neural/neuromod/issues/25)

## Purpose

`neuromod` is the **core reusable SNN library** providing biologically grounded neuron models, a topology-neutral dynamic `SpikingNetwork`, generic neuromodulation, and foundational plasticity primitives.

It is a pure computation library: no I/O, no hardware, no application orchestration, and no domain-specific reward or topology logic. It is intended to be depended upon by supervisor runtimes, training loops, encoders, meshes, and export layers across the Limen-Neural org.

## Owns

- Neuron model implementations (Lapicque, LIF, GIF, Izhikevich, FitzHugh-Nagumo, Hodgkin-Huxley)
- `SpikingNetwork` (including `new()`, `with_dimensions()`, `step()`, state accessors, `StepError`)
- `NeuroModulators` (dopamine, serotonin, acetylcholine, norepinephrine), `SignalProfile` (default + hardware_calibrated), `Observation`, `GenericReward` trait, `UnitReward`, `apply_reward`, `apply_neuromodulation`
- Foundational plasticity building blocks:
  - Reward-modulated STDP (`rm_stdp`: `EligibilityTrace`, `RmStdpConfig`, constants)
  - Classical Hebbian STDP (`hebbian`: `apply_classical_stdp`, `StdpParams`, `HebbianIzhikevichNetwork`)
- Domain-agnostic contracts and errors for downstream integration

## Does Not Own

- Sensory / spike encoding algorithms or `Encoder` trait (owned by `axon-encoder`)
- Topology generation, synaptic wiring, axonal delays, or mesh infrastructure (owned by `synaptic-mesh`)
- Reward shaping, credit assignment, `Environment` trait, or modulator mapping from objectives (owned by `limbic-critic`)
- Full training loops, orchestration, progress tracking, or checkpointing (owned by `plasticity-lab`)
- IPC protocols, transports, backends, or messaging (owned by `corpus-ipc`)
- Headless runtime execution, daemon loop, service registry, or TOML config (owned by `brainstem-daemon`)
- FPGA parameter export, Q8.8 fixed-point conversion, UART bridge, or metrics (owned by `silicon-bridge`)
- HDL / SystemVerilog implementations or hardware coordination (owned by `silicon-hdl` / `Spikenaut-Hardware`)
- Tensor math, MoE extraction, hybrid ANN-SNN orchestration, or projectors (owned by `cortex-tensor`, `engram-parser`, `hybrid-fusion`)
- Streaming feature extraction / stochastic signal stats (owned by `kinetic-signals`)
- Energy / metabolic modeling or ledgers (owned by `metabolic-ledger`)
- Supervision / relay (CPU+GPU) (owned by `thalamic-relay`)

## Allowed Dependencies

- `serde` (with derive) — for serialization of core types
- `rand` — for stochastic elements inside neuron models
- Minimal std + the above; keep the crate lightweight and portable

## Forbidden Dependencies / Domains

- Any I/O, networking, async runtimes (tokio, zmq, etc.)
- Hardware, FPGA, or fixed-point crates
- Domain-specific reward or environment implementations
- Training / optimization frameworks
- Application / supervisor orchestration logic
- Julia FFI or cross-language concerns (handled in consumer crates)

## Core-Library vs Supervisor/App vs Deployment/Hardware Boundaries

| Layer                  | Responsibility                                      | Example Repos                          |
|------------------------|-----------------------------------------------------|----------------------------------------|
| **Core Library**       | Neuron dynamics, network step, generic modulators, plasticity primitives, foundational shared traits | `neuromod`            |
| **Supervisor/App**     | Encoding, topology, training loops, IPC, runtime daemon, reward shaping | `axon-encoder`, `synaptic-mesh`, `plasticity-lab`, `corpus-ipc`, `brainstem-daemon`, `thalamic-relay`, `limbic-critic` |
| **Deployment/Hardware**| Parameter export, fixed-point, UART, HDL synthesis  | `silicon-bridge`, `Spikenaut-Hardware` |

## Domain Leaks

1. Historical naming (pre-0.5 "spikenaut", mining/HFT references) — cleaned in 0.5.0; docs and code are now domain-agnostic.
2. `GenericReward` + `UnitReward` are intentionally minimal examples; real reward logic must live in `limbic-critic` or domain adapters.

## Migration Risks

| Risk                              | Severity | Mitigation |
|-----------------------------------|----------|------------|
| Downstream crates assuming concrete types instead of future traits | Medium | #37 + #43 (traits + ADR) |
| Version skew on `neuromod` (some crates pin old 0.4 / revs) | Medium | #42 (remove rev pins), use `branch = "main"` going forward |
| Accidental domain logic creeping into core models | Low | Strict review + CI domain-agnostic grep |

## Sequencing Questions

1. When #37 lands, should `SpikingNetwork` (and neuron types) become trait-based for easier substitution?
2. Should basic `GenericReward` example stay in neuromod, or move example impls strictly to `limbic-critic`?
3. How do topology-aware networks in `synaptic-mesh` compose with the flat `SpikingNetwork` here?

## Related Boundary Issues

- [neuromod #11](https://github.com/Limen-Neural/neuromod/issues/11)
- [neuromod #25](https://github.com/Limen-Neural/neuromod/issues/25)
- limbic-critic boundary matrix (docs/BOUNDARY_MATRIX.md)
- [brainstem-daemon #4](https://github.com/Limen-Neural/brainstem-daemon/issues/4)
- [silicon-bridge #3](https://github.com/Limen-Neural/silicon-bridge/issues/3)
- [Spikenaut-Hardware #3](https://github.com/Limen-Neural/Spikenaut-Hardware/issues/3)
- hybrid-fusion LIM-9 boundary notes
- See also the org modularization index (#44) and ADR (#43)

## Validation

- This document is planning-only (no implementation changes).
- Cross-checked against current source (`src/lib.rs`, `src/modulators.rs`, `src/engine.rs`, neuron modules) and sibling READMEs / boundary docs.
- Output linkable from Linear LIM-9 and the listed GitHub issues.
