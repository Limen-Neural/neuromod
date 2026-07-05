# ADR 001: Shared traits live in neuromod

**Status:** Accepted (after #37 merges)

**Date:** 2026-07-04

## Context

The Limen-Neural org maintains a growing collection of Rust crates for spiking neural networks and related neuromorphic infrastructure (encoding, topology, runtime, deployment, training, hardware). Multiple crates need common interfaces (e.g. for neurons, networks, modulators, reward) so that:

- Downstream code (plasticity-lab, brainstem-daemon, hybrid-fusion, thalamic-relay, etc.) can depend on contracts without pulling in concrete implementations or creating tight coupling.
- Changes to shared interfaces are governed by a single semver surface.
- Duplication and drift are avoided.

A separate `limen-traits` crate was considered but would add repo overhead, publishing complexity, and another moving part for a small number of focused contracts. Per-repo copies risk divergence.

After the trait audit (#35) and design (#36), the decision point is where the canonical definitions should live.

## Decision

Shared traits (and closely related core types where the trait is the primary public surface) are defined and exported from the `neuromod` crate.

No new dedicated traits-only repository will be created.

`neuromod` already hosts the foundational dynamics (neuron models, `SpikingNetwork`, `NeuroModulators`, `GenericReward`). It is the natural home for the contracts that describe those dynamics.

## Consequences

- Sibling repos depend on `neuromod` (via git `branch = "main"` or published version) for the shared interfaces.
- Semver rules apply to trait changes: breaking changes to a trait bump the major version of `neuromod`.
- `neuromod` must remain focused (core library layer) — it does not absorb encoding, topology, reward shaping, or runtime concerns (see boundary matrix in `docs/neuromod-boundary-matrix.md`).
- Concrete implementations in `neuromod` (e.g. `LifNeuron`, `SpikingNetwork`) will implement the traits; downstream crates can provide alternative implementations behind the same traits.
- Documentation and examples must clearly separate "the trait" from "the neuromod-provided implementation".
- The existing `GenericReward` trait serves as the precedent and proof-of-concept for this placement.

## Alternatives Considered

- **Separate `limen-traits` crate** — Rejected. Extra repository, separate release cadence, and maintenance burden for what is a thin set of contracts tightly related to neuromod's core models. Increases friction for the small team.
- **Per-repo trait copies** — Rejected. High risk of drift (different method signatures, documentation, semantics) leading to integration bugs and duplicated maintenance.
- **Move traits into `synaptic-mesh` or another peer** — Rejected. `neuromod` is the lowest layer for neuron/network dynamics; other crates are either peers (mesh, encoder) or consumers.

## References

- Parent epic: #44 (org modularization standards and index)
- Depends on: #37 (implement and export core traits from neuromod)
- Related: #35 (audit), #36 (design), #43 (this ADR)
- `GenericReward` definition: `src/modulators.rs`
- Cross-repo usage examples: plasticity-lab, hybrid-fusion, brainstem-daemon, limbic-critic boundary notes
- Boundary matrix: `docs/neuromod-boundary-matrix.md`
- Git dep standard (see `docs/org-modularization.md`)

## Acceptance

- [ ] Traits defined and exported per the #36 spec and #37 implementation
- [ ] `cargo doc --no-deps` and tests pass
- [ ] README and rustdoc updated
- [ ] This file committed and linked from README
- [ ] Sibling repos can consume the traits (follow-up work)
