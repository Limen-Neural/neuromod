# Org Modularization Standards and Issue Index

This document is the durable, in-repo reference for the Limen-Neural modularization program. It replaces scattered GitHub issue threads as the canonical place maintainers and agents consult.

See also:
- `docs/neuromod-boundary-matrix.md`
- `docs/adr/001-traits-in-neuromod.md`

## 1. Program overview

`neuromod` hosts shared traits and core SNN contracts for the org (no separate traits crate).

The org maintains ~22 repositories (Rust crates for core infrastructure + Julia research layers + one hardware HDL repo). Hard boundaries are enforced so each owns one layer.

Why `neuromod` for shared traits:

- Lowest layer for neuron dynamics, network stepping, generic neuromodulation, and plasticity primitives.
- Already depended on (directly or indirectly) by runtime, training, relay, and hybrid crates.
- Avoids extra repo overhead and duplication risk.

## 2. Workstream index

| Issue | Phase    | Summary |
|-------|----------|---------|
| #35   | Traits   | Audit trait boundaries across org |
| #36   | Traits   | Design trait API contract |
| #37   | Traits   | Implement traits in neuromod |
| #43   | Traits   | ADR: traits live in neuromod |
| #38   | Deps     | Fix git dependency URLs |
| #42   | Deps     | Remove pinned `rev` pins |
| #40   | Build    | Validate neuromod release profile |
| #41   | Build    | Roll out release profiles org-wide |
| #39   | Tooling  | Initialize beads in missing repos |

(Full current list of open issues in the repo can be found via GitHub search.)

## 3. Execution order

```
Traits:  #35 → #36 → #37 → #43
Deps:    #38 → #42
Build:   #40 → #41
Tooling: #39 (parallel)
```

## 4. Cross-cutting standards

### Git dependencies

- Inter-repo: `git = "https://github.com/Limen-Neural/<repo>", branch = "main"`
- No pinned `rev` unless a documented exception with justification and removal plan.
- Prefer version pins once crates are published to crates.io.
- Example (good):
  ```toml
  neuromod = { git = "https://github.com/Limen-Neural/neuromod", branch = "main" }
  ```
- Current examples of pinned usage (to be cleaned by #42 work) exist in plasticity-lab, etc.

### Release profile template

Every active Rust crate should include:

```toml
[profile.release]
opt-level = 3
lto = true
codegen-units = 1
```

(This is already present in `neuromod/Cargo.toml` and was validated under #40.)

### Beads

- Every active org repo should have `.beads/` committed (complements GitHub Issues; does not replace it).
- See #39.

## 5. Audit commands (copy-paste)

```bash
# Pinned revisions (should be rare)
rg 'rev\s*=' --glob '**/Cargo.toml'

# Git dependencies
rg 'git\s*=' --glob '**/Cargo.toml'

# Trait definitions (to audit surface)
rg '^(pub\s+)?trait\s+' --glob 'src/**/*.rs'

# Beads presence
test -d .beads && echo OK || echo MISSING

# Release profile
rg '^\[profile\.release\]' --glob '**/Cargo.toml' -A 3

# Domain leakage in docs (CI guard)
! grep -riE 'spikenaut|\bhft\b|\bmining\b|\bcrypto\b|eagle-lander' target/doc/<crate>/ || echo "leak"
```

## Related

- Parent: this doc was extracted from the body of #44.
- Siblings: #35–#43.
- Boundary matrix and ADR live alongside this file.
- Org repo inventory (as of 2026-07): axon-encoder, neuromod, synaptic-mesh, kinetic-signals, corpus-ipc, silicon-bridge, thalamic-relay, limbic-critic, engram-parser, cortex-tensor, plasticity-lab, metabolic-ledger, brainstem-daemon, hybrid-fusion, silicon-hdl, plus Julia crates and Spikenaut-Hardware.

Update this document as child issues complete.
