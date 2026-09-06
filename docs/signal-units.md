# Signal Unit Conventions

> Scope: [`SignalProfile`](../src/modulators.rs) and `NeuroModulators::from_signals`.
> Tracked by [neuromod #75](https://github.com/Limen-Neural/neuromod/issues/75).

`neuromod` is a pure neuron-dynamics library. It has no opinion about what a
caller's signals measure, and it cannot validate them. This page states the one
contract it *does* enforce, and how a downstream crate is expected to declare its
own units.

## The contract

| Side | Rule |
|------|------|
| **Outputs** | Every `NeuroModulators` field (`dopamine`, `serotonin`, `acetylcholine`, `norepinephrine`) is a **dimensionless level**, with `0.0..=1.0` as the intended range. `from_signals` and `decay()` keep levels inside it for finite inputs. See [Range caveats](#range-caveats) for the two ways a level leaves it. |
| **Inputs** | The four channels passed to `from_signals` (thermal, power, throughput, timing) are **bare `f32`s with no unit**. |
| **Profiles** | Each `SignalProfile` field is expressed in the **same unit as the channel it pairs with**, so every ratio inside `from_signals` cancels to a dimensionless number. |

Consequence: units are declared exactly once per consumer, in the profile. There
is no conversion layer, no unit tag on the wire, and no runtime check. Feeding a
threshold in °C against a signal in °F is a caller bug this crate cannot detect.

## Channels

| Channel | Profile field(s) | Field unit | Drives |
|---------|------------------|-----------|--------|
| thermal | `thermal_threshold` | thermal units | norepinephrine (max with power) |
| power | `power_baseline`, `power_scale` | power units | norepinephrine (max with thermal) |
| throughput | `throughput_scale`, `stability_target` | throughput units | dopamine, serotonin |
| timing | `timing_scale` | timing units | acetylcholine |

## Mapping

```text
dopamine       = clamp(throughput / throughput_scale)
acetylcholine  = clamp(timing / timing_scale)
serotonin      = clamp(1 - 2 * |throughput - stability_target|)
norepinephrine = max(thermal_stress, power_stress)
  thermal_stress = clamp((thermal - thermal_threshold) / thermal_threshold)  [0 at or below threshold]
  power_stress   = clamp((power - power_baseline) / power_scale)
```

`clamp(x)` is `x.clamp(0.0, 1.0)`. Negative signals read as `0.0`; over-range
signals saturate instead of wrapping. A divisor within `f32::EPSILON` of zero —
or a `NaN` profile field, which fails the same `abs() > f32::EPSILON` guard —
yields `0.0` for that term rather than an infinity or `NaN`.

Reference points worth remembering:

- `throughput_scale` is the throughput that maps to dopamine `1.0`.
- `timing_scale` is the timing value that maps to acetylcholine `1.0`.
- Thermal stress starts at `thermal_threshold` and saturates at `2 x thermal_threshold`.
- Power stress starts at `power_baseline` and saturates at `power_baseline + power_scale`.

## Range caveats

`0.0..=1.0` is the intended range, not an invariant the type enforces. Two
documented ways out of it:

- **Negative amounts.** `add_reward`, `add_serotonin`, `boost_focus`, and
  `add_norepinephrine` cap the upper bound with `.min(1.0)` and do **not** clamp
  the lower one, so `add_reward(-0.5)` on a level of `0.0` yields `-0.5`. Keeping
  those amounts non-negative is the caller's job. (`decay()` does clamp low, so a
  negative level recovers to `0.0` on the next decay.)
- **`NaN` signals.** `f32::clamp` returns `NaN` for a `NaN` input, so
  `from_signals` propagates a `NaN` throughput into `dopamine` and `serotonin`,
  and a `NaN` timing into `acetylcholine`. `norepinephrine` is the exception and
  reads `0.0`: the thermal comparison is false for `NaN`, and `f32::max` discards
  a `NaN` operand. A `NaN` **profile** field is caught by the near-zero divisor
  guard and yields `0.0` for that term. Validate signals upstream if they can be
  `NaN`.

Both are preserved behavior, documented rather than changed — tightening either
would move existing callers' trajectories.

## Known wart: serotonin is not scale-normalized

Serotonin measures the deviation from `stability_target` in **raw throughput
units**, with a fixed half-width of `0.5` — it is the only term that does not
divide by a profile scale. A profile whose throughput unit is far from `1.0`
therefore pins serotonin at `0.0` for virtually every input.

The deprecated `SignalProfile::hardware_calibrated()` is exactly that case:
`throughput_scale` is `0.0105` while `stability_target` is `1.05`, two orders of
magnitude apart, so the two throughput-derived channels are never informative at
the same time.

| throughput | dopamine | serotonin |
|-----------|----------|-----------|
| `0.005` | `0.48` | `0.0` |
| `0.0105` | `1.0` | `0.0` |
| `0.55` | `1.0` | `0.0` |
| `1.0` | `1.0` | `0.9` |
| `1.05` | `1.0` | `1.0` |
| `1.55` | `1.0` | `0.0` |

Dopamine saturates at any throughput at or above `0.0105`; serotonin is non-zero
only within `0.5` of `1.05`. Across the range where dopamine still varies,
serotonin is pinned at `0.0`; across the range where serotonin varies, dopamine
is already pinned at `1.0`.

Keep `stability_target` and `throughput_scale` in the same range, or feed a
pre-normalized throughput channel. The behavior is documented rather than
changed: fixing it silently would move every existing caller's serotonin
trajectory.

## Choosing a profile

### Pre-normalized signals — `SignalProfile::default()`

For callers that already map their signals into `0.0..=1.0`. Every scale is
`1.0`, so each channel passes through unchanged; `thermal_threshold` is `0.5`, so
the upper half of the normalized thermal channel spans the full stress range.
This is the recommended starting point, and it keeps the serotonin wart above out
of play.

```rust
use neuromod::{NeuroModulators, SignalProfile};

let mods = NeuroModulators::from_signals(&SignalProfile::default(), 0.75, 0.4, 1.0, 0.6);
// dopamine 1.0, serotonin 1.0, acetylcholine 0.6, norepinephrine 0.5
```

### Physical units — construct the struct

All fields are public. Declare your own reference values:

```rust
use neuromod::SignalProfile;

// thermal °C, power W, throughput items/s, timing samples
let profile = SignalProfile {
    throughput_scale: 500.0,
    thermal_threshold: 80.0,
    power_baseline: 120.0,
    power_scale: 40.0,
    timing_scale: 1024.0,
    stability_target: 400.0,
};
```

Note that `stability_target: 400.0` with a raw-unit throughput channel puts
serotonin at `0.0` unless throughput sits within `0.5 items/s` of the target —
another reason to normalize upstream when serotonin matters.

## Status of `hardware_calibrated()`

`SignalProfile::hardware_calibrated()` is **deprecated since 0.6.0** and still
returns the values it always has. Nothing is removed and nothing is renamed; the
constructor stays compiling so downstream crates migrate on their own schedule.

Calibration constants describe a *deployment*, not neuron dynamics, so per
[docs/neuromod-boundary-matrix.md](neuromod-boundary-matrix.md) they belong to
the consuming crate. To migrate, copy the literal into your own code:

```rust
use neuromod::SignalProfile;

let profile = SignalProfile {
    throughput_scale: 0.0105,
    thermal_threshold: 83.0,
    power_baseline: 400.0,
    power_scale: 50.0,
    timing_scale: 2640.0,
    stability_target: 1.05,
};
```

A unit test asserts this literal stays equal to what the deprecated constructor
returns, so the two cannot drift while it remains in the crate. Removal, if it
happens, is a separate breaking change and gets its own issue, CHANGELOG entry,
and minor version bump.
