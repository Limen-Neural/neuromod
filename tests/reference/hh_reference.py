"""
Independent float64 HH reference, Codex (OpenAI), 2026-09-17.

Derives rates and ionic currents directly from the documented HH differential
equations; does not import or call neuromod. Every preset input is rounded to
f32 once, matching the public constructor parameters, then integrated in f64.
The half-step rerun estimates reference integration error. Event criterion is
an upward crossing of 0 mV absolute (65 mV relative to nominal -65 mV rest).
No gate clamping is applied to these trajectories. This is a numerical
reference, not biological calibration. Regenerate with:
    python3 tests/reference/hh_reference.py
The 0.001/0.0005 ms comparison converges in endpoint state to about 1e-11;
the cortical sampled peak differs by 0.001153 mV due to grid alignment.
Production f32 tolerances are 0.005 mV peak, 0.001 mV endpoint voltage,
1e-5 endpoint gates, and dt + 0.0005 + 1e-6 ms for sampled event times.
"""

import json
import math
import struct
from pathlib import Path


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def rates(v):
    def ratio(x, divisor):
        return divisor if x == 0 else x / math.expm1(x / divisor)

    return (
        0.1 * ratio(25 - v, 10),
        4 * math.exp(-v / 18),
        0.07 * math.exp(-v / 20),
        1 / (math.exp((30 - v) / 10) + 1),
        0.01 * ratio(10 - v, 10),
        0.125 * math.exp(-v / 80),
    )


def reference(cortical, current, duration, dt):
    shift = 65 if cortical else 0
    reversals = tuple(map(f32, (50, -77, -54.387) if cortical else (115, -12, 10.6)))
    temperature = f32(37 if cortical else 6.3)
    phi = 3 ** ((temperature - 6.3) / 10)
    am, bm, ah, bh, an, bn = rates(0)
    y = (-float(shift), am / (am + bm), ah / (ah + bh), an / (an + bn))
    threshold = 65 - shift

    def rhs(z):
        v, m, h, n = z
        a, b, c, d, e, f = rates(v + shift)
        ina = f32(120) * m**3 * h * (v - reversals[0])
        ik = f32(36) * n**4 * (v - reversals[1])
        il = f32(0.3) * (v - reversals[2])
        return (
            current - ina - ik - il,
            phi * (a * (1 - m) - b * m),
            phi * (c * (1 - h) - d * h),
            phi * (e * (1 - n) - f * n),
        )

    def stage(y, k, h):
        return tuple(a + h * b for a, b in zip(y, k))

    peak = y[0]
    crossings = []
    for i in range(round(duration / dt)):
        a = rhs(y)
        b = rhs(stage(y, a, dt / 2))
        c = rhs(stage(y, b, dt / 2))
        d = rhs(stage(y, c, dt))
        z = tuple(
            v + dt * (p + 2 * q + 2 * r + s) / 6 for v, p, q, r, s in zip(y, a, b, c, d)
        )
        if y[0] < threshold <= z[0]:
            crossings.append((i + 1) * dt)
        peak = max(peak, z[0])
        y = z
    return {
        "dt": dt,
        "duration": duration,
        "current": current,
        "convention": "absolute" if cortical else "relative_to_rest",
        "peak_voltage": peak,
        "crossing_times": crossings,
        "endpoint": list(y),
    }


if __name__ == "__main__":
    out = {}
    for name, args in [
        ("squid_spike", (False, 10, 5)),
        ("squid_subthreshold", (False, 2, 25)),
        ("cortical_subthreshold", (True, 20, 25)),
        ("cortical_spike", (True, 500, 25)),
    ]:
        out[name] = [reference(*args, dt) for dt in (0.001, 0.0005)]
    Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
