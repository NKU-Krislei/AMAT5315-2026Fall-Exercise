"""Shared peak-location helpers for the boundary-MPS and TRG studies.

The specific heat is computed from the saved g(K) tables by the central
difference c_V(K;h) ~= K^2 [g(K+h) - 2 g(K) + g(K-h)] / h^2. Peaks are fitted
with a three-point parabola in K, and finite-size peaks are extrapolated with
T_peak(L) = T_c + a/L.
"""

import json
import math
from pathlib import Path

ONSAGER_TC = 2.0 / math.log(1.0 + math.sqrt(2.0))


def load_runs(run_root: Path):
    """Load (k, g) pairs from a directory of <name>/run.json subdirectories,
    sorted by K."""
    records = []
    for d in run_root.iterdir():
        rj = d / "run.json"
        if not rj.is_file():
            continue
        meta = json.loads(rj.read_text())
        records.append((float(meta["settings"]["k"]),
                        float(meta["g"]),
                        float(meta.get("solver_time_sec", 0.0))))
    records.sort(key=lambda r: r[0])
    ks = [r[0] for r in records]
    gs = [r[1] for r in records]
    times = [r[2] for r in records]
    return ks, gs, times


def cv_curve(ks, gs, h):
    """Central-difference specific heat on the interior of a table."""
    out_k, out_cv = [], []
    for i in range(1, len(ks) - 1):
        # guard against uneven spacing
        if abs((ks[i + 1] - ks[i]) - h) > 1e-9 or abs((ks[i] - ks[i - 1]) - h) > 1e-9:
            continue
        cv = ks[i] ** 2 * (gs[i + 1] - 2.0 * gs[i] + gs[i - 1]) / h**2
        out_k.append(ks[i])
        out_cv.append(cv)
    return out_k, out_cv


def parabola_vertex(xs, ys):
    """Vertex abscissa of the parabola through three points."""
    assert len(xs) == 3 and len(ys) == 3
    (x0, y0), (x1, y1), (x2, y2) = zip(xs, ys)
    # Solve the 3x3 Vandermonde system y = a x^2 + b x + c exactly.
    den = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / den
    if a == 0:
        raise ValueError("degenerate parabola")
    return -b / (2.0 * a)


def peak_from_cv(ks_cv, cvs):
    """Three-point parabola peak through the maximum and its neighbours."""
    n = len(ks_cv)
    # the maximum must have both neighbours available
    lo = 1 if n >= 3 else 0
    hi = n - 2 if n >= 3 else n - 1
    imax = max(range(lo, hi + 1), key=lambda i: cvs[i])
    xs = [ks_cv[imax - 1], ks_cv[imax], ks_cv[imax + 1]]
    ys = [cvs[imax - 1], cvs[imax], cvs[imax + 1]]
    k_peak = parabola_vertex(xs, ys)
    return k_peak, cvs[imax]


def linfit(xs, ys):
    """Least-squares line y = slope*x + intercept."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, my - slope * mx


def infer_h(ks):
    hs = {round(ks[i + 1] - ks[i], 12) for i in range(len(ks) - 1)}
    assert len(hs) == 1, f"uneven spacing: {hs}"
    return hs.pop()