#!/usr/bin/env python3
"""Part 2 analysis and plot: open-square specific-heat peaks and the
extrapolation T_peak(L) = T_c + a/L.

Writes artifacts/boundary-critical.png and runs/boundary/peak_summary.json,
and prints the peak/refinement tables (chi and step comparisons, both size
fits, deviation from Onsager).
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from peaks import (ONSAGER_TC, cv_curve, infer_h, linfit, load_runs,  # noqa: E402
                   peak_from_cv)

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs" / "boundary"
ART = ROOT / "artifacts"
SIZES = [16, 32, 64]
STEP_BROAD = 0.0025
STEP_REF = 0.00125


def refined(l, chi):
    return RUNS / f"L{l}" / f"chi{chi}" / "refined"


def broad(l):
    return RUNS / f"L{l}" / "chi16" / "broad"


def peak_chi(l, chi, step):
    ks, gs, _ = load_runs(refined(l, chi))
    if step == STEP_BROAD:  # every other refined sample
        ks, gs = ks[::2], gs[::2]
    kcv, cvv = cv_curve(ks, gs, step)
    k_peak, cv_peak = peak_from_cv(kcv, cvv)
    return k_peak, 1.0 / k_peak, cv_peak


def main():
    # Broad curves for the figure.
    broad_curves = {}
    for l in SIZES:
        ks, gs, _ = load_runs(broad(l))
        kcv, cvv = cv_curve(ks, gs, STEP_BROAD)
        broad_curves[l] = ([1.0 / k for k in kcv], cvv)

    # Refined peaks at chi=16 and 32 (h=0.00125), plus the wider-step
    # estimate at chi=32 (every other refined sample, h=0.0025).
    table = {}
    for l in SIZES:
        k16, t16, c16 = peak_chi(l, 16, STEP_REF)
        k32, t32, c32 = peak_chi(l, 32, STEP_REF)
        k32w, t32w, c32w = peak_chi(l, 32, STEP_BROAD)
        table[l] = {
            "k_peak_chi16_h00125": k16, "t_peak_chi16_h00125": t16, "cv_chi16": c16,
            "k_peak_chi32_h00125": k32, "t_peak_chi32_h00125": t32, "cv_chi32": c32,
            "k_peak_chi32_h0025": k32w, "t_peak_chi32_h0025": t32w, "cv_chi32_wide": c32w,
        }
        ks, gs, _ = load_runs(refined(l, 32))
        kcv, cvv = cv_curve(ks, gs, STEP_REF)
        broad_curves[(l, "refined")] = ([1.0 / k for k in kcv], cvv)

    # Size fits at chi=32, h=0.00125.
    xs_all = [1.0 / l for l in SIZES]
    ys_all = [table[l]["t_peak_chi32_h00125"] for l in SIZES]
    _, tc3 = linfit(xs_all, ys_all)
    _, tc2 = linfit(xs_all[1:], ys_all[1:])

    # Stability checks (within 1%).
    checks = {}
    ok = True
    for l in SIZES:
        d_chi = abs(table[l]["t_peak_chi16_h00125"] - table[l]["t_peak_chi32_h00125"]) / table[l]["t_peak_chi32_h00125"]
        d_step = abs(table[l]["t_peak_chi32_h0025"] - table[l]["t_peak_chi32_h00125"]) / table[l]["t_peak_chi32_h00125"]
        checks[l] = {"rank_shift_pct": 100 * d_chi, "step_shift_pct": 100 * d_step}
        ok = ok and d_chi < 0.01 and d_step < 0.01

    print("open-square peak refinement (chi=32 unless stated)")
    print(f"{'L':>4} {'K_peak':>10} {'T_peak':>10} {'c_V':>8} "
          f"{'T(16,h.00125)':>14} {'T(32,h.0025)':>13}")
    for l in SIZES:
        t = table[l]
        print(f"{l:>4} {t['k_peak_chi32_h00125']:>10.5f} "
              f"{t['t_peak_chi32_h00125']:>10.4f} {t['cv_chi32']:>8.4f} "
              f"{t['t_peak_chi16_h00125']:>14.4f} {t['t_peak_chi32_h0025']:>13.4f}")
    print(f"T_c fit (all sizes):     {tc3:.4f}")
    print(f"T_c fit (largest two):   {tc2:.4f}")
    print(f"Onsager 2/log(1+sqrt2):  {ONSAGER_TC:.6f}")
    print(f"deviation all sizes:     {100*abs(tc3-ONSAGER_TC)/ONSAGER_TC:.3f}%")
    print(f"deviation largest two:   {100*abs(tc2-ONSAGER_TC)/ONSAGER_TC:.3f}%")
    print("peak stability under rank and step changes: "
          + ("PASS (<1%)" if ok else "FAIL"))
    for l in SIZES:
        c = checks[l]
        print(f"  L={l}: rank shift {c['rank_shift_pct']:.3f}%, "
              f"step shift {c['step_shift_pct']:.3f}%")

    summary = {
        "geometry": "open",
        "method": "boundary",
        "peaks": {str(l): table[l] for l in SIZES},
        "tc_all_sizes": tc3,
        "tc_largest_two": tc2,
        "onsager": ONSAGER_TC,
        "stability_checks": {str(l): checks[l] for l in SIZES},
        "stability_pass": ok,
    }
    (RUNS / "peak_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- figure ----
    ART.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    colors = {16: "#1a7f37", 32: "#1f6fb4", 64: "#d14836"}
    for l in SIZES:
        ts, cv = broad_curves[l]
        ax1.plot(ts, cv, color=colors[l], lw=1.5, label=f"L = {l}")
        tr, cvr = broad_curves[(l, "refined")]
        ax1.plot(tr, cvr, "o", ms=2.6, color=colors[l], alpha=0.7)
    ax1.set_xlabel("Temperature $T$")
    ax1.set_ylabel("Specific heat per spin $c_V$")
    ax1.set_xlim(2.05, 2.45)
    ax1.set_ylim(0.0, 1.8)
    ax1.set_title("(a) Open specific-heat peaks")
    ax1.text(0.03, 0.97, "solid: $\\chi=16$, $h=0.0025$\npoints: $\\chi=32$, $h=0.00125$",
             transform=ax1.transAxes, va="top", ha="left", fontsize=8)
    ax1.legend(fontsize=8, loc="center right")

    for l in SIZES:
        ax2.plot(1.0 / l, table[l]["t_peak_chi32_h00125"], "o", color=colors[l],
                 ms=6, label=f"L = {l}" if l == SIZES[0] else None, zorder=3)
    for l in SIZES:
        ax2.plot([1.0 / l], [table[l]["t_peak_chi32_h00125"]], "o", color=colors[l], ms=6)
    xfit = [0.0, 1.0 / SIZES[0]]
    ax2.plot(xfit, [tc3 + (1.0 / SIZES[0]) * (ys_all[0] - tc3) * SIZES[0] / 1,
                    tc3], color="#444444", lw=1.0)
    # draw the two least-squares lines properly
    s3, i3 = linfit(xs_all, ys_all)
    s2, i2 = linfit(xs_all[1:], ys_all[1:])
    ax2.plot([0, 1 / SIZES[0]], [i3, i3 + s3 / SIZES[0]], color="#666666", lw=1.2,
             label="all sizes")
    ax2.plot([0, 1 / SIZES[1]], [i2, i2 + s2 / SIZES[1]], color="#b8860b", lw=1.2, ls="--",
             label="largest two")
    ax2.axhline(ONSAGER_TC, color="#d14836", ls=":", lw=1.2)
    ax2.annotate("Onsager", xy=(0.045, ONSAGER_TC), xytext=(0.045, ONSAGER_TC - 0.012),
                 fontsize=8, color="#d14836")
    ax2.annotate(f"$T_c$ = {tc3:.3f}", xy=(0.0, tc3), xytext=(0.004, tc3 + 0.010),
                 fontsize=8, color="#444444")
    ax2.annotate(f"{tc2:.3f}", xy=(0.0, tc2), xytext=(0.004, tc2 - 0.022),
                 fontsize=8, color="#b8860b")
    ax2.set_xlabel("Inverse side length $1/L$")
    ax2.set_ylabel("Peak temperature")
    ax2.set_xlim(0.0, 0.07)
    ax2.set_ylim(2.08, 2.30)
    ax2.set_title("(b) Extrapolate the peak")
    ax2.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(ART / "boundary-critical.png", dpi=160)
    print(f"wrote {ART / 'boundary-critical.png'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
