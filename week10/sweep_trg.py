#!/usr/bin/env python3
"""Part 3 sweep: periodic squares L=64,256,1024 by TRG contraction.

Same protocol as the open-square study: K = 0.40..0.50 step 0.0025 at
chi=16, nine refined points spaced 0.00125 around each broad peak at
chi=16 and chi=32, then peak fits in plot_trg.py.
"""

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BIN = ROOT / "target" / "release" / "thermo"
RUNS = ROOT / "runs" / "trg"
SIZES = [64, 256, 1024]
K0, STEP, NK = 0.40, 0.0025, 41
REFINED_STEP = 0.00125
REFINED_M = range(-4, 5)
CHIS = [16, 32]

sys.path.insert(0, str(ROOT))
from peaks import cv_curve, load_runs, peak_from_cv  # noqa: E402
from sweep_boundary import write_table  # noqa: E402


def run_one(l, k, chi, out_dir: Path):
    rj = out_dir / "run.json"
    if rj.is_file():
        return json.loads(rj.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(BIN), "--method", "trg", "--l", str(l),
        "--k", f"{k:.10f}", "--chi", str(chi), "--out", str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd} failed: {proc.stderr}")
    return json.loads(rj.read_text())


def parallel(jobs, workers=4):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda j: run_one(*j), jobs))


def broad_root(l, chi):
    return RUNS / f"L{l}" / f"chi{chi}" / "broad"


def refined_root(l, chi):
    return RUNS / f"L{l}" / f"chi{chi}" / "refined"


def main():
    grid = [round(K0 + i * STEP, 10) for i in range(NK)]

    jobs = []
    for l in SIZES:
        for i, k in enumerate(grid):
            jobs.append((l, k, 16, broad_root(l, 16) / f"K{i:03d}"))
    print(f"broad scan: {len(jobs)} TRG runs at chi=16 ...")
    parallel(jobs)

    peaks = {}
    for l in SIZES:
        ks, gs, _ = load_runs(broad_root(l, 16))
        assert len(ks) == NK, f"L={l}: {len(ks)} runs"
        kcv, cvv = cv_curve(ks, gs, STEP)
        k_peak, cv_peak = peak_from_cv(kcv, cvv)
        i_star = min(range(NK), key=lambda i: abs(grid[i] - k_peak))
        peaks[l] = (grid[i_star], k_peak, cv_peak)
        print(f"L={l}: broad c_V peak at K ~ {k_peak:.5f} (T ~ {1.0/k_peak:.4f}), "
              f"nearest grid point K = {grid[i_star]:.4f}, c_V = {cv_peak:.4f}")

    jobs = []
    for l in SIZES:
        center = peaks[l][0]
        for chi in CHIS:
            for m in REFINED_M:
                k = round(center + m * REFINED_STEP, 10)
                jobs.append((l, k, chi, refined_root(l, chi) / f"R{m:+03d}"))
    print(f"refinement: {len(jobs)} TRG runs at chi=16,32 ...")
    parallel(jobs)

    tables = RUNS / "tables"
    for l in SIZES:
        write_table(tables / f"trg_broad_L{l}_chi16.csv", broad_root(l, 16))
        for chi in CHIS:
            write_table(tables / f"trg_refined_L{l}_chi{chi}.csv", refined_root(l, chi))

    summary = {
        "method": "trg",
        "geometry": "periodic",
        "sizes": SIZES,
        "broad": {str(l): {"k_grid_center_of_peak": peaks[l][0],
                           "k_peak_broad": peaks[l][1],
                           "cv_peak_broad": peaks[l][2],
                           "t_peak_broad": 1.0 / peaks[l][1]} for l in SIZES},
        "k_step_broad": STEP,
        "k_step_refined": REFINED_STEP,
        "chis": CHIS,
    }
    (RUNS / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {RUNS / 'sweep_summary.json'} and tables under {tables}")


if __name__ == "__main__":
    main()
