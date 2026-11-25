#!/usr/bin/env python3
"""Part 2 sweep: open squares L=16,32,64 by boundary-MPS contraction.

Stage 1 sweeps K = 0.40..0.50 step 0.0025 at chi=16 for each size, computes
the specific heat from the saved g values, and locates each peak on the grid.
Stage 2 samples nine points spaced 0.00125 around the nearest grid point to
each peak, at chi=16 and chi=32. Every run writes its own evidence directory
under runs/boundary/; g(K) tables are saved as CSV.
"""

import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BIN = ROOT / "target" / "release" / "thermo"
RUNS = ROOT / "runs" / "boundary"
SIZES = [16, 32, 64]
K0, STEP, NK = 0.40, 0.0025, 41
REFINED_STEP = 0.00125
REFINED_M = range(-4, 5)
CHIS = [16, 32]

sys.path.insert(0, str(ROOT))
from peaks import cv_curve, infer_h, load_runs, peak_from_cv  # noqa: E402


def k_grid():
    return [round(K0 + i * STEP, 10) for i in range(NK)]


def run_one(method, l, k, chi, out_dir: Path):
    rj = out_dir / "run.json"
    if rj.is_file():
        return json.loads(rj.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(BIN), "--method", method, "--l", str(l),
        "--k", f"{k:.10f}", "--chi", str(chi), "--out", str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd} failed: {proc.stderr}")
    return json.loads(rj.read_text())


def parallel(jobs, workers=4):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda j: run_one(*j), jobs))
    return results


def broad_root(l, chi):
    return RUNS / f"L{l}" / f"chi{chi}" / "broad"


def refined_root(l, chi):
    return RUNS / f"L{l}" / f"chi{chi}" / "refined"


def write_table(path: Path, run_root: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in sorted(run_root.iterdir()):
        rj = d / "run.json"
        if not rj.is_file():
            continue
        meta = json.loads(rj.read_text())
        rows.append((
            meta["settings"]["k"],
            meta["g"],
            meta["log_z"],
            meta["solver_time_sec"],
        ))
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["k", "g", "log_z", "solver_time_sec"])
        for k, g, lz, t in rows:
            w.writerow([f"{k:.10f}", f"{g:.16e}", f"{lz:.16e}", f"{t:.6f}"])


def main():
    grid = k_grid()

    # Stage 1: broad scan at chi=16.
    jobs = []
    for l in SIZES:
        for i, k in enumerate(grid):
            jobs.append(("boundary", l, k, 16, broad_root(l, 16) / f"K{i:03d}"))
    print(f"broad scan: {len(jobs)} runs at chi=16 ...")
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

    # Stage 2: refinement around the nearest grid point, at chi=16 and 32.
    jobs = []
    for l in SIZES:
        center = peaks[l][0]
        for chi in CHIS:
            for m in REFINED_M:
                k = round(center + m * REFINED_STEP, 10)
                jobs.append(("boundary", l, k, chi,
                             refined_root(l, chi) / f"R{m:+03d}"))
    print(f"refinement: {len(jobs)} runs at chi=16,32 ...")
    parallel(jobs)

    # g(K) tables.
    tables = RUNS / "tables"
    for l in SIZES:
        write_table(tables / f"boundary_broad_L{l}_chi16.csv", broad_root(l, 16))
        for chi in CHIS:
            write_table(tables / f"boundary_refined_L{l}_chi{chi}.csv",
                        refined_root(l, chi))

    summary = {
        "method": "boundary",
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
