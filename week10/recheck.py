#!/usr/bin/env python3
"""Independent recomputation of log Z from the saved evidence.

For a boundary run: contract the saved final cores, closing every physical
leg with b(K) = [1/sqrt(cosh K), 0], and compare log Z = ell + log(scalar)
with the saved value. For a TRG run: close the saved final tensor with
z = sum_{i,j} A[i,j,i,j] and compare log Z = ell + log z.

Usage: python3 recheck.py runs/boundary64 [runs/trg1024 ...]
"""

import json
import math
import sys
from pathlib import Path

import numpy as np


def recheck_boundary(run_dir: Path):
    meta = json.loads((run_dir / "run.json").read_text())
    k = float(meta["settings"]["k"])
    ell = float(meta["ell"])
    dims = meta["final_cores_dims"]
    flat = np.fromfile(run_dir / meta["final_cores_file"], dtype="<f8")
    b = np.array([1.0 / math.sqrt(math.cosh(k)), 0.0])
    v = np.array([1.0])
    ofs = 0
    for dl, dp, dr in dims:
        core = flat[ofs:ofs + dl * dp * dr].reshape(dl, dp, dr)
        ofs += dl * dp * dr
        # v_beta = sum_{alpha, p} v_alpha * b_p * core[alpha, p, beta]
        v = np.einsum("a,p,apb->b", v, b, core)
    assert v.size == 1, f"chain did not close: v has shape {v.shape}"
    log_z = ell + math.log(float(v[0]))
    saved = float(meta["log_z"])
    return saved, log_z


def recheck_trg(run_dir: Path):
    meta = json.loads((run_dir / "run.json").read_text())
    ell = float(meta["ell"])
    dims = meta["final_tensor_dims"]
    a = np.fromfile(run_dir / meta["final_tensor_file"], dtype="<f8")
    a = a.reshape(dims)
    z = float(np.einsum("ijij->", a))
    log_z = ell + math.log(z)
    saved = float(meta["log_z"])
    return saved, log_z


def main():
    dirs = [Path(p) for p in sys.argv[1:]]
    if not dirs:
        print(__doc__)
        return 2
    worst = 0.0
    for d in dirs:
        meta = json.loads((d / "run.json").read_text())
        if meta["method"] == "boundary":
            saved, recomputed = recheck_boundary(d)
        elif meta["method"] == "trg":
            saved, recomputed = recheck_trg(d)
        else:
            raise SystemExit(f"unknown method {meta['method']}")
        diff = abs(saved - recomputed)
        worst = max(worst, diff)
        print(f"{d}: method={meta['method']} L={meta['settings']['l']} "
              f"K={meta['settings']['k']} chi={meta['settings']['chi']} "
              f"log Z saved={saved:.12f} recomputed={recomputed:.12f} "
              f"|diff|={diff:.2e}")
    print(f"max |diff| over {len(dirs)} runs: {worst:.3e}")
    return 0 if worst < 1e-9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
