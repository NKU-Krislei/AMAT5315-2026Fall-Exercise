#!/usr/bin/env python3
"""Part 1 plot: singular spectrum and reconstruction error vs retained rank.

Independently recomputes the relative Frobenius errors from the saved binary
matrices with numpy, cross-checks them against the saved singular values, and
writes artifacts/svd.png.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"


def load_f64(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype="<f8")


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "svd"
    meta = json.loads((out / "meta.json").read_text())
    n = int(meta["size"])
    m = load_f64(out / "matrix.bin").reshape(n, n)
    s = load_f64(out / "singular_values.bin")
    norm = float(np.linalg.norm(m))

    rows = []
    for entry in meta["reconstructions"]:
        chi = int(entry["chi"])
        rec = load_f64(out / entry["file"]).reshape(n, n)
        e_meas = float(np.linalg.norm(m - rec) / norm)
        tail = float(np.sqrt(np.sum(s[chi:] ** 2)))
        e_pred = tail / norm
        rows.append((chi, e_meas, e_pred))
        diff = abs(e_meas - e_pred)
        status = "PASS" if diff < 1e-8 else "FAIL"
        print(f"chi={chi:4d}  e_measured={e_meas:.9f}  e_predicted={e_pred:.9f}  "
              f"|diff|={diff:.2e}  {status}")
        if diff >= 1e-8:
            return 1

    e_full = rows[-1][1]
    if e_full >= 1e-8:
        print(f"full-rank error {e_full:.2e} >= 1e-8")
        return 1

    # Spectrum-wide prediction curve and crossing points.
    chis = np.arange(0, n + 1)
    # tail_sums[i] = sum_{alpha >= i} s_alpha^2, so e_pred(chi) = sqrt(tail_sums[chi])/norm.
    tail_sums = np.concatenate([np.cumsum((s ** 2)[::-1])[::-1], [0.0]])
    e_pred_curve = np.sqrt(tail_sums) / norm
    storage_parity = n * n // (2 * n)
    chi10 = int(np.argmax(e_pred_curve[1:] <= 0.10) + 1)
    storage_ratio = (2 * n * chi10) / (n * n)
    print(f"two-factor storage equals the original at chi = {storage_parity}")
    print(f"error first reaches 10% at chi = {chi10} "
          f"(factor storage = {storage_ratio:.3f}x original)")

    ART.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    ax1.plot(np.arange(1, n + 1), s, color="#1f6fb4", lw=1.4)
    ax1.set_xlabel("Singular-value index")
    ax1.set_ylabel("Singular value")
    ax1.set_xlim(0, n)
    ax1.set_xticks([0, storage_parity, n])
    ax1.set_title("(a) Singular spectrum")

    ax2.plot(chis, e_pred_curve, color="#1f6fb4", lw=1.4, label="SVD tail prediction")
    meas_chis = [r[0] for r in rows]
    meas_es = [r[1] for r in rows]
    ax2.plot(meas_chis, meas_es, "o", color="#d14836", ms=5, label="Reconstruction")
    ax2.axvline(storage_parity, color="#666666", ls="--", lw=1.0)
    ax2.annotate("no storage saving", xy=(storage_parity, 0.52), xytext=(storage_parity + 12, 0.55),
                 fontsize=8, color="#444444")
    ax2.axhline(0.10, color="#999999", ls=":", lw=1.0)
    ax2.annotate("10%", xy=(4, 0.105), fontsize=8, color="#666666")
    ax2.annotate(f"$\\chi$ = {chi10}", xy=(chi10, 0.10), xytext=(chi10 - 110, 0.22),
                 fontsize=9, color="#d14836",
                 arrowprops=dict(arrowstyle="->", color="#d14836", lw=0.9))
    ax2.set_xlabel("Retained rank $\\chi$")
    ax2.set_ylabel("Relative Frobenius error")
    ax2.set_xlim(0, n)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xticks([0, storage_parity, n])
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_title("(b) Error versus retained rank")

    fig.suptitle(f"Gaussian {n}x{n} matrix, seed {meta['seed']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(ART / "svd.png", dpi=160)
    print(f"wrote {ART / 'svd.png'}")
    print("10%% accuracy needs chi = %d, i.e. %.2fx the original storage: "
          "no storage-saving low-rank approximation exists at 10%% error."
          % (chi10, storage_ratio))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
