"""Plot loopy BP residual against marginal error for the three ASIA patients."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CASES = ["prior", "AD", "ADX"]


def load(case):
    return json.loads((ROOT / "artifacts" / f"bp-{case}" / "run.json").read_text())


def main():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    for case in CASES:
        run = load(case)
        sweeps = [row["sweep"] for row in run["history"]]
        residual = [row["residual"] for row in run["history"]]
        error = [row["marginal_error"] for row in run["history"]]
        axes[0].semilogy(sweeps, residual, label=case)
        axes[1].semilogy(sweeps, error, label=case)
    axes[0].axhline(1e-10, color="#52606d", linewidth=0.8, linestyle="--")
    axes[0].set_xlabel("sweep")
    axes[0].set_ylabel("undamped message residual")
    axes[0].set_title("messages can stop")
    axes[1].set_xlabel("sweep")
    axes[1].set_ylabel("max |belief − exact marginal|")
    axes[1].set_title("the probabilities need not")
    for ax in axes:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "evidence" / "bp.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
