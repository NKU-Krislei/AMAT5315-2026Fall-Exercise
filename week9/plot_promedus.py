"""Compare Promedus contraction cost with the loopy BP trace."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def load(path):
    return json.loads((ROOT / path).read_text())


def main():
    greedy = load("runs/promedus-greedy/run.json")
    treesa = load("runs/promedus-treesa/run.json")
    bp = load("runs/promedus-bp/run.json")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))
    labels = ["greedy", "TreeSA"]
    work = [greedy["work"], treesa["work"]]
    peak = [
        greedy["largest_intermediate_entries"],
        treesa["largest_intermediate_entries"],
    ]
    positions = [0, 1]
    axes[0].bar(
        [p - 0.18 for p in positions],
        work,
        width=0.36,
        label="contraction work",
    )
    axes[0].bar(
        [p + 0.18 for p in positions],
        peak,
        width=0.36,
        label="largest intermediate",
    )
    axes[0].set_yscale("log")
    axes[0].set_xticks(positions, labels)
    axes[0].set_title("same partition, different cost")
    axes[0].legend(frameon=False)

    sweeps = [row["sweep"] for row in bp["history"]]
    axes[1].semilogy(sweeps, [row["residual"] for row in bp["history"]], label="residual")
    axes[1].semilogy(
        sweeps,
        [row["max_query_error"] for row in bp["history"]],
        label="max query error",
    )
    axes[1].axhline(1e-10, color="#52606d", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("sweep")
    axes[1].set_title("still short of the tolerance")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "evidence" / "promedus.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
