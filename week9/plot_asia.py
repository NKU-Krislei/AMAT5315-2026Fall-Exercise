"""Plot ASIA disease posteriors and the T,L dependence."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
CASES = ["prior", "AD", "ADX"]
DISEASES = ["T", "L", "B"]
DISEASE_NAMES = {"T": "tuberculosis", "L": "lung cancer", "B": "bronchitis"}


def load(case):
    return json.loads((ROOT / "artifacts" / case / "run.json").read_text())


def posteriors(runs):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(CASES))
    width = 0.12
    colors = {"T": "#b42318", "L": "#7a3e9d", "B": "#b86e00"}
    for index, disease in enumerate(DISEASES):
        enumeration = [run["node_marginals"]["enumeration"][disease][0] for run in runs]
        contraction = [run["node_marginals"]["contraction"][disease][0] for run in runs]
        offset = (index - 1) * 2.2 * width
        ax.bar(
            x + offset,
            enumeration,
            width,
            color=colors[disease],
            label=f"{DISEASE_NAMES[disease]} enumeration",
        )
        ax.bar(
            x + offset + width,
            contraction,
            width,
            color="none",
            edgecolor=colors[disease],
            hatch="///",
            label=f"{DISEASE_NAMES[disease]} contraction",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["prior", "visit and dyspnoea", "plus positive X-ray"])
    ax.set_ylabel("probability of yes")
    ax.set_ylim(0, 1)
    ax.set_title("ASIA disease probabilities")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "evidence" / "posteriors.png", dpi=160)
    plt.close(fig)


def joints(runs):
    fig, axes = plt.subplots(len(CASES), 3, figsize=(8.2, 7.2))
    titles = ["joint p(T, L)", "product of marginals", "joint minus product"]
    for row, (case, run) in enumerate(zip(CASES, runs)):
        joint = np.array(run["joint_TL"]["enumeration"])
        product = np.array(run["joint_TL"]["product_of_marginals_enumeration"])
        difference = np.array(run["joint_TL"]["dependence_enumeration"])
        panels = [joint, product, difference]
        limit = max(joint.max(), product.max())
        for col, (panel, title) in enumerate(zip(panels, titles)):
            ax = axes[row, col]
            if col < 2:
                ax.imshow(panel, vmin=0, vmax=limit, cmap="YlOrBr")
            else:
                span = max(abs(difference.min()), abs(difference.max()), 1e-6)
                ax.imshow(panel, vmin=-span, vmax=span, cmap="coolwarm")
            ax.set_xticks([0, 1], ["L=0", "L=1"])
            ax.set_yticks([0, 1], ["T=0", "T=1"])
            for t in range(2):
                for lung in range(2):
                    ax.text(
                        lung,
                        t,
                        f"{panel[t, lung]:.4f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="black",
                    )
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(case)
    fig.suptitle("T and L stay dependent once the X-ray is observed")
    fig.tight_layout()
    fig.savefig(ROOT / "evidence" / "joint.png", dpi=160)
    plt.close(fig)


def main():
    runs = [load(case) for case in CASES]
    posteriors(runs)
    joints(runs)


if __name__ == "__main__":
    main()
