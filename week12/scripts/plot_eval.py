"""Plot size generalization and the 30-spin time breakdown."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / "artifacts" / "eval_test" / "run.json"
LARGE = ROOT / "artifacts" / "eval_transfer" / "run.json"
OUT = ROOT / "artifacts" / "eval.png"
RULES = ("most-frac", "strong", "neural", "root-neural")


def load(path):
    return json.loads(path.read_text())


def means(document, key):
    return [document["summary"]["policies"][name][key] for name in RULES]


def print_run(title, document):
    print(title)
    print(f"  largest HiGHS error {document['summary']['max_abs_error']:.3e}")
    print(
        "  rule\tmean_nodes\twall_ms\tlp_ms\tinfer_ms\tlp_equals_nodes"
    )
    for name in RULES:
        item = document["summary"]["policies"][name]
        print(
            f"  {name}\t{item['mean_nodes']:.4f}\t{item['mean_time_ms']:.4f}\t"
            f"{item['mean_lp_ms']:.4f}\t{item['mean_infer_ms']:.4f}\t"
            f"{item['lp_equals_nodes']}"
        )


def main():
    small = load(SMALL)
    large = load(LARGE)
    labels = ["most-frac", "strong", "neural", "root-neural"]
    positions = list(range(len(labels)))
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    width = 0.36
    axes[0].bar(
        [p - width / 2 for p in positions],
        means(small, "mean_nodes"),
        width=width,
        label="22 spins",
    )
    axes[0].bar(
        [p + width / 2 for p in positions],
        means(large, "mean_nodes"),
        width=width,
        label="30 spins",
    )
    axes[0].set_xticks(positions, labels, rotation=15)
    axes[0].set_ylabel("mean explored nodes")
    axes[0].set_title("Size generalization")
    axes[0].legend()

    infer = means(large, "mean_infer_ms")
    wall = means(large, "mean_time_ms")
    axes[1].bar(labels, wall, color="C0", label="total wall time")
    axes[1].set_ylabel("mean wall time (ms)")
    axes[1].set_title("30-spin time")
    axes[1].tick_params(axis="x", rotation=15)
    inference_axis = axes[1].twinx()
    inference_axis.plot(
        labels,
        infer,
        "o-",
        color="C1",
        label="features and inference",
    )
    inference_axis.set_ylabel("feature and inference time (ms)")
    handles, legend_labels = axes[1].get_legend_handles_labels()
    extra_handles, extra_labels = inference_axis.get_legend_handles_labels()
    axes[1].legend(
        handles + extra_handles,
        legend_labels + extra_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    figure.tight_layout()
    figure.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"wrote {OUT}")
    print_run("22-spin unseen", small)
    print_run("30-spin transfer", large)


if __name__ == "__main__":
    main()
