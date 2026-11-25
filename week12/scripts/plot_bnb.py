"""Plot branch-and-bound node and LP counts from the two saved policies."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MOST = ROOT / "artifacts" / "bnb_most" / "run.json"
STRONG = ROOT / "artifacts" / "bnb_strong" / "run.json"
OUT = ROOT / "artifacts" / "bnb.png"


def load(path):
    document = json.loads(path.read_text())
    instances = document["evidence"]["instances"]
    return document, instances


def print_policy(name, document):
    summary = document["summary"]
    print(
        f"{name}: mean nodes {summary['mean_nodes']:.4f}, "
        f"LP solves {summary['mean_lp_solves']:.4f}, "
        f"time {summary['mean_time_ms']:.4f} ms, "
        f"max |error| {summary['max_abs_error']:.3e}"
    )


def main():
    most_doc, most = load(MOST)
    strong_doc, strong = load(STRONG)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    series = (
        ("most-frac", most, "C0"),
        ("strong", strong, "C1"),
    )
    for name, instances, color in series:
        axes[0].plot(
            range(len(instances)),
            sorted(item["nodes"] for item in instances),
            "o-",
            color=color,
            label=name,
        )
        axes[1].plot(
            range(len(instances)),
            sorted(item["lp_solves"] for item in instances),
            "o-",
            color=color,
            label=name,
        )
    axes[0].set_title("Explored nodes, sorted")
    axes[1].set_title("LP solves, sorted")
    for axis, ylabel in zip(axes, ("nodes", "LP solves")):
        axis.set_xlabel("instance rank")
        axis.set_ylabel(ylabel)
        axis.legend()
    figure.tight_layout()
    figure.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")
    print_policy("most-frac", most_doc)
    print_policy("strong", strong_doc)
    print("tetrahedron most-frac trace:")
    for node in most_doc["evidence"]["tetrahedron"]:
        lp = "none" if node["lp"] is None else f"{node['lp']:.6f}"
        print(
            f"  node {node['node']} branch {node['branch']} "
            f"status {node['status']} lp {lp} x {node['x']}"
        )


if __name__ == "__main__":
    main()
