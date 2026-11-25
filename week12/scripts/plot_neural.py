"""Plot imitation-learning loss and explored nodes on unseen 22-spin glasses."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "artifacts" / "model.json"
EVAL = ROOT / "artifacts" / "eval_test" / "run.json"
OUT = ROOT / "artifacts" / "neural_bnb.png"


def main():
    model = json.loads(MODEL.read_text())
    evaluation = json.loads(EVAL.read_text())
    history = model["history"]
    epochs = range(1, len(history["train_ce"]) + 1)
    policies = evaluation["evidence"]["policies"]

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot(epochs, history["train_ce"], label="train")
    axes[0].plot(epochs, history["val_ce"], label="validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy")
    axes[0].set_title("Imitation loss")
    axes[0].legend()
    for name in ("most-frac", "neural", "strong"):
        nodes = sorted(item["nodes"] for item in policies[name]["instances"])
        axes[1].plot(range(len(nodes)), nodes, "o-", label=name)
    axes[1].set_xlabel("instance rank")
    axes[1].set_ylabel("explored nodes")
    axes[1].set_title("Unseen 22-spin trees, sorted")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(OUT, dpi=160)

    training = model["training"]
    summary = evaluation["summary"]
    print(f"wrote {OUT}")
    print(
        f"nodes train/val {training['train_nodes']}/{training['validation_nodes']} "
        f"instances {len(training['train_instances'])}/{len(training['validation_instances'])} "
        f"overlap {training['instance_overlap']}"
    )
    print(
        f"best checkpoint epoch {training['checkpoint_epoch']} "
        f"validation cross-entropy {training['checkpoint_val_ce']:.6f}"
    )
    print(f"largest JAX probe probability difference {summary['max_probe_diff']:.3e}")
    print(f"largest HiGHS error {summary['max_abs_error']:.3e}")
    for name in ("most-frac", "neural", "strong"):
        item = summary["policies"][name]
        print(
            f"{name}: mean nodes {item['mean_nodes']:.4f}, "
            f"time {item['mean_time_ms']:.4f} ms, "
            f"LP solves equal nodes {item['lp_equals_nodes']}"
        )
    saving = summary["best_saving"]
    print(
        f"largest neural saving is instance {saving['instance']} "
        f"energy {saving['energy']}, nodes saved {saving['nodes_saved']}"
    )
    print(f"  most-frac splits {saving['most_frac']['splits']} "
          f"nodes {saving['most_frac']['nodes']} time {saving['most_frac']['time_ms']}")
    print(f"  neural splits {saving['neural']['splits']} "
          f"nodes {saving['neural']['nodes']} time {saving['neural']['time_ms']}")


if __name__ == "__main__":
    main()
