"""Plot the integrality gap benchmark from artifacts/gap/run.json."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts" / "gap" / "run.json"
OUT = ROOT / "artifacts" / "gap.png"


def main():
    document = json.loads(RUN.read_text())
    summary = document["summary"]
    instances = document["evidence"]["instances"]
    ids = [item["id"] for item in instances]
    lp = [item["lp"] for item in instances]
    exact = [item["exact"] for item in instances]
    gap = [item["gap"] for item in instances]
    excess = [item["excess"] for item in instances]

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot(ids, lp, "o-", label="LP lower bound")
    axes[0].plot(ids, exact, "s-", label="exact ground state")
    axes[0].set_xlabel("instance")
    axes[0].set_ylabel("energy")
    axes[0].set_title("LP bound and ground-state energy")
    axes[0].legend()
    axes[1].plot(ids, gap, "o-", label="integrality gap")
    axes[1].plot(ids, excess, "s-", label="rounding excess")
    axes[1].set_xlabel("instance")
    axes[1].set_ylabel("energy")
    axes[1].set_title("Gap and naive rounding excess")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")
    print(
        f"square LP {summary['square_lp']:.6f} at {summary['square_lp_x']}, "
        f"exact {summary['square_exact']:.6f}, rounded {summary['square_rounded']:.6f}"
    )
    print(
        f"diet x {summary['diet_x']} cost {summary['diet_cost']:.6f}"
    )
    print(
        "means: "
        f"LP {summary['mean_lp']:.6f}, "
        f"exact {summary['mean_exact']:.6f}, "
        f"gap {summary['mean_gap']:.6f}, "
        f"fractional {summary['mean_fractional_spins']:.6f}, "
        f"suboptimal roundings {summary['suboptimal_roundings']}, "
        f"rounding excess {summary['mean_rounding_excess']:.6f}"
    )
    instance_11 = next(item for item in instances if item["id"] == 11)
    print(f"instance 11 rounding excess {instance_11['excess']:.6f}")


if __name__ == "__main__":
    main()
