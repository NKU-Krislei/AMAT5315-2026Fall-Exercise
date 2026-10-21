"""Write the evidence figures named by the week 7 learning sheet."""

import argparse
import json
from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from mixture import sample_mixture
from schedule import forward_chain, make_schedule

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
ARTIFACTS = ROOT / "artifacts"
FORWARD_LIMIT = 6.0
STILL_LEVELS = (0, 200, 400, 600, 800, 1000)


def _prepare_dirs():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)


def make_forward_movie(n_samples: int = 2000, seed: int = 2026):
    """Follow one batch of mixture samples through forward diffusion."""
    _prepare_dirs()
    schedule = make_schedule()
    key_data, key_chain = jax.random.split(jax.random.PRNGKey(seed))
    x0, components = sample_mixture(key_data, n_samples, return_components=True)
    states = np.asarray(forward_chain(key_chain, x0, schedule))
    labels = np.asarray(components)
    final = states[-1]
    mean = final.mean(axis=0)
    var = final.var(axis=0)
    stats = {
        "n_samples": n_samples,
        "seed": seed,
        "t": 1000,
        "mean": mean.tolist(),
        "variance": var.tolist(),
    }
    print(f"t=1000 coordinate means: {mean[0]:.6f} {mean[1]:.6f}")
    print(f"t=1000 coordinate variances: {var[0]:.6f} {var[1]:.6f}")
    (ARTIFACTS / "part1_forward_stats.json").write_text(json.dumps(stats, indent=2))

    colors = np.array(["#d95f02", "#1b9e77", "#7570b3", "#e7298a"])[labels]
    frame_levels = list(range(0, 1001, 40))

    fig, ax = plt.subplots(figsize=(4.2, 4.2))

    def draw(level):
        ax.clear()
        cloud = states[level]
        ax.scatter(cloud[:, 0], cloud[:, 1], s=4, c=colors, linewidths=0, alpha=0.8)
        ax.set_xlim(-FORWARD_LIMIT, FORWARD_LIMIT)
        ax.set_ylim(-FORWARD_LIMIT, FORWARD_LIMIT)
        ax.set_aspect("equal")
        ax.set_title(f"Forward diffusion  t = {level}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

    draw(0)
    animation = FuncAnimation(fig, lambda frame: draw(frame_levels[frame]), frames=len(frame_levels))
    gif_path = EVIDENCE / "part1_forward.gif"
    animation.save(gif_path, writer=PillowWriter(fps=4))
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(9, 6), sharex=True, sharey=True)
    for ax, level in zip(axes.ravel(), STILL_LEVELS):
        cloud = states[level]
        ax.scatter(cloud[:, 0], cloud[:, 1], s=3, c=colors, linewidths=0, alpha=0.75)
        ax.set_xlim(-FORWARD_LIMIT, FORWARD_LIMIT)
        ax.set_ylim(-FORWARD_LIMIT, FORWARD_LIMIT)
        ax.set_aspect("equal")
        ax.set_title(f"t = {level}")
    fig.suptitle("Forward diffusion of the same 2000 mixture samples")
    fig.tight_layout()
    still_path = EVIDENCE / "part1_forward_frames.png"
    fig.savefig(still_path, dpi=140)
    plt.close(fig)
    print(f"wrote {gif_path}")
    print(f"wrote {still_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["forward"])
    args = parser.parse_args()
    if args.command == "forward":
        make_forward_movie()


if __name__ == "__main__":
    main()
