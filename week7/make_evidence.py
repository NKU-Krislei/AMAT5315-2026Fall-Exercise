"""Write the evidence figures named by the week 7 learning sheet."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from ddpm import ddpm_sample
from ddim import ddim_sample, full_levels
from mixture import CENTERS, cluster_statistics, sample_mixture
from mlp import predict_noise
from schedule import forward_chain, make_schedule
from train_diffusion import load_params

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
ARTIFACTS = ROOT / "artifacts"
FORWARD_LIMIT = 6.0
CLUSTER_LIMIT = 6.0
STILL_LEVELS = (0, 200, 400, 600, 800, 1000)
DIFFUSION_CHECKPOINTS = (0, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 25000)
FIT_STILL_UPDATES = (0, 200, 2000, 25000)
CLUSTER_COLORS = np.array(["#d95f02", "#1b9e77", "#7570b3", "#e7298a"])


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


def _ddpm_runner(schedule):
    @jax.jit
    def run(params, x_t, step_noise):
        def predict(state, level):
            return predict_noise(params, state, jnp.full((state.shape[0],), level))

        return ddpm_sample(predict, schedule, x_t, step_noise, return_trace=True)

    return run


def _draw_target_circles(ax):
    centres = np.asarray(CENTERS)
    for centre in centres:
        circle = plt.Circle(centre, 0.6, fill=False, linestyle="--", linewidth=1.0, color="0.3")
        ax.add_patch(circle)


def _format_cluster_report(rows):
    lines = []
    for row in rows:
        centre = row["centre"]
        spread = row["spread"]
        lines.append(
            f"cluster {row['index']}: fraction {row['fraction']:.4f}  "
            f"centre ({centre[0]:.4f}, {centre[1]:.4f})  "
            f"spread ({spread[0]:.4f}, {spread[1]:.4f})"
        )
    return "\n".join(lines)


def _check_cluster_acceptance(rows):
    centres = np.asarray(CENTERS)
    problems = []
    for row, centre in zip(rows, centres):
        if abs(row["fraction"] - 0.25) > 0.04:
            problems.append(f"fraction {row['fraction']:.4f}")
        if np.max(np.abs(row["centre"] - centre)) > 0.05:
            problems.append(f"centre {row['centre']}")
        if np.max(np.abs(row["spread"] - 0.30)) > 0.03:
            problems.append(f"spread {row['spread']}")
    if problems:
        raise SystemExit("DDPM samples missed the cluster tolerances:\n" + "\n".join(problems))


def make_ddpm_training_movie():
    """Ten shared DDPM paths at each saved training stage, plus the loss so far."""
    _prepare_dirs()
    schedule = make_schedule()
    run = _ddpm_runner(schedule)
    loss = np.load(ARTIFACTS / "diffusion" / "loss.npy")
    key_start, key_noise = jax.random.split(jax.random.PRNGKey(20261))
    x_t = jax.random.normal(key_start, (10, 2), dtype=jnp.float32)
    step_noise = jax.random.normal(key_noise, (1001, 10, 2), dtype=jnp.float32)
    traces = []
    for update in DIFFUSION_CHECKPOINTS:
        params = load_params(ARTIFACTS / "diffusion" / f"checkpoint_{update:05d}.npz")
        _sample, trace = run(params, x_t, step_noise)
        traces.append(np.asarray(trace))
        print(f"sampled training update {update}", flush=True)

    path_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))

    def draw(frame):
        update = DIFFUSION_CHECKPOINTS[frame]
        trace = traces[frame]
        left, right = axes
        left.clear()
        right.clear()
        for index in range(10):
            path = trace[::-1, index, :]
            left.plot(path[:, 0], path[:, 1], color=path_colors[index], linewidth=0.8, alpha=0.35)
            left.scatter(path[0, 0], path[0, 1], s=28, c="white", edgecolors="black", linewidths=0.6, zorder=3)
            left.scatter(path[-1, 0], path[-1, 1], s=46, c=[path_colors[index]], edgecolors="black", linewidths=0.4, zorder=4)
        _draw_target_circles(left)
        left.set_xlim(-CLUSTER_LIMIT, CLUSTER_LIMIT)
        left.set_ylim(-CLUSTER_LIMIT, CLUSTER_LIMIT)
        left.set_aspect("equal")
        left.set_title(f"10 DDPM paths after {update} updates")
        left.set_xlabel("x1")
        left.set_ylabel("x2")
        shown = loss[:update]
        if shown.size:
            right.plot(np.arange(1, shown.size + 1), shown, color="0.75", linewidth=0.4)
            window = 50
            if shown.size >= window:
                kernel = np.ones(window) / window
                smooth = np.convolve(shown, kernel, mode="valid")
                right.plot(np.arange(window, shown.size + 1), smooth, color="#045a8d", linewidth=1.4)
        right.axvline(update, color="#e34a33", linestyle=":", linewidth=1)
        right.set_xlim(0, loss.size)
        right.set_ylim(0, max(1.0, float(np.nanpercentile(loss, 99))))
        right.set_xlabel("update")
        right.set_ylabel("noise MSE")
        right.set_title("Training loss")

    draw(0)
    animation = FuncAnimation(fig, draw, frames=len(DIFFUSION_CHECKPOINTS))
    gif_path = EVIDENCE / "part1_ddpm_training.gif"
    animation.save(gif_path, writer=PillowWriter(fps=1))
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    for ax, update in zip(axes.ravel(), FIT_STILL_UPDATES):
        trace = traces[DIFFUSION_CHECKPOINTS.index(update)]
        for index in range(10):
            path = trace[::-1, index, :]
            ax.plot(path[:, 0], path[:, 1], color=path_colors[index], linewidth=0.8, alpha=0.35)
            ax.scatter(path[0, 0], path[0, 1], s=22, c="white", edgecolors="black", linewidths=0.5, zorder=3)
            ax.scatter(path[-1, 0], path[-1, 1], s=42, c=[path_colors[index]], edgecolors="black", linewidths=0.4, zorder=4)
        _draw_target_circles(ax)
        ax.set_xlim(-CLUSTER_LIMIT, CLUSTER_LIMIT)
        ax.set_ylim(-CLUSTER_LIMIT, CLUSTER_LIMIT)
        ax.set_aspect("equal")
        ax.set_title(f"{update} updates")
    fig.suptitle("Shared DDPM paths as the noise predictor is trained")
    fig.tight_layout()
    still_path = EVIDENCE / "part1_ddpm_training_frames.png"
    fig.savefig(still_path, dpi=140)
    plt.close(fig)
    print(f"wrote {gif_path}")
    print(f"wrote {still_path}")


def make_ddpm_samples(n_samples: int = 2000, seed: int = 20262):
    """Generate 2000 points with 1000 DDPM steps and compare them with the target."""
    _prepare_dirs()
    schedule = make_schedule()
    run = _ddpm_runner(schedule)
    params = load_params(ARTIFACTS / "diffusion" / "checkpoint_25000.npz")
    key_start, key_noise, key_target = jax.random.split(jax.random.PRNGKey(seed), 3)
    x_t = jax.random.normal(key_start, (n_samples, 2), dtype=jnp.float32)
    step_noise = jax.random.normal(key_noise, (1001, n_samples, 2), dtype=jnp.float32)
    sample, trace = run(params, x_t, step_noise)
    trace = np.asarray(trace)
    generated = np.asarray(sample)
    _labels, rows = cluster_statistics(generated)
    report = _format_cluster_report(rows)
    print(report)
    (ARTIFACTS / "part1_ddpm_metrics.json").write_text(
        json.dumps(
            [
                {
                    "index": row["index"],
                    "fraction": row["fraction"],
                    "centre": row["centre"].tolist(),
                    "spread": row["spread"].tolist(),
                }
                for row in rows
            ],
            indent=2,
        )
    )
    (EVIDENCE / "part1_ddpm_metrics.txt").write_text(report + "\n")

    target = np.asarray(sample_mixture(key_target, n_samples))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.2), sharex=True, sharey=True)
    axes[0].scatter(generated[:, 0], generated[:, 1], s=4, c="0.2", linewidths=0, alpha=0.7)
    axes[1].scatter(target[:, 0], target[:, 1], s=4, c="0.2", linewidths=0, alpha=0.7)
    for ax, title in zip(axes, ("DDPM samples", "Exact target samples")):
        _draw_target_circles(ax)
        ax.set_xlim(-CLUSTER_LIMIT, CLUSTER_LIMIT)
        ax.set_ylim(-CLUSTER_LIMIT, CLUSTER_LIMIT)
        ax.set_aspect("equal")
        ax.set_title(title)
    fig.suptitle("2000 samples, 1000 DDPM steps")
    fig.tight_layout()
    compare_path = EVIDENCE / "part1_ddpm_samples.png"
    fig.savefig(compare_path, dpi=140)
    plt.close(fig)

    frame_levels = list(range(1000, -1, -40))
    fig, ax = plt.subplots(figsize=(4.2, 4.2))

    def draw(frame):
        level = frame_levels[frame]
        cloud = trace[level]
        ax.clear()
        ax.scatter(cloud[:, 0], cloud[:, 1], s=4, c="0.25", linewidths=0, alpha=0.7)
        _draw_target_circles(ax)
        ax.set_xlim(-CLUSTER_LIMIT, CLUSTER_LIMIT)
        ax.set_ylim(-CLUSTER_LIMIT, CLUSTER_LIMIT)
        ax.set_aspect("equal")
        ax.set_title(f"DDPM reverse  t = {level}")

    draw(0)
    animation = FuncAnimation(fig, draw, frames=len(frame_levels))
    gif_path = EVIDENCE / "part1_ddpm_reverse.gif"
    animation.save(gif_path, writer=PillowWriter(fps=4))
    plt.close(fig)
    print(f"wrote {compare_path}")
    print(f"wrote {gif_path}")
    _check_cluster_acceptance(rows)


def _ddim_runner(schedule):
    levels = full_levels(int(schedule.betas.shape[0]))

    @jax.jit
    def run(params, x_t):
        def predict(state, level):
            return predict_noise(params, state, level)

        return ddim_sample(predict, schedule, x_t, levels, return_trace=True)

    return run


def _widths_ok(rows, low=0.29, high=0.32):
    return all(low <= value <= high for row in rows for value in row["spread"])


def make_ddim_comparison(n_samples: int = 2000, seed: int = 20271, n_paths: int = 8):
    """Compare two DDIM runs with two DDPM runs that share the same starts."""
    _prepare_dirs()
    schedule = make_schedule()
    params = load_params(ARTIFACTS / "diffusion" / "checkpoint_25000.npz")
    ddim_run = _ddim_runner(schedule)
    ddpm_run = _ddpm_runner(schedule)
    key_start, key_noise_a, key_noise_b, key_target = jax.random.split(jax.random.PRNGKey(seed), 4)
    x_t = jax.random.normal(key_start, (n_samples, 2), dtype=jnp.float32)
    noise_a = jax.random.normal(key_noise_a, (1001, n_samples, 2), dtype=jnp.float32)
    noise_b = jax.random.normal(key_noise_b, (1001, n_samples, 2), dtype=jnp.float32)

    ddim_a, trace_ddim_a = ddim_run(params, x_t)
    ddim_b, _trace_ddim_b = ddim_run(params, x_t)
    ddpm_a, trace_ddpm_a = ddpm_run(params, x_t, noise_a)
    ddpm_b, trace_ddpm_b = ddpm_run(params, x_t, noise_b)
    ddim_a = np.asarray(ddim_a)
    ddim_b = np.asarray(ddim_b)
    ddpm_a = np.asarray(ddpm_a)
    ddpm_b = np.asarray(ddpm_b)
    difference = float(np.max(np.abs(ddim_a - ddim_b)))
    print(f"largest difference between two DDIM runs: {difference}")

    _labels_ddim, rows_ddim = cluster_statistics(ddim_a)
    labels_ddpm_a, rows_ddpm_a = cluster_statistics(ddpm_a)
    labels_ddpm_b, rows_ddpm_b = cluster_statistics(ddpm_b)
    same_cluster = float(np.mean(labels_ddpm_a == labels_ddpm_b))
    print(f"DDPM runs ending in the same cluster: {same_cluster:.4f}")
    print("DDIM")
    print(_format_cluster_report(rows_ddim))
    print("DDPM A")
    print(_format_cluster_report(rows_ddpm_a))
    print("DDPM B")
    print(_format_cluster_report(rows_ddpm_b))

    target = np.asarray(sample_mixture(key_target, n_samples))
    path_colors = plt.cm.tab10(np.linspace(0, 1, n_paths))
    traces = {
        "DDIM": np.asarray(trace_ddim_a),
        "DDPM noise 1": np.asarray(trace_ddpm_a),
        "DDPM noise 2": np.asarray(trace_ddpm_b),
    }
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharex=True, sharey=True)
    for ax, (title, trace) in zip(axes, traces.items()):
        for index in range(n_paths):
            path = trace[:, index, :]
            if title.startswith("DDPM"):
                path = trace[::-1, index, :]
            ax.plot(path[:, 0], path[:, 1], color=path_colors[index], linewidth=0.9, alpha=0.8)
            ax.scatter(path[-1, 0], path[-1, 1], s=18, c=[path_colors[index]], zorder=3)
        _draw_target_circles(ax)
        ax.set_xlim(-CLUSTER_LIMIT, CLUSTER_LIMIT)
        ax.set_ylim(-CLUSTER_LIMIT, CLUSTER_LIMIT)
        ax.set_aspect("equal")
        ax.set_title(title)
    fig.suptitle("Same starts. DDIM has one path; DDPM changes with the reverse noise.")
    fig.tight_layout()
    trajectory_path = EVIDENCE / "part2_trajectories.png"
    fig.savefig(trajectory_path, dpi=140)
    plt.close(fig)

    clouds = (
        ("DDIM", ddim_a),
        ("DDPM noise 1", ddpm_a),
        ("DDPM noise 2", ddpm_b),
        ("Exact target", target),
    )
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.4), sharex=True, sharey=True)
    for ax, (title, cloud) in zip(axes, clouds):
        ax.scatter(cloud[:, 0], cloud[:, 1], s=3, c="0.2", linewidths=0, alpha=0.7)
        _draw_target_circles(ax)
        ax.set_xlim(-CLUSTER_LIMIT, CLUSTER_LIMIT)
        ax.set_ylim(-CLUSTER_LIMIT, CLUSTER_LIMIT)
        ax.set_aspect("equal")
        ax.set_title(title)
    fig.suptitle(f"2000 samples. Two DDIM runs differ by at most {difference:.1e}.")
    fig.tight_layout()
    sample_path = EVIDENCE / "part2_samples.png"
    fig.savefig(sample_path, dpi=140)
    plt.close(fig)

    reference_width = _widths_ok(rows_ddim) and _widths_ok(rows_ddpm_a) and _widths_ok(rows_ddpm_b)
    report = "\n".join(
        [
            f"largest DDIM difference: {difference}",
            f"DDPM same-cluster fraction: {same_cluster:.4f}",
            f"reference width band [0.29, 0.32] met: {reference_width}",
            "DDIM",
            _format_cluster_report(rows_ddim),
            "DDPM A",
            _format_cluster_report(rows_ddpm_a),
            "DDPM B",
            _format_cluster_report(rows_ddpm_b),
        ]
    )
    (EVIDENCE / "part2_metrics.txt").write_text(report + "\n")
    print(f"wrote {trajectory_path}")
    print(f"wrote {sample_path}")
    if difference != 0.0:
        raise SystemExit("DDIM runs did not agree exactly")
    # "Within 0.015" includes the boundary. Float division can land 1 ulp over it.
    if any(abs(row["fraction"] - 0.25) > 0.015 + 1e-6 for row in rows_ddim):
        raise SystemExit("DDIM cluster fractions are outside 0.015 of 0.25")
    for name, rows in (("DDIM", rows_ddim), ("DDPM A", rows_ddpm_a), ("DDPM B", rows_ddpm_b)):
        if any(np.max(np.abs(row["spread"] - 0.30)) > 0.03 for row in rows):
            raise SystemExit(f"{name} widths are more than 0.03 from 0.30")
    print(f"reference width band [0.29, 0.32] met: {reference_width}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["forward", "ddpm-fit", "ddpm-samples", "ddim"])
    args = parser.parse_args()
    if args.command == "forward":
        make_forward_movie()
    elif args.command == "ddpm-fit":
        make_ddpm_training_movie()
    elif args.command == "ddpm-samples":
        make_ddpm_samples()
    elif args.command == "ddim":
        make_ddim_comparison()


if __name__ == "__main__":
    main()
