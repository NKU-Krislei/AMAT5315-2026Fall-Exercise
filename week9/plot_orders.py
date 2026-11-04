"""Draw the saved AD contraction trees and their costs."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
NAMES = "ATSLBEXD"


def load(directory):
    plan = json.loads((ROOT / directory / "order.json").read_text())
    cost = json.loads((ROOT / directory / "complexity.json").read_text())
    return plan, cost


def leaf_name(plan, index):
    letters = "".join(NAMES[label] for label in plan["inputs"][index])
    if index >= 8:
        return f"[{letters}=0]"
    return letters


def walk(node, plan, steps, cursor):
    if node["kind"] == "leaf":
        return {"label": leaf_name(plan, node["tensor"]), "children": [], "entries": None}, cursor
    left, cursor = walk(node["left"], plan, steps, cursor)
    right, cursor = walk(node["right"], plan, steps, cursor)
    step = steps[cursor]
    return {
        "label": ",".join(NAMES[label] for label in step["retained"]) or "scalar",
        "entries": step["entries"],
        "children": [left, right],
    }, cursor + 1


def layout(node, depth, left, positions, edges):
    if not node["children"]:
        positions.append((left, -depth, node))
        return left + 1
    start = left
    for child in node["children"]:
        left = layout(child, depth + 1, left, positions, edges)
    center = (start + left - 1) / 2
    positions.append((center, -depth, node))
    child_positions = [item for item in positions if item[2] in node["children"]]
    for x, y, _child in child_positions:
        edges.append((center, -depth, x, y))
    return left


def draw_tree(ax, plan, cost, title):
    drawn, cursor = walk(plan["tree"], plan, cost["steps"], 0)
    assert cursor == len(cost["steps"])
    positions = []
    edges = []
    layout(drawn, 0, 0, positions, edges)
    largest = cost["largest_intermediate"]["entries"]
    for x0, y0, x1, y1 in edges:
        ax.plot([x0, x1], [y0, y1], color="#52606d", linewidth=1)
    for x, y, node in positions:
        if node["entries"] is None:
            color = "#f7f4ea"
        elif node["entries"] == largest:
            color = "#f3c98b"
        else:
            color = "#e7eef5"
        ax.scatter([x], [y], s=220, c=color, edgecolors="#1f2933", zorder=3)
        text = node["label"] if node["entries"] is None else f"{node['label']} ({node['entries']})"
        ax.text(x, y - 0.28, text, ha="center", va="top", fontsize=7, zorder=4)
    ax.set_title(title)
    ax.axis("off")
    ax.margins(x=0.08, y=0.12)


def main():
    random_plan, random_cost = load(Path("runs/random"))
    greedy_plan, greedy_cost = load(Path("runs/greedy"))
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 8.0))
    draw_tree(axes[0], random_plan, random_cost, caption("random, seed 42", random_cost))
    draw_tree(axes[1], greedy_plan, greedy_cost, caption("omeco GreedyMethod", greedy_cost))
    fig.suptitle("AD evidence contraction: same probability, different cost")
    fig.tight_layout()
    fig.savefig(ROOT / "evidence" / "orders.png", dpi=160)
    plt.close(fig)


def caption(title, cost):
    intermediate = cost["largest_intermediate"]
    labels = ",".join(NAMES[label] for label in intermediate["retained"]) or "scalar"
    return (
        f"{title}\nwork {cost['work']:,}    largest tensor {cost['largest_tensor']:,}"
        f"\nlargest intermediate {intermediate['entries']:,} ({labels})"
    )


if __name__ == "__main__":
    main()
