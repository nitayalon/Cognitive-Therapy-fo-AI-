#!/usr/bin/env python
"""
Render a state-diagram visualization for each trained agent's minimized FSM.

Reads fidelity_train.json::fsm_structure from each
experiments/fsm_train_*/task_*/opp_*/ directory and renders a directed
state-transition diagram. Edges are labeled "<input symbol>/<output action>"
(input symbols: START, CC, CD, DC, DD; output actions: C=cooperate, D=defect).
The initial state is highlighted in green.

For each agent, writes:
  - a PNG rendered with networkx + matplotlib
  - a Graphviz .dot file (for higher-quality rendering elsewhere, e.g.
    `dot -Tpng file.dot -o file.png`, if Graphviz is available)

Output layout:
  <output-dir>/h<hidden_size>/<game_abbr>_opp<opponent>_seed<seed>.png
  <output-dir>/h<hidden_size>/<game_abbr>_opp<opponent>_seed<seed>.dot

Usage:
    python analysis/fsm_visualize_fsms.py
    python analysis/fsm_visualize_fsms.py --hidden-size 2 --game prisoners-dilemma
    python analysis/fsm_visualize_fsms.py --opponent 0.1 --seed 42
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

GAME_ABBR = {
    "prisoners-dilemma": "PD",
    "stag-hunt": "SH",
    "hawk-dove": "HD",
}

ACTION_ABBR = {"COOPERATE": "C", "DEFECT": "D"}


def find_runs(experiments_root: Path, hidden_size=None, game=None, opponent=None, seed=None):
    paths = sorted(experiments_root.glob("fsm_train_*/task_*/opp_*/fidelity_train.json"))
    runs = []
    for fid_path in paths:
        tm_path = fid_path.parent / "train_metrics.json"
        with open(tm_path) as f:
            tm = json.load(f)
        if hidden_size is not None and tm["hidden_size"] != hidden_size:
            continue
        if game is not None and tm["game"] != game:
            continue
        if opponent is not None and abs(tm["opponent_coop"] - opponent) > 1e-9:
            continue
        if seed is not None and tm["seed"] != seed:
            continue
        runs.append((fid_path, tm))
    return runs


def build_graph(fsm_structure):
    g = nx.DiGraph()
    for s in fsm_structure["states"]:
        g.add_node(s)

    edge_labels = {}  # (src, dst) -> list of "symbol/output" strings
    for src_str, transitions in fsm_structure["transitions"].items():
        src = int(src_str)
        for symbol, target in transitions.items():
            if target is None:
                continue
            dst, output = target
            label = f"{symbol}/{ACTION_ABBR.get(output, output)}"
            edge_labels.setdefault((src, dst), []).append(label)

    for (src, dst), labels in edge_labels.items():
        g.add_edge(src, dst, label=", ".join(labels))

    return g


def draw_fsm_png(fsm_structure, title, output_path: Path):
    g = build_graph(fsm_structure)
    initial_state = fsm_structure["initial_state"]
    n = g.number_of_nodes()
    node_size = 1000

    if n <= 1:
        pos = {node: np.array([0.0, 0.0]) for node in g.nodes()}
        scale = 1.0
    else:
        scale = max(1.0, n / 4.0)
        pos = nx.circular_layout(g, scale=scale)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=120)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    margin = scale * 0.7 + 0.6
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_aspect("equal")
    ax.axis("off")

    node_colors = ["#90EE90" if node == initial_state else "#ADD8E6" for node in g.nodes()]
    nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=node_size,
                            edgecolors="black", ax=ax)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=11, font_weight="bold")

    self_loops = [(u, d) for u, v, d in g.edges(data=True) if u == v]
    other_edges = [(u, v) for u, v, d in g.edges(data=True) if u != v]
    other_edge_labels = {(u, v): d["label"] for u, v, d in g.edges(data=True) if u != v}

    nx.draw_networkx_edges(
        g, pos, edgelist=other_edges, ax=ax, connectionstyle="arc3,rad=0.15",
        arrowsize=18, node_size=node_size, min_source_margin=15, min_target_margin=15,
    )
    nx.draw_networkx_edge_labels(
        g, pos, edge_labels=other_edge_labels, ax=ax, font_size=7,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
    )

    # Self-loops: draw a small circle tangent to the node, offset radially
    # outward from the layout center, with its label further out still, so
    # neither overlaps the node's own label.
    fig.canvas.draw()
    node_radius_pts = (node_size / np.pi) ** 0.5
    r_px = node_radius_pts / 72.0 * fig.dpi
    p0_data = ax.transData.inverted().transform((0, 0))
    p1_data = ax.transData.inverted().transform((r_px, 0))
    node_radius_data = abs(p1_data[0] - p0_data[0])

    center = np.mean(np.array(list(pos.values())), axis=0) if n > 1 else np.array([0.0, 0.0])
    loop_r = node_radius_data * 0.7
    for u, d in self_loops:
        p = np.array(pos[u])
        direction = p - center
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-6 else np.array([0.0, 1.0])
        loop_center = p + direction * (node_radius_data + loop_r * 0.9)
        ax.add_patch(mpatches.Circle(loop_center, loop_r, fill=False,
                                      edgecolor="black", lw=1.2, zorder=1))
        label_pos = p + direction * (node_radius_data + 2 * loop_r + 0.08 * scale)
        ax.text(label_pos[0], label_pos[1], d["label"], fontsize=7, ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8), zorder=3)

    ax.set_title(title, fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def write_fsm_dot(fsm_structure, title, output_path: Path):
    g = build_graph(fsm_structure)
    initial_state = fsm_structure["initial_state"]

    dot_title = title.replace('"', '\\"').replace("\n", "\\n")
    lines = ["digraph FSM {", f'  label="{dot_title}"; labelloc=t; fontsize=10;', "  rankdir=LR;"]
    for n in g.nodes():
        shape = "doublecircle" if n == initial_state else "circle"
        lines.append(f'  {n} [shape={shape}];')
    for src, dst, data in g.edges(data=True):
        label = data["label"].replace('"', '\\"')
        lines.append(f'  {src} -> {dst} [label="{label}"];')
    lines.append("}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", type=str, default="experiments")
    parser.add_argument("--output-dir", type=str, default="results/fsm_representation/fsm_diagrams")
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--game", type=str, default=None,
                         choices=["prisoners-dilemma", "stag-hunt", "hawk-dove"])
    parser.add_argument("--opponent", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-dot", action="store_true", help="Skip writing .dot files")
    args = parser.parse_args()

    experiments_root = Path(args.experiments_root)
    output_dir = Path(args.output_dir)

    runs = find_runs(experiments_root, args.hidden_size, args.game, args.opponent, args.seed)
    print(f"Found {len(runs)} matching training runs")

    for i, (fid_path, tm) in enumerate(runs, 1):
        with open(fid_path) as f:
            fid = json.load(f)

        hidden_size = tm["hidden_size"]
        game_abbr = GAME_ABBR[tm["game"]]
        opponent = tm["opponent_coop"]
        seed = tm["seed"]
        n_states = fid["minimized_states"]
        fidelity = fid["fidelity_train"]

        title = (f"H={hidden_size} {game_abbr} opp={opponent} seed={seed}\n"
                 f"states={n_states} fidelity={fidelity:.3f}")
        stem = f"{game_abbr}_opp{opponent}_seed{seed}"
        run_dir = output_dir / f"h{hidden_size}"

        draw_fsm_png(fid["fsm_structure"], title, run_dir / f"{stem}.png")
        if not args.no_dot:
            write_fsm_dot(fid["fsm_structure"], title, run_dir / f"{stem}.dot")

        if i % 50 == 0 or i == len(runs):
            print(f"  [{i}/{len(runs)}] {run_dir / stem}")

    print(f"\nDone. Diagrams written under {output_dir}/")


if __name__ == "__main__":
    main()
