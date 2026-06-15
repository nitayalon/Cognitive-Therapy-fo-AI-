#!/usr/bin/env python
"""
Render a state-diagram visualization for each trained agent's minimized FSM.

Reads fidelity_train.json::fsm_structure from each
experiments/fsm_train_*/task_*/opp_*/ directory and renders a directed
state-transition diagram. Edges are labeled "<input symbol>/<output action>",
with the input symbol drawn in blue and the output action drawn in red
(input symbols: START, CC, CD, DC, DD; output actions: C=cooperate,
D=defect). The initial state is highlighted in green.

The "START" symbol only ever occurs at t=0 and is never re-visited, so any
(state, "START") transition is drawn separately: as an edge from a dedicated
"START" node (gray, no incoming edges, drawn outside the main state ring)
into the corresponding next state, rather than mixed in with that state's
other outgoing transitions.

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
from matplotlib.offsetbox import TextArea, HPacker, VPacker, AnnotationBbox
from matplotlib.path import Path as MplPath

GAME_ABBR = {
    "prisoners-dilemma": "PD",
    "stag-hunt": "SH",
    "hawk-dove": "HD",
}

ACTION_ABBR = {"COOPERATE": "C", "DEFECT": "D"}

START_NODE = "START"


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
    """Build a DiGraph for the FSM.

    Each edge carries a "pairs" attribute: a list of (input_symbol,
    output_action) tuples for that (source, destination) pair. Transitions
    on the "START" symbol are routed through a synthetic START_NODE instead
    of their nominal source state, since START is only ever the input symbol
    at t=0 and that (state, "START") combination is never revisited.
    """
    g = nx.DiGraph()
    for s in fsm_structure["states"]:
        g.add_node(s)

    edge_pairs = {}   # (src, dst) -> list of (symbol, action)
    start_pairs = {}  # dst -> list of (symbol, action)

    for src_str, transitions in fsm_structure["transitions"].items():
        src = int(src_str)
        for symbol, target in transitions.items():
            if target is None:
                continue
            dst, output = target
            action = ACTION_ABBR.get(output, output)
            if symbol == "START":
                start_pairs.setdefault(dst, []).append((symbol, action))
            else:
                edge_pairs.setdefault((src, dst), []).append((symbol, action))

    for (src, dst), pairs in edge_pairs.items():
        g.add_edge(src, dst, pairs=pairs)

    if start_pairs:
        g.add_node(START_NODE)
        for dst, pairs in start_pairs.items():
            g.add_edge(START_NODE, dst, pairs=pairs)

    return g


def _label_box(pairs, fontsize=7):
    """Build an OffsetBox rendering "<symbol>/<action>" pairs, one per row,
    with the input symbol in blue and the output action in red."""
    rows = []
    for symbol, action in pairs:
        rows.append(HPacker(children=[
            TextArea(symbol, textprops=dict(color="blue", fontsize=fontsize)),
            TextArea("/", textprops=dict(color="black", fontsize=fontsize)),
            TextArea(action, textprops=dict(color="red", fontsize=fontsize)),
        ], pad=0, sep=0, align="baseline"))
    if len(rows) == 1:
        return rows[0]
    return VPacker(children=rows, pad=0, sep=1, align="center")


def _add_edge_label(ax, xy, pairs, fontsize=7):
    box = _label_box(pairs, fontsize=fontsize)
    ab = AnnotationBbox(box, xy, frameon=True, pad=0.25,
                         bboxprops=dict(facecolor="white", edgecolor="none", alpha=0.75),
                         zorder=4)
    ax.add_artist(ab)


def draw_fsm_png(fsm_structure, title, output_path: Path):
    g = build_graph(fsm_structure)
    initial_state = fsm_structure["initial_state"]
    state_nodes = list(fsm_structure["states"])
    has_start = START_NODE in g.nodes()
    center = np.array([0.0, 0.0])

    n = len(state_nodes)
    node_size = 1000

    if n <= 1:
        pos = {node: np.array([0.0, 0.0]) for node in state_nodes}
        scale = 1.0
    else:
        scale = max(1.0, n / 4.0)
        pos = nx.circular_layout(g.subgraph(state_nodes), scale=scale)

    if has_start:
        # Place the START node in a fixed corner (up and to the left) of the
        # layout, away from the main ring. Self-loops are drawn radially
        # outward from each state, so a fixed corner placement avoids
        # colliding with the self-loop of whichever state START points into.
        corner_dir = np.array([-1.0, 1.0]) / np.sqrt(2)
        pos[START_NODE] = center + corner_dir * (scale + 1.3)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    margin = scale * 0.7 + 0.6
    x_range = (max(xs) - min(xs)) + 2 * margin
    y_range = (max(ys) - min(ys)) + 2 * margin

    inches_per_unit = 1.3
    fig_w = float(np.clip(x_range * inches_per_unit, 4.0, 14.0))
    fig_h = float(np.clip(y_range * inches_per_unit, 3.5, 14.0))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120)

    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_aspect("equal")
    ax.axis("off")

    node_colors = ["#90EE90" if node == initial_state else "#ADD8E6" for node in state_nodes]
    nx.draw_networkx_nodes(g, pos, nodelist=state_nodes, node_color=node_colors,
                            node_size=node_size, edgecolors="black", ax=ax)
    nx.draw_networkx_labels(g, pos, labels={s: s for s in state_nodes}, ax=ax,
                             font_size=11, font_weight="bold")

    if has_start:
        nx.draw_networkx_nodes(g, pos, nodelist=[START_NODE], node_color="white",
                                node_size=node_size * 0.6, edgecolors="gray",
                                linewidths=1.2, ax=ax)
        nx.draw_networkx_labels(g, pos, labels={START_NODE: START_NODE}, ax=ax,
                                 font_size=8, font_color="gray")

    self_loops = [(u, d) for u, v, d in g.edges(data=True) if u == v]
    other_edges = [(u, v) for u, v, d in g.edges(data=True) if u != v]
    other_edge_pairs = {(u, v): d["pairs"] for u, v, d in g.edges(data=True) if u != v}

    nx.draw_networkx_edges(
        g, pos, edgelist=other_edges, ax=ax, connectionstyle="arc3,rad=0.15",
        arrowsize=18, node_size=node_size, min_source_margin=15, min_target_margin=15,
    )
    for (u, v), pairs in other_edge_pairs.items():
        mid = (pos[u] + pos[v]) / 2
        _add_edge_label(ax, mid, pairs)

    # Self-loops: draw a small circle tangent to the node, offset radially
    # outward from the layout center, with its label further out still, so
    # neither overlaps the node's own label.
    fig.canvas.draw()
    node_radius_pts = (node_size / np.pi) ** 0.5
    r_px = node_radius_pts / 72.0 * fig.dpi
    p0_data = ax.transData.inverted().transform((0, 0))
    p1_data = ax.transData.inverted().transform((r_px, 0))
    node_radius_data = abs(p1_data[0] - p0_data[0])

    loop_r = node_radius_data * 0.7
    loop_gap_deg = 50
    for u, d in self_loops:
        p = np.array(pos[u])
        direction = p - center
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-6 else np.array([0.0, 1.0])
        loop_center = p + direction * (node_radius_data + loop_r * 0.9)

        # Draw the loop as a circular arc with a small gap facing the node,
        # ending in an arrowhead so the self-transition direction is visible.
        gap_dir_deg = np.degrees(np.arctan2(-direction[1], -direction[0]))
        theta1 = gap_dir_deg + loop_gap_deg / 2
        theta2 = gap_dir_deg + 360 - loop_gap_deg / 2
        arc = MplPath.arc(theta1, theta2)
        loop_path = MplPath(arc.vertices * loop_r + loop_center, arc.codes)
        ax.add_patch(mpatches.FancyArrowPatch(
            path=loop_path, arrowstyle="-|>", mutation_scale=10,
            facecolor="black", edgecolor="black", lw=1.2, zorder=1,
        ))

        label_pos = p + direction * (node_radius_data + 2 * loop_r + 0.08 * scale)
        _add_edge_label(ax, label_pos, d["pairs"])

    ax.set_title(title, fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _dot_label(pairs):
    parts = [f'<FONT COLOR="blue">{s}</FONT>/<FONT COLOR="red">{a}</FONT>' for s, a in pairs]
    return "<" + ", ".join(parts) + ">"


def write_fsm_dot(fsm_structure, title, output_path: Path):
    g = build_graph(fsm_structure)
    initial_state = fsm_structure["initial_state"]
    state_nodes = fsm_structure["states"]

    dot_title = title.replace('"', '\\"').replace("\n", "\\n")
    lines = ["digraph FSM {", f'  label="{dot_title}"; labelloc=t; fontsize=10;', "  rankdir=LR;"]
    for n in state_nodes:
        shape = "doublecircle" if n == initial_state else "circle"
        lines.append(f'  {n} [shape={shape}];')
    if START_NODE in g.nodes():
        lines.append(f'  {START_NODE} [shape=plaintext, fontcolor=gray];')
    for src, dst, data in g.edges(data=True):
        lines.append(f'  {src} -> {dst} [label={_dot_label(data["pairs"])}];')
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
