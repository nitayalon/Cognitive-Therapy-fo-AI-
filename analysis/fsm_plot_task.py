#!/usr/bin/env python
"""
Plot the minimized FSM(s) for a single training task.

Given a task directory containing train_metrics.json and fidelity_train.json,
renders a state-transition diagram for the agent's minimized FSM (nodes =
states, edges = "<input symbol>/<output action>" transitions). Handles both:

  - specialist tasks (fidelity_train.json holds a single fsm_structure,
    trained against one fixed opponent), e.g.
    experiments/fsm_train_tiny_946438/task_0/opp_0.3/

  - generalist tasks (fidelity_train.json holds a "results" list with one
    fsm_structure per opponent level in opponent_set), e.g.
    experiments/fsm_train_generalist_tiny_<job>/task_0/

Usage:
    python analysis/fsm_plot_task.py --task-dir experiments/fsm_train_tiny_946438/task_0/opp_0.3
    python analysis/fsm_plot_task.py --task-dir experiments/fsm_train_generalist_tiny_<job>/task_0
    python analysis/fsm_plot_task.py --task-dir <generalist dir> --opponent-coop 0.5
"""

import argparse
import json
from pathlib import Path

from fsm_visualize_fsms import GAME_ABBR, draw_fsm_png, write_fsm_dot


def plot_specialist(tm, fid, output_dir, no_dot):
    game_abbr = GAME_ABBR[tm["game"]]
    hidden_size = tm["hidden_size"]
    seed = tm["seed"]
    opponent = tm["opponent_coop"]
    n_states = fid["minimized_states"]
    fidelity = fid["fidelity_train"]

    title = (f"H={hidden_size} {game_abbr} opp={opponent} seed={seed}\n"
             f"states={n_states} fidelity={fidelity:.3f}")
    stem = f"{game_abbr}_opp{opponent}_seed{seed}_h{hidden_size}"

    draw_fsm_png(fid["fsm_structure"], title, output_dir / f"{stem}.png")
    if not no_dot:
        write_fsm_dot(fid["fsm_structure"], title, output_dir / f"{stem}.dot")
    print(f"Wrote {output_dir / stem}.png")


def plot_generalist(tm, fid, output_dir, no_dot, opponent_coop):
    game_abbr = GAME_ABBR[tm["game"]]
    hidden_size = tm["hidden_size"]
    seed = tm["seed"]

    results = fid["results"]
    if opponent_coop is not None:
        results = [r for r in results if abs(r["opponent_coop"] - opponent_coop) < 1e-9]
        if not results:
            raise SystemExit(f"opponent_coop={opponent_coop} not found in fidelity_train.json "
                              f"(available: {[r['opponent_coop'] for r in fid['results']]})")

    for r in results:
        opponent = r["opponent_coop"]
        n_states = r["minimized_states"]
        fidelity = r["fidelity_train"]

        title = (f"H={hidden_size} {game_abbr} GENERALIST opp={opponent} seed={seed}\n"
                 f"states={n_states} fidelity={fidelity:.3f}")
        stem = f"{game_abbr}_generalist_opp{opponent}_seed{seed}_h{hidden_size}"

        draw_fsm_png(r["fsm_structure"], title, output_dir / f"{stem}.png")
        if not no_dot:
            write_fsm_dot(r["fsm_structure"], title, output_dir / f"{stem}.dot")
        print(f"Wrote {output_dir / stem}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-dir", type=str, required=True,
                         help="Directory containing train_metrics.json and fidelity_train.json")
    parser.add_argument("--opponent-coop", type=float, default=None,
                         help="For generalist tasks, only plot the FSM for this opponent level "
                              "(default: plot all opponent levels)")
    parser.add_argument("--output-dir", type=str, default="results/fsm_representation/fsm_diagrams",
                         help="Directory to write PNG/DOT files to")
    parser.add_argument("--no-dot", action="store_true", help="Skip writing .dot files")
    args = parser.parse_args()

    task_dir = Path(args.task_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(task_dir / "train_metrics.json") as f:
        tm = json.load(f)
    with open(task_dir / "fidelity_train.json") as f:
        fid = json.load(f)

    if tm.get("training_mode") == "generalist":
        plot_generalist(tm, fid, output_dir, args.no_dot, args.opponent_coop)
    else:
        if args.opponent_coop is not None:
            raise SystemExit("--opponent-coop is only meaningful for generalist tasks")
        plot_specialist(tm, fid, output_dir, args.no_dot)


if __name__ == "__main__":
    main()
