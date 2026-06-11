#!/usr/bin/env python
"""
Extract training-curve and FSM-summary CSVs from experiments/fsm_train_*.

Scans experiments/fsm_train_*/task_*/opp_*/ (one directory per
hidden_size x game x seed x opponent training run) and produces:

  training_rewards.csv.gz
      Per-episode training reward (from train_metrics.json::all_rewards).
      columns: hidden_size, game, opponent, seed, episode, reward

  training_policy.csv.gz
      Per-timestep agent cooperation probability for the subsampled
      training episodes saved in trajectories_train.jsonl.gz.
      columns: hidden_size, game, opponent, seed, episode, timestep,
               agent_action, agent_cooperation_prob, opponent_action,
               agent_reward
      `episode` is the original training-episode index (0..n_episodes-1),
      recovered from the trajectory file's subsample stride.

  fsm_summary.csv
      Minimized FSM size and fidelity per training run.
      columns: hidden_size, game, opponent, seed, geometric_states,
               lstar_states, minimized_states, fidelity

Usage:
    python analysis/fsm_extract_training_data.py
    python analysis/fsm_extract_training_data.py --output-dir results/fsm_representation
"""

import argparse
import csv
import gzip
import json
from pathlib import Path


def find_train_metrics(experiments_root: Path):
    return sorted(experiments_root.glob("fsm_train_*/task_*/opp_*/train_metrics.json"))


def load_meta(tm_path: Path):
    with open(tm_path) as f:
        tm = json.load(f)
    meta = {
        "hidden_size": tm["hidden_size"],
        "game": tm["game"],
        "opponent": tm["opponent_coop"],
        "seed": tm["seed"],
    }
    return meta, tm


def open_out(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if str(path).endswith(".gz"):
        return gzip.open(path, "wt", newline="")
    return open(path, "w", newline="")


def extract_rewards(tm_paths, output_path: Path):
    n_rows = 0
    with open_out(output_path) as f:
        writer = csv.writer(f)
        writer.writerow(["hidden_size", "game", "opponent", "seed", "episode", "reward"])
        for tm_path in tm_paths:
            meta, tm = load_meta(tm_path)
            prefix = [meta["hidden_size"], meta["game"], meta["opponent"], meta["seed"]]
            for episode, reward in enumerate(tm["all_rewards"]):
                writer.writerow(prefix + [episode, reward])
                n_rows += 1
    print(f"  Wrote {n_rows} rows to {output_path}")


def extract_policy(tm_paths, output_path: Path):
    n_rows = 0
    n_with_traj = 0
    n_missing_traj = 0
    with open_out(output_path) as f:
        writer = csv.writer(f)
        writer.writerow([
            "hidden_size", "game", "opponent", "seed", "episode", "timestep",
            "agent_action", "agent_cooperation_prob", "opponent_action", "agent_reward",
        ])
        for tm_path in tm_paths:
            traj_path = tm_path.parent / "trajectories_train.jsonl.gz"
            if not traj_path.exists():
                n_missing_traj += 1
                continue
            n_with_traj += 1

            meta, tm = load_meta(tm_path)
            n_episodes = tm["n_episodes"]
            prefix = [meta["hidden_size"], meta["game"], meta["opponent"], meta["seed"]]

            with gzip.open(traj_path, "rt") as tf:
                lines = tf.readlines()

            # Trajectory file re-numbers saved episodes as 0..(K-1); recover the
            # original training-episode index via the subsample stride.
            file_episode_ids = {json.loads(l)["episode"] for l in lines}
            n_saved = max(1, len(file_episode_ids))
            stride = max(1, n_episodes // n_saved)

            for line in lines:
                d = json.loads(line)
                action = d["agent_action"]
                action_prob = d["agent_action_prob"]
                coop_prob = action_prob if action == 0 else 1.0 - action_prob
                orig_episode = d["episode"] * stride
                writer.writerow(prefix + [
                    orig_episode, d["timestep"], action, coop_prob,
                    d["opponent_action"], d["agent_reward"],
                ])
                n_rows += 1

    print(f"  Wrote {n_rows} rows to {output_path}")
    print(f"  ({n_with_traj} runs with trajectories, {n_missing_traj} missing trajectories_train.jsonl.gz)")


def extract_fsm_summary(tm_paths, output_path: Path):
    n_rows = 0
    n_missing = 0
    with open_out(output_path) as f:
        writer = csv.writer(f)
        writer.writerow([
            "hidden_size", "game", "opponent", "seed",
            "geometric_states", "lstar_states", "minimized_states", "fidelity",
        ])
        for tm_path in tm_paths:
            fid_path = tm_path.parent / "fidelity_train.json"
            if not fid_path.exists():
                n_missing += 1
                continue
            meta, _ = load_meta(tm_path)
            with open(fid_path) as ff:
                fid = json.load(ff)
            writer.writerow([
                meta["hidden_size"], meta["game"], meta["opponent"], meta["seed"],
                fid["geometric_states"], fid["lstar_states"], fid["minimized_states"],
                fid["fidelity_train"],
            ])
            n_rows += 1

    print(f"  Wrote {n_rows} rows to {output_path}")
    if n_missing:
        print(f"  WARNING: {n_missing} fidelity_train.json files missing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", type=str, default="experiments")
    parser.add_argument("--output-dir", type=str, default="results/fsm_representation")
    parser.add_argument("--no-compress", action="store_true",
                         help="Write plain .csv instead of .csv.gz for the per-step tables")
    parser.add_argument("--skip-rewards", action="store_true")
    parser.add_argument("--skip-policy", action="store_true")
    parser.add_argument("--skip-fsm-summary", action="store_true")
    args = parser.parse_args()

    experiments_root = Path(args.experiments_root)
    output_dir = Path(args.output_dir)
    ext = "csv" if args.no_compress else "csv.gz"

    tm_paths = find_train_metrics(experiments_root)
    print(f"Found {len(tm_paths)} training runs under {experiments_root}/fsm_train_*/task_*/opp_*/")

    if not args.skip_rewards:
        print("\nExtracting training rewards...")
        extract_rewards(tm_paths, output_dir / f"training_rewards.{ext}")

    if not args.skip_policy:
        print("\nExtracting training policy (cooperation probability)...")
        extract_policy(tm_paths, output_dir / f"training_policy.{ext}")

    if not args.skip_fsm_summary:
        print("\nExtracting FSM summary...")
        extract_fsm_summary(tm_paths, output_dir / "fsm_summary.csv")


if __name__ == "__main__":
    main()
