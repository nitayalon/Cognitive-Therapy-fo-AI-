"""
Utilities for collecting and saving behavioral trajectory data.

This module provides functions to:
1. Convert TrajectoryStep objects to JSON-serializable dicts
2. Save trajectories to compressed JSONL files
3. Load and analyze trajectory data efficiently
"""

import json
import gzip
from pathlib import Path
from typing import List, Dict, Any, Iterator
import numpy as np
import torch

from .reinforce_trainer import TrajectoryStep


def trajectory_step_to_dict(step: TrajectoryStep, episode_id: int) -> Dict[str, Any]:
    """
    Convert TrajectoryStep to JSON-serializable dictionary.
    
    Args:
        step: TrajectoryStep object
        episode_id: Episode number
    
    Returns:
        Dictionary with all behavioral data
    """
    # Compute agent action probability from policy logits
    policy_probs = torch.softmax(step.policy_logits, dim=0)
    agent_action_prob = float(policy_probs[step.action].detach().cpu().numpy())
    
    return {
        "episode": episode_id if step.episode_id == -1 else step.episode_id,
        "timestep": step.timestep,
        
        # Agent data
        "agent_action": step.action,
        "agent_action_prob": agent_action_prob,
        "agent_reward": step.reward,
        
        # Opponent data
        "opponent_action": step.opponent_action,
        "opponent_action_prob": step.opponent_action_prob,
        "opponent_reward": step.opponent_reward,
        
        # Agent internal state
        "value_estimate": float(step.value.detach().cpu().numpy()),
        "observation": step.observation.detach().cpu().numpy().tolist(),
        
        # Metadata
        "done": step.done
    }


def save_trajectories_jsonl(
    trajectories: List[TrajectoryStep],
    output_path: Path,
    compress: bool = True
) -> int:
    """
    Save trajectories to JSONL file (one JSON object per line).
    
    Args:
        trajectories: List of TrajectoryStep objects
        output_path: Path to save file (will add .gz if compress=True)
        compress: If True, use gzip compression
    
    Returns:
        Number of timesteps saved
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if compress and not str(output_path).endswith('.gz'):
        output_path = Path(str(output_path) + '.gz')
    
    # Group by episode
    current_episode = 0
    
    open_fn = gzip.open if compress else open
    mode = 'wt' if compress else 'w'
    
    with open_fn(output_path, mode) as f:
        for step in trajectories:
            # Update episode_id if needed
            if step.episode_id == -1:
                step_dict = trajectory_step_to_dict(step, current_episode)
            else:
                step_dict = trajectory_step_to_dict(step, step.episode_id)
                current_episode = step.episode_id
            
            # Write as single-line JSON
            f.write(json.dumps(step_dict) + '\n')
            
            # Increment episode on done
            if step.done and step.episode_id == -1:
                current_episode += 1
    
    return len(trajectories)


def save_episode_trajectories(
    episode_trajectories: List[List[TrajectoryStep]],
    output_path: Path,
    compress: bool = True,
    save_every_nth: int = 1
) -> int:
    """
    Save trajectories from multiple episodes with optional subsampling.
    
    Args:
        episode_trajectories: List of episodes, each is a list of TrajectorySteps
        output_path: Path to save file
        compress: If True, use gzip compression
        save_every_nth: Save every N-th episode (1 = save all)
    
    Returns:
        Number of episodes saved
    """
    all_steps = []
    episodes_saved = 0
    
    for episode_id, trajectory in enumerate(episode_trajectories):
        if episode_id % save_every_nth == 0:
            # Set episode IDs
            for step in trajectory:
                step.episode_id = episode_id
            all_steps.extend(trajectory)
            episodes_saved += 1
    
    save_trajectories_jsonl(all_steps, output_path, compress=compress)
    return episodes_saved


def load_trajectories_jsonl(
    file_path: Path,
    max_episodes: int = None
) -> Iterator[Dict[str, Any]]:
    """
    Load trajectories from JSONL file (generator for memory efficiency).
    
    Args:
        file_path: Path to JSONL or JSONL.GZ file
        max_episodes: If set, stop after loading this many episodes
    
    Yields:
        Dictionary with trajectory data
    """
    file_path = Path(file_path)
    is_compressed = str(file_path).endswith('.gz')
    
    open_fn = gzip.open if is_compressed else open
    mode = 'rt' if is_compressed else 'r'
    
    episodes_seen = set()
    
    with open_fn(file_path, mode) as f:
        for line in f:
            data = json.loads(line.strip())
            
            # Track episode count
            if max_episodes is not None:
                episodes_seen.add(data['episode'])
                if len(episodes_seen) > max_episodes:
                    break
            
            yield data


def compute_trajectory_statistics(file_path: Path) -> Dict[str, Any]:
    """
    Compute summary statistics from trajectory file.
    
    Args:
        file_path: Path to trajectory JSONL file
    
    Returns:
        Dictionary with statistics
    """
    episodes = {}
    total_timesteps = 0
    
    for data in load_trajectories_jsonl(file_path):
        ep = data['episode']
        
        if ep not in episodes:
            episodes[ep] = {
                'rewards': [],
                'agent_cooperation_rate': 0,
                'opponent_cooperation_rate': 0,
                'timesteps': 0
            }
        
        episodes[ep]['rewards'].append(data['agent_reward'])
        episodes[ep]['agent_cooperation_rate'] += (1 - data['agent_action'])
        episodes[ep]['opponent_cooperation_rate'] += (1 - data['opponent_action'])
        episodes[ep]['timesteps'] += 1
        total_timesteps += 1
    
    # Normalize cooperation rates
    for ep_data in episodes.values():
        if ep_data['timesteps'] > 0:
            ep_data['agent_cooperation_rate'] /= ep_data['timesteps']
            ep_data['opponent_cooperation_rate'] /= ep_data['timesteps']
    
    # Compute summary
    all_rewards = [sum(ep['rewards']) for ep in episodes.values()]
    
    return {
        'num_episodes': len(episodes),
        'total_timesteps': total_timesteps,
        'mean_episode_reward': float(np.mean(all_rewards)),
        'std_episode_reward': float(np.std(all_rewards)),
        'mean_agent_cooperation': float(np.mean([ep['agent_cooperation_rate'] for ep in episodes.values()])),
        'mean_opponent_cooperation': float(np.mean([ep['opponent_cooperation_rate'] for ep in episodes.values()]))
    }


def subsample_trajectories(
    input_path: Path,
    output_path: Path,
    every_nth_episode: int = 10,
    compress: bool = True
) -> int:
    """
    Create subsampled version of trajectory file.
    
    Args:
        input_path: Source trajectory file
        output_path: Destination file
        every_nth_episode: Keep every N-th episode
        compress: Compress output
    
    Returns:
        Number of episodes in output file
    """
    episodes_saved = 0
    current_episode = -1
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if compress and not str(output_path).endswith('.gz'):
        output_path = Path(str(output_path) + '.gz')
    
    open_fn = gzip.open if compress else open
    mode = 'wt' if compress else 'w'
    
    with open_fn(output_path, mode) as out_f:
        for data in load_trajectories_jsonl(input_path):
            if data['episode'] != current_episode:
                current_episode = data['episode']
                if current_episode % every_nth_episode == 0:
                    out_f.write(json.dumps(data) + '\n')
                    if data['done']:
                        episodes_saved += 1
            elif current_episode % every_nth_episode == 0:
                out_f.write(json.dumps(data) + '\n')
    
    return episodes_saved
