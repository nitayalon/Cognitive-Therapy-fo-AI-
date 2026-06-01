"""
Test trajectory collection functionality.

This script validates that:
1. SessionEnvironment returns opponent data correctly
2. Trajectories capture all behavioral information
3. Trajectory saving/loading works
4. Data format matches requirements
"""

import torch
import numpy as np
from pathlib import Path

from cognitive_therapy_ai.games import GameFactory
from cognitive_therapy_ai.opponent import OpponentFactory
from cognitive_therapy_ai.encoding import ObservationEncoder
from cognitive_therapy_ai.representation_agent import RepresentationAgent
from cognitive_therapy_ai.reinforce_trainer import REINFORCETrainer, SessionEnvironment
from cognitive_therapy_ai.trajectory_utils import (
    save_episode_trajectories,
    load_trajectories_jsonl,
    compute_trajectory_statistics
)


def test_trajectory_collection():
    """Test full trajectory collection pipeline."""
    print("="*80)
    print("TESTING TRAJECTORY COLLECTION")
    print("="*80)
    
    # Setup
    game = GameFactory.create_game("prisoners-dilemma")
    opponent = OpponentFactory.create_probabilistic_opponent(defection_probability=0.7)
    encoder = ObservationEncoder(input_condition="no_game")
    
    agent = RepresentationAgent(
        input_dim=encoder.get_input_dim(),
        hidden_size=4
    )
    
    env = SessionEnvironment(
        game=game,
        opponent=opponent,
        encoder=encoder,
        T=100,
        game_name="PD"
    )
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
    trainer = REINFORCETrainer(agent, optimizer)
    
    print("\n1. Collecting trajectories from 5 training episodes...")
    all_trajectories = []
    
    for episode in range(5):
        stats, trajectory = trainer.train_session_rl(
            env, max_steps=100, return_trajectory=True
        )
        all_trajectories.append(trajectory)
        print(f"  Episode {episode}: {len(trajectory)} steps, reward={stats.total_return:.1f}")
    
    print(f"\n✓ Collected {sum(len(t) for t in all_trajectories)} timesteps")
    
    # Validate data structure
    print("\n2. Validating trajectory data structure...")
    first_step = all_trajectories[0][0]
    
    required_fields = [
        'observation', 'action', 'reward', 'policy_logits', 'value',
        'opponent_action', 'opponent_reward', 'opponent_action_prob',
        'episode_id', 'timestep', 'done'
    ]
    
    for field in required_fields:
        assert hasattr(first_step, field), f"Missing field: {field}"
    
    print(f"✓ All {len(required_fields)} required fields present")
    
    # Check data types
    print("\n3. Checking data types...")
    assert isinstance(first_step.action, int), "action should be int"
    assert isinstance(first_step.opponent_action, int), "opponent_action should be int"
    assert isinstance(first_step.reward, float), "reward should be float"
    assert isinstance(first_step.opponent_reward, float), "opponent_reward should be float"
    assert isinstance(first_step.opponent_action_prob, float), "opponent_action_prob should be float"
    
    print("✓ Data types correct")
    
    # Validate opponent probabilities
    print("\n4. Validating opponent action probabilities...")
    opp_probs = [step.opponent_action_prob for traj in all_trajectories for step in traj]
    mean_prob = np.mean(opp_probs)
    print(f"  Mean opponent action prob: {mean_prob:.3f}")
    print(f"  Expected (0.3 coop, 0.7 defect): mix of both")
    
    # Count cooperation
    opp_actions = [step.opponent_action for traj in all_trajectories for step in traj]
    coop_rate = 1 - np.mean(opp_actions)
    print(f"  Opponent cooperation rate: {coop_rate:.3f} (expected ~0.30)")
    
    assert 0.15 < coop_rate < 0.45, f"Cooperation rate {coop_rate} outside expected range"
    print("✓ Opponent probabilities valid")
    
    # Test saving
    print("\n5. Testing trajectory saving...")
    output_dir = Path("results/test_trajectory_collection")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "trajectories_train.jsonl.gz"
    n_saved = save_episode_trajectories(
        all_trajectories,
        output_file,
        compress=True,
        save_every_nth=1
    )
    
    print(f"  Saved {n_saved} episodes to {output_file}")
    print(f"  File size: {output_file.stat().st_size / 1024:.1f} KB")
    
    # Test loading
    print("\n6. Testing trajectory loading...")
    loaded_data = list(load_trajectories_jsonl(output_file, max_episodes=2))
    print(f"  Loaded {len(loaded_data)} timesteps from first 2 episodes")
    
    # Validate loaded data structure
    first_loaded = loaded_data[0]
    required_json_fields = [
        'episode', 'timestep', 'agent_action', 'agent_action_prob', 'agent_reward',
        'opponent_action', 'opponent_action_prob', 'opponent_reward',
        'value_estimate', 'observation', 'done'
    ]
    
    for field in required_json_fields:
        assert field in first_loaded, f"Missing JSON field: {field}"
    
    print(f"✓ All {len(required_json_fields)} JSON fields present")
    
    # Display sample
    print("\n7. Sample trajectory data:")
    print(f"  Episode: {first_loaded['episode']}")
    print(f"  Timestep: {first_loaded['timestep']}")
    print(f"  Agent action: {first_loaded['agent_action']} (prob={first_loaded['agent_action_prob']:.3f})")
    print(f"  Opponent action: {first_loaded['opponent_action']} (prob={first_loaded['opponent_action_prob']:.3f})")
    print(f"  Agent reward: {first_loaded['agent_reward']:.1f}")
    print(f"  Opponent reward: {first_loaded['opponent_reward']:.1f}")
    print(f"  Value estimate: {first_loaded['value_estimate']:.2f}")
    print(f"  Observation dim: {len(first_loaded['observation'])}")
    
    # Compute statistics
    print("\n8. Computing trajectory statistics...")
    stats = compute_trajectory_statistics(output_file)
    print(f"  Episodes: {stats['num_episodes']}")
    print(f"  Total timesteps: {stats['total_timesteps']}")
    print(f"  Mean episode reward: {stats['mean_episode_reward']:.1f} ± {stats['std_episode_reward']:.1f}")
    print(f"  Agent cooperation: {stats['mean_agent_cooperation']:.3f}")
    print(f"  Opponent cooperation: {stats['mean_opponent_cooperation']:.3f}")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED")
    print("="*80)
    print("\nTrajectory collection system is working correctly!")
    print("Ready to integrate into experiment scripts.")
    

if __name__ == "__main__":
    test_trajectory_collection()
