"""
Local test for FSM train-test pipeline

This script validates the train-test paradigm before cluster deployment:
1. Train agent on one opponent (0.1)
2. Extract FSM, attribution, save checkpoint
3. Load checkpoint and test on unseen opponents (0.3, 0.5, 0.9)
4. Extract FSM on test data and measure generalization

Usage:
    python test_fsm_train_test_pipeline.py
"""

import os
import sys
import json
import time
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cognitive_therapy_ai.games import GameFactory, Action
from cognitive_therapy_ai.opponent import OpponentFactory
from cognitive_therapy_ai.encoding import ObservationEncoder
from cognitive_therapy_ai.representation_agent import RepresentationAgent
from cognitive_therapy_ai.reinforce_trainer import REINFORCETrainer, SessionEnvironment
from cognitive_therapy_ai.fsm_extraction import (
    RolloutCollector, HiddenStateClusterer, LStarExtractor, HopcroftMinimizer
)

# Disable CUDA for local test
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(title)
    print("="*80 + "\n")


def train_agent(game_name, opponent_coop, hidden_size, n_episodes, seed):
    """Train agent and save checkpoint."""
    print_section(f"TRAINING: {game_name} vs opp={opponent_coop}, H={hidden_size}, seed={seed}")
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create components
    game = GameFactory.create_game(game_name)
    encoder = ObservationEncoder(input_condition="no_game")
    agent = RepresentationAgent(
        input_dim=encoder.get_input_dim(),
        hidden_size=hidden_size
    )
    opponent = OpponentFactory.create_probabilistic_opponent(
        defection_probability=1.0 - opponent_coop
    )
    
    # Abbreviated game name for encoder
    game_abbr_map = {
        "prisoners-dilemma": "PD",
        "stag-hunt": "SH",
        "hawk-dove": "HD"
    }
    game_abbr = game_abbr_map[game_name]
    
    env = SessionEnvironment(
        game=game,
        opponent=opponent,
        encoder=encoder,
        T=100,
        game_name=game_abbr
    )
    
    # Train
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
    trainer = REINFORCETrainer(agent, optimizer, gamma=0.99)
    
    rewards = []
    start_time = time.time()
    
    print(f"Training for {n_episodes} episodes...")
    for episode in range(n_episodes):
        stats = trainer.train_session_rl(env, max_steps=100)
        rewards.append(stats.total_return)
        
        if (episode + 1) % (n_episodes // 10) == 0:
            recent_mean = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            print(f"  Episode {episode+1}/{n_episodes}: mean reward (last 100) = {recent_mean:.1f}")
    
    train_time = time.time() - start_time
    final_reward = np.mean(rewards[-100:])
    
    print(f"\n✓ Training complete in {train_time:.1f}s")
    print(f"  Final reward (last 100): {final_reward:.1f}")
    
    return agent, encoder, game, rewards


def compute_fidelity(agent, fsm, env, encoder, n_episodes=50):
    """Compute fidelity: fraction of steps where FSM matches agent."""
    total_steps = 0
    matching_steps = 0
    
    for _ in range(n_episodes):
        obs = env.reset()
        hidden_state = agent.reset_hidden_state(batch_size=1)
        fsm_state = fsm.initial_state
        
        for step in range(100):
            # Get agent action
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                agent_action_int, _, hidden_state = agent.select_action(
                    obs_tensor, hidden_state,
                    deterministic=True,
                    epsilon=0.0
                )
            
            # Get FSM action from observation symbol
            symbol = obs_to_symbol(obs, encoder)
            fsm_action = fsm.get_action(fsm_state, symbol)
            
            if fsm_action is not None:
                fsm_action_int = 0 if fsm_action == Action.COOPERATE else 1
                
                if agent_action_int == fsm_action_int:
                    matching_steps += 1
                total_steps += 1
                
                # Transition FSM
                next_fsm_state = fsm.get_next_state(fsm_state, symbol)
                if next_fsm_state is not None:
                    fsm_state = next_fsm_state
            
            # Step environment
            obs, reward, done = env.step(agent_action_int)
            
            if done:
                break
    
    return matching_steps / total_steps if total_steps > 0 else 0.0


def obs_to_symbol(obs, encoder):
    """Convert observation to FSM symbol."""
    history_dim = encoder.history_dim
    if obs[:history_dim].sum() == 0:
        return "START"
    
    # Extract outcome from one-hot
    outcome_start = 4  # After agent(2) and opp(2)
    outcome_bits = obs[outcome_start:outcome_start+4]
    outcome_map = {0: "CC", 1: "CD", 2: "DC", 3: "DD"}
    outcome_idx = np.argmax(outcome_bits)
    return outcome_map[outcome_idx]


def extract_fsm(agent, encoder, game, opponent_coop, game_abbr):
    """Extract FSM from trained agent."""
    print_section("FSM EXTRACTION")
    
    # Create opponent
    opponent = OpponentFactory.create_probabilistic_opponent(
        defection_probability=1.0 - opponent_coop
    )
    
    env = SessionEnvironment(
        game=game,
        opponent=opponent,
        encoder=encoder,
        T=100,
        game_name=game_abbr
    )
    
    # Collect rollouts
    print("Collecting rollout trajectories...")
    collector = RolloutCollector(agent, encoder)
    trajectories = collector.collect_multiple_trajectories(
        env, n_trajectories=100, max_steps=100, epsilon_explore=0.1
    )
    
    # Cluster hidden states
    print("Clustering hidden states...")
    n_clusters = min(agent.hidden_size, 20)  # Auto-detect reasonable cluster count
    clusterer = HiddenStateClusterer(n_clusters=n_clusters)
    clusters = clusterer.fit(trajectories)
    
    print(f"  Found {len(clusters)} clusters")
    
    # Extract FSM with L*
    print("Running L* algorithm...")
    alphabet = ["START", "CC", "CD", "DC", "DD"]
    lstar = LStarExtractor(alphabet)
    fsm = lstar.extract_fsm(trajectories, clusters, clusterer, encoder)
    
    print(f"  Extracted FSM with {fsm.n_states} states")
    
    # Minimize with Hopcroft
    print("Minimizing FSM...")
    minimizer = HopcroftMinimizer()
    fsm_min = minimizer.minimize(fsm)
    
    print(f"  Minimized to {fsm_min.n_states} states")
    
    # Validate fidelity
    print("Computing fidelity...")
    fidelity_on = compute_fidelity(agent, fsm_min, env, encoder, n_episodes=50)
    
    # Off-policy fidelity (average across different opponents)
    fidelity_off_list = []
    for p_coop in [0.0, 0.25, 0.5, 0.75, 1.0]:
        opp_off = OpponentFactory.create_probabilistic_opponent(
            defection_probability=1.0 - p_coop
        )
        env_off = SessionEnvironment(
            game=game,
            opponent=opp_off,
            encoder=encoder,
            T=100,
            game_name=game_abbr
        )
        fid = compute_fidelity(agent, fsm_min, env_off, encoder, n_episodes=20)
        fidelity_off_list.append(fid)
    
    fidelity_off = np.mean(fidelity_off_list)
    
    print(f"\n✓ FSM extraction complete")
    print(f"  On-policy fidelity: {fidelity_on:.3f}")
    print(f"  Off-policy fidelity: {fidelity_off:.3f}")
    
    return {
        'fsm': fsm_min,
        'on_policy': fidelity_on,
        'off_policy': fidelity_off,
        'geometric_states': len(clusters),
        'lstar_states': fsm_min.n_states
    }


def test_on_unseen_opponent(checkpoint_path, game_name, test_opponent_coop, hidden_size):
    """Load checkpoint and test on unseen opponent."""
    print_section(f"TESTING: {game_name} vs unseen opp={test_opponent_coop}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Recreate components
    game = GameFactory.create_game(game_name)
    encoder = ObservationEncoder(input_condition="no_game")
    agent = RepresentationAgent(
        input_dim=encoder.get_input_dim(),
        hidden_size=hidden_size
    )
    
    # Load weights
    agent.load_state_dict(checkpoint['model_state_dict'])
    agent.eval()
    
    print("✓ Checkpoint loaded successfully")
    
    # Test on unseen opponent
    opponent = OpponentFactory.create_probabilistic_opponent(
        defection_probability=1.0 - test_opponent_coop
    )
    
    game_abbr_map = {
        "prisoners-dilemma": "PD",
        "stag-hunt": "SH",
        "hawk-dove": "HD"
    }
    game_abbr = game_abbr_map[game_name]
    
    env = SessionEnvironment(
        game=game,
        opponent=opponent,
        encoder=encoder,
        T=100,
        game_name=game_abbr
    )
    
    # Evaluate
    print("Evaluating on test opponent...")
    test_rewards = []
    for _ in range(100):
        obs = env.reset()
        hidden = agent.reset_hidden_state(batch_size=1)
        total_reward = 0
        
        for _ in range(100):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, _, hidden = agent.select_action(obs_tensor, hidden, deterministic=True)
            
            obs, reward, done = env.step(action)
            total_reward += reward
            
            if done:
                break
        
        test_rewards.append(total_reward)
    
    mean_reward = np.mean(test_rewards)
    std_reward = np.std(test_rewards)
    
    print(f"\n✓ Test evaluation complete")
    print(f"  Mean reward: {mean_reward:.1f} ± {std_reward:.1f}")
    
    # Extract FSM on test data
    print("\nExtracting FSM on test data...")
    fsm_test = extract_fsm(agent, encoder, game, test_opponent_coop, game_abbr)
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'fsm_test': fsm_test
    }


def main():
    """Run full train-test pipeline."""
    print("\n" + "="*80)
    print("FSM TRAIN-TEST PIPELINE VALIDATION")
    print("="*80)
    
    # Test configuration (small for local testing)
    game_name = "prisoners-dilemma"
    train_opponent = 0.1  # Train on low-coop
    test_opponents = [0.3, 0.5, 0.9]  # Test on unseen
    hidden_size = 4  # Small network for quick test
    n_episodes = 100  # Reduced for local test (full: 10000)
    seed = 42
    
    output_dir = Path("results/test_fsm_pipeline")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  Game: {game_name}")
    print(f"  Train opponent: {train_opponent}")
    print(f"  Test opponents: {test_opponents}")
    print(f"  Hidden size: H={hidden_size}")
    print(f"  Training episodes: {n_episodes} (quick test)")
    print(f"  Seed: {seed}")
    print(f"  Output: {output_dir}")
    
    # PHASE 1: Train
    agent, encoder, game, train_rewards = train_agent(
        game_name, train_opponent, hidden_size, n_episodes, seed
    )
    
    # Extract FSM on training data
    game_abbr_map = {
        "prisoners-dilemma": "PD",
        "stag-hunt": "SH",
        "hawk-dove": "HD"
    }
    fsm_train = extract_fsm(agent, encoder, game, train_opponent, game_abbr_map[game_name])
    
    # Save checkpoint
    checkpoint_path = output_dir / "checkpoint.pth"
    print(f"\nSaving checkpoint to {checkpoint_path}...")
    torch.save({
        'model_state_dict': agent.state_dict(),
        'config': {
            'game': game_name,
            'hidden_size': hidden_size,
            'train_opponent': train_opponent,
            'seed': seed
        },
        'train_metrics': {
            'rewards': train_rewards,
            'final_reward': np.mean(train_rewards[-100:])
        },
        'fsm_train': {
            'on_policy': fsm_train['on_policy'],
            'off_policy': fsm_train['off_policy'],
            'n_states': fsm_train['lstar_states']
        }
    }, checkpoint_path)
    print("✓ Checkpoint saved")
    
    # PHASE 2: Test on unseen opponents
    test_results = {}
    for test_opp in test_opponents:
        result = test_on_unseen_opponent(checkpoint_path, game_name, test_opp, hidden_size)
        test_results[f"opp_{test_opp}"] = result
    
    # Summary
    print_section("SUMMARY")
    
    print("TRAINING:")
    print(f"  Opponent: {train_opponent}")
    print(f"  Final reward: {np.mean(train_rewards[-100:]):.1f}")
    print(f"  FSM states: {fsm_train['lstar_states']}")
    print(f"  FSM fidelity: {fsm_train['off_policy']:.3f}")
    
    print("\nTESTING (unseen opponents):")
    for test_opp in test_opponents:
        key = f"opp_{test_opp}"
        result = test_results[key]
        print(f"  Opponent {test_opp}:")
        print(f"    Reward: {result['mean_reward']:.1f} ± {result['std_reward']:.1f}")
        print(f"    FSM states: {result['fsm_test']['lstar_states']}")
        print(f"    FSM fidelity: {result['fsm_test']['off_policy']:.3f}")
    
    # Save results
    results_file = output_dir / "pipeline_test_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'training': {
                'game': game_name,
                'opponent': train_opponent,
                'hidden_size': hidden_size,
                'n_episodes': n_episodes,
                'final_reward': float(np.mean(train_rewards[-100:])),
                'fsm_states': int(fsm_train['lstar_states']),
                'fsm_fidelity': float(fsm_train['off_policy'])
            },
            'testing': {
                str(test_opp): {
                    'mean_reward': float(result['mean_reward']),
                    'std_reward': float(result['std_reward']),
                    'fsm_states': int(result['fsm_test']['lstar_states']),
                    'fsm_fidelity': float(result['fsm_test']['off_policy'])
                }
                for test_opp, result in [(o, test_results[f"opp_{o}"]) for o in test_opponents]
            }
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    print("\n" + "="*80)
    print("PIPELINE TEST COMPLETE!")
    print("="*80)
    print("\n✓ Train-test paradigm validated successfully")
    print("✓ Ready for cluster deployment")
    print()


if __name__ == "__main__":
    main()
