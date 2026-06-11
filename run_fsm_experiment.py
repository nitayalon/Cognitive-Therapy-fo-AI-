#!/usr/bin/env python3
"""
FSM Representation Experiment Runner

Supports two modes:
1. Training mode: Train agent, extract FSM, save checkpoint and trajectories
2. Testing mode: Load checkpoint, test on unseen opponents/games, extract FSM, save trajectories

Usage:
  # Training
  python run_fsm_experiment.py --mode train --game prisoners-dilemma --opponent 0.1 \
      --hidden-size 4 --input-condition no_game --seed 42 --n-episodes 10000 \
      --save-checkpoint --save-trajectories --output-dir experiments/train/condition_1

  # Testing
  python run_fsm_experiment.py --mode test --checkpoint-path path/to/checkpoint.pth \
      --test-games prisoners-dilemma stag-hunt --test-opponents 0.3 0.5 0.9 \
      --n-test-episodes 100 --save-trajectories --output-dir experiments/test/model_1
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import yaml

from cognitive_therapy_ai.games import GameFactory, Action
from cognitive_therapy_ai.opponent import OpponentFactory
from cognitive_therapy_ai.encoding import ObservationEncoder
from cognitive_therapy_ai.representation_agent import RepresentationAgent
from cognitive_therapy_ai.reinforce_trainer import REINFORCETrainer, SessionEnvironment
from cognitive_therapy_ai.fsm_extraction import (
    RolloutCollector, HiddenStateClusterer, LStarExtractor, HopcroftMinimizer
)
from cognitive_therapy_ai.attribution import AttributionAnalyzer
from cognitive_therapy_ai.trajectory_utils import save_episode_trajectories


# Game name abbreviations
GAME_ABBR = {
    "prisoners-dilemma": "PD",
    "stag-hunt": "SH",
    "hawk-dove": "HD"
}


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


def load_base_config():
    """Load base configuration from config/base.yaml."""
    config_path = Path("config/base.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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


def compute_fidelity_score(agent, fsm, env, encoder, n_episodes=50):
    """
    Compute fidelity: fraction of steps where FSM matches agent.
    
    Args:
        agent: RepresentationAgent
        fsm: Extracted FSM
        env: SessionEnvironment
        encoder: ObservationEncoder
        n_episodes: Number of episodes to evaluate
    
    Returns:
        float: Fidelity score (0 to 1)
    """
    total_steps = 0
    matching_steps = 0
    
    for _ in range(n_episodes):
        obs = env.reset()
        hidden_state = agent.reset_hidden_state(batch_size=1)
        fsm_state = fsm.initial_state
        
        for step in range(100):
            # Get agent action
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                agent_action_int, _, hidden_state = agent.select_action(
                    obs_tensor, hidden_state,
                    deterministic=True,
                    epsilon=0.0
                )
            # Handle both tensor and int returns
            if isinstance(agent_action_int, torch.Tensor):
                agent_action_int = agent_action_int.item()
            
            # Get FSM action (predict from current state and input symbol)
            symbol = obs_to_symbol(obs, encoder)
            if (fsm_state, symbol) in fsm.transitions:
                next_fsm_state, fsm_action = fsm.transitions[(fsm_state, symbol)]
                fsm_action_int = 0 if fsm_action == Action.COOPERATE else 1
                
                # Compare
                if agent_action_int == fsm_action_int:
                    matching_steps += 1
                total_steps += 1
                
                fsm_state = next_fsm_state
            else:
                # FSM doesn't have this transition, skip comparison
                pass
            
            # Step environment
            obs, _, done, _, _, _ = env.step(agent_action_int)
            
            if done:
                break
    
    if total_steps == 0:
        return 0.0
    return matching_steps / total_steps


def extract_fsm_with_data(agent, encoder, game, opponent_coop, game_abbr):
    """
    Extract FSM from agent and return comprehensive data.
    
    Returns:
        dict with keys:
            - fsm: Minimized FSM object
            - fidelity: float
            - geometric_states: int
            - lstar_states: int
            - fsm_structure: dict (states, transitions, alphabet)
    """
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
    print("  Collecting rollout trajectories...")
    collector = RolloutCollector(agent, encoder)
    trajectories = collector.collect_multiple_trajectories(
        env, n_trajectories=100, max_steps=100, epsilon_explore=0.1
    )
    
    # Cluster hidden states
    print("  Clustering hidden states...")
    n_clusters = min(agent.hidden_size, 20)
    clusterer = HiddenStateClusterer(n_clusters=n_clusters)
    clusters = clusterer.fit(trajectories)
    
    geometric_states = clusterer.n_clusters
    print(f"    Geometric clusters: {geometric_states}")
    
    # Extract FSM with L*
    print("  Running L* algorithm...")
    alphabet = ["START", "CC", "CD", "DC", "DD"]
    lstar = LStarExtractor(alphabet=alphabet)
    fsm = lstar.extract_fsm(trajectories, clusters, clusterer, encoder)
    
    lstar_states = len(fsm.states)
    print(f"    L* states: {lstar_states}")
    
    # Minimize FSM
    print("  Minimizing FSM...")
    minimizer = HopcroftMinimizer()
    minimized_fsm = minimizer.minimize(fsm)
    
    minimized_states = len(minimized_fsm.states)
    print(f"    Minimized states: {minimized_states}")
    
    # Compute fidelity
    print("  Computing fidelity...")
    fidelity_on = compute_fidelity_score(agent, minimized_fsm, env, encoder, n_episodes=100)
    
    print(f"    Fidelity: {fidelity_on:.3f}")
    
    # FSM structure (convert to JSON-serializable format)
    transitions_serializable = {}
    for state in minimized_fsm.states:
        state_transitions = {}
        for symbol in minimized_fsm.alphabet:
            transition = minimized_fsm.transitions.get((state, symbol), None)
            if transition is not None:
                next_state, action = transition
                # Convert Action enum to string
                action_str = "COOPERATE" if action == Action.COOPERATE else "DEFECT"
                state_transitions[symbol] = [next_state, action_str]
            else:
                state_transitions[symbol] = None
        transitions_serializable[str(state)] = state_transitions
    
    fsm_structure = {
        'states': list(minimized_fsm.states),
        'alphabet': list(minimized_fsm.alphabet),
        'transitions': transitions_serializable,
        'initial_state': minimized_fsm.initial_state,
        'n_states': minimized_fsm.n_states
    }
    
    return {
        'fsm': minimized_fsm,
        'fidelity': float(fidelity_on),
        'geometric_states': int(geometric_states),
        'lstar_states': int(lstar_states),
        'minimized_states': int(minimized_states),
        'fsm_structure': fsm_structure
    }


def _train_one_opponent(args, base_config, encoder, game, game_abbr, opponent_coop, output_dir):
    """
    Train one agent against one opponent, extract FSM, and save all results.
    Called once per opponent when looping over multiple opponents.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dim = encoder.get_input_dim()
    opponent = OpponentFactory.create_probabilistic_opponent(
        defection_probability=1.0 - opponent_coop
    )

    agent = RepresentationAgent(input_dim=input_dim, hidden_size=args.hidden_size)
    optimizer = torch.optim.Adam(agent.parameters(), lr=base_config['train']['lr'])
    trainer = REINFORCETrainer(
        agent=agent,
        optimizer=optimizer,
        gamma=base_config['train']['gamma'],
        gae_lambda=base_config['train']['gae_lambda']
    )
    env = SessionEnvironment(
        game=game, opponent=opponent, encoder=encoder, T=100, game_name=game_abbr
    )

    # Training loop
    print_section(f"TRAINING  opp={opponent_coop:.1f}")
    start_time = time.time()
    all_trajectories = []
    episode_rewards = []

    for episode in range(args.n_episodes):
        need_trajectory = args.save_trajectories and (episode % args.save_every_nth_episode == 0)
        if need_trajectory:
            stats, trajectories = trainer.train_session_rl(env, return_trajectory=True)
            all_trajectories.append(trajectories)
        else:
            stats = trainer.train_session_rl(env, return_trajectory=False)
        episode_rewards.append(stats.total_return)

        if (episode + 1) % 1000 == 0:
            mean_r = np.mean(episode_rewards[-100:])
            std_r = np.std(episode_rewards[-100:])
            print(f"  Episode {episode+1}/{args.n_episodes}: "
                  f"Reward = {mean_r:.1f} ± {std_r:.1f}", flush=True)

    training_time = time.time() - start_time
    print(f"\n  Training completed in {training_time:.1f}s")
    print(f"  Final reward: {np.mean(episode_rewards[-100:]):.1f} ± {np.std(episode_rewards[-100:]):.1f}")

    # Save training metrics (interim save — written before FSM extraction)
    train_metrics = {
        'game': args.game,
        'opponent_coop': opponent_coop,
        'hidden_size': args.hidden_size,
        'input_condition': args.input_condition,
        'n_episodes': args.n_episodes,
        'seed': args.seed,
        'final_reward_mean': float(np.mean(episode_rewards[-100:])),
        'final_reward_std': float(np.std(episode_rewards[-100:])),
        'all_rewards': [float(r) for r in episode_rewards],
        'training_time_sec': float(training_time)
    }
    with open(output_dir / 'train_metrics.json', 'w') as f:
        json.dump(train_metrics, f, indent=2)

    # Save trajectories
    if args.save_trajectories:
        print_section("SAVING TRAJECTORIES")
        traj_path = output_dir / 'trajectories_train.jsonl.gz'
        n_saved = save_episode_trajectories(
            all_trajectories, traj_path, save_every_nth=1
        )
        print(f"  Saved {n_saved} episodes to {traj_path}")

    # FSM extraction
    print_section("FSM EXTRACTION")
    fsm_data = extract_fsm_with_data(agent, encoder, game, opponent_coop, game_abbr)

    fidelity_data = {
        'fidelity_train': fsm_data['fidelity'],
        'geometric_states': fsm_data['geometric_states'],
        'lstar_states': fsm_data['lstar_states'],
        'minimized_states': fsm_data['minimized_states'],
        'fsm_structure': fsm_data['fsm_structure']
    }
    with open(output_dir / 'fidelity_train.json', 'w') as f:
        json.dump(fidelity_data, f, indent=2)

    # Save checkpoint
    if args.save_checkpoint:
        checkpoint = {
            'agent_state_dict': agent.state_dict(),
            'config': {
                'game': args.game,
                'opponent_coop': opponent_coop,
                'hidden_size': args.hidden_size,
                'input_condition': args.input_condition,
                'input_dim': input_dim,
                'seed': args.seed
            },
            'train_metrics': train_metrics,
            'fidelity': fidelity_data
        }
        checkpoint_path = output_dir / 'checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"  Saved checkpoint to {checkpoint_path}", flush=True)

    return train_metrics


def _train_generalist(args, base_config, encoder, game, game_abbr, opponent_set, output_dir):
    """
    Train one agent on a single game whose opponent's cooperation probability is
    resampled every episode from `opponent_set`, extract an FSM at each opponent
    level, and save all results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dim = encoder.get_input_dim()

    agent = RepresentationAgent(input_dim=input_dim, hidden_size=args.hidden_size)
    optimizer = torch.optim.Adam(agent.parameters(), lr=base_config['train']['lr'])
    trainer = REINFORCETrainer(
        agent=agent,
        optimizer=optimizer,
        gamma=base_config['train']['gamma'],
        gae_lambda=base_config['train']['gae_lambda']
    )

    initial_opponent = OpponentFactory.create_probabilistic_opponent(
        defection_probability=1.0 - opponent_set[0]
    )
    env = SessionEnvironment(
        game=game, opponent=initial_opponent, encoder=encoder, T=100, game_name=game_abbr
    )

    # Training loop
    print_section(f"TRAINING (GENERALIST)  opponent_set={opponent_set}")
    start_time = time.time()
    all_trajectories = []
    episode_rewards = []
    episode_opponent_coops = []

    for episode in range(args.n_episodes):
        # Resample the opponent's cooperation probability for this episode
        opponent_coop = float(np.random.choice(opponent_set))
        env.opponent = OpponentFactory.create_probabilistic_opponent(
            defection_probability=1.0 - opponent_coop
        )
        episode_opponent_coops.append(opponent_coop)

        need_trajectory = args.save_trajectories and (episode % args.save_every_nth_episode == 0)
        if need_trajectory:
            stats, trajectories = trainer.train_session_rl(env, return_trajectory=True)
            all_trajectories.append(trajectories)
        else:
            stats = trainer.train_session_rl(env, return_trajectory=False)
        episode_rewards.append(stats.total_return)

        if (episode + 1) % 1000 == 0:
            mean_r = np.mean(episode_rewards[-100:])
            std_r = np.std(episode_rewards[-100:])
            print(f"  Episode {episode+1}/{args.n_episodes}: "
                  f"Reward = {mean_r:.1f} ± {std_r:.1f}", flush=True)

    training_time = time.time() - start_time
    print(f"\n  Training completed in {training_time:.1f}s")
    print(f"  Final reward: {np.mean(episode_rewards[-100:]):.1f} ± {np.std(episode_rewards[-100:]):.1f}")

    # Per-opponent reward stats over the final stretch of training
    final_window = min(len(episode_rewards), 100 * len(opponent_set))
    final_rewards = np.array(episode_rewards[-final_window:])
    final_opponents = np.array(episode_opponent_coops[-final_window:])
    final_reward_by_opponent = {}
    for opp in opponent_set:
        mask = final_opponents == opp
        if mask.any():
            final_reward_by_opponent[f"{opp:.1f}"] = {
                'mean': float(final_rewards[mask].mean()),
                'std': float(final_rewards[mask].std()),
                'n': int(mask.sum())
            }

    # Save training metrics (interim save — written before FSM extraction)
    train_metrics = {
        'game': args.game,
        'training_mode': 'generalist',
        'opponent_set': [float(o) for o in opponent_set],
        'episode_opponent_coop': episode_opponent_coops,
        'hidden_size': args.hidden_size,
        'input_condition': args.input_condition,
        'n_episodes': args.n_episodes,
        'seed': args.seed,
        'final_reward_mean': float(np.mean(episode_rewards[-100:])),
        'final_reward_std': float(np.std(episode_rewards[-100:])),
        'final_reward_by_opponent': final_reward_by_opponent,
        'all_rewards': [float(r) for r in episode_rewards],
        'training_time_sec': float(training_time)
    }
    with open(output_dir / 'train_metrics.json', 'w') as f:
        json.dump(train_metrics, f, indent=2)

    # Save trajectories
    if args.save_trajectories:
        print_section("SAVING TRAJECTORIES")
        traj_path = output_dir / 'trajectories_train.jsonl.gz'
        n_saved = save_episode_trajectories(
            all_trajectories, traj_path, save_every_nth=1
        )
        print(f"  Saved {n_saved} episodes to {traj_path}")

    # FSM extraction at each opponent level
    print_section("FSM EXTRACTION (per opponent level)")
    fsm_results = []
    for opponent_coop in opponent_set:
        print(f"\n  -- opponent_coop={opponent_coop:.1f} --")
        fsm_data = extract_fsm_with_data(agent, encoder, game, opponent_coop, game_abbr)
        fsm_results.append({
            'opponent_coop': float(opponent_coop),
            'fidelity_train': fsm_data['fidelity'],
            'geometric_states': fsm_data['geometric_states'],
            'lstar_states': fsm_data['lstar_states'],
            'minimized_states': fsm_data['minimized_states'],
            'fsm_structure': fsm_data['fsm_structure']
        })

    fidelity_data = {
        'training_mode': 'generalist',
        'opponent_set': [float(o) for o in opponent_set],
        'results': fsm_results
    }
    with open(output_dir / 'fidelity_train.json', 'w') as f:
        json.dump(fidelity_data, f, indent=2)

    # Save checkpoint
    if args.save_checkpoint:
        checkpoint = {
            'agent_state_dict': agent.state_dict(),
            'config': {
                'game': args.game,
                'opponent_coop': None,
                'training_mode': 'generalist',
                'opponent_set': [float(o) for o in opponent_set],
                'hidden_size': args.hidden_size,
                'input_condition': args.input_condition,
                'input_dim': input_dim,
                'seed': args.seed
            },
            'train_metrics': train_metrics,
            'fidelity': fidelity_data
        }
        checkpoint_path = output_dir / 'checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"  Saved checkpoint to {checkpoint_path}", flush=True)

    return train_metrics


def train_mode(args):
    """Run training mode — loops over all opponents within a single job."""
    print_section("TRAINING MODE")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_config = load_base_config()
    game_abbr = GAME_ABBR[args.game]
    game = GameFactory.create_game(args.game)
    encoder = ObservationEncoder(input_condition=args.input_condition)

    base_output = Path(args.output_dir)
    base_output.mkdir(parents=True, exist_ok=True)

    if args.generalist:
        print(f"\nConfiguration:")
        print(f"  Game: {args.game} ({game_abbr})")
        print(f"  Mode: GENERALIST (opponent resampled each episode from {args.opponent_set})")
        print(f"  Hidden size: {args.hidden_size}")
        print(f"  Input condition: {args.input_condition}")
        print(f"  Training episodes: {args.n_episodes}")
        print(f"  Random seed: {args.seed}")
        print(f"  Output directory: {args.output_dir}")
        print(f"  Agent input size: {encoder.get_input_dim()}")
        print(f"  Agent parameters: {sum(p.numel() for p in RepresentationAgent(encoder.get_input_dim(), args.hidden_size).parameters())}")

        _train_generalist(args, base_config, encoder, game, game_abbr, args.opponent_set, base_output)

        print_section("GENERALIST TRAINING COMPLETE")
        print(f"  Output directory: {base_output}")
        return

    # Support both --opponent (single) and --opponents (multi)
    opponent_list = args.opponents if args.opponents else [args.opponent]

    print(f"\nConfiguration:")
    print(f"  Game: {args.game} ({game_abbr})")
    print(f"  Opponents: {opponent_list}")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Input condition: {args.input_condition}")
    print(f"  Training episodes: {args.n_episodes}")
    print(f"  Random seed: {args.seed}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Agent input size: {encoder.get_input_dim()}")
    print(f"  Agent parameters: {sum(p.numel() for p in RepresentationAgent(encoder.get_input_dim(), args.hidden_size).parameters())}")

    for i, opponent_coop in enumerate(opponent_list):
        print_section(f"OPPONENT {i+1}/{len(opponent_list)}: coop={opponent_coop:.1f}")

        # Each opponent gets its own subdirectory; single-opponent keeps flat layout
        if len(opponent_list) > 1:
            opp_dir = base_output / f"opp_{opponent_coop:.1f}"
        else:
            opp_dir = base_output

        _train_one_opponent(args, base_config, encoder, game, game_abbr, opponent_coop, opp_dir)
        print(f"\n  Opponent {opponent_coop:.1f} complete. Results in {opp_dir}", flush=True)

    print_section("ALL OPPONENTS COMPLETE")
    print(f"  Output directory: {base_output}")
    for opponent_coop in opponent_list:
        opp_dir = base_output / f"opp_{opponent_coop:.1f}" if len(opponent_list) > 1 else base_output
        print(f"    opp={opponent_coop:.1f}: {opp_dir}")


def test_mode(args):
    """Run testing mode."""
    print_section("TESTING MODE")
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    
    config = checkpoint['config']
    print(f"  Trained on: {config['game']}")
    if config.get('training_mode') == 'generalist':
        print(f"  Training mode: generalist (opponent_set={config.get('opponent_set')})")
    else:
        print(f"  Opponent cooperation: {config['opponent_coop']}")
    print(f"  Hidden size: {config['hidden_size']}")
    print(f"  Input condition: {config['input_condition']}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create encoder (same as training)
    encoder = ObservationEncoder(
        input_condition=config['input_condition']
    )
    
    # Create agent and load weights
    agent = RepresentationAgent(
        input_dim=config['input_dim'],
        hidden_size=config['hidden_size']
    )
    agent.load_state_dict(checkpoint['agent_state_dict'])
    agent.eval()
    
    print(f"\nAgent loaded successfully")
    
    # Test on all combinations of games and opponents
    test_results = []
    all_trajectories = []
    
    for test_game in args.test_games:
        game_abbr = GAME_ABBR[test_game]
        game = GameFactory.create_game(test_game)
        
        for test_opponent_coop in args.test_opponents:
            if (args.exclude_train_combo and test_game == config['game']
                    and test_opponent_coop == config['opponent_coop']):
                print(f"\n  Skipping {test_game} vs {test_opponent_coop} (training combo)")
                continue

            print_section(f"Testing: {test_game} vs {test_opponent_coop}")
            
            # Create opponent and environment
            opponent = OpponentFactory.create_probabilistic_opponent(
                defection_probability=1.0 - test_opponent_coop
            )
            
            env = SessionEnvironment(
                game=game,
                opponent=opponent,
                encoder=encoder,
                T=100,
                game_name=game_abbr
            )
            
            # Create trainer for evaluation (optimizer not used during eval)
            optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
            trainer = REINFORCETrainer(
                agent=agent,
                optimizer=optimizer
            )
            
            # Run test episodes
            episode_rewards = []
            test_trajectories = []
            
            for episode in range(args.n_test_episodes):
                need_trajectory = args.save_trajectories and (episode % args.save_every_nth_episode == 0)
                if need_trajectory:
                    stats, trajectories = trainer.train_session_rl(env, return_trajectory=True)
                    test_trajectories.append(trajectories)  # list-of-lists
                else:
                    stats = trainer.train_session_rl(env, return_trajectory=False)
                episode_rewards.append(stats.total_return)
            
            # Compute statistics
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            
            print(f"  Test reward: {mean_reward:.1f} ± {std_reward:.1f}")
            
            # Extract FSM on test data
            print(f"  Extracting FSM...")
            fsm_data = extract_fsm_with_data(agent, encoder, game, test_opponent_coop, game_abbr)
            
            # Store results
            result = {
                'test_game': test_game,
                'test_opponent_coop': test_opponent_coop,
                'n_episodes': args.n_test_episodes,
                'reward_mean': float(mean_reward),
                'reward_std': float(std_reward),
                'fidelity_test': fsm_data['fidelity'],
                'geometric_states': fsm_data['geometric_states'],
                'lstar_states': fsm_data['lstar_states'],
                'minimized_states': fsm_data['minimized_states'],
                'fsm_structure': fsm_data['fsm_structure']
            }
            test_results.append(result)
            
            if args.save_trajectories:
                all_trajectories.extend(test_trajectories)  # extend list-of-lists
    
    # Save test results
    print_section("SAVING TEST RESULTS")
    
    test_summary = {
        'checkpoint_path': str(args.checkpoint_path),
        'training_config': config,
        'test_games': args.test_games,
        'test_opponents': args.test_opponents,
        'n_test_episodes': args.n_test_episodes,
        'results': test_results
    }
    
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(test_summary, f, indent=2)
    
    print(f"  Saved test results to {output_dir / 'test_results.json'}")
    
    # Save trajectories
    if args.save_trajectories:
        traj_path = output_dir / 'trajectories_test.jsonl.gz'
        n_saved = save_episode_trajectories(
            all_trajectories,
            traj_path,
            save_every_nth=1  # already subsampled every args.save_every_nth_episode during collection
        )
        print(f"  Saved {n_saved} episodes ({len(all_trajectories)} collected) to {traj_path}")
    
    # Print summary table
    print_section("TEST SUMMARY")
    print(f"\n{'Game':<20} {'Opponent':<10} {'Reward':<15} {'Fidelity':<10} {'States':<10}")
    print('-' * 70)
    for r in test_results:
        reward_str = f"{r['reward_mean']:.1f} ± {r['reward_std']:.1f}"
        fidelity_str = f"{r['fidelity_test']:.3f}"
        states_str = f"{r['minimized_states']}"
        print(f"{r['test_game']:<20} {r['test_opponent_coop']:<10} {reward_str:<15} {fidelity_str:<10} {states_str:<10}")
    
    print(f"\n  Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="FSM Representation Experiment Runner")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'],
                        help='Experiment mode: train or test')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for results')
    
    # Training arguments
    parser.add_argument('--game', type=str, 
                        choices=['prisoners-dilemma', 'stag-hunt', 'hawk-dove'],
                        help='Game to train on (required for train mode)')
    parser.add_argument('--opponent', type=float,
                        help='Single opponent cooperation probability (train mode)')
    parser.add_argument('--opponents', type=float, nargs='+',
                        help='Multiple opponent cooperation probabilities; each trains a separate agent in sequence (train mode). Use instead of --opponent.')
    parser.add_argument('--generalist', action='store_true',
                        help='Train a single agent on --game with the opponent cooperation '
                             'probability resampled every episode from --opponent-set, '
                             'instead of training one agent per fixed opponent (train mode).')
    parser.add_argument('--opponent-set', type=float, nargs='+', default=[0.1, 0.3, 0.5, 0.7, 0.9],
                        help='Opponent cooperation probabilities to sample from during '
                             '--generalist training, and to use for train-time FSM/fidelity '
                             'extraction (default: 0.1 0.3 0.5 0.7 0.9)')
    parser.add_argument('--hidden-size', type=int, default=4,
                        help='LSTM hidden size (default: 4)')
    parser.add_argument('--input-condition', type=str, default='no_game',
                        choices=['no_game', 'game_tag'],
                        help='Input encoding condition (default: no_game)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--n-episodes', type=int, default=10000,
                        help='Number of training episodes (default: 10000)')
    parser.add_argument('--save-checkpoint', action='store_true',
                        help='Save model checkpoint after training')
    
    # Testing arguments
    parser.add_argument('--checkpoint-path', type=str,
                        help='Path to checkpoint file (required for test mode)')
    parser.add_argument('--test-games', type=str, nargs='+',
                        choices=['prisoners-dilemma', 'stag-hunt', 'hawk-dove'],
                        help='Games to test on (required for test mode)')
    parser.add_argument('--test-opponents', type=float, nargs='+',
                        help='Opponent cooperation probabilities for testing (required for test mode)')
    parser.add_argument('--n-test-episodes', type=int, default=100,
                        help='Number of test episodes per condition (default: 100)')
    parser.add_argument('--exclude-train-combo', action='store_true',
                        help='Skip the (game, opponent) combo the checkpoint was trained on (test mode)')
    
    # Trajectory saving arguments
    parser.add_argument('--save-trajectories', action='store_true',
                        help='Save behavioral trajectories')
    parser.add_argument('--save-every-nth-episode', type=int, default=1,
                        help='Subsample trajectories: save every Nth episode (default: 1 = all)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'train':
        if args.game is None:
            parser.error("--game is required for train mode")
        if not args.generalist and args.opponent is None and not args.opponents:
            parser.error("either --opponent, --opponents, or --generalist is required for train mode")
    elif args.mode == 'test':
        if args.checkpoint_path is None or args.test_games is None or args.test_opponents is None:
            parser.error("--checkpoint-path, --test-games, and --test-opponents are required for test mode")
    
    # Run appropriate mode
    if args.mode == 'train':
        train_mode(args)
    else:
        test_mode(args)


if __name__ == '__main__':
    main()
