"""
Main experiment script for training LSTM networks on mixed-motive games.

This script demonstrates how to use the cognitive therapy AI framework to:
1. Train LSTM networks on Hawk-Dove, Prisoner's Dilemma, and Battle of the Sexes
2. Test different opponent defection probabilities
3. Analyze and visualize results
4. Save trained models for further analysis

Usage:
    python main_experiment.py --config config/experiment_config.json
    python main_experiment.py --game prisoners-dilemma --opponents 0.1,0.3,0.5,0.7,0.9
"""

import argparse
import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
import torch

# Import our framework
from cognitive_therapy_ai import (
    GameFactory, 
    OpponentFactory, 
    GameLSTM, 
    GameTrainer,
    MixedMotiveGame
)
from cognitive_therapy_ai.config import (
    NetworkConfig, 
    TrainingConfig, 
    ExperimentConfig
)
from cognitive_therapy_ai.utils import (
    set_random_seeds, 
    setup_logging, 
    create_output_dirs,
    save_results
)


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy data types and PyTorch tensors."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif hasattr(obj, 'item'):  # Handle single-element tensors
            try:
                return obj.item()
            except (ValueError, TypeError):
                pass
        return super().default(obj)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train LSTM networks on mixed-motive games"
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        help='Path to experiment configuration file'
    )
    
    parser.add_argument(
        '--training-games', 
        type=str, 
        default='prisoners-dilemma,hawk-dove',
        help='Comma-separated list of games to train on (e.g., "prisoners-dilemma,hawk-dove,battle-of-sexes")'
    )
    
    parser.add_argument(
        '--test-game', 
        type=str, 
        choices=['hawk-dove', 'prisoners-dilemma', 'battle-of-sexes', 'stag-hunt'],
        default='stag-hunt',
        help='Game to test the trained agent on'
    )
    
    parser.add_argument(
        '--opponents', 
        type=str, 
        default='0.1,0.3,0.5,0.7,0.9',
        help='Comma-separated list of opponent defection probabilities'
    )
    
    parser.add_argument(
        '--num-games', 
        type=int, 
        default=100,
        help='Number of games per partner (T parameter)'
    )
    
    parser.add_argument(
        '--max-epochs', 
        type=int, 
        default=500,
        help='Maximum number of training epochs'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='experiments',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--device', 
        type=str, 
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help='Device to use for training'
    )
    
    parser.add_argument(
        '--adaptive-loss', 
        action='store_true',
        help='Use adaptive loss weighting'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Setup and return the appropriate device."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    return device


def create_network(network_config: NetworkConfig, game: MixedMotiveGame) -> GameLSTM:
    """Create and initialize the LSTM network."""
    # Input size based on game's state representation (payoff matrix + metadata)
    input_size = game.get_state_size()
    
    network = GameLSTM(
        input_size=input_size,
        hidden_size=network_config.hidden_size,
        num_layers=network_config.num_layers,
        dropout=network_config.dropout,
        num_actions=2  # Cooperate/Defect
    )
    
    return network


def run_multi_game_experiment(
    training_games: List[str],
    test_game: str,
    opponent_probs: List[float],
    network_config: NetworkConfig,
    training_config: TrainingConfig,
    device: torch.device,
    output_dirs: Dict[str, str],
    use_adaptive_loss: bool = False
) -> Dict[str, Any]:
    """
    Run experiment with training on multiple games and testing on a separate game.
    
    Args:
        training_games: List of game names to train on
        test_game: Game name to test on
        opponent_probs: List of opponent cooperation probabilities
        network_config: Network configuration
        training_config: Training configuration
        device: Device for computation
        output_dirs: Output directories
        use_adaptive_loss: Whether to use adaptive loss
        
    Returns:
        Dictionary with experiment results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting multi-game training experiment")
    logger.info(f"Training games: {training_games}")
    logger.info(f"Test game: {test_game}")
    
    # Create a sample training game to determine input size
    sample_game = GameFactory.create_game(training_games[0])
    
    # Create network
    network = create_network(network_config, sample_game)
    
    # Create opponents
    opponents = OpponentFactory.create_opponent_set(opponent_probs)
    
    # Create trainer
    trainer = GameTrainer(
        network=network,
        training_config=training_config,
        device=device,
        use_adaptive_loss=use_adaptive_loss
    )
    
    # Phase 1: Simultaneous multi-game training
    logger.info("=== SIMULTANEOUS MULTI-GAME TRAINING PHASE ===")
    
    # Create game configurations for simultaneous training
    game_configs = []
    for game_name in training_games:
        config = GameTrainer.create_game_config(game_name, weight=1.0)
        game_configs.append(config)
        logger.info(f"  Adding {game_name} with weight=1.0")
    
    # Run simultaneous multi-game training
    logger.info(f"Training simultaneously on {len(training_games)} games...")
    training_results = trainer.train_on_multiple_games(
        game_configs=game_configs,
        opponents=opponents,
        save_dir=output_dirs['checkpoints']
    )
    
    logger.info(f"Completed simultaneous training:")
    logger.info(f"  - Games: {', '.join(training_results['games'])}")
    logger.info(f"  - Epochs: {training_results['final_metrics']['total_epochs']}")
    logger.info(f"  - Best loss: {training_results['final_metrics']['best_loss']:.6f}")
    
    # Log per-game performance breakdown
    if training_results['epoch_results']:
        last_epoch = training_results['epoch_results'][-1]
        if 'per_game_losses' in last_epoch:
            logger.info("  Per-game final losses:")
            for game_name, losses in last_epoch['per_game_losses'].items():
                logger.info(f"    {game_name}: {losses.get('total_loss', 'N/A'):.6f}")
        
        logger.info(f"  Sessions per game: {last_epoch['num_sessions_per_game']}")
    
    # Phase 2: Testing on separate game
    logger.info("=== TESTING PHASE ===")
    logger.info(f"Testing trained agent on {test_game}...")
    
    test_game_instance = GameFactory.create_game(test_game)
    test_results = trainer.evaluate(
        game=test_game_instance,
        opponents=opponents,
        num_sessions=50  # More sessions for thorough testing
    )
    
    # Compile complete results
    complete_results = {
        'training_games': training_games,
        'test_game': test_game,
        'opponent_probabilities': opponent_probs,
        'network_config': network_config.__dict__,
        'training_config': training_config.__dict__,
        'training_results': training_results,
        'test_results': test_results,
        'network_parameters': sum(p.numel() for p in network.parameters()),
        'device': str(device)
    }
    
    return complete_results


def run_single_game_experiment(
    game_name: str,
    opponent_probs: List[float],
    network_config: NetworkConfig,
    training_config: TrainingConfig,
    device: torch.device,
    output_dirs: Dict[str, str],
    use_adaptive_loss: bool = False
) -> Dict[str, Any]:
    """
    Run training experiment on a single game.
    
    Args:
        game_name: Name of the game to train on
        opponent_probs: List of opponent defection probabilities
        network_config: Network configuration
        training_config: Training configuration
        device: Device for computations
        output_dirs: Output directories
        use_adaptive_loss: Whether to use adaptive loss
        
    Returns:
        Dictionary with experiment results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting experiment on {game_name}")
    
    # Create game (needed for network input size)
    game = GameFactory.create_game(game_name)
    
    # Create network
    network = create_network(network_config, game)
    
    # Create opponents
    opponents = OpponentFactory.create_opponent_set(opponent_probs)
    
    # Create trainer
    trainer = GameTrainer(
        network=network,
        training_config=training_config,
        device=device,
        use_adaptive_loss=use_adaptive_loss
    )
    
    # Run training
    training_results = trainer.train_on_game(
        game_name=game_name,
        opponents=opponents,
        save_dir=output_dirs['checkpoints']
    )
    
    # Evaluate trained network
    evaluation_results = trainer.evaluate(
        game=game,
        opponents=opponents,
        num_sessions=20
    )
    
    # Compile complete results
    complete_results = {
        'game_name': game_name,
        'opponent_probabilities': opponent_probs,
        'network_config': network_config.__dict__,
        'training_config': training_config.__dict__,
        'training_results': training_results,
        'evaluation_results': evaluation_results,
        'network_summary': trainer.network_manager.get_network_summary()
    }
    
    # Save results
    results_file = os.path.join(output_dirs['results'], f'{game_name}_results.pkl')
    save_results(complete_results, results_file)
    
    logger.info(f"Experiment completed for {game_name}")
    return complete_results


def create_experiment_report(
    all_results: List[Dict[str, Any]], 
    output_dirs: Dict[str, str]
):
    """Create a comprehensive experiment report."""
    logger = logging.getLogger(__name__)
    
    # Create summary report
    report = {
        'experiment_summary': {
            'total_games': len(all_results),
            'games_trained': [r['game_name'] for r in all_results],
            'timestamp': datetime.now().isoformat(),
        },
        'game_results': {}
    }
    
    for result in all_results:
        game_name = result['game_name']
        training_result = result['training_results']
        evaluation_result = result['evaluation_results']
        
        # Extract key metrics
        final_metrics = training_result.get('final_metrics', {})
        
        # Safe convergence check
        convergence_info = final_metrics.get('convergence_info', {})
        if isinstance(convergence_info, dict) and convergence_info:
            convergence_achieved = any(
                info.get('converged', False) if isinstance(info, dict) else False
                for info in convergence_info.values()
            )
        else:
            # Fallback - check if convergence_info itself has a 'converged' key
            convergence_achieved = convergence_info.get('converged', False) if isinstance(convergence_info, dict) else False
        
        report['game_results'][game_name] = {
            'training_epochs': final_metrics.get('total_epochs', 0),
            'final_loss': final_metrics.get('best_loss', float('inf')),
            'convergence_achieved': convergence_achieved,
            'evaluation_performance': {
                opponent: {
                    'avg_reward': stats.get('average_reward', 0) if isinstance(stats, dict) else 0,
                    'cooperation_rate': stats.get('cooperation_rate', 0) if isinstance(stats, dict) else 0
                }
                for opponent, stats in (evaluation_result.items() if isinstance(evaluation_result, dict) else [])
            }
        }
    
    # Save report
    report_file = os.path.join(output_dirs['results'], 'experiment_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, cls=NumpyJSONEncoder)
    
    logger.info(f"Experiment report saved to {report_file}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    for game_name, game_results in report['game_results'].items():
        print(f"\n{game_name.upper()}:")
        print(f"  Training epochs: {game_results['training_epochs']}")
        print(f"  Final loss: {game_results['final_loss']:.6f}")
        print(f"  Converged: {game_results['convergence_achieved']}")
        
        print("  Performance vs opponents:")
        for opponent, perf in game_results['evaluation_performance'].items():
            print(f"    {opponent}: reward={perf['avg_reward']:.3f}, coop={perf['cooperation_rate']:.3f}")


def create_multi_game_report(
    results: Dict[str, Any], 
    output_dirs: Dict[str, str]
):
    """Create a comprehensive report for multi-game training experiment."""
    logger = logging.getLogger(__name__)
    
    # Create detailed report
    report = {
        'experiment_summary': {
            'experiment_type': 'multi_game_training',
            'training_games': results['training_games'],
            'test_game': results['test_game'],
            'timestamp': datetime.now().isoformat(),
            'network_parameters': results.get('network_parameters', 'unknown'),
            'device': results.get('device', 'unknown')
        },
        'training_phase': {},
        'testing_phase': {}
    }
    
    # Process training results - single result from multi-game training
    training_result = results['training_results']
    final_metrics = training_result.get('final_metrics', {})
    
    # Extract overall training performance
    report['training_phase']['multi_game_training'] = {
        'training_epochs': final_metrics.get('total_epochs', 0),
        'final_loss': final_metrics.get('best_loss', float('inf')),
        'games_trained': results['training_games'],
        'convergence_achieved': (
            final_metrics.get('convergence_info', {}).get('converged', False) 
            if isinstance(final_metrics.get('convergence_info'), dict) 
            else False
        )
    }
    
    # If there's per-game breakdown in the epoch results, add that too
    if training_result.get('epoch_results') and training_result['epoch_results']:
        last_epoch = training_result['epoch_results'][-1]
        epoch_results = training_result['epoch_results']
        
        if 'per_game_losses' in last_epoch:
            report['training_phase']['per_game_breakdown'] = {}
            
            # Calculate averages over last 10 epochs for each training game
            num_epochs = len(epoch_results)
            last_10_epochs = epoch_results[-10:] if num_epochs >= 10 else epoch_results
            
            for game_name, losses in last_epoch['per_game_losses'].items():
                # Calculate average reward over last 10 epochs
                avg_reward = 0.0
                avg_policy = [0.0, 0.0]  # [cooperate_prob, defect_prob]
                valid_epochs = 0
                
                for epoch_data in last_10_epochs:
                    # Look for game session stats which contains the metrics we need
                    if 'game_session_stats' in epoch_data and game_name in epoch_data['game_session_stats']:
                        game_sessions = epoch_data['game_session_stats'][game_name]
                        if isinstance(game_sessions, list) and len(game_sessions) > 0:
                            # Average across all sessions for this game in this epoch
                            epoch_avg_reward = 0.0
                            epoch_coop_rate = 0.0
                            valid_sessions = 0
                            
                            for session in game_sessions:
                                if isinstance(session, dict):
                                    if 'average_reward' in session:
                                        epoch_avg_reward += session['average_reward']
                                        valid_sessions += 1
                                    if 'cooperation_rate' in session:
                                        epoch_coop_rate += session['cooperation_rate']
                            
                            if valid_sessions > 0:
                                epoch_avg_reward /= valid_sessions
                                epoch_coop_rate /= valid_sessions
                                
                                avg_reward += epoch_avg_reward
                                avg_policy[0] += epoch_coop_rate  # cooperate_prob
                                avg_policy[1] += (1.0 - epoch_coop_rate)  # defect_prob
                                valid_epochs += 1
                    
                    # Also check if there are agent_policy_probs at epoch level
                    elif 'agent_policy_probs' in epoch_data:
                        agent_probs = epoch_data['agent_policy_probs']
                        if isinstance(agent_probs, dict):
                            coop_prob = agent_probs.get('cooperate', 0.0)
                            defect_prob = agent_probs.get('defect', 0.0)
                            avg_policy[0] += coop_prob
                            avg_policy[1] += defect_prob
                            
                            # Use epoch-level reward if available
                            if 'epoch_cumulative_reward' in epoch_data:
                                # Estimate average reward (this is cumulative, so we'll use it as-is for now)
                                avg_reward += float(epoch_data.get('epoch_cumulative_reward', 0.0))
                                valid_epochs += 1
                
                # Normalize averages
                if valid_epochs > 0:
                    avg_reward /= valid_epochs
                    avg_policy[0] /= valid_epochs
                    avg_policy[1] /= valid_epochs
                
                report['training_phase']['per_game_breakdown'][game_name] = {
                    'final_loss': losses.get('total_loss', float('inf')),
                    'rl_loss': losses.get('rl_loss', 0.0),
                    'opponent_prediction_loss': losses.get('opponent_prediction_loss', 0.0),
                    'avg_reward_last_10_epochs': avg_reward,
                    'learned_policy_last_10_epochs': {
                        'cooperate_prob': avg_policy[0],
                        'defect_prob': avg_policy[1]
                    },
                    'epochs_averaged': min(10, num_epochs)
                }
    
    # Process test results
    test_result = results['test_results']
    report['testing_phase'][results['test_game']] = {
        'performance_metrics': {},
        'test_game_summary': {}
    }
    
    # Calculate overall test game averages across all opponents
    total_avg_reward = 0.0
    total_cooperation_rate = 0.0
    valid_opponents = 0
    
    # Extract test performance for each opponent (with safety checks)
    if isinstance(test_result, dict):
        for opponent_name, opponent_stats in test_result.items():
            if isinstance(opponent_stats, dict):
                avg_reward = opponent_stats.get('average_reward', 0)
                cooperation_rate = opponent_stats.get('cooperation_rate', 0)
                
                report['testing_phase'][results['test_game']]['performance_metrics'][opponent_name] = {
                    'avg_reward': avg_reward,
                    'cooperation_rate': cooperation_rate,
                    'win_rate': opponent_stats.get('win_rate', 0)
                }
                
                # Accumulate for overall averages
                total_avg_reward += avg_reward
                total_cooperation_rate += cooperation_rate
                valid_opponents += 1
            else:
                # Fallback for unexpected data structure
                report['testing_phase'][results['test_game']]['performance_metrics'][opponent_name] = {
                    'avg_reward': 0,
                    'cooperation_rate': 0,
                    'win_rate': 0,
                    'error': 'Unexpected data structure'
                }
        
        # Calculate test game summary averages
        if valid_opponents > 0:
            report['testing_phase'][results['test_game']]['test_game_summary'] = {
                'avg_reward_across_opponents': total_avg_reward / valid_opponents,
                'learned_policy_across_opponents': {
                    'cooperate_prob': total_cooperation_rate / valid_opponents,
                    'defect_prob': 1.0 - (total_cooperation_rate / valid_opponents)
                },
                'num_opponents_tested': valid_opponents
            }
        else:
            report['testing_phase'][results['test_game']]['test_game_summary'] = {
                'avg_reward_across_opponents': 0.0,
                'learned_policy_across_opponents': {
                    'cooperate_prob': 0.0,
                    'defect_prob': 0.0
                },
                'num_opponents_tested': 0,
                'error': 'No valid opponent data found'
            }
    else:
        # Handle case where test_result is not a dict
        report['testing_phase'][results['test_game']]['performance_metrics'] = {
            'error': f'Test results have unexpected type: {type(test_result)}'
        }
        report['testing_phase'][results['test_game']]['test_game_summary'] = {
            'error': 'Unable to process test results'
        }
    
    # Save detailed report
    report_file = os.path.join(output_dirs['results'], 'multi_game_experiment_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, cls=NumpyJSONEncoder)
    
    # Save raw results
    raw_results_file = os.path.join(output_dirs['results'], 'raw_results.json')
    with open(raw_results_file, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyJSONEncoder)
    
    logger.info(f"Multi-game experiment report saved to {report_file}")
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("MULTI-GAME TRAINING EXPERIMENT SUMMARY")
    print("="*70)
    print(f"Training Games: {', '.join(results['training_games'])}")
    print(f"Test Game: {results['test_game']}")
    
    print(f"\n--- TRAINING PHASE ---")
    for section_name, training_info in report['training_phase'].items():
        if section_name == 'multi_game_training':
            print(f"\nMULTI-GAME TRAINING:")
            print(f"  Games: {', '.join(training_info.get('games_trained', []))}")
            print(f"  Epochs: {training_info.get('training_epochs', 0)}")
            print(f"  Final Loss: {training_info.get('final_loss', float('inf')):.6f}")
            print(f"  Converged: {training_info.get('convergence_achieved', False)}")
        elif section_name == 'per_game_breakdown':
            print(f"\nPER-GAME BREAKDOWN:")
            for game_name, game_info in training_info.items():
                print(f"  {game_name.upper()}:")
                print(f"    Final Loss: {game_info.get('final_loss', float('inf')):.6f}")
                print(f"    RL Loss: {game_info.get('rl_loss', 0.0):.6f}")
                print(f"    Opponent Pred Loss: {game_info.get('opponent_prediction_loss', 0.0):.6f}")
                print(f"    Avg Reward (last {game_info.get('epochs_averaged', 0)} epochs): {game_info.get('avg_reward_last_10_epochs', 0.0):.4f}")
                
                learned_policy = game_info.get('learned_policy_last_10_epochs', {})
                coop_prob = learned_policy.get('cooperate_prob', 0.0)
                defect_prob = learned_policy.get('defect_prob', 0.0)
                print(f"    Learned Policy (last {game_info.get('epochs_averaged', 0)} epochs): Coop={coop_prob:.3f}, Defect={defect_prob:.3f}")
        else:
            # Fallback for any other structure
            print(f"\n{section_name.upper()}:")
            print(f"  Epochs: {training_info.get('training_epochs', 0)}")
            print(f"  Final Loss: {training_info.get('final_loss', float('inf')):.6f}")
            print(f"  Converged: {training_info.get('convergence_achieved', False)}")
    
    print(f"\n--- TESTING PHASE ({results['test_game'].upper()}) ---")
    
    # Print overall test game summary first
    test_summary = report['testing_phase'][results['test_game']].get('test_game_summary', {})
    if 'error' not in test_summary:
        print(f"\nOVERALL TEST GAME PERFORMANCE:")
        print(f"  Average Reward (across {test_summary.get('num_opponents_tested', 0)} opponents): {test_summary.get('avg_reward_across_opponents', 0.0):.4f}")
        
        learned_policy = test_summary.get('learned_policy_across_opponents', {})
        coop_prob = learned_policy.get('cooperate_prob', 0.0)
        defect_prob = learned_policy.get('defect_prob', 0.0)
        print(f"  Learned Policy (across opponents): Coop={coop_prob:.3f}, Defect={defect_prob:.3f}")
    
    # Print per-opponent breakdown
    print(f"\nPER-OPPONENT BREAKDOWN:")
    test_metrics = report['testing_phase'][results['test_game']]['performance_metrics']
    for opponent, metrics in test_metrics.items():
        if 'error' not in metrics:
            print(f"\n{opponent}:")
            print(f"  Average Reward: {metrics['avg_reward']:.4f}")
            print(f"  Cooperation Rate: {metrics['cooperation_rate']:.2%}")
            if 'win_rate' in metrics:
                print(f"  Win Rate: {metrics['win_rate']:.2%}")
        else:
            print(f"\n{opponent}: {metrics.get('error', 'Unknown error')}")
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_dirs['results']}")


def main():
    """Main experiment function."""
    args = parse_arguments()
    
    # Set random seed
    set_random_seeds(args.seed)
    
    # Setup device
    device = setup_device(args.device)
    
    # Create experiment name and output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"mixed_motive_experiment_{timestamp}"
    output_dirs = create_output_dirs(args.output_dir, experiment_name)
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    log_file = os.path.join(output_dirs['logs'], 'experiment.log')
    logger = setup_logging(log_level, log_file)
    
    logger.info("Starting Mixed-Motive Game Experiment")
    logger.info(f"Arguments: {vars(args)}")
    
    # Parse training games and test game
    training_games = [game.strip() for game in args.training_games.split(',')]
    test_game = args.test_game
    logger.info(f"Training games: {training_games}")
    logger.info(f"Test game: {test_game}")
    
    # Parse opponent probabilities
    opponent_probs = [float(p.strip()) for p in args.opponents.split(',')]
    logger.info(f"Opponent defection probabilities: {opponent_probs}")
    
    # Create configurations
    network_config = NetworkConfig()
    training_config = TrainingConfig(
        num_games_per_partner=args.num_games,
        max_epochs=args.max_epochs
    )
    experiment_config = ExperimentConfig(
        opponent_defection_probs=opponent_probs,
        random_seed=args.seed
    )
    
    # Save configurations
    config_data = {
        'network_config': network_config.__dict__,
        'training_config': training_config.__dict__,
        'experiment_config': experiment_config.__dict__,
        'training_games': training_games,
        'test_game': test_game,
        'args': vars(args)
    }
    
    config_file = os.path.join(output_dirs['base'], 'experiment_config.json')
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2, cls=NumpyJSONEncoder)
    
    # Run multi-game training and testing experiment
    try:
        result = run_multi_game_experiment(
            training_games=training_games,
            test_game=test_game,
            opponent_probs=opponent_probs,
            network_config=network_config,
            training_config=training_config,
            device=device,
            output_dirs=output_dirs,
            use_adaptive_loss=args.adaptive_loss
        )
        
        # Create final report
        create_multi_game_report(result, output_dirs)
        
    except Exception as e:
        logger.error(f"Error in multi-game experiment: {str(e)}")
        raise
    
    logger.info("All experiments completed successfully!")
    print(f"\nResults saved to: {output_dirs['base']}")


if __name__ == "__main__":
    main()