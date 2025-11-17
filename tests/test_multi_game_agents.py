"""
Multi-Game Agent Testing Script

This script tests trained agents across multiple games to evaluate:
1. Cross-game generalization capabilities
2. Performance consistency across different game types
3. Adaptation to various opponent strategies

Usage:
    python test_multi_game_agents.py --checkpoint path/to/model.pth
    python test_multi_game_agents.py --experiment-dir experiments/mixed_motive_experiment_20241112_123456
    python test_multi_game_agents.py --all-games --opponents 0.1,0.3,0.5,0.7,0.9
"""

import argparse
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import torch
import numpy as np

from cognitive_therapy_ai import (
    GameFactory, 
    OpponentFactory, 
    GameLSTM, 
    GameTrainer,
    NetworkManager
)
from cognitive_therapy_ai.config import NetworkConfig
from cognitive_therapy_ai.utils import setup_logging, create_output_dirs


class MultiGameTester:
    """Test trained agents across multiple games."""
    
    def __init__(self, model_path: str, device: torch.device = None):
        """
        Initialize the multi-game tester.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device for inference
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.network = None
        self.network_manager = None
        
        # Available games for testing
        self.available_games = [
            'prisoners-dilemma',
            'hawk-dove', 
            'battle-of-sexes',
            'stag-hunt'
        ]
        
        self.logger = logging.getLogger(__name__)
        
    def load_model(self) -> bool:
        """Load the trained model from checkpoint."""
        try:
            if not os.path.exists(self.model_path):
                self.logger.error(f"Model checkpoint not found: {self.model_path}")
                return False
                
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Extract model configuration
            model_config = checkpoint.get('model_config', {})
            
            # Create network with same architecture
            self.network = GameLSTM(
                input_size=model_config.get('input_size', 5),
                hidden_size=model_config.get('hidden_size', 128),
                num_layers=model_config.get('num_layers', 4),
                dropout=0.0,  # No dropout during inference
                num_actions=2
            )
            
            # Load trained weights
            self.network.load_state_dict(checkpoint['model_state_dict'])
            self.network.to(self.device)
            self.network.eval()
            
            # Create network manager
            self.network_manager = NetworkManager(self.network, self.device)
            
            self.logger.info(f"Successfully loaded model from {self.model_path}")
            self.logger.info(f"Model configuration: {model_config}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def test_on_game(
        self, 
        game_name: str, 
        opponent_probs: List[float],
        num_sessions: int = 50,
        num_games_per_session: int = 100
    ) -> Dict[str, Any]:
        """
        Test the agent on a specific game.
        
        Args:
            game_name: Name of the game to test on
            opponent_probs: List of opponent defection probabilities
            num_sessions: Number of test sessions per opponent
            num_games_per_session: Number of games per session
            
        Returns:
            Dictionary with test results
        """
        if self.network is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        self.logger.info(f"Testing agent on {game_name}")
        
        # Create game instance
        game = GameFactory.create_game(game_name)
        
        # Create opponents
        opponents = OpponentFactory.create_opponent_set(opponent_probs)
        
        # Create temporary trainer for evaluation
        from cognitive_therapy_ai.config import TrainingConfig
        temp_config = TrainingConfig(
            num_games_per_partner=num_games_per_session,
            learning_rate=0.001,  # Not used for evaluation
            max_epochs=1  # Not used for evaluation
        )
        
        trainer = GameTrainer(
            network=self.network,
            training_config=temp_config,
            device=self.device,
            use_adaptive_loss=False
        )
        
        # Run evaluation
        results = trainer.evaluate(
            game=game,
            opponents=opponents,
            num_sessions=num_sessions
        )
        
        return {
            'game_name': game_name,
            'test_results': results,
            'num_sessions': num_sessions,
            'num_games_per_session': num_games_per_session,
            'total_games': num_sessions * num_games_per_session
        }
    
    def test_all_games(
        self,
        opponent_probs: List[float],
        num_sessions: int = 50,
        num_games_per_session: int = 100,
        games_to_test: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Test the agent on all available games.
        
        Args:
            opponent_probs: List of opponent defection probabilities
            num_sessions: Number of test sessions per opponent
            num_games_per_session: Number of games per session
            games_to_test: Optional list of specific games to test (default: all)
            
        Returns:
            Dictionary with comprehensive test results
        """
        if games_to_test is None:
            games_to_test = self.available_games
            
        self.logger.info(f"Testing agent on {len(games_to_test)} games: {games_to_test}")
        
        all_results = {}
        summary_stats = {}
        
        for game_name in games_to_test:
            try:
                game_results = self.test_on_game(
                    game_name=game_name,
                    opponent_probs=opponent_probs,
                    num_sessions=num_sessions,
                    num_games_per_session=num_games_per_session
                )
                
                all_results[game_name] = game_results
                
                # Calculate summary statistics
                test_results = game_results['test_results']
                if isinstance(test_results, dict):
                    avg_rewards = []
                    cooperation_rates = []
                    
                    for opponent, stats in test_results.items():
                        if isinstance(stats, dict):
                            avg_rewards.append(stats.get('average_reward', 0))
                            cooperation_rates.append(stats.get('cooperation_rate', 0))
                    
                    summary_stats[game_name] = {
                        'mean_reward': np.mean(avg_rewards) if avg_rewards else 0,
                        'std_reward': np.std(avg_rewards) if avg_rewards else 0,
                        'mean_cooperation': np.mean(cooperation_rates) if cooperation_rates else 0,
                        'std_cooperation': np.std(cooperation_rates) if cooperation_rates else 0,
                        'num_opponents_tested': len(avg_rewards)
                    }
                
                self.logger.info(f"Completed testing on {game_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to test on {game_name}: {str(e)}")
                all_results[game_name] = {'error': str(e)}
                summary_stats[game_name] = {'error': str(e)}
        
        return {
            'test_timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'games_tested': games_to_test,
            'opponent_probabilities': opponent_probs,
            'test_parameters': {
                'num_sessions': num_sessions,
                'num_games_per_session': num_games_per_session
            },
            'individual_results': all_results,
            'summary_statistics': summary_stats
        }
    
    def generate_report(self, results: Dict[str, Any], output_dir: str):
        """Generate a comprehensive test report."""
        
        # Save raw results
        results_file = os.path.join(output_dir, 'multi_game_test_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate human-readable report
        report_file = os.path.join(output_dir, 'multi_game_test_report.txt')
        with open(report_file, 'w') as f:
            f.write("MULTI-GAME AGENT TEST REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Test Date: {results['test_timestamp']}\n")
            f.write(f"Model: {results['model_path']}\n")
            f.write(f"Games Tested: {', '.join(results['games_tested'])}\n")
            f.write(f"Opponent Probabilities: {results['opponent_probabilities']}\n\n")
            
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 30 + "\n")
            
            summary_stats = results['summary_statistics']
            for game_name, stats in summary_stats.items():
                if 'error' not in stats:
                    f.write(f"\n{game_name.upper()}:\n")
                    f.write(f"  Mean Reward: {stats['mean_reward']:.4f} ± {stats['std_reward']:.4f}\n")
                    f.write(f"  Mean Cooperation: {stats['mean_cooperation']:.2%} ± {stats['std_cooperation']:.4f}\n")
                    f.write(f"  Opponents Tested: {stats['num_opponents_tested']}\n")
                else:
                    f.write(f"\n{game_name.upper()}: ERROR - {stats['error']}\n")
            
            f.write(f"\nDETAILED RESULTS\n")
            f.write("-" * 30 + "\n")
            
            individual_results = results['individual_results']
            for game_name, game_data in individual_results.items():
                if 'error' not in game_data:
                    f.write(f"\n{game_name.upper()}:\n")
                    test_results = game_data['test_results']
                    
                    if isinstance(test_results, dict):
                        for opponent, stats in test_results.items():
                            if isinstance(stats, dict):
                                f.write(f"  {opponent}:\n")
                                f.write(f"    Average Reward: {stats.get('average_reward', 0):.4f}\n")
                                f.write(f"    Cooperation Rate: {stats.get('cooperation_rate', 0):.2%}\n")
                                if 'win_rate' in stats:
                                    f.write(f"    Win Rate: {stats['win_rate']:.2%}\n")
                else:
                    f.write(f"\n{game_name.upper()}: ERROR - {game_data['error']}\n")
        
        self.logger.info(f"Test report saved to {report_file}")
        
        # Print summary to console
        print("\n" + "=" * 60)
        print("MULTI-GAME AGENT TEST SUMMARY")
        print("=" * 60)
        
        for game_name, stats in summary_stats.items():
            if 'error' not in stats:
                print(f"\n{game_name.upper()}:")
                print(f"  Mean Reward: {stats['mean_reward']:.4f} ± {stats['std_reward']:.4f}")
                print(f"  Mean Cooperation: {stats['mean_cooperation']:.2%}")
                print(f"  Opponents Tested: {stats['num_opponents_tested']}")
            else:
                print(f"\n{game_name.upper()}: ERROR - {stats['error']}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test trained agents on multiple games")
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to model checkpoint file (.pth)'
    )
    
    parser.add_argument(
        '--experiment-dir',
        type=str,
        help='Path to experiment directory (will look for latest checkpoint)'
    )
    
    parser.add_argument(
        '--games',
        type=str,
        default='prisoners-dilemma,hawk-dove,battle-of-sexes,stag-hunt',
        help='Comma-separated list of games to test on'
    )
    
    parser.add_argument(
        '--opponents',
        type=str,
        default='0.1,0.3,0.5,0.7,0.9',
        help='Comma-separated list of opponent defection probabilities'
    )
    
    parser.add_argument(
        '--num-sessions',
        type=int,
        default=50,
        help='Number of test sessions per opponent'
    )
    
    parser.add_argument(
        '--num-games',
        type=int,
        default=100,
        help='Number of games per session'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='multi_game_tests',
        help='Output directory for test results'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help='Device to use for testing'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def find_latest_checkpoint(experiment_dir: str) -> Optional[str]:
    """Find the latest checkpoint in an experiment directory."""
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    
    if not os.path.exists(checkpoints_dir):
        return None
    
    # Look for checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pth')]
    
    if not checkpoint_files:
        return None
    
    # Sort by modification time and get the latest
    checkpoint_paths = [os.path.join(checkpoints_dir, f) for f in checkpoint_files]
    latest_checkpoint = max(checkpoint_paths, key=os.path.getmtime)
    
    return latest_checkpoint


def main():
    """Main testing function."""
    args = parse_arguments()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Determine model path
    model_path = None
    if args.checkpoint:
        model_path = args.checkpoint
    elif args.experiment_dir:
        model_path = find_latest_checkpoint(args.experiment_dir)
        if model_path is None:
            print(f"No checkpoint found in {args.experiment_dir}")
            return
    else:
        print("Either --checkpoint or --experiment-dir must be specified")
        return
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  
    output_dir = os.path.join(args.output_dir, f"multi_game_test_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    log_file = os.path.join(output_dir, 'test.log')
    logger = setup_logging(log_level, log_file)
    
    logger.info("Starting multi-game agent testing")
    logger.info(f"Model path: {model_path}")
    
    # Parse games and opponents
    games_to_test = [game.strip() for game in args.games.split(',')]
    opponent_probs = [float(p.strip()) for p in args.opponents.split(',')]
    
    logger.info(f"Testing games: {games_to_test}")
    logger.info(f"Opponent probabilities: {opponent_probs}")
    
    # Create tester and load model
    tester = MultiGameTester(model_path, device)
    
    if not tester.load_model():
        print("Failed to load model. Exiting.")
        return
    
    # Run tests
    try:
        results = tester.test_all_games(
            opponent_probs=opponent_probs,
            num_sessions=args.num_sessions,
            num_games_per_session=args.num_games,
            games_to_test=games_to_test
        )
        
        # Generate report
        tester.generate_report(results, output_dir)
        
        print(f"\nTesting completed successfully!")
        print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")
        print(f"Testing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()