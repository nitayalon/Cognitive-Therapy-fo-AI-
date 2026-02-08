#!/usr/bin/env python3
"""
Train on PD with opponents [0.1, 0.3], test on HD with opponents [0.1, 0.3].
"""

import sys
import json
import os
sys.path.append('src')

import torch
from cognitive_therapy_ai.trainer import GameTrainer
from cognitive_therapy_ai.network import GameLSTM
from cognitive_therapy_ai.opponent import OpponentFactory
from cognitive_therapy_ai.config import TrainingConfig
from cognitive_therapy_ai.games import GameFactory
from cognitive_therapy_ai.utils import setup_logging


def test_pd_train_hd_test_same_opponents():
    setup_logging("INFO")
    print("=== TEST: PD train -> HD test (same opponents) ===")
    train_game = GameFactory.create_game('prisoners-dilemma')
    test_game = GameFactory.create_game('hawk-dove')
    input_size = train_game.get_state_size()

    network = GameLSTM(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.1,
        num_actions=2
    )

    training_config = TrainingConfig(
        num_games_per_partner=100,
        max_epochs=10000
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = GameTrainer(
        network=network,
        training_config=training_config,
        device=device,
        use_adaptive_loss=False
    )

    train_opponents = OpponentFactory.create_opponent_set([0.1, 0.3])

    print("Starting training...")
    trainer.train_on_game(
        game_name='prisoners-dilemma',
        opponents=train_opponents
    )
    print("Training complete. Starting evaluation...")

    evaluation_results = trainer.evaluate(
        game=test_game,
        opponents=train_opponents,
        num_sessions=20,
        enable_detailed_testing=False
    )
    print("Evaluation complete. Writing report...")

    assert evaluation_results

    output_dir = os.path.join('experiments', 'test_reports')
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'pd_train_hd_test_same_opponents.json')
    with open(report_path, 'w') as f:
        json.dump({
            'train_game': 'prisoners-dilemma',
            'test_game': 'hawk-dove',
            'train_opponents': [0.1, 0.3],
            'test_opponents': [0.1, 0.3],
            'evaluation_results': evaluation_results
        }, f, indent=2)

    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    test_pd_train_hd_test_same_opponents()
    print("✅ PD train -> HD test (same opponents) completed")
