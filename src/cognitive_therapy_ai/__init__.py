"""
Cognitive Therapy for AI: Mixed-Motive Game Training Framework

This package implements a framework for training LSTM networks on mixed-motive games
to study representation changes as a function of training data and opponent behavior.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .games import HawkDove, PrisonersDilemma, BattleOfSexes, StagHunt, GameFactory, MixedMotiveGame
from .opponent import Opponent, OpponentFactory
from .network import GameLSTM
from .trainer import GameTrainer
from .tom_rl_loss import ToMRLLoss, AdaptiveToMRLLoss
from .training_monitor import TrainingMonitor, BatchedTrainingMonitor

__all__ = [
    "HawkDove",
    "PrisonersDilemma", 
    "BattleOfSexes",
    "StagHunt",
    "GameFactory",
    "MixedMotiveGame",
    "Opponent",
    "OpponentFactory",
    "GameLSTM",
    "GameTrainer",
    "ToMRLLoss",
    "AdaptiveToMRLLoss",
    "TrainingMonitor",
    "BatchedTrainingMonitor"
]