"""
Cognitive Therapy for AI: Mixed-Motive Game Training Framework

This package implements a framework for training LSTM networks on mixed-motive games
to study representation changes as a function of training data and opponent behavior.

IMPORT STRUCTURE:
- Lightweight modules (no torch dependency) imported automatically
- Torch-heavy modules (network, trainer, loss) available via explicit import
  to avoid slow package-level imports
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Lightweight imports (no torch dependency) - safe for fast loading
from .games import HawkDove, PrisonersDilemma, StagHunt, GameFactory, MixedMotiveGame, Action
from .opponent import Opponent, OpponentFactory, ProbabilisticOpponent
from .encoding import ObservationEncoder, Outcome
from .best_response import (
    analytic_best_response,
    analytic_best_response_value,
    verify_pd_dominance,
    verify_no_dominance
)

# Torch-heavy imports - available but not imported at package level
# Use: from cognitive_therapy_ai.network import GameLSTM
# Use: from cognitive_therapy_ai.trainer import GameTrainer
# Use: from cognitive_therapy_ai.tom_rl_loss import ToMRLLoss, AdaptiveToMRLLoss, VanillaRLLoss
# Use: from cognitive_therapy_ai.trajectory_utils import save_trajectories_jsonl, load_trajectories_jsonl
# Use: from cognitive_therapy_ai.training_monitor import TrainingMonitor, BatchedTrainingMonitor

__all__ = [
    # Games (lightweight)
    "HawkDove",
    "PrisonersDilemma",
    "StagHunt",
    "GameFactory",
    "MixedMotiveGame",
    "Action",
    # Opponents (lightweight)
    "Opponent",
    "OpponentFactory",
    "ProbabilisticOpponent",
    # Encoding (lightweight - Gate 1)
    "ObservationEncoder",
    "Outcome",
    # Best Response (lightweight - Gate 1)
    "analytic_best_response",
    "analytic_best_response_value",
    "verify_pd_dominance",
    "verify_no_dominance",
]