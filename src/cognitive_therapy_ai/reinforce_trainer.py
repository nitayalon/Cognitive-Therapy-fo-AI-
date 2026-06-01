"""
REINFORCE with GAE for session-based training.

Key features:
1. Session-based structure: T=100 consecutive games per session
2. Hidden state persists across all T games within session
3. Generalized Advantage Estimation (GAE) with λ=0.95
4. Two training modes:
   - RL mode: Learn from rewards via policy gradient
   - BC mode: Supervised learning from analytic_best_response

Mathematical formulation:
    Policy gradient: ∇J = E[∑_t A_t ∇ log π(a_t|s_t)]
    GAE advantage: A_t = ∑_{l=0}^∞ (γλ)^l δ_{t+l}
    TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
    
Where:
    γ = 0.99 (discount factor)
    λ = 0.95 (GAE parameter)
    T = 100 (games per session)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from dataclasses import dataclass

from .representation_agent import RepresentationAgent
from .encoding import ObservationEncoder, Action
from .games import MixedMotiveGame
from .opponent import ProbabilisticOpponent
from .best_response import analytic_best_response


@dataclass
class SessionStats:
    """Statistics for a single training session."""
    total_return: float
    mean_reward: float
    policy_loss: float
    value_loss: float
    br_accuracy: float  # Fraction of actions matching best response
    num_games: int


@dataclass
class TrajectoryStep:
    """Single step in a trajectory with full behavioral data."""
    # Agent data
    observation: torch.Tensor  # (d_in,)
    action: int  # 0 or 1
    reward: float
    policy_logits: torch.Tensor  # (2,)
    value: torch.Tensor  # (1,)
    
    # Opponent data
    opponent_action: int  # 0 or 1
    opponent_reward: float
    opponent_action_prob: float  # For probabilistic opponents
    
    # Episode metadata
    episode_id: int
    timestep: int
    done: bool  # True if end of session


class SessionEnvironment:
    """
    Environment for session-based training.
    
    A session consists of T consecutive games with the same opponent.
    Hidden state persists across all T games within a session.
    """
    
    def __init__(
        self,
        game: MixedMotiveGame,
        opponent: ProbabilisticOpponent,
        encoder: ObservationEncoder,
        T: int = 100,
        game_name: Optional[str] = None
    ):
        self.game = game
        self.opponent = opponent
        self.encoder = encoder
        self.T = T
        self.game_name = game_name
        
        # Session state
        self.game_count = 0
        self.agent_last_action: Optional[Action] = None
        self.opponent_last_action: Optional[Action] = None
    
    def reset(self) -> np.ndarray:
        """
        Reset session to initial state.
        
        Returns:
            Initial observation (START token)
        """
        self.game_count = 0
        self.agent_last_action = None
        self.opponent_last_action = None
        
        return self.encoder.encode_start_token(game_name=self.game_name)
    
    def step(self, agent_action: int) -> Tuple[np.ndarray, float, bool, int, float, float]:
        """
        Execute one game round.
        
        Args:
            agent_action: 0 (Cooperate) or 1 (Defect)
        
        Returns:
            next_obs: Observation encoding current outcome
            agent_reward: Agent's payoff
            done: True if session is complete (game_count >= T)
            opponent_action: Opponent's action (0 or 1)
            opponent_reward: Opponent's payoff
            opponent_action_prob: Probability opponent assigned to their action
        """
        # Convert to Action enum
        agent_action_enum = Action.COOPERATE if agent_action == 0 else Action.DEFECT
        
        # Opponent chooses action (memoryless)
        opponent_action_enum = self.opponent.play_action(game_history=[], round_number=0)
        opponent_action_int = opponent_action_enum.value
        
        # Get opponent action probability (for probabilistic opponents)
        # Check if opponent has a strategy with defection_probability attribute
        if hasattr(self.opponent, 'strategy') and hasattr(self.opponent.strategy, 'defection_probability'):
            if opponent_action_enum == Action.COOPERATE:
                opponent_action_prob = 1.0 - self.opponent.strategy.defection_probability
            else:
                opponent_action_prob = self.opponent.strategy.defection_probability
        else:
            # For deterministic opponents, set prob to 1.0
            opponent_action_prob = 1.0
        
        # Get payoffs (transpose for opponent's perspective)
        payoff_matrix = self.game.get_payoff_matrix()
        agent_reward = payoff_matrix[agent_action, opponent_action_int]
        opponent_reward = payoff_matrix[opponent_action_int, agent_action]  # Transpose
        
        # Update state
        self.agent_last_action = agent_action_enum
        self.opponent_last_action = opponent_action_enum
        self.game_count += 1
        
        # Create next observation
        next_obs = self.encoder.encode_observation(
            agent_last_action=agent_action_enum,
            opponent_last_action=opponent_action_enum,
            game_name=self.game_name
        )
        
        # Check if session is done
        done = (self.game_count >= self.T)
        
        return (
            next_obs, 
            float(agent_reward), 
            done, 
            opponent_action_int,
            float(opponent_reward),
            float(opponent_action_prob)
        )


class REINFORCETrainer:
    """
    REINFORCE trainer with GAE for session-based learning.
    
    Supports two modes:
    1. RL mode: Learn from environment rewards via policy gradient
    2. BC mode: Supervised learning from analytic best response
    """
    
    def __init__(
        self,
        agent: RepresentationAgent,
        optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        value_loss_coef: float = 0.5,
        device: Optional[torch.device] = None
    ):
        self.agent = agent
        self.optimizer = optimizer
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_loss_coef = value_loss_coef
        self.device = device if device is not None else torch.device('cpu')
    
    def compute_gae_advantages(
        self,
        rewards: List[float],
        values: List[torch.Tensor],
        dones: List[bool]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and returns.
        
        Args:
            rewards: List of rewards [r_0, r_1, ..., r_{T-1}]
            values: List of value estimates [V(s_0), V(s_1), ..., V(s_T)]
            dones: List of done flags
        
        Returns:
            advantages: (T,) tensor of GAE advantages
            returns: (T,) tensor of discounted returns
        """
        T = len(rewards)
        advantages = []
        gae = 0.0
        
        # Compute GAE advantages backwards
        for t in reversed(range(T)):
            if dones[t]:
                # End of session: no next value
                delta = rewards[t] - values[t].item()
                gae = delta
            else:
                # TD error: δ_t = r_t + γ V(s_{t+1}) - V(s_t)
                delta = rewards[t] + self.gamma * values[t + 1].item() - values[t].item()
                # GAE: A_t = δ_t + γλ A_{t+1}
                gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        
        # Returns: R_t = A_t + V(s_t)
        values_tensor = torch.cat([v for v in values[:-1]], dim=0).squeeze()  # (T,)
        returns = advantages + values_tensor
        
        return advantages, returns
    
    def train_session_rl(
        self,
        env: SessionEnvironment,
        max_steps: int = 100,
        return_trajectory: bool = False
    ) -> Union[Tuple[SessionStats, List[TrajectoryStep]], SessionStats]:
        """
        Train agent on one session using REINFORCE with GAE.
        
        Args:
            env: Session environment
            max_steps: Maximum steps per session (should match T)
            return_trajectory: If True, return (stats, trajectory) instead of just stats
        
        Returns:
            SessionStats with training metrics, and optionally the full trajectory
        """
        self.agent.train()
        
        # Collect trajectory
        trajectory: List[TrajectoryStep] = []
        obs = env.reset()
        hidden_state = self.agent.reset_hidden_state(batch_size=1)
        
        total_reward = 0.0
        
        for step in range(max_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            
            # Forward pass
            policy_logits, value, hidden_state = self.agent.forward(
                obs_tensor.unsqueeze(0),
                hidden_state
            )
            policy_logits = policy_logits.squeeze(0)  # (2,)
            value = value.squeeze(0)  # (1,)
            
            # Sample action
            policy_probs = F.softmax(policy_logits, dim=0)
            action = torch.multinomial(policy_probs, num_samples=1).item()
            
            # Environment step
            next_obs, reward, done, opp_action, opp_reward, opp_prob = env.step(action)
            total_reward += reward
            
            # Compute agent action probability
            policy_probs_np = policy_probs.detach().cpu().numpy()
            agent_action_prob = float(policy_probs_np[action])
            
            # Store trajectory
            trajectory.append(TrajectoryStep(
                observation=obs_tensor,
                action=action,
                reward=reward,
                policy_logits=policy_logits,
                value=value,
                opponent_action=opp_action,
                opponent_reward=opp_reward,
                opponent_action_prob=opp_prob,
                episode_id=-1,  # Set by caller
                timestep=step,
                done=done
            ))
            
            obs = next_obs
            
            if done:
                break
        
        # Compute final value (for bootstrapping)
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _, final_value, _ = self.agent.forward(obs_tensor.unsqueeze(0), hidden_state)
            final_value = final_value.squeeze(0)
        
        # Extract trajectory components
        rewards = [step.reward for step in trajectory]
        values = [step.value for step in trajectory] + [final_value]
        dones = [step.done for step in trajectory]
        
        # Compute GAE advantages
        advantages, returns = self.compute_gae_advantages(rewards, values, dones)
        
        # Normalize advantages (optional, improves stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute policy loss
        policy_losses = []
        for step, advantage in zip(trajectory, advantages):
            log_prob = F.log_softmax(step.policy_logits, dim=0)[step.action]
            policy_loss = -log_prob * advantage.detach()  # REINFORCE gradient
            policy_losses.append(policy_loss)
        
        policy_loss = torch.stack(policy_losses).mean()
        
        # Compute value loss (MSE between predicted values and returns)
        value_predictions = torch.cat([step.value for step in trajectory], dim=0).squeeze()
        value_loss = F.mse_loss(value_predictions, returns.detach())
        
        # Total loss
        total_loss = policy_loss + self.value_loss_coef * value_loss
        
        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Compute BR accuracy (for monitoring)
        br_accuracy = self._compute_br_accuracy(env, trajectory)
        
        stats = SessionStats(
            total_return=total_reward,
            mean_reward=total_reward / len(trajectory),
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            br_accuracy=br_accuracy,
            num_games=len(trajectory)
        )
        
        if return_trajectory:
            return stats, trajectory
        else:
            return stats
    
    def train_session_bc(
        self,
        env: SessionEnvironment,
        max_steps: int = 100
    ) -> SessionStats:
        """
        Train agent using behavioral cloning (supervised learning from best response).
        
        Args:
            env: Session environment
            max_steps: Maximum steps per session
        
        Returns:
            SessionStats with training metrics
        """
        self.agent.train()
        
        # Get payoff matrix and opponent p_coop
        payoff_matrix = env.game.get_payoff_matrix()
        p_coop = 1.0 - env.opponent.defection_probability
        
        # Compute best response action
        br_action_enum = analytic_best_response(payoff_matrix, p_coop)
        br_action = 0 if br_action_enum == Action.COOPERATE else 1
        
        # Collect trajectory
        trajectory: List[TrajectoryStep] = []
        obs = env.reset()
        hidden_state = self.agent.reset_hidden_state(batch_size=1)
        
        total_reward = 0.0
        
        for step in range(max_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            
            # Forward pass
            policy_logits, value, hidden_state = self.agent.forward(
                obs_tensor.unsqueeze(0),
                hidden_state
            )
            policy_logits = policy_logits.squeeze(0)
            value = value.squeeze(0)
            
            # Use best response action
            action = br_action
            
            # Environment step
            next_obs, reward, done, opp_action, opp_reward, opp_prob = env.step(action)
            total_reward += reward
            
            # Compute agent action probability
            policy_probs = F.softmax(policy_logits, dim=0)
            agent_action_prob = float(policy_probs[action].detach().cpu().numpy())
            
            # Store trajectory
            trajectory.append(TrajectoryStep(
                observation=obs_tensor,
                action=action,
                reward=reward,
                policy_logits=policy_logits,
                value=value,
                opponent_action=opp_action,
                opponent_reward=opp_reward,
                opponent_action_prob=opp_prob,
                episode_id=-1,
                timestep=step,
                done=done
            ))
            
            obs = next_obs
            
            if done:
                break
        
        # Compute supervised loss (cross-entropy)
        policy_losses = []
        for step in trajectory:
            # Target: best response action
            target = torch.tensor([br_action], dtype=torch.long, device=self.device)
            loss = F.cross_entropy(step.policy_logits.unsqueeze(0), target)
            policy_losses.append(loss)
        
        policy_loss = torch.stack(policy_losses).mean()
        
        # Value loss (predict observed returns)
        # For BC, we can use actual rewards as targets
        returns = []
        G = 0.0
        for reward in reversed([step.reward for step in trajectory]):
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        value_predictions = torch.cat([step.value for step in trajectory], dim=0).squeeze()
        value_loss = F.mse_loss(value_predictions, returns_tensor)
        
        # Total loss
        total_loss = policy_loss + self.value_loss_coef * value_loss
        
        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # BR accuracy (should be 100% in BC mode)
        br_accuracy = 1.0
        
        return SessionStats(
            total_return=total_reward,
            mean_reward=total_reward / len(trajectory),
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            br_accuracy=br_accuracy,
            num_games=len(trajectory)
        )
    
    def _compute_br_accuracy(
        self,
        env: SessionEnvironment,
        trajectory: List[TrajectoryStep]
    ) -> float:
        """
        Compute fraction of actions matching best response.
        
        Args:
            env: Session environment
            trajectory: Collected trajectory
        
        Returns:
            Fraction of actions matching BR (0.0 to 1.0)
        """
        payoff_matrix = env.game.get_payoff_matrix()
        p_coop = 1.0 - env.opponent.strategy.defection_probability
        
        br_action_enum = analytic_best_response(payoff_matrix, p_coop)
        br_action = 0 if br_action_enum == Action.COOPERATE else 1
        
        matches = sum(1 for step in trajectory if step.action == br_action)
        return matches / len(trajectory) if trajectory else 0.0
    
    def evaluate_session(
        self,
        env: SessionEnvironment,
        max_steps: int = 100,
        deterministic: bool = True
    ) -> SessionStats:
        """
        Evaluate agent on one session (no training).
        
        Args:
            env: Session environment
            max_steps: Maximum steps per session
            deterministic: If True, use greedy action selection
        
        Returns:
            SessionStats with evaluation metrics
        """
        self.agent.eval()
        
        trajectory: List[TrajectoryStep] = []
        obs = env.reset()
        hidden_state = self.agent.reset_hidden_state(batch_size=1)
        
        total_reward = 0.0
        
        with torch.no_grad():
            for step in range(max_steps):
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                
                # Forward pass
                policy_logits, value, hidden_state = self.agent.forward(
                    obs_tensor.unsqueeze(0),
                    hidden_state
                )
                policy_logits = policy_logits.squeeze(0)
                value = value.squeeze(0)
                
                # Select action
                if deterministic:
                    policy_probs = F.softmax(policy_logits, dim=0)
                    action = torch.argmax(policy_probs).item()
                    agent_action_prob = float(policy_probs[action].detach().cpu().numpy())
                else:
                    policy_probs = F.softmax(policy_logits, dim=0)
                    action = torch.multinomial(policy_probs, num_samples=1).item()
                    agent_action_prob = float(policy_probs[action].detach().cpu().numpy())
                
                # Environment step
                next_obs, reward, done, opp_action, opp_reward, opp_prob = env.step(action)
                total_reward += reward
                
                # Store trajectory
                trajectory.append(TrajectoryStep(
                    observation=obs_tensor,
                    action=action,
                    reward=reward,
                    policy_logits=policy_logits,
                    value=value,
                    opponent_action=opp_action,
                    opponent_reward=opp_reward,
                    opponent_action_prob=opp_prob,
                    episode_id=episode,
                    timestep=step,
                    done=done
                ))
                
                obs = next_obs
                
                if done:
                    break
        
        # Compute BR accuracy
        br_accuracy = self._compute_br_accuracy(env, trajectory)
        
        return SessionStats(
            total_return=total_reward,
            mean_reward=total_reward / len(trajectory),
            policy_loss=0.0,  # Not computed during evaluation
            value_loss=0.0,
            br_accuracy=br_accuracy,
            num_games=len(trajectory)
        )


def train_to_convergence(
    trainer: REINFORCETrainer,
    env: SessionEnvironment,
    max_episodes: int = 1000,
    convergence_window: int = 50,
    convergence_threshold: float = 0.01,
    mode: str = "rl",
    verbose: bool = False
) -> Dict[str, List[float]]:
    """
    Train agent until convergence.
    
    Args:
        trainer: REINFORCE trainer
        env: Session environment
        max_episodes: Maximum training episodes
        convergence_window: Window for checking convergence
        convergence_threshold: Threshold for return variance
        mode: "rl" or "bc"
        verbose: Print progress
    
    Returns:
        Training history dictionary
    """
    history = {
        'returns': [],
        'policy_losses': [],
        'value_losses': [],
        'br_accuracies': []
    }
    
    for episode in range(max_episodes):
        if mode == "rl":
            stats = trainer.train_session_rl(env)
        elif mode == "bc":
            stats = trainer.train_session_bc(env)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        history['returns'].append(stats.total_return)
        history['policy_losses'].append(stats.policy_loss)
        history['value_losses'].append(stats.value_loss)
        history['br_accuracies'].append(stats.br_accuracy)
        
        if verbose and (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}: Return={stats.total_return:.2f}, "
                  f"BR_acc={stats.br_accuracy:.3f}")
        
        # Check convergence
        if episode >= convergence_window:
            recent_returns = history['returns'][-convergence_window:]
            return_std = np.std(recent_returns)
            if return_std < convergence_threshold:
                if verbose:
                    print(f"Converged at episode {episode + 1}")
                break
    
    return history
