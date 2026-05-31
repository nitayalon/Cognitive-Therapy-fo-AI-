"""
Gate 6: Attribution Analysis for FSM States

This module computes input attribution for trained agents to understand
which input features drive state transitions and action selection.

Key techniques:
1. Integrated Gradients: Attribute output to input features
2. Saliency Maps: Identify critical input dimensions
3. Feature Importance: Rank input features by influence

This enables answering: "What aspects of the environment does the agent's
representation capture?" which is central to the research claim.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

from cognitive_therapy_ai.representation_agent import RepresentationAgent
from cognitive_therapy_ai.encoding import ObservationEncoder
from cognitive_therapy_ai.games import Action


@dataclass
class AttributionResult:
    """Attribution scores for a single observation."""
    observation: np.ndarray  # Original one-hot observation
    attributions: np.ndarray  # Attribution scores (same shape as observation)
    predicted_action: Action  # Agent's predicted action
    policy_probs: np.ndarray  # Policy distribution [p(C), p(D)]
    baseline: np.ndarray  # Baseline (reference) input
    method: str  # Attribution method name


@dataclass
class SaliencyMap:
    """Saliency map for state transitions."""
    from_observation: np.ndarray  # Observation before transition
    to_observation: np.ndarray  # Observation after transition
    saliency_scores: np.ndarray  # Gradient magnitudes
    from_action: Action  # Action taken
    from_hidden_state: np.ndarray  # Hidden state before
    to_hidden_state: np.ndarray  # Hidden state after


@dataclass
class FeatureImportance:
    """Aggregated feature importance across observations."""
    feature_names: List[str]  # Human-readable feature names
    mean_attributions: np.ndarray  # Mean attribution per feature
    std_attributions: np.ndarray  # Std attribution per feature
    total_observations: int  # Number of observations analyzed


class IntegratedGradients:
    """
    Integrated Gradients attribution method.
    
    Computes the integral of gradients along a path from baseline to input.
    This satisfies axioms like completeness and sensitivity.
    
    Reference: Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks"
    """
    
    def __init__(
        self,
        agent: RepresentationAgent,
        encoder: ObservationEncoder,
        device: torch.device = None
    ):
        self.agent = agent
        self.encoder = encoder
        self.device = device or torch.device('cpu')
        self.agent.eval()
    
    def attribute(
        self,
        observation: np.ndarray,
        hidden_state: Tuple[torch.Tensor, torch.Tensor],
        baseline: Optional[np.ndarray] = None,
        n_steps: int = 50,
        target_action: Optional[int] = None
    ) -> AttributionResult:
        """
        Compute integrated gradients for an observation.
        
        Args:
            observation: One-hot encoded observation
            hidden_state: Current LSTM hidden state
            baseline: Reference input (default: all zeros)
            n_steps: Number of interpolation steps
            target_action: Action to attribute (default: predicted action)
        
        Returns:
            AttributionResult with attribution scores
        """
        # Default baseline: START token (all zeros)
        if baseline is None:
            baseline = self.encoder.encode_start_token()
        
        # Convert to tensors
        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device)
        baseline_tensor = torch.tensor(baseline, dtype=torch.float32, device=self.device)
        
        # Get predicted action if not specified
        if target_action is None:
            with torch.no_grad():
                policy_logits, _, _ = self.agent.forward(obs_tensor.unsqueeze(0), hidden_state)
                target_action = torch.argmax(policy_logits, dim=1).item()
        
        # Compute gradients along path
        attributions = torch.zeros_like(obs_tensor)
        
        for step in range(n_steps):
            # Interpolate between baseline and input
            alpha = (step + 1) / n_steps
            interpolated = baseline_tensor + alpha * (obs_tensor - baseline_tensor)
            interpolated.requires_grad_(True)
            
            # Forward pass
            policy_logits, _, _ = self.agent.forward(interpolated.unsqueeze(0), hidden_state)
            target_logit = policy_logits[0, target_action]
            
            # Backward pass
            target_logit.backward()
            
            # Accumulate gradients
            if interpolated.grad is not None:
                attributions += interpolated.grad
        
        # Average and scale by (input - baseline)
        attributions = attributions / n_steps
        attributions = attributions * (obs_tensor - baseline_tensor)
        
        # Get final policy for result
        with torch.no_grad():
            policy_logits, _, _ = self.agent.forward(obs_tensor.unsqueeze(0), hidden_state)
            policy_probs = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            predicted_action = Action(torch.argmax(policy_logits, dim=1).item())
        
        return AttributionResult(
            observation=observation,
            attributions=attributions.detach().cpu().numpy(),
            predicted_action=predicted_action,
            policy_probs=policy_probs,
            baseline=baseline,
            method='integrated_gradients'
        )
    
    def batch_attribute(
        self,
        observations: List[np.ndarray],
        hidden_states: List[Tuple[torch.Tensor, torch.Tensor]],
        baseline: Optional[np.ndarray] = None,
        n_steps: int = 50
    ) -> List[AttributionResult]:
        """Compute attributions for multiple observations."""
        results = []
        for obs, h_state in zip(observations, hidden_states):
            result = self.attribute(obs, h_state, baseline, n_steps)
            results.append(result)
        return results


class SaliencyAnalyzer:
    """
    Compute saliency maps via gradient-based methods.
    
    Saliency identifies which input dimensions have largest gradient magnitudes,
    indicating sensitivity of the output to those inputs.
    """
    
    def __init__(
        self,
        agent: RepresentationAgent,
        device: torch.device = None
    ):
        self.agent = agent
        self.device = device or torch.device('cpu')
        self.agent.eval()
    
    def compute_saliency(
        self,
        observation: np.ndarray,
        hidden_state: Tuple[torch.Tensor, torch.Tensor],
        target_action: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute saliency map for observation.
        
        Args:
            observation: One-hot encoded observation
            hidden_state: LSTM hidden state
            target_action: Action to compute saliency for (default: predicted)
        
        Returns:
            Saliency scores (gradient magnitudes)
        """
        # Convert to tensor
        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device)
        obs_tensor.requires_grad_(True)
        
        # Forward pass
        policy_logits, _, _ = self.agent.forward(obs_tensor.unsqueeze(0), hidden_state)
        
        # Get target action
        if target_action is None:
            target_action = torch.argmax(policy_logits, dim=1).item()
        
        # Backward pass
        target_logit = policy_logits[0, target_action]
        target_logit.backward()
        
        # Saliency = absolute gradient
        saliency = torch.abs(obs_tensor.grad).detach().cpu().numpy()
        
        return saliency
    
    def compute_saliency_map(
        self,
        from_obs: np.ndarray,
        to_obs: np.ndarray,
        from_hidden: np.ndarray,
        to_hidden: np.ndarray,
        from_action: Action,
        from_hidden_state: Tuple[torch.Tensor, torch.Tensor]
    ) -> SaliencyMap:
        """
        Compute saliency map for a state transition.
        
        Args:
            from_obs: Observation before transition
            to_obs: Observation after transition
            from_hidden: Hidden state before
            to_hidden: Hidden state after
            from_action: Action taken
            from_hidden_state: LSTM state before transition
        
        Returns:
            SaliencyMap object
        """
        saliency = self.compute_saliency(from_obs, from_hidden_state, from_action.value)
        
        return SaliencyMap(
            from_observation=from_obs,
            to_observation=to_obs,
            saliency_scores=saliency,
            from_action=from_action,
            from_hidden_state=from_hidden,
            to_hidden_state=to_hidden
        )


class AttributionAnalyzer:
    """
    High-level attribution analysis for trained agents.
    
    Provides utilities to:
    - Compute feature importance across trajectories
    - Identify which input dimensions matter most
    - Analyze attribution patterns by game/opponent
    """
    
    def __init__(
        self,
        agent: RepresentationAgent,
        encoder: ObservationEncoder,
        device: torch.device = None
    ):
        self.agent = agent
        self.encoder = encoder
        self.device = device or torch.device('cpu')
        
        self.ig = IntegratedGradients(agent, encoder, device)
        self.saliency = SaliencyAnalyzer(agent, device)
    
    def analyze_trajectory(
        self,
        observations: List[np.ndarray],
        actions: List[Action],
        hidden_states: List[np.ndarray],
        lstm_states: List[Tuple[torch.Tensor, torch.Tensor]],
        method: str = 'integrated_gradients'
    ) -> List[AttributionResult]:
        """
        Analyze attribution across a trajectory.
        
        Args:
            observations: List of observations
            actions: List of actions taken
            hidden_states: List of LSTM hidden states (h_t)
            lstm_states: List of LSTM (h, c) tuples
            method: 'integrated_gradients' or 'saliency'
        
        Returns:
            List of AttributionResult objects
        """
        if method == 'integrated_gradients':
            return self.ig.batch_attribute(observations, lstm_states)
        elif method == 'saliency':
            results = []
            for obs, action, lstm_state in zip(observations, actions, lstm_states):
                saliency = self.saliency.compute_saliency(obs, lstm_state, action.value)
                # Convert to AttributionResult format
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                    policy_logits, _, _ = self.agent.forward(obs_tensor.unsqueeze(0), lstm_state)
                    policy_probs = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
                    predicted_action = Action(torch.argmax(policy_logits, dim=1).item())
                
                results.append(AttributionResult(
                    observation=obs,
                    attributions=saliency,
                    predicted_action=predicted_action,
                    policy_probs=policy_probs,
                    baseline=self.encoder.encode_start_token(),
                    method='saliency'
                ))
            return results
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def compute_feature_importance(
        self,
        attribution_results: List[AttributionResult]
    ) -> FeatureImportance:
        """
        Aggregate attributions to compute feature importance.
        
        Args:
            attribution_results: List of attribution results
        
        Returns:
            FeatureImportance with mean and std per feature
        """
        if len(attribution_results) == 0:
            raise ValueError("No attribution results provided")
        
        # Stack attributions
        all_attributions = np.array([r.attributions for r in attribution_results])
        
        # Compute statistics
        mean_attr = np.mean(np.abs(all_attributions), axis=0)
        std_attr = np.std(np.abs(all_attributions), axis=0)
        
        # Get feature names from encoder
        feature_names = self._get_feature_names()
        
        return FeatureImportance(
            feature_names=feature_names,
            mean_attributions=mean_attr,
            std_attributions=std_attr,
            total_observations=len(attribution_results)
        )
    
    def _get_feature_names(self) -> List[str]:
        """Generate human-readable feature names."""
        input_dim = self.encoder.get_input_dim()
        
        if self.encoder.input_condition == 'no_game':
            # 8D: agent_action(2) + opp_action(2) + outcome(4)
            return [
                'agent_cooperate', 'agent_defect',
                'opp_cooperate', 'opp_defect',
                'outcome_CC', 'outcome_CD', 'outcome_DC', 'outcome_DD'
            ]
        elif self.encoder.input_condition == 'game_tag':
            # 11D: history(8) + game_tag(3)
            return [
                'agent_cooperate', 'agent_defect',
                'opp_cooperate', 'opp_defect',
                'outcome_CC', 'outcome_CD', 'outcome_DC', 'outcome_DD',
                'game_PD', 'game_SH', 'game_HD'
            ]
        else:
            # Generic names
            return [f'feature_{i}' for i in range(input_dim)]
    
    def rank_features_by_importance(
        self,
        feature_importance: FeatureImportance,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Rank features by mean attribution.
        
        Args:
            feature_importance: FeatureImportance object
            top_k: Number of top features to return
        
        Returns:
            List of (feature_name, mean_attribution) tuples
        """
        indices = np.argsort(feature_importance.mean_attributions)[::-1]
        ranked = [
            (feature_importance.feature_names[i], feature_importance.mean_attributions[i])
            for i in indices[:top_k]
        ]
        return ranked
    
    def analyze_by_input_symbol(
        self,
        attribution_results: List[AttributionResult],
        observations: List[np.ndarray]
    ) -> Dict[str, FeatureImportance]:
        """
        Analyze attribution separately for each input symbol (CC, CD, DC, DD).
        
        Args:
            attribution_results: List of attribution results
            observations: Corresponding observations
        
        Returns:
            Dict mapping input symbol to FeatureImportance
        """
        symbol_results = defaultdict(list)
        
        for result, obs in zip(attribution_results, observations):
            symbol = self._observation_to_symbol(obs)
            symbol_results[symbol].append(result)
        
        # Compute feature importance per symbol
        importance_by_symbol = {}
        for symbol, results in symbol_results.items():
            if len(results) > 0:
                importance_by_symbol[symbol] = self.compute_feature_importance(results)
        
        return importance_by_symbol
    
    def _observation_to_symbol(self, obs: np.ndarray) -> str:
        """Convert observation to symbol (START, CC, CD, DC, DD)."""
        history_dim = self.encoder.history_dim
        
        # Check if START token
        if obs[:history_dim].sum() == 0:
            return "START"
        
        # Extract outcome from one-hot encoding
        outcome_start = 4  # After agent(2) and opp(2)
        outcome_bits = obs[outcome_start:outcome_start+4]
        
        outcome_map = {0: "CC", 1: "CD", 2: "DC", 3: "DD"}
        outcome_idx = np.argmax(outcome_bits)
        return outcome_map[outcome_idx]


def visualize_attribution(
    attribution_result: AttributionResult,
    feature_names: List[str],
    title: str = "Attribution Scores"
) -> str:
    """
    Create text visualization of attribution scores.
    
    Args:
        attribution_result: Attribution result to visualize
        feature_names: Feature names
        title: Plot title
    
    Returns:
        String representation of attribution scores
    """
    lines = [f"\n{title}", "=" * len(title)]
    lines.append(f"Predicted Action: {attribution_result.predicted_action.name}")
    lines.append(f"Policy: C={attribution_result.policy_probs[0]:.3f}, D={attribution_result.policy_probs[1]:.3f}")
    lines.append(f"Method: {attribution_result.method}\n")
    
    # Sort features by attribution magnitude
    indices = np.argsort(np.abs(attribution_result.attributions))[::-1]
    
    lines.append("Feature Attributions (ranked by magnitude):")
    for i in indices:
        attr_val = attribution_result.attributions[i]
        obs_val = attribution_result.observation[i]
        name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        lines.append(f"  {name:20s}: {attr_val:+.4f}  (input={obs_val:.0f})")
    
    return "\n".join(lines)


def visualize_feature_importance(
    feature_importance: FeatureImportance,
    top_k: int = 10,
    title: str = "Feature Importance"
) -> str:
    """
    Create text visualization of feature importance.
    
    Args:
        feature_importance: FeatureImportance object
        top_k: Number of top features to show
        title: Title
    
    Returns:
        String representation
    """
    lines = [f"\n{title}", "=" * len(title)]
    lines.append(f"Total Observations: {feature_importance.total_observations}\n")
    
    # Rank features
    indices = np.argsort(feature_importance.mean_attributions)[::-1]
    
    lines.append("Top Features (by mean absolute attribution):")
    for rank, i in enumerate(indices[:top_k], 1):
        mean_val = feature_importance.mean_attributions[i]
        std_val = feature_importance.std_attributions[i]
        name = feature_importance.feature_names[i]
        lines.append(f"  {rank:2d}. {name:20s}: {mean_val:.4f} ± {std_val:.4f}")
    
    return "\n".join(lines)
