#!/usr/bin/env python3
"""
Complexity Metrics Computation
==============================

Computes metrics across four complexity domains:
1. Task Complexity: Game-theoretic properties
2. Opponent Complexity: Predictability and adaptation requirements
3. Behavioral Complexity: Reciprocity and policy characteristics
4. Representational Complexity: Network structure and dimensionality

Author: Research Team
Date: May 12, 2026
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, List
from scipy.stats import entropy
from sklearn.decomposition import PCA


def compute_task_complexity(game_name: str, payoff_matrix: np.ndarray = None) -> Dict[str, float]:
    """
    Compute task complexity metrics for a game.
    
    Args:
        game_name: Name of the game
        payoff_matrix: Optional 2x2 payoff matrix [[R, S], [T, P]]
    
    Returns:
        Dictionary with complexity metrics
    """
    # Standard payoff matrices for each game
    game_payoffs = {
        'prisoners-dilemma': {'R': 3, 'S': 0, 'T': 5, 'P': 1},
        'hawk-dove': {'R': 2, 'S': 1, 'T': 3, 'P': 0},
        'stag-hunt': {'R': 4, 'S': 0, 'T': 3, 'P': 2}
    }
    
    if payoff_matrix is None and game_name in game_payoffs:
        params = game_payoffs[game_name]
        payoff_matrix = np.array([
            [params['R'], params['S']],
            [params['T'], params['P']]
        ])
    
    if payoff_matrix is None:
        return {'error': f'Unknown game: {game_name}'}
    
    R, S = payoff_matrix[0, 0], payoff_matrix[0, 1]
    T, P = payoff_matrix[1, 0], payoff_matrix[1, 1]
    
    # 1. Nash equilibrium count (simplified - pure strategy only)
    nash_count = 0
    # (C, C) is NE if R >= T and R >= S (no incentive to deviate)
    if R >= T and R >= S:
        nash_count += 1
    # (D, D) is NE if P >= S and P >= T
    if P >= S and P >= T:
        nash_count += 1
    # For mixed strategy, check if no pure NE exists
    if nash_count == 0:
        nash_count = 0.5  # Indicate mixed strategy NE
    
    # 2. Payoff variance (measure of game complexity)
    payoff_variance = np.var([R, S, T, P])
    
    # 3. Social dilemma strength
    # Classic: T - R (temptation to defect over mutual cooperation)
    # Generalized: max(T, R) - min(S, P)
    social_dilemma_strength = max(T, R) - min(S, P)
    temptation_to_defect = T - R
    
    # 4. Coordination bonus
    # R - max(S, T) for coordination games
    # Positive = coordination beneficial, negative = dilemma
    coordination_bonus = R - max(S, T)
    
    # 5. Risk dominance (for coordination games)
    # (R - S) vs (T - P): which equilibrium is less risky
    risk_coop = R - S  # Gain from mutual coop vs being exploited
    risk_defect = P - T  # "Gain" from mutual defect vs exploiting
    risk_dominance = risk_coop - risk_defect
    
    # 6. Payoff range (normalized complexity measure)
    payoff_range = max(R, S, T, P) - min(R, S, T, P)
    
    return {
        'game_name': game_name,
        'nash_equilibrium_count': nash_count,
        'payoff_variance': payoff_variance,
        'social_dilemma_strength': social_dilemma_strength,
        'temptation_to_defect': temptation_to_defect,
        'coordination_bonus': coordination_bonus,
        'risk_dominance': risk_dominance,
        'payoff_range': payoff_range,
        'R': R, 'S': S, 'T': T, 'P': P
    }


def compute_opponent_complexity(opponent_defect_prob: float, 
                                action_history: np.ndarray = None) -> Dict[str, float]:
    """
    Compute opponent complexity metrics.
    
    Args:
        opponent_defect_prob: Opponent's defection probability (0.1-0.9)
        action_history: Optional array of opponent actions (0=coop, 1=defect)
    
    Returns:
        Dictionary with complexity metrics
    """
    # 1. Stationarity (1.0 for probabilistic opponents)
    stationarity = 1.0
    
    # 2. Predictability (1 - entropy of action distribution)
    p_defect = opponent_defect_prob
    p_coop = 1 - p_defect
    # Avoid log(0) issues
    if p_defect == 0 or p_defect == 1:
        action_entropy = 0
    else:
        action_entropy = -p_coop * np.log2(p_coop) - p_defect * np.log2(p_defect)
    predictability = 1 - action_entropy  # Max entropy = 1 (p=0.5), so predictability in [0, 1]
    
    # 3. Cooperation rate
    cooperation_rate = 1 - opponent_defect_prob
    
    # 4. Adaptation requirement (how much agent must vary strategy)
    # For probabilistic opponents: higher entropy = more adaptation needed
    # Proxy: distance from extremes
    adaptation_requirement = 1 - abs(2 * opponent_defect_prob - 1)  # Max at 0.5, min at 0/1
    
    # 5. Empirical metrics from action history (if available)
    if action_history is not None:
        empirical_defect_rate = action_history.mean()
        empirical_entropy = entropy([1 - empirical_defect_rate, empirical_defect_rate], base=2)
        # Temporal consistency: autocorrelation
        if len(action_history) > 1:
            temporal_consistency = np.corrcoef(action_history[:-1], action_history[1:])[0, 1]
        else:
            temporal_consistency = 1.0
    else:
        empirical_defect_rate = opponent_defect_prob
        empirical_entropy = action_entropy
        temporal_consistency = stationarity
    
    return {
        'opponent_defect_prob': opponent_defect_prob,
        'stationarity': stationarity,
        'predictability': predictability,
        'cooperation_rate': cooperation_rate,
        'adaptation_requirement': adaptation_requirement,
        'action_entropy': action_entropy,
        'empirical_defect_rate': empirical_defect_rate,
        'empirical_entropy': empirical_entropy,
        'temporal_consistency': temporal_consistency
    }


def compute_behavioral_complexity(actions: np.ndarray, 
                                  opponent_actions: np.ndarray,
                                  policy_probs: np.ndarray = None) -> Dict[str, float]:
    """
    Compute behavioral complexity metrics from agent actions.
    
    Args:
        actions: Agent actions (0=coop, 1=defect)
        opponent_actions: Opponent actions (0=coop, 1=defect)
        policy_probs: Optional policy probabilities P(cooperate) per step
    
    Returns:
        Dictionary with complexity metrics
    """
    # Convert to binary
    agent_coop = (actions == 0).astype(int)
    opp_coop = (opponent_actions == 0).astype(int)
    
    # 1. Reciprocity strength: P(coop|opp_coop_t-1) - P(coop|opp_defect_t-1)
    if len(agent_coop) > 1:
        opp_prev = opp_coop[:-1]
        agent_curr = agent_coop[1:]
        
        opp_coop_mask = opp_prev == 1
        opp_defect_mask = opp_prev == 0
        
        if opp_coop_mask.sum() > 0:
            p_coop_given_opp_coop = agent_curr[opp_coop_mask].mean()
        else:
            p_coop_given_opp_coop = np.nan
        
        if opp_defect_mask.sum() > 0:
            p_coop_given_opp_defect = agent_curr[opp_defect_mask].mean()
        else:
            p_coop_given_opp_defect = np.nan
        
        if not np.isnan(p_coop_given_opp_coop) and not np.isnan(p_coop_given_opp_defect):
            reciprocity_strength = p_coop_given_opp_coop - p_coop_given_opp_defect
        else:
            reciprocity_strength = 0.0
    else:
        reciprocity_strength = 0.0
        p_coop_given_opp_coop = np.nan
        p_coop_given_opp_defect = np.nan
    
    # 2. Policy entropy
    if policy_probs is not None:
        # Average entropy over all steps
        policy_entropy_values = []
        for p_coop in policy_probs:
            p_defect = 1 - p_coop
            if p_coop == 0 or p_coop == 1:
                ent = 0
            else:
                ent = -p_coop * np.log2(p_coop) - p_defect * np.log2(p_defect)
            policy_entropy_values.append(ent)
        policy_entropy = np.mean(policy_entropy_values)
    else:
        # Use empirical action distribution
        p_coop = agent_coop.mean()
        p_defect = 1 - p_coop
        if p_coop == 0 or p_coop == 1:
            policy_entropy = 0
        else:
            policy_entropy = -p_coop * np.log2(p_coop) - p_defect * np.log2(p_defect)
    
    # 3. Behavioral variability (std of cooperation rate)
    behavioral_variability = agent_coop.std()
    
    # 4. Temporal consistency (autocorrelation in actions)
    if len(agent_coop) > 1:
        temporal_consistency = np.corrcoef(agent_coop[:-1], agent_coop[1:])[0, 1]
        if np.isnan(temporal_consistency):
            temporal_consistency = 1.0  # Constant strategy
    else:
        temporal_consistency = 1.0
    
    # 5. Mean cooperation rate
    mean_cooperation = agent_coop.mean()
    
    return {
        'reciprocity_strength': reciprocity_strength,
        'p_coop_given_opp_coop': p_coop_given_opp_coop,
        'p_coop_given_opp_defect': p_coop_given_opp_defect,
        'policy_entropy': policy_entropy,
        'behavioral_variability': behavioral_variability,
        'temporal_consistency': temporal_consistency,
        'mean_cooperation': mean_cooperation
    }


def compute_representational_complexity(
    model: torch.nn.Module,
    test_states: torch.Tensor,
    device: torch.device
) -> Dict[str, float]:
    """
    Compute representational complexity metrics from network.
    
    Args:
        model: Trained GameLSTM model
        test_states: Batch of test states (B, seq_len, input_size)
        device: Torch device
    
    Returns:
        Dictionary with complexity metrics
    """
    model.eval()
    model.to(device)
    test_states = test_states.to(device)
    
    results = {}
    
    # 1. Embedding weight magnitudes (reuse from metric 4)
    embedding_weights = {}
    if hasattr(model, 'payoff_matrix_embed'):
        embedding_weights['env_payoff'] = model.payoff_matrix_embed[0].weight.norm().item()
    if hasattr(model, 'round_number_embed'):
        embedding_weights['env_round'] = model.round_number_embed[0].weight.norm().item()
    if hasattr(model, 'opponent_action_embed'):
        embedding_weights['soc_opp_action'] = model.opponent_action_embed[0].weight.norm().item()
    if hasattr(model, 'agent_action_embed'):
        embedding_weights['soc_agent_action'] = model.agent_action_embed[0].weight.norm().item()
    if hasattr(model, 'agent_reward_embed'):
        embedding_weights['soc_agent_reward'] = model.agent_reward_embed[0].weight.norm().item()
    if hasattr(model, 'opponent_reward_embed'):
        embedding_weights['soc_opp_reward'] = model.opponent_reward_embed[0].weight.norm().item()
    
    # Environmental and social totals
    env_total = embedding_weights.get('env_payoff', 0) + embedding_weights.get('env_round', 0)
    soc_total = (embedding_weights.get('soc_opp_action', 0) + 
                 embedding_weights.get('soc_agent_action', 0) +
                 embedding_weights.get('soc_agent_reward', 0) +
                 embedding_weights.get('soc_opp_reward', 0))
    
    total = env_total + soc_total
    if total > 0:
        social_ratio = soc_total / total
    else:
        social_ratio = 0.0
    
    results['social_ratio'] = social_ratio
    results['env_weight_total'] = env_total
    results['soc_weight_total'] = soc_total
    
    # 2. Total weight L2 norm
    total_weight_norm = 0
    for param in model.parameters():
        total_weight_norm += param.norm().item() ** 2
    total_weight_norm = np.sqrt(total_weight_norm)
    results['weight_l2_norm'] = total_weight_norm
    
    # 3. Effective dimensionality from hidden states
    with torch.no_grad():
        # Forward pass to get hidden states
        batch_size = test_states.size(0)
        seq_len = test_states.size(1)
        
        # Initialize hidden state
        h0 = torch.zeros(model.lstm.num_layers, batch_size, model.lstm.hidden_size).to(device)
        c0 = torch.zeros(model.lstm.num_layers, batch_size, model.lstm.hidden_size).to(device)
        
        # Get embeddings
        if hasattr(model, 'get_embeddings'):
            embeddings = model.get_embeddings(test_states)
        else:
            # Manual embedding extraction
            payoff = test_states[:, :, :4]
            round_num = test_states[:, :, 4:5]
            opp_action = test_states[:, :, 5:6]
            agent_action = test_states[:, :, 6:7]
            agent_reward = test_states[:, :, 7:8]
            opp_reward = test_states[:, :, 8:9]
            
            embed_list = []
            if hasattr(model, 'payoff_matrix_embed'):
                embed_list.append(model.payoff_matrix_embed(payoff))
            if hasattr(model, 'round_number_embed'):
                embed_list.append(model.round_number_embed(round_num))
            if hasattr(model, 'opponent_action_embed'):
                embed_list.append(model.opponent_action_embed(opp_action))
            if hasattr(model, 'agent_action_embed'):
                embed_list.append(model.agent_action_embed(agent_action))
            if hasattr(model, 'agent_reward_embed'):
                embed_list.append(model.agent_reward_embed(agent_reward))
            if hasattr(model, 'opponent_reward_embed'):
                embed_list.append(model.opponent_reward_embed(opp_reward))
            
            embeddings = torch.cat(embed_list, dim=-1)
        
        # LSTM forward
        lstm_out, (hn, cn) = model.lstm(embeddings, (h0, c0))
        
        # Extract final hidden state across all samples
        # Shape: (batch_size, hidden_size)
        final_hidden = hn[-1, :, :]  # Last layer
        
        # PCA to measure intrinsic dimensionality
        hidden_np = final_hidden.cpu().numpy()
        
        # Standardize
        hidden_mean = hidden_np.mean(axis=0)
        hidden_std = hidden_np.std(axis=0) + 1e-8
        hidden_standardized = (hidden_np - hidden_mean) / hidden_std
        
        # PCA
        pca = PCA()
        pca.fit(hidden_standardized)
        
        # Effective dimensionality: number of components for 95% variance
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        effective_dim = np.argmax(cumsum_var >= 0.95) + 1
        
        results['effective_dimensionality'] = effective_dim
        results['pca_variance_ratio_95'] = cumsum_var[effective_dim - 1] if effective_dim > 0 else 0
        
        # 4. Activation sparsity (L1/L2 ratio)
        l1_norm = final_hidden.abs().mean().item()
        l2_norm = final_hidden.norm(dim=1).mean().item()
        if l2_norm > 0:
            activation_sparsity = l1_norm / l2_norm
        else:
            activation_sparsity = 0
        results['activation_sparsity'] = activation_sparsity
        
        # 5. Embedding specialization (variance in embedding weight magnitudes)
        if len(embedding_weights) > 0:
            weight_values = list(embedding_weights.values())
            embedding_specialization = np.std(weight_values)
        else:
            embedding_specialization = 0
        results['embedding_specialization'] = embedding_specialization
    
    return results


def generate_test_states(n_samples: int = 1000, 
                         games: List[str] = None,
                         seed: int = 42) -> torch.Tensor:
    """
    Generate diverse test states for representation analysis.
    
    Args:
        n_samples: Number of test states to generate
        games: Optional list of games to sample from
        seed: Random seed
    
    Returns:
        Tensor of shape (n_samples, 1, 9) with test states
    """
    np.random.seed(seed)
    
    if games is None:
        games = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']
    
    # Payoff matrices
    game_payoffs = {
        'prisoners-dilemma': np.array([[3, 0], [5, 1]]),
        'hawk-dove': np.array([[2, 1], [3, 0]]),
        'stag-hunt': np.array([[4, 0], [3, 2]])
    }
    
    states = []
    for _ in range(n_samples):
        # Random game
        game = np.random.choice(games)
        payoff = game_payoffs[game].flatten()  # [R, S, T, P]
        
        # Random round (0-99, normalized to 0-1)
        round_num = np.random.rand()
        
        # Random actions (0=coop, 1=defect)
        opp_action = np.random.randint(0, 2)
        agent_action = np.random.randint(0, 2)
        
        # Compute rewards
        agent_reward = payoff[agent_action * 2 + opp_action]
        opp_reward = payoff[opp_action * 2 + agent_action]
        
        # Normalize rewards to [0, 1] (assuming max=5, min=0)
        agent_reward = agent_reward / 5.0
        opp_reward = opp_reward / 5.0
        
        # Concatenate: [payoff(4), round(1), opp_action(1), agent_action(1), agent_reward(1), opp_reward(1)]
        state = np.concatenate([
            payoff / 5.0,  # Normalize payoffs
            [round_num],
            [opp_action],
            [agent_action],
            [agent_reward],
            [opp_reward]
        ])
        
        states.append(state)
    
    # Convert to tensor with shape (n_samples, 1, 9)
    states_tensor = torch.FloatTensor(states).unsqueeze(1)
    return states_tensor
