"""
Metric 4: Embedding Analysis - Social vs Environmental Learning

This module implements comprehensive embedding analysis to determine whether
agents learn social strategies (using opponent history) or task-only strategies
(Nash equilibrium, ignoring social information).

Research Question: Do agents integrate opponent behavior into decision-making,
or do they only learn task structure?

Methods:
1. Weight Magnitude Analysis - Measure learned structure in each embedding
2. Activation Variance Analysis - Measure discrimination ability of embeddings
3. Ablation Analysis - Measure policy dependence on each embedding
4. Embedding Visualization - Visualize social embedding spaces
5. Correlation with Behavior - Link embeddings to cooperation patterns
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple, Optional
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cognitive_therapy_ai.network import GameLSTM


#############################################################################
# Method 1: Weight Magnitude Analysis
#############################################################################

def analyze_embedding_weights(model: GameLSTM) -> Dict[str, float]:
    """
    Compute L2 norm of weights for each embedding pathway.
    
    Interpretation:
    - Large magnitude = embedding has learned structure (is being used)
    - Small magnitude = embedding near initialization (is being ignored)
    
    Returns:
        Dictionary with weight magnitudes and social ratio
    """
    results = {}
    
    # ENVIRONMENTAL EMBEDDINGS
    payoff_weights = model.payoff_matrix_embed[0].weight  # (embed_dim, 4)
    round_weights = model.round_number_embed[0].weight    # (embed_dim, 1)
    
    results['env_payoff_magnitude'] = torch.norm(payoff_weights).item()
    results['env_round_magnitude'] = torch.norm(round_weights).item()
    results['env_total'] = results['env_payoff_magnitude'] + results['env_round_magnitude']
    
    # SOCIAL EMBEDDINGS
    opp_action_weights = model.opponent_action_embed[0].weight
    agent_action_weights = model.agent_action_embed[0].weight
    agent_reward_weights = model.agent_reward_embed[0].weight
    opp_reward_weights = model.opponent_reward_embed[0].weight
    
    results['soc_opp_action'] = torch.norm(opp_action_weights).item()
    results['soc_agent_action'] = torch.norm(agent_action_weights).item()
    results['soc_agent_reward'] = torch.norm(agent_reward_weights).item()
    results['soc_opp_reward'] = torch.norm(opp_reward_weights).item()
    results['soc_total'] = sum([results[k] for k in results if k.startswith('soc_')])
    
    # RATIO: Social vs Total
    total_magnitude = results['env_total'] + results['soc_total']
    results['social_ratio'] = results['soc_total'] / total_magnitude if total_magnitude > 0 else 0.0
    
    return results


#############################################################################
# Method 2: Activation Variance Analysis
#############################################################################

def analyze_activation_variance(
    model: GameLSTM,
    states_tensor: torch.Tensor,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Measure variance of embedding activations across states.
    
    High variance = embedding discriminates different inputs
    Low variance = embedding produces constant output (unused)
    
    Args:
        model: Trained GameLSTM network
        states_tensor: Tensor of shape (N, 9) with test states
        device: Device to run on
        
    Returns:
        Dictionary with variance for each embedding
    """
    model.eval()
    model = model.to(device)
    states = states_tensor.to(device)
    
    activations = {
        'env_payoff': [],
        'env_round': [],
        'soc_opp_action': [],
        'soc_agent_action': [],
        'soc_agent_reward': [],
        'soc_opp_reward': []
    }
    
    with torch.no_grad():
        # Extract components from states
        payoff = states[:, 0:4]
        round_num = states[:, 4:5]
        opp_act = states[:, 5:6]
        agent_act = states[:, 6:7]
        agent_rew = states[:, 7:8]
        opp_rew = states[:, 8:9]
        
        # Pass through embeddings
        activations['env_payoff'] = model.payoff_matrix_embed(payoff).cpu()
        activations['env_round'] = model.round_number_embed(round_num).cpu()
        activations['soc_opp_action'] = model.opponent_action_embed(opp_act).cpu()
        activations['soc_agent_action'] = model.agent_action_embed(agent_act).cpu()
        activations['soc_agent_reward'] = model.agent_reward_embed(agent_rew).cpu()
        activations['soc_opp_reward'] = model.opponent_reward_embed(opp_rew).cpu()
    
    # Compute variance for each embedding (average across embedding dimensions)
    variances = {}
    for name, acts in activations.items():
        # acts is (N, embed_dim), compute variance per dimension then average
        variances[name] = acts.var(dim=0).mean().item()
    
    # Compute totals and ratio
    env_var = variances['env_payoff'] + variances['env_round']
    soc_var = sum(v for k, v in variances.items() if k.startswith('soc_'))
    total_var = env_var + soc_var
    
    variances['env_total'] = env_var
    variances['soc_total'] = soc_var
    variances['social_ratio'] = soc_var / total_var if total_var > 0 else 0.0
    
    return variances


#############################################################################
# Method 3: Ablation Analysis
#############################################################################

def ablation_analysis(
    model: GameLSTM,
    states_tensor: torch.Tensor,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Measure policy change when zeroing out each embedding.
    
    Large KL divergence = embedding is critical for policy
    Small KL divergence = embedding is not used
    
    Args:
        model: Trained GameLSTM network
        states_tensor: Tensor of shape (N, 9) with test states
        device: Device to run on
        
    Returns:
        Dictionary with KL divergence for each embedding ablation
    """
    model.eval()
    model = model.to(device)
    states = states_tensor.to(device).unsqueeze(1)  # Add sequence dimension
    
    # Get baseline policy
    with torch.no_grad():
        hidden = None
        policy_logits, _, _, hidden = model(states, hidden)
        baseline_policy = torch.softmax(policy_logits.squeeze(1), dim=-1)
    
    results = {}
    
    # Test each embedding ablation
    embeddings_to_test = [
        ('payoff_matrix_embed', 'env_payoff'),
        ('round_number_embed', 'env_round'),
        ('opponent_action_embed', 'soc_opp_action'),
        ('agent_action_embed', 'soc_agent_action'),
        ('agent_reward_embed', 'soc_agent_reward'),
        ('opponent_reward_embed', 'soc_opp_reward')
    ]
    
    for embed_attr, result_key in embeddings_to_test:
        # Save original embedding
        original_embed = getattr(model, embed_attr)
        
        # Get input and output dimensions from original embedding
        input_dim = original_embed[0].in_features
        output_dim = original_embed[0].out_features
        
        # Create zero embedding that outputs zeros with correct dimensions
        class ZeroLinear(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.out_features = out_features
            
            def forward(self, x):
                batch_size = x.shape[0]
                return torch.zeros(batch_size, self.out_features, device=x.device)
        
        zero_embed = nn.Sequential(
            ZeroLinear(input_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        ).to(device)
        
        # Replace embedding temporarily
        setattr(model, embed_attr, zero_embed)
        
        # Get ablated policy
        with torch.no_grad():
            hidden = None
            ablated_logits, _, _, hidden = model(states, hidden)
            ablated_policy = torch.softmax(ablated_logits.squeeze(1), dim=-1)
        
        # Compute KL divergence (how much did policy change?)
        kl_div = torch.nn.functional.kl_div(
            ablated_policy.log(),
            baseline_policy,
            reduction='batchmean'
        ).item()
        
        results[result_key] = kl_div
        
        # Restore original embedding
        setattr(model, embed_attr, original_embed)
    
    return results


#############################################################################
# Method 4: Embedding Visualization (t-SNE/PCA)
#############################################################################

def extract_embedding_activations(
    model: GameLSTM,
    states_tensor: torch.Tensor,
    embedding_name: str,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Extract activations from a specific embedding layer.
    
    Args:
        model: Trained network
        states_tensor: Input states (N, 9)
        embedding_name: Which embedding to extract
        device: Device to run on
        
    Returns:
        Numpy array of activations (N, embed_dim)
    """
    model.eval()
    model = model.to(device)
    states = states_tensor.to(device)
    
    # Map embedding names to model attributes
    embed_map = {
        'payoff': (model.payoff_matrix_embed, states[:, 0:4]),
        'round': (model.round_number_embed, states[:, 4:5]),
        'opp_action': (model.opponent_action_embed, states[:, 5:6]),
        'agent_action': (model.agent_action_embed, states[:, 6:7]),
        'agent_reward': (model.agent_reward_embed, states[:, 7:8]),
        'opp_reward': (model.opponent_reward_embed, states[:, 8:9])
    }
    
    if embedding_name not in embed_map:
        raise ValueError(f"Unknown embedding: {embedding_name}")
    
    embed_layer, embed_input = embed_map[embedding_name]
    
    with torch.no_grad():
        activations = embed_layer(embed_input)
    
    return activations.cpu().numpy()


def visualize_embedding_tsne(
    activations: np.ndarray,
    labels: np.ndarray,
    label_name: str,
    title: str,
    output_path: Path,
    cmap: str = 'RdYlGn'
):
    """
    Create t-SNE visualization of embedding space.
    
    Args:
        activations: Embedding activations (N, embed_dim)
        labels: Labels for coloring points (N,)
        label_name: Name of label (for colorbar)
        title: Plot title
        output_path: Where to save
        cmap: Colormap to use
    """
    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(activations) // 2))
    embeds_2d = tsne.fit_transform(activations)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embeds_2d[:, 0],
        embeds_2d[:, 1],
        c=labels,
        cmap=cmap,
        alpha=0.6,
        s=20,
        edgecolors='black',
        linewidth=0.5
    )
    plt.colorbar(scatter, label=label_name)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=11)
    plt.ylabel('t-SNE Dimension 2', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_pca_variance(
    activations_dict: Dict[str, np.ndarray],
    output_path: Path
):
    """
    Show PCA explained variance for environmental vs social embeddings.
    
    Args:
        activations_dict: Dict mapping embedding names to activations
        output_path: Where to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Environmental embeddings
    env_acts = np.concatenate([
        activations_dict['env_payoff'],
        activations_dict['env_round']
    ], axis=1)
    
    pca_env = PCA()
    pca_env.fit(env_acts)
    
    axes[0].bar(range(len(pca_env.explained_variance_ratio_)), 
                pca_env.explained_variance_ratio_)
    axes[0].set_title('Environmental Embeddings PCA', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].grid(True, alpha=0.3)
    
    # Social embeddings
    soc_acts = np.concatenate([
        activations_dict['soc_opp_action'],
        activations_dict['soc_agent_action'],
        activations_dict['soc_agent_reward'],
        activations_dict['soc_opp_reward']
    ], axis=1)
    
    pca_soc = PCA()
    pca_soc.fit(soc_acts)
    
    axes[1].bar(range(len(pca_soc.explained_variance_ratio_)),
                pca_soc.explained_variance_ratio_)
    axes[1].set_title('Social Embeddings PCA', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Principal Component')
    axes[1].set_ylabel('Explained Variance Ratio')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


#############################################################################
# Comprehensive Analysis Function
#############################################################################

def analyze_model_embeddings(
    checkpoint_path: Path,
    test_states: torch.Tensor,
    model_metadata: Dict,
    device: str = 'cpu'
) -> Dict:
    """
    Run complete embedding analysis on a single model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        test_states: Test states tensor (N, 9)
        model_metadata: Dict with train_game, train_opponent, seed, model_id
        device: Device to use
        
    Returns:
        Dictionary with all analysis results
    """
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model state dict (checkpoint may be nested)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model = GameLSTM(input_size=9, hidden_size=128, num_layers=2, dropout=0.1)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    results = model_metadata.copy()
    
    # Method 1: Weight magnitudes
    weight_results = analyze_embedding_weights(model)
    results.update(weight_results)
    
    # Method 2: Activation variance
    variance_results = analyze_activation_variance(model, test_states, device)
    # Prefix with 'var_' to avoid name collision
    results.update({f'var_{k}': v for k, v in variance_results.items()})
    
    # Method 3: Ablation analysis
    ablation_results = ablation_analysis(model, test_states[:1000], device)  # Use subset
    # Prefix with 'abl_'
    results.update({f'abl_{k}': v for k, v in ablation_results.items()})
    
    return results


#############################################################################
# Method 5: Full Network Representation Analysis
#############################################################################

def analyze_full_network_weights(model: GameLSTM) -> Dict[str, float]:
    """
    Analyze weight magnitudes across the entire network, not just input embeddings.
    
    Returns:
        Dictionary with L2 norms for each network component
    """
    results = {}
    
    # Input embeddings (already computed separately)
    embed_total = 0
    for name in ['payoff_matrix_embed', 'round_number_embed', 'opponent_action_embed',
                 'agent_action_embed', 'agent_reward_embed', 'opponent_reward_embed']:
        embed = getattr(model, name)
        if isinstance(embed, nn.Sequential):
            linear_layer = embed[0]
            embed_total += torch.norm(linear_layer.weight).item() ** 2
    results['input_embeddings_total'] = np.sqrt(embed_total)
    
    # LSTM weights
    lstm_total = 0
    for name, param in model.lstm.named_parameters():
        lstm_total += torch.norm(param).item() ** 2
    results['lstm_total'] = np.sqrt(lstm_total)
    
    # Separate LSTM input-hidden from hidden-hidden
    ih_total = 0  # Input to hidden
    hh_total = 0  # Hidden to hidden
    for name, param in model.lstm.named_parameters():
        if 'weight_ih' in name:
            ih_total += torch.norm(param).item() ** 2
        elif 'weight_hh' in name:
            hh_total += torch.norm(param).item() ** 2
    results['lstm_input_hidden'] = np.sqrt(ih_total)
    results['lstm_hidden_hidden'] = np.sqrt(hh_total)
    
    # Output heads
    results['policy_head'] = torch.norm(model.policy_head[0].weight).item()
    results['opponent_pred_head'] = torch.norm(model.opponent_policy_head[0].weight).item()
    results['value_head'] = torch.norm(model.value_head[0].weight).item()
    
    # Total network magnitude
    total = 0
    for param in model.parameters():
        total += torch.norm(param).item() ** 2
    results['total_network'] = np.sqrt(total)
    
    return results


def extract_hidden_state_representations(
    model: GameLSTM,
    states_tensor: torch.Tensor,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Extract LSTM hidden states across test states.
    These capture the full learned representation, not just input processing.
    
    Args:
        model: Trained network
        states_tensor: Test states (N, 9)
        device: Device
        
    Returns:
        Hidden states (N, hidden_size)
    """
    model.eval()
    model = model.to(device)
    states = states_tensor.to(device).unsqueeze(1)  # Add sequence dimension
    
    with torch.no_grad():
        hidden = None
        _, _, _, (h_n, c_n) = model(states, hidden)
        # h_n shape: (num_layers, batch, hidden_size)
        # Take last layer's hidden state
        hidden_states = h_n[-1, :, :].cpu().numpy()
    
    return hidden_states


def compute_representational_similarity(
    hidden_states_1: np.ndarray,
    hidden_states_2: np.ndarray,
    method: str = 'cka'
) -> float:
    """
    Compute similarity between two sets of representations.
    
    Methods:
    - 'cka': Centered Kernel Alignment (scale-invariant, rotation-invariant)
    - 'correlation': Mean correlation between dimensions
    - 'cosine': Mean cosine similarity across states
    
    Args:
        hidden_states_1: (N, D) representations from model 1
        hidden_states_2: (N, D) representations from model 2
        method: Similarity metric
        
    Returns:
        Similarity score in [0, 1]
    """
    if method == 'cka':
        # Centered Kernel Alignment
        def centering(K):
            n = K.shape[0]
            unit = np.ones([n, n])
            I = np.eye(n)
            H = I - unit / n
            return np.dot(np.dot(H, K), H)
        
        def linear_kernel(X, Y):
            return X @ Y.T
        
        K1 = linear_kernel(hidden_states_1, hidden_states_1)
        K2 = linear_kernel(hidden_states_2, hidden_states_2)
        K12 = linear_kernel(hidden_states_1, hidden_states_2)
        
        K1_centered = centering(K1)
        K2_centered = centering(K2)
        K12_centered = centering(K12)
        
        hsic = np.sum(K1_centered * K2_centered)
        var1 = np.sqrt(np.sum(K1_centered * K1_centered))
        var2 = np.sqrt(np.sum(K2_centered * K2_centered))
        
        cka = hsic / (var1 * var2 + 1e-10)
        return max(0, min(1, cka))  # Clip to [0, 1]
    
    elif method == 'correlation':
        # Mean absolute correlation between dimensions
        corr_matrix = np.corrcoef(hidden_states_1.T, hidden_states_2.T)
        n_dims = hidden_states_1.shape[1]
        cross_corr = corr_matrix[:n_dims, n_dims:]
        return np.abs(cross_corr).mean()
    
    elif method == 'cosine':
        # Mean cosine similarity across states
        from sklearn.metrics.pairwise import cosine_similarity
        cos_sim = cosine_similarity(hidden_states_1, hidden_states_2)
        return np.diag(cos_sim).mean()
    
    else:
        raise ValueError(f"Unknown method: {method}")

