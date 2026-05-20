#!/usr/bin/env python3
"""
Network-Level Representation Analysis - Experiment 918988/918989
================================================================

Implements 6 comprehensive methods to analyze learned representations:
1. Singular Value Decomposition (SVD) - rank, spectrum, dominant vectors
2. Gradient Flow Analysis - gradient norms, alignment, sensitivity
3. Activation Pattern Analysis - hidden states, similarity, UMAP/t-SNE
4. Linear Probing - frozen representation classifiers
5. Network Compression Metrics - effective rank, intrinsic dimensionality, pruning
6. Information Flow Analysis - mutual information, compression, retention

Usage:
    python analysis/network_representation_analysis_918988.py

Author: Research Team
Date: May 19, 2026
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from scipy.linalg import svd
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from cognitive_therapy_ai import GameLSTM, GameFactory

# Try to import UMAP (optional)
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Install with: pip install umap-learn")


def get_game_abbreviation(game_name: str) -> str:
    """Get consistent game abbreviation."""
    abbreviations = {
        'prisoners-dilemma': 'PD',
        'hawk-dove': 'HD',
        'stag-hunt': 'SH'
    }
    return abbreviations.get(game_name, game_name[:2].upper())


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> GameLSTM:
    """Load model checkpoint with correct architecture."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Architecture for 918988/918989
    model = GameLSTM(
        input_size=9,
        hidden_size=128,
        num_layers=2,
        dropout=0.1
    )
    
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)


def generate_test_episodes(game_name: str, opponent_prob: float, num_episodes: int = 100, 
                          episode_length: int = 100, device: torch.device = 'cpu'):
    """Generate standardized test episodes for representation extraction."""
    game = GameFactory.create_game(game_name)
    payoff_matrix = game.get_payoff_matrix().flatten()  # 4 elements
    
    states_list = []
    opp_actions_list = []
    
    for ep in range(num_episodes):
        episode_states = []
        episode_opp_actions = []
        
        # Previous actions/rewards for this episode
        prev_opp_action = -1.0
        prev_agent_action = -1.0
        prev_agent_reward = 0.0
        prev_opp_reward = 0.0
        
        for t in range(episode_length):
            # State: [payoff_matrix(4), round_num(1), prev_opp_action(1), prev_agent_action(1), prev_agent_reward(1), prev_opp_reward(1)]
            round_norm = t / 100.0
            state = np.concatenate([
                payoff_matrix,
                [round_norm],
                [prev_opp_action],
                [prev_agent_action],
                [prev_agent_reward],
                [prev_opp_reward]
            ])
            episode_states.append(state)
            
            # Opponent action (probabilistic)
            opp_action = 1 if np.random.random() < opponent_prob else 0
            episode_opp_actions.append(opp_action)
            
            # Update previous actions (dummy agent action)
            prev_opp_action = float(opp_action)
            prev_agent_action = 0.0  # Dummy
            prev_agent_reward = 0.0  # Would need game logic to compute
            prev_opp_reward = 0.0
        
        states_list.append(np.array(episode_states))
        opp_actions_list.append(np.array(episode_opp_actions))
    
    # Stack into batches (num_episodes, episode_length, 10)
    states = torch.FloatTensor(np.array(states_list)).to(device)
    opp_actions = torch.LongTensor(np.array(opp_actions_list)).to(device)
    
    return states, opp_actions


# ============================================================================
# METHOD 1: SINGULAR VALUE DECOMPOSITION (SVD)
# ============================================================================

def analyze_svd(models_data: dict, output_dir: Path):
    """
    Analyze weight matrices using SVD.
    
    Metrics:
    - Singular value spectrum
    - Effective rank (number of singular values > threshold)
    - Dominant singular vectors
    - Compression ratio
    """
    print("\n" + "="*80)
    print("METHOD 1: SINGULAR VALUE DECOMPOSITION (SVD)")
    print("="*80)
    
    results = []
    
    for (game, opponent), model_info in models_data.items():
        model = model_info['model']
        
        # Extract weight matrices from LSTM
        lstm = model.lstm
        
        # LSTM weight matrices: weight_ih_l0, weight_hh_l0, weight_ih_l1, weight_hh_l1
        matrices = {
            'lstm_l0_ih': lstm.weight_ih_l0.detach().cpu().numpy(),
            'lstm_l0_hh': lstm.weight_hh_l0.detach().cpu().numpy(),
            'lstm_l1_ih': lstm.weight_ih_l1.detach().cpu().numpy(),
            'lstm_l1_hh': lstm.weight_hh_l1.detach().cpu().numpy(),
            'policy_head': model.policy_head[0].weight.detach().cpu().numpy(),  # First linear layer
            'value_head': model.value_head[0].weight.detach().cpu().numpy(),    # First linear layer
        }
        
        for mat_name, W in matrices.items():
            # Compute SVD
            U, s, Vt = svd(W, full_matrices=False)
            
            # Metrics
            total_variance = np.sum(s**2)
            cumsum = np.cumsum(s**2) / total_variance
            
            # Effective rank (90% variance)
            effective_rank_90 = np.searchsorted(cumsum, 0.90) + 1
            # Effective rank (95% variance)
            effective_rank_95 = np.searchsorted(cumsum, 0.95) + 1
            
            # Participation ratio (inverse of normalized sum of squares)
            participation_ratio = (np.sum(s)**2) / np.sum(s**2)
            
            # Condition number
            condition_number = s[0] / s[-1] if s[-1] > 1e-10 else np.inf
            
            # Spectral entropy
            p = (s**2) / np.sum(s**2)
            spectral_entropy = -np.sum(p * np.log(p + 1e-10))
            
            results.append({
                'game': game,
                'opponent': opponent,
                'matrix': mat_name,
                'effective_rank_90': effective_rank_90,
                'effective_rank_95': effective_rank_95,
                'participation_ratio': participation_ratio,
                'condition_number': condition_number,
                'spectral_entropy': spectral_entropy,
                'max_singular_value': s[0],
                'min_singular_value': s[-1],
                'singular_values': s.tolist()
            })
    
    df_svd = pd.DataFrame(results)
    
    if len(df_svd) == 0:
        print("Warning: No models loaded. Cannot perform SVD analysis.")
        return df_svd
    
    # Save results
    df_svd.drop(columns=['singular_values']).to_csv(
        output_dir / 'svd_analysis.csv', index=False
    )
    
    # Plot singular value spectra
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    matrices_to_plot = ['lstm_l0_ih', 'lstm_l0_hh', 'lstm_l1_ih', 'lstm_l1_hh', 'policy_head', 'value_head']
    
    for idx, mat_name in enumerate(matrices_to_plot):
        ax = axes[idx]
        
        for game in ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']:
            subset = df_svd[(df_svd['matrix'] == mat_name) & (df_svd['game'] == game)]
            
            if len(subset) > 0:
                # Average singular values across opponents
                all_sv = [sv for sv_list in subset['singular_values'] for sv in sv_list]
                avg_sv = np.mean([subset.iloc[i]['singular_values'] for i in range(len(subset))], axis=0)
                
                ax.semilogy(avg_sv, label=game, linewidth=2)
        
        ax.set_title(f'{mat_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Singular Value Index')
        ax.set_ylabel('Singular Value (log scale)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'svd_spectrum.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot effective rank comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # By game
    summary = df_svd.groupby(['game', 'matrix'])[['effective_rank_90', 'effective_rank_95']].mean().reset_index()
    
    games = summary['game'].unique()
    x = np.arange(len(matrices_to_plot))
    width = 0.25
    
    for i, game in enumerate(games):
        game_data = summary[summary['game'] == game]
        axes[0].bar(x + i*width, game_data['effective_rank_90'], width, label=game, alpha=0.8)
    
    axes[0].set_xlabel('Weight Matrix')
    axes[0].set_ylabel('Effective Rank (90% variance)')
    axes[0].set_title('Effective Rank by Game')
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(matrices_to_plot, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # By opponent
    summary_opp = df_svd.groupby(['opponent', 'matrix'])[['effective_rank_90']].mean().reset_index()
    
    for i, opp in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
        opp_data = summary_opp[summary_opp['opponent'] == opp]
        axes[1].plot(matrices_to_plot, opp_data['effective_rank_90'], marker='o', label=f'p={opp}', linewidth=2)
    
    axes[1].set_xlabel('Weight Matrix')
    axes[1].set_ylabel('Effective Rank (90% variance)')
    axes[1].set_title('Effective Rank by Opponent Type')
    axes[1].set_xticklabels(matrices_to_plot, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'svd_effective_rank.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"SVD analysis complete. Analyzed {len(df_svd)} weight matrices.")
    print(f"Mean effective rank (90%): {df_svd['effective_rank_90'].mean():.2f}")
    print(f"Mean participation ratio: {df_svd['participation_ratio'].mean():.2f}")
    
    return df_svd


# ============================================================================
# METHOD 2: GRADIENT FLOW ANALYSIS
# ============================================================================

def analyze_gradient_flow(models_data: dict, device: torch.device, output_dir: Path):
    """
    Analyze gradient flow through networks.
    
    Metrics:
    - Gradient norms per layer
    - Gradient alignment across conditions
    - Input sensitivity (gradient w.r.t. inputs)
    """
    print("\n" + "="*80)
    print("METHOD 2: GRADIENT FLOW ANALYSIS")
    print("="*80)
    
    if len(models_data) == 0:
        print("Warning: No models loaded. Cannot perform gradient flow analysis.")
        return pd.DataFrame(), np.array([])
    
    results = []
    gradient_maps = {}  # Store gradients for alignment analysis
    
    for (game, opponent), model_info in models_data.items():
        model = model_info['model']
        model.train()  # Enable gradients
        
        # Generate test batch
        states, opp_actions = generate_test_episodes(game, opponent, num_episodes=10, 
                                                     episode_length=50, device=device)
        
        # Flatten batch: (10, 50, 9) -> (500, 9)
        batch_size = states.shape[0] * states.shape[1]
        states_flat = states.reshape(batch_size, 9)
        states_flat.requires_grad = True
        
        # Forward pass (single timestep)
        policy_logits, _, _, _ = model(states_flat, None)
        
        # Compute loss (policy entropy as proxy)
        probs = torch.softmax(policy_logits, dim=-1)
        loss = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Extract gradient norms per layer
        grad_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms[name] = param.grad.norm().item()
        
        # Input sensitivity (gradient w.r.t. input)
        if states_flat.grad is not None:
            input_sensitivity = states_flat.grad.norm(dim=-1).mean().item()
        else:
            input_sensitivity = 0.0
        
        # Store results
        for layer_name, grad_norm in grad_norms.items():
            results.append({
                'game': game,
                'opponent': opponent,
                'layer': layer_name,
                'grad_norm': grad_norm,
                'input_sensitivity': input_sensitivity
            })
        
        # Store full gradient vector for alignment analysis
        grad_vector = []
        for param in model.parameters():
            if param.grad is not None:
                grad_vector.append(param.grad.flatten().cpu().numpy())
        
        gradient_maps[(game, opponent)] = np.concatenate(grad_vector)
        
        model.eval()
    
    df_grad = pd.DataFrame(results)
    df_grad.to_csv(output_dir / 'gradient_flow.csv', index=False)
    
    # Compute gradient alignment matrix (cosine similarity)
    conditions = list(gradient_maps.keys())
    n_cond = len(conditions)
    alignment_matrix = np.zeros((n_cond, n_cond))
    
    for i, cond_i in enumerate(conditions):
        for j, cond_j in enumerate(conditions):
            grad_i = gradient_maps[cond_i]
            grad_j = gradient_maps[cond_j]
            
            # Cosine similarity with epsilon to prevent numerical instability from vanishing gradients
            norm_i = np.linalg.norm(grad_i) + 1e-8
            norm_j = np.linalg.norm(grad_j) + 1e-8
            alignment = np.dot(grad_i, grad_j) / (norm_i * norm_j)
            alignment_matrix[i, j] = alignment
    
    # Plot gradient alignment heatmap only
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Gradient alignment heatmap
    im = ax.imshow(alignment_matrix, cmap='RdYlGn', vmin=-0.2, vmax=1, aspect='auto')
    ax.set_title('Gradient Alignment Matrix (Cosine Similarity)', fontsize=14, pad=15)
    ax.set_xlabel('Condition', fontsize=12)
    ax.set_ylabel('Condition', fontsize=12)
    
    # Add condition labels
    condition_labels = [f"{get_game_abbreviation(g)}-{o}" for g, o in conditions]
    ax.set_xticks(np.arange(n_cond))
    ax.set_yticks(np.arange(n_cond))
    ax.set_xticklabels(condition_labels, rotation=90, fontsize=8)
    ax.set_yticklabels(condition_labels, fontsize=8)
    
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gradient_flow_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Gradient flow analysis complete.")
    print(f"Mean gradient norm: {df_grad['grad_norm'].mean():.6f}")
    print(f"Mean input sensitivity: {df_grad['input_sensitivity'].mean():.6f}")
    
    # Save alignment matrix
    np.save(output_dir / 'gradient_alignment_matrix.npy', alignment_matrix)
    
    return df_grad, alignment_matrix


# ============================================================================
# METHOD 3: ACTIVATION PATTERN ANALYSIS
# ============================================================================

def analyze_activation_patterns(models_data: dict, device: torch.device, output_dir: Path):
    """
    Analyze hidden state activation patterns.
    
    Metrics:
    - Hidden state similarity matrices
    - UMAP/t-SNE visualization
    - Activation statistics (mean, std, sparsity)
    """
    print("\n" + "="*80)
    print("METHOD 3: ACTIVATION PATTERN ANALYSIS")
    print("="*80)
    
    if len(models_data) == 0:
        print("Warning: No models loaded. Cannot perform activation pattern analysis.")
        return pd.DataFrame(), np.array([])
    
    all_activations = {}
    activation_stats = []
    
    for (game, opponent), model_info in models_data.items():
        model = model_info['model']
        
        # Generate standardized test episodes
        states, opp_actions = generate_test_episodes(game, opponent, num_episodes=50, 
                                                     episode_length=100, device=device)
        
        # Extract hidden states
        with torch.no_grad():
            # Initialize LSTM hidden state
            h = torch.zeros(2, states.shape[0], 128).to(device)  # (num_layers, batch, hidden_size)
            c = torch.zeros(2, states.shape[0], 128).to(device)
            
            hidden_states_list = []
            
            for t in range(states.shape[1]):  # Iterate over timesteps
                state_t = states[:, t, :]  # (batch, 9)
                
                # Forward through network, returns (policy, opp_policy, value, new_hidden)
                _, _, _, (h, c) = model(state_t, (h, c))
                
                # Store hidden state from layer 1 (final layer)
                hidden_states_list.append(h[1, :, :].cpu().numpy())  # (batch, 128)
            
            # Average across timesteps and batch
            hidden_states = np.mean(hidden_states_list, axis=(0, 1))  # (128,)
        
        all_activations[(game, opponent)] = hidden_states
        
        # Compute statistics
        activation_stats.append({
            'game': game,
            'opponent': opponent,
            'mean_activation': hidden_states.mean(),
            'std_activation': hidden_states.std(),
            'sparsity': (np.abs(hidden_states) < 0.01).mean(),  # Fraction near zero
            'max_activation': np.abs(hidden_states).max()
        })
    
    df_activation = pd.DataFrame(activation_stats)
    df_activation.to_csv(output_dir / 'activation_statistics.csv', index=False)
    
    # Compute similarity matrix (cosine similarity)
    conditions = list(all_activations.keys())
    n_cond = len(conditions)
    similarity_matrix = np.zeros((n_cond, n_cond))
    
    for i, cond_i in enumerate(conditions):
        for j, cond_j in enumerate(conditions):
            act_i = all_activations[cond_i]
            act_j = all_activations[cond_j]
            
            similarity = np.dot(act_i, act_j) / (np.linalg.norm(act_i) * np.linalg.norm(act_j) + 1e-10)
            similarity_matrix[i, j] = similarity
    
    np.save(output_dir / 'activation_similarity_matrix.npy', similarity_matrix)
    
    # Prepare data for dimensionality reduction
    activation_array = np.array([all_activations[cond] for cond in conditions])  # (15, 128)
    labels_game = np.array([cond[0] for cond in conditions])
    labels_opponent = np.array([cond[1] for cond in conditions])
    
    # PCA
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(activation_array)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(conditions)-1))
    tsne_coords = tsne.fit_transform(activation_array)
    
    # UMAP (if available)
    if UMAP_AVAILABLE and len(conditions) > 5:
        umap_reducer = UMAP(n_components=2, random_state=42, n_neighbors=min(5, len(conditions)-1))
        umap_coords = umap_reducer.fit_transform(activation_array)
    else:
        umap_coords = None
    
    # Plotting
    n_plots = 4 if umap_coords is not None else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
    
    # Color maps
    game_colors = {'prisoners-dilemma': 'red', 'hawk-dove': 'blue', 'stag-hunt': 'green'}
    opponent_colors = {0.1: 'purple', 0.3: 'orange', 0.5: 'brown', 0.7: 'pink', 0.9: 'gray'}
    
    # Similarity heatmap
    im = axes[0].imshow(similarity_matrix, cmap='viridis', aspect='auto')
    axes[0].set_title('Activation Similarity Matrix')
    condition_labels = [f"{get_game_abbreviation(g)}-{o}" for g, o in conditions]
    axes[0].set_xticks(np.arange(n_cond))
    axes[0].set_yticks(np.arange(n_cond))
    axes[0].set_xticklabels(condition_labels, rotation=90, fontsize=6)
    axes[0].set_yticklabels(condition_labels, fontsize=6)
    plt.colorbar(im, ax=axes[0])
    
    # PCA (colored by game)
    for game in game_colors:
        mask = labels_game == game
        axes[1].scatter(pca_coords[mask, 0], pca_coords[mask, 1], 
                       c=game_colors[game], label=game, s=100, alpha=0.7)
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    axes[1].set_title('PCA - Colored by Game')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # t-SNE (colored by opponent)
    for opp in opponent_colors:
        mask = labels_opponent == opp
        axes[2].scatter(tsne_coords[mask, 0], tsne_coords[mask, 1], 
                       c=opponent_colors[opp], label=f'p={opp}', s=100, alpha=0.7)
    axes[2].set_xlabel('t-SNE 1')
    axes[2].set_ylabel('t-SNE 2')
    axes[2].set_title('t-SNE - Colored by Opponent')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # UMAP (if available)
    if umap_coords is not None:
        for game in game_colors:
            mask = labels_game == game
            axes[3].scatter(umap_coords[mask, 0], umap_coords[mask, 1], 
                           c=game_colors[game], label=game, s=100, alpha=0.7)
        axes[3].set_xlabel('UMAP 1')
        axes[3].set_ylabel('UMAP 2')
        axes[3].set_title('UMAP - Colored by Game')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'activation_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Activation pattern analysis complete.")
    print(f"Mean activation: {df_activation['mean_activation'].mean():.6f}")
    print(f"Mean sparsity: {df_activation['sparsity'].mean():.2%}")
    
    return df_activation, similarity_matrix


# ============================================================================
# COMBINED VISUALIZATION: ACTIVATION & GRADIENT MATRICES
# ============================================================================

def create_combined_matrix_figure(models_data: dict, act_similarity: np.ndarray, 
                                  grad_alignment: np.ndarray, output_dir: Path):
    """
    Create unified 2-panel figure:
    (A) Gradient Alignment Matrix
    (B) Activation Similarity Matrix
    
    Shows the relationship between learning dynamics and final representations.
    """
    print("\n" + "="*80)
    print("CREATING COMBINED MATRIX VISUALIZATION")
    print("="*80)
    
    # Extract conditions for labels
    conditions = list(models_data.keys())
    condition_labels = [f"{get_game_abbreviation(g)}-{o}" for g, o in conditions]
    n_cond = len(conditions)
    
    # Create figure with 2 panels
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # ========================================================================
    # Panel A: Gradient Alignment Matrix
    # ========================================================================
    ax = axes[0]
    im1 = ax.imshow(grad_alignment, cmap='RdYlGn', vmin=-0.2, vmax=1, aspect='auto')
    ax.set_title('(A) Gradient Alignment Matrix', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Condition', fontsize=13)
    ax.set_ylabel('Condition', fontsize=13)
    
    ax.set_xticks(np.arange(n_cond))
    ax.set_yticks(np.arange(n_cond))
    ax.set_xticklabels(condition_labels, rotation=90, fontsize=9)
    ax.set_yticklabels(condition_labels, fontsize=9)
    
    cbar1 = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    cbar1.set_label('Cosine Similarity', fontsize=11)
    
    # ========================================================================
    # Panel B: Activation Similarity Matrix  
    # ========================================================================
    ax = axes[1]
    im2 = ax.imshow(act_similarity, cmap='RdYlGn', vmin=-0.2, vmax=1, aspect='auto')
    ax.set_title('(B) Activation Similarity Matrix', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Condition', fontsize=13)
    ax.set_ylabel('Condition', fontsize=13)
    
    ax.set_xticks(np.arange(n_cond))
    ax.set_yticks(np.arange(n_cond))
    ax.set_xticklabels(condition_labels, rotation=90, fontsize=9)
    ax.set_yticklabels(condition_labels, fontsize=9)
    
    cbar2 = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cbar2.set_label('Cosine Similarity', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_activation_gradient_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: combined_activation_gradient_matrices.png")
    
    # ========================================================================
    # Also create standalone activation matrix plot
    # ========================================================================
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    im = ax.imshow(act_similarity, cmap='RdYlGn', aspect='auto', vmin=-0.2, vmax=1)
    ax.set_title('Activation Similarity Matrix', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Condition', fontsize=12)
    ax.set_ylabel('Condition', fontsize=12)
    
    ax.set_xticks(np.arange(n_cond))
    ax.set_yticks(np.arange(n_cond))
    ax.set_xticklabels(condition_labels, rotation=90, fontsize=8)
    ax.set_yticklabels(condition_labels, fontsize=8)
    
    cbar = plt.colorbar(im, ax=ax, label='Cosine Similarity')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'activation_similarity_matrix_standalone.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: activation_similarity_matrix_standalone.png")


# ============================================================================
# METHOD 4: LINEAR PROBING
# ============================================================================

def analyze_linear_probing(models_data: dict, device: torch.device, output_dir: Path):
    """
    Train linear classifiers on frozen representations from 3 layers:
    1. Input embeddings (6 separate embeddings before LSTM)
    2. LSTM hidden states (final layer)
    3. Policy head outputs (2D policy logits)
    
    Tasks:
    - Classify game type (3-way)
    - Classify opponent type (5-way)
    - Classify task-opponent condition (15-way)
    
    Enhanced analyses:
    1. Analyze coefficient sparsity (separating hyperplane dimensionality)
    2. Correlate with behavioral clusters
    3. Compare input → LSTM → output information preservation
    """
    print("\n" + "="*80)
    print("METHOD 4: LINEAR PROBING (3 LAYERS)")
    print("="*80)
    
    if len(models_data) == 0:
        print("Warning: No models loaded. Cannot perform linear probing.")
        return {}
    
    # Collect representations from all three layers
    representations_embedding = []  # Input embeddings
    representations_lstm = []       # LSTM hidden states
    representations_policy = []     # Policy head outputs
    labels_game = []
    labels_opponent = []
    labels_condition = []
    condition_keys = []  # For behavioral correlation
    
    game_to_idx = {'prisoners-dilemma': 0, 'hawk-dove': 1, 'stag-hunt': 2}
    opponent_to_idx = {0.1: 0, 0.3: 1, 0.5: 2, 0.7: 3, 0.9: 4}
    
    # Build condition to index mapping (15 conditions: 3 games × 5 opponents)
    condition_to_idx = {}
    idx = 0
    for game in sorted(game_to_idx.keys()):
        for opponent in sorted(opponent_to_idx.keys()):
            condition_to_idx[f"{game}_{opponent}"] = idx
            idx += 1
    
    for (game, opponent), model_info in models_data.items():
        model = model_info['model']
        
        # Generate test episodes
        states, opp_actions = generate_test_episodes(game, opponent, num_episodes=20, 
                                                     episode_length=100, device=device)
        
        # Extract representations from all three layers
        with torch.no_grad():
            h = torch.zeros(2, states.shape[0], 128).to(device)
            c = torch.zeros(2, states.shape[0], 128).to(device)
            
            for t in range(states.shape[1]):
                state_t = states[:, t, :]  # (batch, 9)
                
                # =================================================================
                # LAYER 1: Extract input embeddings (before LSTM)
                # =================================================================
                # Parse input components
                payoff_matrix = state_t[:, :4]           # (batch, 4)
                round_num = state_t[:, 4:5]              # (batch, 1)
                opponent_action = state_t[:, 5:6]        # (batch, 1)
                agent_action = state_t[:, 6:7]           # (batch, 1)
                agent_reward = state_t[:, 7:8]           # (batch, 1)
                opponent_reward = state_t[:, 8:9]        # (batch, 1)
                
                # Embed each component
                embed_payoff = model.payoff_matrix_embed(payoff_matrix)
                embed_round = model.round_number_embed(round_num)
                embed_opp_action = model.opponent_action_embed(opponent_action)
                embed_agent_action = model.agent_action_embed(agent_action)
                embed_agent_reward = model.agent_reward_embed(agent_reward)
                embed_opp_reward = model.opponent_reward_embed(opponent_reward)
                
                # Concatenate all embeddings
                embedded = torch.cat([
                    embed_payoff, embed_round, embed_opp_action,
                    embed_agent_action, embed_agent_reward, embed_opp_reward
                ], dim=-1)  # (batch, total_embed_dim)
                
                repr_embedding = embedded.cpu().numpy()
                
                # =================================================================
                # LAYER 2 & 3: Full forward pass for LSTM hidden states and policy logits
                # =================================================================
                policy_logits, _, _, (h, c) = model(state_t, (h, c))
                
                # Extract LSTM hidden state (final layer)
                repr_lstm = h[1, :, :].cpu().numpy()  # Final LSTM layer (batch, 128)
                
                # Extract policy head output
                repr_policy = policy_logits.cpu().numpy()  # (batch, 2)
                
                # =================================================================
                # Store all representations with labels
                # =================================================================
                for i in range(repr_lstm.shape[0]):
                    representations_embedding.append(repr_embedding[i])  # (embed_dim*6,)
                    representations_lstm.append(repr_lstm[i])             # (128,)
                    representations_policy.append(repr_policy[i])         # (2,)
                    labels_game.append(game_to_idx[game])
                    labels_opponent.append(opponent_to_idx[opponent])
                    condition_label = f"{game}_{opponent}"
                    labels_condition.append(condition_to_idx[condition_label])
                    condition_keys.append((game, opponent))
    
    
    # Convert to numpy arrays
    X_embedding = np.array(representations_embedding)  # (N, embed_dim*6)
    X_lstm = np.array(representations_lstm)            # (N, 128)
    X_policy = np.array(representations_policy)        # (N, 2)
    
    # Ensure X_policy is 2D
    if X_policy.ndim == 1:
        X_policy = X_policy.reshape(-1, 1)
    
    y_game = np.array(labels_game)
    y_opponent = np.array(labels_opponent)
    y_condition = np.array(labels_condition)
    
    print(f"\nData collected: {X_lstm.shape[0]} samples")
    print(f"  Input embeddings: {X_embedding.shape}")
    print(f"  LSTM hidden:      {X_lstm.shape}")
    print(f"  Policy output:    {X_policy.shape}")
    
    # Split train/test (80/20)
    n_samples = len(X_lstm)
    n_train = int(0.8 * n_samples)
    
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    # Split all representations
    X_embedding_train, X_embedding_test = X_embedding[train_idx], X_embedding[test_idx]
    X_lstm_train, X_lstm_test = X_lstm[train_idx], X_lstm[test_idx]
    X_policy_train, X_policy_test = X_policy[train_idx], X_policy[test_idx]
    
    y_game_train, y_game_test = y_game[train_idx], y_game[test_idx]
    y_opponent_train, y_opponent_test = y_opponent[train_idx], y_opponent[test_idx]
    y_condition_train, y_condition_test = y_condition[train_idx], y_condition[test_idx]
    
    
    # =========================================================================
    # TRAIN LINEAR PROBES ON ALL THREE LAYERS
    # =========================================================================
    print("\n" + "="*60)
    print("Training Linear Probes on 3 Layers")
    print("="*60)
    
    results = {}
    accuracies = {
        'layer': [],
        'task': [],
        'accuracy': []
    }
    
    # Helper function to train and evaluate probes
    def train_probe(X_train, X_test, y_train, y_test, task_name, layer_name):
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        accuracies['layer'].append(layer_name)
        accuracies['task'].append(task_name)
        accuracies['accuracy'].append(acc)
        return clf, acc
    
    # Train probes for all 3 layers × 3 tasks = 9 classifiers
    print("\nTraining classifiers...")
    
    # Layer 1: Input Embeddings
    clf_game_embed, game_acc_embed = train_probe(
        X_embedding_train, X_embedding_test, y_game_train, y_game_test, 'game', 'Input Embedding'
    )
    clf_opponent_embed, opponent_acc_embed = train_probe(
        X_embedding_train, X_embedding_test, y_opponent_train, y_opponent_test, 'opponent', 'Input Embedding'
    )
    clf_condition_embed, condition_acc_embed = train_probe(
        X_embedding_train, X_embedding_test, y_condition_train, y_condition_test, 'condition', 'Input Embedding'
    )
    
    # Layer 2: LSTM Hidden States
    clf_game_lstm, game_acc_lstm = train_probe(
        X_lstm_train, X_lstm_test, y_game_train, y_game_test, 'game', 'LSTM Hidden'
    )
    clf_opponent_lstm, opponent_acc_lstm = train_probe(
        X_lstm_train, X_lstm_test, y_opponent_train, y_opponent_test, 'opponent', 'LSTM Hidden'
    )
    clf_condition_lstm, condition_acc_lstm = train_probe(
        X_lstm_train, X_lstm_test, y_condition_train, y_condition_test, 'condition', 'LSTM Hidden'
    )
    
    # Layer 3: Policy Head Outputs
    clf_game_policy, game_acc_policy = train_probe(
        X_policy_train, X_policy_test, y_game_train, y_game_test, 'game', 'Policy Output'
    )
    clf_opponent_policy, opponent_acc_policy = train_probe(
        X_policy_train, X_policy_test, y_opponent_train, y_opponent_test, 'opponent', 'Policy Output'
    )
    clf_condition_policy, condition_acc_policy = train_probe(
        X_policy_train, X_policy_test, y_condition_train, y_condition_test, 'condition', 'Policy Output'
    )
    
    # Print results
    print(f"\n{'Layer':<20} {'Game':<12} {'Opponent':<12} {'Condition':<12}")
    print("="*60)
    print(f"{'Input Embedding':<20} {game_acc_embed:>10.2%}  {opponent_acc_embed:>10.2%}  {condition_acc_embed:>10.2%}")
    print(f"{'LSTM Hidden':<20} {game_acc_lstm:>10.2%}  {opponent_acc_lstm:>10.2%}  {condition_acc_lstm:>10.2%}")
    print(f"{'Policy Output':<20} {game_acc_policy:>10.2%}  {opponent_acc_policy:>10.2%}  {condition_acc_policy:>10.2%}")
    
    # Store results
    results['accuracies'] = accuracies
    results['probes_embedding'] = {
        'game_accuracy': game_acc_embed,
        'opponent_accuracy': opponent_acc_embed,
        'condition_accuracy': condition_acc_embed
    }
    results['probes_lstm'] = {
        'game_accuracy': game_acc_lstm,
        'opponent_accuracy': opponent_acc_lstm,
        'condition_accuracy': condition_acc_lstm
    }
    results['probes_policy'] = {
        'game_accuracy': game_acc_policy,
        'opponent_accuracy': opponent_acc_policy,
        'condition_accuracy': condition_acc_policy
    }
    
    # =========================================================================
    # ANALYSIS 1: Coefficient Sparsity (LSTM Layer)
    # =========================================================================
    print("\n" + "="*60)
    print("ANALYSIS 1: Coefficient Sparsity Analysis (LSTM Layer)")
    print("="*60)
    
    coef_analysis = []
    
    def analyze_coefficients(clf, task_name):
        """Analyze coefficient sparsity and effective dimensionality."""
        coef = clf.coef_  # Shape: (n_classes, n_features) or (1, n_features) for binary
        
        # Flatten if multi-class (take norm across classes for each feature)
        if coef.ndim == 2 and coef.shape[0] > 1:
            coef_importance = np.linalg.norm(coef, axis=0)  # L2 norm across classes
        else:
            coef_importance = np.abs(coef.flatten())
        
        # Sparsity analysis at different thresholds
        total_dims = len(coef_importance)
        max_coef = coef_importance.max()
        
        sparsity_results = {}
        for threshold in [0.01, 0.05, 0.1, 0.2]:
            active_dims = (coef_importance > threshold * max_coef).sum()
            sparsity_results[f'active_dims_{int(threshold*100)}pct'] = active_dims
            sparsity_results[f'sparsity_{int(threshold*100)}pct'] = 1 - (active_dims / total_dims)
        
        # Effective dimensionality (participation ratio)
        coef_norm = coef_importance / (coef_importance.sum() + 1e-10)
        effective_dims = 1.0 / (np.sum(coef_norm ** 2) + 1e-10)
        
        return {
            'task': task_name,
            'total_dims': total_dims,
            'effective_dims': effective_dims,
            'max_coef': max_coef,
            **sparsity_results
        }
    
    # Analyze coefficients from LSTM classifiers
    coef_analysis.append(analyze_coefficients(clf_game_lstm, 'game'))
    coef_analysis.append(analyze_coefficients(clf_opponent_lstm, 'opponent'))
    coef_analysis.append(analyze_coefficients(clf_condition_lstm, 'condition'))
    
    df_coef = pd.DataFrame(coef_analysis)
    
    print(f"\nEffective Dimensionality (lower = more sparse separation):")
    for _, row in df_coef.iterrows():
        print(f"  {row['task']:9s}: {row['effective_dims']:6.1f} / {row['total_dims']:3d} dims " +
              f"({row['effective_dims']/row['total_dims']:.1%})")
    
    print(f"\nActive Dimensions at 10% threshold:")
    for _, row in df_coef.iterrows():
        print(f"  {row['task']:9s}: {row['active_dims_10pct']:3d} dims " +
              f"({row['sparsity_10pct']:.1%} sparsity)")
    
    # =========================================================================
    # ANALYSIS 2: Behavioral Cluster Correlation
    # =========================================================================
    print("\n" + "="*60)
    print("ANALYSIS 2: Behavioral Cluster Correlation")
    print("="*60)
    
    # Load behavioral analysis if available
    behavioral_csv = Path("results/task_opponent_918988_analysis/behavioral_clusters.csv")
    if behavioral_csv.exists():
        df_behavior = pd.read_csv(behavioral_csv)
        
        # Create mapping from (game, opponent) to cluster
        behavior_mapping = {}
        for _, row in df_behavior.iterrows():
            game = row['game']
            opponent = row['opponent']
            cluster = row['cluster']
            behavior_mapping[(game, opponent)] = cluster
        
        # Compute average LSTM representation per condition
        condition_representations = {}
        for cond_idx, (game, opponent) in enumerate(sorted(set(condition_keys))):
            # Get all representations for this condition
            mask = np.array([(g, o) == (game, opponent) for g, o in condition_keys])
            if mask.sum() > 0:
                condition_representations[(game, opponent)] = X_lstm[mask].mean(axis=0)
        
        # Compute within-cluster vs between-cluster similarity
        clusters = list(set(behavior_mapping.values()))
        cluster_similarities = {c: [] for c in clusters}
        between_cluster_similarities = []
        
        for (game_i, opp_i), repr_i in condition_representations.items():
            cluster_i = behavior_mapping.get((game_i, opp_i))
            if cluster_i is None:
                continue
                
            for (game_j, opp_j), repr_j in condition_representations.items():
                if (game_i, opp_i) == (game_j, opp_j):
                    continue  # Skip self-comparison
                
                cluster_j = behavior_mapping.get((game_j, opp_j))
                if cluster_j is None:
                    continue
                
                # Cosine similarity
                sim = np.dot(repr_i, repr_j) / (np.linalg.norm(repr_i) * np.linalg.norm(repr_j) + 1e-10)
                
                if cluster_i == cluster_j:
                    cluster_similarities[cluster_i].append(sim)
                else:
                    between_cluster_similarities.append(sim)
        
        # Report statistics
        print(f"\nWithin-cluster representation similarity:")
        for cluster in sorted(clusters):
            if len(cluster_similarities[cluster]) > 0:
                mean_sim = np.mean(cluster_similarities[cluster])
                std_sim = np.std(cluster_similarities[cluster])
                print(f"  Cluster {cluster}: {mean_sim:.3f} ± {std_sim:.3f} (n={len(cluster_similarities[cluster])})")
        
        if len(between_cluster_similarities) > 0:
            mean_between = np.mean(between_cluster_similarities)
            std_between = np.std(between_cluster_similarities)
            print(f"\nBetween-cluster similarity: {mean_between:.3f} ± {std_between:.3f}")
            
            # Compute effect size
            within_all = [s for sims in cluster_similarities.values() for s in sims]
            if len(within_all) > 0:
                mean_within = np.mean(within_all)
                print(f"Overall within-cluster:     {mean_within:.3f}")
                print(f"Effect (within - between):  {mean_within - mean_between:+.3f}")
        
        # Save cluster correlation analysis
        cluster_corr_data = []
        for cluster in sorted(clusters):
            if len(cluster_similarities[cluster]) > 0:
                cluster_corr_data.append({
                    'cluster': cluster,
                    'within_similarity_mean': np.mean(cluster_similarities[cluster]),
                    'within_similarity_std': np.std(cluster_similarities[cluster]),
                    'n_comparisons': len(cluster_similarities[cluster])
                })
        
        if len(between_cluster_similarities) > 0:
            cluster_corr_data.append({
                'cluster': 'between',
                'within_similarity_mean': np.mean(between_cluster_similarities),
                'within_similarity_std': np.std(between_cluster_similarities),
                'n_comparisons': len(between_cluster_similarities)
            })
        
        df_cluster_corr = pd.DataFrame(cluster_corr_data)
        df_cluster_corr.to_csv(output_dir / 'behavioral_cluster_correlation.csv', index=False)
        
    else:
        print(f"\nBehavioral cluster data not found at {behavioral_csv}")
        print("Skipping behavioral correlation analysis.")
    
    # Save coefficient analysis
    df_coef.to_csv(output_dir / 'coefficient_sparsity_analysis.csv', index=False)
    
    # Save probing results for all three layers
    probe_results_df = pd.DataFrame(accuracies)
    probe_results_df.to_csv(output_dir / 'linear_probing_results.csv', index=False)
    
    # Plot 3-layer comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    tasks = ['Game', 'Opponent', 'Condition']
    layers = ['Input Embedding', 'LSTM Hidden', 'Policy Output']
    colors = ['steelblue', 'coral', 'forestgreen']
    
    for task_idx, task in enumerate(['game', 'opponent', 'condition']):
        ax = axes[task_idx]
        task_data = probe_results_df[probe_results_df['task'] == task]
        
        # Get accuracies in layer order
        accs = []
        for layer in layers:
            acc = task_data[task_data['layer'] == layer]['accuracy'].values[0]
            accs.append(acc)
        
        x = np.arange(len(layers))
        bars = ax.bar(x, accs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Classification Accuracy', fontsize=11)
        ax.set_title(f'{tasks[task_idx]} Classification', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=15, ha='right', fontsize=9)
        ax.set_ylim([0, 1.05])
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, acc) in enumerate(zip(bars, accs)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{acc:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'linear_probing_layer_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot confusion matrices (LSTM layer)
    from sklearn.metrics import confusion_matrix
    
    y_game_pred = clf_game_lstm.predict(X_lstm_test)
    y_opponent_pred = clf_opponent_lstm.predict(X_lstm_test)
    y_condition_pred = clf_condition_lstm.predict(X_lstm_test)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Game confusion matrix
    cm_game = confusion_matrix(y_game_test, y_game_pred)
    im1 = axes[0].imshow(cm_game, cmap='Blues', aspect='auto')
    axes[0].set_title(f'Game Classification (LSTM)\n(Accuracy: {game_acc_lstm:.2%})')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_xticks(range(3))
    axes[0].set_yticks(range(3))
    axes[0].set_xticklabels(['PD', 'HD', 'SH'])
    axes[0].set_yticklabels(['PD', 'HD', 'SH'])
    
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, str(cm_game[i, j]), ha='center', va='center')
    
    plt.colorbar(im1, ax=axes[0])
    
    # Opponent confusion matrix
    cm_opponent = confusion_matrix(y_opponent_test, y_opponent_pred)
    im2 = axes[1].imshow(cm_opponent, cmap='Greens', aspect='auto')
    axes[1].set_title(f'Opponent Classification (LSTM)\n(Accuracy: {opponent_acc_lstm:.2%})')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_xticks(range(5))
    axes[1].set_yticks(range(5))
    axes[1].set_xticklabels(['0.1', '0.3', '0.5', '0.7', '0.9'])
    axes[1].set_yticklabels(['0.1', '0.3', '0.5', '0.7', '0.9'])
    
    for i in range(5):
        for j in range(5):
            axes[1].text(j, i, str(cm_opponent[i, j]), ha='center', va='center', fontsize=8)
    
    plt.colorbar(im2, ax=axes[1])
    
    # Condition confusion matrix (15x15)
    cm_condition = confusion_matrix(y_condition_test, y_condition_pred)
    im3 = axes[2].imshow(cm_condition, cmap='Oranges', aspect='auto')
    axes[2].set_title(f'Condition Classification (LSTM)\n(Accuracy: {condition_acc_lstm:.2%})')
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('True')
    
    # Create abbreviated labels for 15 conditions
    condition_labels_short = []
    for game_name in sorted(game_to_idx.keys()):
        game_abbr = get_game_abbreviation(game_name)
        for opp in sorted(opponent_to_idx.keys()):
            condition_labels_short.append(f"{game_abbr}-{opp}")
    
    axes[2].set_xticks(range(15))
    axes[2].set_yticks(range(15))
    axes[2].set_xticklabels(condition_labels_short, rotation=90, fontsize=6)
    axes[2].set_yticklabels(condition_labels_short, fontsize=6)
    
    # Add text annotations for diagonal only (too many cells otherwise)
    for i in range(15):
        axes[2].text(i, i, str(cm_condition[i, i]), ha='center', va='center', fontsize=6, fontweight='bold')
    
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'linear_probing_confusion.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nLinear probing analysis complete.")
    print(f"Results saved to {output_dir}")
    
    return results


# ============================================================================
# METHOD 5: NETWORK COMPRESSION METRICS
# ============================================================================

def analyze_compression_metrics(models_data: dict, device: torch.device, output_dir: Path):
    """
    Analyze network compression and pruning sensitivity.
    
    Metrics:
    - Weight magnitude distribution
    - Pruning sensitivity (performance vs. sparsity)
    - Intrinsic dimensionality
    """
    print("\n" + "="*80)
    print("METHOD 5: NETWORK COMPRESSION METRICS")
    print("="*80)
    
    if len(models_data) == 0:
        print("Warning: No models loaded. Cannot perform compression analysis.")
        return pd.DataFrame()
    
    results = []
    
    for (game, opponent), model_info in models_data.items():
        model = model_info['model']
        
        # Collect all weights
        all_weights = []
        for param in model.parameters():
            all_weights.append(param.data.flatten().cpu().numpy())
        
        all_weights = np.concatenate(all_weights)
        
        # Weight statistics
        weight_stats = {
            'game': game,
            'opponent': opponent,
            'total_params': len(all_weights),
            'mean_weight': np.mean(all_weights),
            'std_weight': np.std(all_weights),
            'median_weight': np.median(np.abs(all_weights)),
            'max_weight': np.max(np.abs(all_weights)),
            'sparsity_0.001': (np.abs(all_weights) < 0.001).mean(),
            'sparsity_0.01': (np.abs(all_weights) < 0.01).mean(),
            'sparsity_0.1': (np.abs(all_weights) < 0.1).mean(),
        }
        
        # Compute effective number of parameters (via magnitude thresholding)
        sorted_weights = np.sort(np.abs(all_weights))[::-1]
        cumsum = np.cumsum(sorted_weights**2)
        total_power = cumsum[-1]
        
        # Effective params (90% of total power)
        effective_params_90 = np.searchsorted(cumsum, 0.90 * total_power) + 1
        # Effective params (95% of total power)
        effective_params_95 = np.searchsorted(cumsum, 0.95 * total_power) + 1
        
        weight_stats['effective_params_90'] = effective_params_90
        weight_stats['effective_params_95'] = effective_params_95
        weight_stats['compression_ratio_90'] = len(all_weights) / effective_params_90
        weight_stats['compression_ratio_95'] = len(all_weights) / effective_params_95
        
        results.append(weight_stats)
    
    df_compression = pd.DataFrame(results)
    df_compression.to_csv(output_dir / 'compression_metrics.csv', index=False)
    
    # Plot weight distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Weight magnitude histogram (by game)
    for game in ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']:
        subset = df_compression[df_compression['game'] == game]
        axes[0, 0].hist(subset['median_weight'], bins=20, alpha=0.5, label=game)
    
    axes[0, 0].set_xlabel('Median Absolute Weight')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Weight Magnitude Distribution by Game')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Sparsity levels
    sparsity_cols = ['sparsity_0.001', 'sparsity_0.01', 'sparsity_0.1']
    x = np.arange(len(sparsity_cols))
    width = 0.25
    
    for i, game in enumerate(['prisoners-dilemma', 'hawk-dove', 'stag-hunt']):
        game_data = df_compression[df_compression['game'] == game][sparsity_cols].mean()
        axes[0, 1].bar(x + i*width, game_data, width, label=game, alpha=0.8)
    
    axes[0, 1].set_xlabel('Sparsity Threshold')
    axes[0, 1].set_ylabel('Fraction of Weights')
    axes[0, 1].set_title('Sparsity Levels by Game')
    axes[0, 1].set_xticks(x + width)
    axes[0, 1].set_xticklabels(['< 0.001', '< 0.01', '< 0.1'])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Compression ratio
    games = df_compression['game'].unique()
    opponents = df_compression['opponent'].unique()
    
    for opp in opponents:
        opp_data = df_compression[df_compression['opponent'] == opp].groupby('game')['compression_ratio_90'].mean()
        axes[1, 0].plot(games, opp_data, marker='o', label=f'p={opp}', linewidth=2)
    
    axes[1, 0].set_xlabel('Game')
    axes[1, 0].set_ylabel('Compression Ratio (90%)')
    axes[1, 0].set_title('Network Compression Potential')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticklabels(['PD', 'HD', 'SH'])
    
    # Effective parameters
    summary = df_compression.groupby('game')[['effective_params_90', 'effective_params_95', 'total_params']].mean()
    
    x = np.arange(len(games))
    width = 0.25
    
    axes[1, 1].bar(x - width, summary['total_params'], width, label='Total', alpha=0.8)
    axes[1, 1].bar(x, summary['effective_params_95'], width, label='Effective (95%)', alpha=0.8)
    axes[1, 1].bar(x + width, summary['effective_params_90'], width, label='Effective (90%)', alpha=0.8)
    
    axes[1, 1].set_xlabel('Game')
    axes[1, 1].set_ylabel('Number of Parameters')
    axes[1, 1].set_title('Effective vs Total Parameters')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(['PD', 'HD', 'SH'])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'compression_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Compression analysis complete.")
    print(f"Mean compression ratio (90%): {df_compression['compression_ratio_90'].mean():.2f}x")
    print(f"Mean sparsity (< 0.01): {df_compression['sparsity_0.01'].mean():.2%}")
    
    return df_compression


# ============================================================================
# METHOD 6: INFORMATION FLOW ANALYSIS
# ============================================================================

def analyze_information_flow(models_data: dict, device: torch.device, output_dir: Path):
    """
    Analyze information flow from inputs → hidden states → outputs.
    Measures how much information is preserved and compressed at each layer.
    
    Metrics:
    - I(X; H): Mutual information between inputs and hidden states
    - I(H; Y): Mutual information between hidden states and outputs  
    - Compression ratio: I(H; Y) / I(X; H)
    - Information retention: % of input info that reaches output
    """
    print("\n" + "="*60)
    print("METHOD 6: Information Flow Analysis")
    print("="*60)
    
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    
    results = []
    
    # For each model, generate test episodes and compute info flow
    for (game, opponent), data in models_data.items():
        model = data['model']
        model.eval()
        
        # Generate test episodes (100 episodes, 50 rounds each)
        n_episodes = 100
        n_rounds = 50
        
        inputs_all = []
        hiddens_all = []
        outputs_all = []  # Policy logits
        actions_all = []
        
        game_obj = GameFactory.create_game(game)
        
        with torch.no_grad():
            for episode in range(n_episodes):
                # Initialize hidden state
                h = torch.zeros(model.num_layers, 1, model.hidden_size, device=device)
                c = torch.zeros(model.num_layers, 1, model.hidden_size, device=device)
                hidden = (h, c)
                
                # Reset game state variables
                prev_opp_action = 0
                prev_agent_action = 0
                prev_agent_reward = 0
                prev_opp_reward = 0
                
                for round_num in range(n_rounds):
                    # Construct 9-element state vector
                    payoff_matrix = game_obj.get_payoff_matrix().flatten()
                    round_normalized = round_num / n_rounds
                    
                    state_vector = np.concatenate([
                        payoff_matrix,
                        [round_normalized],
                        [prev_opp_action],
                        [prev_agent_action],
                        [prev_agent_reward],
                        [prev_opp_reward]
                    ])
                    
                    state_t = torch.tensor(state_vector, dtype=torch.float32, device=device).unsqueeze(0)
                    
                    # Forward pass
                    policy_logits, _, _, hidden = model(state_t, hidden)
                    
                    # Store data
                    inputs_all.append(state_vector)
                    hiddens_all.append(hidden[0][:, 0, :].cpu().numpy().flatten())  # Final layer hidden state
                    outputs_all.append(policy_logits.squeeze(0).cpu().numpy())  # Remove batch dimension
                    
                    # Sample action
                    probs = torch.softmax(policy_logits, dim=-1)
                    action = torch.multinomial(probs, 1).item()
                    actions_all.append(action)
                    
                    # Simulate opponent action (probabilistic)
                    opp_action = 1 if np.random.rand() < opponent else 0
                    
                    # Update history variables
                    prev_agent_action = action
                    prev_opp_action = opp_action
                    R = game_obj.get_payoff_matrix()
                    prev_agent_reward = R[action, opp_action]
                    prev_opp_reward = R[opp_action, action]
        
        # Convert to numpy arrays
        X = np.array(inputs_all)  # (n_samples, 9)
        H = np.array(hiddens_all)  # (n_samples, hidden_size)
        Y_logits = np.array(outputs_all)  # (n_samples, 2)
        Y_actions = np.array(actions_all)  # (n_samples,)
        
        print(f"\n{get_game_abbreviation(game)} vs p={opponent}:")
        print(f"  Generated {len(X)} state transitions")
        
        # =========================================================================
        # 1. Compute I(X; H) - Mutual information between inputs and hidden states
        # =========================================================================
        # We'll use the actions as a proxy for information content
        # MI between each input dimension and actions
        mi_x_actions = mutual_info_classif(X, Y_actions, discrete_features=False, random_state=42)
        I_X_Actions = np.sum(mi_x_actions)
        
        # MI between hidden states and actions (how much info about actions is in H)
        # Sample subset of hidden dimensions to avoid computational explosion
        hidden_sample_dims = min(32, H.shape[1])
        H_sample = H[:, :hidden_sample_dims]
        mi_h_actions = mutual_info_classif(H_sample, Y_actions, discrete_features=False, random_state=42)
        I_H_Actions = np.sum(mi_h_actions) * (H.shape[1] / hidden_sample_dims)  # Scale up
        
        # =========================================================================
        # 2. Compute effective information dimensionality
        # =========================================================================
        # How many dimensions of H actually carry information about actions?
        mi_threshold = 0.01  # Bits
        active_dims = np.sum(mi_h_actions > mi_threshold)
        effective_dim_pct = active_dims / hidden_sample_dims
        
        # =========================================================================
        # 3. Compute information retention
        # =========================================================================
        # What fraction of input information makes it to hidden state?
        info_retention = I_H_Actions / max(I_X_Actions, 1e-6)
        
        # =========================================================================
        # 4. Compute output compression
        # =========================================================================
        # Policy logits are 2-dimensional - this is the final compression
        # Compute entropy of action distribution
        from scipy.stats import entropy
        action_probs = np.bincount(Y_actions, minlength=2) / len(Y_actions)
        action_entropy = entropy(action_probs, base=2)  # bits
        
        # Maximum possible entropy for binary actions
        max_entropy = 1.0  # log2(2) = 1 bit
        
        # How much of the information in H is preserved in the action distribution?
        output_compression = action_entropy / max(I_H_Actions, 1e-6)
        
        results.append({
            'game': get_game_abbreviation(game),
            'opponent': opponent,
            'I_X_Actions': I_X_Actions,
            'I_H_Actions': I_H_Actions,
            'info_retention': info_retention,
            'active_dims': active_dims,
            'active_dims_pct': effective_dim_pct,
            'action_entropy': action_entropy,
            'max_entropy': max_entropy,
            'output_compression': output_compression,
            'compression_ratio': I_X_Actions / max(action_entropy, 1e-6)
        })
        
        print(f"  I(X → Actions): {I_X_Actions:.3f} bits")
        print(f"  I(H → Actions): {I_H_Actions:.3f} bits")
        print(f"  Info retention: {info_retention:.2%}")
        print(f"  Active hidden dims: {active_dims}/{hidden_sample_dims} ({effective_dim_pct:.1%})")
        print(f"  Action entropy: {action_entropy:.3f} / {max_entropy:.3f} bits")
        print(f"  Compression ratio: {I_X_Actions / max(action_entropy, 1e-6):.1f}x")
    
    df_info = pd.DataFrame(results)
    df_info.to_csv(output_dir / 'information_flow.csv', index=False)
    
    # =========================================================================
    # Visualization 1: Information flow diagram
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Information retention by game
    ax = axes[0, 0]
    games_order = ['PD', 'HD', 'SH']
    for game_abbrev in games_order:
        game_data = df_info[df_info['game'] == game_abbrev]
        ax.plot(game_data['opponent'], game_data['info_retention'], 
                marker='o', label=game_abbrev, linewidth=2)
    ax.set_xlabel('Opponent Defection Probability', fontsize=11)
    ax.set_ylabel('Information Retention (I(H→Y) / I(X→Y))', fontsize=11)
    ax.set_title('Information Flow: Input → Hidden → Output', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim([0, None])
    
    # Plot 2: Active dimensions by condition
    ax = axes[0, 1]
    x_pos = np.arange(len(df_info))
    colors = [{'PD': 'steelblue', 'HD': 'coral', 'SH': 'forestgreen'}[g] 
              for g in df_info['game']]
    ax.bar(x_pos, df_info['active_dims_pct'], color=colors, alpha=0.7)
    ax.set_xlabel('Condition Index', fontsize=11)
    ax.set_ylabel('Active Hidden Dimensions (%)', fontsize=11)
    ax.set_title('Information-Carrying Dimensions', fontweight='bold')
    ax.set_ylim([0, 1])
    ax.axhline(y=df_info['active_dims_pct'].mean(), color='red', 
               linestyle='--', linewidth=1.5, label=f"Mean: {df_info['active_dims_pct'].mean():.1%}")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: Compression ratio by game
    ax = axes[1, 0]
    for game_abbrev in games_order:
        game_data = df_info[df_info['game'] == game_abbrev]
        ax.plot(game_data['opponent'], game_data['compression_ratio'], 
                marker='s', label=game_abbrev, linewidth=2)
    ax.set_xlabel('Opponent Defection Probability', fontsize=11)
    ax.set_ylabel('Compression Ratio (Input/Output bits)', fontsize=11)
    ax.set_title('Network Compression: Input → Output', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim([0, None])
    
    # Plot 4: Action entropy (output complexity)
    ax = axes[1, 1]
    for game_abbrev in games_order:
        game_data = df_info[df_info['game'] == game_abbrev]
        ax.plot(game_data['opponent'], game_data['action_entropy'], 
                marker='^', label=game_abbrev, linewidth=2)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Max entropy (1 bit)')
    ax.set_xlabel('Opponent Defection Probability', fontsize=11)
    ax.set_ylabel('Action Entropy (bits)', fontsize=11)
    ax.set_title('Output Complexity (Action Distribution)', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'information_flow_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Summary statistics
    # =========================================================================
    print("\n" + "="*60)
    print("INFORMATION FLOW SUMMARY")
    print("="*60)
    print(f"\nMean information retention: {df_info['info_retention'].mean():.2%}")
    print(f"Mean active dimensions: {df_info['active_dims_pct'].mean():.1%}")
    print(f"Mean compression ratio: {df_info['compression_ratio'].mean():.1f}x")
    print(f"Mean action entropy: {df_info['action_entropy'].mean():.3f} bits (max: 1.0)")
    
    print("\nBy game:")
    for game_abbrev in games_order:
        game_data = df_info[df_info['game'] == game_abbrev]
        print(f"  {game_abbrev}: retention={game_data['info_retention'].mean():.2%}, "
              f"compression={game_data['compression_ratio'].mean():.1f}x, "
              f"entropy={game_data['action_entropy'].mean():.3f} bits")
    
    return df_info


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("NETWORK-LEVEL REPRESENTATION ANALYSIS")
    print("Experiment: 918988 (train) / 918989 (test)")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup paths
    train_dir = project_root / 'experiments' / 'generalization_matrix_train_918988' / 'training'
    output_dir = project_root / 'Results' / 'task_opponent_918988_analysis' / 'network_representation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all models (15 conditions, use seed 0 - first seed)
    games = ['prisoners-dilemma', 'hawk-dove', 'stag-hunt']
    opponents = [0.1, 0.3, 0.5, 0.7, 0.9]
    seed = 0
    
    models_data = {}
    
    print("\nLoading models...")
    for game_idx, game in enumerate(games):
        for opp_idx, opponent in enumerate(opponents):
            condition_id = game_idx * 5 + opp_idx
            
            # Find the experiment directory (there should be one timestamped dir)
            condition_dir = train_dir / f'condition_{condition_id}_seed_{seed}'
            if condition_dir.exists():
                # Get the first (should be only) experiment directory
                experiment_dirs = list(condition_dir.glob('generalization_matrix_task_*'))
                if len(experiment_dirs) > 0:
                    # Checkpoint is named {game}_final_checkpoint.pth
                    checkpoint_path = experiment_dirs[0] / 'checkpoints' / f'{game}_final_checkpoint.pth'
                else:
                    checkpoint_path = None
            else:
                checkpoint_path = None
            
            if checkpoint_path is not None and checkpoint_path.exists():
                model = load_checkpoint(checkpoint_path, device)
                models_data[(game, opponent)] = {
                    'model': model,
                    'condition_id': condition_id,
                    'seed': seed
                }
                print(f"  Loaded: {game} vs p={opponent} (condition {condition_id})")
            else:
                print(f"  Warning: Missing checkpoint for {game} vs p={opponent}")
    
    print(f"\nLoaded {len(models_data)} models.")
    
    # Run all 5 methods
    print("\n" + "="*80)
    print("RUNNING ALL 6 NETWORK REPRESENTATION METHODS")
    print("="*80)
    
    # Method 1: SVD
    df_svd = analyze_svd(models_data, output_dir)
    
    # Method 2: Gradient Flow
    df_grad, grad_alignment = analyze_gradient_flow(models_data, device, output_dir)
    
    # Method 3: Activation Patterns
    df_activation, act_similarity = analyze_activation_patterns(models_data, device, output_dir)
    
    # Create combined activation & gradient matrix figure
    create_combined_matrix_figure(models_data, act_similarity, grad_alignment, output_dir)
    
    # Method 4: Linear Probing
    probe_results = analyze_linear_probing(models_data, device, output_dir)
    
    # Method 5: Compression Metrics
    df_compression = analyze_compression_metrics(models_data, device, output_dir)
    
    # Method 6: Information Flow
    df_info_flow = analyze_information_flow(models_data, device, output_dir)
    
    # Final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - svd_analysis.csv, svd_spectrum.png, svd_effective_rank.png")
    print(f"  - gradient_flow.csv, gradient_flow_analysis.png, gradient_alignment_matrix.npy")
    print(f"  - activation_statistics.csv, activation_patterns.png, activation_similarity_matrix.npy")
    print(f"  - activation_similarity_matrix_standalone.png")
    print(f"  - combined_activation_gradient_matrices.png (A: Activation, B: Gradient)")
    print(f"  - linear_probing_results.csv, linear_probing_confusion.png")
    print(f"  - compression_metrics.csv, compression_analysis.png")
    print(f"  - information_flow.csv, information_flow_analysis.png")
    
    print("\n" + "="*80)
    print("KEY FINDINGS SUMMARY")
    print("="*80)
    
    if len(models_data) == 0:
        print("\nNo models were loaded. Cannot generate summary.")
    else:
        print(f"\n1. SVD Analysis:")
        if len(df_svd) > 0:
            print(f"   - Mean effective rank (90%): {df_svd['effective_rank_90'].mean():.2f}")
            print(f"   - Mean participation ratio: {df_svd['participation_ratio'].mean():.2f}")
        else:
            print(f"   - No data available")
        
        print(f"\n2. Gradient Flow:")
        if len(df_grad) > 0:
            print(f"   - Mean gradient norm: {df_grad['grad_norm'].mean():.6f}")
            print(f"   - Gradient alignment range: [{grad_alignment.min():.3f}, {grad_alignment.max():.3f}]")
        else:
            print(f"   - No data available")
        
        print(f"\n3. Activation Patterns:")
        if len(df_activation) > 0:
            print(f"   - Mean activation sparsity: {df_activation['sparsity'].mean():.2%}")
            print(f"   - Activation similarity range: [{act_similarity.min():.3f}, {act_similarity.max():.3f}]")
        else:
            print(f"   - No data available")
        
        print(f"\n4. Linear Probing (3 Layers):")
        if len(probe_results) > 0 and 'probes_embedding' in probe_results:
            print(f"   Input Embeddings:")
            print(f"     - Game: {probe_results['probes_embedding']['game_accuracy']:.2%}, " +
                  f"Opponent: {probe_results['probes_embedding']['opponent_accuracy']:.2%}, " +
                  f"Condition: {probe_results['probes_embedding']['condition_accuracy']:.2%}")
            print(f"   LSTM Hidden States:")
            print(f"     - Game: {probe_results['probes_lstm']['game_accuracy']:.2%}, " +
                  f"Opponent: {probe_results['probes_lstm']['opponent_accuracy']:.2%}, " +
                  f"Condition: {probe_results['probes_lstm']['condition_accuracy']:.2%}")
            print(f"   Policy Outputs:")
            print(f"     - Game: {probe_results['probes_policy']['game_accuracy']:.2%}, " +
                  f"Opponent: {probe_results['probes_policy']['opponent_accuracy']:.2%}, " +
                  f"Condition: {probe_results['probes_policy']['condition_accuracy']:.2%}")
        else:
            print(f"   - No data available")
        
        print(f"\n5. Compression Metrics:")
        if len(df_compression) > 0:
            print(f"   - Mean compression ratio (90%): {df_compression['compression_ratio_90'].mean():.2f}x")
            print(f"   - Mean sparsity (< 0.01): {df_compression['sparsity_0.01'].mean():.2%}")
        else:
            print(f"   - No data available")
        
        print(f"\n6. Information Flow:")
        if len(df_info_flow) > 0:
            print(f"   - Mean information retention: {df_info_flow['info_retention'].mean():.2%}")
            print(f"   - Mean active dimensions: {df_info_flow['active_dims_pct'].mean():.1%}")
            print(f"   - Mean compression ratio: {df_info_flow['compression_ratio'].mean():.1f}x")
            print(f"   - Mean action entropy: {df_info_flow['action_entropy'].mean():.3f} bits")
        else:
            print(f"   - No data available")


if __name__ == '__main__':
    main()
