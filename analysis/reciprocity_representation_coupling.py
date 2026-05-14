#!/usr/bin/env python3
"""
Reciprocity-Representation Coupling Analysis
============================================

Analyzes relationships between:
- Task complexity and representation structure
- Opponent complexity and social learning
- Representation-behavior coupling (degeneracy)
- Generalization from representation similarity
- Integrated complexity framework

Author: Research Team
Date: May 12, 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, List
from scipy.stats import spearmanr, pearsonr, f_oneway
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import linear_kernel
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

GAME_DISPLAY_NAMES = {
    'prisoners-dilemma': 'PD',
    'hawk-dove': 'HD',
    'stag-hunt': 'SH'
}

OPPONENT_COLORS = {
    0.1: '#2E86AB',  # Blue (cooperative)
    0.3: '#6BAED6',
    0.5: '#FDAE6B',  # Orange (neutral)
    0.7: '#E15759',
    0.9: '#C1121F'   # Red (defective)
}


def component_1_task_representation(complexity_df: pd.DataFrame, 
                                   output_dir: Path) -> Tuple[pd.DataFrame, Dict[str, plt.Figure]]:
    """
    Component 1: Task Complexity → Representation Structure
    
    Hypothesis: Complex games (SH) require higher-dimensional representations
    
    Returns:
        - Summary statistics DataFrame
        - Dictionary of figures
    """
    print("\n" + "="*80)
    print("COMPONENT 1: Task Complexity → Representation Structure")
    print("="*80)
    
    figures = {}
    
    # Group by training game
    game_groups = complexity_df.groupby('train_game')
    
    # Statistical test: effective_dimensionality ~ train_game
    game_dims = [group['effective_dimensionality'].values 
                 for name, group in game_groups]
    f_stat, p_value = f_oneway(*game_dims)
    
    print(f"\nANOVA: effective_dimensionality ~ train_game")
    print(f"  F-statistic: {f_stat:.4f}, p-value: {p_value:.4e}")
    
    # Summary statistics
    summary = game_groups.agg({
        'effective_dimensionality': ['mean', 'std', 'count'],
        'embedding_specialization': ['mean', 'std'],
        'weight_l2_norm': ['mean', 'std'],
        'social_ratio': ['mean', 'std'],
        'payoff_variance': ['mean'],
        'coordination_bonus': ['mean']
    }).round(4)
    
    print("\nSummary by Training Game:")
    print(summary)
    
    # Visualization 1: Effective dimensionality by game
    fig1, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Panel A: Bar chart of effective dimensionality
    ax = axes[0]
    game_names = sorted(complexity_df['train_game'].unique())
    game_labels = [GAME_DISPLAY_NAMES.get(g, g) for g in game_names]
    
    means = [game_groups.get_group(g)['effective_dimensionality'].mean() for g in game_names]
    stds = [game_groups.get_group(g)['effective_dimensionality'].std() for g in game_names]
    
    ax.bar(game_labels, means, yerr=stds, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    ax.set_ylabel('Effective Dimensionality\n(95% Variance)', fontsize=11)
    ax.set_xlabel('Training Game', fontsize=11)
    ax.set_title('A. Representation Dimensionality by Game', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Panel B: Embedding specialization by game
    ax = axes[1]
    means = [game_groups.get_group(g)['embedding_specialization'].mean() for g in game_names]
    stds = [game_groups.get_group(g)['embedding_specialization'].std() for g in game_names]
    
    ax.bar(game_labels, means, yerr=stds, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    ax.set_ylabel('Embedding Specialization\n(Std of Weight Magnitudes)', fontsize=11)
    ax.set_xlabel('Training Game', fontsize=11)
    ax.set_title('B. Embedding Specialization by Game', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Panel C: Scatter - payoff variance vs effective dimensionality
    ax = axes[2]
    for game in game_names:
        game_data = complexity_df[complexity_df['train_game'] == game]
        label = GAME_DISPLAY_NAMES.get(game, game)
        ax.scatter(game_data['payoff_variance'], 
                  game_data['effective_dimensionality'],
                  label=label, alpha=0.6, s=50)
    
    ax.set_xlabel('Payoff Variance', fontsize=11)
    ax.set_ylabel('Effective Dimensionality', fontsize=11)
    ax.set_title('C. Task Variance vs Representation Complexity', fontsize=12, fontweight='bold')
    ax.legend(title='Game', fontsize=9)
    ax.grid(alpha=0.3)
    
    # Compute correlation
    corr, corr_p = pearsonr(complexity_df['payoff_variance'], 
                           complexity_df['effective_dimensionality'])
    ax.text(0.05, 0.95, f'r = {corr:.3f}\np = {corr_p:.4f}', 
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    figures['component_1'] = fig1
    
    return summary, figures


def component_2_opponent_social_learning(complexity_df: pd.DataFrame,
                                        output_dir: Path) -> Tuple[pd.DataFrame, Dict[str, plt.Figure]]:
    """
    Component 2: Opponent Complexity → Social Learning
    
    Hypothesis: Unpredictable opponents require more social tracking
    
    Returns:
        - Summary statistics DataFrame
        - Dictionary of figures
    """
    print("\n" + "="*80)
    print("COMPONENT 2: Opponent Complexity → Social Learning")
    print("="*80)
    
    figures = {}
    
    # Check if opponent data exists (task-opponent setup)
    if 'train_opponent' not in complexity_df.columns:
        print("  Skipping: No opponent-specific training (Task setup)")
        return pd.DataFrame(), figures
    
    # Correlations
    corr_social_pred, p_social_pred = spearmanr(
        complexity_df['predictability'],
        complexity_df['social_ratio']
    )
    corr_recip_adapt, p_recip_adapt = spearmanr(
        complexity_df['adaptation_requirement'],
        complexity_df['reciprocity_strength']
    )
    
    print(f"\nCorrelation: opponent_predictability ~ social_ratio")
    print(f"  ρ = {corr_social_pred:.4f}, p = {p_social_pred:.4e}")
    print(f"\nCorrelation: adaptation_requirement ~ reciprocity_strength")
    print(f"  ρ = {corr_recip_adapt:.4f}, p = {p_recip_adapt:.4e}")
    
    # Summary by opponent
    opp_groups = complexity_df.groupby('train_opponent')
    summary = opp_groups.agg({
        'social_ratio': ['mean', 'std'],
        'reciprocity_strength': ['mean', 'std'],
        'predictability': ['mean'],
        'adaptation_requirement': ['mean']
    }).round(4)
    
    print("\nSummary by Training Opponent:")
    print(summary)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel A: Social ratio vs opponent defection probability
    ax = axes[0]
    games = sorted(complexity_df['train_game'].unique())
    
    for game in games:
        game_data = complexity_df[complexity_df['train_game'] == game]
        game_label = GAME_DISPLAY_NAMES.get(game, game)
        
        # Group by opponent and compute means
        opp_means = game_data.groupby('train_opponent').agg({
            'social_ratio': 'mean'
        }).reset_index()
        
        ax.plot(opp_means['train_opponent'], opp_means['social_ratio'],
               marker='o', label=game_label, linewidth=2, markersize=8, alpha=0.7)
    
    ax.set_xlabel('Training Opponent Defection Probability', fontsize=11)
    ax.set_ylabel('Social Ratio', fontsize=11)
    ax.set_title('A. Social Learning vs Opponent Type', fontsize=12, fontweight='bold')
    ax.legend(title='Game', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    
    # Panel B: Reciprocity strength vs opponent defection probability
    ax = axes[1]
    for game in games:
        game_data = complexity_df[complexity_df['train_game'] == game]
        game_label = GAME_DISPLAY_NAMES.get(game, game)
        
        opp_means = game_data.groupby('train_opponent').agg({
            'reciprocity_strength': 'mean'
        }).reset_index()
        
        ax.plot(opp_means['train_opponent'], opp_means['reciprocity_strength'],
               marker='s', label=game_label, linewidth=2, markersize=8, alpha=0.7)
    
    ax.set_xlabel('Training Opponent Defection Probability', fontsize=11)
    ax.set_ylabel('Reciprocity Strength', fontsize=11)
    ax.set_title('B. Reciprocity vs Opponent Type', fontsize=12, fontweight='bold')
    ax.legend(title='Game', fontsize=10)
    ax.grid(alpha=0.3)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    figures['component_2'] = fig
    
    return summary, figures


def component_3_representation_behavior_coupling(
    complexity_df: pd.DataFrame,
    cka_matrix: np.ndarray,
    output_dir: Path
) -> Tuple[pd.DataFrame, Dict[str, plt.Figure]]:
    """
    Component 3: Representation-Behavior Coupling (Degeneracy)
    
    Hypothesis: Similar behaviors can emerge from different representations
    
    Args:
        complexity_df: DataFrame with all metrics
        cka_matrix: CKA similarity matrix (N×N)
    
    Returns:
        - Coupling analysis DataFrame
        - Dictionary of figures
    """
    print("\n" + "="*80)
    print("COMPONENT 3: Representation-Behavior Coupling")
    print("="*80)
    
    figures = {}
    
    # Behavioral clustering (K-means on cooperation and reciprocity)
    X_behavior = complexity_df[['mean_cooperation', 'reciprocity_strength']].values
    
    # Handle NaN in reciprocity
    X_behavior = np.nan_to_num(X_behavior, nan=0.0)
    
    # Standardize
    scaler = StandardScaler()
    X_behavior_scaled = scaler.fit_transform(X_behavior)
    
    # Cluster (k=5)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    behavior_clusters = kmeans.fit_predict(X_behavior_scaled)
    complexity_df['behavior_cluster'] = behavior_clusters
    
    print(f"\nBehavioral Clustering (k=5):")
    for cluster_id in range(5):
        cluster_data = complexity_df[complexity_df['behavior_cluster'] == cluster_id]
        mean_coop = cluster_data['mean_cooperation'].mean()
        mean_recip = cluster_data['reciprocity_strength'].mean()
        count = len(cluster_data)
        print(f"  Cluster {cluster_id}: n={count:2d}, coop={mean_coop:.3f}, recip={mean_recip:.3f}")
    
    # Compute representational diversity within vs between clusters
    # Use CKA matrix to compute similarity
    if cka_matrix is not None and cka_matrix.shape[0] == len(complexity_df):
        # Convert CKA similarity to distance
        cka_distance = 1 - cka_matrix
        
        within_cluster_distances = []
        between_cluster_distances = []
        
        for i in range(len(complexity_df)):
            for j in range(i+1, len(complexity_df)):
                dist = cka_distance[i, j]
                if behavior_clusters[i] == behavior_clusters[j]:
                    within_cluster_distances.append(dist)
                else:
                    between_cluster_distances.append(dist)
        
        within_mean = np.mean(within_cluster_distances)
        between_mean = np.mean(between_cluster_distances)
        diversity_ratio = within_mean / between_mean if between_mean > 0 else 0
        
        print(f"\nRepresentational Diversity:")
        print(f"  Within-cluster distance:  {within_mean:.4f}")
        print(f"  Between-cluster distance: {between_mean:.4f}")
        print(f"  Ratio (within/between):   {diversity_ratio:.4f}")
        print(f"  Interpretation: {'High degeneracy' if diversity_ratio > 0.5 else 'Low degeneracy'}")
        
        coupling_summary = pd.DataFrame({
            'metric': ['within_cluster_dist', 'between_cluster_dist', 'diversity_ratio'],
            'value': [within_mean, between_mean, diversity_ratio]
        })
    else:
        print("  Warning: CKA matrix not provided or size mismatch")
        coupling_summary = pd.DataFrame()
        within_cluster_distances = []
        between_cluster_distances = []
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Panel A: Behavioral clustering
    ax = axes[0]
    scatter = ax.scatter(complexity_df['mean_cooperation'],
                        complexity_df['reciprocity_strength'],
                        c=behavior_clusters, cmap='tab10', s=80, alpha=0.7,
                        edgecolors='black', linewidths=0.5)
    ax.set_xlabel('Mean Cooperation Rate', fontsize=11)
    ax.set_ylabel('Reciprocity Strength', fontsize=11)
    ax.set_title('A. Behavioral Clustering', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.colorbar(scatter, ax=ax, label='Cluster ID')
    
    # Panel B: Within vs between cluster diversity
    ax = axes[1]
    if len(within_cluster_distances) > 0 and len(between_cluster_distances) > 0:
        box_data = [within_cluster_distances, between_cluster_distances]
        bp = ax.boxplot(box_data, labels=['Within\nCluster', 'Between\nCluster'],
                       patch_artist=True, widths=0.6)
        bp['boxes'][0].set_facecolor('#66c2a5')
        bp['boxes'][1].set_facecolor('#fc8d62')
        ax.set_ylabel('Representational Distance\n(1 - CKA)', fontsize=11)
        ax.set_title('B. Representation Diversity', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'CKA data not available', 
               ha='center', va='center', transform=ax.transAxes)
    
    # Panel C: CKA similarity heatmap
    ax = axes[2]
    if cka_matrix is not None and cka_matrix.shape[0] == len(complexity_df):
        # Sort by behavioral cluster for visualization
        sorted_idx = np.argsort(behavior_clusters)
        cka_sorted = cka_matrix[sorted_idx, :][:, sorted_idx]
        
        im = ax.imshow(cka_sorted, cmap='viridis', vmin=0, vmax=1, aspect='auto')
        ax.set_xlabel('Agent Index (sorted by cluster)', fontsize=11)
        ax.set_ylabel('Agent Index (sorted by cluster)', fontsize=11)
        ax.set_title('C. CKA Similarity Matrix', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='CKA Similarity')
        
        # Add cluster boundaries
        cluster_boundaries = []
        current_cluster = behavior_clusters[sorted_idx[0]]
        for i, cluster in enumerate(behavior_clusters[sorted_idx]):
            if cluster != current_cluster:
                cluster_boundaries.append(i)
                current_cluster = cluster
        for boundary in cluster_boundaries:
            ax.axhline(boundary, color='white', linestyle='--', linewidth=1, alpha=0.5)
            ax.axvline(boundary, color='white', linestyle='--', linewidth=1, alpha=0.5)
    else:
        ax.text(0.5, 0.5, 'CKA matrix not available',
               ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    figures['component_3'] = fig
    
    return coupling_summary, figures


def component_4_generalization_similarity(
    complexity_df: pd.DataFrame,
    test_performance_df: pd.DataFrame,
    cka_matrix: np.ndarray,
    output_dir: Path
) -> Tuple[pd.DataFrame, Dict[str, plt.Figure]]:
    """
    Component 4: Generalization from Representation Similarity
    
    Hypothesis: Similar representations → similar generalization patterns
    
    Returns:
        - Correlation analysis DataFrame
        - Dictionary of figures
    """
    print("\n" + "="*80)
    print("COMPONENT 4: Generalization from Representation Similarity")
    print("="*80)
    
    figures = {}
    
    # Compute generalization similarity matrix
    # Each agent has a vector of test performances across conditions
    if test_performance_df is None or len(test_performance_df) == 0:
        print("  Skipping: No test performance data available")
        return pd.DataFrame(), figures
    
    # Pivot to get performance matrix: agents × test_conditions
    if 'model_id' in test_performance_df.columns:
        perf_pivot = test_performance_df.pivot_table(
            index='model_id',
            columns=['test_game', 'test_opponent'],
            values='normalized_reward',
            aggfunc='mean'
        )
    elif 'task_id' in test_performance_df.columns:
        perf_pivot = test_performance_df.pivot_table(
            index='task_id',
            columns=['test_game', 'test_opponent'],
            values='normalized_reward',
            aggfunc='mean'
        )
    else:
        print("  Error: No model_id or task_id in test_performance_df")
        return pd.DataFrame(), figures
    
    # Fill NaN with mean (in case some test conditions missing)
    perf_matrix = perf_pivot.fillna(perf_pivot.mean()).values
    
    # Compute pairwise correlations (generalization similarity)
    gen_similarity = np.corrcoef(perf_matrix)
    
    # Extract upper triangle (excluding diagonal) for both matrices
    if cka_matrix is not None and cka_matrix.shape == gen_similarity.shape:
        n = cka_matrix.shape[0]
        triu_indices = np.triu_indices(n, k=1)
        
        cka_flat = cka_matrix[triu_indices]
        gen_flat = gen_similarity[triu_indices]
        
        # Remove NaN
        valid_mask = ~(np.isnan(cka_flat) | np.isnan(gen_flat))
        cka_valid = cka_flat[valid_mask]
        gen_valid = gen_flat[valid_mask]
        
        # Correlation
        corr, p_value = pearsonr(cka_valid, gen_valid)
        
        print(f"\nCorrelation: CKA Similarity ~ Generalization Similarity")
        print(f"  r = {corr:.4f}, p = {p_value:.4e}")
        print(f"  n pairs = {len(cka_valid)}")
        
        summary = pd.DataFrame({
            'metric': ['pearson_r', 'p_value', 'n_pairs'],
            'value': [corr, p_value, len(cka_valid)]
        })
    else:
        print("  Warning: CKA matrix not available or size mismatch")
        summary = pd.DataFrame()
        cka_valid = None
        gen_valid = None
        corr = np.nan
        p_value = np.nan
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: Scatter plot - CKA vs generalization similarity
    ax = axes[0]
    if cka_valid is not None and gen_valid is not None:
        ax.scatter(cka_valid, gen_valid, alpha=0.3, s=20, color='steelblue')
        ax.set_xlabel('Representational Similarity (CKA)', fontsize=11)
        ax.set_ylabel('Generalization Similarity\n(Performance Correlation)', fontsize=11)
        ax.set_title('A. Representation-Generalization Coupling', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Add regression line
        if len(cka_valid) > 1:
            z = np.polyfit(cka_valid, gen_valid, 1)
            p_fit = np.poly1d(z)
            x_line = np.linspace(cka_valid.min(), cka_valid.max(), 100)
            ax.plot(x_line, p_fit(x_line), "r--", alpha=0.8, linewidth=2)
        
        # Add stats text
        ax.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_value:.4e}',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'Data not available',
               ha='center', va='center', transform=ax.transAxes)
    
    # Panel B: Joint heatmap (upper=CKA, lower=generalization)
    ax = axes[1]
    if cka_matrix is not None and cka_matrix.shape == gen_similarity.shape:
        n = cka_matrix.shape[0]
        joint_matrix = np.zeros((n, n))
        
        # Upper triangle: CKA
        triu_mask = np.triu_indices(n, k=0)
        joint_matrix[triu_mask] = cka_matrix[triu_mask]
        
        # Lower triangle: Generalization
        tril_mask = np.tril_indices(n, k=-1)
        joint_matrix[tril_mask] = gen_similarity[tril_mask]
        
        im = ax.imshow(joint_matrix, cmap='RdYlGn', vmin=-0.5, vmax=1.0, aspect='auto')
        ax.set_xlabel('Agent Index', fontsize=11)
        ax.set_ylabel('Agent Index', fontsize=11)
        ax.set_title('B. Joint Similarity\n(Upper=CKA, Lower=Generalization)', 
                    fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Similarity')
    else:
        ax.text(0.5, 0.5, 'Data not available',
               ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    figures['component_4'] = fig
    
    return summary, figures


def component_5_integrated_complexity(
    complexity_df: pd.DataFrame,
    output_dir: Path
) -> Tuple[pd.DataFrame, Dict[str, plt.Figure]]:
    """
    Component 5: Integrated Complexity Analysis
    
    Hypothesis: Complexity metrics form coherent structure
    
    Returns:
        - PCA and correlation results DataFrame
        - Dictionary of figures
    """
    print("\n" + "="*80)
    print("COMPONENT 5: Integrated Complexity Analysis")
    print("="*80)
    
    figures = {}
    
    # Select complexity metrics from all 4 domains
    complexity_cols = [
        # Task
        'payoff_variance', 'coordination_bonus', 'social_dilemma_strength',
        # Opponent (if available)
        # 'predictability', 'adaptation_requirement',
        # Behavior
        'reciprocity_strength', 'policy_entropy', 'mean_cooperation',
        # Representation
        'social_ratio', 'effective_dimensionality', 'weight_l2_norm'
    ]
    
    # Add opponent metrics if available
    if 'predictability' in complexity_df.columns:
        complexity_cols.extend(['predictability', 'adaptation_requirement'])
    
    # Extract available columns
    available_cols = [col for col in complexity_cols if col in complexity_df.columns]
    X = complexity_df[available_cols].copy()
    
    # Handle NaN
    X = X.fillna(X.mean())
    
    # Correlation matrix
    corr_matrix = X.corr()
    
    print("\nCorrelation Matrix (top correlations):")
    # Get upper triangle
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append({
                'var1': corr_matrix.columns[i],
                'var2': corr_matrix.columns[j],
                'correlation': corr_matrix.iloc[i, j]
            })
    corr_df = pd.DataFrame(corr_pairs).sort_values('correlation', key=abs, ascending=False)
    print(corr_df.head(10))
    
    # PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA()
    pca_scores = pca.fit_transform(X_scaled)
    
    print(f"\nPCA Results:")
    print(f"  PC1 variance explained: {pca.explained_variance_ratio_[0]:.3f}")
    print(f"  PC2 variance explained: {pca.explained_variance_ratio_[1]:.3f}")
    print(f"  Cumulative (PC1+PC2):   {pca.explained_variance_ratio_[:2].sum():.3f}")
    
    # PC loadings
    loadings = pd.DataFrame(
        pca.components_[:2].T,
        columns=['PC1', 'PC2'],
        index=available_cols
    )
    print("\nPCA Loadings:")
    print(loadings.round(3))
    
    summary = pd.DataFrame({
        'PC': ['PC1', 'PC2'],
        'variance_explained': pca.explained_variance_ratio_[:2]
    })
    
    # Visualization
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Panel A: Correlation matrix heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
               center=0, vmin=-1, vmax=1, square=True, ax=ax1,
               cbar_kws={'label': 'Correlation'})
    ax1.set_title('A. Complexity Correlation Matrix', fontsize=12, fontweight='bold')
    
    # Panel B: PCA biplot
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Scatter plot colored by training game
    if 'train_game' in complexity_df.columns:
        games = sorted(complexity_df['train_game'].unique())
        for game in games:
            mask = complexity_df['train_game'] == game
            label = GAME_DISPLAY_NAMES.get(game, game)
            ax2.scatter(pca_scores[mask, 0], pca_scores[mask, 1],
                       label=label, alpha=0.6, s=50)
        ax2.legend(title='Game', fontsize=9)
    else:
        ax2.scatter(pca_scores[:, 0], pca_scores[:, 1], alpha=0.6, s=50)
    
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
    ax2.set_title('B. PCA Biplot', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.3)
    ax2.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.3)
    
    # Panel C: Loading plot
    ax3 = fig.add_subplot(gs[1, 0])
    loadings_sorted = loadings.iloc[loadings['PC1'].abs().argsort()]
    loadings_sorted['PC1'].plot(kind='barh', ax=ax3, color='steelblue', alpha=0.7)
    ax3.set_xlabel('PC1 Loading', fontsize=11)
    ax3.set_title('C. PC1 Feature Loadings', fontsize=12, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    ax3.axvline(0, color='black', linewidth=0.8)
    
    # Panel D: Scree plot
    ax4 = fig.add_subplot(gs[1, 1])
    n_components = min(10, len(pca.explained_variance_ratio_))
    ax4.plot(range(1, n_components + 1), 
            pca.explained_variance_ratio_[:n_components],
            marker='o', linewidth=2, markersize=8)
    ax4.plot(range(1, n_components + 1),
            np.cumsum(pca.explained_variance_ratio_[:n_components]),
            marker='s', linewidth=2, markersize=8, linestyle='--', alpha=0.7)
    ax4.set_xlabel('Principal Component', fontsize=11)
    ax4.set_ylabel('Variance Explained', fontsize=11)
    ax4.set_title('D. Scree Plot', fontsize=12, fontweight='bold')
    ax4.legend(['Individual', 'Cumulative'], fontsize=9)
    ax4.grid(alpha=0.3)
    ax4.set_xticks(range(1, n_components + 1))
    
    figures['component_5'] = fig
    
    return summary, figures
