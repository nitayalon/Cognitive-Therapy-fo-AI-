"""
Complete Behavior-Representation Coupling Analysis
==================================================

Addresses three key questions:
1. Do models with same behavior (e.g., defectors) from different games have similar weights?
2. Can we measure how different representations produce identical behavioral outputs?
3. Why weren't Stag-Hunt and Hawk-Dove checkpoints loading? (FIXED: Wrong condition mapping)

Fixed Issues:
- Condition mapping: 0-4=PD, 5-9=HD, 10-14=SH (NOT 0-14 sequential by game/opponent pairs)
- Now loads all 75 models successfully

Analysis Components:
1. Weight-space embeddings (t-SNE, UMAP) with all 75 models
2. Behavioral equivalence class analysis (defectors across games)
3. Within-class vs between-class weight diversity
4. Behavior-representation correlation with proper alignment
5. CKA (Centered Kernel Alignment) for robust representation comparison

Outputs: experiments/representation_coupling_complete/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
from scipy.spatial.distance import euclidean, cosine, cdist
from scipy.stats import spearmanr, pearsonr
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import linear_kernel
import warnings
warnings.filterwarnings('ignore')

# Import UMAP if available
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not installed")

# Import network architecture
import sys
from pathlib import Path
# Add src directory to path (go up from analysis_scripts to experiments, then to root, then to src)
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir / 'src'))
from cognitive_therapy_ai.network import GameLSTM

class CompleteRepresentationAnalyzer:
    def __init__(self, kld_csv_path, checkpoint_base_dir, output_dir):
        self.kld_csv_path = Path(kld_csv_path)
        self.checkpoint_base_dir = Path(checkpoint_base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir = self.output_dir / 'figures'
        self.fig_dir.mkdir(exist_ok=True)
        self.data_dir = self.output_dir / 'data'
        self.data_dir.mkdir(exist_ok=True)
        
        self.games = ['prisoners-dilemma', 'stag-hunt', 'hawk-dove']
        self.game_abbrev = {'prisoners-dilemma': 'PD', 'stag-hunt': 'SH', 'hawk-dove': 'HD'}
        self.opp_ranges = ['very_low', 'low', 'mid', 'high', 'very_high']
        self.opp_labels = ['VL', 'L', 'M', 'H', 'VH']
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load behavioral data (optional - only needed for some analyses)
        if self.kld_csv_path.exists():
            self.df = pd.read_csv(kld_csv_path)
        else:
            self.df = None
            print(f"Note: Behavioral data CSV not found, skipping (only needed for correlation analysis)")
        
        # CORRECTED MAPPING: Discovered from actual checkpoint files
        # Conditions 0-4: PD (5 opponent ranges)
        # Conditions 5-9: HD (5 opponent ranges)
        # Conditions 10-14: SH (5 opponent ranges)
        self.condition_to_game_map = {}
        for i in range(5):
            self.condition_to_game_map[i] = ('prisoners-dilemma', self.opp_ranges[i])
            self.condition_to_game_map[5 + i] = ('hawk-dove', self.opp_ranges[i])
            self.condition_to_game_map[10 + i] = ('stag-hunt', self.opp_ranges[i])
    
    def load_all_model_weights(self):
        """Load all 75 model checkpoints with corrected condition mapping."""
        print("\n" + "="*80)
        print("LOADING ALL MODEL WEIGHTS (CORRECTED MAPPING)")
        print("="*80)
        print("Condition mapping discovered:")
        print("  0-4:   Prisoner's Dilemma")
        print("  5-9:   Hawk-Dove")
        print("  10-14: Stag-Hunt")
        print()
        
        weight_vectors = []
        metadata = []
        
        loaded_count = 0
        failed_count = 0
        
        for condition_id in range(15):
            game, opp = self.condition_to_game_map[condition_id]
            
            for seed in range(5):
                # Checkpoint path
                checkpoint_pattern = f"condition_{condition_id}_seed_{seed}/generalization_matrix_task_*/checkpoints/{game}_final_checkpoint.pth"
                matches = list(self.checkpoint_base_dir.glob(checkpoint_pattern))
                
                if matches:
                    checkpoint_path = matches[0]
                    
                    try:
                        # Load checkpoint
                        checkpoint = torch.load(checkpoint_path, map_location=self.device)
                        hidden_size = checkpoint.get('hidden_size', 128)
                        
                        # Try different num_layers
                        model = None
                        for num_layers in [4, 3, 2, 1]:
                            try:
                                model = GameLSTM(
                                    input_size=5,
                                    hidden_size=hidden_size,
                                    num_actions=2,
                                    num_layers=num_layers
                                ).to(self.device)
                                model.load_state_dict(checkpoint['model_state_dict'])
                                break
                            except RuntimeError:
                                continue
                        
                        if model is None:
                            raise RuntimeError("No matching architecture")
                        
                        # Extract weight vector
                        weight_vector = []
                        for param in model.parameters():
                            weight_vector.append(param.detach().cpu().numpy().flatten())
                        weight_vector = np.concatenate(weight_vector)
                        
                        weight_vectors.append(weight_vector)
                        metadata.append({
                            'game': game,
                            'opponent': opp,
                            'seed': seed,
                            'condition_id': condition_id,
                            'num_params': len(weight_vector)
                        })
                        
                        loaded_count += 1
                        
                    except Exception as e:
                        print(f"  ⚠ Failed to load condition {condition_id} ({game}/{opp}) seed {seed}: {e}")
                        failed_count += 1
                else:
                    print(f"  ⚠ Checkpoint not found: condition {condition_id} ({game}/{opp}) seed {seed}")
                    failed_count += 1
        
        print(f"\n{'='*80}")
        print(f"✓ Successfully loaded: {loaded_count}/75 models")
        print(f"✗ Failed to load: {failed_count}/75 models")
        print(f"{'='*80}")
        
        if loaded_count == 0:
            print("\n❌ CRITICAL: No models loaded!")
            return False
        
        self.weight_vectors = np.array(weight_vectors)
        self.weight_metadata = pd.DataFrame(metadata)
        
        print(f"\n✓ Weight matrix shape: {self.weight_vectors.shape}")
        print(f"✓ Parameters per model: {self.weight_vectors.shape[1]:,}")
        
        # Distribution by game
        game_counts = self.weight_metadata['game'].value_counts()
        print(f"\nModels per game:")
        for game, count in game_counts.items():
            print(f"  {self.game_abbrev[game]}: {count}")
        
        # Save
        np.save(self.data_dir / 'all_weight_vectors.npy', self.weight_vectors)
        self.weight_metadata.to_csv(self.data_dir / 'all_weight_metadata.csv', index=False)
        
        return True
    
    def compute_weight_distances(self):
        """Compute pairwise weight distances."""
        print("\n" + "="*80)
        print("COMPUTING WEIGHT-SPACE DISTANCES")
        print("="*80)
        
        # L2 distance
        self.weight_l2_dist = cdist(self.weight_vectors, self.weight_vectors, metric='euclidean')
        
        # Cosine similarity
        self.weight_cos_sim = 1 - cdist(self.weight_vectors, self.weight_vectors, metric='cosine')
        
        print(f"✓ L2 distance range: [{self.weight_l2_dist[self.weight_l2_dist>0].min():.2f}, {self.weight_l2_dist.max():.2f}]")
        print(f"✓ Cosine similarity range: [{self.weight_cos_sim[self.weight_cos_sim<1].min():.4f}, 1.0000]")
        
        # Within-game vs between-game statistics
        within_game_dists = []
        between_game_dists = []
        
        for i in range(len(self.weight_vectors)):
            for j in range(i+1, len(self.weight_vectors)):
                dist = self.weight_l2_dist[i, j]
                if self.weight_metadata.iloc[i]['game'] == self.weight_metadata.iloc[j]['game']:
                    within_game_dists.append(dist)
                else:
                    between_game_dists.append(dist)
        
        print(f"\nWithin-game weight distance: {np.mean(within_game_dists):.2f} ± {np.std(within_game_dists):.2f}")
        print(f"Between-game weight distance: {np.mean(between_game_dists):.2f} ± {np.std(between_game_dists):.2f}")
        print(f"Ratio (between/within): {np.mean(between_game_dists)/np.mean(within_game_dists):.3f}x")
        
        # Save
        np.save(self.data_dir / 'weight_l2_dist.npy', self.weight_l2_dist)
        np.save(self.data_dir / 'weight_cos_sim.npy', self.weight_cos_sim)
    
    def plot_weight_embeddings(self):
        """Create t-SNE and UMAP embeddings of weight space."""
        print("\n" + "="*80)
        print("CREATING WEIGHT-SPACE EMBEDDINGS")
        print("="*80)
        
        # Standardize
        scaler = StandardScaler()
        weights_scaled = scaler.fit_transform(self.weight_vectors)
        
        # t-SNE
        print("  Computing t-SNE...")
        tsne = TSNE(n_components=2, perplexity=min(30, len(weights_scaled)//4),
                   random_state=42, max_iter=1000, learning_rate='auto', init='pca')
        tsne_embedding = tsne.fit_transform(weights_scaled)
        np.save(self.data_dir / 'tsne_weights.npy', tsne_embedding)
        
        # UMAP
        umap_embedding = None
        if UMAP_AVAILABLE:
            print("  Computing UMAP...")
            umap = UMAP(n_components=2, n_neighbors=min(15, len(weights_scaled)//5),
                       min_dist=0.1, random_state=42)
            umap_embedding = umap.fit_transform(weights_scaled)
            np.save(self.data_dir / 'umap_weights.npy', umap_embedding)
        
        # Plot
        n_plots = 2 if UMAP_AVAILABLE else 1
        fig, axes = plt.subplots(2, n_plots, figsize=(9*n_plots, 14))
        if n_plots == 1:
            axes = axes.reshape(-1, 1)
        
        colors = {'prisoners-dilemma': 'red', 'stag-hunt': 'blue', 'hawk-dove': 'green'}
        
        # t-SNE by game
        ax = axes[0, 0]
        for i, row in self.weight_metadata.iterrows():
            ax.scatter(tsne_embedding[i, 0], tsne_embedding[i, 1],
                      c=colors[row['game']], alpha=0.7, s=100,
                      edgecolors='black', linewidth=0.7)
        for game, color in colors.items():
            ax.scatter([], [], c=color, label=self.game_abbrev[game], s=120, alpha=0.9)
        ax.legend(fontsize=13, loc='best', title='Training Game')
        ax.set_title('t-SNE: Weight Space by Game', fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Dim 1', fontsize=12)
        ax.set_ylabel('t-SNE Dim 2', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # t-SNE by opponent
        ax = axes[1, 0]
        opp_colors = {'very_low': '#440154', 'low': '#31688e', 'mid': '#35b779',
                     'high': '#fde724', 'very_high': '#ff6e3a'}
        for i, row in self.weight_metadata.iterrows():
            ax.scatter(tsne_embedding[i, 0], tsne_embedding[i, 1],
                      c=opp_colors[row['opponent']], alpha=0.7, s=100,
                      edgecolors='black', linewidth=0.7)
        for opp, color in opp_colors.items():
            ax.scatter([], [], c=color, label=self.opp_labels[self.opp_ranges.index(opp)], 
                      s=120, alpha=0.9)
        ax.legend(fontsize=13, loc='best', title='Opponent', ncol=2)
        ax.set_title('t-SNE: Weight Space by Opponent', fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Dim 1', fontsize=12)
        ax.set_ylabel('t-SNE Dim 2', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        if UMAP_AVAILABLE:
            # UMAP by game
            ax = axes[0, 1]
            for i, row in self.weight_metadata.iterrows():
                ax.scatter(umap_embedding[i, 0], umap_embedding[i, 1],
                          c=colors[row['game']], alpha=0.7, s=100,
                          edgecolors='black', linewidth=0.7)
            for game, color in colors.items():
                ax.scatter([], [], c=color, label=self.game_abbrev[game], s=120, alpha=0.9)
            ax.legend(fontsize=13, loc='best', title='Training Game')
            ax.set_title('UMAP: Weight Space by Game', fontsize=14, fontweight='bold')
            ax.set_xlabel('UMAP Dim 1', fontsize=12)
            ax.set_ylabel('UMAP Dim 2', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # UMAP by opponent
            ax = axes[1, 1]
            for i, row in self.weight_metadata.iterrows():
                ax.scatter(umap_embedding[i, 0], umap_embedding[i, 1],
                          c=opp_colors[row['opponent']], alpha=0.7, s=100,
                          edgecolors='black', linewidth=0.7)
            for opp, color in opp_colors.items():
                ax.scatter([], [], c=color, label=self.opp_labels[self.opp_ranges.index(opp)],
                          s=120, alpha=0.9)
            ax.legend(fontsize=13, loc='best', title='Opponent', ncol=2)
            ax.set_title('UMAP: Weight Space by Opponent', fontsize=14, fontweight='bold')
            ax.set_xlabel('UMAP Dim 1', fontsize=12)
            ax.set_ylabel('UMAP Dim 2', fontsize=12)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'weight_space_embeddings_complete.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: weight_space_embeddings_complete.png")
        plt.close()
    
    def analyze_behavioral_equivalence_classes(self):
        """Analyze weight diversity within behavioral equivalence classes."""
        print("\n" + "="*80)
        print("TASK 1: BEHAVIORAL EQUIVALENCE CLASS ANALYSIS")
        print("="*80)
        print("Question: Do behaviorally-equivalent models from different games")
        print("          have similar or different weight representations?")
        print()
        
        # Categorize behaviors
        def categorize_behavior(coop_rate):
            if coop_rate < 0.2:
                return 'Defector'
            elif coop_rate < 0.4:
                return 'Mostly Defect'
            elif coop_rate < 0.6:
                return 'Mixed'
            elif coop_rate < 0.8:
                return 'Mostly Cooperate'
            else:
                return 'Cooperator'
        
        # Match weights with behaviors
        weight_behaviors = []
        for _, row in self.weight_metadata.iterrows():
            matching = self.df[
                (self.df['train_game'] == row['game']) &
                (self.df['train_opponent_range'] == row['opponent'])
            ]
            if len(matching) > 0:
                coop_rate = matching['test_coop_rate'].mean()
                behavior = categorize_behavior(coop_rate)
                weight_behaviors.append(behavior)
            else:
                weight_behaviors.append('Unknown')
        
        self.weight_metadata['behavior_class'] = weight_behaviors
        
        # Analyze defectors specifically
        defector_mask = self.weight_metadata['behavior_class'] == 'Defector'
        defector_indices = np.where(defector_mask)[0]
        
        print(f"Found {len(defector_indices)} defector models:")
        for game in self.games:
            count = ((self.weight_metadata['behavior_class'] == 'Defector') & 
                    (self.weight_metadata['game'] == game)).sum()
            print(f"  {self.game_abbrev[game]}: {count} defectors")
        
        # Within-game vs between-game defector distances
        within_defector_dists = []
        between_defector_dists = []
        
        for i in range(len(defector_indices)):
            for j in range(i+1, len(defector_indices)):
                idx1, idx2 = defector_indices[i], defector_indices[j]
                dist = self.weight_l2_dist[idx1, idx2]
                
                if self.weight_metadata.iloc[idx1]['game'] == self.weight_metadata.iloc[idx2]['game']:
                    within_defector_dists.append(dist)
                else:
                    between_defector_dists.append(dist)
        
        print(f"\n📊 DEFECTOR WEIGHT ANALYSIS:")
        print(f"  Within-game defector distance: {np.mean(within_defector_dists):.2f} ± {np.std(within_defector_dists):.2f}")
        print(f"  Between-game defector distance: {np.mean(between_defector_dists):.2f} ± {np.std(between_defector_dists):.2f}")
        print(f"  Ratio: {np.mean(between_defector_dists)/np.mean(within_defector_dists):.3f}x")
        print()
        print("💡 INTERPRETATION:")
        if np.mean(between_defector_dists) > np.mean(within_defector_dists) * 1.2:
            print("  ✓ DIFFERENT REPRESENTATIONS: Defectors from different games")
            print("    achieve the same behavior via different weight configurations.")
        else:
            print("  ✓ SIMILAR REPRESENTATIONS: Defectors converge to similar")
            print("    weight configurations regardless of training game.")
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Box plot
        ax = axes[0]
        bp = ax.boxplot([within_defector_dists, between_defector_dists],
                        labels=['Within Game', 'Between Games'],
                        patch_artist=True, widths=0.6)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax.set_ylabel('Weight Distance (L2)', fontsize=13, fontweight='bold')
        ax.set_title('Defector Weight Diversity:\nSame Behavior, Different Representations?',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.text(0.5, 0.95,
               f'Within: {np.mean(within_defector_dists):.1f}\\n'
               f'Between: {np.mean(between_defector_dists):.1f}\\n'
               f'Ratio: {np.mean(between_defector_dists)/np.mean(within_defector_dists):.2f}x',
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Histogram overlay
        ax = axes[1]
        ax.hist(within_defector_dists, bins=20, alpha=0.6, color='blue',
               label='Within Game', density=True)
        ax.hist(between_defector_dists, bins=20, alpha=0.6, color='red',
               label='Between Games', density=True)
        ax.set_xlabel('Weight Distance (L2)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Density', fontsize=13, fontweight='bold')
        ax.set_title('Distribution of Defector Weight Distances',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'defector_weight_analysis.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: defector_weight_analysis.png")
        plt.close()
        
        # Save results
        results = {
            'within_mean': np.mean(within_defector_dists),
            'within_std': np.std(within_defector_dists),
            'between_mean': np.mean(between_defector_dists),
            'between_std': np.std(between_defector_dists),
            'ratio': np.mean(between_defector_dists) / np.mean(within_defector_dists)
        }
        pd.DataFrame([results]).to_csv(self.data_dir / 'defector_analysis_results.csv', index=False)
    
    def compute_cka_matrix(self):
        """Compute Centered Kernel Alignment between all model pairs."""
        print("\n" + "="*80)
        print("TASK 2: CKA (CENTERED KERNEL ALIGNMENT) ANALYSIS")
        print("="*80)
        print("Question: How similar are representations using geometry-aware metrics?")
        print()
        
        def cka_rbf(X, Y, sigma=None):
            """Compute CKA with RBF kernel."""
            if sigma is None:
                sigma = 1.0
            
            # RBF kernel
            def rbf_kernel(X):
                dists = cdist(X, X, metric='sqeuclidean')
                return np.exp(-dists / (2 * sigma**2))
            
            K = rbf_kernel(X)
            L = rbf_kernel(Y)
            
            # Center kernels
            n = K.shape[0]
            H = np.eye(n) - np.ones((n, n)) / n
            K_c = H @ K @ H
            L_c = H @ L @ H
            
            # CKA
            numerator = np.trace(K_c @ L_c)
            denominator = np.sqrt(np.trace(K_c @ K_c) * np.trace(L_c @ L_c))
            
            return numerator / denominator if denominator > 0 else 0
        
        n_models = len(self.weight_vectors)
        cka_matrix = np.zeros((n_models, n_models))
        
        print("  Computing CKA matrix (this may take a few minutes)...")
        
        # Use a subset of weights for computational efficiency
        subset_size = min(5000, self.weight_vectors.shape[1])
        indices = np.random.choice(self.weight_vectors.shape[1], subset_size, replace=False)
        weights_subset = self.weight_vectors[:, indices]
        
        for i in range(n_models):
            cka_matrix[i, i] = 1.0
            for j in range(i+1, n_models):
                cka_val = cka_rbf(weights_subset[i:i+1].T, weights_subset[j:j+1].T)
                cka_matrix[i, j] = cka_val
                cka_matrix[j, i] = cka_val
            
            if (i+1) % 15 == 0:
                print(f"    Progress: {i+1}/{n_models} models")
        
        self.cka_matrix = cka_matrix
        np.save(self.data_dir / 'cka_matrix.npy', cka_matrix)
        
        print(f"✓ CKA range: [{cka_matrix[cka_matrix<1].min():.4f}, 1.0000]")
        
        # Within vs between game CKA
        within_cka = []
        between_cka = []
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                cka = cka_matrix[i, j]
                if self.weight_metadata.iloc[i]['game'] == self.weight_metadata.iloc[j]['game']:
                    within_cka.append(cka)
                else:
                    between_cka.append(cka)
        
        print(f"\n📊 CKA SIMILARITY:")
        print(f"  Within-game: {np.mean(within_cka):.4f} ± {np.std(within_cka):.4f}")
        print(f"  Between-game: {np.mean(between_cka):.4f} ± {np.std(between_cka):.4f}")
        print(f"  Difference: {np.mean(within_cka) - np.mean(between_cka):.4f}")
        
        # Visualize
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(cka_matrix, cmap='viridis', vmin=0, vmax=1, aspect='auto')
        ax.set_title('CKA Similarity Matrix\n(Geometry-Aware Representation Comparison)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Model Index', fontsize=12)
        ax.set_ylabel('Model Index', fontsize=12)
        plt.colorbar(im, ax=ax, label='CKA Similarity')
        
        # Add game boundaries
        for i in [25, 50]:
            ax.axhline(i-0.5, color='white', linewidth=2, linestyle='--')
            ax.axvline(i-0.5, color='white', linewidth=2, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'cka_similarity_matrix.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: cka_similarity_matrix.png")
        plt.close()
    
    def behavior_representation_correlation(self):
        """Correlate behavioral and representational distance matrices."""
        print("\n" + "="*80)
        print("TASK 3: BEHAVIOR-REPRESENTATION COUPLING")
        print("="*80)
        print("Question: Are models with similar behaviors also similar in weight space?")
        print()
        
        # Create behavioral vectors for loaded models
        behavioral_profiles = []
        for _, row in self.weight_metadata.iterrows():
            model_data = self.df[
                (self.df['train_game'] == row['game']) &
                (self.df['train_opponent_range'] == row['opponent'])
            ]
            
            if len(model_data) > 0:
                # Use test cooperation rates
                profile = []
                for test_game in self.games:
                    for test_opp in self.opp_ranges:
                        test_vals = model_data[
                            (model_data['test_game'] == test_game) &
                            (model_data['test_opponent_range'] == test_opp)
                        ]['test_coop_rate']
                        if len(test_vals) > 0:
                            profile.append(test_vals.values[0])
                        else:
                            profile.append(np.nan)
                behavioral_profiles.append(profile)
            else:
                behavioral_profiles.append([np.nan] * 15)
        
        behavioral_profiles = np.array(behavioral_profiles)
        
        # Compute behavioral distance
        behavior_dist = cdist(behavioral_profiles, behavioral_profiles, metric='euclidean')
        
        # Correlate
        behavior_flat = behavior_dist[np.triu_indices_from(behavior_dist, k=1)]
        weight_flat = self.weight_l2_dist[np.triu_indices_from(self.weight_l2_dist, k=1)]
        
        # Remove NaN pairs
        valid_mask = ~(np.isnan(behavior_flat) | np.isnan(weight_flat))
        behavior_flat = behavior_flat[valid_mask]
        weight_flat = weight_flat[valid_mask]
        
        spearman_r, spearman_p = spearmanr(behavior_flat, weight_flat)
        pearson_r, pearson_p = pearsonr(behavior_flat, weight_flat)
        
        print(f"📊 CORRELATION RESULTS:")
        print(f"  Spearman r = {spearman_r:.4f}, p = {spearman_p:.4e}")
        print(f"  Pearson  r = {pearson_r:.4f}, p = {pearson_p:.4e}")
        print()
        print("💡 INTERPRETATION:")
        if spearman_r > 0.5:
            print("  ✓ STRONG COUPLING: Similar behaviors → Similar weights")
        elif spearman_r > 0.3:
            print("  ✓ MODERATE COUPLING: Partial behavior-weight relationship")
        elif spearman_r > 0:
            print("  ✓ WEAK COUPLING: Many-to-one mapping (different weights → same behavior)")
        else:
            print("  ✓ NO COUPLING: Behavior and weights are largely independent")
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # Scatter plot
        ax = axes[0]
        ax.scatter(behavior_flat, weight_flat, alpha=0.3, s=10, c='steelblue')
        ax.set_xlabel('Behavioral Distance (L2)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Weight Distance (L2)', fontsize=13, fontweight='bold')
        ax.set_title(f'Behavior-Representation Coupling\\nSpearman r = {spearman_r:.3f}, p = {spearman_p:.2e}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(behavior_flat, weight_flat, 1)
        p = np.poly1d(z)
        ax.plot(sorted(behavior_flat), p(sorted(behavior_flat)), 
               "r--", linewidth=2, alpha=0.7, label=f'Linear fit')
        ax.legend(fontsize=11)
        
        # Hexbin density
        ax = axes[1]
        hb = ax.hexbin(behavior_flat, weight_flat, gridsize=30, cmap='Blues', mincnt=1)
        ax.set_xlabel('Behavioral Distance (L2)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Weight Distance (L2)', fontsize=13, fontweight='bold')
        ax.set_title('Density Plot: Behavior vs Weight Distance',
                    fontsize=14, fontweight='bold')
        plt.colorbar(hb, ax=ax, label='Count')
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'behavior_weight_correlation.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: behavior_weight_correlation.png")
        plt.close()
        
        # Save correlation data
        results = {
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p
        }
        pd.DataFrame([results]).to_csv(self.data_dir / 'correlation_results.csv', index=False)
    
    def load_and_aggregate_weights_by_policy(self):
        """Memory-efficient: Load weights and immediately aggregate by (game, opponent, policy)."""
        print("\n" + "="*80)
        print("MEMORY-EFFICIENT POLICY-BASED WEIGHT AGGREGATION")
        print("="*80)
        print("Strategy: Load → Classify → Aggregate → Discard individual models")
        print()
        
        # Dictionary to accumulate weights: {(game, opponent, policy): [weight_vectors]}
        weight_accumulator = {}
        policy_info = []
        
        loaded_count = 0
        failed_count = 0
        
        for condition_id in range(15):
            game, opponent = self.condition_to_game_map[condition_id]
            
            for seed in range(5):
                # Find checkpoint and metrics
                checkpoint_pattern = f"condition_{condition_id}_seed_{seed}/generalization_matrix_task_*/checkpoints/{game}_final_checkpoint.pth"
                metrics_pattern = f"condition_{condition_id}_seed_{seed}/generalization_matrix_task_*/results/training_task_*_metrics.csv"
                
                checkpoint_matches = list(self.checkpoint_base_dir.glob(checkpoint_pattern))
                metrics_matches = list(self.checkpoint_base_dir.glob(metrics_pattern))
                
                if len(checkpoint_matches) == 1 and len(metrics_matches) == 1:
                    try:
                        # Load metrics to classify policy
                        metrics_df = pd.read_csv(metrics_matches[0])
                        final_coop_rate = metrics_df['epoch_average_cooperation_rate'].iloc[-1]
                        
                        # Classify policy
                        if final_coop_rate >= 0.7:
                            policy_type = 'cooperative'
                        elif final_coop_rate <= 0.3:
                            policy_type = 'defective'
                        else:
                            policy_type = 'mixed'
                        
                        # Load checkpoint
                        checkpoint = torch.load(checkpoint_matches[0], map_location=self.device)
                        hidden_size = checkpoint.get('hidden_size', 128)
                        
                        # Load model
                        model = None
                        for num_layers in [4, 3, 2, 1]:
                            try:
                                model = GameLSTM(
                                    input_size=5,
                                    hidden_size=hidden_size,
                                    num_actions=2,
                                    num_layers=num_layers
                                ).to(self.device)
                                model.load_state_dict(checkpoint['model_state_dict'])
                                break
                            except RuntimeError:
                                continue
                        
                        if model is None:
                            raise RuntimeError("No matching architecture")
                        
                        # Extract weight vector
                        weight_vector = []
                        for param in model.parameters():
                            weight_vector.append(param.detach().cpu().numpy().flatten())
                        weight_vector = np.concatenate(weight_vector)
                        
                        # Accumulate by (game, opponent, policy)
                        key = (game, opponent, policy_type)
                        if key not in weight_accumulator:
                            weight_accumulator[key] = []
                        weight_accumulator[key].append(weight_vector)
                        
                        # Track policy info
                        policy_info.append({
                            'game': game,
                            'opponent': opponent,
                            'seed': seed,
                            'policy_type': policy_type,
                            'final_coop_rate': final_coop_rate
                        })
                        
                        loaded_count += 1
                        
                        # Clear memory
                        del model, checkpoint, weight_vector
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        
                    except Exception as e:
                        failed_count += 1
                        if failed_count <= 5:  # Only show first 5 errors
                            print(f"  ⚠ Failed: {game}-{opponent}-seed{seed}: {e}")
                else:
                    failed_count += 1
        
        print(f"\n{'='*80}")
        print(f"✓ Loaded: {loaded_count}/75 models")
        print(f"✗ Failed: {failed_count}/75 models")
        print(f"{'='*80}")
        
        if loaded_count == 0:
            print("\n❌ No models loaded!")
            return False
        
        # Save policy classifications
        self.policy_classifications = pd.DataFrame(policy_info)
        policy_counts = self.policy_classifications['policy_type'].value_counts()
        print(f"\nPolicy distribution:")
        for policy, count in policy_counts.items():
            print(f"  {policy}: {count} models")
        self.policy_classifications.to_csv(self.data_dir / 'policy_classifications.csv', index=False)
        
        # Aggregate weights
        print(f"\nAggregating {len(weight_accumulator)} unique (game, opponent, policy) conditions...")
        
        aggregated_weights = []
        aggregated_metadata = []
        
        for (game, opponent, policy), weights_list in weight_accumulator.items():
            # Average across seeds
            mean_weight = np.mean(weights_list, axis=0)
            aggregated_weights.append(mean_weight)
            
            # Metadata
            matching_policies = self.policy_classifications[
                (self.policy_classifications['game'] == game) &
                (self.policy_classifications['opponent'] == opponent) &
                (self.policy_classifications['policy_type'] == policy)
            ]
            
            aggregated_metadata.append({
                'game': game,
                'opponent': opponent,
                'policy_type': policy,
                'n_seeds': len(weights_list),
                'mean_coop_rate': matching_policies['final_coop_rate'].mean(),
                'std_coop_rate': matching_policies['final_coop_rate'].std()
            })
        
        self.aggregated_weights = np.array(aggregated_weights)
        self.aggregated_metadata = pd.DataFrame(aggregated_metadata)
        
        print(f"✓ Created {len(self.aggregated_weights)} seed-averaged representations")
        print(f"  Shape: {self.aggregated_weights.shape}")
        print(f"  Parameters per model: {self.aggregated_weights.shape[1]:,}")
        print(f"\nBreakdown:")
        print(f"  Games: {self.aggregated_metadata['game'].nunique()}")
        print(f"  Opponent ranges: {self.aggregated_metadata['opponent'].nunique()}")
        print(f"  Policy types: {self.aggregated_metadata['policy_type'].nunique()}")
        
        # Save
        np.save(self.data_dir / 'aggregated_weights.npy', self.aggregated_weights)
        self.aggregated_metadata.to_csv(self.data_dir / 'aggregated_metadata.csv', index=False)
        print(f"✓ Saved: aggregated_weights.npy, aggregated_metadata.csv")
        
        return True
    
    # NOTE: This method removed - aggregation now done during loading for memory efficiency
    
    def analyze_policy_conditioned_representations(self):
        """Analyze representation structure across task x opponent x policy dimensions."""
        print("\n" + "="*80)
        print("POLICY-CONDITIONED REPRESENTATION ANALYSIS")
        print("="*80)
        
        # Compute pairwise distances
        print("\nComputing pairwise distances...")
        agg_l2_dist = cdist(self.aggregated_weights, self.aggregated_weights, metric='euclidean')
        agg_cos_sim = 1 - cdist(self.aggregated_weights, self.aggregated_weights, metric='cosine')
        
        n = len(self.aggregated_weights)
        print(f"  ✓ Computed {n}x{n} distance matrix")
        
        # Analyze clustering by each dimension
        print("\n" + "-"*80)
        print("CLUSTERING ANALYSIS")
        print("-"*80)
        
        def compute_clustering_ratio(metadata, distance_matrix, dimension):
            """Compute within vs between distance ratio for a dimension."""
            within_dists = []
            between_dists = []
            
            for i in range(n):
                for j in range(i+1, n):
                    dist = distance_matrix[i, j]
                    if metadata.iloc[i][dimension] == metadata.iloc[j][dimension]:
                        within_dists.append(dist)
                    else:
                        between_dists.append(dist)
            
            within_mean = np.mean(within_dists)
            within_std = np.std(within_dists)
            between_mean = np.mean(between_dists)
            between_std = np.std(between_dists)
            ratio = between_mean / within_mean if within_mean > 0 else 0
            
            return {
                'within_mean': within_mean,
                'within_std': within_std,
                'within_n': len(within_dists),
                'between_mean': between_mean,
                'between_std': between_std,
                'between_n': len(between_dists),
                'ratio': ratio
            }
        
        # Analyze each dimension
        clustering_results = {}
        for dim, dim_name in [('game', 'Task'), ('opponent', 'Opponent Type'), ('policy_type', 'Policy')]:
            result = compute_clustering_ratio(self.aggregated_metadata, agg_l2_dist, dim)
            clustering_results[dim] = result
            
            print(f"\n{dim_name.upper()}-DRIVEN CLUSTERING:")
            print(f"  Within-{dim_name.lower()} distance:  {result['within_mean']:.2f} ± {result['within_std']:.2f}")
            print(f"  Between-{dim_name.lower()} distance: {result['between_mean']:.2f} ± {result['between_std']:.2f}")
            print(f"  Ratio (between/within): {result['ratio']:.3f}x")
            
            if result['ratio'] > 1.5:
                print(f"  → STRONG {dim_name} clustering")
            elif result['ratio'] > 1.1:
                print(f"  → MODERATE {dim_name} clustering")
            else:
                print(f"  → WEAK/NO {dim_name} clustering")
        
        # Save clustering results
        clustering_df = pd.DataFrame(clustering_results).T
        clustering_df.to_csv(self.data_dir / 'policy_conditioned_clustering.csv')
        print(f"\n✓ Saved: policy_conditioned_clustering.csv")
        
        # Visualization 1: t-SNE colored by each dimension
        print("\n" + "-"*80)
        print("CREATING VISUALIZATIONS")
        print("-"*80)
        print("  Computing t-SNE embedding...")
        
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n-1))
        weights_embedded = tsne.fit_transform(self.aggregated_weights)
        
        print(f"  ✓ t-SNE embedding computed: {weights_embedded.shape}")
        print(f"  ✓ Total models to plot: {n}")
        
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        
        # Define DISCRETE color palettes
        game_color_map = {
            'prisoners-dilemma': '#e41a1c',  # Red
            'hawk-dove': '#377eb8',          # Blue
            'stag-hunt': '#4daf4a'           # Green
        }
        
        opponent_color_map = {
            'very_low': '#440154',   # Dark purple
            'low': '#3b528b',        # Blue
            'mid': '#21918c',        # Teal
            'high': '#5ec962',       # Green
            'very_high': '#fde724'   # Yellow
        }
        
        policy_color_map = {
            'cooperative': '#2ca02c',   # Green
            'defective': '#d62728',     # Red
            'mixed': '#ff7f0e'          # Orange
        }
        
        # Panel 1: Color by GAME (Task)
        ax = axes[0]
        for game, color in game_color_map.items():
            mask = self.aggregated_metadata['game'] == game
            indices = np.where(mask)[0]
            if len(indices) > 0:
                ax.scatter(weights_embedded[indices, 0], weights_embedded[indices, 1],
                          c=color, s=200, alpha=0.85, edgecolors='black', linewidths=1.5,
                          label=self.game_abbrev.get(game, game), marker='o')
        
        ax.set_title(f'Task Clustering (ratio={clustering_results["game"]["ratio"]:.2f}x)\n{n} models plotted',
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
        ax.legend(fontsize=11, loc='best', framealpha=0.9, edgecolor='black')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f8f8')
        
        # Panel 2: Color by OPPONENT
        ax = axes[1]
        opponent_labels = {'very_low': 'Very Low', 'low': 'Low', 'mid': 'Mid', 
                          'high': 'High', 'very_high': 'Very High'}
        for opponent, color in opponent_color_map.items():
            mask = self.aggregated_metadata['opponent'] == opponent
            indices = np.where(mask)[0]
            if len(indices) > 0:
                ax.scatter(weights_embedded[indices, 0], weights_embedded[indices, 1],
                          c=color, s=200, alpha=0.85, edgecolors='black', linewidths=1.5,
                          label=opponent_labels[opponent], marker='s')
        
        ax.set_title(f'Opponent Clustering (ratio={clustering_results["opponent"]["ratio"]:.2f}x)\n{n} models plotted',
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
        ax.legend(fontsize=10, loc='best', framealpha=0.9, edgecolor='black', ncol=1)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f8f8')
        
        # Panel 3: Color by POLICY
        ax = axes[2]
        policy_labels = {'cooperative': 'Cooperative (≥0.7)', 'defective': 'Defective (≤0.3)', 
                        'mixed': 'Mixed (0.3-0.7)'}
        policy_markers = {'cooperative': '^', 'defective': 'v', 'mixed': 'D'}
        for policy, color in policy_color_map.items():
            mask = self.aggregated_metadata['policy_type'] == policy
            indices = np.where(mask)[0]
            if len(indices) > 0:
                ax.scatter(weights_embedded[indices, 0], weights_embedded[indices, 1],
                          c=color, s=200, alpha=0.85, edgecolors='black', linewidths=1.5,
                          label=policy_labels[policy], marker=policy_markers[policy])
        
        ax.set_title(f'Policy Clustering (ratio={clustering_results["policy_type"]["ratio"]:.2f}x)\n{n} models plotted',
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
        ax.legend(fontsize=10, loc='best', framealpha=0.9, edgecolor='black')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f8f8')
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'policy_conditioned_tsne.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: policy_conditioned_tsne.png")
        plt.close()
        
        # Verify all models were plotted
        print(f"\n  📊 Verification:")
        print(f"    Total aggregated models: {n}")
        for game in game_color_map.keys():
            count = (self.aggregated_metadata['game'] == game).sum()
            print(f"    {self.game_abbrev.get(game, game)}: {count} models")
        for policy in policy_color_map.keys():
            count = (self.aggregated_metadata['policy_type'] == policy).sum()
            print(f"    {policy.capitalize()}: {count} models")
        
        # Visualization 2: Distance matrices heatmap with hierarchical labels
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Create hierarchical labels: Game | Opponent | Policy
        tick_labels = []
        for _, row in self.aggregated_metadata.iterrows():
            game_abbr = self.game_abbrev.get(row['game'], row['game'])
            opp_abbr = {'very_low': 'VL', 'low': 'L', 'mid': 'M', 'high': 'H', 'very_high': 'VH'}[row['opponent']]
            policy_abbr = {'cooperative': 'C', 'defective': 'D', 'mixed': 'M'}[row['policy_type']]
            tick_labels.append(f"{game_abbr}|{opp_abbr}|{policy_abbr}")
        
        # Calculate boundaries for visual separation by game
        game_boundaries = []
        prev_game = None
        for i, row in self.aggregated_metadata.iterrows():
            if prev_game is not None and row['game'] != prev_game:
                game_boundaries.append(i - 0.5)
            prev_game = row['game']
        
        # L2 distance
        ax = axes[0]
        im = ax.imshow(agg_l2_dist, cmap='viridis', aspect='auto')
        ax.set_title('L2 Distance Matrix (Policy-Aggregated)\nLabels: Game|Opponent|Policy', 
                    fontsize=13, fontweight='bold')
        
        # Set tick labels
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=8, ha='center')
        ax.set_yticklabels(tick_labels, fontsize=8)
        
        # Add game boundary lines
        for boundary in game_boundaries:
            ax.axhline(boundary, color='white', linewidth=2, linestyle='-', alpha=0.8)
            ax.axvline(boundary, color='white', linewidth=2, linestyle='-', alpha=0.8)
        
        plt.colorbar(im, ax=ax, label='L2 Distance', fraction=0.046, pad=0.04)
        
        # Cosine similarity
        ax = axes[1]
        im = ax.imshow(agg_cos_sim, cmap='plasma', aspect='auto', vmin=0, vmax=1)
        ax.set_title('Cosine Similarity (Policy-Aggregated)\nLabels: Game|Opponent|Policy',
                    fontsize=13, fontweight='bold')
        
        # Set tick labels
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=8, ha='center')
        ax.set_yticklabels(tick_labels, fontsize=8)
        
        # Add game boundary lines
        for boundary in game_boundaries:
            ax.axhline(boundary, color='white', linewidth=2, linestyle='-', alpha=0.8)
            ax.axvline(boundary, color='white', linewidth=2, linestyle='-', alpha=0.8)
        
        plt.colorbar(im, ax=ax, label='Cosine Similarity', fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'policy_conditioned_distance_matrices.png', dpi=200, bbox_inches='tight')
        print(f"  ✓ Saved: policy_conditioned_distance_matrices.png")
        plt.close()
        
        # Visualization 3: Clustering ratios comparison
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        dimensions = ['Task', 'Opponent Type', 'Policy']
        ratios = [clustering_results['game']['ratio'],
                 clustering_results['opponent']['ratio'],
                 clustering_results['policy_type']['ratio']]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        bars = ax.bar(dimensions, ratios, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='No clustering (ratio=1.0)')
        ax.axhline(y=1.5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Strong clustering threshold')
        
        ax.set_ylabel('Clustering Ratio (Between/Within)', fontsize=13, fontweight='bold')
        ax.set_title('Representation Clustering by Dimension\n(Seed-Averaged Weights)',
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(ratios) * 1.2)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, ratio in zip(bars, ratios):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{ratio:.2f}x', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'policy_conditioned_clustering_ratios.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: policy_conditioned_clustering_ratios.png")
        plt.close()
        
        print("\n✅ POLICY-CONDITIONED ANALYSIS COMPLETE")
    
    def analyze_pca(self):
        """Analyze principal components of weight space."""
        print("\n" + "="*80)
        print("PCA ANALYSIS")
        print("="*80)
        
        # Standardize weights
        scaler = StandardScaler()
        weights_scaled = scaler.fit_transform(self.aggregated_weights)
        
        # Perform PCA
        print("\n  Computing PCA...")
        pca = PCA()
        weights_pca = pca.fit_transform(weights_scaled)
        
        # Save PC embeddings
        np.save(self.data_dir / 'pca_weights.npy', weights_pca)
        
        # Variance explained
        var_explained = pca.explained_variance_ratio_
        cumsum_var = np.cumsum(var_explained)
        
        print(f"\n  PC1 variance explained: {var_explained[0]:.4f} ({var_explained[0]*100:.2f}%)")
        print(f"  PC2 variance explained: {var_explained[1]:.4f} ({var_explained[1]*100:.2f}%)")
        print(f"  PC1+PC2 total: {cumsum_var[1]:.4f} ({cumsum_var[1]*100:.2f}%)")
        print(f"  Components for 90% variance: {np.argmax(cumsum_var >= 0.90) + 1}")
        print(f"  Components for 95% variance: {np.argmax(cumsum_var >= 0.95) + 1}")
        
        # Visualization 1: Variance explained plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scree plot
        ax = axes[0]
        n_components_show = min(20, len(var_explained))
        ax.bar(range(1, n_components_show+1), var_explained[:n_components_show], alpha=0.7, color='steelblue')
        ax.plot(range(1, n_components_show+1), var_explained[:n_components_show], 'ro-', linewidth=2, markersize=6)
        ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
        ax.set_ylabel('Variance Explained', fontsize=12, fontweight='bold')
        ax.set_title(f'Scree Plot (Top {n_components_show} Components)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(range(1, n_components_show+1, 2))
        
        # Cumulative variance
        ax = axes[1]
        ax.plot(range(1, len(cumsum_var)+1), cumsum_var, 'b-', linewidth=2.5)
        ax.axhline(y=0.90, color='orange', linestyle='--', linewidth=2, label='90% variance', alpha=0.7)
        ax.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95% variance', alpha=0.7)
        ax.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Variance Explained', fontsize=12, fontweight='bold')
        ax.set_title('Cumulative Variance Explained', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        ax.set_xlim(0, min(50, len(cumsum_var)))
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'pca_variance_explained.png', dpi=150, bbox_inches='tight')
        print("\n  ✓ Saved: pca_variance_explained.png")
        plt.close()
        
        # Visualization 2: PCA embeddings colored by dimensions
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        
        game_color_map = {'prisoners-dilemma': '#e41a1c', 'hawk-dove': '#377eb8', 'stag-hunt': '#4daf4a'}
        opponent_color_map = {'very_low': '#440154', 'low': '#3b528b', 'mid': '#21918c', 
                             'high': '#5ec962', 'very_high': '#fde724'}
        policy_color_map = {'cooperative': '#2ca02c', 'defective': '#d62728', 'mixed': '#ff7f0e'}
        
        # PC1 vs PC2: By game
        ax = axes[0]
        for game, color in game_color_map.items():
            mask = self.aggregated_metadata['game'] == game
            indices = np.where(mask)[0]
            if len(indices) > 0:
                ax.scatter(weights_pca[indices, 0], weights_pca[indices, 1],
                          c=color, s=200, alpha=0.85, edgecolors='black', linewidths=1.5,
                          label=self.game_abbrev.get(game, game), marker='o')
        ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}% var)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}% var)', fontsize=11, fontweight='bold')
        ax.set_title('PCA: Task Clustering', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best', framealpha=0.9, edgecolor='black')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f8f8')
        
        # PC1 vs PC2: By opponent
        ax = axes[1]
        opponent_labels = {'very_low': 'Very Low', 'low': 'Low', 'mid': 'Mid', 
                          'high': 'High', 'very_high': 'Very High'}
        for opponent, color in opponent_color_map.items():
            mask = self.aggregated_metadata['opponent'] == opponent
            indices = np.where(mask)[0]
            if len(indices) > 0:
                ax.scatter(weights_pca[indices, 0], weights_pca[indices, 1],
                          c=color, s=200, alpha=0.85, edgecolors='black', linewidths=1.5,
                          label=opponent_labels[opponent], marker='s')
        ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}% var)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}% var)', fontsize=11, fontweight='bold')
        ax.set_title('PCA: Opponent Clustering', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best', framealpha=0.9, edgecolor='black', ncol=1)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f8f8')
        
        # PC1 vs PC2: By policy
        ax = axes[2]
        policy_labels = {'cooperative': 'Cooperative (≥0.7)', 'defective': 'Defective (≤0.3)', 
                        'mixed': 'Mixed (0.3-0.7)'}
        policy_markers = {'cooperative': '^', 'defective': 'v', 'mixed': 'D'}
        for policy, color in policy_color_map.items():
            mask = self.aggregated_metadata['policy_type'] == policy
            indices = np.where(mask)[0]
            if len(indices) > 0:
                ax.scatter(weights_pca[indices, 0], weights_pca[indices, 1],
                          c=color, s=200, alpha=0.85, edgecolors='black', linewidths=1.5,
                          label=policy_labels[policy], marker=policy_markers[policy])
        ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}% var)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}% var)', fontsize=11, fontweight='bold')
        ax.set_title('PCA: Policy Clustering', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best', framealpha=0.9, edgecolor='black')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f8f8')
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'pca_embeddings.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: pca_embeddings.png")
        plt.close()
        
        # Save variance summary
        pca_summary = pd.DataFrame({
            'PC': range(1, min(21, len(var_explained)+1)),
            'variance_explained': var_explained[:20],
            'cumulative_variance': cumsum_var[:20]
        })
        pca_summary.to_csv(self.data_dir / 'pca_variance_summary.csv', index=False)
        print("  ✓ Saved: pca_variance_summary.csv")
        
        print("\n✅ PCA ANALYSIS COMPLETE")
    
    def analyze_weight_magnitudes(self):
        """Analyze distribution of weight magnitudes per game."""
        print("\n" + "="*80)
        print("WEIGHT MAGNITUDE ANALYSIS")
        print("="*80)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        game_colors = {'prisoners-dilemma': '#e41a1c', 'hawk-dove': '#377eb8', 'stag-hunt': '#4daf4a'}
        
        # Plot 1: Distribution of all weights per game (KDE)
        ax = axes[0, 0]
        for game in self.games:
            mask = self.aggregated_metadata['game'] == game
            indices = np.where(mask)[0]
            if len(indices) > 0:
                # Flatten all weights for this game
                game_weights = self.aggregated_weights[indices].flatten()
                ax.hist(game_weights, bins=100, alpha=0.5, density=True, 
                       label=self.game_abbrev[game], color=game_colors[game])
        ax.set_xlabel('Weight Value', fontsize=11, fontweight='bold')
        ax.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax.set_title('Weight Value Distribution by Game', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Mean absolute weight per game
        ax = axes[0, 1]
        game_mean_abs = []
        game_labels = []
        for game in self.games:
            mask = self.aggregated_metadata['game'] == game
            indices = np.where(mask)[0]
            if len(indices) > 0:
                mean_abs = np.mean(np.abs(self.aggregated_weights[indices]))
                game_mean_abs.append(mean_abs)
                game_labels.append(self.game_abbrev[game])
                print(f"\n  {self.game_abbrev[game]}: Mean |weight| = {mean_abs:.4f}")
        
        bars = ax.bar(game_labels, game_mean_abs, 
                     color=[game_colors[g] for g in self.games if g in game_colors],
                     alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Mean Absolute Weight', fontsize=11, fontweight='bold')
        ax.set_title('Average Weight Magnitude by Game', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, game_mean_abs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Plot 3: Weight variance per game
        ax = axes[1, 0]
        game_std = []
        for game in self.games:
            mask = self.aggregated_metadata['game'] == game
            indices = np.where(mask)[0]
            if len(indices) > 0:
                std = np.std(self.aggregated_weights[indices])
                game_std.append(std)
                print(f"  {self.game_abbrev[game]}: Weight std = {std:.4f}")
        
        bars = ax.bar(game_labels, game_std,
                     color=[game_colors[g] for g in self.games if g in game_colors],
                     alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Weight Standard Deviation', fontsize=11, fontweight='bold')
        ax.set_title('Weight Variability by Game', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, game_std):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Plot 4: L2 norm per game
        ax = axes[1, 1]
        game_l2_norms = []
        for game in self.games:
            mask = self.aggregated_metadata['game'] == game
            indices = np.where(mask)[0]
            if len(indices) > 0:
                # Average L2 norm across models
                l2_norms = [np.linalg.norm(self.aggregated_weights[i]) for i in indices]
                mean_l2 = np.mean(l2_norms)
                game_l2_norms.append(mean_l2)
                print(f"  {self.game_abbrev[game]}: Mean L2 norm = {mean_l2:.2f}")
        
        bars = ax.bar(game_labels, game_l2_norms,
                     color=[game_colors[g] for g in self.games if g in game_colors],
                     alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Mean L2 Norm', fontsize=11, fontweight='bold')
        ax.set_title('Weight Vector Magnitude by Game', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, game_l2_norms):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'weight_magnitude_analysis.png', dpi=150, bbox_inches='tight')
        print("\n  ✓ Saved: weight_magnitude_analysis.png")
        plt.close()
        
        # Save statistics
        magnitude_stats = pd.DataFrame({
            'game': [self.game_abbrev[g] for g in self.games],
            'mean_abs_weight': game_mean_abs,
            'weight_std': game_std,
            'mean_l2_norm': game_l2_norms
        })
        magnitude_stats.to_csv(self.data_dir / 'weight_magnitude_stats.csv', index=False)
        print("  ✓ Saved: weight_magnitude_stats.csv")
        
        print("\n✅ WEIGHT MAGNITUDE ANALYSIS COMPLETE")
    
    def analyze_layer_contributions(self):
        """Analyze contribution of different network components to clustering."""
        print("\n" + "="*80)
        print("LAYER-WISE CONTRIBUTION ANALYSIS")
        print("="*80)
        print("\nExtracting layer-specific weights...")
        
        # Reload one model to get layer structure
        sample_condition = list(self.condition_to_game_map.keys())[0]
        checkpoint_pattern = f"condition_{sample_condition}_seed_0/generalization_matrix_task_*/checkpoints/*.pth"
        matches = list(self.checkpoint_base_dir.glob(checkpoint_pattern))
        
        if not matches:
            print("  ⚠ Could not find sample model for layer analysis")
            return
        
        # Load sample model to get architecture
        checkpoint = torch.load(matches[0], map_location=self.device)
        hidden_size = checkpoint.get('hidden_size', 128)
        
        # Try different num_layers
        model = None
        for num_layers in [4, 3, 2, 1]:
            try:
                model = GameLSTM(input_size=5, hidden_size=hidden_size, num_actions=2, num_layers=num_layers).to(self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                break
            except RuntimeError:
                continue
        
        if model is None:
            print("  ⚠ Could not load model for layer analysis")
            return
        
        # Get parameter names and sizes
        layer_info = []
        cumulative_params = 0
        for name, param in model.named_parameters():
            num_params = param.numel()
            layer_info.append({
                'name': name,
                'shape': tuple(param.shape),
                'num_params': num_params,
                'start_idx': cumulative_params,
                'end_idx': cumulative_params + num_params
            })
            cumulative_params += num_params
            print(f"  {name}: {tuple(param.shape)} -> {num_params:,} params")
        
        print(f"\n  Total parameters: {cumulative_params:,}")
        
        # Group into components
        lstm_indices = []
        policy_indices = []
        opponent_indices = []
        value_indices = []
        
        for info in layer_info:
            name = info['name']
            start, end = info['start_idx'], info['end_idx']
            
            if 'lstm' in name.lower():
                lstm_indices.extend(range(start, end))
            elif 'policy' in name.lower():
                policy_indices.extend(range(start, end))
            elif 'opponent' in name.lower():
                opponent_indices.extend(range(start, end))
            elif 'value' in name.lower() or 'critic' in name.lower():
                value_indices.extend(range(start, end))
        
        print(f"\nComponent breakdown:")
        print(f"  LSTM: {len(lstm_indices):,} params")
        print(f"  Policy head: {len(policy_indices):,} params")
        print(f"  Opponent head: {len(opponent_indices):,} params")
        print(f"  Value head: {len(value_indices):,} params")
        
        # Compute clustering metrics for each component
        def compute_component_clustering(indices, component_name):
            if len(indices) == 0:
                return None
            
            # Extract component weights
            component_weights = self.aggregated_weights[:, indices]
            
            # Compute distances
            component_dists = cdist(component_weights, component_weights, metric='euclidean')
            
            # Within vs between game distances
            within_dists = []
            between_dists = []
            n = len(component_weights)
            
            for i in range(n):
                for j in range(i+1, n):
                    dist = component_dists[i, j]
                    if self.aggregated_metadata.iloc[i]['game'] == self.aggregated_metadata.iloc[j]['game']:
                        within_dists.append(dist)
                    else:
                        between_dists.append(dist)
            
            if not within_dists or not between_dists:
                return None
            
            within_mean = np.mean(within_dists)
            between_mean = np.mean(between_dists)
            ratio = between_mean / within_mean if within_mean > 0 else 0
            
            return {
                'component': component_name,
                'num_params': len(indices),
                'within_mean': within_mean,
                'between_mean': between_mean,
                'ratio': ratio
            }
        
        print("\nComputing clustering ratios per component...")
        results = []
        
        for indices, name in [(lstm_indices, 'LSTM'),
                              (policy_indices, 'Policy Head'),
                              (opponent_indices, 'Opponent Head'),
                              (value_indices, 'Value Head')]:
            result = compute_component_clustering(indices, name)
            if result:
                results.append(result)
                print(f"\n  {name}:")
                print(f"    Within-game: {result['within_mean']:.2f}")
                print(f"    Between-game: {result['between_mean']:.2f}")
                print(f"    Ratio: {result['ratio']:.3f}x")
        
        # Visualization
        if results:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            components = [r['component'] for r in results]
            ratios = [r['ratio'] for r in results]
            params = [r['num_params'] for r in results]
            
            # Plot 1: Clustering ratio by component
            ax = axes[0]
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            bars = ax.bar(components, ratios, color=colors[:len(components)], 
                         alpha=0.7, edgecolor='black', linewidth=2)
            ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='No clustering')
            ax.set_ylabel('Clustering Ratio (Between/Within)', fontsize=11, fontweight='bold')
            ax.set_title('Task Clustering by Network Component', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, max(ratios) * 1.2)
            
            for bar, ratio in zip(bars, ratios):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{ratio:.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Plot 2: Parameter count by component
            ax = axes[1]
            bars = ax.bar(components, params, color=colors[:len(components)],
                         alpha=0.7, edgecolor='black', linewidth=2)
            ax.set_ylabel('Number of Parameters', fontsize=11, fontweight='bold')
            ax.set_title('Parameter Distribution Across Components', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, param_count in zip(bars, params):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{param_count:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(self.fig_dir / 'layer_contribution_analysis.png', dpi=150, bbox_inches='tight')
            print("\n  ✓ Saved: layer_contribution_analysis.png")
            plt.close()
            
            # Save results
            results_df = pd.DataFrame(results)
            results_df.to_csv(self.data_dir / 'layer_contribution_stats.csv', index=False)
            print("  ✓ Saved: layer_contribution_stats.csv")
        
        print("\n✅ LAYER-WISE ANALYSIS COMPLETE")
    
    def run_policy_conditioned_analysis_only(self):
        """Run ONLY the memory-efficient policy-conditioned analysis."""
        print("\n" + "="*80)
        print("POLICY-CONDITIONED REPRESENTATION ANALYSIS (MEMORY-EFFICIENT)")
        print("="*80)
        print(f"Checkpoints: {self.checkpoint_base_dir}")
        print(f"Output: {self.output_dir}")
        print()
        print("Analysis dimensions:")
        print("  - Task (3 games): Prisoner's Dilemma, Hawk-Dove, Stag Hunt")
        print("  - Opponent Type (5 ranges): Very Low, Low, Mid, High, Very High")
        print("  - Policy (3 types): Cooperative (>0.7), Defective (<0.3), Mixed")
        print()
        
        # Memory-efficient load and aggregate
        success = self.load_and_aggregate_weights_by_policy()
        if not success:
            print("\n❌ Analysis aborted: Could not load model weights")
            return
        
        # Analyze 3D structure: task x opponent x policy
        self.analyze_policy_conditioned_representations()
        
        # Additional analyses
        self.analyze_pca()
        self.analyze_weight_magnitudes()
        self.analyze_layer_contributions()
        
        print("\n" + "="*80)
        print("✅ POLICY-CONDITIONED ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nAll outputs saved to: {self.output_dir}")
        print(f"  Figures: {self.fig_dir}")
        print(f"  Data: {self.data_dir}")
        print(f"\nKey files:")
        print(f"  - policy_classifications.csv: Individual model policies")
        print(f"  - aggregated_weights.npy: Seed-averaged weight vectors")
        print(f"  - aggregated_metadata.csv: Metadata for aggregated models")
        print(f"  - policy_conditioned_clustering.csv: Clustering statistics")
        print(f"  - policy_conditioned_tsne.png: 3-panel t-SNE visualization")
        print(f"  - policy_conditioned_clustering_ratios.png: Bar chart comparison")
    
    def run_complete_analysis(self):
        """Run all three analysis tasks (MEMORY INTENSIVE)."""
        print("\n" + "="*80)
        print("COMPLETE BEHAVIOR-REPRESENTATION COUPLING ANALYSIS")
        print("="*80)
        print(f"Input: {self.kld_csv_path}")
        print(f"Checkpoints: {self.checkpoint_base_dir}")
        print(f"Output: {self.output_dir}")
        print()
        
        # Load weights
        success = self.load_all_model_weights()
        if not success:
            print("\n❌ Analysis aborted: Could not load model weights")
            return
        
        # Compute distances
        self.compute_weight_distances()
        
        # Embeddings
        self.plot_weight_embeddings()
        
        # Task 1: Behavioral equivalence
        self.analyze_behavioral_equivalence_classes()
        
        # Task 2: CKA
        self.compute_cka_matrix()
        
        # Task 3: Correlation
        self.behavior_representation_correlation()
        
        print("\n" + "="*80)
        print("✅ COMPLETE ANALYSIS FINISHED")
        print("="*80)
        print(f"\nAll outputs saved to: {self.output_dir}")
        print(f"Figures: {self.fig_dir}")
        print(f"Data: {self.data_dir}")

def main():
    # Use correct checkpoint directory (WITH training subdirectory)
    checkpoint_dir = "experiments/generalization_matrix_train_888509/training"
    output_dir = "experiments/representation_coupling_complete"
    
    # Dummy KLD path (not used in policy-conditioned analysis)
    kld_csv = "experiments/generalization_matrix_train_888509/analysis_results/dummy.csv"
    
    analyzer = CompleteRepresentationAnalyzer(kld_csv, checkpoint_dir, output_dir)
    
    # Run memory-efficient policy-conditioned analysis only
    print("\n⚡ Running MEMORY-EFFICIENT policy-conditioned analysis")
    print("   (skipping full 75-model analysis to save memory)\n")
    analyzer.run_policy_conditioned_analysis_only()

if __name__ == "__main__":
    main()
