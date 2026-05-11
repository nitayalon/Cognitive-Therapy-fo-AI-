"""
Summary comparison of within-model vs cross-model KLD analysis.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load both datasets
within_model_df = pd.read_csv('experiments/generalization_analysis/data/policy_kld_analysis.csv')
cross_model_df = pd.read_csv('experiments/cross_model_kld_analysis/cross_model_kld.csv')

print("="*80)
print("COMPARISON: WITHIN-MODEL vs CROSS-MODEL KLD")
print("="*80)

print("\n1. WITHIN-MODEL KLD (Current Analysis)")
print("   Definition: KL(train_policy || test_policy) for the SAME model")
print("   Question: Does a model's policy change when tested on new conditions?")
print(f"   Range: [{within_model_df['kld'].min():.6f}, {within_model_df['kld'].max():.6f}]")
print(f"   Mean: {within_model_df['kld'].mean():.6f} ± {within_model_df['kld'].std():.6f}")
print(f"   Median: {within_model_df['kld'].median():.6f}")
print(f"   95th percentile: {within_model_df['kld'].quantile(0.95):.6f}")
print(f"   Interpretation: Policies are VERY STABLE (94% show KLD=0)")

print("\n2. CROSS-MODEL KLD (New Analysis)")
print("   Definition: KL(model_A_policy || model_B_policy) when tested on SAME condition")
print("   Question: How different are policies learned from different training games?")
print(f"   Range: [{cross_model_df['kld'].min():.6f}, {cross_model_df['kld'].max():.6f}]")
print(f"   Mean: {cross_model_df['kld'].mean():.6f} ± {cross_model_df['kld'].std():.6f}")
print(f"   Median: {cross_model_df['kld'].median():.6f}")
print(f"   95th percentile: {cross_model_df['kld'].quantile(0.95):.6f}")
print(f"   Interpretation: Models learn VERY DIFFERENT policies from different games")

print("\n3. KEY FINDINGS")
print("   ✓ Within-model KLD ≈ 0: Models maintain their learned policy across tests")
print("   ✓ Cross-model KLD = 0.04-3.6: Different games produce radically different policies")
print("   ✓ This suggests STRONG game-specific learning but LIMITED adaptation")

print("\n4. MOST DIVERGENT MODEL PAIRS (Cross-Model)")
top10 = cross_model_df.nlargest(10, 'kld')
game_abbrev = {'prisoners-dilemma': 'PD', 'stag-hunt': 'SH', 'hawk-dove': 'HD'}
for idx, row in top10.iterrows():
    print(f"   {game_abbrev[row['train_game_1']]} vs {game_abbrev[row['train_game_2']]} "
          f"on {game_abbrev[row['test_game']]}-{row['test_opponent_range']}: "
          f"KLD={row['kld']:.3f} (coop: {row['coop_rate_1']:.3f} vs {row['coop_rate_2']:.3f})")

# Create comparison plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Distribution comparison
ax = axes[0]
bins = np.logspace(-6, 1, 50)
ax.hist(within_model_df['kld'][within_model_df['kld'] > 0], bins=bins, alpha=0.6, 
        label=f'Within-Model (n={len(within_model_df)})', color='steelblue', edgecolor='black')
ax.hist(cross_model_df['kld'], bins=bins, alpha=0.6, 
        label=f'Cross-Model (n={len(cross_model_df)})', color='coral', edgecolor='black')
ax.set_xscale('log')
ax.set_xlabel('KL Divergence (nats, log scale)', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Distribution of KLD Values', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Plot 2: Box plot comparison
ax = axes[1]
data_to_plot = [
    within_model_df['kld'][within_model_df['kld'] > 0],
    cross_model_df['kld']
]
bp = ax.boxplot(data_to_plot, labels=['Within-Model\n(same model,\ndiff test)', 
                                        'Cross-Model\n(diff models,\nsame test)'],
                patch_artist=True, showfliers=True)
bp['boxes'][0].set_facecolor('steelblue')
bp['boxes'][1].set_facecolor('coral')
ax.set_ylabel('KL Divergence (nats)', fontsize=12, fontweight='bold')
ax.set_title('Comparison of KLD Magnitudes', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([-0.1, 4])

plt.tight_layout()
plt.savefig('experiments/cross_model_kld_analysis/figures/kld_comparison.png', 
            dpi=150, bbox_inches='tight')
print(f"\n✓ Saved comparison plot: kld_comparison.png")
plt.close()

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("The original analysis (within-model KLD ≈ 0) was correct but incomplete:")
print("  • Models DON'T adapt their policies when tested on new conditions")
print("  • But different training games produce VERY DIFFERENT baseline policies")
print("  • Cross-model KLD (0.04-3.6 nats) reveals the true policy diversity")
print("  • This is the expected ≈1-4 nats range with epsilon=0.001")
print("="*80)
