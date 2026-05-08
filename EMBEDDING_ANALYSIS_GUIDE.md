# Embedding Architecture Analysis Guide
**Isolating Social vs Environmental Effects**

**Date:** May 8, 2026  
**Architecture:** Separate Embeddings (commit 3a10f2d)

---

## Overview

The separate embedding architecture allows us to **quantitatively measure** which inputs the network actually uses for decision-making, addressing the key research question: **Do networks learn social strategies (reciprocity) or task-only strategies (Nash equilibrium)?**

## Architecture Design

### Input Decomposition (9 elements → 6 pathways)

```
INPUT VECTOR (9 elements)
│
├─ ENVIRONMENTAL INPUTS (game structure, 5 elements)
│  ├─ payoff_matrix[0:4]    → payoff_matrix_embed    (4→21 dims)
│  └─ round_number[4:5]     → round_number_embed     (1→21 dims)
│
└─ SOCIAL INPUTS (trial history, 4 elements)
   ├─ opponent_action[5:6]  → opponent_action_embed  (1→21 dims)
   ├─ agent_action[6:7]     → agent_action_embed     (1→21 dims)
   ├─ agent_reward[7:8]     → agent_reward_embed     (1→21 dims)
   └─ opponent_reward[8:9]  → opponent_reward_embed  (1→21 dims)

CONCATENATE (6×21 = 126 dims) → LSTM → policy/value/ToM heads
```

### Embedding Layer Structure
Each embedding pathway:
```python
nn.Sequential(
    nn.Linear(input_dim, embed_dim),  # Projects input to embed_dim
    nn.ReLU(),                         # Nonlinearity
    nn.LayerNorm(embed_dim)            # Stabilization
)
```

Where `embed_dim = hidden_size // 6` (e.g., 128 → 21 per pathway)

---

## Why This Enables Social/Environmental Isolation

### 1. **Separate Gradients**
Each embedding has its own learnable weights. During backpropagation:
- If the network **uses** an input → gradients flow, weights update
- If the network **ignores** an input → minimal gradients, weights stay near initialization

### 2. **Deactivation Detection**
If a network learns Nash equilibrium (task-only):
- **Social embedding weights** should remain small/random (near initialization)
- **Environmental embedding weights** should be large/structured

If a network learns reciprocity (social strategy):
- **Both** social and environmental embeddings should show significant learned structure

### 3. **Interpretable Dimensions**
The concatenated embedding space has clear structure:
```
[env_21_dims | env_21_dims | soc_21_dims | soc_21_dims | soc_21_dims | soc_21_dims]
 ↑ payoff     ↑ round       ↑ opp_action  ↑ agent_act  ↑ agent_rew  ↑ opp_reward
```

We can measure contribution of each block to LSTM input.

---

## Analysis Methods

### Method 1: **Weight Magnitude Analysis**

**Hypothesis:**  
- Social learners: Large weights in social embeddings
- Nash learners: Small weights in social embeddings

**Implementation:**
```python
import torch
import numpy as np

def analyze_embedding_weights(model):
    """Compute weight magnitudes for each embedding pathway."""
    
    results = {}
    
    # ENVIRONMENTAL EMBEDDINGS
    payoff_weights = model.payoff_matrix_embed[0].weight  # (embed_dim, 4)
    round_weights = model.round_number_embed[0].weight    # (embed_dim, 1)
    
    results['env_payoff_magnitude'] = torch.norm(payoff_weights).item()
    results['env_round_magnitude'] = torch.norm(round_weights).item()
    results['env_total'] = results['env_payoff_magnitude'] + results['env_round_magnitude']
    
    # SOCIAL EMBEDDINGS
    opp_action_weights = model.opponent_action_embed[0].weight  # (embed_dim, 1)
    agent_action_weights = model.agent_action_embed[0].weight  # (embed_dim, 1)
    agent_reward_weights = model.agent_reward_embed[0].weight  # (embed_dim, 1)
    opp_reward_weights = model.opponent_reward_embed[0].weight  # (embed_dim, 1)
    
    results['soc_opp_action'] = torch.norm(opp_action_weights).item()
    results['soc_agent_action'] = torch.norm(agent_action_weights).item()
    results['soc_agent_reward'] = torch.norm(agent_reward_weights).item()
    results['soc_opp_reward'] = torch.norm(opp_reward_weights).item()
    results['soc_total'] = sum([results[k] for k in results if k.startswith('soc_')])
    
    # RATIO: How much does network rely on social vs environmental?
    results['social_ratio'] = results['soc_total'] / (results['env_total'] + results['soc_total'])
    
    return results

# Usage
checkpoint = torch.load('model.pth')
model.load_state_dict(checkpoint)
weights = analyze_embedding_weights(model)

print(f"Environmental weight norm: {weights['env_total']:.3f}")
print(f"Social weight norm: {weights['soc_total']:.3f}")
print(f"Social ratio: {weights['social_ratio']:.3%}")
```

**Interpretation:**
- `social_ratio < 0.3` → Nash equilibrium learner (ignores social inputs)
- `social_ratio > 0.5` → Social learner (uses history for reciprocity)

---

### Method 2: **Gradient Flow Analysis**

**Hypothesis:**  
Track gradient magnitudes during training to see which embeddings receive updates.

**Implementation:**
```python
def track_embedding_gradients(model, dataloader, device='cuda'):
    """Track gradient flow through each embedding during one epoch."""
    
    model.train()
    gradient_stats = {
        'env_payoff': [],
        'env_round': [],
        'soc_opp_action': [],
        'soc_agent_action': [],
        'soc_agent_reward': [],
        'soc_opp_reward': []
    }
    
    for batch in dataloader:
        states, actions, rewards = batch
        states = states.to(device)
        
        # Forward pass
        policy_logits, _, values, _ = model(states)
        loss = compute_loss(policy_logits, actions, values, rewards)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Capture gradients
        gradient_stats['env_payoff'].append(
            model.payoff_matrix_embed[0].weight.grad.norm().item()
        )
        gradient_stats['env_round'].append(
            model.round_number_embed[0].weight.grad.norm().item()
        )
        gradient_stats['soc_opp_action'].append(
            model.opponent_action_embed[0].weight.grad.norm().item()
        )
        gradient_stats['soc_agent_action'].append(
            model.agent_action_embed[0].weight.grad.norm().item()
        )
        gradient_stats['soc_agent_reward'].append(
            model.agent_reward_embed[0].weight.grad.norm().item()
        )
        gradient_stats['soc_opp_reward'].append(
            model.opponent_reward_embed[0].weight.grad.norm().item()
        )
    
    # Average gradients
    return {k: np.mean(v) for k, v in gradient_stats.items()}

# Usage
grad_flow = track_embedding_gradients(model, train_loader)
for name, grad_norm in grad_flow.items():
    print(f"{name}: {grad_norm:.6f}")
```

**Interpretation:**
- Large gradients → embedding is being actively used
- Small gradients → embedding is ignored

---

### Method 3: **Ablation Analysis**

**Hypothesis:**  
Remove embeddings and measure policy change. Important embeddings will cause large policy shifts.

**Implementation:**
```python
def ablation_analysis(model, test_states, device='cuda'):
    """
    Measure policy change when zeroing out each embedding.
    
    Returns policy KL-divergence for each ablation.
    """
    model.eval()
    test_states = test_states.to(device)
    
    # Get baseline policy
    with torch.no_grad():
        baseline_logits, _, _, _ = model(test_states)
        baseline_policy = torch.softmax(baseline_logits, dim=-1)
    
    results = {}
    
    # Test each embedding ablation
    embeddings_to_test = [
        ('payoff_matrix_embed', 'env'),
        ('round_number_embed', 'env'),
        ('opponent_action_embed', 'social'),
        ('agent_action_embed', 'social'),
        ('agent_reward_embed', 'social'),
        ('opponent_reward_embed', 'social')
    ]
    
    for embed_name, category in embeddings_to_test:
        # Create modified forward pass that zeros this embedding
        original_forward = model.forward
        
        def ablated_forward(x, hidden=None):
            # ... [copy forward pass code, but zero out the target embedding]
            # Example: if embed_name == 'opponent_action_embed':
            #     opp_action_embed = torch.zeros_like(opp_action_embed)
            pass
        
        # Monkey patch temporarily
        model.forward = ablated_forward
        
        with torch.no_grad():
            ablated_logits, _, _, _ = model(test_states)
            ablated_policy = torch.softmax(ablated_logits, dim=-1)
        
        # Compute KL divergence
        kl_div = torch.nn.functional.kl_div(
            ablated_policy.log(), 
            baseline_policy, 
            reduction='batchmean'
        ).item()
        
        results[embed_name] = kl_div
        
        # Restore original forward
        model.forward = original_forward
    
    return results

# Usage
kl_divergences = ablation_analysis(model, test_states)
for embed, kl in sorted(kl_divergences.items(), key=lambda x: -x[1]):
    print(f"{embed}: KL = {kl:.6f}")
```

**Interpretation:**
- Large KL → embedding is critical for policy
- Small KL → embedding is not used

---

### Method 4: **Embedding Activation Analysis**

**Hypothesis:**  
Measure variance of embedding outputs. Used embeddings have high variance.

**Implementation:**
```python
def embedding_activation_variance(model, dataloader, device='cuda'):
    """
    Measure variance of embedding activations across dataset.
    High variance → embedding is used for discrimination.
    Low variance → embedding output is nearly constant (unused).
    """
    model.eval()
    
    activations = {
        'env_payoff': [],
        'env_round': [],
        'soc_opp_action': [],
        'soc_agent_action': [],
        'soc_agent_reward': [],
        'soc_opp_reward': []
    }
    
    with torch.no_grad():
        for states, _, _ in dataloader:
            states = states.to(device)
            batch_size = states.size(0)
            seq_len = states.size(1)
            
            # Extract components
            payoff = states[..., 0:4].reshape(-1, 4)
            round_num = states[..., 4:5].reshape(-1, 1)
            opp_act = states[..., 5:6].reshape(-1, 1)
            agent_act = states[..., 6:7].reshape(-1, 1)
            agent_rew = states[..., 7:8].reshape(-1, 1)
            opp_rew = states[..., 8:9].reshape(-1, 1)
            
            # Pass through embeddings
            activations['env_payoff'].append(model.payoff_matrix_embed(payoff))
            activations['env_round'].append(model.round_number_embed(round_num))
            activations['soc_opp_action'].append(model.opponent_action_embed(opp_act))
            activations['soc_agent_action'].append(model.agent_action_embed(agent_act))
            activations['soc_agent_reward'].append(model.agent_reward_embed(agent_rew))
            activations['soc_opp_reward'].append(model.opponent_reward_embed(opp_rew))
    
    # Compute variance for each embedding
    variances = {}
    for name, acts in activations.items():
        all_acts = torch.cat(acts, dim=0)  # (N, embed_dim)
        variances[name] = all_acts.var(dim=0).mean().item()  # Average variance across dims
    
    return variances

# Usage
variances = embedding_activation_variance(model, test_loader)
print("\nEmbedding Activation Variances:")
for name, var in variances.items():
    category = 'ENV' if 'env' in name else 'SOC'
    print(f"  [{category}] {name}: {var:.6f}")

# Compute social vs environmental ratio
env_var = sum(v for k, v in variances.items() if 'env' in k)
soc_var = sum(v for k, v in variances.items() if 'soc' in k)
print(f"\nSocial variance ratio: {soc_var / (env_var + soc_var):.3%}")
```

**Interpretation:**
- High variance → embedding discriminates between different inputs
- Low variance → embedding produces similar outputs regardless of input (unused)

---

### Method 5: **Embedding Visualization (t-SNE/PCA)**

**Hypothesis:**  
Visualize embedding space to see if social embeddings cluster by opponent behavior.

**Implementation:**
```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def visualize_social_embeddings(model, dataloader, device='cuda'):
    """
    Extract and visualize social embedding activations.
    Color by opponent cooperation rate to see if embeddings detect social patterns.
    """
    model.eval()
    
    opp_action_embeds = []
    cooperation_rates = []
    
    with torch.no_grad():
        for states, _, metadata in dataloader:
            states = states.to(device)
            
            # Extract opponent action component
            opp_actions = states[..., 5:6].reshape(-1, 1)
            
            # Get embeddings
            embeds = model.opponent_action_embed(opp_actions)
            opp_action_embeds.append(embeds.cpu())
            
            # Track cooperation rate for coloring
            coop_rate = (opp_actions == 0).float().mean().item()
            cooperation_rates.extend([coop_rate] * embeds.size(0))
    
    # Concatenate all embeddings
    all_embeds = torch.cat(opp_action_embeds, dim=0).numpy()
    
    # Dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    embeds_2d = tsne.fit_transform(all_embeds)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embeds_2d[:, 0], 
        embeds_2d[:, 1],
        c=cooperation_rates,
        cmap='RdYlGn',
        alpha=0.6,
        s=10
    )
    plt.colorbar(scatter, label='Opponent Cooperation Rate')
    plt.title('Opponent Action Embedding Space (t-SNE)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    plt.savefig('opponent_action_embedding_tsne.png', dpi=300)
    plt.close()

# Usage
visualize_social_embeddings(model, test_loader)
```

**Interpretation:**
- Clear clustering by cooperation rate → embedding learns social patterns
- Random scatter → embedding doesn't discriminate social information

---

## Comprehensive Analysis Script

Here's a complete script that runs all analyses:

```python
import torch
import numpy as np
from pathlib import Path
import json

def comprehensive_embedding_analysis(
    checkpoint_path,
    model_class,
    test_dataloader,
    output_dir='embedding_analysis',
    device='cuda'
):
    """
    Run all 5 analysis methods and generate report.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    checkpoint = torch.load(checkpoint_path)
    model = model_class(input_size=9, hidden_size=128, num_layers=2)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    results = {}
    
    # 1. Weight Magnitude Analysis
    print("Running weight magnitude analysis...")
    results['weight_magnitudes'] = analyze_embedding_weights(model)
    
    # 2. Gradient Flow (requires training mode)
    print("Running gradient flow analysis...")
    results['gradient_flow'] = track_embedding_gradients(model, test_dataloader, device)
    
    # 3. Ablation Analysis
    print("Running ablation analysis...")
    test_states = next(iter(test_dataloader))[0][:100]  # Sample 100 states
    results['ablation_kl'] = ablation_analysis(model, test_states, device)
    
    # 4. Activation Variance
    print("Running activation variance analysis...")
    results['activation_variance'] = embedding_activation_variance(model, test_dataloader, device)
    
    # 5. Visualization
    print("Generating visualizations...")
    visualize_social_embeddings(model, test_dataloader, device)
    
    # Compute summary statistics
    results['summary'] = {
        'social_weight_ratio': results['weight_magnitudes']['social_ratio'],
        'social_variance_ratio': sum(v for k, v in results['activation_variance'].items() if 'soc' in k) / 
                                 sum(results['activation_variance'].values()),
        'most_important_embedding': max(results['ablation_kl'].items(), key=lambda x: x[1])[0],
        'least_important_embedding': min(results['ablation_kl'].items(), key=lambda x: x[1])[0]
    }
    
    # Save results
    with open(output_dir / 'embedding_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate human-readable report
    with open(output_dir / 'REPORT.md', 'w') as f:
        f.write("# Embedding Analysis Report\n\n")
        f.write(f"**Model:** {checkpoint_path}\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- **Social Weight Ratio:** {results['summary']['social_weight_ratio']:.2%}\n")
        f.write(f"- **Social Variance Ratio:** {results['summary']['social_variance_ratio']:.2%}\n")
        f.write(f"- **Most Important:** {results['summary']['most_important_embedding']}\n")
        f.write(f"- **Least Important:** {results['summary']['least_important_embedding']}\n\n")
        
        f.write("## Interpretation\n\n")
        if results['summary']['social_weight_ratio'] < 0.3:
            f.write("**Conclusion:** This network appears to be a **NASH EQUILIBRIUM learner**.\n")
            f.write("Social embeddings have low weight magnitude, suggesting the network ignores ")
            f.write("opponent actions and rewards in favor of pure game-theoretic reasoning.\n")
        else:
            f.write("**Conclusion:** This network appears to be a **SOCIAL learner**.\n")
            f.write("Social embeddings have significant weight magnitude, suggesting the network ")
            f.write("uses opponent history for reciprocal strategies.\n")
    
    print(f"\nAnalysis complete. Results saved to {output_dir}/")
    return results

# Usage
results = comprehensive_embedding_analysis(
    checkpoint_path='experiments/generalization_matrix_train_<JOB_ID>/training/model_0/checkpoint_final.pth',
    model_class=GameLSTM,
    test_dataloader=test_loader,
    output_dir='experiments/embedding_analysis_model_0'
)
```

---

## Expected Findings

### Nash Equilibrium Networks (predicted based on reciprocity analysis)
```
Weight Magnitudes:
  Environmental: 15.3 (82%)
  Social: 3.2 (18%)

Ablation KL-Divergence:
  payoff_matrix: 0.85 (critical)
  round_number: 0.12 (minor)
  opponent_action: 0.03 (negligible)
  agent_action: 0.02 (negligible)
  agent_reward: 0.01 (negligible)
  opponent_reward: 0.01 (negligible)
```

### Social Networks (hypothetical)
```
Weight Magnitudes:
  Environmental: 12.1 (45%)
  Social: 14.8 (55%)

Ablation KL-Divergence:
  payoff_matrix: 0.72 (critical)
  opponent_action: 0.51 (important)
  agent_reward: 0.38 (important)
  round_number: 0.15 (minor)
  agent_action: 0.12 (minor)
  opponent_reward: 0.08 (minor)
```

---

## Next Steps

1. **Create analysis script** in `experiments/analysis_scripts/analyze_embeddings.py`
2. **Run on trained models** from current/future experiments
3. **Compare across conditions:**
   - Task-opponent vs task-only paradigms
   - Different games (PD, HD, SH)
   - Different training opponent types
4. **Correlate with behavior:**
   - Do high social ratios predict reciprocity?
   - Do low social ratios predict Nash convergence?

This analysis will definitively answer: **Does the network learn to ignore social inputs despite having access to them?**
