# Whole Population Analysis - Task 3: Network Size Reduction Recommendations

## Current Network Architecture

Based on [network.py](../src/cognitive_therapy_ai/network.py):

### Current Specifications
- **LSTM Hidden Size**: 128
- **LSTM Layers**: 2 (default config)
- **Embedding Dimension**: 128 / 6 = 21 per component
- **Total Embedding Size**: 21 × 6 = 126
- **Policy Head**: 128 → 64 → 2
- **Opponent Policy Head**: 128 → 64 → 32 → 2
- **Value Head**: 128 → 64 → 1

### Total Parameter Count (Approximate)
- LSTM: ~67,000 parameters (128 hidden × 2 layers)
- Embeddings: ~3,000 parameters
- Heads: ~10,000 parameters
- **Total: ~80,000 parameters**

## Evidence for Over-Capacity

### From Component 1 Analysis Results:
- **Effective Dimensionality**: Only 1.8 - 3.8 dimensions (out of 128!)
- **Prisoners Dilemma**: 2.0 effective dimensions (1.6% of capacity)
- **Hawk-Dove**: 2.8 effective dimensions (2.2% of capacity)
- **Stag Hunt**: 2.5 effective dimensions (2.0% of capacity)

**Conclusion**: The network is using less than 3% of its representational capacity. This massive over-parametrization may:
1. Make it harder to find significant learning effects
2. Reduce sample efficiency
3. Create unnecessarily complex representations
4. Mask important architectural differences between conditions

## Recommended Size Reductions

### Option 1: Moderate Reduction (Recommended for Initial Tests)
```python
network_config = {
    "hidden_size": 64,      # 128 → 64 (2x reduction)
    "num_layers": 2,        # Keep 2 layers
    "dropout": 0.1,         # Keep same
    "embed_dim": 11         # 64 / 6 ≈ 11
}
```
- Total parameters: ~20,000 (4x reduction)
- Effective capacity: Still 32x the observed usage
- Lower risk of under-fitting

### Option 2: Aggressive Reduction (Recommended)
```python
network_config = {
    "hidden_size": 32,      # 128 → 32 (4x reduction)
    "num_layers": 1,        # 2 → 1 (simpler)
    "dropout": 0.05,        # Reduce dropout
    "embed_dim": 5          # 32 / 6 ≈ 5
}
```
- Total parameters: ~3,500 (23x reduction!)
- Effective capacity: Still 10x the observed usage
- May reveal more significant architectural effects

### Option 3: Minimal Network (Exploratory)
```python
network_config = {
    "hidden_size": 16,      # 128 → 16 (8x reduction)
    "num_layers": 1,        # Single layer
    "dropout": 0.0,         # No dropout
    "embed_dim": 3          # 16 / 6 ≈ 3
}
```
- Total parameters: ~1,000 (80x reduction!)
- Effective capacity: ~5x the observed usage
- Closer to theoretical minimum
- Risk: May be too constrained for some tasks

## Expected Benefits

### 1. **Stronger Learning Signals**
- Smaller networks must use capacity more efficiently
- Architectural differences become more pronounced
- Easier to detect statistical effects

### 2. **Faster Training**
- 4-23x fewer parameters to update
- Faster convergence
- Lower computational cost

### 3. **Better Interpretability**
- Less redundancy in representations
- Clearer embedding specialization patterns
- More meaningful dimensionality analysis

### 4. **Improved Sample Efficiency**
- Fewer parameters → less overfitting risk
- Better generalization with fewer training samples
- Could reduce seed requirements

## Implementation Plan

### Step 1: Create New Config Files
Create variants of `whole_population_config.json`:
- `whole_population_config_h64.json` (Option 1)
- `whole_population_config_h32.json` (Option 2)
- `whole_population_config_h16.json` (Option 3)

### Step 2: Pilot Study (Recommended)
Run small-scale comparison on single game (e.g., Prisoners Dilemma):
- 3 seeds × 4 network sizes = 12 quick training runs
- Compare: convergence speed, final performance, effective dimensionality
- Select best configuration for full experiment

### Step 3: Full Re-run (if pilot successful)
- Use selected network size
- Run full 60-task array (20 seeds × 3 games)
- Compare with original 128-hidden results

## Risk Assessment

| Network Size | Risk Level | Notes |
|--------------|-----------|-------|
| 64 hidden    | **Low**   | Safe reduction, unlikely to hurt performance |
| 32 hidden    | **Low-Medium** | Still well above effective dimensionality |
| 16 hidden    | **Medium** | May approach capacity limits in some conditions |

## Code Changes Required

Only need to modify config files - no code changes needed:

```json
{
  "network_config": {
    "hidden_size": 32,
    "num_layers": 1,
    "dropout": 0.05,
    "input_size": 9,
    "agent_type": "vanilla"
  }
}
```

## Expected Timeline

- **Pilot study** (Option 2, 3 seeds, 1 game): ~6 hours
- **Full re-run** (Option 2, 60 tasks): ~3 days
- **Analysis**: 1 day

**Total**: ~4-5 days for complete comparison

## Recommendation

**Proceed with Option 2 (32 hidden units, 1 layer)** because:
1. Still 10x above observed effective dimensionality (safe margin)
2. 23x parameter reduction (substantial efficiency gain)
3. Expected to reveal stronger architectural effects
4. Fast enough for quick iteration

If Option 2 performs well, consider Option 3 for even stronger effects.
