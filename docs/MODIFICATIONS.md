# Modification Documentation Template

## Overview
Document all changes to loss functions, training procedures, and opponents with precise technical details.

---

## Loss Function Modifications

### Change opponent action prediction loss function - 20/10/2025 - Nitay

**File(s) Modified**: 
- `src/cognitive_therapy_ai/tom_rl_loss.py`
- `src/cognitive_therapy_ai/loss.py` (if legacy changes)

**Change Type**: Replace the cross entropy loss with a KL divergence between the network's opponent's policy prediction and the opponent's true policy. Normalize this size to unity.

**Technical Details**:
```python
# BEFORE (original implementation)
def _calculate_opponent_prediction_loss(self, opponent_coop_probs, opponent_actions):
    # Binary cross-entropy loss
    bce_loss = F.binary_cross_entropy(opponent_coop_probs, opponent_actions_float, reduction='mean')
    return bce_loss

# Network output: opponent_coop_prob (batch_size, 1) - single cooperation probability

# AFTER (modified implementation)  
def _calculate_opponent_policy_loss(self, predicted_opponent_logits, true_opponent_policy, temperature=0.1):
    # Convert predicted logits to policy distribution
    predicted_policy = F.softmax(predicted_opponent_logits / temperature, dim=-1)
    # KL divergence between true opponent policy and predicted policy
    kl_loss = F.kl_div(predicted_policy.log(), true_opponent_policy, reduction='batchmean')
    return kl_loss

# Network output: opponent_policy_logits (batch_size, 2) - [defect_logit, coop_logit]
```

**Mathematical Description**:
- **Original**: L_Op = BCE(p̂_coop, o_t) where o_t ∈ {0,1} is observed action
- **Modified**: L_Op = KL(π_true || π_pred) where π_true = (1-p_d, p_d) from opponent type
- **True Policy Source**: Opponent type parameters (e.g., p_d = defection probability)
- **Temperature**: τ = 0.1 (low temperature for sharp policy predictions)
- **Open Issue**: KL divergence normalization method (to be determined)

**Rationale**: 
Instead of predicting pointwise actions, incentivize reasoning about opponent's **policy distribution**. The true opponent policy π_true = (1-p_d, p_d) is available from opponent type parameters during training. This encourages the network to:
1. Model opponent decision-making process rather than just predicting single actions
2. Develop understanding of opponent's behavioral tendencies (shallow mentalization)
3. Learn policy-level representations that generalize better across contexts

**Expected Impact**:
- Training convergence: May require more episodes due to distributional vs pointwise prediction
- Performance: Better generalization to opponent behavior patterns
- Theory of Mind development: Deeper understanding of opponent policies vs actions

**Open Issues**:
- **KL Divergence Normalization**: How to normalize KL(π_true || π_pred) to unity scale?
  - Option 1: Divide by max possible KL divergence
  - Option 2: Use running average for adaptive normalization  
  - Option 3: Learn normalization factor as parameter
  - **Decision**: To be determined through experimentation

**Testing Plan**:
- [ ] Unit test for new loss components
- [ ] Integration test with existing training
- [ ] Comparison experiment: old vs new loss

---

## Training Procedure Modifications

### Simoultanious game training - 20/10/2025

**File(s) Modified**:
- `src/cognitive_therapy_ai/trainer.py`
- `main_experiment.py` (if experiment changes)

**Change Type**: Instead of sampling a game with uniform probability, the network is trained on all 3 games at every iteration. This ensures full coverage of the environment, but requires elongated training

**Technical Details**:
```python
# BEFORE (probabilistic sampling)
for opponent in opponents:
    for game_name in game_names:
        # Probability of including this game for this opponent
        if random.random() < game_weights[game_name]:  # ← Probabilistic sampling
            game = games[game_name]
            session = GameSession(game, opponent, self.network, ...)
            # May skip games due to random chance

# AFTER (deterministic all-games training)
for opponent in opponents:
    for game_name in game_names:
        # Always train on every game (deterministic)
        game = games[game_name]  # ← No probability check
        session = GameSession(game, opponent, self.network, ...)
        # Guaranteed to train on all games every epoch
```

**Algorithm Changes**:
1. **Deterministic Coverage**: Every opponent trains on every game each epoch
2. **Removed Probabilistic Sampling**: Eliminated `random.random() < game_weights[game_name]` check
3. **Increased Training Volume**: 3x more sessions per epoch (if 3 games) 
4. **Balanced Exposure**: Equal training time across all games per epoch

**Parameter Impact**:
| Aspect | Old Behavior | New Behavior | Impact |
|--------|-------------|--------------|---------|
| **Games per Epoch** | Variable (probabilistic) | Fixed (all games) | +200% more sessions |
| **Training Stability** | May skip games | Guaranteed coverage | More consistent |
| **Memory Usage** | Lower (fewer sessions) | Higher (more sessions) | +200% memory |
| **Convergence** | Faster per epoch | Slower per epoch | Better final performance |

**Rationale**:
The probabilistic game sampling could lead to:
1. **Uneven exposure**: Some games might be undersampled in certain epochs
2. **Training instability**: Inconsistent game coverage across training
3. **Convergence issues**: Network might not see balanced examples of all environments

Deterministic all-games training ensures:
1. **Balanced learning**: Equal exposure to all game environments per epoch
2. **Consistent gradients**: Predictable training signal from all games
3. **Robust generalization**: Network learns from complete environment distribution

**Expected Impact**:
- Training time: **+200% longer per epoch** (3 games vs ~1 game average)
- Memory usage: **+200% higher** (more sessions per batch)
- Convergence: **Slower per epoch, better final performance**
- Stability: **More consistent training dynamics**

**Testing Plan**:
- [ ] Performance comparison: probabilistic vs deterministic training
- [ ] Memory usage monitoring during multi-game training
- [ ] Convergence speed analysis (epochs vs wall-clock time)
- [ ] Cross-game generalization evaluation

---

## Opponent Modifications

### [Modification Name] - [Date]

**File(s) Modified**:
- `src/cognitive_therapy_ai/opponent.py`

**Change Type**: [New Strategy | Strategy Modification | Parameter Tuning]

**Strategy Implementation**:
```python
class NewOpponentStrategy(OpponentStrategy):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
    
    def choose_action(self, game_history, round_number):
        # Strategy logic here
        return action
```

**Strategy Description**:
- **Name**: [Strategy name]
- **Behavior**: [Detailed description of decision-making]
- **Parameters**: [List and explain all parameters]
- **Theoretical Basis**: [Game theory/research foundation]

**Research Purpose**:
[Why this opponent was created/modified - what aspect of ToM or cooperation it tests]

**Integration**:
- **Factory Registration**: Added to `OpponentFactory.create_opponent()`
- **Configuration**: Available in experiment configs as "[strategy-name]"
- **Testing**: Compatible with all existing games

---

## Experiment Configuration Changes

### [Modification Name] - [Date]

**Configuration Files Modified**:
- `config/default_config.json`
- `config/quick_test_config.json`

**New Parameters**:
```json
{
  "new_parameter": {
    "value": 0.5,
    "description": "Controls aspect X of training",
    "valid_range": [0.0, 1.0]
  }
}
```

**Backward Compatibility**:
- [ ] Old configs still work with defaults
- [ ] Migration guide provided
- [ ] Version compatibility documented

---

## Testing and Validation

### Regression Testing
- [ ] All existing unit tests pass
- [ ] Integration tests with multi-game training
- [ ] Performance benchmarks maintained

### Ablation Studies
- [ ] Isolated effect of each modification
- [ ] Statistical significance testing
- [ ] Comparison with baseline results

### Documentation Updates
- [ ] Updated README.md
- [ ] Updated .github/copilot-instructions.md
- [ ] Updated docstrings and comments

---

## Migration Guide

### For Existing Experiments
1. **Backup**: Save current experiment results
2. **Configuration**: Update config files with new parameters
3. **Code**: Update any custom training scripts
4. **Testing**: Run quick validation experiment

### For Comparison Studies
1. **Baseline**: Document exact configuration of "old" system
2. **New System**: Document exact configuration of "new" system  
3. **Metrics**: Ensure comparable evaluation metrics
4. **Statistical**: Plan appropriate statistical tests

---

## Notes for AI Assistant

### Key Files to Monitor
- `src/cognitive_therapy_ai/tom_rl_loss.py` - Primary loss function
- `src/cognitive_therapy_ai/trainer.py` - Training orchestration
- `src/cognitive_therapy_ai/opponent.py` - Opponent strategies
- `main_experiment.py` - Experiment configuration

### Critical Changes to Track
1. **Loss Function Signatures**: Any change to input/output format
2. **Training Data Flow**: Changes to batch processing or session management
3. **Opponent Interface**: Changes to strategy API
4. **Configuration Schema**: New parameters or deprecations

### Validation Checklist
- [ ] All imports still work
- [ ] No breaking changes to public APIs
- [ ] Backward compatibility maintained where possible
- [ ] Performance impact documented
- [ ] Research implications explained