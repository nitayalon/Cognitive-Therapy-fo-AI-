# Reciprocity Analysis Summary
**Date:** May 8, 2026  
**Analysis Type:** Social Learning vs Nash Equilibrium Behavior  

---

## Research Question
Does adding the opponent's previous action to the network input enable **social learning** (reciprocity) or do agents still converge to **pure Nash equilibrium** strategies (ignoring opponent history)?

**Reciprocity Metric:**  
`Reciprocity Strength = P(agent cooperates | opponent cooperated) - P(agent cooperates | opponent defected)`

**Interpretation:**
- **Reciprocity > 0**: Social learning - agent rewards opponent's cooperation
- **Reciprocity ≈ 0**: Nash equilibrium - agent ignores opponent's history
- **Reciprocity < 0**: Anti-reciprocal - agent exploits opponent's cooperation

---

## Data Summary

### Whole Population (Task-Only Paradigm)
- **Test Job ID:** 912631
- **Training Job ID:** 911034
- **Episodes Analyzed:** 450,000
- **Games Analyzed:** 100
- **Network Input:** 5 elements (payoff matrix + round number)
- **Test Coverage:** Prisoner's Dilemma only with opponent defection probability 0.1

### Task-Opponent (With Opponent Action Input)
- **Test Job ID:** 911088
- **Training Job ID:** 910969
- **Episodes Analyzed:** 4,000,000  
- **Games Analyzed:** 100
- **Network Input:** 6 elements (payoff matrix + round number + opponent's previous action)
- **Test Coverage:** Prisoner's Dilemma only with opponent defection probability 0.2

---

## Key Findings

### Whole Population (Task-Only)
**Prisoner's Dilemma vs 0.1 Defector:**
- P(cooperate | opponent cooperated) = **18.7%** ± 1.8%
- P(cooperate | opponent defected) = **26.9%** ± 1.1%
- **Reciprocity Strength = -8.2%** ± 1.7%

**Interpretation:** **Anti-reciprocal behavior** - The agent is MORE likely to cooperate after the opponent defects. This is counterintuitive and suggests the agent may have learned a mixed strategy that exploits the opponent's low defection rate, defecting more when the opponent "wastes" their rare defections on cooperation.

### Task-Opponent (With Opponent Action Input)
**Prisoner's Dilemma vs 0.2 Defector:**
- P(cooperate | opponent cooperated) = **31.3%** ± 0.5%
- P(cooperate | opponent defected) = **31.1%** ± 0.5%
- **Reciprocity Strength = +0.2%** ± 0.4%

**Interpretation:** **Near-zero reciprocity** - Despite having explicit access to the opponent's previous action, the network learned to essentially ignore it. The agent cooperates ~31% of the time regardless of opponent's previous move. This is consistent with a **pure Nash equilibrium** strategy.

---

## Conclusions

### Main Findings
1. **No Social Learning Detected:** Neither paradigm shows evidence of reciprocal cooperation dynamics
2. **Task-Only Shows Anti-Reciprocity:** Paradoxically, the network WITHOUT opponent action input shows stronger (albeit negative) sensitivity to opponent history
3. **Opponent Action Input Converges to Nash:** Adding opponent's previous action as input did NOT induce social learning - instead, the network learned to ignore this feature and play a mixed Nash strategy

### Theoretical Implications
**Why No Reciprocity in Prisoner's Dilemma?**
- In PD, mutual defection is the only Nash equilibrium
- Reciprocal cooperation (Tit-for-Tat) is suboptimal against probabilistic opponents
- The network correctly learned that responding to individual opponent actions is less important than tracking overall opponent defection probability

**The Anti-Reciprocity Paradox:**
The negative reciprocity in the task-only paradigm may reflect:
- Learned exploitation of probabilistic patterns
- The network detecting when opponent "wastes" rare defections on cooperation
- A meta-strategy that defects more aggressively after opponent cooperation events

### Limitations
⚠️ **Important:** This analysis is based on limited test data:
- Only Prisoner's Dilemma results available
- Only single opponent types per paradigm (0.1 and 0.2 defection probabilities)
- Hawk-Dove and Stag-Hunt data not yet analyzed (may show different patterns)

**Expected Behavior in Other Games:**
- **Hawk-Dove:** Should show reciprocity (coordination benefits)
- **Stag-Hunt:** Should show strong reciprocity (trust-building dynamics)

---

## Next Steps

### Immediate Actions
1. ✅ **Complete test data download** for all games and opponent types
2. ⏳ **Re-run analysis** on full dataset including:
   - All 3 games (PD, HD, SH)
   - All 5 opponent types (0.1, 0.3, 0.5, 0.7, 0.9)
   - Full generalization matrix

### Extended Analysis
3. **Compare reciprocity across games:**
   - Do coordination games (HD, SH) show more reciprocity?
   - How does reciprocity vary with opponent type?

4. **Training vs Testing reciprocity:**
   - Does reciprocity appear during training?
   - Is lack of reciprocity a training vs testing difference?

5. **Baseline comparison:**
   - How does 5-element input (no opponent action) compare to 6-element?
   - Does explicit opponent action input change learning dynamics?

---

## Generated Files

### Whole Population
- `reciprocity_whole_population_912631/reciprocity_per_game.csv` - Raw per-game metrics
- `reciprocity_whole_population_912631/reciprocity_aggregated.csv` - Summary statistics
- `reciprocity_whole_population_912631/reciprocity_whole_population_(task-only).png` - Conditional cooperation plot
- `reciprocity_whole_population_912631/reciprocity_strength_whole_population_(task-only).png` - Reciprocity strength plot

### Task-Opponent
- `reciprocity_task_opponent_911088/reciprocity_per_game.csv` - Raw per-game metrics
- `reciprocity_task_opponent_911088/reciprocity_aggregated.csv` - Summary statistics
- `reciprocity_task_opponent_911088/reciprocity_task-opponent_(with_opponent_action_input).png` - Conditional cooperation plot
- `reciprocity_task_opponent_911088/reciprocity_strength_task-opponent_(with_opponent_action_input).png` - Reciprocity strength plot

---

## Statistical Notes

**Sample Sizes:**
- Whole Population: 100 games with ~4,500 trials each
- Task-Opponent: 100 games with ~40,000 trials each

**Standard Errors:**
- All metrics reported as mean ± standard deviation across games
- High confidence in near-zero reciprocity finding (SE < 0.5%)

**Effect Sizes:**
- Whole population anti-reciprocity: Cohen's d ≈ -4.8 (very large effect)
- Task-opponent near-zero reciprocity: Cohen's d ≈ 0.05 (negligible effect)
